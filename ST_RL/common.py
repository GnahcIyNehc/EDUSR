"""Shared tensor utilities for GRPO and HRPO.

The functions in this module operate on sampled-response log probabilities.
They are intentionally independent of a particular rollout engine so that the
same losses can be plugged into Transformers, vLLM/verl, or a small local loop.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def sequence_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Return a boolean mask with shape ``[batch, max_length]``."""
    if lengths.dim() != 1:
        raise ValueError("lengths must be a 1D tensor")
    if max_length is None:
        max_length = int(lengths.max().item()) if lengths.numel() else 0
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.long().unsqueeze(1)


def phase_masks_from_lengths(
    streaming_lengths: torch.Tensor,
    deep_lengths: torch.Tensor,
    max_length: Optional[int] = None,
):
    """Build streaming/deep/response masks for concatenated response tokens.

    The expected response layout is ``[streaming tokens][deep tokens][padding]``.
    """
    if streaming_lengths.shape != deep_lengths.shape:
        raise ValueError("streaming_lengths and deep_lengths must have the same shape")
    total_lengths = streaming_lengths.long() + deep_lengths.long()
    if max_length is None:
        max_length = int(total_lengths.max().item()) if total_lengths.numel() else 0
    positions = torch.arange(max_length, device=streaming_lengths.device).unsqueeze(0)
    stream_mask = positions < streaming_lengths.long().unsqueeze(1)
    response_mask = positions < total_lengths.unsqueeze(1)
    deep_mask = response_mask & ~stream_mask
    return stream_mask, deep_mask, response_mask


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum ``values`` over ``dim`` while ignoring masked positions."""
    return (values * mask.to(dtype=values.dtype)).sum(dim=dim)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean ``values`` over valid mask positions."""
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum(dim=dim).clamp_min(eps)
    return (values * mask_f).sum(dim=dim) / denom


def group_normalize(
    values: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize scalar values within each prompt group.

    GRPO-style training samples multiple completions for the same prompt and
    uses the group mean and standard deviation as a critic-free baseline.

    Args:
        values: Tensor of shape ``[batch]``.
        group_ids: Tensor of shape ``[batch]``. Samples with the same id are
            normalized together. If omitted, the whole batch is one group.
        eps: Small positive constant for numerical stability.
        valid_mask: Optional boolean tensor of shape ``[batch]``. Statistics are
            computed only over valid samples; invalid samples receive zero
            advantage.
    """
    if values.dim() != 1:
        raise ValueError("values must be a 1D tensor")
    values = values.float()
    if group_ids is None:
        group_ids = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    else:
        group_ids = group_ids.to(device=values.device)
    if valid_mask is None:
        valid_mask = torch.ones_like(values, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)

    advantages = torch.zeros_like(values)
    for group in torch.unique(group_ids):
        group_mask = group_ids == group
        stats_mask = group_mask & valid_mask
        if int(stats_mask.sum().item()) <= 1:
            continue
        group_values = values[stats_mask]
        mean = group_values.mean()
        std = group_values.std(unbiased=False)
        if float(std.item()) < eps:
            continue
        advantages[group_mask] = (values[group_mask] - mean) / std.clamp_min(eps)
        advantages[group_mask & ~valid_mask] = 0.0
    return advantages


def expand_scalar_to_tokens(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[batch]`` tensor to a token tensor shape."""
    if values.dim() == 1:
        return values.unsqueeze(-1).expand_as(like)
    if values.shape != like.shape:
        raise ValueError("token-level values must already match the target shape")
    return values


def ppo_clipped_surrogate(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """PPO/GRPO clipped surrogate ``min(r A, clip(r) A)`` per token."""
    if log_probs.shape != old_log_probs.shape:
        raise ValueError("log_probs and old_log_probs must have the same shape")
    token_advantages = expand_scalar_to_tokens(advantages, log_probs).to(dtype=log_probs.dtype)
    ratios = torch.exp(log_probs - old_log_probs)
    unclipped = ratios * token_advantages
    clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * token_advantages
    return torch.min(unclipped, clipped)


def grpo_kl_penalty(log_probs: torch.Tensor, ref_log_probs: Optional[torch.Tensor]) -> torch.Tensor:
    """Non-negative sampled-token KL estimator used by common GRPO codebases.

    Given sampled token log probabilities from the current and reference
    policies, this computes ``exp(ref - current) - (ref - current) - 1``.
    It is zero when the two log probabilities match.
    """
    if ref_log_probs is None:
        return torch.zeros_like(log_probs)
    if log_probs.shape != ref_log_probs.shape:
        raise ValueError("log_probs and ref_log_probs must have the same shape")
    delta = ref_log_probs - log_probs
    return torch.exp(delta) - delta - 1.0


def gather_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities of ``token_ids`` from logits.

    Args:
        logits: Tensor with shape ``[batch, seq, vocab]``.
        token_ids: Tensor with shape ``[batch, seq]``.
    """
    if logits.dim() != 3 or token_ids.dim() != 2:
        raise ValueError("expected logits [B, T, V] and token_ids [B, T]")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token_ids sequence dimensions must match")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def response_log_probs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sampled response-token log probabilities from LM logits.

    ``logits[:, t]`` predicts ``input_ids[:, t + 1]``. The returned tensor is
    aligned to the response positions ``input_ids[:, 1:]`` and has shape
    ``[batch, seq - 1]``; callers should pass ``response_mask[:, 1:]``.
    """
    shifted_logits = logits[:, :-1, :]
    shifted_ids = input_ids[:, 1:]
    shifted_mask = response_mask[:, 1:]
    return gather_log_probs(shifted_logits, shifted_ids) * shifted_mask.to(dtype=logits.dtype)

