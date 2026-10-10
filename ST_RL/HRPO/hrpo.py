"""HRPO loss and reward utilities for AdaSR-style streaming reasoning.

The implementation mirrors AdaSR's paper formulation:

* streaming-local advantages supervise streaming tokens;
* deep-local advantages supervise final deep-reasoning tokens;
* global advantages supervise the full trajectory;
* local and global objectives are combined with PPO/GRPO clipping.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ST_RL.common import group_normalize, grpo_kl_penalty, masked_mean, ppo_clipped_surrogate


@dataclass
class HRPOConfig:
    """Configuration for HRPO and AdaSR reward composition."""

    clip_range: float = 0.2
    kl_coef: float = 0.0
    local_weight: float = 0.05
    normalize_eps: float = 1e-8
    accuracy_weight: float = 1.0
    format_weight: float = 2.0
    length_weight: float = 0.05
    latency_alpha: float = 0.5
    reward_tau: Optional[float] = None
    mask_local_length_by_format: bool = True


@dataclass
class HRPORewardComponents:
    """Scalar reward components used to compose HRPO advantages."""

    streaming_format: torch.Tensor
    deep_format: torch.Tensor
    accuracy: torch.Tensor
    streaming_length_penalty: torch.Tensor
    deep_length_penalty: torch.Tensor
    efficiency: torch.Tensor
    streaming_total_length: torch.Tensor
    deep_length: torch.Tensor


@dataclass
class HRPOAdvantages:
    """Streaming-local, deep-local, and global advantages."""

    streaming: torch.Tensor
    deep: torch.Tensor
    global_: torch.Tensor
    components: HRPORewardComponents


def _as_float_1d(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.dim() != 1:
        raise ValueError("%s must be a 1D tensor" % name)
    return value.float()


def _streaming_length_penalty(
    streaming_segment_lengths: Optional[torch.Tensor],
    streaming_total_lengths: Optional[torch.Tensor],
    segment_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute AdaSR Eq. 6 streaming length penalty."""
    if streaming_segment_lengths is not None:
        lengths = streaming_segment_lengths.float()
        if lengths.dim() != 2:
            raise ValueError("streaming_segment_lengths must have shape [batch, segments]")
        penalties = -torch.log1p(lengths)
        if segment_mask is None:
            segment_mask = lengths > 0
        mask = segment_mask.to(device=lengths.device, dtype=lengths.dtype)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        penalty = (penalties * mask).sum(dim=-1) / denom
        total = (lengths * mask).sum(dim=-1)
        return penalty, total
    if streaming_total_lengths is None:
        raise ValueError("provide streaming_segment_lengths or streaming_total_lengths")
    total = streaming_total_lengths.float()
    return -torch.log1p(total), total


def compute_hrpo_reward_components(
    streaming_format: torch.Tensor,
    deep_format: torch.Tensor,
    accuracy: torch.Tensor,
    deep_lengths: torch.Tensor,
    streaming_segment_lengths: Optional[torch.Tensor] = None,
    streaming_total_lengths: Optional[torch.Tensor] = None,
    streaming_segment_mask: Optional[torch.Tensor] = None,
    config: Optional[HRPOConfig] = None,
) -> HRPORewardComponents:
    """Compute AdaSR format, accuracy, length, and efficiency rewards.

    ``streaming_format``, ``deep_format``, and ``accuracy`` can be binary
    rewards or soft scores in ``[0, 1]``. Lengths are token counts.
    """
    if config is None:
        config = HRPOConfig()
    streaming_format = _as_float_1d("streaming_format", streaming_format)
    deep_format = _as_float_1d("deep_format", deep_format).to(device=streaming_format.device)
    accuracy = _as_float_1d("accuracy", accuracy).to(device=streaming_format.device)
    deep_lengths = _as_float_1d("deep_lengths", deep_lengths).to(device=streaming_format.device)
    if not (streaming_format.shape == deep_format.shape == accuracy.shape == deep_lengths.shape):
        raise ValueError("format, accuracy, and deep_lengths tensors must share shape [batch]")

    stream_penalty, stream_total = _streaming_length_penalty(
        streaming_segment_lengths=streaming_segment_lengths,
        streaming_total_lengths=streaming_total_lengths,
        segment_mask=streaming_segment_mask,
    )
    stream_penalty = stream_penalty.to(device=streaming_format.device)
    stream_total = stream_total.to(device=streaming_format.device)

    deep_penalty = -torch.log1p(deep_lengths)
    if config.reward_tau is None:
        tau = (stream_total + deep_lengths).detach().float().mean().clamp_min(1.0)
    else:
        tau = torch.tensor(float(config.reward_tau), device=streaming_format.device)

    gate = accuracy * streaming_format * deep_format
    lower_bound_gate = (1.0 - torch.exp(-deep_lengths / tau)) * (1.0 - torch.exp(-stream_total / tau))
    latency_cost = torch.exp(-(config.latency_alpha * stream_total + deep_lengths) / tau)
    efficiency = gate * lower_bound_gate * latency_cost

    return HRPORewardComponents(
        streaming_format=streaming_format,
        deep_format=deep_format,
        accuracy=accuracy,
        streaming_length_penalty=stream_penalty,
        deep_length_penalty=deep_penalty,
        efficiency=efficiency,
        streaming_total_length=stream_total,
        deep_length=deep_lengths,
    )


def compute_hrpo_advantages(
    streaming_format: torch.Tensor,
    deep_format: torch.Tensor,
    accuracy: torch.Tensor,
    deep_lengths: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    streaming_segment_lengths: Optional[torch.Tensor] = None,
    streaming_total_lengths: Optional[torch.Tensor] = None,
    streaming_segment_mask: Optional[torch.Tensor] = None,
    config: Optional[HRPOConfig] = None,
) -> HRPOAdvantages:
    """Compose AdaSR Eq. 8 streaming/deep/global advantages."""
    if config is None:
        config = HRPOConfig()
    components = compute_hrpo_reward_components(
        streaming_format=streaming_format,
        deep_format=deep_format,
        accuracy=accuracy,
        deep_lengths=deep_lengths,
        streaming_segment_lengths=streaming_segment_lengths,
        streaming_total_lengths=streaming_total_lengths,
        streaming_segment_mask=streaming_segment_mask,
        config=config,
    )
    device = components.accuracy.device
    if group_ids is not None:
        group_ids = group_ids.to(device=device)

    stream_valid = components.streaming_format > 0.5 if config.mask_local_length_by_format else None
    deep_valid = components.deep_format > 0.5 if config.mask_local_length_by_format else None

    stream_advantage = (
        config.format_weight
        * group_normalize(components.streaming_format, group_ids, config.normalize_eps)
        + config.length_weight
        * group_normalize(
            components.streaming_length_penalty,
            group_ids,
            config.normalize_eps,
            valid_mask=stream_valid,
        )
    )
    deep_advantage = (
        config.format_weight
        * group_normalize(components.deep_format, group_ids, config.normalize_eps)
        + config.length_weight
        * group_normalize(
            components.deep_length_penalty,
            group_ids,
            config.normalize_eps,
            valid_mask=deep_valid,
        )
    )
    global_advantage = (
        config.accuracy_weight * group_normalize(components.accuracy, group_ids, config.normalize_eps)
        + config.length_weight * group_normalize(components.efficiency, group_ids, config.normalize_eps)
    )

    return HRPOAdvantages(
        streaming=stream_advantage,
        deep=deep_advantage,
        global_=global_advantage,
        components=components,
    )


def hrpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    streaming_mask: torch.Tensor,
    deep_mask: torch.Tensor,
    streaming_advantages: torch.Tensor,
    deep_advantages: torch.Tensor,
    global_advantages: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    config: Optional[HRPOConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the HRPO loss from sampled response log probabilities.

    Masks are token-level tensors of shape ``[batch, response_len]``. Streaming
    and deep masks should be disjoint subsets of ``response_mask``.
    """
    if config is None:
        config = HRPOConfig()
    expected_shape = log_probs.shape
    for name, tensor in [
        ("old_log_probs", old_log_probs),
        ("response_mask", response_mask),
        ("streaming_mask", streaming_mask),
        ("deep_mask", deep_mask),
    ]:
        if tensor.shape != expected_shape:
            raise ValueError("%s must have shape %s" % (name, tuple(expected_shape)))

    response_mask_f = response_mask.to(dtype=log_probs.dtype)
    streaming_mask_f = streaming_mask.to(dtype=log_probs.dtype) * response_mask_f
    deep_mask_f = deep_mask.to(dtype=log_probs.dtype) * response_mask_f

    streaming_advantages = streaming_advantages.to(device=log_probs.device, dtype=log_probs.dtype)
    deep_advantages = deep_advantages.to(device=log_probs.device, dtype=log_probs.dtype)
    global_advantages = global_advantages.to(device=log_probs.device, dtype=log_probs.dtype)

    streaming_surrogate = ppo_clipped_surrogate(
        log_probs, old_log_probs, streaming_advantages, config.clip_range
    )
    deep_surrogate = ppo_clipped_surrogate(log_probs, old_log_probs, deep_advantages, config.clip_range)
    global_surrogate = ppo_clipped_surrogate(log_probs, old_log_probs, global_advantages, config.clip_range)

    streaming_objective = masked_mean(streaming_surrogate, streaming_mask_f, dim=-1)
    deep_objective = masked_mean(deep_surrogate, deep_mask_f, dim=-1)
    global_objective = masked_mean(global_surrogate, response_mask_f, dim=-1)

    kl_per_token = grpo_kl_penalty(log_probs, ref_log_probs) * response_mask_f
    kl = masked_mean(kl_per_token, response_mask_f, dim=-1)

    objective = (
        config.local_weight * streaming_objective
        + config.local_weight * deep_objective
        + global_objective
        - config.kl_coef * kl
    )
    loss = -objective.mean()

    metrics = {
        "loss": loss.detach(),
        "objective": objective.detach().mean(),
        "streaming_objective": streaming_objective.detach().mean(),
        "deep_objective": deep_objective.detach().mean(),
        "global_objective": global_objective.detach().mean(),
        "kl": kl.detach().mean(),
        "streaming_advantage_mean": streaming_advantages.detach().float().mean(),
        "deep_advantage_mean": deep_advantages.detach().float().mean(),
        "global_advantage_mean": global_advantages.detach().float().mean(),
        "streaming_len_mean": streaming_mask_f.detach().sum(dim=-1).float().mean(),
        "deep_len_mean": deep_mask_f.detach().sum(dim=-1).float().mean(),
        "response_len_mean": response_mask_f.detach().sum(dim=-1).float().mean(),
    }
    return loss, metrics

