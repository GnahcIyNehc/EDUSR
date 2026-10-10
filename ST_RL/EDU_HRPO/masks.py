"""EDU-BLHRPO 的阶段 token mask。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class StageMasks:
    streaming: torch.Tensor
    deep: torch.Tensor
    answer: torch.Tensor
    response: torch.Tensor


def stage_masks_from_lengths(
    streaming_lengths: torch.Tensor,
    deep_lengths: torch.Tensor,
    answer_lengths: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
) -> StageMasks:
    """构造连续布局 `[stream][deep][answer][padding]` 的互斥 mask。"""

    if streaming_lengths.dim() != 1 or deep_lengths.shape != streaming_lengths.shape:
        raise ValueError("streaming_lengths and deep_lengths must be one-dimensional tensors of equal shape.")
    if answer_lengths is None:
        answer_lengths = torch.zeros_like(streaming_lengths)
    if answer_lengths.shape != streaming_lengths.shape:
        raise ValueError("answer_lengths must have the same shape as streaming_lengths.")
    stream = streaming_lengths.long().clamp_min(0)
    deep = deep_lengths.long().clamp_min(0)
    answer = answer_lengths.long().clamp_min(0)
    total = stream + deep + answer
    if max_length is None:
        max_length = int(total.max().item()) if total.numel() else 0
    if max_length < int(total.max().item() if total.numel() else 0):
        raise ValueError("max_length is smaller than the actual response length.")
    positions = torch.arange(max_length, device=stream.device).unsqueeze(0)
    stream_end = stream.unsqueeze(1)
    deep_end = (stream + deep).unsqueeze(1)
    answer_end = total.unsqueeze(1)
    stream_mask = positions < stream_end
    deep_mask = (positions >= stream_end) & (positions < deep_end)
    answer_mask = (positions >= deep_end) & (positions < answer_end)
    response_mask = positions < answer_end
    masks = StageMasks(
        streaming=stream_mask,
        deep=deep_mask,
        answer=answer_mask,
        response=response_mask,
    )
    validate_stage_masks(masks)
    return masks


def validate_stage_masks(masks: StageMasks) -> None:
    """保证阶段 mask 互斥，且所有局部 token 都属于响应。"""

    shape = masks.response.shape
    for name in ("streaming", "deep", "answer"):
        if getattr(masks, name).shape != shape:
            raise ValueError("All stage masks must have the same shape.")
    if bool((masks.streaming & masks.deep).any().item()):
        raise ValueError("Streaming and deep masks must not overlap.")
    if bool((masks.streaming & masks.answer).any().item()):
        raise ValueError("Streaming and answer masks must not overlap.")
    if bool((masks.deep & masks.answer).any().item()):
        raise ValueError("Deep and answer masks must not overlap.")
    local_union = masks.streaming | masks.deep | masks.answer
    if bool((local_union & ~masks.response).any().item()):
        raise ValueError("Stage tokens must be a subset of response tokens.")


def compose_token_advantages(
    masks: StageMasks,
    streaming_advantage: torch.Tensor,
    deep_advantage: torch.Tensor,
    global_advantage: torch.Tensor,
    local_weight: float = 0.05,
) -> torch.Tensor:
    """显式展示每类 token 接收的全局和局部优势。"""

    batch_size = masks.response.shape[0]
    for name, value in (
        ("streaming_advantage", streaming_advantage),
        ("deep_advantage", deep_advantage),
        ("global_advantage", global_advantage),
    ):
        if value.dim() != 1 or value.shape[0] != batch_size:
            raise ValueError("%s must have shape [batch]." % name)
    dtype = global_advantage.dtype
    result = global_advantage.unsqueeze(-1).expand_as(masks.response).to(dtype=dtype)
    result = result + float(local_weight) * (
        streaming_advantage.unsqueeze(-1) * masks.streaming.to(dtype=dtype)
        + deep_advantage.unsqueeze(-1) * masks.deep.to(dtype=dtype)
    )
    return result * masks.response.to(dtype=dtype)

