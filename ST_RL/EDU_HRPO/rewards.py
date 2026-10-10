"""EDU-BLHRPO 的三视图归因和分量级优势组合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ST_RL.common import group_normalize

from .config import EDUBLHRPOConfig
from .latency import LatencyBatch, compute_latency_batch


@dataclass
class EDURewardBatch:
    """从完整 rollout 记录整理出的奖励输入张量。"""

    group_ids: torch.Tensor
    exact: torch.Tensor
    part: torch.Tensor
    stream_probe_exact: torch.Tensor
    stream_probe_part: torch.Tensor
    no_stream_exact: torch.Tensor
    no_stream_part: torch.Tensor
    stream_probe_valid: torch.Tensor
    no_stream_probe_valid: torch.Tensor
    stream_probe_truncated: torch.Tensor
    no_stream_probe_truncated: torch.Tensor
    stream_verifier: torch.Tensor
    stream_format: torch.Tensor
    deep_format: torch.Tensor
    deep_lengths: torch.Tensor
    thought_lengths: torch.Tensor
    input_lengths: torch.Tensor
    group_edu_counts: torch.Tensor
    group_mask: torch.Tensor
    think_group_mask: torch.Tensor
    control_uncertainty: torch.Tensor
    relevance: torch.Tensor
    boundary_times: Optional[torch.Tensor] = None
    next_boundary_times: Optional[torch.Tensor] = None
    slack_tokens: Optional[torch.Tensor] = None

    def validate(self) -> None:
        batch_size = int(self.exact.shape[0])
        for name in (
            "group_ids",
            "exact",
            "part",
            "stream_probe_exact",
            "stream_probe_part",
            "no_stream_exact",
            "no_stream_part",
            "stream_probe_valid",
            "no_stream_probe_valid",
            "stream_probe_truncated",
            "no_stream_probe_truncated",
            "stream_verifier",
            "stream_format",
            "deep_format",
            "deep_lengths",
        ):
            value = getattr(self, name)
            if value.dim() != 1 or int(value.shape[0]) != batch_size:
                raise ValueError("%s must have shape [batch]." % name)
        group_shape = self.thought_lengths.shape
        if len(group_shape) != 2 or int(group_shape[0]) != batch_size:
            raise ValueError("thought_lengths must have shape [batch, groups].")
        for name in (
            "input_lengths",
            "group_edu_counts",
            "group_mask",
            "think_group_mask",
            "control_uncertainty",
            "relevance",
        ):
            if getattr(self, name).shape != group_shape:
                raise ValueError("%s must have shape %s." % (name, tuple(group_shape)))
        for name in ("boundary_times", "next_boundary_times", "slack_tokens"):
            value = getattr(self, name)
            if value is not None and value.shape != group_shape:
                raise ValueError("%s must have shape %s." % (name, tuple(group_shape)))
        for name in (
            "exact",
            "part",
            "stream_probe_exact",
            "stream_probe_part",
            "no_stream_exact",
            "no_stream_part",
            "stream_probe_valid",
            "no_stream_probe_valid",
            "stream_probe_truncated",
            "no_stream_probe_truncated",
            "stream_verifier",
            "stream_format",
            "deep_format",
        ):
            value = getattr(self, name)
            if bool(((value < 0) | (value > 1)).any().item()):
                raise ValueError("%s must be in [0, 1]." % name)


@dataclass
class EDUBLHRPOComponents:
    """用于训练、诊断和消融的未标准化奖励分量。"""

    quality: torch.Tensor
    stream_quality: torch.Tensor
    no_stream_quality: torch.Tensor
    stream_causal_help: torch.Tensor
    stream_causal_harm: torch.Tensor
    stream_attribution: torch.Tensor
    deep_attribution: torch.Tensor
    correction_burden: torch.Tensor
    stream_length_reward: torch.Tensor
    deep_length_reward: torch.Tensor
    efficiency_reward: torch.Tensor
    stream_probe_valid: torch.Tensor
    no_stream_probe_valid: torch.Tensor
    stream_probe_truncated: torch.Tensor
    no_stream_probe_truncated: torch.Tensor
    latency: LatencyBatch


@dataclass
class EDUBLHRPOAdvantages:
    """保持 stage-level 的流式、深度和全局优势。"""

    streaming: torch.Tensor
    deep: torch.Tensor
    global_: torch.Tensor
    components: EDUBLHRPOComponents
    normalized_components: Dict[str, torch.Tensor]


def _quality(exact: torch.Tensor, part: torch.Tensor, gamma_part: float) -> torch.Tensor:
    return (exact.float() + float(gamma_part) * part.float()) / (1.0 + float(gamma_part))


def _normalize(
    value: torch.Tensor,
    group_ids: torch.Tensor,
    config: EDUBLHRPOConfig,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return group_normalize(
        value.float(),
        group_ids=group_ids,
        eps=float(config.normalize_eps),
        valid_mask=valid_mask,
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在有效探针上求均值；空集合返回零。"""

    mask_float = mask.to(dtype=value.dtype)
    return (value * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def compute_edu_blhrpo_advantages(
    batch: EDURewardBatch,
    config: EDUBLHRPOConfig,
    reward_tau: float,
    latency: Optional[LatencyBatch] = None,
) -> EDUBLHRPOAdvantages:
    """实现实验计划第 5、6、7、8 和 10 节的完整优势公式。"""

    config.validate(require_derived=True)
    batch.validate()
    if reward_tau <= 0:
        raise ValueError("reward_tau must be positive.")
    if latency is None:
        latency = compute_latency_batch(
            thought_lengths=batch.thought_lengths,
            input_lengths=batch.input_lengths,
            group_edu_counts=batch.group_edu_counts,
            group_mask=batch.group_mask,
            think_group_mask=batch.think_group_mask,
            control_uncertainty=batch.control_uncertainty,
            relevance=batch.relevance,
            deep_lengths=batch.deep_lengths,
            config=config,
            boundary_times=batch.boundary_times,
            next_boundary_times=batch.next_boundary_times,
            slack_tokens=batch.slack_tokens,
        )

    quality = _quality(batch.exact, batch.part, config.gamma_part)
    stream_quality = _quality(
        batch.stream_probe_exact,
        batch.stream_probe_part,
        config.gamma_part,
    )
    no_stream_quality = _quality(
        batch.no_stream_exact,
        batch.no_stream_part,
        config.gamma_part,
    )
    stream_probe_valid = batch.stream_probe_valid > 0.5
    no_stream_probe_valid = batch.no_stream_probe_valid > 0.5
    both_probes_valid = stream_probe_valid & no_stream_probe_valid
    if config.deep_budget is None:
        raise ValueError("Derive deep_budget from the sentence baseline.")
    correction_burden = (quality - stream_quality).clamp_min(0.0) * torch.clamp(
        batch.deep_lengths.float() / max(float(config.deep_budget), 1.0),
        min=0.0,
        max=1.0,
    )
    correction_burden = torch.where(
        stream_probe_valid,
        correction_burden,
        torch.zeros_like(correction_burden),
    )

    # 共享 Q0 若仅作减法，会在 prompt 内标准化时严格抵消。以下门控式分解
    # 直接实现归因判定表：QS 优于 Q0 是帮助，Q0 优于 QS 与最终 Q 是伤害。
    stream_causal_help = (stream_quality - no_stream_quality).clamp_min(0.0)
    stream_causal_harm = (
        no_stream_quality - torch.maximum(stream_quality, quality)
    ).clamp_min(0.0)
    causal_attribution = stream_causal_help - stream_causal_harm
    causal_attribution = torch.where(
        both_probes_valid,
        causal_attribution,
        torch.where(
            stream_probe_valid,
            stream_quality,
            torch.zeros_like(stream_quality),
        ),
    )
    stream_causal_help = torch.where(
        both_probes_valid,
        stream_causal_help,
        torch.zeros_like(stream_causal_help),
    )
    stream_causal_harm = torch.where(
        both_probes_valid,
        stream_causal_harm,
        torch.zeros_like(stream_causal_harm),
    )
    stream_attribution = (
        causal_attribution
        + float(config.rho_v) * batch.stream_verifier.float()
        - float(config.rho_fix) * correction_burden
    )
    deep_attribution = torch.where(
        stream_probe_valid,
        quality - stream_quality,
        torch.zeros_like(quality),
    )
    efficiency = (
        batch.exact.float()
        * batch.stream_format.float()
        * batch.deep_format.float()
        * torch.exp(-latency.exposure_cost.float() / float(reward_tau))
    )

    normalized = {
        "stream_format": _normalize(batch.stream_format, batch.group_ids, config),
        "stream_length": _normalize(latency.stream_length_reward, batch.group_ids, config),
        "stream_attribution": _normalize(
            stream_attribution,
            batch.group_ids,
            config,
            valid_mask=stream_probe_valid,
        ),
        "deep_format": _normalize(batch.deep_format, batch.group_ids, config),
        "deep_length": _normalize(latency.deep_length_reward, batch.group_ids, config),
        "deep_attribution": _normalize(
            deep_attribution,
            batch.group_ids,
            config,
            valid_mask=stream_probe_valid,
        ),
        "exact": _normalize(batch.exact, batch.group_ids, config),
        "part": _normalize(batch.part, batch.group_ids, config),
        "efficiency": _normalize(efficiency, batch.group_ids, config),
    }
    streaming = (
        float(config.format_weight) * normalized["stream_format"]
        + float(config.beta_s) * normalized["stream_length"]
        + float(config.gamma_attr_s) * normalized["stream_attribution"]
    )
    deep = (
        float(config.format_weight) * normalized["deep_format"]
        + float(config.beta_d) * normalized["deep_length"]
        + float(config.gamma_attr_d) * normalized["deep_attribution"]
    )
    global_advantage = (
        float(config.exact_weight) * normalized["exact"]
        + float(config.gamma_part) * normalized["part"]
        + float(config.beta_eff) * normalized["efficiency"]
    )
    components = EDUBLHRPOComponents(
        quality=quality,
        stream_quality=stream_quality,
        no_stream_quality=no_stream_quality,
        stream_causal_help=stream_causal_help,
        stream_causal_harm=stream_causal_harm,
        stream_attribution=stream_attribution,
        deep_attribution=deep_attribution,
        correction_burden=correction_burden,
        stream_length_reward=latency.stream_length_reward,
        deep_length_reward=latency.deep_length_reward,
        efficiency_reward=efficiency,
        stream_probe_valid=stream_probe_valid.float(),
        no_stream_probe_valid=no_stream_probe_valid.float(),
        stream_probe_truncated=batch.stream_probe_truncated.float(),
        no_stream_probe_truncated=batch.no_stream_probe_truncated.float(),
        latency=latency,
    )
    return EDUBLHRPOAdvantages(
        streaming=streaming,
        deep=deep,
        global_=global_advantage,
        components=components,
        normalized_components=normalized,
    )


def attribution_diagnostics(
    advantages: EDUBLHRPOAdvantages,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """计算论文 Block 2 所需的错误流向指标。"""

    components = advantages.components
    stream_valid = components.stream_probe_valid > 0.5
    no_stream_valid = components.no_stream_probe_valid > 0.5
    both_valid = stream_valid & no_stream_valid
    stream_harm = (components.stream_causal_harm > eps).float()
    deep_regression = (components.stream_quality > components.quality + eps).float()
    deep_correction = (components.quality > components.stream_quality + eps).float()
    stream_correct = components.stream_quality > 0.5
    final_correct = components.quality > 0.5
    return {
        "stream_probe_valid_rate": stream_valid.float().mean(),
        "no_stream_probe_valid_rate": no_stream_valid.float().mean(),
        "stream_probe_truncation_rate": components.stream_probe_truncated.float().mean(),
        "no_stream_probe_truncation_rate": components.no_stream_probe_truncated.float().mean(),
        "stream_harm_rate": _masked_mean(stream_harm, both_valid),
        "deep_regression_rate": _masked_mean(deep_regression, stream_valid),
        "deep_correction_rate": _masked_mean(deep_correction, stream_valid),
        "deep_regression_given_stream_correct": _masked_mean(
            deep_regression,
            stream_valid & stream_correct,
        ),
        "deep_correction_given_stream_wrong": _masked_mean(
            deep_correction,
            stream_valid & ~stream_correct,
        ),
        "stream_final_both_correct_rate": _masked_mean(
            (stream_correct & final_correct).float(),
            stream_valid,
        ),
        "stream_final_both_wrong_rate": _masked_mean(
            (~stream_correct & ~final_correct).float(),
            stream_valid,
        ),
        "mean_stream_causal_help": _masked_mean(
            components.stream_causal_help.float(),
            both_valid,
        ),
        "mean_stream_causal_harm": _masked_mean(
            components.stream_causal_harm.float(),
            both_valid,
        ),
        "mean_stream_attribution": _masked_mean(
            components.stream_attribution.float(),
            stream_valid,
        ),
        "mean_deep_attribution": _masked_mean(
            components.deep_attribution.float(),
            stream_valid,
        ),
        "mean_correction_burden": _masked_mean(
            components.correction_burden.float(),
            stream_valid,
        ),
        "hidden_stream_tokens": components.latency.hidden_token_count.float().mean(),
        "exposed_stream_tokens": components.latency.exposed_token_count.float().mean(),
        "think_rate": components.latency.think_rate.float().mean(),
    }
