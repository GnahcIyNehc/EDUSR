"""EDU 暴露 token、动态预算和延迟奖励。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from .config import EDUBLHRPOConfig
from .controller import ControlAction, EDUGroupRecord


@dataclass(frozen=True)
class TokenExposure:
    thought_tokens: int
    slack_tokens: int
    hidden_tokens: int
    exposed_tokens: int
    cost: float


@dataclass
class LatencyBatch:
    slack_tokens: torch.Tensor
    hidden_tokens: torch.Tensor
    exposed_tokens: torch.Tensor
    group_costs: torch.Tensor
    group_budgets: torch.Tensor
    stream_length_reward: torch.Tensor
    deep_length_reward: torch.Tensor
    stream_cost: torch.Tensor
    exposure_cost: torch.Tensor
    hidden_token_count: torch.Tensor
    exposed_token_count: torch.Tensor
    think_rate: torch.Tensor


def split_hidden_exposed_tokens(
    thought_tokens: int,
    boundary_time: float,
    next_boundary_time: float,
    generation_tokens_per_second: float,
    alpha_in: float = 0.10,
    alpha_over: float = 1.0,
    mode: str = "overlap_aware",
) -> TokenExposure:
    """按下一输入到达窗口拆分可隐藏与暴露 token。"""

    if thought_tokens < 0:
        raise ValueError("thought_tokens must be nonnegative.")
    if next_boundary_time < boundary_time:
        raise ValueError("The next boundary time cannot precede the current boundary.")
    if generation_tokens_per_second <= 0:
        raise ValueError("Generation throughput must be positive.")
    if mode not in {"overlap_aware", "uniform_small"}:
        raise ValueError("Unknown streaming cost mode.")
    slack = max(
        int(math.floor((next_boundary_time - boundary_time) * generation_tokens_per_second)),
        0,
    )
    hidden = min(int(thought_tokens), slack)
    exposed = int(thought_tokens) - hidden
    if mode == "uniform_small":
        cost = float(alpha_in) * float(thought_tokens)
    else:
        cost = float(alpha_in) * hidden + float(alpha_over) * exposed
    return TokenExposure(
        thought_tokens=int(thought_tokens),
        slack_tokens=slack,
        hidden_tokens=hidden,
        exposed_tokens=exposed,
        cost=cost,
    )


def compute_edu_budget(
    input_tokens: float,
    uncertainty: float,
    relevance: float,
    config: EDUBLHRPOConfig,
) -> float:
    """计算单个 EDU 组的免费思考预算。"""

    if input_tokens < 0:
        raise ValueError("EDU input token counts must be nonnegative.")
    if not 0.0 <= uncertainty <= 1.0 or not 0.0 <= relevance <= 1.0:
        raise ValueError("Uncertainty and relevance must be in [0, 1].")
    raw = (
        config.budget_base
        + config.budget_input_scale * input_tokens
        + config.budget_uncertainty_scale * uncertainty
        + config.budget_relevance_scale * relevance
    )
    return min(max(float(raw), config.budget_min), config.budget_max)


def annotate_group_exposure(
    groups: Sequence[EDUGroupRecord],
    config: EDUBLHRPOConfig,
) -> None:
    """把实测 token 分解回写到可审计 rollout 记录。"""

    if config.generation_tokens_per_second is None:
        raise ValueError("Measure generation_tokens_per_second first.")
    for group in groups:
        if group.next_boundary_time is None:
            raise ValueError("The EDU group is missing the next boundary or end-of-input time.")
        exposure = split_hidden_exposed_tokens(
            thought_tokens=group.token_count,
            boundary_time=group.boundary_time,
            next_boundary_time=group.next_boundary_time,
            generation_tokens_per_second=config.generation_tokens_per_second,
            alpha_in=config.alpha_in,
            alpha_over=config.alpha_over,
            mode=config.stream_cost_mode,
        )
        group.slack_tokens = exposure.slack_tokens
        group.hidden_tokens = exposure.hidden_tokens
        group.exposed_tokens = exposure.exposed_tokens


def _validate_matrix(name: str, value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if value.shape != shape:
        raise ValueError("%s must have shape %s." % (name, tuple(shape)))
    return value.float()


def compute_latency_batch(
    thought_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    group_edu_counts: torch.Tensor,
    group_mask: torch.Tensor,
    think_group_mask: torch.Tensor,
    control_uncertainty: torch.Tensor,
    relevance: torch.Tensor,
    deep_lengths: torch.Tensor,
    config: EDUBLHRPOConfig,
    boundary_times: Optional[torch.Tensor] = None,
    next_boundary_times: Optional[torch.Tensor] = None,
    slack_tokens: Optional[torch.Tensor] = None,
) -> LatencyBatch:
    """批量计算第 6 节的 EDU 预算、触发惩罚和暴露成本。"""

    if thought_lengths.dim() != 2:
        raise ValueError("thought_lengths must have shape [batch, groups].")
    shape = thought_lengths.shape
    thought_lengths = thought_lengths.float().clamp_min(0.0)
    input_lengths = _validate_matrix("input_lengths", input_lengths, shape).clamp_min(0.0)
    group_edu_counts = _validate_matrix(
        "group_edu_counts", group_edu_counts, shape
    ).clamp_min(0.0)
    uncertainty = _validate_matrix(
        "control_uncertainty", control_uncertainty, shape
    )
    relevance = _validate_matrix("relevance", relevance, shape)
    group_mask_b = _validate_matrix("group_mask", group_mask, shape).bool()
    think_mask_b = _validate_matrix("think_group_mask", think_group_mask, shape).bool()
    think_mask_b = think_mask_b & group_mask_b
    if bool(((uncertainty < 0) | (uncertainty > 1)).any().item()):
        raise ValueError("control_uncertainty must be in [0, 1].")
    if bool(((relevance < 0) | (relevance > 1)).any().item()):
        raise ValueError("relevance must be in [0, 1].")
    mask_f = group_mask_b.to(dtype=thought_lengths.dtype)
    thought_lengths = thought_lengths * mask_f

    if slack_tokens is not None:
        slack = _validate_matrix("slack_tokens", slack_tokens, shape).floor().clamp_min(0.0)
    else:
        if boundary_times is None or next_boundary_times is None:
            raise ValueError("Provide slack_tokens or complete boundary times.")
        if config.generation_tokens_per_second is None:
            raise ValueError("Measure generation_tokens_per_second first.")
        boundary_times = _validate_matrix("boundary_times", boundary_times, shape)
        next_boundary_times = _validate_matrix("next_boundary_times", next_boundary_times, shape)
        if bool(((next_boundary_times < boundary_times) & group_mask_b).any().item()):
            raise ValueError("The next boundary time cannot precede the current boundary.")
        slack = torch.floor(
            (next_boundary_times - boundary_times).clamp_min(0.0)
            * float(config.generation_tokens_per_second)
        )
    slack = slack * mask_f
    hidden = torch.minimum(thought_lengths, slack)
    exposed = (thought_lengths - hidden).clamp_min(0.0)
    if config.stream_cost_mode == "uniform_small":
        costs = float(config.alpha_in) * thought_lengths
    elif config.stream_cost_mode == "overlap_aware":
        costs = float(config.alpha_in) * hidden + float(config.alpha_over) * exposed
    else:
        raise ValueError("Unknown stream_cost_mode: %s" % config.stream_cost_mode)

    budgets = (
        float(config.budget_base)
        + float(config.budget_input_scale) * input_lengths
        + float(config.budget_uncertainty_scale) * uncertainty
        + float(config.budget_relevance_scale) * relevance
    ).clamp(min=float(config.budget_min), max=float(config.budget_max))
    budgets = budgets * mask_f

    total_input = (input_lengths * mask_f).sum(dim=-1)
    total_edus = (group_edu_counts * mask_f).sum(dim=-1)
    if bool((total_input <= 0).any().item()):
        raise ValueError("Each trajectory requires at least one EDU group with positive input length.")
    if bool((total_edus <= 0).any().item()):
        raise ValueError("Each trajectory requires at least one EDU.")
    weights = input_lengths * mask_f / total_input.unsqueeze(-1)
    over_budget = (costs - budgets).clamp_min(0.0)
    budget_penalty = -torch.log1p(over_budget)
    think_count = think_mask_b.to(dtype=thought_lengths.dtype).sum(dim=-1)
    think_rate = think_count / total_edus.clamp_min(1.0)
    stream_reward = (weights * budget_penalty).sum(dim=-1) - float(
        config.trigger_penalty
    ) * think_rate

    if deep_lengths.dim() != 1 or deep_lengths.shape[0] != shape[0]:
        raise ValueError("deep_lengths must have shape [batch].")
    if config.deep_budget is None:
        raise ValueError("Derive deep_budget from the sentence baseline.")
    deep_lengths = deep_lengths.float().clamp_min(0.0)
    deep_reward = -torch.log1p((deep_lengths - float(config.deep_budget)).clamp_min(0.0))
    stream_cost = (costs * mask_f).sum(dim=-1)
    exposure_cost = stream_cost + deep_lengths

    return LatencyBatch(
        slack_tokens=slack,
        hidden_tokens=hidden,
        exposed_tokens=exposed,
        group_costs=costs,
        group_budgets=budgets,
        stream_length_reward=stream_reward,
        deep_length_reward=deep_reward,
        stream_cost=stream_cost,
        exposure_cost=exposure_cost,
        hidden_token_count=(hidden * mask_f).sum(dim=-1),
        exposed_token_count=(exposed * mask_f).sum(dim=-1),
        think_rate=think_rate,
    )


class CorrectExposureTau:
    """仅用正确且格式合法的训练轨迹估计 tau，并在预热后冻结。"""

    def __init__(self, warmup_correct_rollouts: int = 512) -> None:
        if warmup_correct_rollouts < 1:
            raise ValueError("warmup_correct_rollouts must be positive.")
        self.warmup_correct_rollouts = int(warmup_correct_rollouts)
        self.total = 0.0
        self.count = 0
        self.frozen = False
        self._frozen_value = None  # type: Optional[float]

    def update(
        self,
        exposure_cost: torch.Tensor,
        exact: torch.Tensor,
        stream_format: torch.Tensor,
        deep_format: torch.Tensor,
    ) -> Optional[float]:
        if self.frozen:
            return self.value
        valid = (exact > 0.5) & (stream_format > 0.5) & (deep_format > 0.5)
        values = exposure_cost.detach().float()[valid].cpu().tolist()
        self.total += sum(float(item) for item in values)
        self.count += len(values)
        if self.count >= self.warmup_correct_rollouts:
            self._frozen_value = max(self.total / float(self.count), 1.0)
            self.frozen = True
        return self.current_value

    @property
    def current_value(self) -> Optional[float]:
        if self.count == 0:
            return None
        if self.frozen:
            return self._frozen_value
        return max(self.total / float(self.count), 1.0)

    @property
    def value(self) -> float:
        current = self.current_value
        if current is None:
            raise RuntimeError("No correct trajectories are available to estimate tau.")
        return float(current)

    def state_dict(self) -> Dict[str, object]:
        return {
            "warmup_correct_rollouts": self.warmup_correct_rollouts,
            "total": self.total,
            "count": self.count,
            "frozen": self.frozen,
            "frozen_value": self._frozen_value,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if int(state["warmup_correct_rollouts"]) != self.warmup_correct_rollouts:
            raise ValueError("The tau tracker warmup sample count differs from the current configuration.")
        self.total = float(state["total"])
        self.count = int(state["count"])
        self.frozen = bool(state["frozen"])
        frozen_value = state.get("frozen_value")
        self._frozen_value = None if frozen_value is None else float(frozen_value)

