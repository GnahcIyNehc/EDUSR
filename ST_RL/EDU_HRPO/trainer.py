"""把 EDU-BLHRPO 奖励接入现有 stage-level HRPO loss。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from ST_RL.HRPO import HRPOConfig, hrpo_loss

from .config import EDUBLHRPOConfig
from .controller import ControlAction
from .latency import CorrectExposureTau, compute_latency_batch
from .rewards import (
    EDUBLHRPOAdvantages,
    EDURewardBatch,
    attribution_diagnostics,
    compute_edu_blhrpo_advantages,
)
from .rollout import EDURolloutRecord


def _tensor(values, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _valid_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """只汇总结构有效的反事实答案。"""

    valid_float = valid.float()
    return (values.float() * valid_float).sum() / valid_float.sum().clamp_min(1.0)


def reward_batch_from_records(
    records: Sequence[EDURolloutRecord],
    device: Optional[torch.device] = None,
    expected_rollouts_per_prompt: Optional[int] = None,
) -> EDURewardBatch:
    """把可审计 rollout 记录补齐成批量奖励张量。"""

    if not records:
        raise ValueError("Rollout records must not be empty.")
    if expected_rollouts_per_prompt is not None and expected_rollouts_per_prompt < 2:
        raise ValueError("Each prompt must expect at least 2 rollouts.")
    for record in records:
        record.validate()
    prompt_indices = {}  # type: Dict[str, int]
    no_stream_scores = {}  # type: Dict[str, Tuple[str, float, float, bool, str, int, bool]]
    prompt_bases = {}  # type: Dict[str, Tuple[str, str, int]]
    group_ids = []
    for record in records:
        if record.prompt_id not in prompt_indices:
            prompt_indices[record.prompt_id] = len(prompt_indices)
        group_ids.append(prompt_indices[record.prompt_id])
        current = (
            record.no_stream_probe.answer,
            record.no_stream_probe.exact,
            record.no_stream_probe.part,
            record.no_stream_probe.valid,
            record.no_stream_probe.finish_reason,
            record.no_stream_probe.token_count,
            record.no_stream_probe.truncated,
        )
        previous = no_stream_scores.get(record.prompt_id)
        if previous is not None and previous != current:
            raise ValueError("Shared no-stream answers or scores differ for the same prompt.")
        no_stream_scores[record.prompt_id] = current
        base = (record.full_input, record.answer, len(record.edu_events))
        previous_base = prompt_bases.get(record.prompt_id)
        if previous_base is not None and previous_base != base:
            raise ValueError("Inputs, gold answers, or EDU counts differ for the same prompt_id.")
        prompt_bases[record.prompt_id] = base

    if expected_rollouts_per_prompt is not None:
        counts = {prompt_id: 0 for prompt_id in prompt_indices}
        for record in records:
            counts[record.prompt_id] += 1
        invalid = {
            prompt_id: count
            for prompt_id, count in counts.items()
            if count != expected_rollouts_per_prompt
        }
        if invalid:
            details = ", ".join(
                "%s=%d" % item for item in sorted(invalid.items())
            )
            raise ValueError(
                "Each prompt must have exactly %d rollouts; got %s."
                % (expected_rollouts_per_prompt, details)
            )

    max_groups = max(len(record.stream_groups) for record in records)
    if max_groups < 1:
        raise ValueError("Each rollout requires at least one EDU group.")

    def rows(default=0.0):
        return [[default for _ in range(max_groups)] for _ in records]

    thought_lengths = rows()
    input_lengths = rows()
    edu_counts = rows()
    group_mask = rows()
    think_mask = rows()
    uncertainty = rows()
    relevance = rows()
    boundary_times = rows()
    next_boundary_times = rows()
    slack = rows()
    all_have_slack = True
    for row_index, record in enumerate(records):
        for group_index, group in enumerate(record.stream_groups):
            thought_lengths[row_index][group_index] = float(group.token_count)
            input_lengths[row_index][group_index] = float(group.input_token_count)
            edu_counts[row_index][group_index] = float(group.edu_count)
            group_mask[row_index][group_index] = 1.0
            think_mask[row_index][group_index] = float(group.action == ControlAction.THINK)
            uncertainty[row_index][group_index] = float(group.control_entropy)
            relevance[row_index][group_index] = float(group.relevance)
            boundary_times[row_index][group_index] = float(group.boundary_time)
            next_boundary_times[row_index][group_index] = float(group.next_boundary_time)
            if group.slack_tokens is None:
                all_have_slack = False
            else:
                slack[row_index][group_index] = float(group.slack_tokens)

    batch = EDURewardBatch(
        group_ids=_tensor(group_ids, device, dtype=torch.long),
        exact=_tensor([record.scores.exact for record in records], device),
        part=_tensor([record.scores.part for record in records], device),
        stream_probe_exact=_tensor(
            [record.stream_probe.exact for record in records], device
        ),
        stream_probe_part=_tensor(
            [record.stream_probe.part for record in records], device
        ),
        no_stream_exact=_tensor(
            [record.no_stream_probe.exact for record in records], device
        ),
        no_stream_part=_tensor(
            [record.no_stream_probe.part for record in records], device
        ),
        stream_probe_valid=_tensor(
            [record.stream_probe.valid for record in records], device
        ),
        no_stream_probe_valid=_tensor(
            [record.no_stream_probe.valid for record in records], device
        ),
        stream_probe_truncated=_tensor(
            [record.stream_probe.truncated for record in records], device
        ),
        no_stream_probe_truncated=_tensor(
            [record.no_stream_probe.truncated for record in records], device
        ),
        stream_verifier=_tensor(
            [record.scores.stream_verifier for record in records], device
        ),
        stream_format=_tensor(
            [record.scores.stream_format for record in records], device
        ),
        deep_format=_tensor([record.scores.deep_format for record in records], device),
        deep_lengths=_tensor([record.deep.token_count for record in records], device),
        thought_lengths=_tensor(thought_lengths, device),
        input_lengths=_tensor(input_lengths, device),
        group_edu_counts=_tensor(edu_counts, device),
        group_mask=_tensor(group_mask, device),
        think_group_mask=_tensor(think_mask, device),
        control_uncertainty=_tensor(uncertainty, device),
        relevance=_tensor(relevance, device),
        boundary_times=_tensor(boundary_times, device),
        next_boundary_times=_tensor(next_boundary_times, device),
        slack_tokens=_tensor(slack, device) if all_have_slack else None,
    )
    batch.validate()
    return batch


class EDUBLHRPOTrainerCore:
    """不绑定 rollout 引擎或 FSDP，负责奖励、优势和 HRPO loss。"""

    def __init__(self, config: EDUBLHRPOConfig) -> None:
        config.validate(require_derived=True)
        self.config = config
        self.tau_tracker = CorrectExposureTau(config.tau_warmup_correct_rollouts)

    def compute_advantages(self, batch: EDURewardBatch) -> Tuple[EDUBLHRPOAdvantages, float]:
        latency = compute_latency_batch(
            thought_lengths=batch.thought_lengths,
            input_lengths=batch.input_lengths,
            group_edu_counts=batch.group_edu_counts,
            group_mask=batch.group_mask,
            think_group_mask=batch.think_group_mask,
            control_uncertainty=batch.control_uncertainty,
            relevance=batch.relevance,
            deep_lengths=batch.deep_lengths,
            config=self.config,
            boundary_times=batch.boundary_times,
            next_boundary_times=batch.next_boundary_times,
            slack_tokens=batch.slack_tokens,
        )
        if self.config.reward_tau is not None:
            tau = float(self.config.reward_tau)
        else:
            tau_value = self.tau_tracker.update(
                latency.exposure_cost,
                batch.exact,
                batch.stream_format,
                batch.deep_format,
            )
            if tau_value is None:
                raise RuntimeError("The batch has no correct, well-formatted trajectories to initialize tau.")
            tau = float(tau_value)
        advantages = compute_edu_blhrpo_advantages(
            batch=batch,
            config=self.config,
            reward_tau=tau,
            latency=latency,
        )
        return advantages, tau

    def loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        streaming_mask: torch.Tensor,
        deep_mask: torch.Tensor,
        reward_batch: EDURewardBatch,
        ref_log_probs: Optional[torch.Tensor] = None,
    ):
        """计算改进优势，并复用已经验证的 HRPO clipped loss。"""

        advantages, tau = self.compute_advantages(reward_batch)
        loss_config = HRPOConfig(
            clip_range=self.config.clip_range,
            kl_coef=self.config.kl_coef,
            local_weight=self.config.local_weight,
            normalize_eps=self.config.normalize_eps,
        )
        loss, metrics = hrpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            streaming_mask=streaming_mask,
            deep_mask=deep_mask,
            streaming_advantages=advantages.streaming,
            deep_advantages=advantages.deep,
            global_advantages=advantages.global_,
            ref_log_probs=ref_log_probs,
            config=loss_config,
        )
        metrics = dict(metrics)
        metrics.update(attribution_diagnostics(advantages))
        metrics.update(
            {
                "reward_tau": torch.tensor(tau, device=log_probs.device),
                "answer_accuracy": reward_batch.exact.float().mean(),
                "partial_score": reward_batch.part.float().mean(),
                "stream_probe_accuracy": _valid_mean(
                    reward_batch.stream_probe_exact,
                    reward_batch.stream_probe_valid,
                ),
                "no_stream_accuracy": _valid_mean(
                    reward_batch.no_stream_exact,
                    reward_batch.no_stream_probe_valid,
                ),
                "stream_format_reward": reward_batch.stream_format.float().mean(),
                "deep_format_reward": reward_batch.deep_format.float().mean(),
                "control_entropy": (
                    reward_batch.control_uncertainty.float()
                    * reward_batch.group_mask.float()
                ).sum()
                / reward_batch.group_mask.float().sum().clamp_min(1.0),
                "stream_length_reward": advantages.components.stream_length_reward.mean(),
                "deep_length_reward": advantages.components.deep_length_reward.mean(),
                "efficiency_reward": advantages.components.efficiency_reward.mean(),
                "exposure_cost": advantages.components.latency.exposure_cost.mean(),
            }
        )
        return loss, metrics, advantages

    def state_dict(self) -> Dict[str, object]:
        """优化器之外至少持久化被冻结的 tau，保证续训奖励一致。"""

        return {"tau_tracker": self.tau_tracker.state_dict()}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.tau_tracker.load_state_dict(state["tau_tracker"])
