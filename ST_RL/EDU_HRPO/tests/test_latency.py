import torch

from ST_RL.EDU_HRPO.config import EDUBLHRPOConfig
from ST_RL.EDU_HRPO.latency import (
    CorrectExposureTau,
    compute_edu_budget,
    compute_latency_batch,
    split_hidden_exposed_tokens,
)


def make_config(mode="overlap_aware"):
    return EDUBLHRPOConfig(
        stream_cost_mode=mode,
        generation_tokens_per_second=10.0,
        deep_budget=4.0,
        budget_min=0.0,
        budget_max=64.0,
        budget_base=0.0,
        budget_input_scale=0.0,
        budget_uncertainty_scale=0.0,
        budget_relevance_scale=0.0,
    )


def test_hidden_and_exposed_tokens_exactly_partition_thought():
    exposure = split_hidden_exposed_tokens(
        thought_tokens=12,
        boundary_time=1.0,
        next_boundary_time=1.7,
        generation_tokens_per_second=10.0,
    )
    assert exposure.slack_tokens == 7
    assert exposure.hidden_tokens == 7
    assert exposure.exposed_tokens == 5
    assert exposure.hidden_tokens + exposure.exposed_tokens == exposure.thought_tokens
    assert abs(exposure.cost - 5.7) < 1e-6


def test_overlap_aware_penalizes_exposed_tokens_more_than_uniform_small():
    kwargs = dict(
        thought_lengths=torch.tensor([[12.0]]),
        input_lengths=torch.tensor([[4.0]]),
        group_edu_counts=torch.tensor([[1.0]]),
        group_mask=torch.tensor([[1.0]]),
        think_group_mask=torch.tensor([[1.0]]),
        control_uncertainty=torch.tensor([[0.0]]),
        relevance=torch.tensor([[0.0]]),
        deep_lengths=torch.tensor([4.0]),
        slack_tokens=torch.tensor([[7.0]]),
    )
    overlap = compute_latency_batch(config=make_config("overlap_aware"), **kwargs)
    uniform = compute_latency_batch(config=make_config("uniform_small"), **kwargs)

    assert overlap.hidden_tokens.item() + overlap.exposed_tokens.item() == 12.0
    assert overlap.group_costs.item() == torch.tensor(5.7).item()
    assert uniform.group_costs.item() == torch.tensor(1.2).item()
    assert overlap.stream_length_reward.item() < uniform.stream_length_reward.item()


def test_dynamic_budget_uses_input_uncertainty_and_relevance_then_clips():
    config = EDUBLHRPOConfig(
        budget_min=4,
        budget_max=20,
        budget_base=4,
        budget_input_scale=0.5,
        budget_uncertainty_scale=4,
        budget_relevance_scale=8,
    )
    assert compute_edu_budget(4, 0.5, 0.5, config) == 12.0
    assert compute_edu_budget(100, 1.0, 1.0, config) == 20.0


def test_tau_uses_only_correct_format_valid_rollouts_and_freezes():
    tracker = CorrectExposureTau(warmup_correct_rollouts=2)
    value = tracker.update(
        exposure_cost=torch.tensor([10.0, 100.0, 20.0]),
        exact=torch.tensor([1.0, 0.0, 1.0]),
        stream_format=torch.tensor([1.0, 1.0, 1.0]),
        deep_format=torch.tensor([1.0, 1.0, 1.0]),
    )
    assert value == 15.0
    assert tracker.frozen
    tracker.update(
        torch.tensor([1000.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )
    assert tracker.value == 15.0

