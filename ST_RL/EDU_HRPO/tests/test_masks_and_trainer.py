import torch

from ST_RL.EDU_HRPO.config import EDUBLHRPOConfig
from ST_RL.EDU_HRPO.controller import ControlAction, EDUBoundaryEvent, EDUGroupRecord
from ST_RL.EDU_HRPO.masks import compose_token_advantages, stage_masks_from_lengths
from ST_RL.EDU_HRPO.rollout import (
    DeepRecord,
    EDURolloutRecord,
    ProbeRecord,
    ScoreRecord,
)
from ST_RL.EDU_HRPO.trainer import EDUBLHRPOTrainerCore, reward_batch_from_records


def test_stage_masks_apply_local_advantage_only_to_its_stage():
    masks = stage_masks_from_lengths(
        streaming_lengths=torch.tensor([2]),
        deep_lengths=torch.tensor([2]),
        answer_lengths=torch.tensor([1]),
    )
    values = compose_token_advantages(
        masks,
        streaming_advantage=torch.tensor([2.0]),
        deep_advantage=torch.tensor([4.0]),
        global_advantage=torch.tensor([1.0]),
        local_weight=0.5,
    )
    assert values.tolist() == [[2.0, 2.0, 3.0, 3.0, 1.0]]


def make_record(prompt_id, exact, stream_exact, no_stream_exact, thought_tokens):
    event = EDUBoundaryEvent(
        edu_id=0,
        text="one edu",
        input_token_count=4,
        boundary_after=4,
        emitted_at=1.0,
        visible_at=1.0,
    )
    group = EDUGroupRecord(
        edu_ids=[0],
        action=ControlAction.THINK,
        input_token_count=4,
        boundary_time=1.0,
        next_boundary_time=1.5,
        thought="reason",
        token_count=thought_tokens,
        control_entropy=0.5,
        relevance=0.5,
        slack_tokens=5,
        hidden_tokens=min(thought_tokens, 5),
        exposed_tokens=max(thought_tokens - 5, 0),
    )
    return EDURolloutRecord(
        prompt_id=prompt_id,
        full_input="question",
        answer="2",
        edu_events=[event],
        stream_groups=[group],
        stream_probe=ProbeRecord("2", stream_exact, 0.0),
        no_stream_probe=ProbeRecord("2", no_stream_exact, 0.0),
        deep=DeepRecord("deep", 4 + thought_tokens),
        final_answer="2",
        scores=ScoreRecord(exact, 0.0, 1.0, 1.0, 0.0),
    )


def test_trainer_core_builds_reward_batch_and_backpropagates():
    records = [
        make_record("p0", 1.0, 1.0, 0.0, 4),
        make_record("p0", 0.0, 1.0, 0.0, 8),
        make_record("p1", 1.0, 0.0, 1.0, 4),
        make_record("p1", 0.0, 0.0, 1.0, 8),
    ]
    reward_batch = reward_batch_from_records(
        records,
        expected_rollouts_per_prompt=2,
    )
    config = EDUBLHRPOConfig(
        generation_tokens_per_second=10.0,
        deep_budget=4.0,
        reward_tau=20.0,
        rollouts_per_prompt=2,
    )
    core = EDUBLHRPOTrainerCore(config)
    masks = stage_masks_from_lengths(
        streaming_lengths=torch.tensor([4, 8, 4, 8]),
        deep_lengths=torch.tensor([8, 12, 8, 12]),
    )
    log_probs = torch.full(masks.response.shape, -0.3, requires_grad=True)
    old_log_probs = log_probs.detach() - 0.01
    loss, metrics, advantages = core.loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        response_mask=masks.response,
        streaming_mask=masks.streaming,
        deep_mask=masks.deep,
        reward_batch=reward_batch,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["stream_harm_rate"])
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["control_entropy"] == 0.5
    assert advantages.streaming.shape == torch.Size([4])
    loss.backward()
    assert log_probs.grad is not None


def test_rollout_record_round_trip_and_group_size_validation():
    first = make_record("p0", 1.0, 1.0, 0.0, 4)
    restored = EDURolloutRecord.from_dict(first.to_dict())

    assert restored.to_dict() == first.to_dict()
    try:
        reward_batch_from_records(
            [first],
            expected_rollouts_per_prompt=2,
        )
    except ValueError as error:
        assert "exactly 2 rollouts" in str(error)
    else:
        raise AssertionError("Incomplete rollout groups must be rejected before reward computation.")
