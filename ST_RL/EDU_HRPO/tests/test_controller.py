import pytest

from ST_RL.EDU_HRPO.controller import (
    ControlAction,
    EDUBoundaryEvent,
    EDUController,
)
from ST_RL.EDU_HRPO.rollout import ControlPolicyOutput, run_controlled_stream


def make_event(index, emitted_at=None):
    visible_at = float(index + 1)
    return EDUBoundaryEvent(
        edu_id=index,
        text="edu-%d" % index,
        input_token_count=3,
        boundary_after=(index + 1) * 3,
        emitted_at=visible_at if emitted_at is None else emitted_at,
        visible_at=visible_at,
        confidence=0.9,
    )


def test_boundary_event_cannot_be_emitted_before_visible_context():
    with pytest.raises(ValueError, match="cannot precede"):
        make_event(1, emitted_at=1.5)


def test_defer_never_exceeds_merge_limit():
    controller = EDUController(max_merge=3)
    assert not controller.push(make_event(0), ControlAction.DEFER).should_emit
    assert not controller.push(make_event(1), ControlAction.DEFER).should_emit
    decision = controller.push(make_event(2), ControlAction.DEFER)

    assert decision.should_emit
    assert decision.action == ControlAction.THINK
    assert decision.forced_by_merge_limit
    record = controller.commit(decision, thought="combined", token_count=2)
    assert record.edu_ids == [0, 1, 2]
    assert record.forced_by_merge_limit
    assert controller.pending_events == ()


def test_controlled_stream_records_skip_defer_and_think_groups():
    events = [make_event(index) for index in range(3)]
    controller = EDUController(max_merge=2)
    actions = iter([ControlAction.DEFER, ControlAction.DEFER, ControlAction.SKIP])

    result = run_controlled_stream(
        events=events,
        controller=controller,
        choose_action=lambda pending, state: ControlPolicyOutput(
            action=next(actions), entropy=0.5, relevance=0.25
        ),
        generate_thought=lambda pending, state: "reason about %s" % pending[-1].text,
        token_counter=lambda text: len(text.split()),
        input_end_time=4.0,
    )

    assert [group.action for group in result.groups] == [ControlAction.THINK, ControlAction.SKIP]
    assert result.groups[0].edu_count == 2
    assert result.groups[0].next_boundary_time == events[2].emitted_at
    assert result.groups[1].next_boundary_time == 4.0
    assert len(result.stream_state) == 1

