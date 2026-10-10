"""EDU 动态控制循环和可审计 rollout 记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .controller import (
    ControlAction,
    EDUBoundaryEvent,
    EDUController,
    EDUGroupRecord,
    parse_control_action,
)


@dataclass(frozen=True)
class ControlPolicyOutput:
    action: ControlAction
    entropy: float = 0.0
    relevance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", parse_control_action(self.action))
        if not 0.0 <= self.entropy <= 1.0 or not 0.0 <= self.relevance <= 1.0:
            raise ValueError("Control entropy and relevance must be in [0, 1].")


@dataclass(frozen=True)
class ProbeRecord:
    answer: str
    exact: float
    part: float = 0.0
    valid: bool = True
    finish_reason: str = ""
    token_count: int = 0
    truncated: bool = False

    @property
    def q(self) -> float:
        return self.exact


@dataclass(frozen=True)
class DeepRecord:
    text: str
    token_count: int


@dataclass(frozen=True)
class ScoreRecord:
    exact: float
    part: float
    stream_format: float
    deep_format: float
    stream_verifier: float = 0.0


@dataclass(frozen=True)
class LatencyRecord:
    first_thought: float = 0.0
    post_input_ttfa: float = 0.0
    total: float = 0.0


@dataclass
class EDURolloutRecord:
    """训练奖励所需的完整轨迹，不能退化成只有最终文本。"""

    prompt_id: str
    full_input: str
    answer: str
    edu_events: List[EDUBoundaryEvent]
    stream_groups: List[EDUGroupRecord]
    stream_probe: ProbeRecord
    no_stream_probe: ProbeRecord
    deep: DeepRecord
    final_answer: str
    scores: ScoreRecord
    latency: LatencyRecord = field(default_factory=LatencyRecord)

    def validate(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must not be empty.")
        event_id_list = [event.edu_id for event in self.edu_events]
        event_ids = set(event_id_list)
        if len(event_id_list) != len(event_ids):
            raise ValueError("BoundaryEvent edu_id values must be unique within a rollout.")
        grouped_ids = [edu_id for group in self.stream_groups for edu_id in group.edu_ids]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise ValueError("The same EDU cannot belong to multiple dynamic groups.")
        if set(grouped_ids) != event_ids:
            raise ValueError("Each BoundaryEvent must belong to exactly one dynamic group.")
        for score in (
            self.stream_probe.exact,
            self.stream_probe.part,
            self.no_stream_probe.exact,
            self.no_stream_probe.part,
            self.scores.exact,
            self.scores.part,
            self.scores.stream_format,
            self.scores.deep_format,
            self.scores.stream_verifier,
        ):
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError("Rollout scores must be in [0, 1].")
        if self.deep.token_count < 0:
            raise ValueError("Deep token counts must be nonnegative.")
        for probe in (self.stream_probe, self.no_stream_probe):
            if probe.token_count < 0:
                raise ValueError("Counterfactual probe token counts must be nonnegative.")
            if not isinstance(probe.valid, bool) or not isinstance(probe.truncated, bool):
                raise ValueError("Counterfactual probe valid/truncated fields must be booleans.")
        for group in self.stream_groups:
            if group.next_boundary_time is None:
                raise ValueError("Each EDU group must record its next boundary or end-of-input time.")
            if group.hidden_tokens is not None or group.exposed_tokens is not None:
                hidden = int(group.hidden_tokens or 0)
                exposed = int(group.exposed_tokens or 0)
                if hidden + exposed != group.token_count:
                    raise ValueError("hidden_tokens + exposed_tokens must equal the thought token count.")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "prompt_id": self.prompt_id,
            "full_input": self.full_input,
            "answer": self.answer,
            "edu_events": [event.to_dict() for event in self.edu_events],
            "stream_groups": [group.to_dict() for group in self.stream_groups],
            "stream_probe": {
                "answer": self.stream_probe.answer,
                "exact": self.stream_probe.exact,
                "part": self.stream_probe.part,
                "valid": self.stream_probe.valid,
                "finish_reason": self.stream_probe.finish_reason,
                "token_count": self.stream_probe.token_count,
                "truncated": self.stream_probe.truncated,
            },
            "no_stream_probe": {
                "answer": self.no_stream_probe.answer,
                "exact": self.no_stream_probe.exact,
                "part": self.no_stream_probe.part,
                "valid": self.no_stream_probe.valid,
                "finish_reason": self.no_stream_probe.finish_reason,
                "token_count": self.no_stream_probe.token_count,
                "truncated": self.no_stream_probe.truncated,
            },
            "deep": {"text": self.deep.text, "token_count": self.deep.token_count},
            "final_answer": self.final_answer,
            "scores": {
                "exact": self.scores.exact,
                "part": self.scores.part,
                "stream_format": self.scores.stream_format,
                "deep_format": self.scores.deep_format,
                "stream_verifier": self.scores.stream_verifier,
            },
            "latency": {
                "first_thought": self.latency.first_thought,
                "post_input_ttfa": self.latency.post_input_ttfa,
                "total": self.latency.total,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EDURolloutRecord":
        groups = []
        for item in value.get("stream_groups", []):
            groups.append(
                EDUGroupRecord(
                    edu_ids=[int(x) for x in item["edu_ids"]],
                    action=parse_control_action(item["action"]),
                    input_token_count=int(item.get("input_token_count", 0)),
                    boundary_time=float(item["boundary_time"]),
                    next_boundary_time=float(item["next_boundary_time"]),
                    thought=str(item.get("thought", "")),
                    token_count=int(item.get("token_count", 0)),
                    control_entropy=float(item.get("control_entropy", 0.0)),
                    relevance=float(item.get("relevance", 0.0)),
                    forced_by_merge_limit=bool(item.get("forced_by_merge_limit", False)),
                    slack_tokens=item.get("slack_tokens"),
                    hidden_tokens=item.get("hidden_tokens"),
                    exposed_tokens=item.get("exposed_tokens"),
                )
            )
        stream_probe = value["stream_probe"]
        no_stream_probe = value["no_stream_probe"]
        deep = value["deep"]
        scores = value["scores"]
        latency = value.get("latency", {})
        record = cls(
            prompt_id=str(value["prompt_id"]),
            full_input=str(value.get("full_input", "")),
            answer=str(value.get("answer", "")),
            edu_events=[EDUBoundaryEvent.from_mapping(item) for item in value.get("edu_events", [])],
            stream_groups=groups,
            stream_probe=ProbeRecord(
                answer=str(stream_probe.get("answer", "")),
                exact=float(stream_probe.get("exact", stream_probe.get("q", 0.0))),
                part=float(stream_probe.get("part", 0.0)),
                valid=bool(stream_probe.get("valid", True)),
                finish_reason=str(stream_probe.get("finish_reason", "")),
                token_count=int(stream_probe.get("token_count", 0)),
                truncated=bool(stream_probe.get("truncated", False)),
            ),
            no_stream_probe=ProbeRecord(
                answer=str(no_stream_probe.get("answer", "")),
                exact=float(no_stream_probe.get("exact", no_stream_probe.get("q", 0.0))),
                part=float(no_stream_probe.get("part", 0.0)),
                valid=bool(no_stream_probe.get("valid", True)),
                finish_reason=str(no_stream_probe.get("finish_reason", "")),
                token_count=int(no_stream_probe.get("token_count", 0)),
                truncated=bool(no_stream_probe.get("truncated", False)),
            ),
            deep=DeepRecord(text=str(deep.get("text", "")), token_count=int(deep["token_count"])),
            final_answer=str(value.get("final_answer", "")),
            scores=ScoreRecord(
                exact=float(scores["exact"]),
                part=float(scores.get("part", 0.0)),
                stream_format=float(scores.get("stream_format", 0.0)),
                deep_format=float(scores.get("deep_format", 0.0)),
                stream_verifier=float(scores.get("stream_verifier", 0.0)),
            ),
            latency=LatencyRecord(
                first_thought=float(latency.get("first_thought", 0.0)),
                post_input_ttfa=float(latency.get("post_input_ttfa", 0.0)),
                total=float(latency.get("total", 0.0)),
            ),
        )
        record.validate()
        return record


@dataclass(frozen=True)
class ControlledStreamResult:
    events: Tuple[EDUBoundaryEvent, ...]
    groups: Tuple[EDUGroupRecord, ...]
    stream_state: Tuple[str, ...]


def run_controlled_stream(
    events: Sequence[EDUBoundaryEvent],
    controller: EDUController,
    choose_action: Callable[[Sequence[EDUBoundaryEvent], Sequence[str]], ControlPolicyOutput],
    generate_thought: Callable[[Sequence[EDUBoundaryEvent], Sequence[str]], str],
    token_counter: Callable[[str], int],
    input_end_time: float,
) -> ControlledStreamResult:
    """执行动态 EDU 控制，模型推理通过回调注入。"""

    stream_state = []  # type: List[str]
    for event in events:
        proposal = choose_action(controller.pending_events + (event,), tuple(stream_state))
        decision = controller.push(event, proposal.action)
        if not decision.should_emit:
            continue
        thought = ""
        token_count = 0
        if decision.action == ControlAction.THINK:
            thought = str(generate_thought(controller.pending_events, tuple(stream_state)))
            token_count = int(token_counter(thought))
            stream_state.append(thought)
        controller.commit(
            decision,
            thought=thought,
            token_count=token_count,
            control_entropy=proposal.entropy,
            relevance=proposal.relevance,
        )

    if controller.pending_events:
        proposal = choose_action(controller.pending_events, tuple(stream_state))
        action = proposal.action
        if action == ControlAction.DEFER:
            action = controller.merge_limit_action
        thought = ""
        token_count = 0
        if action == ControlAction.THINK:
            thought = str(generate_thought(controller.pending_events, tuple(stream_state)))
            token_count = int(token_counter(thought))
            stream_state.append(thought)
        controller.flush(
            input_end_time=float(input_end_time),
            action=action,
            thought=thought,
            token_count=token_count,
            control_entropy=proposal.entropy,
            relevance=proposal.relevance,
        )
    elif controller.groups and controller.groups[-1].next_boundary_time is None:
        controller.groups[-1].next_boundary_time = float(input_end_time)

    return ControlledStreamResult(
        events=tuple(events),
        groups=tuple(controller.groups),
        stream_state=tuple(stream_state),
    )
