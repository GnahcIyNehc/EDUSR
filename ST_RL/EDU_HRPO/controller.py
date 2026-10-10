"""EDU 级 skip/defer/think 状态机。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class ControlAction(str, Enum):
    SKIP = "skip"
    DEFER = "defer"
    THINK = "think"


CONTROL_TOKENS = {
    ControlAction.SKIP: "<skip>",
    ControlAction.DEFER: "<defer>",
    ControlAction.THINK: "<think>",
}


def parse_control_action(value: Any) -> ControlAction:
    """同时接受枚举、动作名和控制 token。"""

    if isinstance(value, ControlAction):
        return value
    text = str(value).strip().lower()
    for action, token in CONTROL_TOKENS.items():
        if text in {action.value, token.lower()}:
            return action
    raise ValueError("Unknown EDU control action: %s" % value)


@dataclass
class EDUBoundaryEvent:
    """在线分割器交给控制器的最小边界事件。"""

    edu_id: int
    text: str
    input_token_count: int
    boundary_after: int
    emitted_at: float
    visible_at: float
    confidence: float = 1.0
    forced: bool = False

    def __post_init__(self) -> None:
        if self.edu_id < 0 or self.input_token_count < 0 or self.boundary_after < 0:
            raise ValueError("EDU identifiers and token counts must be nonnegative.")
        if self.emitted_at < self.visible_at:
            raise ValueError("BoundaryEvent cannot precede its visible context.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Boundary confidence must be in [0, 1].")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EDUBoundaryEvent":
        emitted_at = float(value["emitted_at"])
        return cls(
            edu_id=int(value.get("edu_id", value.get("segment_index", 0))),
            text=str(value.get("text", value.get("completed_edu", ""))),
            input_token_count=int(value.get("input_token_count", value.get("token_count", 0))),
            boundary_after=int(value.get("boundary_after", 0)),
            emitted_at=emitted_at,
            visible_at=float(value.get("visible_at", value.get("earliest_emit_at", emitted_at))),
            confidence=float(value.get("confidence", value.get("boundary_score", 1.0) or 0.0)),
            forced=bool(value.get("forced", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edu_id": self.edu_id,
            "text": self.text,
            "input_token_count": self.input_token_count,
            "boundary_after": self.boundary_after,
            "emitted_at": self.emitted_at,
            "visible_at": self.visible_at,
            "confidence": self.confidence,
            "forced": self.forced,
        }


@dataclass
class ControlDecision:
    """控制器解析 defer 上限后的实际动作。"""

    requested_action: ControlAction
    action: ControlAction
    edu_ids: Tuple[int, ...]
    should_emit: bool
    forced_by_merge_limit: bool = False


@dataclass
class EDUGroupRecord:
    """一个动态 EDU 组及其流式行为。"""

    edu_ids: List[int]
    action: ControlAction
    input_token_count: int
    boundary_time: float
    next_boundary_time: Optional[float]
    thought: str = ""
    token_count: int = 0
    control_entropy: float = 0.0
    relevance: float = 0.0
    forced_by_merge_limit: bool = False
    slack_tokens: Optional[int] = None
    hidden_tokens: Optional[int] = None
    exposed_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        self.action = parse_control_action(self.action)
        if not self.edu_ids:
            raise ValueError("An EDU group must not be empty.")
        if self.input_token_count < 0 or self.token_count < 0:
            raise ValueError("Group token counts must be nonnegative.")
        if self.action != ControlAction.THINK and self.token_count != 0:
            raise ValueError("skip/defer groups must not contain thought tokens.")
        if not 0.0 <= self.control_entropy <= 1.0:
            raise ValueError("Control action entropy must be in [0, 1].")
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError("Semantic relevance must be in [0, 1].")
        if self.next_boundary_time is not None and self.next_boundary_time < self.boundary_time:
            raise ValueError("The next boundary time cannot precede the current boundary.")

    @property
    def edu_count(self) -> int:
        return len(self.edu_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edu_ids": list(self.edu_ids),
            "action": self.action.value,
            "input_token_count": self.input_token_count,
            "boundary_time": self.boundary_time,
            "next_boundary_time": self.next_boundary_time,
            "thought": self.thought,
            "token_count": self.token_count,
            "control_entropy": self.control_entropy,
            "relevance": self.relevance,
            "forced_by_merge_limit": self.forced_by_merge_limit,
            "slack_tokens": self.slack_tokens,
            "hidden_tokens": self.hidden_tokens,
            "exposed_tokens": self.exposed_tokens,
        }


class EDUController:
    """维护 EDU 缓冲区，并硬性限制连续 defer 的合并规模。"""

    def __init__(self, max_merge: int = 3, merge_limit_action: Any = ControlAction.THINK):
        if max_merge < 1:
            raise ValueError("max_merge must be at least 1.")
        fallback = parse_control_action(merge_limit_action)
        if fallback == ControlAction.DEFER:
            raise ValueError("The action cannot remain defer after reaching the merge limit.")
        self.max_merge = int(max_merge)
        self.merge_limit_action = fallback
        self._buffer = []  # type: List[EDUBoundaryEvent]
        self.groups = []  # type: List[EDUGroupRecord]

    @property
    def pending_events(self) -> Tuple[EDUBoundaryEvent, ...]:
        return tuple(self._buffer)

    def push(self, event: EDUBoundaryEvent, requested_action: Any) -> ControlDecision:
        """接收一个合法边界，并返回等待或提交组的决定。"""

        if self.groups and self.groups[-1].next_boundary_time is None:
            self.groups[-1].next_boundary_time = event.emitted_at
        self._buffer.append(event)
        requested = parse_control_action(requested_action)
        if requested == ControlAction.DEFER and len(self._buffer) < self.max_merge:
            return ControlDecision(
                requested_action=requested,
                action=ControlAction.DEFER,
                edu_ids=tuple(item.edu_id for item in self._buffer),
                should_emit=False,
            )
        forced = requested == ControlAction.DEFER
        action = self.merge_limit_action if forced else requested
        return ControlDecision(
            requested_action=requested,
            action=action,
            edu_ids=tuple(item.edu_id for item in self._buffer),
            should_emit=True,
            forced_by_merge_limit=forced,
        )

    def commit(
        self,
        decision: ControlDecision,
        thought: str = "",
        token_count: int = 0,
        control_entropy: float = 0.0,
        relevance: float = 0.0,
    ) -> EDUGroupRecord:
        """提交非 defer 决定并清空缓冲区。"""

        current_ids = tuple(item.edu_id for item in self._buffer)
        if not decision.should_emit or decision.action == ControlAction.DEFER:
            raise ValueError("A pending defer decision cannot be committed.")
        if decision.edu_ids != current_ids:
            raise ValueError("The commit decision does not match the current EDU buffer.")
        if decision.action == ControlAction.SKIP:
            thought = ""
            token_count = 0
        record = EDUGroupRecord(
            edu_ids=list(current_ids),
            action=decision.action,
            input_token_count=sum(item.input_token_count for item in self._buffer),
            boundary_time=max(item.emitted_at for item in self._buffer),
            next_boundary_time=None,
            thought=str(thought),
            token_count=int(token_count),
            control_entropy=float(control_entropy),
            relevance=float(relevance),
            forced_by_merge_limit=decision.forced_by_merge_limit,
        )
        self.groups.append(record)
        self._buffer = []
        return record

    def flush(
        self,
        input_end_time: float,
        action: Any = ControlAction.THINK,
        thought: str = "",
        token_count: int = 0,
        control_entropy: float = 0.0,
        relevance: float = 0.0,
    ) -> Optional[EDUGroupRecord]:
        """输入结束时提交剩余 EDU，并闭合最后一个暴露窗口。"""

        if self.groups and self.groups[-1].next_boundary_time is None:
            self.groups[-1].next_boundary_time = float(input_end_time)
        if not self._buffer:
            return None
        resolved = parse_control_action(action)
        if resolved == ControlAction.DEFER:
            resolved = self.merge_limit_action
        decision = ControlDecision(
            requested_action=parse_control_action(action),
            action=resolved,
            edu_ids=tuple(item.edu_id for item in self._buffer),
            should_emit=True,
            forced_by_merge_limit=parse_control_action(action) == ControlAction.DEFER,
        )
        record = self.commit(
            decision,
            thought=thought,
            token_count=token_count,
            control_entropy=control_entropy,
            relevance=relevance,
        )
        record.next_boundary_time = float(input_end_time)
        return record

