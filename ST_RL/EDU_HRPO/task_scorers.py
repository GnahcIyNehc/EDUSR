"""受 exact 锚点约束的任务评分器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from ST_RL.HRPO.rewards import (
    exact_match_accuracy_reward,
    extract_boxed_answer,
    normalize_answer,
)


@dataclass(frozen=True)
class TaskScore:
    exact: float
    part: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.exact <= 1.0 or not 0.0 <= self.part <= 1.0:
            raise ValueError("Task scores must be in [0, 1].")
        if float(self.exact) not in {0.0, 1.0}:
            raise ValueError("exact must be a binary anchor; store soft scores in part.")


def _clip_score(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _answer_content(value: str) -> str:
    """优先抽取 boxed 内容，使分类和抽取任务与正式答案格式兼容。"""
    boxed = extract_boxed_answer(str(value or ""))
    return boxed if boxed is not None else str(value or "")


def _has_complete_boxed_answer(value: str) -> bool:
    """探针必须闭合 boxed，避免把截断解释或空输出计作错误答案。"""

    boxed = extract_boxed_answer(str(value or ""))
    return boxed is not None and bool(normalize_answer(boxed))


class MathTaskScorer:
    """数学 exact/符号等价锚点和可选冻结部分奖励。"""

    def __init__(
        self,
        require_boxed: bool = True,
        symbolic_equivalence_fn: Optional[Callable[[str, str], float]] = None,
        partial_score_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.require_boxed = bool(require_boxed)
        self.symbolic_equivalence_fn = symbolic_equivalence_fn
        self.partial_score_fn = partial_score_fn

    def score(self, prediction: str, answer: str) -> TaskScore:
        exact = exact_match_accuracy_reward(
            prediction,
            answer,
            prefer_boxed=True,
            require_boxed=self.require_boxed,
        )
        if exact < 1.0 and self.symbolic_equivalence_fn is not None:
            symbolic = float(self.symbolic_equivalence_fn(prediction, answer))
            if symbolic not in {0.0, 1.0}:
                raise ValueError("Symbolic equivalence scoring must return 0 or 1.")
            exact = max(exact, symbolic)
        part = 0.0
        if self.partial_score_fn is not None:
            part = _clip_score(self.partial_score_fn(prediction, answer))
        return TaskScore(exact=_clip_score(exact), part=part)

    def is_valid_probe(self, prediction: str) -> bool:
        """短探针沿用 SFT 最终答案格式，必须包含完整 boxed。"""

        return _has_complete_boxed_answer(prediction)


def token_f1_score(prediction: str, answer: str) -> float:
    """抽取式 QA 使用的确定性 token F1。"""

    pred_tokens = re.findall(r"\w+", normalize_answer(_answer_content(prediction)))
    gold_tokens = re.findall(r"\w+", normalize_answer(_answer_content(answer)))
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    remaining = list(gold_tokens)
    common = 0
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / float(len(pred_tokens))
    recall = common / float(len(gold_tokens))
    return 2.0 * precision * recall / (precision + recall)


class ExtractiveQAScorer:
    def score(self, prediction: str, answer: str) -> TaskScore:
        exact = float(
            normalize_answer(_answer_content(prediction))
            == normalize_answer(_answer_content(answer))
        )
        return TaskScore(exact=exact, part=token_f1_score(prediction, answer))

    def is_valid_probe(self, prediction: str) -> bool:
        """抽取探针也要求完整 boxed，以区分错误答案与生成失败。"""

        return _has_complete_boxed_answer(prediction)


class ClassificationScorer:
    """多选和是非任务默认不引入部分答案奖励。"""

    def score(self, prediction: str, answer: str) -> TaskScore:
        exact = float(
            normalize_answer(_answer_content(prediction))
            == normalize_answer(_answer_content(answer))
        )
        return TaskScore(exact=exact, part=0.0)

    def is_valid_probe(self, prediction: str) -> bool:
        """分类探针只要给出非空且闭合的 boxed 标签即为有效。"""

        return _has_complete_boxed_answer(prediction)
