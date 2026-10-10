from __future__ import annotations

import copy
import bisect
import hashlib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import IterableDataset

from dataloader_hf import StreamingDataCollator


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent
STREAMING_EDU_SRC = WORKSPACE_ROOT / "Streaming_EDU" / "src"
BOUNDARY_RIGHT_SPACE_RE = re.compile(r"<(?:EOE|EOS)>\s*")
FIXED_PUNCTUATION_MARKS = frozenset({",", ";", ":", ".", "?", "!"})
SENTENCE_PUNCTUATION_MARKS = frozenset({".", "?", "!"})
DETERMINISTIC_BOUNDARY_DETECTORS = frozenset(
    {"sentence_punctuation", "fixed_punctuation"}
)
ASR_DROPOUT_PUNCTUATION = frozenset({",", ".", "，", "。"})
ASR_DROPOUT_SEED = 42
EOS_AFTER_PERIOD_RE = re.compile(r"[\s\"'”’）)\]]*?(<EOS>)")
FIXED_PUNCTUATION_NUMERIC_MARKS = frozenset({",", ".", ":"})
FIXED_PUNCTUATION_ABBREVIATIONS = frozenset(
    {
        "dr",
        "etc",
        "fig",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vs",
    }
)

if str(STREAMING_EDU_SRC) not in sys.path:
    sys.path.insert(0, str(STREAMING_EDU_SRC))

try:
    from final_edu import EDUSegmenter
    from final_edu.segmenter import TOKEN_PATTERN
except ImportError as error:  # pragma: no cover - environment error path
    raise ImportError(
        f"Could not import final_edu from {STREAMING_EDU_SRC}. "
        "Keep Streaming_EDU next to StreamingThinker or install its wheel."
    ) from error


def normalize_eos_spacing(text: str) -> str:
    """Ensure EDU markers are followed by a prompt-token separator."""

    return BOUNDARY_RIGHT_SPACE_RE.sub(lambda match: f"{match.group(0)[:5]} ", text).rstrip()


def _is_asr_dropout_candidate(text: str, index: int) -> bool:
    """判断字符是否属于可模拟缺失的逗号或句号。"""

    punctuation = text[index]
    if punctuation not in ASR_DROPOUT_PUNCTUATION:
        return False
    if punctuation == ".":
        left = text[index - 1] if index > 0 else ""
        right = text[index + 1] if index + 1 < len(text) else ""
        if left.isdigit() and right.isdigit():
            return False
    return True


def _should_drop_punctuation(
    *,
    sample_key: str,
    source: str,
    character_index: int,
    percent: float,
) -> bool:
    """使用稳定哈希决定是否删除标点，保证不同读取策略得到同一份扰动输入。"""

    if percent <= 0:
        return False
    if percent >= 100:
        return True
    payload = (
        f"{ASR_DROPOUT_SEED}\0{sample_key}\0{source}\0{character_index}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value < (percent / 100.0) * (1 << 64)


def apply_asr_punctuation_dropout(
    *,
    instruct: str,
    context: str,
    percent: float,
) -> tuple[str, str, dict]:
    """模拟 ASR 丢失逗号和句号，并移除被删句号对应的源句界。"""

    if not 0.0 <= percent <= 100.0:
        raise ValueError("punctuation_dropout_percent must be between 0 and 100")

    sample_key = hashlib.blake2b(
        f"{instruct}\0{context}".encode("utf-8"), digest_size=16
    ).hexdigest()
    totals = {
        "punctuation_dropout_percent": float(percent),
        "punctuation_dropout_seed": ASR_DROPOUT_SEED,
        "punctuation_candidates": 0,
        "punctuation_dropped": 0,
        "comma_candidates": 0,
        "commas_dropped": 0,
        "period_candidates": 0,
        "periods_dropped": 0,
        "source_eos_removed_by_punctuation_dropout": 0,
    }

    def perturb(text: str, source: str) -> str:
        removed_indices: set[int] = set()
        replacement_indices: dict[int, str] = {}
        for index, character in enumerate(text):
            if not _is_asr_dropout_candidate(text, index):
                continue
            totals["punctuation_candidates"] += 1
            punctuation_kind = (
                "comma" if character in {",", "，"} else "period"
            )
            totals[f"{punctuation_kind}_candidates"] += 1
            if not _should_drop_punctuation(
                sample_key=sample_key,
                source=source,
                character_index=index,
                percent=percent,
            ):
                continue

            removed_indices.add(index)
            totals["punctuation_dropped"] += 1
            totals[f"{punctuation_kind}s_dropped"] += 1
            if punctuation_kind != "period":
                continue

            # 句号缺失时同步去掉紧邻的人工 <EOS>，避免泄露原始句子边界。
            eos_match = EOS_AFTER_PERIOD_RE.match(text, index + 1)
            if eos_match is None:
                continue
            eos_start = eos_match.start(1)
            removed_indices.update(range(eos_start, eos_start + len("<EOS>")))
            replacement_indices[eos_start] = " "
            totals["source_eos_removed_by_punctuation_dropout"] += 1

        if not removed_indices:
            return text
        pieces = []
        for index, character in enumerate(text):
            if index in replacement_indices:
                pieces.append(replacement_indices[index])
            elif index not in removed_indices:
                pieces.append(character)
        return "".join(pieces)

    perturbed_instruct = perturb(instruct, "instruct")
    perturbed_context = perturb(context, "context")
    candidate_count = totals["punctuation_candidates"]
    totals["effective_punctuation_dropout_percent"] = (
        100.0 * totals["punctuation_dropped"] / candidate_count
        if candidate_count
        else 0.0
    )
    return perturbed_instruct, perturbed_context, totals


@dataclass
class OnlineEDUStats:
    wall_time_s: float = 0.0
    processing_time_s: float = 0.0
    arrival_wait_time_s: float = 0.0
    skipped_arrival_wait_time_s: float = 0.0
    input_tokens: int = 0
    emitted_edus: int = 0
    predicted_boundaries: int = 0
    forced_boundaries: int = 0
    utterances: int = 0

    def update(self, other: "OnlineEDUStats") -> None:
        self.wall_time_s += other.wall_time_s
        self.processing_time_s += other.processing_time_s
        self.arrival_wait_time_s += other.arrival_wait_time_s
        self.skipped_arrival_wait_time_s += other.skipped_arrival_wait_time_s
        self.input_tokens += other.input_tokens
        self.emitted_edus += other.emitted_edus
        self.predicted_boundaries += other.predicted_boundaries
        self.forced_boundaries += other.forced_boundaries
        self.utterances += other.utterances


@dataclass
class OnlineEDUSegment:
    text: str
    boundary_score: float | None
    forced: bool
    source: Literal["instruct", "context"]
    emitted_at: float
    final_source_segment: bool = False
    detected_boundary: bool = True
    utterance_index: int = -1
    utterance_final_segment: bool = False


@dataclass
class _FixedBoundarySpan:
    end: int


@dataclass
class _FixedBoundaryDecision:
    edu: _FixedBoundarySpan
    score: float | None
    forced: bool


def is_fixed_punctuation_boundary(
    text: str,
    token_matches,
    token_index: int,
    boundary_detector: Literal["sentence_punctuation", "fixed_punctuation"],
) -> bool:
    """判断当前标点是否是可立即提交的确定性流式边界。"""

    if token_index < 0 or token_index >= len(token_matches) - 1:
        return False
    token = token_matches[token_index].group(0)
    punctuation_marks = (
        SENTENCE_PUNCTUATION_MARKS
        if boundary_detector == "sentence_punctuation"
        else FIXED_PUNCTUATION_MARKS
    )
    if token not in punctuation_marks:
        return False

    match = token_matches[token_index]
    left_text = text[: match.start()].rstrip()
    right_text = text[match.end() :].lstrip()
    left_char = left_text[-1] if left_text else ""
    right_char = right_text[0] if right_text else ""
    if (
        token in FIXED_PUNCTUATION_NUMERIC_MARKS
        and left_char.isdigit()
        and right_char.isdigit()
    ):
        return False

    if token == ".":
        previous_token = (
            token_matches[token_index - 1].group(0) if token_index > 0 else ""
        )
        if previous_token.lower() in FIXED_PUNCTUATION_ABBREVIATIONS:
            return False
        if len(previous_token) == 1 and previous_token.isalpha():
            return False
    return True


class IncrementalOnlineEDUBoundaryStream:
    """Produce exactly one source segment per call, as soon as a boundary is known."""

    def __init__(
        self,
        *,
        segmenter: EDUSegmenter | None,
        scope: Literal["sentence", "context", "full"],
        instruct: str,
        context: str,
        end_instruct: str,
        source_token_rate: float | None = None,
        source_words_per_minute: float | None = None,
        prompt_prefill_mode: Literal["immediate"] = "immediate",
        use_eoe_boundaries: bool = False,
        fast: bool = False,
        boundary_detector: Literal[
            "model", "sentence_punctuation", "fixed_punctuation"
        ] = "model",
        punctuation_dropout_metadata: dict | None = None,
    ) -> None:
        if prompt_prefill_mode != "immediate":
            raise ValueError(
                "The fixed task prompt must be prefilled immediately. "
                "Only the sample question and context may be rate limited."
            )
        self.segmenter = segmenter
        self.scope = scope
        self.instruct = instruct
        self.context = context
        self.end_instruct = end_instruct
        self.prompt_prefill_mode = prompt_prefill_mode
        self.source_token_rate = (
            float(source_token_rate)
            if source_token_rate is not None and source_token_rate > 0
            else None
        )
        self.source_words_per_minute = (
            float(source_words_per_minute)
            if source_words_per_minute is not None
            and source_words_per_minute > 0
            else None
        )
        self.use_eoe_boundaries = bool(use_eoe_boundaries)
        self.fast = bool(fast)
        if boundary_detector not in {"model", *DETERMINISTIC_BOUNDARY_DETECTORS}:
            raise ValueError(f"Unsupported boundary detector: {boundary_detector}")
        if scope != "sentence" and boundary_detector == "model" and segmenter is None:
            raise ValueError("Model boundary detection requires an EDU segmenter.")
        self.boundary_detector = boundary_detector
        self.punctuation_dropout_metadata = punctuation_dropout_metadata or {}

        self.stats = OnlineEDUStats()
        self.events: list[dict] = []
        self.rendered_segments: list[str] = []
        self._segment_index = 0
        self._current_source: Literal["instruct", "context"] | None = None
        self._utterances = self._build_utterances()
        self._utterance_index = -1
        self._utterance_text = ""
        self._token_matches = []
        self._token_cursor = 0
        self._char_cursor = 0
        self._pending_decisions = []
        self._flush_complete = False
        self._static_segments = []
        self._peeked_segment: OnlineEDUSegment | None = None
        self._exhausted = False
        self.context_started_at: float | None = None
        self.context_start_arrival_token_index: int | None = None
        self._arrival_start_time: float | None = None
        self._fast_forwarded_wait_s = 0.0
        self._arrival_processed_tokens = 0
        self._arrival_token_due_times: list[float] = []
        self._arrival_word_index_by_token: list[int] = []
        self._source_arrival_duration_s = 0.0
        self.context_start_arrival_word_index: int | None = None
        self._context_start_due_s = 0.0
        self._build_arrival_schedule()

    def _boundary_marker(self, utterance_final_segment: bool) -> str:
        if self.use_eoe_boundaries and not utterance_final_segment:
            return "<EOE>"
        return "<EOS>"

    def _chunk_utterance(self, source: str, text: str):
        matches = list(TOKEN_PATTERN.finditer(text))
        if not matches:
            return []
        chunks = []
        max_tokens = self.segmenter.max_tokens if self.segmenter is not None else len(matches)
        for start in range(0, len(matches), max_tokens):
            chunk_matches = matches[start : start + max_tokens]
            # Preserve whitespace before the first token of each original
            # utterance, such as the space following an input <EOS> marker.
            char_start = 0 if start == 0 else chunk_matches[0].start()
            char_end = (
                matches[start + max_tokens].start()
                if start + max_tokens < len(matches)
                else len(text)
            )
            chunks.append((source, text[char_start:char_end]))
        return chunks

    def _build_utterances(self):
        utterances = []
        if self.instruct.strip():
            if self.scope == "full":
                utterances.extend(self._chunk_utterance("instruct", self.instruct))
            else:
                utterances.append(("instruct", self.instruct))
        for sentence in self.context.split("<EOS>"):
            if sentence.strip():
                utterances.extend(self._chunk_utterance("context", sentence))
        return utterances

    @staticmethod
    def _is_spoken_word(token: str) -> bool:
        return any(character.isalnum() or character == "_" for character in token)

    def _build_arrival_schedule(self) -> None:
        total_arrival_tokens = sum(
            1
            for _, text in self._utterances
            for _ in TOKEN_PATTERN.finditer(text)
        )
        if self.source_words_per_minute is None:
            if self.source_token_rate is not None:
                self._source_arrival_duration_s = (
                    total_arrival_tokens / self.source_token_rate
                )
            return
        words_per_second = self.source_words_per_minute / 60.0
        cumulative_words = 0
        context_seen = False
        for source, text in self._utterances:
            if source == "context" and not context_seen:
                context_seen = True
                self.context_start_arrival_token_index = len(
                    self._arrival_token_due_times
                )
                self.context_start_arrival_word_index = cumulative_words
                self._context_start_due_s = cumulative_words / words_per_second
            for match in TOKEN_PATTERN.finditer(text):
                if self._is_spoken_word(match.group(0)):
                    cumulative_words += 1
                self._arrival_token_due_times.append(
                    cumulative_words / words_per_second
                )
                self._arrival_word_index_by_token.append(cumulative_words)
        if self.context_start_arrival_word_index is None:
            self.context_start_arrival_word_index = cumulative_words
            self._context_start_due_s = cumulative_words / words_per_second
        self._source_arrival_duration_s = (
            self._arrival_token_due_times[-1]
            if self._arrival_token_due_times
            else 0.0
        )

    def _start_next_utterance(self) -> bool:
        self._utterance_index += 1
        if self._utterance_index >= len(self._utterances):
            self._exhausted = True
            return False
        self._current_source, self._utterance_text = self._utterances[
            self._utterance_index
        ]
        if self._current_source == "context" and self.context_started_at is None:
            if self._arrival_start_time is not None:
                self.context_started_at = (
                    self._arrival_start_time + self._context_start_due_s
                )
            else:
                self.context_started_at = time.perf_counter()
            self.context_start_arrival_token_index = self._arrival_processed_tokens
        self._token_matches = list(TOKEN_PATTERN.finditer(self._utterance_text))
        self._token_cursor = 0
        self._char_cursor = 0
        self._pending_decisions = []
        self._flush_complete = False
        if self.segmenter is not None:
            self.segmenter.reset()
        self.stats.input_tokens += len(self._token_matches)
        self.stats.utterances += 1
        return True

    def start_arrival_clock(self) -> None:
        if self._arrival_start_time is None:
            self._arrival_start_time = time.perf_counter()
            self.context_started_at = (
                self._arrival_start_time + self._context_start_due_s
            )
            if self.context_start_arrival_token_index is None:
                self.context_start_arrival_token_index = 0

    def latency_now(self) -> float:
        """返回包含已跳过输入等待的虚拟时间。"""

        return time.perf_counter() + self._fast_forwarded_wait_s

    @property
    def source_start_time(self) -> float | None:
        """返回样本问题与上下文开始到达的虚拟时间。"""

        return self._arrival_start_time

    @property
    def source_arrival_complete_time(self) -> float | None:
        """返回最后一个源输入单元按设定速率到达的虚拟时间。"""

        if self._arrival_start_time is None:
            return None
        return self._arrival_start_time + self._source_arrival_duration_s

    @property
    def source_arrival_duration_s(self) -> float:
        """返回从源输入开始到最后一个输入单元到达的计划时长。"""

        return self._source_arrival_duration_s

    def _available_arrival_tokens(self) -> int | None:
        if self.source_words_per_minute is not None:
            self.start_arrival_clock()
            elapsed = max(0.0, self.latency_now() - self._arrival_start_time)
            return bisect.bisect_right(self._arrival_token_due_times, elapsed)
        if self.source_token_rate is None:
            return None
        self.start_arrival_clock()
        elapsed = max(0.0, self.latency_now() - self._arrival_start_time)
        return int(elapsed * self.source_token_rate)

    def _seconds_until_next_arrival(self) -> float:
        if (
            self.source_words_per_minute is not None
            and self._arrival_start_time is not None
        ):
            if self._arrival_processed_tokens >= len(self._arrival_token_due_times):
                return 0.0
            next_arrival = (
                self._arrival_start_time
                + self._arrival_token_due_times[self._arrival_processed_tokens]
            )
            return max(0.0, next_arrival - self.latency_now())
        if self.source_token_rate is None or self._arrival_start_time is None:
            return 0.0
        next_arrival = (
            self._arrival_start_time
            + (self._arrival_processed_tokens + 1) / self.source_token_rate
        )
        return max(0.0, next_arrival - self.latency_now())

    def _sentence_boundary_segment(self) -> OnlineEDUSegment:
        text = self._utterance_text[self._char_cursor :] + "<EOS> "
        final_source_segment = self._utterance_index == len(self._utterances) - 1
        if final_source_segment:
            text += self.end_instruct
        self._char_cursor = len(self._utterance_text)
        self._token_matches = []
        self._flush_complete = False
        self.stats.emitted_edus += 1
        self.stats.forced_boundaries += 1
        return OnlineEDUSegment(
            text=text,
            boundary_score=None,
            forced=True,
            source=self._current_source,
            emitted_at=self.latency_now(),
            final_source_segment=final_source_segment,
            utterance_index=self._utterance_index,
            utterance_final_segment=True,
        )

    def _complete_instruct_segment(self) -> OnlineEDUSegment:
        text = self._utterance_text[self._char_cursor :] + "<EOQ>"
        final_source_segment = self._utterance_index == len(self._utterances) - 1
        if final_source_segment:
            text += self.end_instruct
        self._char_cursor = len(self._utterance_text)
        self._token_matches = []
        self._flush_complete = False
        return OnlineEDUSegment(
            text=text,
            boundary_score=None,
            forced=True,
            source="instruct",
            emitted_at=self.latency_now(),
            final_source_segment=final_source_segment,
            detected_boundary=False,
            utterance_index=self._utterance_index,
            utterance_final_segment=True,
        )

    def _decision_to_segment(self, decision) -> OnlineEDUSegment | None:
        if decision.edu is None:
            return None
        token_end = self._token_matches[decision.edu.end].end()
        is_final_in_utterance = decision.edu.end == len(self._token_matches) - 1
        text_end = len(self._utterance_text) if is_final_in_utterance else token_end
        text = self._utterance_text[self._char_cursor:text_end]
        self._char_cursor = text_end

        if self._current_source == "instruct":
            final_instruct_segment = is_final_in_utterance and not any(
                source == "instruct"
                for source, _text in self._utterances[self._utterance_index + 1 :]
            )
            if final_instruct_segment:
                text += "<EOQ>"
            else:
                text += f"{self._boundary_marker(is_final_in_utterance)} "
        else:
            text += f"{self._boundary_marker(is_final_in_utterance)} "

        final_source_segment = (
            self._utterance_index == len(self._utterances) - 1
            and is_final_in_utterance
        )
        if final_source_segment:
            text += self.end_instruct

        emitted_at = self.latency_now()
        segment = OnlineEDUSegment(
            text=text,
            boundary_score=(
                float(decision.score) if decision.score is not None else None
            ),
            forced=bool(decision.forced),
            source=self._current_source,
            emitted_at=emitted_at,
            final_source_segment=final_source_segment,
            utterance_index=self._utterance_index,
            utterance_final_segment=is_final_in_utterance,
        )
        self.stats.emitted_edus += 1
        if decision.forced:
            self.stats.forced_boundaries += 1
        else:
            self.stats.predicted_boundaries += 1
        return segment

    def _register_segment(self, segment: OnlineEDUSegment) -> OnlineEDUSegment:
        event = {
            "segment_index": self._segment_index,
            "source": segment.source,
            "boundary_score": segment.boundary_score,
            "forced": segment.forced,
            "emitted_at": segment.emitted_at,
            "llm_read_started_at": None,
            "llm_read_finished_at": None,
            "handoff_to_llm_s": None,
            "llm_read_duration_s": None,
            "text": segment.text,
            "final_source_segment": segment.final_source_segment,
            "utterance_index": segment.utterance_index,
            "utterance_final_segment": segment.utterance_final_segment,
            "detected_boundary": segment.detected_boundary,
            "target_token_index_at_read": None,
            "arrival_token_index": self._arrival_processed_tokens,
            "arrival_word_index": (
                self._arrival_word_index_by_token[
                    self._arrival_processed_tokens - 1
                ]
                if self._arrival_processed_tokens > 0
                and self._arrival_word_index_by_token
                else 0
            ),
        }
        self.events.append(event)
        self.rendered_segments.append(segment.text)
        self._segment_index += 1
        return segment

    def _advance_to_next_segment(
        self,
        token_budget: int | None = None,
    ) -> OnlineEDUSegment | None:
        processed_now = 0
        if self._static_segments:
            segment = self._static_segments.pop(0)
            segment.emitted_at = self.latency_now()
            return self._register_segment(segment)

        while not self._exhausted:
            if self._pending_decisions:
                decision = self._pending_decisions.pop(0)
                segment = self._decision_to_segment(decision)
                if segment is not None:
                    return self._register_segment(segment)
                continue

            if self._flush_complete:
                self._token_matches = []
                self._flush_complete = False
                continue

            if not self._token_matches and not self._start_next_utterance():
                return None

            if self._token_cursor < len(self._token_matches):
                if token_budget is not None and processed_now >= token_budget:
                    return None
                token = self._token_matches[self._token_cursor].group(0)
                self._token_cursor += 1
                processed_now += 1
                self._arrival_processed_tokens += 1
                if self.scope == "sentence" or (
                    self._current_source == "instruct"
                    and self.scope == "context"
                ):
                    continue
                if self.boundary_detector in DETERMINISTIC_BOUNDARY_DETECTORS:
                    token_index = self._token_cursor - 1
                    if is_fixed_punctuation_boundary(
                        self._utterance_text,
                        self._token_matches,
                        token_index,
                        self.boundary_detector,
                    ):
                        decision = _FixedBoundaryDecision(
                            edu=_FixedBoundarySpan(end=token_index),
                            score=None,
                            forced=False,
                        )
                        segment = self._decision_to_segment(decision)
                        if segment is not None:
                            return self._register_segment(segment)
                    continue

                decision = self.segmenter.push_token(token)
                if decision is not None:
                    segment = self._decision_to_segment(decision)
                    if segment is not None:
                        return self._register_segment(segment)
                continue

            if (
                self._current_source == "instruct"
                and self.scope in {"sentence", "context"}
            ):
                return self._register_segment(self._complete_instruct_segment())
            if self.scope == "sentence":
                return self._register_segment(self._sentence_boundary_segment())
            if self.boundary_detector in DETERMINISTIC_BOUNDARY_DETECTORS:
                self._flush_complete = True
                decision = _FixedBoundaryDecision(
                    edu=_FixedBoundarySpan(end=len(self._token_matches) - 1),
                    score=None,
                    forced=True,
                )
                segment = self._decision_to_segment(decision)
                if segment is not None:
                    return self._register_segment(segment)
                continue
            self._pending_decisions = self.segmenter.flush(force_final=True)
            self._flush_complete = True
        return None

    def next_segment(self, block: bool = True) -> OnlineEDUSegment | None:
        """Return the next boundary, optionally waiting for its token arrival."""

        if self._peeked_segment is not None:
            segment = self._peeked_segment
            self._peeked_segment = None
            return segment

        if self._static_segments:
            return self._advance_to_next_segment()

        start = time.perf_counter()
        try:
            while not self._exhausted:
                available_tokens = self._available_arrival_tokens()
                token_budget = (
                    None
                    if available_tokens is None
                    else max(0, available_tokens - self._arrival_processed_tokens)
                )
                processing_start = time.perf_counter()
                segment = self._advance_to_next_segment(token_budget=token_budget)
                self.stats.processing_time_s += (
                    time.perf_counter() - processing_start
                )
                if segment is not None:
                    return segment
                if self._exhausted or not block:
                    return None
                sleep_s = self._seconds_until_next_arrival()
                if sleep_s > 0:
                    if self.fast:
                        self._fast_forwarded_wait_s += sleep_s
                        self.stats.skipped_arrival_wait_time_s += sleep_s
                        continue
                    sleep_start = time.perf_counter()
                    time.sleep(min(sleep_s, 0.01))
                    self.stats.arrival_wait_time_s += (
                        time.perf_counter() - sleep_start
                    )
        finally:
            self.stats.wall_time_s += time.perf_counter() - start
        return None

    def poll_segment(self) -> OnlineEDUSegment | None:
        return self.next_segment(block=False)

    def peek_segment(self, block: bool = False) -> OnlineEDUSegment | None:
        """Return the next segment without marking it as read by the LLM.

        Peeking still advances the EDU detector and records the boundary event,
        so the event timestamp reflects when the boundary became available. A
        later ``next_segment`` call returns the same segment for actual reading.
        """

        if self._peeked_segment is None:
            self._peeked_segment = self.next_segment(block=block)
        return self._peeked_segment

    @property
    def exhausted(self) -> bool:
        return self._exhausted and not self._static_segments

    def mark_llm_read_started(self, segment_index: int) -> None:
        event = self.events[segment_index]
        now = self.latency_now()
        event["llm_read_started_at"] = now
        event["handoff_to_llm_s"] = now - event["emitted_at"]

    def mark_llm_read_finished(self, segment_index: int) -> None:
        event = self.events[segment_index]
        now = self.latency_now()
        event["llm_read_finished_at"] = now
        if event["llm_read_started_at"] is not None:
            event["llm_read_duration_s"] = now - event["llm_read_started_at"]

    def metadata(self) -> dict:
        handoff = [
            event["handoff_to_llm_s"]
            for event in self.events
            if event["detected_boundary"]
            and event["handoff_to_llm_s"] is not None
        ]
        read_durations = [
            event["llm_read_duration_s"]
            for event in self.events
            if event["detected_boundary"]
            and event["llm_read_duration_s"] is not None
        ]
        metadata = {
            "scope": self.scope,
            "boundary_detector": (
                "sentence" if self.scope == "sentence" else self.boundary_detector
            ),
            "deterministic_punctuation_marks": (
                sorted(
                    SENTENCE_PUNCTUATION_MARKS
                    if self.boundary_detector == "sentence_punctuation"
                    else FIXED_PUNCTUATION_MARKS
                )
                if self.boundary_detector in DETERMINISTIC_BOUNDARY_DETECTORS
                else None
            ),
            "device": str(self.segmenter.device) if self.segmenter is not None else None,
            "threshold": self.segmenter.threshold if self.segmenter is not None else None,
            "lookahead_tokens": self.segmenter.lookahead if self.segmenter is not None else 0,
            "maximum_tokens_per_utterance": (
                self.segmenter.max_tokens if self.segmenter is not None else None
            ),
            "source_token_rate": self.source_token_rate,
            "source_words_per_minute": self.source_words_per_minute,
            "source_start_time": self.source_start_time,
            "source_arrival_complete_time": self.source_arrival_complete_time,
            "source_arrival_duration_s": self.source_arrival_duration_s,
            "prompt_prefill_mode": self.prompt_prefill_mode,
            "use_eoe_boundaries": self.use_eoe_boundaries,
            "fast": self.fast,
            "source_rate_applies_to": ["instruct", "context"],
            "wall_time_s": self.stats.wall_time_s,
            "processing_time_s": self.stats.processing_time_s,
            "arrival_wait_time_s": self.stats.arrival_wait_time_s,
            "skipped_arrival_wait_time_s": self.stats.skipped_arrival_wait_time_s,
            "simulated_arrival_wait_time_s": (
                self.stats.arrival_wait_time_s
                + self.stats.skipped_arrival_wait_time_s
            ),
            "input_tokens": self.stats.input_tokens,
            "emitted_edus": self.stats.emitted_edus,
            "predicted_boundaries": self.stats.predicted_boundaries,
            "forced_boundaries": self.stats.forced_boundaries,
            "utterances": self.stats.utterances,
            "avg_boundary_to_llm_read_s": (
                sum(handoff) / len(handoff) if handoff else None
            ),
            "avg_llm_read_duration_s": (
                sum(read_durations) / len(read_durations) if read_durations else None
            ),
            "events": self.events,
            "rendered_source_text": "".join(self.rendered_segments),
        }
        metadata.update(self.punctuation_dropout_metadata)
        return metadata


class OnlineEDUTransformer:
    """Apply final_edu incrementally and insert StreamingThinker boundary markers."""

    def __init__(
        self,
        *,
        scope: Literal["sentence", "context", "full"] = "context",
        device: str = "cpu",
        checkpoint: str | Path | None = None,
        threshold: float | None = None,
        delimiter_guard: bool = False,
        syntax_guard: bool = False,
        use_eoe_boundaries: bool = False,
        boundary_detector: Literal[
            "model", "sentence_punctuation", "fixed_punctuation"
        ] = "model",
    ) -> None:
        if scope not in {"sentence", "context", "full"}:
            raise ValueError(f"Unsupported online EDU scope: {scope}")
        self.scope = scope
        self.use_eoe_boundaries = bool(use_eoe_boundaries)
        if boundary_detector not in {"model", *DETERMINISTIC_BOUNDARY_DETECTORS}:
            raise ValueError(f"Unsupported boundary detector: {boundary_detector}")
        self.boundary_detector = boundary_detector
        self.segmenter = (
            None
            if scope == "sentence"
            or boundary_detector in DETERMINISTIC_BOUNDARY_DETECTORS
            else EDUSegmenter(
                checkpoint=checkpoint,
                device=device,
                threshold=threshold,
                delimiter_guard=delimiter_guard,
                syntax_guard=syntax_guard,
            )
        )

    def _offline_boundary_marker(self, token_index: int, token_matches) -> str:
        if self.use_eoe_boundaries and token_index < len(token_matches) - 1:
            return "<EOE>"
        return "<EOS>"

    def _insert_boundaries(self, text: str, token_matches, boundary_indices: list[int]) -> str:
        if not token_matches:
            return text

        pieces = []
        cursor = 0
        for token_index in boundary_indices:
            end = token_matches[token_index].end()
            pieces.append(text[cursor:end])
            pieces.append(f"{self._offline_boundary_marker(token_index, token_matches)} ")
            cursor = end
        pieces.append(text[cursor:])
        return normalize_eos_spacing("".join(pieces))

    def _segment_token_chunk(
        self,
        text: str,
        token_matches,
        token_offset: int,
    ) -> tuple[list[int], OnlineEDUStats]:
        tokens = [match.group(0) for match in token_matches]
        self.segmenter.reset()
        boundaries = []
        stats = OnlineEDUStats(input_tokens=len(tokens), utterances=1)

        start = time.perf_counter()
        for token in tokens:
            decision = self.segmenter.push_token(token)
            if decision is not None and decision.edu is not None:
                boundaries.append(token_offset + decision.token_index)
                stats.emitted_edus += 1
                if decision.forced:
                    stats.forced_boundaries += 1
                else:
                    stats.predicted_boundaries += 1

        for decision in self.segmenter.flush(force_final=True):
            if decision.edu is not None:
                boundaries.append(token_offset + decision.token_index)
                stats.emitted_edus += 1
                if decision.forced:
                    stats.forced_boundaries += 1
                else:
                    stats.predicted_boundaries += 1
        stats.wall_time_s = time.perf_counter() - start
        return boundaries, stats

    def segment_utterance(self, text: str) -> tuple[str, OnlineEDUStats]:
        token_matches = list(TOKEN_PATTERN.finditer(text))
        if not token_matches:
            return text, OnlineEDUStats()

        if self.boundary_detector in DETERMINISTIC_BOUNDARY_DETECTORS:
            boundaries = [
                token_index
                for token_index in range(len(token_matches) - 1)
                if is_fixed_punctuation_boundary(
                    text,
                    token_matches,
                    token_index,
                    self.boundary_detector,
                )
            ]
            boundaries.append(len(token_matches) - 1)
            stats = OnlineEDUStats(
                input_tokens=len(token_matches),
                emitted_edus=len(boundaries),
                predicted_boundaries=max(0, len(boundaries) - 1),
                forced_boundaries=1,
                utterances=1,
            )
            return self._insert_boundaries(text, token_matches, boundaries), stats

        start = time.perf_counter()
        spans = self.segmenter.segment_text(text)
        stats = OnlineEDUStats(
            wall_time_s=time.perf_counter() - start,
            input_tokens=len(token_matches),
            emitted_edus=len(spans),
            predicted_boundaries=sum(not span.forced for span in spans),
            forced_boundaries=sum(span.forced for span in spans),
            utterances=1,
        )
        boundaries = [span.end for span in spans]

        return self._insert_boundaries(text, token_matches, boundaries), stats

    def segment_sentence_tagged_text(self, text: str) -> tuple[str, OnlineEDUStats]:
        """Use existing sentence <EOS> markers only as utterance flush points."""

        utterances = text.split("<EOS>")
        transformed = []
        stats = OnlineEDUStats()
        for utterance in utterances:
            if not utterance.strip():
                continue
            segmented, utterance_stats = self.segment_utterance(utterance)
            transformed.append(segmented)
            stats.update(utterance_stats)
        return "".join(transformed), stats

    def transform_sample(self, sample: dict) -> tuple[dict, dict]:
        transformed = copy.deepcopy(sample)
        instruct_context = transformed["Instruct_Context"]

        total_stats = OnlineEDUStats()
        context, context_stats = self.segment_sentence_tagged_text(
            instruct_context.get("context", "")
        )
        instruct_context["context"] = context
        total_stats.update(context_stats)

        instruct_stats = OnlineEDUStats()
        if self.scope == "full" and instruct_context.get("instruct", "").strip():
            instruct, instruct_stats = self.segment_utterance(
                instruct_context["instruct"]
            )
            instruct_context["instruct"] = instruct
            total_stats.update(instruct_stats)

        metadata = {
            "scope": self.scope,
            "boundary_detector": (
                "sentence" if self.scope == "sentence" else self.boundary_detector
            ),
            "device": str(self.segmenter.device) if self.segmenter is not None else None,
            "threshold": self.segmenter.threshold if self.segmenter is not None else None,
            "lookahead_tokens": (
                self.segmenter.lookahead if self.segmenter is not None else 0
            ),
            "maximum_tokens_per_utterance": (
                self.segmenter.max_tokens if self.segmenter is not None else None
            ),
            "use_eoe_boundaries": self.use_eoe_boundaries,
            "wall_time_s": total_stats.wall_time_s,
            "input_tokens": total_stats.input_tokens,
            "emitted_edus": total_stats.emitted_edus,
            "predicted_boundaries": total_stats.predicted_boundaries,
            "forced_boundaries": total_stats.forced_boundaries,
            "utterances": total_stats.utterances,
            "context_input_tokens": context_stats.input_tokens,
            "context_emitted_edus": context_stats.emitted_edus,
            "instruct_input_tokens": instruct_stats.input_tokens,
            "instruct_emitted_edus": instruct_stats.emitted_edus,
        }
        return transformed, metadata


class OnlineEDUStreamingDataCollator(StreamingDataCollator):
    def __init__(self, *args, online_edu_transformer: OnlineEDUTransformer, **kwargs):
        super().__init__(*args, **kwargs)
        self.online_edu_transformer = online_edu_transformer

    def generator(self, samples):
        for sample in samples:
            transformed, metadata = self.online_edu_transformer.transform_sample(sample)
            inputs, labels, source_txt_lt, target_txt_lt = self.extract_instruct(transformed)
            yield {
                "input_txt": inputs,
                "labels": labels,
                "Instruct": self.Instruct,
                "source_txt_lt": source_txt_lt,
                "target_txt_lt": target_txt_lt,
                "online_edu_metadata": metadata,
            }

    def dataset_loader(self, file_paths=None):
        if file_paths is None:
            file_paths = self.file_path
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        collator = self

        class OnlineEDUIterableDataset(IterableDataset):
            def __iter__(self):
                for file_path in file_paths:
                    yield from collator.generator(collator._load_samples(file_path))

        return OnlineEDUIterableDataset()

    def collate_fn_inference(self, batch_data):
        batch = super().collate_fn_inference(batch_data)
        batch["online_edu_metadata"] = [
            item["online_edu_metadata"] for item in batch_data
        ]
        return batch


class IncrementalOnlineEDUStreamingDataCollator(StreamingDataCollator):
    """Build only the prompt prefix; EDU source segments are produced during generation."""

    def __init__(
        self,
        *args,
        online_edu_transformer: OnlineEDUTransformer,
        source_token_rate: float | None = None,
        source_words_per_minute: float | None = None,
        prompt_prefill_mode: Literal["immediate"] = "immediate",
        use_eoe_boundaries: bool | None = None,
        fast: bool = False,
        punctuation_dropout_percent: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if prompt_prefill_mode != "immediate":
            raise ValueError(
                "Incremental online EDU evaluation supports only immediate "
                "task-prompt prefill."
            )
        self.online_edu_transformer = online_edu_transformer
        self.source_token_rate = source_token_rate
        self.source_words_per_minute = source_words_per_minute
        self.prompt_prefill_mode = prompt_prefill_mode
        self.fast = bool(fast)
        if not 0.0 <= punctuation_dropout_percent <= 100.0:
            raise ValueError(
                "punctuation_dropout_percent must be between 0 and 100"
            )
        self.punctuation_dropout_percent = float(punctuation_dropout_percent)
        self.use_eoe_boundaries = (
            self.online_edu_transformer.use_eoe_boundaries
            if use_eoe_boundaries is None
            else bool(use_eoe_boundaries)
        )
        self._sample_count_cache: dict[tuple[str, ...], int] = {}

    def prepare_streaming_input(self, sample: dict) -> tuple[str, str, dict]:
        """生成当前样本可复现的 ASR 标点缺失输入。"""

        instruct_context = sample["Instruct_Context"]
        return apply_asr_punctuation_dropout(
            instruct=instruct_context.get("instruct", ""),
            context=instruct_context.get("context", ""),
            percent=self.punctuation_dropout_percent,
        )

    def generator(self, samples):
        for sample in samples:
            prompt = self.Instruct[0]
            instruct, context, dropout_metadata = self.prepare_streaming_input(
                sample
            )
            stream = IncrementalOnlineEDUBoundaryStream(
                segmenter=self.online_edu_transformer.segmenter,
                scope=self.online_edu_transformer.scope,
                instruct=instruct,
                context=context,
                end_instruct=self.end_Instruct,
                source_token_rate=self.source_token_rate,
                source_words_per_minute=self.source_words_per_minute,
                prompt_prefill_mode=self.prompt_prefill_mode,
                use_eoe_boundaries=self.use_eoe_boundaries,
                fast=self.fast,
                boundary_detector=self.online_edu_transformer.boundary_detector,
                punctuation_dropout_metadata=dropout_metadata,
            )
            yield {
                "prefix_text": prompt + "<SEP>",
                "labels": sample.get("answer", ""),
                "online_edu_boundary_stream": stream,
            }

    def dataset_loader(self, file_paths=None):
        if file_paths is None:
            file_paths = self.file_path
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        collator = self
        cache_key = tuple(str(file_path) for file_path in file_paths)
        if cache_key not in self._sample_count_cache:
            self._sample_count_cache[cache_key] = sum(
                1
                for file_path in file_paths
                for _sample in self._load_samples(file_path)
            )
        total_samples = self._sample_count_cache[cache_key]

        class IncrementalOnlineEDUDataset(IterableDataset):
            def __iter__(self):
                for file_path in file_paths:
                    yield from collator.generator(collator._load_samples(file_path))

            def __len__(self):
                return total_samples

        return IncrementalOnlineEDUDataset()

    def collate_fn_inference(self, batch_data):
        if len(batch_data) != 1:
            raise ValueError("Incremental online EDU evaluation requires batch size 1.")
        item = batch_data[0]
        source_tokens = self.tokenizer(
            [item["prefix_text"]],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        target_tokens = self.tokenizer(
            [item["labels"]],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        prefix_len = int(source_tokens["input_ids"].shape[1])
        return {
            "source_tokens": source_tokens["input_ids"],
            "attention_mask": source_tokens["attention_mask"],
            "labels": target_tokens["input_ids"],
            "_lengths": [
                {
                    "source_token_len": prefix_len,
                    "source_seg_len": [prefix_len],
                    "prefix_source_segments": 1,
                    "prompt_prefill_mode": self.prompt_prefill_mode,
                    "online_source": True,
                }
            ],
            "position_ids": torch.arange(prefix_len).unsqueeze(0),
            "inference_mode": self.inference_mode,
            "split_mode": self.split_mode,
            "assistant_token": torch.tensor(self.assistant_Instruct_token),
            "source_txt": [item["prefix_text"]],
            "online_edu_boundary_stream": [item["online_edu_boundary_stream"]],
        }
