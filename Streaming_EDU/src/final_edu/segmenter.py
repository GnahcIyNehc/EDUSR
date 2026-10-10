from __future__ import annotations

import os
import pathlib
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from .features import (
    balanced_delimiter_span_mask,
    biomedical_boundary_needs_more_context,
    biomedical_parenthetical_span_mask,
    code_span_mask,
    CURRENCY_SYMBOLS,
    encode_tokens,
    esurf_boundary_needs_more_context,
    esurf_ngram_closure_mask,
    is_decimal_point,
    is_likely_sentence_period,
    is_nonterminal_period,
    is_numeric_internal_colon,
    math_boundary_needs_more_context,
    math_span_mask,
    measurement_span_mask,
    mwe_boundary_closure_mask,
    mwe_boundary_needs_more_context,
    predicate_ngram_closure_mask,
    syntactic_closure_mask,
)
from .model import StreamingTransformerEDU, lookahead_mask


TOKEN_PATTERN = re.compile(r"https?://\S+|www\.\S+|[\w]+(?:['-][\w]+)*|[^\w\s]", re.UNICODE)
NO_SPACE_BEFORE = {".", ",", ";", ":", "?", "!", "%", ")", "]", "}"}
NO_SPACE_AFTER = {"(", "[", "{"} | CURRENCY_SYMBOLS
NO_SPACE_AROUND = {"/", "-"}
ATTACHED_RIGHT_PUNCTUATION = NO_SPACE_BEFORE
FALLBACK_CHUNK_BOUNDARY_TOKENS = {",", ";", ")", "]", "}"}


def _numeric_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?", token))


def _no_space_before(tokens: tuple[str, ...], index: int) -> bool:
    token = tokens[index]
    previous = tokens[index - 1] if index > 0 else ""
    before_previous = tokens[index - 2] if index > 1 else ""
    if token in NO_SPACE_BEFORE or token in NO_SPACE_AROUND:
        return True
    if previous in NO_SPACE_AFTER or previous in NO_SPACE_AROUND:
        return True
    if previous in {".", ":"} and _numeric_token(before_previous) and token.isdigit():
        return True
    return False


def detokenize_tokens(tokens: Iterable[str]) -> str:
    """Render tokens without inventing spaces before punctuation."""

    token_tuple = tuple(tokens)
    if not token_tuple:
        return ""
    pieces = [token_tuple[0]]
    for index in range(1, len(token_tuple)):
        token = token_tuple[index]
        if _no_space_before(token_tuple, index):
            pieces.append(token)
        else:
            pieces.extend((" ", token))
    return "".join(pieces)


def _load_checkpoint_portably(path: Path) -> dict:
    """Load checkpoints created on Windows from POSIX hosts as well."""

    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath
    return torch.load(path, map_location="cpu", weights_only=False)


def simple_tokenize(text: str) -> list[str]:
    """Small English-oriented fallback tokenizer.

    Production integrations should normally pass the tokens produced by the
    upstream ASR or NLP tokenizer.
    """

    return TOKEN_PATTERN.findall(text)


def _is_text_chunk_sentence_end(tokens: list[str], index: int) -> bool:
    """判断长文本切块点是否落在真实句末。"""

    if index < 0 or index >= len(tokens):
        return False
    token = tokens[index]
    if token in {"!", "?"}:
        return True
    return (
        token == "."
        and not is_decimal_point(tokens, index)
        and not is_nonterminal_period(tokens, index)
        and is_likely_sentence_period(tokens, index)
    )


def _bad_chunk_end(tokens: list[str], cut: int) -> bool:
    """检查长文本模型窗口是否把附着表达切开。"""

    if cut <= 0 or cut >= len(tokens):
        return False
    previous = tokens[cut - 1]
    following = tokens[cut]
    if following in ATTACHED_RIGHT_PUNCTUATION:
        return True
    if previous in NO_SPACE_AFTER or previous in NO_SPACE_AROUND:
        return True
    if following in NO_SPACE_AROUND:
        return True
    if previous.isdigit() and following == ".":
        return True
    if previous == "." and following.isdigit():
        return True
    return False


def _is_fallback_chunk_boundary(tokens: list[str], index: int) -> bool:
    if index < 0 or index >= len(tokens):
        return False
    return (
        tokens[index] in FALLBACK_CHUNK_BOUNDARY_TOKENS
        and not _bad_chunk_end(tokens, index + 1)
    )


def _adjust_hard_chunk_end(tokens: list[str], start: int, hard_end: int) -> int:
    for index in range(hard_end - 1, start, -1):
        if _is_fallback_chunk_boundary(tokens, index):
            return index + 1
    cut = hard_end
    while cut > start + 1 and _bad_chunk_end(tokens, cut):
        cut -= 1
    return max(start + 1, cut)


def _sentence_aware_chunk_ranges(tokens: list[str], max_tokens: int) -> list[tuple[int, int]]:
    """将长文本按句末附近切块，避免超过模型位置长度。"""

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(tokens):
        hard_end = min(start + max_tokens, len(tokens))
        if hard_end == len(tokens):
            ranges.append((start, hard_end))
            break
        cut = None
        for index in range(hard_end - 1, start, -1):
            if _is_text_chunk_sentence_end(tokens, index):
                cut = index + 1
                break
        if cut is None or cut <= start:
            cut = _adjust_hard_chunk_end(tokens, start, hard_end)
        ranges.append((start, cut))
        start = cut
    return ranges


@dataclass(frozen=True)
class EDUSpan:
    tokens: tuple[str, ...]
    start: int
    end: int
    boundary_score: float
    forced: bool = False
    surface_text: str | None = None

    @property
    def text(self) -> str:
        return self.surface_text if self.surface_text is not None else detokenize_tokens(self.tokens)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tokens"] = list(self.tokens)
        data["text"] = self.text
        return data


@dataclass(frozen=True)
class BoundaryDecision:
    token_index: int
    token: str
    score: float
    is_boundary: bool
    edu: EDUSpan | None = None
    forced: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["edu"] = self.edu.to_dict() if self.edu is not None else None
        return data


class EDUSegmenter:
    """Low-latency token-stream EDU segmenter.

    The bundled model uses one-token lookahead. After token ``i + 1`` arrives,
    ``push_token`` returns the decision for token ``i``.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        threshold: float | None = None,
        decimal_guard: bool = True,
        period_guard: bool = True,
        numeric_colon_guard: bool = True,
        phrase_guard: bool = True,
        measurement_guard: bool = True,
        predicate_guard: bool = True,
        esurf_guard: bool = True,
        delimiter_guard: bool = False,
        syntax_guard: bool = False,
        code_guard: bool = True,
        math_guard: bool = True,
    ) -> None:
        if checkpoint is None:
            checkpoint = (
                Path(__file__).resolve().parent
                / "models"
                / "grid8ref_hybrid_nep_codra_k1_e30_period_safe.pt"
            )
        self.checkpoint_path = Path(checkpoint).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        self.device = self._resolve_device(device)
        checkpoint_data = _load_checkpoint_portably(self.checkpoint_path)
        self.vocab: dict[str, int] = checkpoint_data["vocab"]
        self.lookahead = int(checkpoint_data["lookahead"])
        self.threshold = (
            float(threshold)
            if threshold is not None
            else float(checkpoint_data["best_dev"]["threshold"])
        )
        self.char_buckets = int(checkpoint_data.get("char_buckets", 4096))
        self.char_slots = int(checkpoint_data.get("char_slots", 18))
        self.model_config = dict(checkpoint_data["model_config"])
        self.max_tokens = int(self.model_config.get("max_len", 256))
        self.decimal_guard = decimal_guard
        self.period_guard = period_guard
        self.numeric_colon_guard = numeric_colon_guard
        self.phrase_guard = phrase_guard
        self.measurement_guard = measurement_guard
        self.predicate_guard = predicate_guard
        self.esurf_guard = esurf_guard
        self.delimiter_guard = delimiter_guard
        self.syntax_guard = syntax_guard
        self.code_guard = code_guard
        self.math_guard = math_guard

        self.model = StreamingTransformerEDU(len(self.vocab), **self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint_data["model_state"], strict=True)
        self.model.eval()

        self._tokens: list[str] = []
        self._next_decision = 0
        self._edu_start = 0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device(device)

    @property
    def buffered_tokens(self) -> tuple[str, ...]:
        return tuple(self._tokens)

    def reset(self) -> None:
        self._tokens.clear()
        self._next_decision = 0
        self._edu_start = 0

    @torch.inference_mode()
    def score_tokens(self, tokens: list[str]) -> list[float]:
        if len(tokens) > self.max_tokens:
            raise ValueError(
                f"the bundled model supports at most {self.max_tokens} tokens per utterance; "
                "flush/reset at sentence or utterance boundaries"
            )
        encoded = encode_tokens(tokens, self.vocab, self.char_buckets, self.char_slots, self.device)
        attention_mask = lookahead_mask(len(tokens), self.lookahead, self.device)
        output = self.model(
            encoded.token_ids,
            encoded.shape_ids,
            encoded.char_ids,
            encoded.padding_mask,
            attention_mask,
            encoded.cue_ids,
            encoded.stream_ids,
        )
        logits = output[0] if isinstance(output, tuple) else output
        scores = torch.sigmoid(logits[0]).detach().cpu().tolist()
        sentence_period_mask = [False] * len(tokens)
        if self.decimal_guard:
            for i in range(len(tokens)):
                if is_decimal_point(tokens, i):
                    scores[i] = 0.0
        if self.period_guard:
            for i in range(len(tokens)):
                if is_nonterminal_period(tokens, i):
                    scores[i] = 0.0
                elif is_likely_sentence_period(tokens, i):
                    sentence_period_mask[i] = True
                    scores[i] = 1.0
        if self.numeric_colon_guard:
            for i in range(len(tokens)):
                if is_numeric_internal_colon(tokens, i):
                    scores[i] = 0.0
        if self.phrase_guard:
            phrase_mask = mwe_boundary_closure_mask(tokens)
            for i, is_phrase_internal in enumerate(phrase_mask):
                if is_phrase_internal:
                    scores[i] = 0.0
            biomedical_mask = biomedical_parenthetical_span_mask(tokens)
            for i, is_biomedical_internal in enumerate(biomedical_mask):
                if is_biomedical_internal:
                    scores[i] = 0.0
        if self.measurement_guard:
            measurement_mask = measurement_span_mask(tokens)
            for i, is_measurement_internal in enumerate(measurement_mask):
                if is_measurement_internal:
                    scores[i] = 0.0
        if self.predicate_guard:
            predicate_mask = predicate_ngram_closure_mask(tokens)
            for i, is_predicate_incomplete in enumerate(predicate_mask):
                if is_predicate_incomplete:
                    scores[i] = 0.0
        if self.esurf_guard:
            esurf_mask = esurf_ngram_closure_mask(tokens)
            for i, is_esurf_internal in enumerate(esurf_mask):
                if is_esurf_internal:
                    scores[i] = 0.0
        if self.delimiter_guard:
            delimiter_mask = balanced_delimiter_span_mask(tokens)
            for i, is_delimiter_internal in enumerate(delimiter_mask):
                if is_delimiter_internal:
                    scores[i] = 0.0
        if self.syntax_guard:
            syntax_mask = syntactic_closure_mask(tokens)
            for i, is_syntax_incomplete in enumerate(syntax_mask):
                if is_syntax_incomplete:
                    scores[i] = 0.0
        if self.code_guard:
            code_mask = code_span_mask(tokens)
            for i, is_code in enumerate(code_mask):
                if is_code:
                    scores[i] = 0.0
        if self.math_guard:
            math_mask = math_span_mask(tokens)
            for i, is_math_internal in enumerate(math_mask):
                if is_math_internal:
                    scores[i] = 0.0
        for i in range(len(tokens) - 1):
            if tokens[i + 1] in ATTACHED_RIGHT_PUNCTUATION:
                scores[i] = 0.0
        if self.period_guard:
            for i, is_sentence_period in enumerate(sentence_period_mask):
                if is_sentence_period:
                    scores[i] = 1.0
        return scores

    def _should_hold_decision(self, index: int, force_final: bool = False) -> bool:
        if self.period_guard and is_likely_sentence_period(self._tokens, index):
            return False
        if self.math_guard and not force_final and math_boundary_needs_more_context(self._tokens, index):
            return True
        if self.phrase_guard and mwe_boundary_needs_more_context(self._tokens, index):
            return True
        if (
            self.phrase_guard
            and not force_final
            and biomedical_boundary_needs_more_context(self._tokens, index)
        ):
            return True
        if self.esurf_guard and esurf_boundary_needs_more_context(self._tokens, index):
            return True
        if not self.phrase_guard:
            return False
        if force_final and index == len(self._tokens) - 1:
            return mwe_boundary_closure_mask(self._tokens)[index]
        return False

    def _make_decision(self, index: int, scores: list[float], force_boundary: bool = False) -> BoundaryDecision:
        score = float(scores[index])
        sentence_period_boundary = (
            self.period_guard and is_likely_sentence_period(self._tokens, index)
        )
        if self.predicate_guard and not force_boundary and not sentence_period_boundary and index + 1 < len(self._tokens):
            span_tokens = self._tokens[self._edu_start : index + 2]
            span_index = index - self._edu_start
            if predicate_ngram_closure_mask(span_tokens)[span_index]:
                score = 0.0
        if self.esurf_guard and not force_boundary and not sentence_period_boundary and index + 1 < len(self._tokens):
            span_tokens = self._tokens[self._edu_start : index + 2]
            span_index = index - self._edu_start
            if esurf_ngram_closure_mask(span_tokens)[span_index]:
                score = 0.0
        if sentence_period_boundary:
            score = 1.0
        is_boundary = force_boundary or score >= self.threshold
        edu = None
        if is_boundary:
            edu = EDUSpan(
                tokens=tuple(self._tokens[self._edu_start : index + 1]),
                start=self._edu_start,
                end=index,
                boundary_score=score,
                forced=force_boundary,
            )
            self._edu_start = index + 1
        self._next_decision = index + 1
        return BoundaryDecision(
            token_index=index,
            token=self._tokens[index],
            score=score,
            is_boundary=is_boundary,
            edu=edu,
            forced=force_boundary,
        )

    def push_token(self, token: str) -> BoundaryDecision | None:
        """Push one token and return the newly available boundary decision."""

        if not token:
            raise ValueError("token must not be empty")
        if len(self._tokens) >= self.max_tokens:
            raise ValueError(
                f"stream reached {self.max_tokens} tokens; call flush() and start a new utterance"
            )
        self._tokens.append(token)
        target = len(self._tokens) - 1 - self.lookahead
        if target < self._next_decision:
            return None
        index = self._next_decision
        if self._should_hold_decision(index):
            return None
        return self._make_decision(index, self.score_tokens(self._tokens))

    def flush(self, force_final: bool = True, reset: bool = True) -> list[BoundaryDecision]:
        """Resolve tokens still waiting for lookahead.

        ``force_final=True`` closes the final EDU at the application-provided
        utterance boundary. The returned final decision is marked ``forced``.
        """

        if not self._tokens:
            return []
        scores = self.score_tokens(self._tokens)
        decisions: list[BoundaryDecision] = []
        last_index = len(self._tokens) - 1
        while self._next_decision <= last_index:
            index = self._next_decision
            if self._should_hold_decision(index, force_final=force_final):
                break
            decisions.append(
                self._make_decision(
                    index,
                    scores,
                    force_boundary=force_final and index == last_index,
                )
            )
        if reset and self._next_decision > last_index:
            self.reset()
        return decisions

    def segment_tokens(self, tokens: Iterable[str], force_final: bool = True) -> list[EDUSpan]:
        """Simulate streaming input and return emitted EDU spans."""

        self.reset()
        spans: list[EDUSpan] = []
        for token in tokens:
            decision = self.push_token(token)
            if decision is not None and decision.edu is not None:
                spans.append(decision.edu)
        for decision in self.flush(force_final=force_final):
            if decision.edu is not None:
                spans.append(decision.edu)
        return spans

    def segment_text(self, text: str, force_final: bool = True) -> list[EDUSpan]:
        token_matches = list(TOKEN_PATTERN.finditer(text))
        if not token_matches:
            return []
        tokens = [match.group(0) for match in token_matches]
        rendered_spans: list[EDUSpan] = []
        for chunk_start, chunk_end in _sentence_aware_chunk_ranges(tokens, self.max_tokens):
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_force_final = force_final or chunk_end < len(tokens)
            for span in self.segment_tokens(chunk_tokens, force_final=chunk_force_final):
                global_start = chunk_start + span.start
                global_end = chunk_start + span.end
                rendered_spans.append(
                    EDUSpan(
                        tokens=span.tokens,
                        start=global_start,
                        end=global_end,
                        boundary_score=span.boundary_score,
                        forced=span.forced,
                        surface_text=text[
                            token_matches[global_start].start() : token_matches[global_end].end()
                        ],
                    )
                )
        return rendered_spans

    def benchmark(self, tokens: list[str], warmup: int = 5, repeats: int = 50) -> dict[str, float | str | int]:
        if not tokens:
            raise ValueError("tokens must not be empty")
        for _ in range(warmup):
            self.score_tokens(tokens)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        for _ in range(repeats):
            self.score_tokens(tokens)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        return {
            "device": str(self.device),
            "tokens": len(tokens),
            "repeats": repeats,
            "mean_inference_ms": 1000.0 * elapsed / repeats,
            "lookahead_tokens": self.lookahead,
        }
