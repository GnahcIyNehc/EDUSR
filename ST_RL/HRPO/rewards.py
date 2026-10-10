"""Text-level reward helpers for AdaSR/HRPO.

These helpers keep parsing separate from tensor loss computation. In a full RL
pipeline, call them on raw rollouts before repacking the batch into token
tensors for log-probability evaluation.
"""

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence


_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")
_MAX_NUMERIC_DIGITS = 10_000
_MAX_DECIMAL_EXPONENT = 10_000


def normalize_answer(text: object) -> str:
    """Normalize a textual answer for lightweight exact matching."""
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(".。 \t\r\n")
    return normalized


def _parse_simple_number(text: object) -> Optional[Fraction]:
    """解析常见整数、小数、分数和 LaTeX 分数。"""
    normalized = normalize_answer(text).replace(" ", "")
    latex_fraction = re.fullmatch(r"\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}", normalized)
    if latex_fraction:
        numerator = _parse_simple_number(latex_fraction.group(1))
        denominator = _parse_simple_number(latex_fraction.group(2))
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator
    if normalized.endswith("%"):
        percentage = _parse_simple_number(normalized[:-1])
        return None if percentage is None else percentage / 100

    def parse_decimal(value: str) -> Optional[Fraction]:
        decimal_value = Decimal(value)
        if not decimal_value.is_finite():
            return None
        decimal_tuple = decimal_value.as_tuple()
        exponent = decimal_tuple.exponent
        if not isinstance(exponent, int):
            return None
        if len(decimal_tuple.digits) > _MAX_NUMERIC_DIGITS or abs(exponent) > _MAX_DECIMAL_EXPONENT:
            return None
        return Fraction(decimal_value)

    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            parsed_numerator = parse_decimal(numerator)
            parsed_denominator = parse_decimal(denominator)
            if parsed_numerator is None or parsed_denominator in (None, 0):
                return None
            return parsed_numerator / parsed_denominator
        return parse_decimal(normalized)
    except (InvalidOperation, OverflowError, ValueError, ZeroDivisionError):
        return None


def _candidate_sets_match(pred_candidates: Sequence[str], gold_candidates: Sequence[str]) -> bool:
    """先比较归一化文本，再比较可安全解析的数值。"""
    norm_pred = {normalize_answer(item) for item in pred_candidates if normalize_answer(item)}
    norm_gold = {normalize_answer(item) for item in gold_candidates if normalize_answer(item)}
    if norm_pred & norm_gold:
        return True
    pred_numbers = {_parse_simple_number(item) for item in pred_candidates}
    gold_numbers = {_parse_simple_number(item) for item in gold_candidates}
    pred_numbers.discard(None)
    gold_numbers.discard(None)
    return bool(pred_numbers & gold_numbers)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last ``\\boxed{...}`` answer if present."""
    source = text or ""
    matches: List[str] = []
    for start_match in _BOXED_START_RE.finditer(source):
        opening_brace = start_match.end() - 1
        depth = 0
        for index in range(opening_brace, len(source)):
            character = source[index]
            preceding_slashes = 0
            slash_index = index - 1
            while slash_index >= 0 and source[slash_index] == "\\":
                preceding_slashes += 1
                slash_index -= 1
            is_escaped = preceding_slashes % 2 == 1
            if character == "{" and not is_escaped:
                depth += 1
            elif character == "}" and not is_escaped:
                depth -= 1
                if depth == 0:
                    matches.append(source[opening_brace + 1 : index].strip())
                    break
    return matches[-1] if matches else None


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last number-like span from text."""
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1]


def exact_match_accuracy_reward(
    prediction: str,
    answer: str,
    prefer_boxed: bool = True,
    require_boxed: bool = False,
) -> float:
    """计算二值最终答案奖励。

    严格模式要求模型输出 ``\\boxed{...}``，但仍允许从带解释的标准答案中
    提取最后一个数字。宽松模式保留原有的全文和末尾数字兜底行为。
    """
    pred = prediction or ""
    gold = answer or ""
    pred_candidates: List[str] = []
    gold_candidates: List[str] = []

    boxed_pred = extract_boxed_answer(pred)
    boxed_gold = extract_boxed_answer(gold)
    gold_num = extract_last_number(gold)

    if require_boxed:
        if boxed_pred is None:
            return 0.0
        pred_candidates.append(boxed_pred)
        if boxed_gold is not None:
            gold_candidates.append(boxed_gold)
        gold_candidates.append(gold)
        if gold_num is not None:
            gold_candidates.append(gold_num)
        return float(_candidate_sets_match(pred_candidates, gold_candidates))

    if prefer_boxed:
        if boxed_pred is not None:
            pred_candidates.append(boxed_pred)
        if boxed_gold is not None:
            gold_candidates.append(boxed_gold)

    pred_candidates.append(pred)
    gold_candidates.append(gold)

    pred_num = extract_last_number(pred)
    if pred_num is not None:
        pred_candidates.append(pred_num)
    if gold_num is not None:
        gold_candidates.append(gold_num)

    return float(_candidate_sets_match(pred_candidates, gold_candidates))


def _is_valid_streaming_content(content: str, skip_token: str) -> bool:
    content = content.strip()
    return bool(content) or content == skip_token


def streaming_format_reward(
    streaming_segments: Sequence[str],
    eot_token: str = "<EOT>",
    skip_token: str = "<skip>",
) -> float:
    """Binary streaming-format reward.

    Each streaming segment must terminate with ``eot_token``. The segment
    content before the terminator can be a natural thought or ``skip_token``.
    """
    if not streaming_segments:
        return 0.0
    for segment in streaming_segments:
        if segment is None:
            return 0.0
        text = str(segment).strip()
        if not text.endswith(eot_token):
            return 0.0
        content = text[: -len(eot_token)].strip()
        if not _is_valid_streaming_content(content, skip_token):
            return 0.0
    return 1.0


def split_streaming_segments(text: str, eot_token: str = "<EOT>") -> List[str]:
    """Split a raw streaming text into segments that keep their EOT marker."""
    if not text:
        return []
    parts = text.split(eot_token)
    segments = [part + eot_token for part in parts[:-1]]
    tail = parts[-1].strip()
    if tail:
        segments.append(tail)
    return segments


def deep_format_reward(deep_text: str, eor_token: str = "<EOR>") -> float:
    """Binary deep-reasoning format reward."""
    if deep_text is None:
        return 0.0
    text = str(deep_text).strip()
    if not text.endswith(eor_token):
        return 0.0
    content = text[: -len(eor_token)].strip()
    return float(bool(content))


def token_lengths_from_texts(texts: Iterable[str], tokenizer=None) -> List[int]:
    """Return token lengths with an optional HuggingFace tokenizer."""
    lengths = []
    for text in texts:
        if tokenizer is None:
            lengths.append(len(str(text).split()))
        else:
            lengths.append(len(tokenizer(str(text), add_special_tokens=False)["input_ids"]))
    return lengths
