from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_ROOT = Path("data/gum_edu")
PAD = "<pad>"
UNK = "<unk>"
DECIMAL_POINT = "<decimal_point>"
NONTERMINAL_PERIOD = "<nonterminal_period>"


@dataclass(frozen=True)
class Document:
    tokens: list[str]
    labels: list[int]
    starts: list[int] | None = None
    hard_negative: list[int] | None = None


def split_path(split: str, data_root: Path = DATA_ROOT) -> Path:
    return data_root / split / f"gum_{split}.online.jsonl"


def read_split(split: str, data_root: Path = DATA_ROOT) -> list[Document]:
    docs: list[Document] = []
    with split_path(split, data_root).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(tokens=row["tokens"], labels=row["end_labels"], starts=row.get("start_labels")))
    return docs


def normalize_token(token: str) -> str:
    token = token.lower()
    if re.fullmatch(r"\d+([.,:/-]\d+)*", token):
        return "<num>"
    if re.fullmatch(r"https?://\S+|www\.\S+", token):
        return "<url>"
    if "@" in token and "." in token:
        return "<email>"
    return token


def build_vocab(docs: list[Document], min_freq: int = 2, max_size: int = 50000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(normalize_token(tok) for tok in doc.tokens)
    vocab = {PAD: 0, UNK: 1}
    for tok, freq in counts.most_common(max_size - len(vocab)):
        if freq < min_freq:
            break
        vocab[tok] = len(vocab)
    return vocab


PUNCT_SET = set(".,;:!?)]}\"'")
OPEN_SET = set("([{\"'")
CLAUSE_PUNCT_SET = set(",;:!?")
COORD_CUES = {"and", "but", "or", "nor", "yet", "so"}
SUBORD_CUES = {
    "after",
    "although",
    "as",
    "because",
    "before",
    "even",
    "if",
    "once",
    "since",
    "though",
    "unless",
    "until",
    "when",
    "whenever",
    "whereas",
    "whether",
    "while",
}
ADVERBIAL_CUES = {
    "accordingly",
    "additionally",
    "also",
    "alternatively",
    "consequently",
    "furthermore",
    "hence",
    "however",
    "instead",
    "meanwhile",
    "moreover",
    "nevertheless",
    "nonetheless",
    "otherwise",
    "therefore",
    "thus",
}
RELATIVE_CUES = {"that", "which", "who", "whom", "whose", "where"}
REPORTING_CUES = {"said", "says", "say", "reported", "told", "asked", "according"}


def is_numeric_piece(token: str) -> bool:
    return bool(re.fullmatch(r"\d+", token))


def is_decimal_point(tokens: list[str], index: int) -> bool:
    return (
        tokens[index] == "."
        and index > 0
        and index + 1 < len(tokens)
        and is_numeric_piece(tokens[index - 1])
        and is_numeric_piece(tokens[index + 1])
    )


TITLE_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st"}
NUMBER_ABBREVIATIONS = {"fig", "no", "eq", "ch", "sec", "pp", "vol"}
MONTH_ABBREVIATIONS = {"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
GENERAL_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "approx", "dept", "inc", "ltd", "co"}
DOMAIN_PREFIXES = {"www", "docs", "api", "mail", "shop", "blog", "support", "portal"}
DOMAIN_SUFFIXES = {"com", "org", "net", "edu", "gov", "io", "ai", "co", "uk", "de", "fr", "cn", "jp"}
FILE_EXTENSIONS = {"pdf", "txt", "csv", "json", "xml", "html", "htm", "py", "js", "ts", "md", "doc", "docx", "xls", "xlsx", "png", "jpg", "jpeg"}


def is_nonterminal_period(tokens: list[str], index: int) -> bool:
    if tokens[index] != "." or index == 0 or index + 1 >= len(tokens):
        return False
    if is_decimal_point(tokens, index):
        return False
    previous = tokens[index - 1]
    following = tokens[index + 1]
    previous_lower = previous.lower()
    following_lower = following.lower()

    if following == ".":
        return True
    if previous == ".":
        continuation_cues = COORD_CUES | SUBORD_CUES | ADVERBIAL_CUES
        return following_lower not in continuation_cues
    if previous_lower in TITLE_ABBREVIATIONS and following[:1].isupper():
        if index >= 2 and tokens[index - 2].lower() in {"a", "an", "the"}:
            return False
        return True
    if previous_lower in NUMBER_ABBREVIATIONS and is_numeric_piece(following):
        return True
    if previous_lower in MONTH_ABBREVIATIONS and is_numeric_piece(following):
        return True
    if previous_lower in GENERAL_ABBREVIATIONS and (not following[:1].isupper() or following in ",;:)"):
        return True
    if len(previous) == 1 and previous.isalpha():
        if len(following) == 1 and following.isalpha():
            return True
        if index >= 2 and tokens[index - 2] == ".":
            return True
    if following == following_lower and following_lower in DOMAIN_SUFFIXES | FILE_EXTENSIONS:
        return True
    if (
        previous_lower in DOMAIN_PREFIXES
        and following[:1].islower()
        and re.fullmatch(r"[A-Za-z0-9-]+", following)
    ):
        return True
    return False


def is_likely_sentence_period(tokens: list[str], index: int) -> bool:
    if (
        tokens[index] != "."
        or index + 1 >= len(tokens)
        or is_decimal_point(tokens, index)
        or is_nonterminal_period(tokens, index)
    ):
        return False
    following = tokens[index + 1]
    return len(following) > 1 and following.isupper()


def shape_id(token: str, decimal_point: bool = False, nonterminal_period: bool = False) -> int:
    if decimal_point:
        return 9
    if nonterminal_period:
        return 10
    if token in PUNCT_SET:
        return 1
    if token in OPEN_SET:
        return 2
    if token.isdigit():
        return 3
    if token.isupper() and len(token) > 1:
        return 4
    if token[:1].isupper():
        return 5
    if any(ch.isdigit() for ch in token):
        return 6
    if "-" in token:
        return 7
    return 8


def contextual_shape_ids(tokens: list[str]) -> list[int]:
    return [
        shape_id(token, is_decimal_point(tokens, i), is_nonterminal_period(tokens, i))
        for i, token in enumerate(tokens)
    ]


def contextual_token_id(tokens: list[str], index: int, vocab: dict[str, int]) -> int:
    if is_decimal_point(tokens, index):
        normalized = DECIMAL_POINT
    elif is_nonterminal_period(tokens, index):
        normalized = NONTERMINAL_PERIOD
    else:
        normalized = normalize_token(tokens[index])
    if normalized in vocab:
        return vocab[normalized]
    return vocab.get(normalize_token(tokens[index]), vocab[UNK])


def cue_id(token: str) -> int:
    tok = normalize_token(token)
    if tok in COORD_CUES:
        return 1
    if tok in SUBORD_CUES:
        return 2
    if tok in ADVERBIAL_CUES:
        return 3
    if tok in RELATIVE_CUES:
        return 4
    if tok in REPORTING_CUES:
        return 5
    return 0


def bucket(value: int, cap: int) -> int:
    return max(0, min(value, cap))


def stream_feature_ids(tokens: list[str]) -> list[list[int]]:
    feats: list[list[int]] = []
    since_punct = 16
    depth = 0
    for i, token in enumerate(tokens):
        pos_bucket = bucket(i, 31)
        since_bucket = bucket(since_punct, 16)
        depth_bucket = bucket(depth, 8)
        feats.append([pos_bucket, since_bucket, depth_bucket])
        if token in "([{":
            depth += 1
        elif token in ")]}":
            depth = max(0, depth - 1)
        if token in CLAUSE_PUNCT_SET:
            since_punct = 0
        else:
            since_punct += 1

    return feats


def distance_to_boundary_ids(labels: list[int], cap: int = 8) -> list[int]:
    boundary_positions = [i for i, label in enumerate(labels) if label]
    if not boundary_positions:
        return [cap] * len(labels)
    out: list[int] = []
    j = 0
    for i in range(len(labels)):
        while j < len(boundary_positions) - 1 and boundary_positions[j] < i:
            j += 1
        out.append(bucket(boundary_positions[j] - i, cap))
    return out


def age_since_boundary_ids(labels: list[int], cap: int = 24) -> list[int]:
    out: list[int] = []
    age = 0
    for label in labels:
        out.append(bucket(age, cap))
        if label:
            age = 0
        else:
            age += 1
    return out


def rule_boundary_labels(tokens: list[str]) -> list[float]:
    labels: list[float] = []
    for token in tokens:
        tok = normalize_token(token)
        labels.append(float(token in CLAUSE_PUNCT_SET or tok in {"therefore", "however", "thus", "hence"}))
    return labels


def stable_hash(text: str, modulo: int) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % modulo


def char_ngram_ids(token: str, buckets: int = 2048, slots: int = 8) -> list[int]:
    text = f"<{normalize_token(token)}>"
    grams: list[str] = []
    for n in (2, 3, 4):
        grams.extend(text[i : i + n] for i in range(max(0, len(text) - n + 1)))
    ids = [stable_hash(g, buckets) + 1 for g in grams[:slots]]
    ids.extend([0] * (slots - len(ids)))
    return ids


def encode_docs(
    docs: list[Document],
    vocab: dict[str, int],
    past: int,
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = past + lookahead + 1
    xs = []
    shapes = []
    labels = []
    is_final = []
    for doc in docs:
        ids = [contextual_token_id(doc.tokens, i, vocab) for i in range(len(doc.tokens))]
        sids = contextual_shape_ids(doc.tokens)
        n = len(ids)
        for i in range(n):
            win = []
            swin = []
            for j in range(i - past, i + lookahead + 1):
                if 0 <= j < n:
                    win.append(ids[j])
                    swin.append(sids[j])
                else:
                    win.append(0)
                    swin.append(0)
            xs.append(win)
            shapes.append(swin)
            labels.append(doc.labels[i])
            is_final.append(i == n - 1)
    return (
        np.asarray(xs, dtype=np.int64).reshape(-1, width),
        np.asarray(shapes, dtype=np.int64).reshape(-1, width),
        np.asarray(labels, dtype=np.float32),
        np.asarray(is_final, dtype=bool),
    )


def encode_sequence_docs(
    docs: list[Document],
    vocab: dict[str, int],
    char_buckets: int = 2048,
    char_slots: int = 8,
) -> list[dict[str, np.ndarray]]:
    encoded = []
    for doc in docs:
        starts = doc.starts if doc.starts is not None else [0] * len(doc.tokens)
        hard_negative = doc.hard_negative if doc.hard_negative is not None else [0] * len(doc.tokens)
        encoded.append(
            {
                "tokens": np.asarray(
                    [contextual_token_id(doc.tokens, i, vocab) for i in range(len(doc.tokens))],
                    dtype=np.int64,
                ),
                "shapes": np.asarray(contextual_shape_ids(doc.tokens), dtype=np.int64),
                "cues": np.asarray([cue_id(tok) for tok in doc.tokens], dtype=np.int64),
                "stream_feats": np.asarray(stream_feature_ids(doc.tokens), dtype=np.int64),
                "chars": np.asarray([char_ngram_ids(tok, char_buckets, char_slots) for tok in doc.tokens], dtype=np.int64),
                "labels": np.asarray(doc.labels, dtype=np.float32),
                "starts": np.asarray(starts, dtype=np.float32),
                "dist": np.asarray(distance_to_boundary_ids(doc.labels), dtype=np.int64),
                "age": np.asarray(age_since_boundary_ids(doc.labels), dtype=np.int64),
                "decimal": np.asarray([is_decimal_point(doc.tokens, i) for i in range(len(doc.tokens))], dtype=bool),
                "nonterminal_period": np.asarray(
                    [is_nonterminal_period(doc.tokens, i) for i in range(len(doc.tokens))],
                    dtype=bool,
                ),
                "rules": np.asarray(rule_boundary_labels(doc.tokens), dtype=np.float32),
                "hard_negative": np.asarray(hard_negative, dtype=bool),
                "final": np.asarray([i == len(doc.tokens) - 1 for i in range(len(doc.tokens))], dtype=bool),
            }
        )
    return encoded


def make_decimal_augmentation(count: int, seed: int = 0) -> list[Document]:
    """Create decimal hard negatives without using the GSM evaluation set."""

    if count <= 0:
        return []
    rng = random.Random(seed)
    units = ["dollars", "euros", "pounds", "liters", "meters", "kilograms", "hours", "percent"]
    subjects = ["price", "value", "rate", "distance", "weight", "result", "measurement", "total"]
    decimal_templates = [
        lambda w, f, u, s: ["The", s, "is", w, ".", f, u, "."],
        lambda w, f, u, s: ["A", "recorded", s, "was", w, ".", f, u, "."],
        lambda w, f, u, s: ["They", "used", w, ".", f, u, "in", "total", "."],
        lambda w, f, u, s: ["The", "calculation", "gave", w, ".", f, "."],
        lambda w, f, u, s: ["It", "costs", "$", w, ".", f, "per", "item", "."],
        lambda w, f, u, s: ["The", s, "changed", "from", w, ".", f, "to", str(int(w) + 1), ".", f, "."],
    ]
    integer_templates = [
        lambda w, u, s: ["The", s, "is", w, "."],
        lambda w, u, s: ["They", "used", w, u, "."],
        lambda w, u, s: ["The", "final", "answer", "is", w, "."],
    ]

    docs: list[Document] = []
    for i in range(count):
        whole = str(rng.randint(0, 999))
        fraction = str(rng.randint(0, 99)).zfill(rng.choice([1, 2]))
        unit = rng.choice(units)
        subject = rng.choice(subjects)
        if i % 4 == 0:
            tokens = rng.choice(integer_templates)(whole, unit, subject)
        else:
            tokens = rng.choice(decimal_templates)(whole, fraction, unit, subject)
        labels = [0] * len(tokens)
        labels[-1] = 1
        starts = [0] * len(tokens)
        starts[0] = 1
        docs.append(Document(tokens=tokens, labels=labels, starts=starts))
    return docs


def make_period_augmentation(count: int, seed: int = 0) -> list[Document]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    names = ["Adams", "Brown", "Clark", "Davis", "Evans", "Garcia", "Harris", "Patel"]
    verbs = ["answered", "arrived", "called", "continued", "replied", "returned", "spoke"]
    nouns = ["chart", "document", "example", "record", "result", "table"]
    domains = ["alpha", "beta", "research", "service", "status", "team"]
    suffixes = sorted(DOMAIN_SUFFIXES)
    extensions = sorted(FILE_EXTENSIONS)
    titles = sorted(TITLE_ABBREVIATIONS - {"st"})
    number_abbreviations = sorted(NUMBER_ABBREVIATIONS)
    months = sorted(MONTH_ABBREVIATIONS)

    def make_case(case_id: int) -> list[str]:
        name = rng.choice(names)
        verb = rng.choice(verbs)
        noun = rng.choice(nouns)
        number = str(rng.randint(1, 99))
        case_type = case_id % 16
        if case_type == 0:
            return [rng.choice(titles).title(), ".", name, verb, "."]
        if case_type == 1:
            return ["The", rng.choice(["E", "U", "N"]), ".", rng.choice(["U", "S", "K"]), ".", noun, verb, "."]
        if case_type == 2:
            return [rng.choice(["A", "C", "M", "R"]), ".", rng.choice(["B", "D", "L", "T"]), ".", name, verb, "."]
        if case_type == 3:
            return ["Choose", noun, ",", "e", ".", "g", ".", ",", rng.choice(nouns), "."]
        if case_type == 4:
            return ["The", "term", "is", ",", "i", ".", "e", ".", ",", "specific", "."]
        if case_type == 5:
            return ["Return", "at", number, rng.choice(["a", "p"]), ".", "m", ".", "today", "."]
        if case_type == 6:
            return ["Consult", rng.choice(number_abbreviations).title(), ".", number, "before", "continuing", "."]
        if case_type == 7:
            return ["Delivery", "is", rng.choice(months).title(), ".", number, "."]
        if case_type == 8:
            return ["Browse", rng.choice(domains), ".", rng.choice(suffixes), "for", "updates", "."]
        if case_type == 9:
            return ["Check", rng.choice(sorted(DOMAIN_PREFIXES)), ".", rng.choice(domains), ".", rng.choice(suffixes), "now", "."]
        if case_type == 10:
            return ["Load", noun, ".", rng.choice(extensions), "before", "starting", "."]
        if case_type == 11:
            return ["Hmm", ".", ".", ".", "perhaps", "tomorrow", "."]
        if case_type == 12:
            return ["They", "work", "on", "St", ".", name, "Avenue", "."]
        if case_type == 13:
            return ["Acme", "Inc", ".", "reported", "growth", "."]
        if case_type == 14:
            return ["Compare", "the", "old", "method", "vs", ".", "the", "new", "one", "."]
        return ["This", noun, "is", "complete", "."]
    docs = []
    for i in range(count):
        tokens = make_case(i)
        labels = [0] * len(tokens)
        labels[-1] = 1
        starts = [0] * len(tokens)
        starts[0] = 1
        docs.append(Document(tokens=tokens, labels=labels, starts=starts))
    return docs


def make_biomedical_parenthetical_augmentation(count: int, seed: int = 0) -> list[Document]:
    """构造医学统计括号、P 值和缩写同位语的困难负样本。"""

    if count <= 0:
        return []
    rng = random.Random(seed)
    symptoms = ["hemiplegia", "numbness", "sleep", "pain", "mood", "falls"]
    cortexes = ["M1", "S1", "S2"]
    abbreviations = ["rTMS", "EMS", "CI", "RCT"]

    def finish(tokens: list[str]) -> Document:
        labels = [0] * len(tokens)
        labels[-1] = 1
        starts = [0] * len(tokens)
        starts[0] = 1
        hard_negative = [0] * len(tokens)
        for index in risky_internal_indexes(tokens):
            hard_negative[index] = 1
        return Document(tokens=tokens, labels=labels, starts=starts, hard_negative=hard_negative)

    def finish_prefix(tokens: list[str]) -> Document:
        labels = [0] * len(tokens)
        starts = [0] * len(tokens)
        starts[0] = 1
        hard_negative = [0] * len(tokens)
        for index in risky_internal_indexes(tokens):
            hard_negative[index] = 1
        return Document(tokens=tokens, labels=labels, starts=starts, hard_negative=hard_negative)

    def risky_internal_indexes(tokens: list[str]) -> list[int]:
        indexes = []
        for index, token in enumerate(tokens[:-1]):
            following = tokens[index + 1]
            if token in {")", "]", "<", "=", "/", "%"}:
                indexes.append(index)
            elif following in {"(", "[", "/"}:
                indexes.append(index)
        return indexes

    def short_p_value() -> list[str]:
        return [".", rng.choice(["001", "005", "01", "05"])]

    def ranged_p_value() -> list[str]:
        left = rng.choice(["000", "013", "021", "034"])
        right = rng.choice(["001", "046", "085", "115"])
        return ["0", ".", left, "-", "0", ".", right]

    def decimal_percent() -> list[str]:
        whole = str(rng.randint(50, 95))
        frac = str(rng.randint(1, 9))
        return [whole, ".", frac, "%"]

    def decimal_range() -> list[str]:
        left_whole = str(rng.randint(40, 95))
        left_frac = str(rng.randint(1, 99)).zfill(2)
        right_whole = str(rng.randint(100, 150))
        right_frac = str(rng.randint(1, 99)).zfill(2)
        return [left_whole, ".", left_frac + "-" + right_whole, ".", right_frac]

    def make_case(case_id: int) -> list[str]:
        total = str(rng.randint(300, 1200))
        before = str(rng.randint(100, 600))
        after = str(rng.randint(100, 600))
        months = str(rng.randint(2, 8))
        age = str(rng.choice([55, 60, 65, 70, 75]))
        case_type = case_id % 12
        if case_type == 0:
            return [
                "A", "total", "of", total, "people", "responded", "before",
                "(", "n", "=", before, ")", "and", months, "months", "after",
                "(", "n", "=", after, ")", "the", "campaign", ".",
            ]
        if case_type == 1:
            return [
                "Of", "these", ",", total, "(", *decimal_percent(), ")",
                "were", "aged", age, "years", "or", "older", ".",
            ]
        if case_type == 2:
            return [
                "The", "percentages", "of", "correct", "answers", "for",
                rng.choice(symptoms), "(", "P", "<", *short_p_value(), "for",
                "both", ")", "and", "calling", "emergency", "medical",
                "services", "(", "EMS", ")", "on", "identifying", "stroke",
                "occurrence", "(", "P", "<", *short_p_value(), ")", "were",
                "higher", "after", "the", "campaign", ".",
            ]
        if case_type == 3:
            return [
                "The", "effect", "was", "significant", "(", "P", "=",
                *ranged_p_value(), "in", "sleep", "scores", ")", "but",
                "the", "outcome", "did", "not", "change", ".",
            ]
        if case_type == 4:
            return [
                "Mechanisms", "underlying", "alleviation", "of", "pain", "by",
                "repetitive", "transcranial", "magnetic", "stimulation",
                "(", "rTMS", ")", "of", "primary", "motor", "cortex",
                "(", "M1", ")", "and", "right", "secondary",
                "somatosensory", "cortex", "(", "S2", ")", "are", "only",
                "partly", "known", ".",
            ]
        if case_type == 5:
            return [
                "Intensity", "and", "interference", "of", "pain", ",",
                "mood", ",", "and", "sleep", "were", "evaluated", "after",
                months, "active", "(", "primary", "somatosensory",
                "cortex", "[", "S1", "]", "/", "M1", "and", "S2", ")",
                "and", "placebo", "rTMS", "treatments", ".",
            ]
        if case_type == 6:
            return [
                "The", "respondents", "aged", age, "years", "or", "older",
                "had", "higher", "scores", "(", "P", "<", *short_p_value(),
                "and", "P", "<", *short_p_value(), ",", "respectively", ")",
                ".",
            ]
        if case_type == 7:
            return [
                "Patients", "received", "treatment", "(", "n", "=",
                before, ")", "or", "placebo", "(", "n", "=", after, ")",
                "during", "the", "trial", ".",
            ]
        if case_type == 8:
            return [
                "The", "incidence", "was", *decimal_percent(), "(", "95",
                "%", "CI", *decimal_range(), ")", "after", "follow-up", ".",
            ]
        if case_type == 9:
            return [
                "The", rng.choice(cortexes), "/", "M1", "stimulation",
                "improved", "sleep", "without", "significant", "analgesic",
                "effect", "(", "P", "=", *ranged_p_value(), "in", "sleep",
                "scores", ")", ".",
            ]
        if case_type == 10:
            return [
                "The", "service", "name", "(", rng.choice(abbreviations), ")",
                "was", "recorded", "inside", "the", "study", ".",
            ]
        return [
            "Participants", "were", "followed", "for", months, "months",
            "(", "95", "%", "CI", *decimal_range(), ")", "and", "then",
            "interviewed", ".",
        ]

    docs = []
    for i in range(count):
        tokens = make_case(i)
        if i % 2 == 0:
            docs.append(finish(tokens))
            continue
        risks = risky_internal_indexes(tokens)
        if not risks:
            docs.append(finish(tokens))
            continue
        end = min(len(tokens), rng.choice(risks) + 2)
        docs.append(finish_prefix(tokens[:end]))
    return docs


def collate_sequences(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    max_len = max(len(item["tokens"]) for item in batch)
    char_slots = batch[0]["chars"].shape[1]
    out = {
        "tokens": np.zeros((len(batch), max_len), dtype=np.int64),
        "shapes": np.zeros((len(batch), max_len), dtype=np.int64),
        "cues": np.zeros((len(batch), max_len), dtype=np.int64),
        "stream_feats": np.zeros((len(batch), max_len, 3), dtype=np.int64),
        "chars": np.zeros((len(batch), max_len, char_slots), dtype=np.int64),
        "labels": np.zeros((len(batch), max_len), dtype=np.float32),
        "starts": np.zeros((len(batch), max_len), dtype=np.float32),
        "dist": np.zeros((len(batch), max_len), dtype=np.int64),
        "age": np.zeros((len(batch), max_len), dtype=np.int64),
        "decimal": np.zeros((len(batch), max_len), dtype=bool),
        "nonterminal_period": np.zeros((len(batch), max_len), dtype=bool),
        "rules": np.zeros((len(batch), max_len), dtype=np.float32),
        "hard_negative": np.zeros((len(batch), max_len), dtype=bool),
        "mask": np.zeros((len(batch), max_len), dtype=bool),
        "final": np.zeros((len(batch), max_len), dtype=bool),
    }
    for i, item in enumerate(batch):
        n = len(item["tokens"])
        out["tokens"][i, :n] = item["tokens"]
        out["shapes"][i, :n] = item["shapes"]
        out["cues"][i, :n] = item["cues"]
        out["stream_feats"][i, :n] = item["stream_feats"]
        out["chars"][i, :n] = item["chars"]
        out["labels"][i, :n] = item["labels"]
        out["starts"][i, :n] = item["starts"]
        out["dist"][i, :n] = item["dist"]
        out["age"][i, :n] = item["age"]
        out["decimal"][i, :n] = item["decimal"]
        out["nonterminal_period"][i, :n] = item["nonterminal_period"]
        out["rules"][i, :n] = item["rules"]
        out["hard_negative"][i, :n] = item["hard_negative"]
        out["mask"][i, :n] = True
        out["final"][i, :n] = item["final"]
    return out


def describe_split(split: str, data_root: Path = DATA_ROOT) -> dict[str, float]:
    docs = read_split(split, data_root)
    n_docs = len(docs)
    n_tokens = sum(len(d.tokens) for d in docs)
    n_boundaries = sum(sum(d.labels) for d in docs)
    return {
        "docs": n_docs,
        "tokens": n_tokens,
        "boundaries": n_boundaries,
        "boundary_rate": n_boundaries / max(n_tokens, 1),
    }
