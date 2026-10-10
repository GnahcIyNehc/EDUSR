from __future__ import annotations

import re
from dataclasses import dataclass

import torch


PAD = "<pad>"
UNK = "<unk>"
DECIMAL_POINT = "<decimal_point>"
NONTERMINAL_PERIOD = "<nonterminal_period>"
PUNCT_SET = set(".,;:!?)]}\"'")
OPEN_SET = set("([{\"'")
CLAUSE_PUNCT_SET = set(",;:!?")
COORD_CUES = {"and", "but", "or", "nor", "yet", "so"}
SUBORD_CUES = {
    "after", "although", "as", "because", "before", "even", "if", "once",
    "since", "though", "unless", "until", "when", "whenever", "whereas",
    "whether", "while",
}
ADVERBIAL_CUES = {
    "accordingly", "additionally", "also", "alternatively", "consequently",
    "furthermore", "hence", "however", "instead", "meanwhile", "moreover",
    "nevertheless", "nonetheless", "otherwise", "therefore", "thus",
}
RELATIVE_CUES = {"that", "which", "who", "whom", "whose", "where"}
REPORTING_CUES = {"said", "says", "say", "reported", "told", "asked", "according"}
TITLE_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st"}
NUMBER_ABBREVIATIONS = {"fig", "no", "eq", "ch", "sec", "pp", "vol"}
MONTH_ABBREVIATIONS = {"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
GENERAL_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "approx", "dept", "inc", "ltd", "co", "al", "cont",
}
DOMAIN_PREFIXES = {"www", "docs", "api", "mail", "shop", "blog", "support", "portal"}
DOMAIN_SUFFIXES = {"com", "org", "net", "edu", "gov", "io", "ai", "co", "uk", "de", "fr", "cn", "jp"}
FILE_EXTENSIONS = {
    "pdf", "txt", "csv", "json", "xml", "html", "htm", "py", "js", "ts",
    "md", "doc", "docx", "xls", "xlsx", "png", "jpg", "jpeg",
}
CODE_BLOCK_LANGS = {"asy", "asymptote"}
CODE_FUNCTION_KEYWORDS = {
    "add", "arc", "axes", "axis", "clip", "cycle", "dot", "draw", "fill",
    "filldraw", "label", "pair", "path", "pen", "real", "size", "unitsize",
}
CODE_BRACKETS = {"(", ")", "[", "]", "{", "}"}
CODE_PUNCT = CODE_BRACKETS | {",", ";", ":", "=", "+", "-", "*", "/", "<", ">"}
LATEX_MATH_COMMANDS = {
    "alpha", "approx", "arccos", "arcsin", "arctan", "arg", "bar", "beta",
    "binom", "cap", "cdot", "ceil", "cos", "cot", "cup", "delta", "div",
    "equiv", "exists", "exp", "forall", "frac", "gamma", "ge", "geq", "gt",
    "in", "infty", "int", "langle", "lceil", "left", "le", "leq", "lfloor",
    "lim", "ln", "log", "lt", "mathbb", "mathbf", "mathcal", "mathrm",
    "neq", "notin", "overline", "pi", "pm", "prod", "rangle", "rceil",
    "rfloor", "right", "sin", "sqrt", "sum", "tan", "theta", "times", "to",
    "underline", "varphi",
}
LATEX_MATH_ENVIRONMENTS = {
    "align", "align*", "displaymath", "equation", "equation*", "gather",
    "gather*", "math", "multline", "multline*",
}
MATH_SYMBOL_TOKENS = {
    "$", "\\", "^", "_", "{", "}", "|", "=", "+", "-", "*", "/", "<", ">",
    "≤", "≥", "≠", "≈", "±", "×", "÷",
}
CURRENCY_SYMBOLS = {
    "$",
    "\u00a3",  # pound sterling
    "\u00a5",  # yen/yuan
    "\u20ac",  # euro
    "\u20ba",  # Turkish lira
    "\u20b9",  # Indian rupee
    "\u20a9",  # won
    "\u20bd",  # ruble
    "\uffe5",  # full-width yen/yuan
}
PERCENT_SYMBOLS = {"%"}
STATISTICAL_VARIABLES = {"p", "n", "ci", "sd", "se", "df", "hr", "or", "rr"}
STATISTICAL_OPERATORS = {"<", ">", "=", "\u2264", "\u2265"}
BIOMEDICAL_ABBREVIATION_HOST_CUES = {
    "cortex", "cortices", "mood", "occurrence", "pain", "score", "scores",
    "service", "services", "situ", "sleep", "stimulation", "therapy",
}
MWE_FIXED_PATTERNS = (
    ("according", "to"),
    ("after", "all"),
    ("as", "a", "result"),
    ("as", "long", "as"),
    ("as", "soon", "as"),
    ("as", "well", "as"),
    ("because", "of"),
    ("by", "contrast"),
    ("due", "to"),
    ("even", "if"),
    ("even", "though"),
    ("for", "example"),
    ("for", "instance"),
    ("given", "that"),
    ("in", "addition"),
    ("in", "addition", "to"),
    ("in", "fact"),
    ("in", "front", "of"),
    ("in", "order", "to"),
    ("note", "that"),
    ("on", "the", "other", "hand"),
    ("instead", "of"),
    ("adjusted", "opening", "pressure"),
    ("mean", "midazolam", "pretreatment", "dose"),
    ("opening", "pressure"),
    ("on", "behalf", "of"),
    ("prior", "to"),
    ("provided", "that"),
    ("rather", "than"),
    ("receiver", "operating", "curve"),
    ("receiver", "operating", "characteristic", "curve"),
    ("so", "as", "to"),
    ("such", "as"),
    ("with", "respect", "to"),
    ("as", "much", "as"),
    ("as", "many", "as"),
    ("more", "than"),
    ("less", "than"),
    ("fewer", "than"),
)
MWE_QUANTITY_MODIFIERS = {
    "about", "almost", "approximately", "around", "exactly", "just",
    "nearly", "only", "roughly",
}
MWE_QUANTITY_WORDS = {
    "a", "an", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion",
}
MWE_QUANTITY_SYMBOLS = CURRENCY_SYMBOLS | PERCENT_SYMBOLS
FRACTION_WORDS = {
    "half", "quarter", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "one-half", "one-third", "one-fourth",
    "one-quarter", "one-fifth", "one-sixth", "two-thirds", "three-fourths",
}
MEASUREMENT_UNITS = {
    "acre", "acres", "calorie", "calories", "cent", "cents", "centimeter", "centimeters", "cm",
    "cup", "cups", "day", "days", "dozen", "dozens", "dollar", "dollars",
    "euro", "euros", "feet", "foot", "ft", "g", "gallon", "gallons",
    "gram", "grams", "h2o", "hour", "hours", "inch", "inches", "kg", "kilogram",
    "kilograms", "kilometer", "kilometers", "km", "lb", "lbs", "liter",
    "liters", "litre", "litres", "m", "meter", "meters", "metre", "metres",
    "mg", "milligram", "milligrams",
    "mile", "miles", "milliliter", "milliliters", "ml", "minute", "minutes",
    "month", "months", "ounce", "ounces", "oz", "page", "pages", "percent",
    "percentage", "piece", "pieces", "pint", "pints", "pound", "pounds",
    "quart", "quarts", "rupee", "rupees", "second", "seconds", "sq", "square",
    "serving", "servings", "ton", "tons", "week", "weeks", "yard", "yards", "year", "years", "yen",
}
RATE_CONNECTORS = {"a", "an", "each", "every", "per"}
PERCENT_NOUNS = {
    "chance", "commission", "decrease", "discount", "fee", "increase",
    "interest", "markup", "off", "rate", "rise", "sale", "tax", "tip",
}
SIZE_ADJECTIVES = {"deep", "high", "long", "tall", "thick", "wide"}
QUOTED_FORMULA_CUES = {"buy", "discount", "get", "off", "per", "sale", "save"}
LEFT_QUOTES = {"\"", "\u201c", "\u2018"}
RIGHT_QUOTES = {"\"", "\u201d", "\u2019"}
OPEN_BRACKETS = {"(": ")", "[": "]", "{": "}"}
CLOSE_BRACKETS = {")", "]", "}"}
FINITE_AUXILIARIES = {
    "am", "are", "can", "could", "did", "do", "does", "had", "has", "have",
    "is", "may", "might", "must", "shall", "should", "was", "were", "will",
    "would",
}
MODAL_AUXILIARIES = {
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
}
COMPLEMENT_TAKING_VERBS = {
    "agree", "agrees", "agreed", "ask", "asks", "asked", "assume", "assumes",
    "assumed", "begin", "begins", "began", "calculate", "calculates",
    "calculated", "check", "checks", "checked", "decide", "decides", "decided",
    "expect", "expects", "expected", "find", "finds", "found", "give", "gives",
    "gave", "given", "has", "have", "had", "hope", "hopes", "hoped", "learn",
    "knows", "know", "known", "knew", "learns", "learn", "learned", "need",
    "needs", "needed", "plan", "plans", "planned", "note", "notes", "noted",
    "realize", "realizes", "realized", "say", "says", "said", "show", "shows",
    "showed", "start", "starts", "started", "suppose", "supposes", "supposed",
    "think", "thinks", "thought", "try", "tries", "tried", "want", "wants",
    "wanted",
}
TRANSITIVE_COMPLETION_VERBS = {
    "bought", "buy", "buys", "completed", "complete", "completes", "cost",
    "costs", "divide", "divided", "fill", "filled", "find", "finds", "found",
    "gave", "give", "gives", "got", "has", "have", "had", "made", "make",
    "makes", "need", "needs", "needed", "receive", "receives", "received",
    "use", "uses", "used", "want", "wants", "wanted",
}
PREPOSITIONAL_COMPLEMENT_CUES = {"about", "at", "by", "for", "from", "in", "into", "of", "on", "to", "with"}
SUBJECT_PRONOUNS = {"he", "her", "i", "it", "she", "they", "we", "who", "you"}
COMMON_FINITE_VERBS = {
    "apply", "applies", "are", "buy", "buys", "cost", "costs", "did", "do",
    "does", "get", "gets", "had", "has", "have", "is", "make", "makes",
    "receive", "receives", "use", "uses", "was", "were",
}
PREDICATE_HEADS = (
    COMMON_FINITE_VERBS
    | COMPLEMENT_TAKING_VERBS
    | TRANSITIVE_COMPLETION_VERBS
    | {
        "bake", "bakes", "breathe", "breathes", "dance", "dances", "develop",
        "develops", "eat", "eats", "exchange", "exchanges", "focus",
        "focuses", "go", "goes", "hold", "holds", "hurl", "hurling",
        "like", "likes", "look", "looks", "play", "plays", "see", "sees",
        "step", "steps", "swallow", "swallows", "swim", "swims",
        "compare", "compares", "compared", "earn", "earning", "earnings",
        "go", "going", "take", "takes", "watch", "watches", "watched",
        "yell", "yells",
    }
)
DISCOURSE_MARKER_PREFIXES = ADVERBIAL_CUES | {"finally", "first", "second", "then"}
FUNCTION_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "if", "in",
    "of", "on", "or", "the", "to", "with",
}
MINIMAL_PP_HEADS = PREPOSITIONAL_COMPLEMENT_CUES | {"after", "before", "during", "over", "under", "until"}
SCALAR_ANAPHOR_HEADS = {"time", "times"}
MATRIX_COMPLEMENT_CLOSURE_VERBS = {
    "assume", "assumes", "assumed", "find", "finds", "found", "know", "knows",
    "knew", "known", "note", "notes", "noted", "realize", "realizes",
    "realized", "see", "sees", "saw", "seen", "think", "thinks", "thought",
}
COMPLEMENTIZERS = {"that", "whether"}
COMPARATIVE_MULTIPLIERS = {
    "double", "doubled", "half", "thrice", "triple", "tripled", "twice",
}
COMPARATIVE_SCALARS = {"many", "much"}
COMPARATIVE_DEGREE_HEADS = {
    "big", "deep", "far", "fast", "high", "large", "long", "many", "much",
    "old", "small", "tall", "wide", "young",
}
ATTRIBUTIVE_MODIFIERS = {
    "annual", "average", "daily", "final", "first", "former", "future",
    "hourly", "last", "monthly", "normal", "only", "original", "other",
    "remaining", "same", "second", "total", "weekly", "yearly",
}
NOMINAL_HEADS = {
    "amount", "answer", "brand", "class", "cost", "day", "earning",
    "earnings", "episode", "episodes", "hour", "income", "kilogram",
    "kilograms", "minute", "minutes", "money", "pay", "price", "profit",
    "question", "questions", "revenue", "salary", "serving", "servings",
    "speed", "student", "students", "total", "wage", "wages", "watch",
    "week", "wind",
}
ESURF_CONTROL_INFINITIVE_VERBS = {
    "agree", "agrees", "agreed", "arrange", "arranges", "arranged",
    "choose", "chooses", "chose", "decide", "decides", "decided",
    "expect", "expects", "expected", "hope", "hopes", "hoped",
    "need", "needs", "needed", "plan", "plans", "planned",
    "prepare", "prepares", "prepared", "promise", "promises", "promised",
    "try", "tries", "tried", "want", "wants", "wanted",
}
ESURF_LIGHT_NOUN_OF_HEADS = {
    "amount", "bag", "bunch", "couple", "group", "kind", "lot", "lots",
    "number", "pair", "piece", "set", "sort", "type", "variety",
}
ESURF_OBJECT_LIST_ANCHORS = {
    "ate", "bought", "buy", "buys", "choose", "chooses", "chose",
    "collect", "collects", "collected", "eat", "eats", "find", "finds",
    "found", "get", "gets", "got", "pick", "picks", "picked", "purchase",
    "purchases", "purchased", "sell", "selling", "sells", "sold",
}
PURPOSE_INFINITIVE_ANCHORS = {
    "gave", "get", "gets", "give", "given", "gives", "got", "had", "has",
    "have",
}
ZERO_RELATIVE_CLASSIFIER_HEADS = {"brand", "kind", "sort", "type", "variety"}
FREE_RELATIVE_PRONOUNS = {"what", "whatever", "whichever", "whoever", "whomever"}
OBJECT_PRONOUNS = {"him", "her", "it", "me", "them", "us", "you"}
PREVERBAL_ADVERBS = {
    "also", "already", "eventually", "finally", "first", "just", "next",
    "now", "often", "only", "still", "then",
}
POSSESSIVE_DETERMINERS = {"his", "her", "its", "my", "our", "their", "your"}
DEICTIC_DETERMINERS = {"this", "that", "these", "those"}
NOMINAL_INFINITIVE_LEFT_DETERMINERS = (
    POSSESSIVE_DETERMINERS | DEICTIC_DETERMINERS | {"a", "an", "no", "the"}
)
KINSHIP_OR_ROLE_HEADS = {
    "aunt", "boss", "brother", "cousin", "daughter", "father", "friend",
    "grandfather", "grandmother", "mother", "sister", "son", "uncle",
}
RESTRICTIVE_OF_HEADS = {
    "amount", "batch", "class", "group", "lot", "number", "part", "portion",
    "rest", "set", "share", "total",
}
RANGE_OPENERS = {"between", "from"}
RANGE_CLOSERS = {"between": "and", "from": "to"}
ESURF_NOMINAL_STOPWORDS = (
    FUNCTION_WORDS
    | FINITE_AUXILIARIES
    | COORD_CUES
    | SUBORD_CUES
    | RELATIVE_CUES
    | ADVERBIAL_CUES
)
FRACTION_DENOMINATORS = {
    "half", "halves", "third", "thirds", "fourth", "fourths", "quarter",
    "quarters", "fifth", "fifths", "sixth", "sixths", "seventh",
    "sevenths", "eighth", "eighths", "ninth", "ninths", "tenth",
    "tenths",
}
TEMPORAL_FREQUENCY_DETERMINERS = {"each", "every"}
TEMPORAL_FREQUENCY_UNITS = {
    "day", "days", "hour", "hours", "minute", "minutes", "month", "months",
    "morning", "mornings", "night", "nights", "second", "seconds", "week",
    "weeks", "year", "years",
}
INLINE_FORMULA_INTRO_CUES = {
    "calculated", "computed", "defined", "equation", "follows", "formula",
    "given",
}
INLINE_FORMULA_OPERATORS = {"/", "*", "×", "÷", "+", "-"}


def normalize_token(token: str) -> str:
    token = token.lower()
    if re.fullmatch(r"\d+([.,:/-]\d+)*", token):
        return "<num>"
    if re.fullmatch(r"https?://\S+|www\.\S+", token):
        return "<url>"
    if "@" in token and "." in token:
        return "<email>"
    return token


def is_numeric_piece(token: str) -> bool:
    return bool(re.fullmatch(r"\d+", token))


def _is_numeric_decimal_neighbor(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:-\d+)?", token))


def is_decimal_point(tokens: list[str], index: int) -> bool:
    if tokens[index] != "." or index == 0 or index + 1 >= len(tokens):
        return False
    if (
        _is_numeric_decimal_neighbor(tokens[index - 1])
        and _is_numeric_decimal_neighbor(tokens[index + 1])
    ):
        return True
    return (
        tokens[index - 1] in STATISTICAL_OPERATORS
        and _is_numeric_decimal_neighbor(tokens[index + 1])
    )


def is_numeric_internal_colon(tokens: list[str], index: int) -> bool:
    return (
        tokens[index] == ":"
        and index > 0
        and index + 1 < len(tokens)
        and is_numeric_piece(tokens[index - 1])
        and is_numeric_piece(tokens[index + 1])
    )


def _lower_tokens(tokens: list[str]) -> list[str]:
    return [token.lower() for token in tokens]


def _matches_pattern(lowered: list[str], start: int, pattern: tuple[str, ...]) -> bool:
    end = start + len(pattern)
    return end <= len(lowered) and tuple(lowered[start:end]) == pattern


def _is_quantity_start(token: str) -> bool:
    lowered = token.lower()
    return (
        is_numeric_piece(token)
        or lowered in MWE_QUANTITY_WORDS
        or token in MWE_QUANTITY_SYMBOLS
        or bool(re.fullmatch(r"\d+(?:[.,]\d+)?%?", token))
    )


def _has_quantity_modifier_before(lowered: list[str], index: int) -> bool:
    return index > 0 and lowered[index - 1] in MWE_QUANTITY_MODIFIERS


def _quantity_up_to_is_confirmed(tokens: list[str], lowered: list[str], index: int) -> bool:
    if lowered[index] != "up" or index + 1 >= len(tokens) or lowered[index + 1] != "to":
        return False
    if _has_quantity_modifier_before(lowered, index):
        return True
    return index + 2 < len(tokens) and _is_quantity_start(tokens[index + 2])


def _is_attributive_modifier(token: str) -> bool:
    lowered = token.lower().rstrip(".")
    return (
        lowered in ATTRIBUTIVE_MODIFIERS
        or lowered.endswith("year")
        or lowered.endswith("minute")
        or lowered.endswith("day")
        or lowered.endswith("month")
        or lowered.endswith("week")
    )


def _is_nominal_head_candidate(token: str) -> bool:
    lowered = token.lower().rstrip(".")
    if lowered in NOMINAL_HEADS or _is_measurement_unit(token):
        return True
    if lowered in PREDICATE_HEADS or lowered in FINITE_AUXILIARIES:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*(?:s|ing|tion|ment|ness|ity|er|or)?", token))


def _is_discourse_marker_prefix_boundary(tokens: list[str], index: int) -> bool:
    if tokens[index] not in CLAUSE_PUNCT_SET or index + 1 >= len(tokens):
        return False
    start = 0
    for position in range(index - 1, -1, -1):
        if tokens[position] in PUNCT_SET | CLAUSE_PUNCT_SET:
            start = position + 1
            break
    lowered = _lower_tokens(tokens[start : index + 1])
    content = [token for token in lowered if token not in {",", ";", ":"}]
    return len(content) == 1 and content[0] in DISCOURSE_MARKER_PREFIXES


def _is_subject_candidate_token(token: str) -> bool:
    lowered = token.lower()
    if lowered in SUBJECT_PRONOUNS:
        return True
    if lowered in FUNCTION_WORDS or lowered in FINITE_AUXILIARIES or lowered in PREDICATE_HEADS:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Za-z'-]*|[A-Za-z]+(?:'s)", token))


def _is_direct_subject_candidate_token(token: str) -> bool:
    lowered = token.lower()
    if lowered in {"he", "i", "it", "she", "they", "we", "you"}:
        return True
    if lowered in FUNCTION_WORDS or lowered in FINITE_AUXILIARIES or lowered in PREDICATE_HEADS:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Za-z'-]*|[A-Za-z]+(?:'s)", token))


def mwe_boundary_closure_mask(tokens: list[str]) -> list[bool]:
    """Mask boundaries that would split a lexicalized multiword expression.

    The entries are lexicalized MWE/complex-preposition patterns rather than
    surface error patches. A boundary may occur before or after the expression,
    but not after a non-final token inside the expression. The same layer also
    applies a minimal PP chunk constraint: a preposition cannot be emitted as an
    EDU before its local complement begins.
    """

    mask = [False] * len(tokens)
    lowered = _lower_tokens(tokens)
    for start in range(len(tokens)):
        for pattern in MWE_FIXED_PATTERNS:
            if _matches_pattern(lowered, start, pattern):
                for index in range(start, start + len(pattern) - 1):
                    mask[index] = True

    for index in range(len(tokens)):
        if _is_discourse_marker_prefix_boundary(tokens, index):
            mask[index] = True
        if (
            lowered[index] in MINIMAL_PP_HEADS
            and index + 1 < len(tokens)
            and tokens[index + 1] not in PUNCT_SET | CLAUSE_PUNCT_SET
        ):
            mask[index] = True
        if lowered[index] in SCALAR_ANAPHOR_HEADS and index + 1 < len(tokens) and lowered[index + 1] == "that":
            mask[index] = True
        if (
            lowered[index] in COMPARATIVE_MULTIPLIERS
            and index + 1 < len(tokens)
            and lowered[index + 1] == "as"
        ):
            mask[index] = True
        if (
            lowered[index] == "as"
            and index + 1 < len(tokens)
            and lowered[index + 1] in COMPARATIVE_SCALARS
        ):
            mask[index] = True
        if (
            lowered[index] in COMPARATIVE_SCALARS
            and index > 0
            and lowered[index - 1] == "as"
            and index + 1 < len(tokens)
            and lowered[index + 1] == "as"
        ):
            mask[index] = True
        if (
            lowered[index] == "as"
            and index > 1
            and lowered[index - 1] in COMPARATIVE_SCALARS
            and lowered[index - 2] == "as"
            and index + 1 < len(tokens)
            and tokens[index + 1] not in PUNCT_SET | CLAUSE_PUNCT_SET
        ):
            mask[index] = True
        if (
            _is_attributive_modifier(tokens[index])
            and index + 1 < len(tokens)
            and (
                _is_attributive_modifier(tokens[index + 1])
                or _is_nominal_head_candidate(tokens[index + 1])
            )
        ):
            mask[index] = True
        if _quantity_up_to_is_confirmed(tokens, lowered, index):
            mask[index] = True
        elif lowered[index] == "up" and _has_quantity_modifier_before(lowered, index):
            if index + 1 == len(tokens) or lowered[index + 1] == "to":
                mask[index] = True
    return mask


def mwe_boundary_needs_more_context(tokens: list[str], index: int) -> bool:
    """Return true for ambiguous MWE prefixes that need one extra token.

    Example: after seeing "up to" the stream needs the next token to decide
    whether this is a quantity upper-bound MWE ("up to 10") or a particle verb
    followed by an infinitival clause ("ramping up to have ...").
    """

    lowered = _lower_tokens(tokens)
    return (
        0 <= index < len(tokens)
        and (
            (
                lowered[index] == "up"
                and index + 1 < len(tokens)
                and lowered[index + 1] == "to"
                and index + 2 >= len(tokens)
                and not _has_quantity_modifier_before(lowered, index)
            )
            or (
                lowered[index] in COMPARATIVE_MULTIPLIERS
                and index + 1 < len(tokens)
                and lowered[index + 1] == "as"
                and index + 2 >= len(tokens)
            )
            or (
                lowered[index] in COMPARATIVE_SCALARS
                and index > 0
                and lowered[index - 1] == "as"
                and index + 1 < len(tokens)
                and lowered[index + 1] == "as"
                and index + 2 >= len(tokens)
            )
            or (
                lowered[index] in FINITE_AUXILIARIES
                and index + 1 < len(tokens)
                and _is_subject_candidate_token(tokens[index + 1])
                and index + 2 >= len(tokens)
            )
            or (
                index + 1 < len(tokens)
                and lowered[index + 1] in TEMPORAL_FREQUENCY_DETERMINERS
                and _is_measurement_unit(tokens[index])
                and _has_quantity_before(tokens, index)
                and index + 2 >= len(tokens)
            )
            or (
                lowered[index] in FREE_RELATIVE_PRONOUNS
                and index + 1 < len(tokens)
                and lowered[index + 1] in SUBJECT_PRONOUNS
                and index + 2 >= len(tokens)
            )
            or _is_zero_relative_classifier_boundary(
                tokens,
                index,
                require_predicate=False,
            )
        )
    )


def _mark_internal(mask: list[bool], start: int, end: int) -> None:
    for position in range(max(0, start), min(end, len(mask) - 1) + 1):
        mask[position] = True


def _is_number_word(token: str) -> bool:
    lowered = token.lower()
    return lowered in MWE_QUANTITY_WORDS or lowered in FRACTION_WORDS


def _numeric_span_end(tokens: list[str], start: int) -> int | None:
    if start >= len(tokens):
        return None
    token = tokens[start]
    lowered = token.lower()
    if is_numeric_piece(token):
        end = start
        if (
            start + 2 < len(tokens)
            and tokens[start + 1] in {".", ","}
            and is_numeric_piece(tokens[start + 2])
        ):
            end = start + 2
        elif (
            start + 2 < len(tokens)
            and tokens[start + 1] == "/"
            and is_numeric_piece(tokens[start + 2])
        ):
            end = start + 2
        return end
    if _is_number_word(token):
        return start
    if re.fullmatch(r"\d+(?:[.,]\d+)?%?", token):
        return start
    if re.fullmatch(r"\d+(?:st|nd|rd|th)", lowered):
        return start
    if re.fullmatch(r"\d+(?:[a-z]+)", lowered):
        suffix = re.sub(r"^\d+", "", lowered)
        if suffix in MEASUREMENT_UNITS:
            return start
    return None


def _is_measurement_unit(token: str) -> bool:
    return token.lower().rstrip(".") in MEASUREMENT_UNITS


def _is_measurement_fused_token(token: str) -> bool:
    lowered = token.lower()
    match = re.fullmatch(r"\d+(?:[.,]\d+)?([a-z]+)", lowered)
    return bool(match and match.group(1) in MEASUREMENT_UNITS)


def _is_percentage_marker(token: str) -> bool:
    return token in PERCENT_SYMBOLS or token.lower() in {"percent", "percentage"}


def measurement_span_mask(tokens: list[str]) -> list[bool]:
    """Mask boundaries inside quantity, currency, unit, and rate expressions.

    This is a shallow measurement-expression chunker: it treats prices,
    percentages, physical units, fractions, and rate phrases as local semantic
    units, so an EDU boundary cannot fall inside the quantity phrase.
    """

    mask = [False] * len(tokens)
    lowered = _lower_tokens(tokens)
    for index, token in enumerate(tokens):
        numeric_end = _numeric_span_end(tokens, index)
        if numeric_end is not None and numeric_end > index:
            _mark_internal(mask, index, numeric_end - 1)

        if token in CURRENCY_SYMBOLS:
            amount_end = _numeric_span_end(tokens, index + 1)
            if amount_end is not None:
                _mark_internal(mask, index, amount_end - 1)
                after_amount = amount_end + 1
                if after_amount < len(tokens) and lowered[after_amount] in RATE_CONNECTORS:
                    mask[amount_end] = True
                    if (
                        after_amount + 1 < len(tokens)
                        and _is_measurement_unit(tokens[after_amount + 1])
                    ):
                        mask[after_amount] = True
                elif after_amount < len(tokens) and lowered[after_amount] in {"apiece", "each", "total"}:
                    mask[amount_end] = True
            continue

        if numeric_end is None:
            if (
                _is_measurement_fused_token(token)
                and index + 1 < len(tokens)
                and lowered[index + 1] in SIZE_ADJECTIVES
            ):
                mask[index] = True
            if lowered[index] == "per" and index + 1 < len(tokens) and _is_measurement_unit(tokens[index + 1]):
                mask[index] = True
            if lowered[index] in FRACTION_WORDS and index + 1 < len(tokens) and lowered[index + 1] == "off":
                mask[index] = True
            continue

        after = numeric_end + 1
        if after >= len(tokens):
            continue
        if _is_measurement_fused_token(token) and lowered[after] in SIZE_ADJECTIVES:
            mask[numeric_end] = True
        if _is_percentage_marker(tokens[after]):
            mask[numeric_end] = True
            if after + 1 < len(tokens) and lowered[after + 1] in PERCENT_NOUNS:
                mask[after] = True
        elif _is_measurement_unit(tokens[after]):
            mask[numeric_end] = True
            if (
                after + 2 < len(tokens)
                and lowered[after + 1] in RATE_CONNECTORS
                and _is_measurement_unit(tokens[after + 2])
            ):
                mask[after] = True
                mask[after + 1] = True
            if (
                after + 2 < len(tokens)
                and lowered[after + 1] in TEMPORAL_FREQUENCY_DETERMINERS
                and lowered[after + 2] in TEMPORAL_FREQUENCY_UNITS
            ):
                mask[after] = True
                mask[after + 1] = True
        elif lowered[after] in RATE_CONNECTORS:
            mask[numeric_end] = True
            if after + 1 < len(tokens) and _is_measurement_unit(tokens[after + 1]):
                mask[after] = True
        elif lowered[after] in {"apiece", "each", "off", "total"}:
            mask[numeric_end] = True
    return mask


def _span_has_quantity_or_symbolic_cue(tokens: list[str]) -> bool:
    lowered = _lower_tokens(tokens)
    for index, token in enumerate(tokens):
        if (
            token in CURRENCY_SYMBOLS
            or token in PERCENT_SYMBOLS
            or lowered[index] in QUOTED_FORMULA_CUES
            or lowered[index] in RATE_CONNECTORS
            or _is_measurement_unit(token)
            or _numeric_span_end(tokens, index) is not None
            or token.lower() in CODE_FUNCTION_KEYWORDS
        ):
            return True
    return False


def _mark_span(mask: list[bool], start: int, end: int) -> None:
    if start >= end:
        return
    for index in range(start, min(end, len(mask) - 1) + 1):
        mask[index] = True


def balanced_delimiter_span_mask(tokens: list[str]) -> list[bool]:
    """Mask boundaries inside formula-like quotation or bracket spans.

    EDU corpora often allow internal discourse boundaries inside ordinary
    quotes. This guard is therefore selective: it closes delimiter spans only
    when the span looks like a quantity/formula/code chunk.
    """

    mask = [False] * len(tokens)
    quote_start: int | None = None
    for index, token in enumerate(tokens):
        if token not in LEFT_QUOTES | RIGHT_QUOTES:
            continue
        if quote_start is None:
            quote_start = index
        else:
            if _span_has_quantity_or_symbolic_cue(tokens[quote_start + 1 : index]):
                _mark_span(mask, quote_start, index - 1)
            quote_start = None
    if quote_start is not None and _span_has_quantity_or_symbolic_cue(tokens[quote_start + 1 :]):
        _mark_span(mask, quote_start, len(tokens) - 1)

    stack: list[tuple[str, int]] = []
    for index, token in enumerate(tokens):
        if token in OPEN_BRACKETS:
            stack.append((OPEN_BRACKETS[token], index))
        elif token in CLOSE_BRACKETS and stack and stack[-1][0] == token:
            _, start = stack.pop()
            if _span_has_quantity_or_symbolic_cue(tokens[start + 1 : index]):
                _mark_span(mask, start, index - 1)
    for _, start in stack:
        if _span_has_quantity_or_symbolic_cue(tokens[start + 1 :]):
            _mark_span(mask, start, len(tokens) - 1)
    return mask


def _is_short_scientific_abbreviation(token: str) -> bool:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9/-]{0,7}", token):
        return False
    lowered = token.lower()
    if lowered in FUNCTION_WORDS or lowered in PREDICATE_HEADS:
        return False
    upper_count = sum(char.isupper() for char in token)
    has_digit = any(char.isdigit() for char in token)
    return upper_count >= 2 or (has_digit and upper_count >= 1)


def _has_letter_and_digit(token: str) -> bool:
    return any(char.isalpha() for char in token) and any(char.isdigit() for char in token)


def _is_mixed_case_abbreviation(token: str) -> bool:
    return (
        _is_short_scientific_abbreviation(token)
        and any(char.islower() for char in token)
        and any(char.isupper() for char in token)
    )


def _is_compact_percentage_parenthetical(tokens: list[str]) -> bool:
    content = [token for token in tokens if token not in set(OPEN_BRACKETS) | CLOSE_BRACKETS]
    if len(content) not in {2, 4} or not _is_percentage_marker(content[-1]):
        return False
    numeric_end = _numeric_span_end(content, 0)
    return numeric_end is not None and numeric_end == len(content) - 2


def _span_has_confidence_interval_phrase(tokens: list[str]) -> bool:
    lowered = _lower_tokens(tokens)
    return "ci" in lowered or ("confidence" in lowered and "interval" in lowered)


def _span_has_numeric_interval_bracket(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if token not in {"[", "("}:
            continue
        closer = {"[": "]", "(": ")"}[token]
        end = next(
            (position for position in range(index + 1, len(tokens)) if tokens[position] == closer),
            len(tokens),
        )
        inner = tokens[index + 1 : end]
        has_decimal_value = any(
            is_numeric_piece(inner[position])
            and position + 2 < len(inner)
            and inner[position + 1] == "."
            and is_numeric_piece(inner[position + 2])
            for position in range(len(inner))
        )
        has_interval_separator = "," in inner or "-" in inner
        if has_decimal_value and has_interval_separator:
            return True
    return False


def _span_has_numeric_statistical_estimate(tokens: list[str]) -> bool:
    content = [token for token in tokens if token not in set(OPEN_BRACKETS) | CLOSE_BRACKETS]
    if not content:
        return False
    first_numeric = _numeric_span_end(content, 0)
    starts_as_estimate = (
        first_numeric is not None
        and (is_numeric_piece(content[0]) or bool(re.fullmatch(r"\d+(?:[.,]\d+)?%?", content[0])))
    ) or (
        content[0] in {"±", "+/-", "?"}
        and len(content) > 1
        and _numeric_span_end(content, 1) is not None
    )
    if not starts_as_estimate:
        return False
    lowered = _lower_tokens(content)
    has_p_value = any(
        token == "p"
        and index + 1 < len(content)
        and content[index + 1] in STATISTICAL_OPERATORS
        for index, token in enumerate(lowered)
    )
    has_plus_minus = any(token in {"±", "+/-"} for token in content)
    has_numeric_interval = _span_has_numeric_interval_bracket(tokens)
    return (
        _span_has_confidence_interval_phrase(content)
        or has_p_value
        or has_plus_minus
        or has_numeric_interval
    )


def _span_has_statistical_expression(tokens: list[str]) -> bool:
    if (
        _is_compact_percentage_parenthetical(tokens)
        or _span_has_confidence_interval_phrase(tokens)
        or _span_has_numeric_statistical_estimate(tokens)
    ):
        return True
    lowered = _lower_tokens(tokens)
    for index, token in enumerate(tokens):
        if (
            lowered[index] in STATISTICAL_VARIABLES
            and index + 1 < len(tokens)
            and tokens[index + 1] in STATISTICAL_OPERATORS
        ):
            return True
        if token in STATISTICAL_OPERATORS and index + 1 < len(tokens):
            following = tokens[index + 1]
            if _numeric_span_end(tokens, index + 1) is not None:
                return True
            if following == "." and index + 2 < len(tokens) and _is_numeric_decimal_neighbor(tokens[index + 2]):
                return True
    return False


def _span_has_abbreviation_appositive(tokens: list[str], left_token: str | None = None) -> bool:
    content = [token for token in tokens if token not in set(OPEN_BRACKETS) | CLOSE_BRACKETS | {"/"}]
    if not content:
        return False
    left_lower = left_token.lower() if left_token else ""
    if any(_is_mixed_case_abbreviation(token) for token in content):
        return True
    if (
        left_lower in BIOMEDICAL_ABBREVIATION_HOST_CUES
        and any(_is_short_scientific_abbreviation(token) for token in content)
    ):
        return True
    return any(token in {"/", "[", "]"} for token in tokens) and any(
        _has_letter_and_digit(token) for token in content
    )


def _span_has_biomedical_parenthetical_cue(
    tokens: list[str],
    left_token: str | None = None,
) -> bool:
    return _span_has_statistical_expression(tokens) or _span_has_abbreviation_appositive(tokens, left_token)


def biomedical_parenthetical_span_mask(tokens: list[str]) -> list[bool]:
    """闭合医学统计括号、P 值和缩写同位语内部的候选边界。"""

    mask = [False] * len(tokens)
    stack: list[tuple[str, int]] = []
    for index, token in enumerate(tokens):
        if token in OPEN_BRACKETS:
            stack.append((OPEN_BRACKETS[token], index))
            continue
        if token not in CLOSE_BRACKETS or not stack or stack[-1][0] != token:
            continue
        _, start = stack.pop()
        span_tokens = tokens[start + 1 : index]
        left_token = tokens[start - 1] if start > 0 else None
        if not _span_has_biomedical_parenthetical_cue(span_tokens, left_token):
            continue
        if start > 0 and tokens[start - 1] not in (PUNCT_SET | CLAUSE_PUNCT_SET):
            mask[start - 1] = True
        _mark_span(mask, start, index - 1)
        if index + 1 < len(tokens):
            mask[index] = True

    for _, start in stack:
        span_tokens = tokens[start + 1 :]
        left_token = tokens[start - 1] if start > 0 else None
        if not _span_has_biomedical_parenthetical_cue(span_tokens, left_token):
            continue
        if start > 0 and tokens[start - 1] not in (PUNCT_SET | CLAUSE_PUNCT_SET):
            mask[start - 1] = True
        _mark_span(mask, start, len(tokens) - 1)

    lowered = _lower_tokens(tokens)
    for index in range(len(tokens) - 1):
        current = tokens[index]
        following = tokens[index + 1]
        if (
            lowered[index] in STATISTICAL_VARIABLES
            and following in STATISTICAL_OPERATORS
        ):
            mask[index] = True
        if (
            current in STATISTICAL_OPERATORS
            and (
                _numeric_span_end(tokens, index + 1) is not None
                or (
                    following == "."
                    and index + 2 < len(tokens)
                    and _is_numeric_decimal_neighbor(tokens[index + 2])
                )
            )
        ):
            mask[index] = True
        if (
            _is_short_scientific_abbreviation(current)
            and following.lower() in COORD_CUES
            and index + 2 < len(tokens)
            and _is_short_scientific_abbreviation(tokens[index + 2])
        ):
            mask[index] = True
            mask[index + 1] = True
        if current == "/" and index > 0 and index + 1 < len(tokens):
            if _is_short_scientific_abbreviation(tokens[index - 1].strip("[]")) or _is_short_scientific_abbreviation(following):
                mask[index] = True
    return mask


def biomedical_boundary_needs_more_context(tokens: list[str], index: int) -> bool:
    """对括号同位语先等待内部证据，避免把普通括号一律当成不可切块。"""

    if (
        index < 0
        or index + 1 >= len(tokens)
        or tokens[index + 1] not in OPEN_BRACKETS
        or tokens[index] in (PUNCT_SET | CLAUSE_PUNCT_SET)
    ):
        return False
    open_index = index + 1
    close_token = OPEN_BRACKETS[tokens[open_index]]
    span_tokens = tokens[open_index + 1 :]
    if _span_has_biomedical_parenthetical_cue(span_tokens, tokens[index]):
        return False
    depth = 0
    for position in range(open_index, len(tokens)):
        token = tokens[position]
        if token == tokens[open_index]:
            depth += 1
        elif token == close_token:
            depth -= 1
            if depth == 0:
                return False
    return len(tokens) - open_index <= 12


def _recent_complement_taking_verb(lowered: list[str], index: int, window: int = 6) -> bool:
    start = max(0, index - window + 1)
    return any(token in COMPLEMENT_TAKING_VERBS for token in lowered[start : index + 1])


def _could_end_subject(tokens: list[str], index: int) -> bool:
    token = tokens[index]
    lowered = token.lower()
    if token in PUNCT_SET or lowered in FUNCTION_WORDS:
        return False
    if index > 0 and tokens[index - 1] in CLAUSE_PUNCT_SET:
        return False
    return True


def _is_discourse_marker_orphan(tokens: list[str], index: int) -> bool:
    lowered = _lower_tokens(tokens[: index + 1])
    if not lowered:
        return False
    content = [token for token in lowered if token not in {",", ";", ":"}]
    return len(content) == 1 and content[0] in DISCOURSE_MARKER_PREFIXES


def _tokens_since_clause_punct(tokens: list[str], index: int) -> int:
    start = 0
    for position in range(index - 1, -1, -1):
        if tokens[position] in PUNCT_SET | CLAUSE_PUNCT_SET:
            start = position + 1
            break
    return index - start + 1


def _local_clause_tokens(tokens: list[str], index: int) -> list[str]:
    start = 0
    for position in range(index - 1, -1, -1):
        if tokens[position] in PUNCT_SET | CLAUSE_PUNCT_SET:
            start = position + 1
            break
    return tokens[start : index + 1]


def _is_fraction_like(token: str) -> bool:
    lowered = token.lower()
    if lowered in FRACTION_WORDS:
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", lowered):
        return True
    if "-" not in lowered:
        return False
    numerator, denominator = lowered.rsplit("-", 1)
    return numerator in MWE_QUANTITY_WORDS and denominator in FRACTION_DENOMINATORS


def _is_nominal_ngram_token(token: str) -> bool:
    lowered = token.lower().rstrip(".")
    if token in PUNCT_SET | CLAUSE_PUNCT_SET | OPEN_SET:
        return False
    if lowered in ESURF_NOMINAL_STOPWORDS or lowered in PREDICATE_HEADS:
        return False
    if lowered in PREVERBAL_ADVERBS or lowered in MWE_QUANTITY_MODIFIERS:
        return False
    if lowered in {"not", "n't", "normally", "right"}:
        return False
    if lowered in POSSESSIVE_DETERMINERS | DEICTIC_DETERMINERS:
        return False
    if lowered not in NOMINAL_HEADS and (lowered.endswith("ed") or lowered.endswith("ing")):
        return False
    if token in MWE_QUANTITY_SYMBOLS:
        return False
    return _is_nominal_head_candidate(token) or _is_quantity_start(token)


def _has_quantity_before(tokens: list[str], index: int, window: int = 3) -> bool:
    start = max(0, index - window)
    return any(_is_quantity_start(token) or _is_fraction_like(token) for token in tokens[start:index])


def _is_zero_relative_classifier_boundary(
    tokens: list[str],
    index: int,
    *,
    require_predicate: bool,
) -> bool:
    if not (0 <= index + 1 < len(tokens)):
        return False
    lowered = _lower_tokens(tokens)
    following = lowered[index + 1]
    if following not in SUBJECT_PRONOUNS or following in RELATIVE_CUES:
        return False
    if not _is_nominal_ngram_token(tokens[index]):
        return False
    local_tokens = _local_clause_tokens(tokens, index)
    if len(local_tokens) > 10 or not _is_probable_subject_prefix(local_tokens):
        return False
    local_lowered = _lower_tokens(local_tokens[:-1])
    if "of" not in local_lowered:
        return False
    if not any(token in ZERO_RELATIVE_CLASSIFIER_HEADS for token in local_lowered):
        return False
    if not require_predicate:
        return index + 2 >= len(tokens)
    return index + 2 < len(tokens) and _is_infinitive_predicate_like(tokens[index + 2])


def _is_quantified_object_tail(tokens: list[str], index: int) -> bool:
    if _is_quantity_start(tokens[index]):
        return True
    return _is_measurement_unit(tokens[index]) and _has_quantity_before(tokens, index)


def _recent_purpose_infinitive_anchor(tokens: list[str], index: int, window: int = 8) -> bool:
    lowered = _lower_tokens(tokens)
    start = max(0, index - window + 1)
    for position in range(index, start - 1, -1):
        if tokens[position] in {".", ";", ":", "!", "?"}:
            break
        if lowered[position] in PURPOSE_INFINITIVE_ANCHORS:
            return True
    return False


def _is_times_comparative_anchor(tokens: list[str], index: int) -> bool:
    if tokens[index].lower() not in {"time", "times"}:
        return False
    return index > 0 and (_is_quantity_start(tokens[index - 1]) or _is_fraction_like(tokens[index - 1]))


def _range_opener_position(lowered: list[str], index: int, opener: str) -> int | None:
    start = 0
    for position in range(index - 1, -1, -1):
        if lowered[position] in PUNCT_SET | CLAUSE_PUNCT_SET:
            start = position + 1
            break
    closer = RANGE_CLOSERS[opener]
    for position in range(index, start - 1, -1):
        if lowered[position] == closer:
            return None
        if lowered[position] == opener:
            return position
    return None


def _has_unclosed_range(lowered: list[str], index: int, opener: str) -> bool:
    return _range_opener_position(lowered, index, opener) is not None


def _is_restrictive_of_head(tokens: list[str], index: int) -> bool:
    lowered = tokens[index].lower().rstrip(".")
    if lowered in RESTRICTIVE_OF_HEADS:
        return True
    if _is_quantity_start(tokens[index]) or _is_fraction_like(tokens[index]):
        return True
    return _is_measurement_unit(tokens[index]) and _has_quantity_before(tokens, index)


def _has_recent_restrictive_of(tokens: list[str], index: int, window: int = 5) -> bool:
    lowered = _lower_tokens(tokens)
    for position in range(max(1, index - window), index):
        if lowered[position] == "of" and _is_restrictive_of_head(tokens, position - 1):
            return True
    return False


def _is_price_start_token(token: str) -> bool:
    return token in CURRENCY_SYMBOLS or _is_quantity_start(token)


def _is_coordinate_subject_start(tokens: list[str], index: int) -> bool:
    if index + 3 >= len(tokens):
        return False
    lowered = _lower_tokens(tokens)
    if lowered[index + 1] != "and":
        return False
    if not (_is_subject_candidate_token(tokens[index]) and _is_subject_candidate_token(tokens[index + 2])):
        return False
    predicate_index = index + 3
    if lowered[predicate_index] in PREVERBAL_ADVERBS and predicate_index + 1 < len(tokens):
        predicate_index += 1
    return (
        lowered[predicate_index] in PREDICATE_HEADS
        or lowered[predicate_index] in FINITE_AUXILIARIES
        or _is_lexical_predicate_candidate(tokens[predicate_index])
    )


def _recent_esurf_anchor(tokens: list[str], index: int, anchors: set[str], window: int = 10) -> bool:
    local = _local_clause_tokens(tokens, index)
    start = max(0, len(local) - window)
    return any(token in anchors for token in _lower_tokens(local[start:]))


def _recent_control_infinitive_anchor(tokens: list[str], index: int) -> bool:
    start = max(0, index - 5)
    for position in range(index - 1, start - 1, -1):
        if tokens[position] in PUNCT_SET | CLAUSE_PUNCT_SET:
            break
        if _is_verbal_control_infinitive(tokens, position):
            return True
    return False


def _is_verbal_control_infinitive(tokens: list[str], index: int) -> bool:
    lowered = _lower_tokens(tokens)
    current = lowered[index]
    previous = lowered[index - 1] if index > 0 else ""
    if current not in ESURF_CONTROL_INFINITIVE_VERBS:
        return False
    if previous in NOMINAL_INFINITIVE_LEFT_DETERMINERS:
        return False
    if current.endswith("s") or current.endswith("ed") or current in {"chose", "tried"}:
        return True
    local = _local_clause_tokens(tokens, index)
    return any(_is_subject_candidate_token(token) for token in local[:-1])


def _recent_object_list_anchor(tokens: list[str], index: int, window: int = 24) -> bool:
    local = _local_clause_tokens(tokens, index)
    start = max(0, len(local) - window)
    return any(token in ESURF_OBJECT_LIST_ANCHORS for token in _lower_tokens(local[start:]))


def _recent_object_list_anchor_across_commas(tokens: list[str], index: int, window: int = 24) -> bool:
    lowered = _lower_tokens(tokens)
    start = max(0, index - window + 1)
    for position in range(index, start - 1, -1):
        if tokens[position] in {".", ";", ":", "!", "?"}:
            break
        if lowered[position] in ESURF_OBJECT_LIST_ANCHORS:
            return True
    return False


def _is_object_list_item(token: str) -> bool:
    lowered = token.lower()
    if lowered in SUBJECT_PRONOUNS or lowered in FINITE_AUXILIARIES:
        return False
    return _is_nominal_ngram_token(token)


def _is_infinitive_predicate_like(token: str) -> bool:
    lowered = token.lower()
    if token in PUNCT_SET | CLAUSE_PUNCT_SET:
        return False
    if lowered in FUNCTION_WORDS or lowered in FINITE_AUXILIARIES:
        return False
    return _is_lexical_predicate_candidate(token) or bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", token))


def _has_left_comparative_as_many(tokens: list[str]) -> bool:
    lowered = _lower_tokens(tokens)
    for position in range(len(lowered) - 1):
        if lowered[position] == "as" and lowered[position + 1] in COMPARATIVE_SCALARS:
            return True
    return False


def _is_appositive_name(token: str) -> bool:
    lowered = token.lower()
    return lowered not in SUBJECT_PRONOUNS and bool(re.fullmatch(r"[A-Z][A-Za-z'-]{1,}", token))


def _is_appositive_prefix(tokens: list[str]) -> bool:
    content = [token for token in tokens if token not in PUNCT_SET | CLAUSE_PUNCT_SET]
    lowered = _lower_tokens(content)
    return (
        bool(content)
        and not _has_predicate_head(content)
        and any(token.endswith("'s") or token in KINSHIP_OR_ROLE_HEADS for token in lowered)
    )


def _has_predicate_head(tokens: list[str]) -> bool:
    lowered = _lower_tokens(tokens)
    return any(
        token in PREDICATE_HEADS
        or token in FINITE_AUXILIARIES
        or (token.endswith("ed") and token not in NOMINAL_HEADS)
        or token.endswith("ing")
        for token in lowered
    )


def _is_lexical_predicate_candidate(token: str) -> bool:
    lowered = token.lower()
    if lowered in NOMINAL_HEADS and lowered not in PREDICATE_HEADS:
        return False
    return (
        lowered in PREDICATE_HEADS
        or lowered.endswith("ed")
        or lowered.endswith("ing")
        or lowered.endswith("s")
    )


def esurf_ngram_closure_mask(tokens: list[str]) -> list[bool]:
    """ESURF-style local n-gram closure for low-latency streaming.

    ESURF represents each candidate boundary with small lexical/character
    n-grams around the boundary. This deterministic adaptation turns a small
    subset of high-precision n-gram configurations into boundary vetoes, so the
    neural scorer cannot emit inside tightly bound predicate-argument,
    coordination, comparative, and appositive spans.
    """

    mask = [False] * len(tokens)
    lowered = _lower_tokens(tokens)
    for index in range(len(tokens) - 1):
        current = lowered[index]
        following = lowered[index + 1]
        previous = lowered[index - 1] if index > 0 else ""
        local_tokens = _local_clause_tokens(tokens, index)

        if following == "to" and _is_verbal_control_infinitive(tokens, index):
            mask[index] = True
        if (
            current == "to"
            and _recent_control_infinitive_anchor(tokens, index)
            and _is_lexical_predicate_candidate(tokens[index + 1])
        ):
            mask[index] = True

        if current in {"a", "an", "the"} and following in ESURF_LIGHT_NOUN_OF_HEADS:
            mask[index] = True
        if current in ESURF_LIGHT_NOUN_OF_HEADS and following == "of":
            mask[index] = True
        if following == "of" and _is_restrictive_of_head(tokens, index):
            mask[index] = True
        if (
            current.endswith("ing")
            and following in {"at", "for", "in", "on", "with"}
            and _has_recent_restrictive_of(tokens, index)
        ):
            mask[index] = True
        if (
            following == "to"
            and _has_recent_restrictive_of(tokens, index)
            and index + 2 < len(tokens)
            and _is_infinitive_predicate_like(tokens[index + 2])
        ):
            mask[index] = True
        if (
            following == "at"
            and _is_object_list_item(tokens[index])
            and _recent_object_list_anchor_across_commas(tokens, index)
            and index + 2 < len(tokens)
            and _is_price_start_token(tokens[index + 2])
        ):
            mask[index] = True
        if (
            following == "for"
            and index + 2 < len(tokens)
            and lowered[index + 2] in OBJECT_PRONOUNS
            and _is_nominal_ngram_token(tokens[index])
            and _recent_object_list_anchor(tokens, index)
        ):
            mask[index] = True

        if (
            tokens[index] == ","
            and _recent_object_list_anchor_across_commas(tokens, index)
            and index > 0
            and _is_object_list_item(tokens[index - 1])
            and _is_object_list_item(tokens[index + 1])
        ):
            mask[index] = True
        if (
            tokens[index] == ","
            and following in {"and", "or"}
            and _recent_object_list_anchor_across_commas(tokens, index)
            and index + 2 < len(tokens)
            and _is_object_list_item(tokens[index + 2])
        ):
            mask[index] = True
        if (
            following in {"and", "or"}
            and _is_object_list_item(tokens[index])
            and (
                _recent_object_list_anchor(tokens, index)
                or (
                    "," in tokens[max(0, index - 8) : index]
                    and _recent_object_list_anchor_across_commas(tokens, index)
                )
            )
            and index + 2 < len(tokens)
            and _is_object_list_item(tokens[index + 2])
            and not _is_lexical_predicate_candidate(tokens[index + 2])
        ):
            mask[index] = True
        if (
            current in {"and", "or"}
            and _recent_object_list_anchor_across_commas(tokens, index)
            and "," in tokens[max(0, index - 8) : index]
            and _is_object_list_item(tokens[index + 1])
            and not _is_lexical_predicate_candidate(tokens[index + 1])
        ):
            mask[index] = True

        if _is_coordinate_subject_start(tokens, index):
            mask[index] = True
        if index > 0 and current == "and" and _is_coordinate_subject_start(tokens, index - 1):
            mask[index] = True
        if index >= 2 and lowered[index - 1] == "and" and _is_coordinate_subject_start(tokens, index - 2):
            mask[index] = True
        if index > 1 and lowered[index - 2] == "and" and _is_coordinate_subject_start(tokens, index - 2):
            mask[index] = True

        if _is_times_comparative_anchor(tokens, index) and following == "as":
            mask[index] = True
        if (
            current == "as"
            and index > 0
            and _is_times_comparative_anchor(tokens, index - 1)
            and following in COMPARATIVE_DEGREE_HEADS
        ):
            mask[index] = True
        if (
            current in COMPARATIVE_DEGREE_HEADS
            and previous == "as"
            and index + 1 < len(tokens)
            and lowered[index + 1] == "as"
        ):
            mask[index] = True
        if (
            current == "as"
            and index > 1
            and lowered[index - 1] in COMPARATIVE_DEGREE_HEADS
            and lowered[index - 2] == "as"
            and index + 1 < len(tokens)
            and tokens[index + 1] not in PUNCT_SET | CLAUSE_PUNCT_SET
        ):
            mask[index] = True
        if _is_fraction_like(tokens[index]) and following == "as":
            mask[index] = True
        if current == "as" and (
            _is_fraction_like(tokens[index - 1]) if index > 0 else False
        ):
            mask[index] = True
        if current == "as" and following in COMPARATIVE_SCALARS:
            mask[index] = True
        if (
            current in COMPARATIVE_SCALARS
            and previous == "as"
            and _is_nominal_ngram_token(tokens[index + 1])
        ):
            mask[index] = True
        if (
            following == "as"
            and _is_nominal_ngram_token(tokens[index])
            and _has_left_comparative_as_many(local_tokens)
        ):
            mask[index] = True
        if (
            current == "as"
            and index > 0
            and _is_nominal_ngram_token(tokens[index - 1])
            and _has_left_comparative_as_many(local_tokens)
        ):
            mask[index] = True

        if following == "to" and _has_unclosed_range(lowered, index, "from"):
            mask[index] = True
        if (
            current == "to"
            and _has_unclosed_range(lowered, index - 1, "from")
            and tokens[index + 1] not in PUNCT_SET | CLAUSE_PUNCT_SET
        ):
            mask[index] = True
        if following == "and" and _has_unclosed_range(lowered, index, "between"):
            mask[index] = True
        if (
            current == "and"
            and _has_unclosed_range(lowered, index - 1, "between")
            and tokens[index + 1] not in PUNCT_SET | CLAUSE_PUNCT_SET
        ):
            mask[index] = True

        if (
            tokens[index] == ","
            and _is_appositive_name(tokens[index + 1])
            and _is_appositive_prefix(_local_clause_tokens(tokens, index - 1))
        ):
            mask[index] = True
        if (
            tokens[index + 1] == ","
            and _is_appositive_name(tokens[index])
            and _is_appositive_prefix(local_tokens)
        ):
            mask[index] = True
    return mask


def esurf_boundary_needs_more_context(tokens: list[str], index: int) -> bool:
    """Delay one token for ESURF-style ambiguous coordination n-grams."""

    if not (0 <= index + 1 < len(tokens)):
        return False
    lowered = _lower_tokens(tokens)
    return (
        (
            lowered[index + 1] in {"and", "or"}
            and _is_object_list_item(tokens[index])
            and (
                _recent_object_list_anchor(tokens, index)
                or (
                    "," in tokens[max(0, index - 8) : index]
                    and _recent_object_list_anchor_across_commas(tokens, index)
                )
            )
            and index + 2 >= len(tokens)
        )
        or (
            lowered[index + 1] == "at"
            and _is_object_list_item(tokens[index])
            and _recent_object_list_anchor_across_commas(tokens, index)
            and index + 2 >= len(tokens)
        )
        or (
            lowered[index + 1] == "to"
            and _has_recent_restrictive_of(tokens, index)
            and index + 2 >= len(tokens)
        )
        or (
            lowered[index + 1] == "to"
            and _is_quantified_object_tail(tokens, index)
            and _recent_purpose_infinitive_anchor(tokens, index)
            and index + 2 >= len(tokens)
        )
        or (
            lowered[index + 1] == "and"
            and _is_direct_subject_candidate_token(tokens[index])
            and (
                index + 2 >= len(tokens)
                or (
                    _is_direct_subject_candidate_token(tokens[index + 2])
                    and index + 3 >= len(tokens)
                )
            )
        )
        or (
            lowered[index + 1] in PREVERBAL_ADVERBS
            and _is_direct_subject_candidate_token(tokens[index])
            and index + 2 >= len(tokens)
        )
    )


def _is_probable_subject_prefix(tokens: list[str]) -> bool:
    content = [
        token
        for token in _lower_tokens(tokens)
        if token not in FUNCTION_WORDS and token not in COORD_CUES and token not in {",", ";", ":"}
    ]
    return bool(content) and not _has_predicate_head(tokens)


def _subordinate_subject_tail(tokens: list[str]) -> list[str]:
    lowered = _lower_tokens(tokens)
    for position in range(len(lowered) - 1, -1, -1):
        if lowered[position] in SUBORD_CUES:
            return tokens[position + 1 :]
    return []


def _embedded_clause_subject_tail(tokens: list[str]) -> list[str]:
    lowered = _lower_tokens(tokens)
    for position in range(len(lowered) - 1, -1, -1):
        if lowered[position] in COMPLEMENTIZERS | RELATIVE_CUES:
            return tokens[position + 1 :]
    return []


def predicate_ngram_closure_mask(tokens: list[str]) -> list[bool]:
    """Mask high-confidence predicate-bearing fragment splits.

    This is a narrow n-gram/shallow-syntax layer, not a full parser. It closes
    modal-verb bigrams, number-subject predicates, duration-infinitive
    constructions, and local subject-copula splits while leaving broader
    relative/complement clause decisions to the learned EDU model.
    """

    mask = [False] * len(tokens)
    lowered = _lower_tokens(tokens)
    for index in range(len(tokens) - 1):
        current = lowered[index]
        following = lowered[index + 1]
        local_tokens = _local_clause_tokens(tokens, index)
        local_len = len(local_tokens)
        if (
            current in MODAL_AUXILIARIES
            and tokens[index + 1] not in PUNCT_SET | CLAUSE_PUNCT_SET
            and (following in PREDICATE_HEADS or _is_lexical_predicate_candidate(following))
        ):
            mask[index] = True
        if (
            current in FINITE_AUXILIARIES
            and index + 2 < len(tokens)
            and _is_subject_candidate_token(tokens[index + 1])
            and _is_lexical_predicate_candidate(tokens[index + 2])
        ):
            mask[index] = True
        if (
            index > 0
            and lowered[index - 1] in FINITE_AUXILIARIES
            and _is_subject_candidate_token(tokens[index])
            and (following in PREDICATE_HEADS or _is_lexical_predicate_candidate(following))
        ):
            mask[index] = True
        if (
            local_len <= 8
            and _is_direct_subject_candidate_token(tokens[index])
            and _is_probable_subject_prefix(local_tokens)
            and (
                following in PREDICATE_HEADS
                or following in FINITE_AUXILIARIES
                or _is_lexical_predicate_candidate(tokens[index + 1])
            )
        ):
            mask[index] = True
        if (
            local_len <= 8
            and _is_direct_subject_candidate_token(tokens[index])
            and _is_probable_subject_prefix(local_tokens)
            and following in PREVERBAL_ADVERBS
            and index + 2 < len(tokens)
            and (
                lowered[index + 2] in PREDICATE_HEADS
                or lowered[index + 2] in FINITE_AUXILIARIES
                or _is_lexical_predicate_candidate(tokens[index + 2])
            )
        ):
            mask[index] = True
        if _is_zero_relative_classifier_boundary(tokens, index, require_predicate=True):
            mask[index] = True
        if (
            current in FREE_RELATIVE_PRONOUNS
            and following in SUBJECT_PRONOUNS
            and index + 2 < len(tokens)
            and _is_infinitive_predicate_like(tokens[index + 2])
        ):
            mask[index] = True
        if (
            current in MATRIX_COMPLEMENT_CLOSURE_VERBS
            and following in COMPLEMENTIZERS
            and local_len <= 5
            and any(_is_subject_candidate_token(token) for token in local_tokens[:-1])
        ):
            mask[index] = True
        if (
            (is_numeric_piece(tokens[index]) or _is_number_word(tokens[index]))
            and (following in PREDICATE_HEADS or _is_lexical_predicate_candidate(following))
        ):
            mask[index] = True
        if (
            following in {"am", "are", "is", "was", "were"}
            and local_len <= 8
            and _is_probable_subject_prefix(local_tokens)
        ):
            mask[index] = True
        subordinate_tail = _subordinate_subject_tail(local_tokens)
        if (
            following in FINITE_AUXILIARIES
            and subordinate_tail
            and len(subordinate_tail) <= 8
            and _is_probable_subject_prefix(subordinate_tail)
        ):
            mask[index] = True
        embedded_tail = _embedded_clause_subject_tail(local_tokens)
        if (
            following in FINITE_AUXILIARIES
            and embedded_tail
            and len(embedded_tail) <= 8
            and _is_probable_subject_prefix(embedded_tail)
        ):
            mask[index] = True
        if (
            following == "to"
            and current in MEASUREMENT_UNITS
            and any(token in {"take", "takes", "took", "taken"} for token in lowered[max(0, index - 6) : index + 1])
        ):
            mask[index] = True
        if (
            following == "to"
            and index + 2 < len(tokens)
            and _is_quantified_object_tail(tokens, index)
            and _recent_purpose_infinitive_anchor(tokens, index)
            and _is_infinitive_predicate_like(tokens[index + 2])
        ):
            mask[index] = True
        if (
            current.endswith("ed")
            and following in {"at", "by", "for", "from", "in", "on", "with"}
            and any(
                token in {"am", "are", "be", "been", "being", "is", "was", "were", "going"}
                for token in lowered[max(0, index - 5) : index]
            )
        ):
            mask[index] = True
    return mask


def syntactic_closure_mask(tokens: list[str]) -> list[bool]:
    """Mask boundaries that break shallow predicate-argument closure.

    The constraints are local and class-based: they block subject-copula
    splits, complement-taking verb splits, restrictive modifier splits, and
    orphaned discourse markers without adding sample-specific rules.
    """

    mask = [False] * len(tokens)
    lowered = _lower_tokens(tokens)
    for index in range(len(tokens) - 1):
        current = lowered[index]
        following = lowered[index + 1]
        local_len = _tokens_since_clause_punct(tokens, index)
        if tokens[index] in CLAUSE_PUNCT_SET and _is_discourse_marker_orphan(tokens, index):
            mask[index] = True
        if (
            local_len <= 8
            and current in COMPLEMENT_TAKING_VERBS
            and following in {"if", "that", "to", "whether"}
        ):
            mask[index] = True
        if (
            local_len <= 10
            and following == "to"
            and tokens[index] not in PUNCT_SET
            and _recent_complement_taking_verb(lowered, index)
        ):
            mask[index] = True
        if (
            local_len <= 6
            and following in RELATIVE_CUES
            and tokens[index] not in PUNCT_SET
            and (index == 0 or tokens[index - 1] != ",")
        ):
            mask[index] = True
        if local_len <= 4 and following in FINITE_AUXILIARIES and _could_end_subject(tokens, index):
            mask[index] = True
        if (
            local_len <= 6
            and index + 2 < len(tokens)
            and following in SUBJECT_PRONOUNS
            and lowered[index + 2] in COMMON_FINITE_VERBS
            and tokens[index] not in PUNCT_SET
        ):
            mask[index] = True
        if (
            local_len <= 5
            and current in TRANSITIVE_COMPLETION_VERBS
            and following in (MWE_QUANTITY_MODIFIERS | MWE_QUANTITY_WORDS | {"the", "a", "an"})
        ):
            mask[index] = True
        if (
            local_len <= 4
            and current.endswith("ed")
            and following in PREPOSITIONAL_COMPLEMENT_CUES
            and tokens[index] not in PUNCT_SET
        ):
            mask[index] = True
    return mask


def _is_latex_command_at(tokens: list[str], index: int) -> bool:
    return (
        index + 1 < len(tokens)
        and tokens[index] == "\\"
        and bool(re.fullmatch(r"[A-Za-z]+|.", tokens[index + 1]))
    )


def _is_latex_math_command_at(tokens: list[str], index: int) -> bool:
    return (
        index + 1 < len(tokens)
        and tokens[index] == "\\"
        and tokens[index + 1].lower() in LATEX_MATH_COMMANDS
    )


def _latex_begin_environment(tokens: list[str], index: int) -> str | None:
    if not (
        index + 4 < len(tokens)
        and tokens[index] == "\\"
        and tokens[index + 1].lower() == "begin"
        and tokens[index + 2] == "{"
        and tokens[index + 4] == "}"
    ):
        return None
    environment = tokens[index + 3].lower()
    return environment if environment in LATEX_MATH_ENVIRONMENTS else None


def _latex_end_environment_at(tokens: list[str], index: int, environment: str) -> bool:
    return (
        index + 4 < len(tokens)
        and tokens[index] == "\\"
        and tokens[index + 1].lower() == "end"
        and tokens[index + 2] == "{"
        and tokens[index + 3].lower() == environment
        and tokens[index + 4] == "}"
    )


def _is_currency_dollar_at(tokens: list[str], index: int) -> bool:
    return (
        tokens[index] == "$"
        and index + 1 < len(tokens)
        and (_is_quantity_start(tokens[index + 1]) or is_numeric_piece(tokens[index + 1]))
    )


def _math_delimiter_at(tokens: list[str], index: int) -> tuple[int, tuple[str, ...] | None] | None:
    if tokens[index] == "$" and not _is_currency_dollar_at(tokens, index):
        return index, ("$",)
    if index + 1 < len(tokens) and tokens[index] == "\\":
        if tokens[index + 1] == "[":
            return index + 1, ("\\", "]")
        if tokens[index + 1] == "(":
            return index + 1, ("\\", ")")
    environment = _latex_begin_environment(tokens, index)
    if environment is not None:
        return index + 4, ("\\end", environment)
    return None


def _find_math_closer(tokens: list[str], start: int, closer: tuple[str, ...] | None) -> int | None:
    if closer is None:
        return None
    if closer == ("$",):
        for index in range(start + 1, len(tokens)):
            if tokens[index] == "$":
                return index
        return None
    if closer in {("\\", "]"), ("\\", ")")}:
        for index in range(start + 1, len(tokens) - 1):
            if tokens[index] == closer[0] and tokens[index + 1] == closer[1]:
                return index + 1
        return None
    if len(closer) == 2 and closer[0] == "\\end":
        for index in range(start + 1, len(tokens) - 4):
            if _latex_end_environment_at(tokens, index, closer[1]):
                return index + 4
    return None


def _mark_math_internal(mask: list[bool], start: int, end: int) -> None:
    for index in range(start, min(end, len(mask) - 1) + 1):
        mask[index] = True


def _math_attachment_left(tokens: list[str], start: int) -> bool:
    if start <= 0:
        return False
    previous = tokens[start - 1].lower()
    return (
        previous in FUNCTION_WORDS
        or previous in COMPLEMENTIZERS
        or previous in PREPOSITIONAL_COMPLEMENT_CUES
        or previous in {"let", "such", "than", "equals", "equal", "evaluate", "evaluating"}
    )


def _math_attachment_right(tokens: list[str], end: int) -> bool:
    if end + 1 >= len(tokens):
        return False
    following = tokens[end + 1].lower()
    return (
        following in FINITE_AUXILIARIES
        or following in PREDICATE_HEADS
        or following in SUBORD_CUES
        or following in {"after", "be", "equals", "is"}
    )


def _mark_latex_command(mask: list[bool], tokens: list[str], index: int) -> None:
    if not _is_latex_command_at(tokens, index):
        return
    mask[index] = True
    command = tokens[index + 1].lower()
    if command in LATEX_MATH_COMMANDS:
        mask[index + 1] = True
    if index + 2 < len(tokens) and tokens[index + 2] in {"{", "(", "["}:
        opener = tokens[index + 2]
        closer = {"{": "}", "(": ")", "[": "]"}[opener]
        depth = 0
        for position in range(index + 2, len(tokens)):
            if tokens[position] == opener:
                depth += 1
            elif tokens[position] == closer:
                depth -= 1
                if depth == 0:
                    _mark_math_internal(mask, index, position - 1)
                    return


def _has_mathish_unclosed_span(tokens: list[str], start: int) -> bool:
    return any(
        token in MATH_SYMBOL_TOKENS
        or is_numeric_piece(token)
        or _is_latex_math_command_at(tokens, index)
        for index, token in enumerate(tokens[start:], start)
    )


def _previous_clause_break(tokens: list[str], index: int) -> int:
    for position in range(index - 1, -1, -1):
        if tokens[position] in CLAUSE_PUNCT_SET or tokens[position] in {".", "!", "?"}:
            return position
    return -1


def _has_recent_formula_intro(tokens: list[str], start: int, window: int = 8) -> bool:
    lowered = _lower_tokens(tokens)
    begin = max(0, start - window)
    return any(token in INLINE_FORMULA_INTRO_CUES for token in lowered[begin:start])


def _is_unit_parenthetical(tokens: list[str], index: int) -> bool:
    return (
        index + 2 < len(tokens)
        and tokens[index] == "("
        and _is_measurement_unit(tokens[index + 1])
        and tokens[index + 2] == ")"
    )


def _inline_formula_has_evidence(tokens: list[str], start: int, end: int) -> bool:
    span = tokens[start : end + 1]
    if "=" not in span:
        return False
    if _has_recent_formula_intro(tokens, start):
        return True
    return any(token in INLINE_FORMULA_OPERATORS for token in span) or any(
        _is_unit_parenthetical(tokens, position)
        for position in range(start, end + 1)
    )


def _inline_formula_end(tokens: list[str], equals_index: int) -> int:
    for position in range(equals_index + 1, len(tokens)):
        if tokens[position] == "?" and _numeric_span_end(tokens, position + 1) is not None:
            continue
        if (
            tokens[position] in {".", "!", "?"}
            and not is_decimal_point(tokens, position)
            and not is_nonterminal_period(tokens, position)
        ):
            return position
    return len(tokens) - 1


def _plain_inline_equation_span_mask(tokens: list[str]) -> list[bool]:
    """闭合普通文本里的赋值型公式，如 ``x = a / b × 100``。"""

    mask = [False] * len(tokens)
    for equals_index, token in enumerate(tokens):
        if token != "=":
            continue
        start = _previous_clause_break(tokens, equals_index) + 1
        if equals_index - start > 6:
            continue
        end = _inline_formula_end(tokens, equals_index)
        if not _inline_formula_has_evidence(tokens, start, end):
            continue
        formula_end = end - 1 if end < len(tokens) and tokens[end] in {".", "!", "?"} else end
        _mark_math_internal(mask, start, formula_end)
    return mask


def _inline_formula_prefix_needs_more_context(tokens: list[str], index: int) -> bool:
    if not (0 <= index + 1 < len(tokens)):
        return False
    start = _previous_clause_break(tokens, index) + 1
    if start > index or not _has_recent_formula_intro(tokens, start):
        return False
    prefix = tokens[start : index + 2]
    if "=" in prefix or len(prefix) > 6:
        return False
    return all(
        re.fullmatch(r"[A-Za-z][A-Za-z-]*", token)
        or token in {"(", ")"}
        or _is_measurement_unit(token)
        for token in prefix
    )


def math_span_mask(tokens: list[str]) -> list[bool]:
    """Mask EDU boundaries inside LaTeX/math expressions.

    The detector treats TeX math delimiters and math commands as structural
    spans. It closes boundaries inside the formula and, for complement-taking
    contexts such as "such that \\[...\\]" or "of $...$", also closes the
    boundary immediately before the formula.
    """

    mask = [False] * len(tokens)
    index = 0
    while index < len(tokens):
        delimiter = _math_delimiter_at(tokens, index)
        if delimiter is not None:
            opener_end, closer = delimiter
            close_index = _find_math_closer(tokens, opener_end, closer)
            end = close_index if close_index is not None else len(tokens) - 1
            if close_index is not None or _has_mathish_unclosed_span(tokens, opener_end + 1):
                if _math_attachment_left(tokens, index):
                    mask[index - 1] = True
                _mark_math_internal(mask, index, end - (0 if close_index is None else 1))
                if close_index is not None and _math_attachment_right(tokens, close_index):
                    mask[close_index] = True
                index = end + 1
                continue
        if _is_latex_math_command_at(tokens, index):
            _mark_latex_command(mask, tokens, index)
            if _math_attachment_left(tokens, index):
                mask[index - 1] = True
            index += 2
            continue
        index += 1
    equation_mask = _plain_inline_equation_span_mask(tokens)
    for index, is_equation_internal in enumerate(equation_mask):
        if is_equation_internal:
            mask[index] = True
    return mask


def math_boundary_needs_more_context(tokens: list[str], index: int) -> bool:
    """在 TeX 或普通文本公式前缀仍不明确时延迟边界。"""

    if not (0 <= index + 1 < len(tokens)):
        return False
    if tokens[index + 1] == "\\" and index + 2 >= len(tokens):
        return True
    if tokens[index + 1] == "$" and math_span_mask(tokens)[index]:
        return True
    if _inline_formula_prefix_needs_more_context(tokens, index):
        return True
    return False


def _is_code_close_marker(tokens: list[str], index: int) -> bool:
    return (
        index + 3 < len(tokens)
        and tokens[index] == "["
        and tokens[index + 1] == "/"
        and tokens[index + 2].lower() in CODE_BLOCK_LANGS
        and tokens[index + 3] == "]"
    )


def _has_coordinate_tuple(tokens: list[str]) -> bool:
    for index in range(len(tokens) - 4):
        if (
            tokens[index] == "("
            and is_numeric_piece(tokens[index + 1])
            and tokens[index + 2] == ","
            and is_numeric_piece(tokens[index + 3])
            and tokens[index + 4] == ")"
        ):
            return True
    return False


def _has_coordinate_fragment(tokens: list[str]) -> bool:
    return any(
        is_numeric_piece(tokens[index])
        and tokens[index + 1] == ","
        and is_numeric_piece(tokens[index + 2])
        for index in range(len(tokens) - 2)
    )


def _has_path_connector(tokens: list[str]) -> bool:
    return any(
        token == "--"
        or (token == "-" and index + 1 < len(tokens) and tokens[index + 1] == "-")
        for index, token in enumerate(tokens)
    )


def _has_coordinate_prefix(tokens: list[str]) -> bool:
    return any(
        token == "-" and index + 1 < len(tokens) and tokens[index + 1] == "("
        for index, token in enumerate(tokens)
    )


def _has_code_marker(tokens: list[str]) -> bool:
    bracketed = {f"[{language}]" for language in CODE_BLOCK_LANGS} | {
        f"[/{language}]" for language in CODE_BLOCK_LANGS
    }
    return any(token.lower() in bracketed | CODE_BLOCK_LANGS for token in tokens) or any(
        _is_code_close_marker(tokens, index) for index in range(len(tokens))
    )


def _code_function_start(tokens: list[str], index: int) -> bool:
    if (
        tokens[index].lower() not in CODE_FUNCTION_KEYWORDS
        or index + 1 >= len(tokens)
        or tokens[index + 1] != "("
    ):
        return False
    depth = 0
    close_index = None
    for position in range(index + 1, min(len(tokens), index + 24)):
        if tokens[position] == "(":
            depth += 1
        elif tokens[position] == ")":
            depth -= 1
            if depth == 0:
                close_index = position
                break
    if close_index is None:
        return True
    argument_tokens = tokens[index + 2 : close_index]
    if len(argument_tokens) == 1 and _is_measurement_unit(argument_tokens[0]):
        return False
    return any(
        token in CODE_PUNCT
        or is_numeric_piece(token)
        or token.lower() in CODE_FUNCTION_KEYWORDS
        or token in {"--", "\"", "'"}
        for token in argument_tokens
    )


def code_span_mask(tokens: list[str]) -> list[bool]:
    """Return token positions that should be treated as code/symbolic text.

    The mask is intentionally conservative: natural-language punctuation is not
    suppressed unless the local buffer contains code-block markers, function-call
    syntax, or coordinate/path syntax. Asymptote is one member of this symbolic
    domain, but the detector is based on domain cues rather than one example.
    """

    mask = [False] * len(tokens)
    if not tokens:
        return mask

    has_marker = _has_code_marker(tokens)
    has_function = any(_code_function_start(tokens, index) for index in range(len(tokens)))
    has_coordinate = _has_coordinate_tuple(tokens) or _has_coordinate_fragment(tokens)
    has_coordinate_prefix = _has_coordinate_prefix(tokens)
    has_coordinate_bracket = "(" in tokens or ")" in tokens
    codeish_token_count = sum(
        token in CODE_PUNCT
        or token == "--"
        or is_numeric_piece(token)
        or token.lower() in CODE_FUNCTION_KEYWORDS
        for token in tokens
    )
    coordinate_only = (
        (has_coordinate and has_coordinate_bracket) or has_coordinate_prefix
    ) and codeish_token_count >= max(1, len(tokens) - 1)
    has_coordinate_path = (has_coordinate or has_coordinate_prefix) and (
        _has_path_connector(tokens)
        or any(token.lower() in CODE_FUNCTION_KEYWORDS for token in tokens)
        or ";" in tokens
        or coordinate_only
    )
    suffix_coordinate_path = (
        _has_path_connector(tokens)
        and has_coordinate
        and codeish_token_count >= max(3, len(tokens) // 2)
    )
    if not (has_marker or has_function or has_coordinate_path or suffix_coordinate_path):
        return mask

    if has_marker:
        return [True] * len(tokens)

    index = 0
    while index < len(tokens):
        if _code_function_start(tokens, index):
            depth = 0
            end = index
            for end in range(index, len(tokens)):
                if tokens[end] in "([{":
                    depth += 1
                elif tokens[end] in ")]}":
                    depth = max(0, depth - 1)
                if tokens[end] == ";" and depth == 0:
                    break
            for pos in range(index, min(end + 1, len(tokens))):
                mask[pos] = True
            index = end + 1
            continue
        index += 1

    if has_coordinate_path or suffix_coordinate_path:
        for index, token in enumerate(tokens):
            if (
                token in CODE_PUNCT
                or is_numeric_piece(token)
                or token.lower() in CODE_FUNCTION_KEYWORDS
                or token in {"--"}
            ):
                mask[index] = True
        first = next((i for i, value in enumerate(mask) if value), None)
        last = len(mask) - 1 - next((i for i, value in enumerate(reversed(mask)) if value), None)
        if first is not None:
            for pos in range(first, last + 1):
                mask[pos] = True
    return mask

def is_nonterminal_period(tokens: list[str], index: int) -> bool:
    if tokens[index] != "." or index == 0 or index + 1 >= len(tokens):
        return False
    if is_decimal_point(tokens, index):
        return False
    previous = tokens[index - 1]
    following = tokens[index + 1]
    previous_lower = previous.lower()
    following_lower = following.lower()

    if following in {",", ";"}:
        return True
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
    if previous_lower in GENERAL_ABBREVIATIONS and (
        not following[:1].isupper() or following in ",;:)"
    ):
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
    previous = tokens[index - 1] if index > 0 else ""
    previous_lower = previous.lower()
    if previous_lower in TITLE_ABBREVIATIONS | NUMBER_ABBREVIATIONS | MONTH_ABBREVIATIONS:
        return False
    if previous_lower in GENERAL_ABBREVIATIONS and previous_lower != "etc":
        return False
    if previous in CLOSE_BRACKETS and following[:1].isupper():
        return True
    if (
        previous == "-"
        and index >= 2
        and following[:1].isupper()
        and re.fullmatch(r"[A-Za-z0-9]+", tokens[index - 2])
    ):
        return True
    if (
        following[:1].isupper()
        and len(previous) > 1
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", previous)
    ):
        return True
    if previous in PERCENT_SYMBOLS and following[:1].isupper():
        return True
    return len(following) > 1 and following.isupper()


def shape_id(
    token: str,
    decimal_point: bool = False,
    nonterminal_period: bool = False,
) -> int:
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


def cue_id(token: str) -> int:
    token = normalize_token(token)
    if token in COORD_CUES:
        return 1
    if token in SUBORD_CUES:
        return 2
    if token in ADVERBIAL_CUES:
        return 3
    if token in RELATIVE_CUES:
        return 4
    if token in REPORTING_CUES:
        return 5
    return 0


def stable_hash(text: str, modulo: int) -> int:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value % modulo


def char_ngram_ids(token: str, buckets: int, slots: int) -> list[int]:
    text = f"<{normalize_token(token)}>"
    grams: list[str] = []
    for size in (2, 3, 4):
        grams.extend(text[i : i + size] for i in range(max(0, len(text) - size + 1)))
    ids = [stable_hash(gram, buckets) + 1 for gram in grams[:slots]]
    return ids + [0] * (slots - len(ids))


def stream_feature_ids(tokens: list[str]) -> list[list[int]]:
    features: list[list[int]] = []
    since_punct = 16
    depth = 0
    for index, token in enumerate(tokens):
        features.append([min(index, 31), min(since_punct, 16), min(depth, 8)])
        if token in "([{":
            depth += 1
        elif token in ")]}":
            depth = max(0, depth - 1)
        since_punct = 0 if token in CLAUSE_PUNCT_SET else since_punct + 1
    return features


@dataclass(frozen=True)
class EncodedTokens:
    token_ids: torch.Tensor
    shape_ids: torch.Tensor
    cue_ids: torch.Tensor
    stream_ids: torch.Tensor
    char_ids: torch.Tensor
    padding_mask: torch.Tensor


def encode_tokens(
    tokens: list[str],
    vocab: dict[str, int],
    char_buckets: int,
    char_slots: int,
    device: torch.device,
) -> EncodedTokens:
    if not tokens:
        raise ValueError("tokens must not be empty")
    unk = vocab[UNK]
    token_ids = []
    shape_ids = []
    for i, token in enumerate(tokens):
        decimal_point = is_decimal_point(tokens, i)
        nonterminal_period = is_nonterminal_period(tokens, i)
        if decimal_point:
            normalized = DECIMAL_POINT
        elif nonterminal_period:
            normalized = NONTERMINAL_PERIOD
        else:
            normalized = normalize_token(token)
        token_ids.append(vocab.get(normalized, vocab.get(normalize_token(token), unk)))
        shape_ids.append(shape_id(token, decimal_point, nonterminal_period))
    return EncodedTokens(
        token_ids=torch.tensor([token_ids], dtype=torch.long, device=device),
        shape_ids=torch.tensor([shape_ids], dtype=torch.long, device=device),
        cue_ids=torch.tensor([[cue_id(token) for token in tokens]], dtype=torch.long, device=device),
        stream_ids=torch.tensor([stream_feature_ids(tokens)], dtype=torch.long, device=device),
        char_ids=torch.tensor(
            [[char_ngram_ids(token, char_buckets, char_slots) for token in tokens]],
            dtype=torch.long,
            device=device,
        ),
        padding_mask=torch.ones((1, len(tokens)), dtype=torch.bool, device=device),
    )
