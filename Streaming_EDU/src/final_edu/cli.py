from __future__ import annotations

import argparse
import json
import sys
import time

from .segmenter import EDUSegmenter, simple_tokenize


def emit_json(data: object) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


def emit_json_line(data: object) -> None:
    text = json.dumps(data, ensure_ascii=False)
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency streaming EDU segmentation")
    parser.add_argument("--text", help="English text to segment; reads stdin when omitted")
    parser.add_argument("--tokens-json", help='JSON token array, for example: ["I", "agree", "."]')
    parser.add_argument("--token", dest="token_items", action="append", help="Add one pre-tokenized token")
    parser.add_argument("--checkpoint", help="Optional alternative checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--threshold", type=float, help="Override the bundled development threshold")
    parser.add_argument(
        "--no-decimal-guard",
        action="store_true",
        help="Disable the deterministic numeric-context decimal guard",
    )
    parser.add_argument(
        "--no-period-guard",
        action="store_true",
        help="Disable the deterministic abbreviation/domain/ellipsis period guard",
    )
    parser.add_argument(
        "--no-numeric-colon-guard",
        action="store_true",
        help="Disable the deterministic digit-colon-digit guard for ratios, times, and scores",
    )
    parser.add_argument(
        "--no-phrase-guard",
        action="store_true",
        help="Disable the lexicalized MWE boundary-closure guard",
    )
    parser.add_argument(
        "--no-measurement-guard",
        action="store_true",
        help="Disable the quantity/currency/unit/rate expression guard",
    )
    parser.add_argument(
        "--no-predicate-guard",
        action="store_true",
        help="Disable the narrow predicate-bearing n-gram closure guard",
    )
    parser.add_argument(
        "--no-esurf-guard",
        action="store_true",
        help="Disable the ESURF-style local n-gram closure guard",
    )
    parser.add_argument(
        "--delimiter-guard",
        action="store_true",
        help="Enable the formula-like quotation/bracket span guard for reasoning streams",
    )
    parser.add_argument(
        "--syntax-guard",
        action="store_true",
        help="Enable the shallow predicate-argument closure guard for reasoning streams",
    )
    parser.add_argument(
        "--no-code-guard",
        action="store_true",
        help="Disable the symbolic/code-domain span guard",
    )
    parser.add_argument(
        "--no-math-guard",
        action="store_true",
        help="Disable the LaTeX/math-expression span guard",
    )
    parser.add_argument("--stream", action="store_true", help="Feed tokens incrementally")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="Optional delay between streamed tokens")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark model inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_text = None
    if args.token_items:
        tokens = args.token_items
    elif args.tokens_json:
        try:
            tokens = json.loads(args.tokens_json)
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"invalid --tokens-json: {error}. On PowerShell, repeated --token arguments are more reliable."
            ) from error
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise SystemExit("--tokens-json must be a JSON array of strings")
    else:
        source_text = args.text if args.text is not None else sys.stdin.read()
        tokens = simple_tokenize(source_text)

    if not tokens:
        raise SystemExit("no input tokens")

    segmenter = EDUSegmenter(
        args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        decimal_guard=not args.no_decimal_guard,
        period_guard=not args.no_period_guard,
        numeric_colon_guard=not args.no_numeric_colon_guard,
        phrase_guard=not args.no_phrase_guard,
        measurement_guard=not args.no_measurement_guard,
        predicate_guard=not args.no_predicate_guard,
        esurf_guard=not args.no_esurf_guard,
        delimiter_guard=args.delimiter_guard,
        syntax_guard=args.syntax_guard,
        code_guard=not args.no_code_guard,
        math_guard=not args.no_math_guard,
    )
    if args.benchmark:
        emit_json(segmenter.benchmark(tokens))
        return

    if not args.stream:
        spans = (
            segmenter.segment_text(source_text)
            if source_text is not None
            else segmenter.segment_tokens(tokens)
        )
        result = {
            "tokens": tokens,
            "edus": [span.to_dict() for span in spans],
            "lookahead_tokens": segmenter.lookahead,
            "threshold": segmenter.threshold,
            "device": str(segmenter.device),
        }
        emit_json(result)
        return

    for token in tokens:
        decision = segmenter.push_token(token)
        if decision is not None:
            emit_json_line(decision.to_dict())
        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)
    for decision in segmenter.flush():
        emit_json_line(decision.to_dict())


if __name__ == "__main__":
    main()
