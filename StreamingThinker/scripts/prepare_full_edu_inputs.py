from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online_edu_segmenter import OnlineEDUTransformer  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "data" / "train" / "streamingcot_train.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "train" / "streamingcot_train_full_edu_inputs.json"


PROTECTED_BOUNDARY_RULES = [
    # Decimal and thousands separators: 1<EOS>.5, 1.<EOS>5, 1,<EOS>000
    (re.compile(r"(?<=\d)([.,])<EOS>(?=\d)"), r"\1"),
    (re.compile(r"(?<=\d)<EOS>(?=[.,]\d)"), ""),
    # Currency and percentages: $<EOS>20, 20<EOS>%
    (re.compile(r"(?<=[$€£¥])<EOS>(?=\s*\d)"), ""),
    (re.compile(r"(?<=\d)<EOS>(?=\s*%)"), ""),
    # Simple fractions, ratios, negative ranges, and clock-like expressions.
    (re.compile(r"(?<=\d)<EOS>(?=\s*[/:\-]\s*\d)"), ""),
    (re.compile(r"(?<=\d)\s*[/:\-]\s*<EOS>(?=\d)"), ""),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the online EDU segmenter to split both Instruct_Context.instruct "
            "and Instruct_Context.context into Full-EDU units."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--format-version",
        choices=["v1", "v3"],
        default="v1",
        help=(
            "v1 writes every EDU boundary as <EOS>. "
            "v3 writes intra-sentence EDU boundaries as <EOE> and sentence-final "
            "EDU boundaries as <EOS>."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes. Each worker loads one EDU segmenter. "
            "Use 2-4 on GPU only if VRAM is sufficient."
        ),
    )
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=64,
        help="Number of samples processed per worker task when --num-workers > 1.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--no-repair-protected-boundaries",
        action="store_true",
        help="Disable light post-processing that removes <EOS> inside numbers.",
    )
    parser.add_argument(
        "--keep-original-input",
        action="store_true",
        help="Store original instruct/context under full_edu_metadata for auditing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def repair_protected_boundaries(text: str) -> tuple[str, int]:
    total_repairs = 0
    for pattern, replacement in PROTECTED_BOUNDARY_RULES:
        while True:
            text, count = pattern.subn(replacement, text)
            total_repairs += count
            if count == 0:
                break
    return text, total_repairs


def count_edus(text: str) -> int:
    return sum(1 for part in re.split(r"<EOE>|<EOS>", str(text or "")) if part.strip())


def boundary_counts(text: str) -> dict[str, int]:
    text = str(text or "")
    return {
        "eoe": text.count("<EOE>"),
        "eos": text.count("<EOS>"),
        "units": count_edus(text),
    }


def normalize_segmented_text(text: str, *, format_version: str = "v1") -> str:
    parts = [part.strip() for part in str(text or "").split("<EOS>") if part.strip()]
    if not parts:
        return ""
    if format_version == "v3":
        marked_parts = [f"{part}<EOE>" for part in parts[:-1]]
        marked_parts.append(f"{parts[-1]}<EOS>")
        return " ".join(marked_parts)
    return " ".join(f"{part}<EOS>" for part in parts)


def strip_trailing_eos(text: str) -> str:
    text = str(text or "").strip()
    while text.endswith("<EOS>"):
        text = text[: -len("<EOS>")].rstrip()
    return text


def segment_v3_context(
    text: str,
    *,
    transformer: OnlineEDUTransformer,
    repair_boundaries: bool,
) -> tuple[str, dict[str, Any]]:
    """Segment each original sentence separately, then relabel inner EDU marks."""
    transformed_sentences = []
    repair_count = 0
    stats = {
        "input_tokens": 0,
        "emitted_edus": 0,
        "predicted_boundaries": 0,
        "forced_boundaries": 0,
        "utterances": 0,
        "wall_time_s": 0.0,
    }
    for sentence in str(text or "").split("<EOS>"):
        if not sentence.strip():
            continue
        segmented, sentence_stats = transformer.segment_utterance(sentence)
        if repair_boundaries:
            segmented, repairs = repair_protected_boundaries(segmented)
            repair_count += repairs
        transformed_sentences.append(
            normalize_segmented_text(segmented, format_version="v3")
        )
        stats["input_tokens"] += sentence_stats.input_tokens
        stats["emitted_edus"] += sentence_stats.emitted_edus
        stats["predicted_boundaries"] += sentence_stats.predicted_boundaries
        stats["forced_boundaries"] += sentence_stats.forced_boundaries
        stats["utterances"] += sentence_stats.utterances
        stats["wall_time_s"] += sentence_stats.wall_time_s
    return " ".join(part for part in transformed_sentences if part).strip(), {
        **stats,
        "repair_count": repair_count,
    }


def segment_v3_instruct(
    text: str,
    *,
    transformer: OnlineEDUTransformer,
    repair_boundaries: bool,
) -> tuple[str, dict[str, Any]]:
    segmented, stats_obj = transformer.segment_utterance(str(text or ""))
    repair_count = 0
    if repair_boundaries:
        segmented, repair_count = repair_protected_boundaries(segmented)
    # The question's final boundary is implicit <EOQ>; keep any non-final
    # <EOE>/<EOS> boundaries, but do not add a sentence-final <EOS> after the
    # question mark. The dataloader appends <EOQ> to the final question chunk.
    normalized = strip_trailing_eos(normalize_segmented_text(segmented, format_version="v3"))
    return normalized, {
        "input_tokens": stats_obj.input_tokens,
        "emitted_edus": stats_obj.emitted_edus,
        "predicted_boundaries": stats_obj.predicted_boundaries,
        "forced_boundaries": stats_obj.forced_boundaries,
        "utterances": stats_obj.utterances,
        "wall_time_s": stats_obj.wall_time_s,
        "repair_count": repair_count,
    }


def transform_one(
    index: int,
    sample: dict[str, Any],
    *,
    transformer: OnlineEDUTransformer,
    format_version: str,
    repair_boundaries: bool,
    keep_original_input: bool,
) -> dict[str, Any]:
    original_instruct = str(sample.get("Instruct_Context", {}).get("instruct", ""))
    original_context = str(sample.get("Instruct_Context", {}).get("context", ""))
    if format_version == "v3":
        transformed = json.loads(json.dumps(sample, ensure_ascii=False))
        instruct_context = transformed.get("Instruct_Context", {})
        segmented_context, context_meta = segment_v3_context(
            original_context,
            transformer=transformer,
            repair_boundaries=repair_boundaries,
        )
        segmented_instruct, instruct_meta = segment_v3_instruct(
            original_instruct,
            transformer=transformer,
            repair_boundaries=repair_boundaries,
        )
        instruct_context["instruct"] = segmented_instruct
        instruct_context["context"] = segmented_context
        repair_count = instruct_meta["repair_count"] + context_meta["repair_count"]
        metadata = {
            "scope": transformer.scope,
            "device": str(transformer.segmenter.device),
            "threshold": transformer.segmenter.threshold,
            "lookahead_tokens": transformer.segmenter.lookahead,
            "input_tokens": instruct_meta["input_tokens"] + context_meta["input_tokens"],
            "emitted_edus": instruct_meta["emitted_edus"] + context_meta["emitted_edus"],
            "predicted_boundaries": (
                instruct_meta["predicted_boundaries"]
                + context_meta["predicted_boundaries"]
            ),
            "forced_boundaries": (
                instruct_meta["forced_boundaries"] + context_meta["forced_boundaries"]
            ),
            "utterances": instruct_meta["utterances"] + context_meta["utterances"],
            "wall_time_s": instruct_meta["wall_time_s"] + context_meta["wall_time_s"],
        }
    else:
        transformed, metadata = transformer.transform_sample(sample)
        instruct_context = transformed.get("Instruct_Context", {})

        segmented_instruct = str(instruct_context.get("instruct", ""))
        segmented_context = str(instruct_context.get("context", ""))

        repair_count = 0
        if repair_boundaries:
            segmented_instruct, instruct_repairs = repair_protected_boundaries(
                segmented_instruct
            )
            segmented_context, context_repairs = repair_protected_boundaries(
                segmented_context
            )
            repair_count = instruct_repairs + context_repairs

        # The final question boundary is represented by <EOQ> in the dataloader.
        # Keep internal EDU boundaries, but do not emit a redundant final <EOS>.
        instruct_context["instruct"] = strip_trailing_eos(
            normalize_segmented_text(segmented_instruct)
        )
        instruct_context["context"] = normalize_segmented_text(segmented_context)

    instruct_boundaries = boundary_counts(instruct_context["instruct"])
    context_boundaries = boundary_counts(instruct_context["context"])

    full_edu_metadata = {
        **metadata,
        "source_index": index,
        "format_version": f"full_edu_inputs_{format_version}",
        "repair_protected_boundary_count": repair_count,
        "original_instruct_edus": count_edus(original_instruct),
        "original_context_edus": count_edus(original_context),
        "full_edu_instruct_edus": instruct_boundaries["units"],
        "full_edu_context_edus": context_boundaries["units"],
        "full_edu_instruct_eoe": instruct_boundaries["eoe"],
        "full_edu_instruct_eos": instruct_boundaries["eos"],
        "full_edu_context_eoe": context_boundaries["eoe"],
        "full_edu_context_eos": context_boundaries["eos"],
    }
    if keep_original_input:
        full_edu_metadata["original_instruct"] = original_instruct
        full_edu_metadata["original_context"] = original_context

    transformed["full_edu_metadata"] = full_edu_metadata
    return transformed


def chunked(items: list[tuple[int, dict[str, Any]]], chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def transform_worker_batch(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    device: str,
    checkpoint: str | None,
    threshold: float | None,
    format_version: str,
    repair_boundaries: bool,
    keep_original_input: bool,
) -> list[dict[str, Any]]:
    transformer = OnlineEDUTransformer(
        scope="full",
        device=device,
        checkpoint=Path(checkpoint) if checkpoint else None,
        threshold=threshold,
    )
    return [
        transform_one(
            index,
            sample,
            transformer=transformer,
            format_version=format_version,
            repair_boundaries=repair_boundaries,
            keep_original_input=keep_original_input,
        )
        for index, sample in batch
    ]


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    instruct_counts = [
        row.get("full_edu_metadata", {}).get("full_edu_instruct_edus", 0)
        for row in rows
    ]
    context_counts = [
        row.get("full_edu_metadata", {}).get("full_edu_context_edus", 0)
        for row in rows
    ]
    repairs = [
        row.get("full_edu_metadata", {}).get("repair_protected_boundary_count", 0)
        for row in rows
    ]
    instruct_eoe_counts = [
        row.get("full_edu_metadata", {}).get("full_edu_instruct_eoe", 0)
        for row in rows
    ]
    instruct_eos_counts = [
        row.get("full_edu_metadata", {}).get("full_edu_instruct_eos", 0)
        for row in rows
    ]
    context_eoe_counts = [
        row.get("full_edu_metadata", {}).get("full_edu_context_eoe", 0)
        for row in rows
    ]
    context_eos_counts = [
        row.get("full_edu_metadata", {}).get("full_edu_context_eos", 0)
        for row in rows
    ]
    return {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "format_version": args.format_version,
        "samples": len(rows),
        "start_index": args.start_index,
        "limit": args.limit,
        "device": args.device,
        "num_workers": args.num_workers,
        "worker_chunk_size": args.worker_chunk_size,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "threshold": args.threshold,
        "repair_protected_boundaries": not args.no_repair_protected_boundaries,
        "total_boundary_repairs": sum(repairs),
        "samples_with_boundary_repairs": sum(value > 0 for value in repairs),
        "total_instruct_eoe": sum(instruct_eoe_counts),
        "total_instruct_eos": sum(instruct_eos_counts),
        "total_context_eoe": sum(context_eoe_counts),
        "total_context_eos": sum(context_eos_counts),
        "avg_instruct_edus": (
            sum(instruct_counts) / len(instruct_counts) if instruct_counts else None
        ),
        "min_instruct_edus": min(instruct_counts) if instruct_counts else None,
        "max_instruct_edus": max(instruct_counts) if instruct_counts else None,
        "avg_context_edus": (
            sum(context_counts) / len(context_counts) if context_counts else None
        ),
        "min_context_edus": min(context_counts) if context_counts else None,
        "max_context_edus": max(context_counts) if context_counts else None,
    }


def training_input_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Instruct_Context": row.get("Instruct_Context", {}),
        "answer": row.get("answer", ""),
    }


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}. Pass --overwrite to replace it."
        )

    samples = read_json(args.input)
    selected = []
    for index, sample in enumerate(samples):
        if index < args.start_index:
            continue
        if args.limit is not None and len(selected) >= args.limit:
            break
        selected.append((index, sample))

    if args.num_workers <= 1:
        transformer = OnlineEDUTransformer(
            scope="full",
            device=args.device,
            checkpoint=args.checkpoint,
            threshold=args.threshold,
        )

        rows = []
        for index, sample in tqdm(selected, desc="Full-EDU segment", unit="sample"):
            rows.append(
                transform_one(
                    index,
                    sample,
                    transformer=transformer,
                    format_version=args.format_version,
                    repair_boundaries=not args.no_repair_protected_boundaries,
                    keep_original_input=args.keep_original_input,
                )
            )
    else:
        rows = []
        chunks = list(chunked(selected, args.worker_chunk_size))
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    transform_worker_batch,
                    chunk,
                    device=args.device,
                    checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                    threshold=args.threshold,
                    format_version=args.format_version,
                    repair_boundaries=not args.no_repair_protected_boundaries,
                    keep_original_input=args.keep_original_input,
                )
                for chunk in chunks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Full-EDU segment x{args.num_workers}",
                unit="chunk",
            ):
                rows.extend(future.result())
        rows.sort(
            key=lambda row: row.get("full_edu_metadata", {}).get(
                "source_index", -1
            )
        )

    report_path = args.output.with_name(args.output.stem + ".report.json")
    write_json(report_path, summarize(rows, args))
    write_json(args.output, [training_input_record(row) for row in rows])
    print(f"Wrote {len(rows)} samples to {args.output}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
