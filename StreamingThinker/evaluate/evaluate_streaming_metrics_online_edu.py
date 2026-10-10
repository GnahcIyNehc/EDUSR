from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common import build_parser  # noqa: E402
from evaluate_streaming_metrics_devmap_nosplit import (  # noqa: E402
    average,
    main as run_evaluation,
    median,
    summarize,
)
from online_edu_segmenter import (  # noqa: E402
    IncrementalOnlineEDUStreamingDataCollator,
    OnlineEDUTransformer,
)


def parse_args():
    parser = build_parser(
        default_mode="streaming",
        default_config=(
            REPO_ROOT
            / "configs"
            / "params_qwen_streamingcot_eval_gsm_p2_D3_0_5k_online_context_edu.json"
        ),
        default_attn_implementation="eager",
        description=(
            "Evaluate StreamingThinker while detecting EDU boundaries online "
            "with the bundled Streaming_EDU final_edu model."
        ),
    )
    parser.set_defaults(
        per_device_eval_batch_size=1,
        max_new_tokens=2048,
        temperature=0.0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples.",
    )
    parser.add_argument(
        "--no_sync_cuda_for_latency",
        action="store_true",
        help="Do not synchronize CUDA before latency timestamps.",
    )
    parser.add_argument(
        "--online_edu_scope",
        choices=["sentence", "context", "full"],
        default="context",
        help="Use sentence boundaries, context EDU boundaries, or full EDU boundaries.",
    )
    parser.add_argument(
        "--boundary_detector",
        choices=["model", "sentence_punctuation", "fixed_punctuation"],
        default="model",
        help=(
            "Use the learned online EDU detector or deterministic streaming "
            "punctuation boundaries. Sentence scope always uses source <EOS> markers."
        ),
    )
    parser.add_argument(
        "--online_edu_device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for final_edu. Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--online_edu_checkpoint",
        default=None,
        help="Optional final_edu checkpoint path. Defaults to its bundled checkpoint.",
    )
    parser.add_argument(
        "--online_edu_threshold",
        type=float,
        default=None,
        help="Optional boundary threshold override. Bundled default is 0.38.",
    )
    parser.add_argument(
        "--use_eoe_boundaries",
        action="store_true",
        help=(
            "Render intra-sentence EDU boundaries as <EOE> and sentence-final "
            "boundaries as <EOS>. By default, every EDU boundary keeps the "
            "legacy <EOS> marker."
        ),
    )
    parser.add_argument(
        "--force_complete_source",
        action="store_true",
        default=True,
        help="Prevent termination until all online source segments have been read.",
    )
    parser.add_argument(
        "--max_write_tokens_before_read",
        type=int,
        default=None,
        help="Optional fallback only. Leave unset for arrival-driven preemption.",
    )
    parser.add_argument(
        "--read_policy",
        choices=["arrival_preemption", "finish_eot", "sentence_finish_eot"],
        default="arrival_preemption",
        help=(
            "Choose whether a newly arrived EDU immediately pauses generation "
            "and is read, waits until the current thought emits <EOT>, or "
            "preempts only within the current sentence while sentence "
            "transitions wait for <EOT>."
        ),
    )
    parser.add_argument(
        "--finish_thought_before_read",
        action="store_true",
        help=(
            "Backward-compatible alias for '--read_policy finish_eot'. "
            "Prefer --read_policy in new commands."
        ),
    )
    parser.add_argument(
        "--defer_thinking_until_context",
        action="store_true",
        default=False,
        help=(
            "Read the complete question and first context EDU before generating. "
            "Disabled by default so question reasoning can emit <EOQ> before "
            "the first context segment is read."
        ),
    )
    parser.add_argument(
        "--initial_source_segments_before_thinking",
        type=int,
        default=None,
        help="Reserved for offline sentence-boundary controls.",
    )
    parser.add_argument(
        "--source_token_rate",
        type=float,
        default=None,
        help="Legacy tokenizer-token rate. Prefer --source_words_per_minute.",
    )
    parser.add_argument(
        "--source_words_per_minute",
        type=float,
        default=150.0,
        help=(
            "Spoken rate for the sample question and context only. The fixed "
            "task prompt plus '<SEP>' is always prefilled immediately and is "
            "never delayed by this rate."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip idle wall-clock waits for the next source unit while preserving "
            "the configured source-arrival timeline in reported latency metrics."
        ),
    )
    parser.add_argument(
        "--punctuation_dropout_percent",
        type=float,
        default=0.0,
        help=(
            "Simulate ASR punctuation loss by removing this percentage of "
            "commas and sentence periods from the question and context. "
            "A removed sentence period also removes its adjacent source <EOS>."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.punctuation_dropout_percent <= 100.0:
        parser.error("--punctuation_dropout_percent must be between 0 and 100")
    if args.finish_thought_before_read:
        args.read_policy = "finish_eot"
    args.finish_thought_before_read = args.read_policy == "finish_eot"
    return args


def summarize_online_edu(rows):
    summary = summarize(rows)
    metadata = [row.get("online_edu") for row in rows if row.get("online_edu")]
    read_policies = [
        row.get("latency", {}).get("read_policy")
        for row in rows
        if row.get("latency", {}).get("read_policy")
    ]
    if read_policies:
        read_policy = read_policies[0]
    else:
        finish_thought_flags = [
            row.get("latency", {}).get("finish_thought_before_read")
            for row in rows
            if row.get("latency", {}).get("finish_thought_before_read")
            is not None
        ]
        read_policy = (
            "finish_eot" if any(finish_thought_flags) else "arrival_preemption"
        )

    def metadata_values(key):
        return [item.get(key) for item in metadata]

    online_wall_times = metadata_values("wall_time_s")
    online_processing_times = [
        item.get("processing_time_s", item.get("wall_time_s"))
        for item in metadata
    ]
    online_arrival_wait_times = metadata_values("arrival_wait_time_s")
    skipped_arrival_wait_times = metadata_values("skipped_arrival_wait_time_s")
    simulated_arrival_wait_times = metadata_values("simulated_arrival_wait_time_s")
    input_tokens = metadata_values("input_tokens")
    punctuation_candidates = metadata_values("punctuation_candidates")
    punctuation_dropped = metadata_values("punctuation_dropped")
    total_punctuation_candidates = sum(
        value for value in punctuation_candidates if value is not None
    )
    total_punctuation_dropped = sum(
        value for value in punctuation_dropped if value is not None
    )
    boundary_handoff = metadata_values("avg_boundary_to_llm_read_s")
    llm_read_durations = metadata_values("avg_llm_read_duration_s")
    per_token_ms = [
        1000.0 * processing_time / token_count
        for processing_time, token_count in zip(
            online_processing_times, input_tokens
        )
        if processing_time is not None and token_count
    ]
    answer_with_online = [
        row["latency"]["answer_latency_s"]
        for row in rows
        if row.get("online_edu")
        and row["latency"].get("answer_latency_s") is not None
    ]
    total_with_online = [
        row["latency"]["total_generation_s"]
        for row in rows
        if row.get("online_edu")
        and row["latency"].get("total_generation_s") is not None
    ]
    preemptions_per_sample = []
    target_tokens_at_boundary_reads = []
    arrival_preemption_reasons = {
        "arrival_preemption",
        "post_read_arrival_preemption",
        "same_sentence_arrival_preemption",
        "post_read_same_sentence_preemption",
    }
    for item in metadata:
        events = [
            event
            for event in item.get("events", [])
            if event.get("detected_boundary")
        ]
        preemptions = [
            event
            for event in events
            if (event.get("target_token_index_at_read") or 0) > 0
            and (
                event.get("read_reason") in arrival_preemption_reasons
                or (
                    event.get("read_reason") is None
                    and read_policy == "arrival_preemption"
                )
            )
        ]
        preemptions_per_sample.append(len(preemptions))
        target_tokens_at_boundary_reads.extend(
            event["target_token_index_at_read"] for event in preemptions
        )

    summary.update(
        {
            "online_edu_samples": len(metadata),
            "online_edu_interleaved_with_generation": True,
            "read_policy": read_policy,
            "arrival_preemption_enabled": read_policy
            in {"arrival_preemption", "sentence_finish_eot"},
            "sentence_transition_finish_eot": (
                read_policy == "sentence_finish_eot"
            ),
            "online_edu_scope": metadata[0].get("scope") if metadata else None,
            "boundary_detector": (
                metadata[0].get("boundary_detector") if metadata else None
            ),
            "deterministic_punctuation_marks": (
                metadata[0].get("deterministic_punctuation_marks")
                if metadata
                else None
            ),
            "online_edu_device": metadata[0].get("device") if metadata else None,
            "online_edu_threshold": metadata[0].get("threshold") if metadata else None,
            "online_edu_lookahead_tokens": (
                metadata[0].get("lookahead_tokens") if metadata else None
            ),
            "source_words_per_minute": (
                metadata[0].get("source_words_per_minute")
                if metadata
                else None
            ),
            "fast": metadata[0].get("fast") if metadata else False,
            "punctuation_dropout_percent": (
                metadata[0].get("punctuation_dropout_percent")
                if metadata
                else 0.0
            ),
            "punctuation_dropout_seed": (
                metadata[0].get("punctuation_dropout_seed")
                if metadata
                else None
            ),
            "total_punctuation_candidates": total_punctuation_candidates,
            "total_punctuation_dropped": total_punctuation_dropped,
            "effective_punctuation_dropout_percent": (
                100.0
                * total_punctuation_dropped
                / total_punctuation_candidates
                if total_punctuation_candidates
                else 0.0
            ),
            "total_commas_dropped": sum(
                value
                for value in metadata_values("commas_dropped")
                if value is not None
            ),
            "total_periods_dropped": sum(
                value
                for value in metadata_values("periods_dropped")
                if value is not None
            ),
            "total_source_eos_removed_by_punctuation_dropout": sum(
                value
                for value in metadata_values(
                    "source_eos_removed_by_punctuation_dropout"
                )
                if value is not None
            ),
            "use_eoe_boundaries": (
                metadata[0].get("use_eoe_boundaries")
                if metadata
                else None
            ),
            "prompt_prefill_mode": (
                metadata[0].get("prompt_prefill_mode")
                if metadata
                else None
            ),
            "source_rate_applies_to": (
                metadata[0].get("source_rate_applies_to")
                if metadata
                else None
            ),
            "avg_online_edu_wall_time_s": average(online_wall_times),
            "median_online_edu_wall_time_s": median(online_wall_times),
            "avg_online_edu_processing_time_s": average(
                online_processing_times
            ),
            "avg_source_arrival_wait_time_s": average(
                online_arrival_wait_times
            ),
            "avg_skipped_source_arrival_wait_time_s": average(
                skipped_arrival_wait_times
            ),
            "total_skipped_source_arrival_wait_time_s": sum(
                value for value in skipped_arrival_wait_times if value is not None
            ),
            "avg_simulated_source_arrival_wait_time_s": average(
                simulated_arrival_wait_times
            ),
            "avg_online_edu_ms_per_input_token": average(per_token_ms),
            "avg_online_edu_input_tokens": average(input_tokens),
            "avg_online_edu_emitted_edus": average(
                metadata_values("emitted_edus")
            ),
            "avg_online_edu_predicted_boundaries": average(
                metadata_values("predicted_boundaries")
            ),
            "avg_online_edu_forced_boundaries": average(
                metadata_values("forced_boundaries")
            ),
            "avg_boundary_to_llm_read_s": average(boundary_handoff),
            "median_boundary_to_llm_read_s": median(boundary_handoff),
            "avg_llm_read_duration_per_edu_s": average(llm_read_durations),
            "median_llm_read_duration_per_edu_s": median(llm_read_durations),
            "samples_with_arrival_preemption": sum(
                count > 0 for count in preemptions_per_sample
            ) if read_policy in {"arrival_preemption", "sentence_finish_eot"} else 0,
            "avg_arrival_preemptions_per_sample": average(
                preemptions_per_sample
            ) if read_policy in {"arrival_preemption", "sentence_finish_eot"} else 0.0,
            "avg_target_tokens_at_preempting_read": average(
                target_tokens_at_boundary_reads
            ) if read_policy in {"arrival_preemption", "sentence_finish_eot"} else None,
            "avg_answer_latency_including_online_edu_s": average(
                answer_with_online
            ),
            "median_answer_latency_including_online_edu_s": median(
                answer_with_online
            ),
            "avg_total_including_online_edu_s": average(total_with_online),
            "median_total_including_online_edu_s": median(total_with_online),
        }
    )
    return summary


def main():
    args = parse_args()
    transformer = OnlineEDUTransformer(
        scope=args.online_edu_scope,
        device=args.online_edu_device,
        checkpoint=args.online_edu_checkpoint,
        threshold=args.online_edu_threshold,
        use_eoe_boundaries=args.use_eoe_boundaries,
        boundary_detector=args.boundary_detector,
    )
    collator_factory = partial(
        IncrementalOnlineEDUStreamingDataCollator,
        online_edu_transformer=transformer,
        source_token_rate=args.source_token_rate,
        source_words_per_minute=args.source_words_per_minute,
        prompt_prefill_mode="immediate",
        use_eoe_boundaries=args.use_eoe_boundaries,
        fast=args.fast,
        punctuation_dropout_percent=args.punctuation_dropout_percent,
    )
    run_evaluation(
        args=args,
        data_collator_factory=collator_factory,
        summarize_fn=summarize_online_edu,
        prepare_dataloader=False,
    )


if __name__ == "__main__":
    main()
