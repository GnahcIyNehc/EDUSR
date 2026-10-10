import gc
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig

from common import (
    REPO_ROOT,
    answers_match,
    build_parser,
    build_tokenizer,
    extract_boxed_answer,
    extract_reference_answer,
    gather_objects,
    get_model_class,
    setup_seed,
)

sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402
from dataloader_hf import StreamingDataCollator  # noqa: E402


GIB = 1024 ** 3


def local_checkpoint_size_bytes(model_name_or_path):
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        return None
    weight_files = list(model_path.glob("*.safetensors"))
    if not weight_files:
        return None
    return sum(path.stat().st_size for path in weight_files)


def available_system_memory_bytes():
    if os.name != "nt":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def gpu_memory_bytes(device_index):
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.free), int(info.total), "nvml"
    except Exception:
        pass
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        first_line = output.strip().splitlines()[0]
        free_mib, total_mib = [int(part.strip()) for part in first_line.split(",")]
        return free_mib * 1024 ** 2, total_mib * 1024 ** 2, "nvidia-smi"
    except Exception:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        return int(free_bytes), int(total_bytes), "torch"


def parse_args():
    parser = build_parser(
        default_mode="streaming",
        default_config=REPO_ROOT / "configs" / "params_qwen_streamingcot_eval_gsm_p2_100.json",
        default_attn_implementation="eager",
        description="Evaluate StreamingThinker accuracy and streaming wall-clock latency.",
    )
    parser.set_defaults(per_device_eval_batch_size=1, max_new_tokens=256, temperature=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument(
        "--no_sync_cuda_for_latency",
        action="store_true",
        help="Do not synchronize CUDA before latency timestamps. Faster, but less accurate on GPU.",
    )
    parser.add_argument(
        "--force_complete_source",
        action="store_true",
        help="Prevent generation from terminating before every source segment is read.",
    )
    parser.add_argument(
        "--max_write_tokens_before_read",
        type=int,
        default=None,
        help="Force the next source read after this many target tokens without a separator.",
    )
    parser.add_argument(
        "--defer_thinking_until_context",
        action="store_true",
        help="Read all question/instruct segments and the first context segment before generating.",
    )
    parser.add_argument(
        "--initial_source_segments_before_thinking",
        type=int,
        default=None,
        help="Offline source segments to read before the first generated thought.",
    )
    return parser.parse_args()


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def resolve_sample_range(total_samples, start_index, end_index, limit):
    """将闭区间参数转换成供 islice 使用的左闭右开区间。"""
    start = 0 if start_index is None else int(start_index)
    if start < 0:
        raise ValueError("--sample_start_index must be >= 0.")
    if end_index is not None and int(end_index) < start:
        raise ValueError("--sample_end_index must be >= --sample_start_index.")
    if limit is not None and int(limit) <= 0:
        raise ValueError("--limit must be > 0.")

    stop = int(end_index) + 1 if end_index is not None else None
    if total_samples is not None:
        total_samples = int(total_samples)
        if start >= total_samples:
            raise ValueError(
                f"sample_start_index={start} is outside a dataset with "
                f"{total_samples} samples."
            )
        stop = total_samples if stop is None else min(stop, total_samples)
    if limit is not None:
        limit_stop = start + int(limit)
        stop = limit_stop if stop is None else min(stop, limit_stop)
    if stop is not None and stop <= start:
        raise ValueError("The requested sample range is empty.")
    return start, stop


def summarize(rows):
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    empty_predictions = sum(1 for row in rows if not row["prediction"])
    reflection_rows = [
        row
        for row in rows
        if row.get("forced_reflection", {}).get("enabled")
    ]
    draft_rows = [
        row for row in reflection_rows if row.get("draft_prediction")
    ]

    def latency_values(key):
        return [row["latency"].get(key) for row in rows]

    def generation_phase_token_values():
        streaming_values = []
        post_input_values = []
        for row in rows:
            latency = row["latency"]
            total_tokens = latency.get("generated_target_tokens")
            streaming_tokens = latency.get("streaming_stage_target_tokens")
            if streaming_tokens is None:
                streaming_tokens = latency.get("input_complete_target_token_index")
            post_input_tokens = latency.get("post_input_complete_target_tokens")
            if (
                post_input_tokens is None
                and total_tokens is not None
                and streaming_tokens is not None
            ):
                post_input_tokens = max(0, total_tokens - streaming_tokens)
            if streaming_tokens is not None:
                streaming_values.append(streaming_tokens)
            if post_input_tokens is not None:
                post_input_values.append(post_input_tokens)
        return streaming_values, post_input_values

    detected_answers = sum(1 for row in rows if row["latency"].get("answer_time") is not None)
    answers_after_input_complete = sum(
        1 for row in rows if row["latency"].get("answer_after_input_complete_s") is not None
    )
    streaming_stage_token_values, post_input_token_values = generation_phase_token_values()
    answer_read_ratios = []
    for row in rows:
        latency = row["latency"]
        read_tokens = latency.get("source_tokens_read_at_answer")
        total_tokens = latency.get("total_source_tokens")
        if read_tokens is not None and total_tokens:
            answer_read_ratios.append(read_tokens / total_tokens)

    summary = {
        "samples": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "empty_predictions": empty_predictions,
        "forced_reflection_enabled_samples": len(reflection_rows),
        "forced_reflection_triggered_samples": sum(
            1
            for row in reflection_rows
            if row.get("forced_reflection", {}).get("triggered")
        ),
        "forced_reflection_completed_samples": sum(
            1
            for row in reflection_rows
            if row.get("forced_reflection", {}).get("completed")
        ),
        "forced_reflection_missing_final_answer_samples": sum(
            1 for row in reflection_rows if not row.get("prediction")
        ),
        "draft_answer_samples": len(draft_rows),
        "draft_correct": sum(
            1 for row in draft_rows if row.get("draft_correct")
        ),
        "draft_accuracy": (
            sum(1 for row in draft_rows if row.get("draft_correct"))
            / len(draft_rows)
            if draft_rows
            else None
        ),
        "reflection_wrong_to_right": sum(
            1
            for row in draft_rows
            if not row.get("draft_correct") and row.get("correct")
        ),
        "reflection_right_to_wrong": sum(
            1
            for row in draft_rows
            if row.get("draft_correct") and not row.get("correct")
        ),
        "avg_reflection_stage_target_tokens": average(
            latency_values("reflection_stage_target_tokens")
        ),
        "median_reflection_stage_target_tokens": median(
            latency_values("reflection_stage_target_tokens")
        ),
        "avg_reflection_generation_s": average(
            latency_values("reflection_generation_s")
        ),
        "median_reflection_generation_s": median(
            latency_values("reflection_generation_s")
        ),
        "detected_answers": detected_answers,
        "answers_after_input_complete": answers_after_input_complete,
        "missing_exact_answer_time": sum(1 for row in rows if row["latency"].get("answer_time") is None),
        "missing_input_complete_time": sum(1 for row in rows if row["latency"].get("input_complete_time") is None),
        "missing_source_start_time": sum(
            1 for row in rows if row["latency"].get("source_start_time") is None
        ),
        "missing_source_arrival_complete_time": sum(
            1
            for row in rows
            if row["latency"].get("source_arrival_complete_time") is None
        ),
        "missing_model_read_complete_time": sum(
            1
            for row in rows
            if row["latency"].get("model_read_complete_time") is None
        ),
        "missing_answer_time": sum(
            1 for row in rows if row["latency"].get("answer_after_input_complete_s") is None
        ),
        "posthoc_answer_time_fallbacks": sum(
            1 for row in rows if row["latency"].get("answer_time_source") == "fallback_generation_end"
        ),
        "answers_before_input_complete": sum(
            1 for row in rows if row["latency"].get("answer_before_input_complete") is True
        ),
        "early_answers_suppressed_before_input_complete": sum(
            1 for row in rows if row["latency"].get("early_answer_time") is not None
        ),
        "samples_with_forced_reads": sum(
            1 for row in rows if row["latency"].get("forced_read_count", 0) > 0
        ),
        "source_completion_rate": (
            sum(1 for row in rows if row["latency"].get("input_completed")) / n
            if n
            else None
        ),
        "avg_first_thought_latency_s": average(latency_values("first_thought_latency_s")),
        "median_first_thought_latency_s": median(latency_values("first_thought_latency_s")),
        "avg_first_thought_after_prefix_s": average(latency_values("first_thought_after_prefix_s")),
        "median_first_thought_after_prefix_s": median(latency_values("first_thought_after_prefix_s")),
        "avg_first_thought_after_content_start_s": average(
            latency_values("first_thought_after_content_start_s")
        ),
        "median_first_thought_after_content_start_s": median(
            latency_values("first_thought_after_content_start_s")
        ),
        "avg_content_tokens_until_first_thought": average(
            latency_values("content_tokens_until_first_thought")
        ),
        "median_content_tokens_until_first_thought": median(
            latency_values("content_tokens_until_first_thought")
        ),
        "avg_content_arrival_tokens_until_first_thought": average(
            latency_values("content_arrival_tokens_until_first_thought")
        ),
        "median_content_arrival_tokens_until_first_thought": median(
            latency_values("content_arrival_tokens_until_first_thought")
        ),
        "avg_content_arrival_words_until_first_thought": average(
            latency_values("content_arrival_words_until_first_thought")
        ),
        "median_content_arrival_words_until_first_thought": median(
            latency_values("content_arrival_words_until_first_thought")
        ),
        "avg_answer_latency_s": average(latency_values("answer_latency_s")),
        "median_answer_latency_s": median(latency_values("answer_latency_s")),
        "avg_answer_after_prefix_s": average(latency_values("answer_after_prefix_s")),
        "median_answer_after_prefix_s": median(latency_values("answer_after_prefix_s")),
        "avg_answer_after_input_complete_s": average(latency_values("answer_after_input_complete_s")),
        "median_answer_after_input_complete_s": median(latency_values("answer_after_input_complete_s")),
        "avg_source_start_to_answer_s": average(
            latency_values("source_start_to_answer_s")
        ),
        "median_source_start_to_answer_s": median(
            latency_values("source_start_to_answer_s")
        ),
        "avg_source_arrival_complete_to_answer_s": average(
            latency_values("source_arrival_complete_to_answer_s")
        ),
        "median_source_arrival_complete_to_answer_s": median(
            latency_values("source_arrival_complete_to_answer_s")
        ),
        "avg_model_read_complete_to_answer_s": average(
            latency_values("model_read_complete_to_answer_s")
        ),
        "median_model_read_complete_to_answer_s": median(
            latency_values("model_read_complete_to_answer_s")
        ),
        "avg_source_arrival_duration_s": average(
            latency_values("source_arrival_duration_s")
        ),
        "median_source_arrival_duration_s": median(
            latency_values("source_arrival_duration_s")
        ),
        "avg_input_complete_to_end_s": average(latency_values("input_complete_to_end_s")),
        "median_input_complete_to_end_s": median(latency_values("input_complete_to_end_s")),
        "avg_total_after_prefix_s": average(latency_values("total_after_prefix_s")),
        "median_total_after_prefix_s": median(latency_values("total_after_prefix_s")),
        "avg_total_generation_s": average(latency_values("total_generation_s")),
        "median_total_generation_s": median(latency_values("total_generation_s")),
        "avg_generated_target_tokens": average(latency_values("generated_target_tokens")),
        "median_generated_target_tokens": median(latency_values("generated_target_tokens")),
        "samples_with_generation_phase_token_split": len(streaming_stage_token_values),
        "avg_streaming_stage_target_tokens": average(streaming_stage_token_values),
        "median_streaming_stage_target_tokens": median(streaming_stage_token_values),
        "avg_post_input_complete_target_tokens": average(post_input_token_values),
        "median_post_input_complete_target_tokens": median(post_input_token_values),
        "avg_forced_read_count": average(latency_values("forced_read_count")),
        "avg_prompt_prefix_tokens": average(latency_values("prompt_prefix_tokens")),
        "avg_raw_first_read_tokens": average(latency_values("raw_first_read_tokens")),
        "avg_raw_total_source_tokens": average(latency_values("raw_total_source_tokens")),
        "avg_first_read_tokens": average(latency_values("first_read_tokens")),
        "avg_total_source_tokens": average(latency_values("total_source_tokens")),
        "avg_streaming_source_segments": average(latency_values("streaming_source_segments")),
        "avg_source_tokens_read_at_first_thought": average(
            latency_values("source_tokens_read_at_first_thought")
        ),
        "avg_raw_source_tokens_read_at_first_thought": average(
            latency_values("raw_source_tokens_read_at_first_thought")
        ),
        "avg_source_segments_read_at_answer": average(latency_values("source_segments_read_at_answer")),
        "median_source_segments_read_at_answer": median(latency_values("source_segments_read_at_answer")),
        "avg_source_tokens_read_at_answer": average(latency_values("source_tokens_read_at_answer")),
        "avg_raw_source_tokens_read_at_answer": average(latency_values("raw_source_tokens_read_at_answer")),
        "median_source_tokens_read_at_answer": median(latency_values("source_tokens_read_at_answer")),
        "avg_source_read_ratio_at_answer": average(answer_read_ratios),
        "median_source_read_ratio_at_answer": median(answer_read_ratios),
    }
    return summary


def main(
    args=None,
    data_collator_factory=StreamingDataCollator,
    summarize_fn=summarize,
    prepare_dataloader=True,
):
    args = args or parse_args()
    if args.inference_mode != "streaming":
        raise ValueError("This metrics script only supports --inference_mode streaming.")
    if args.per_device_eval_batch_size != 1:
        raise ValueError("Use --per_device_eval_batch_size 1 for interpretable streaming latency metrics.")

    setup_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = utils.Params(args.data_config)
    params.file_path = utils.resolve_path(params.file_path, REPO_ROOT)

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config._attn_implementation = args.attn_implementation

    tokenizer = build_tokenizer(
        tokenizer_name_or_path,
        config,
        padding_side="right",
        use_eoe_tokens=(
            getattr(args, "tokenizer_uses_eoe_tokens", False)
            or getattr(args, "use_eoe_boundaries", False)
        ),
        use_fast=getattr(args, "use_fast_tokenizer", True),
    )

    model_cls = get_model_class(args.model_backbone)
    torch_dtype = (
        torch.bfloat16
        if args.mixed_precision == "bf16"
        else torch.float16
        if args.mixed_precision == "fp16"
        else None
    )
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        split_at = config.num_hidden_layers // 2
        device_map = {
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
            "model.norm": 1,
            "lm_head": 0,
        }
        for layer_idx in range(config.num_hidden_layers):
            device_map[f"model.layers.{layer_idx}"] = 0 if layer_idx < split_at else 1
        max_memory = {0: "45GiB", 1: "45GiB", "cpu": "128GiB"}
    else:
        free_gpu_bytes, total_gpu_bytes, memory_source = gpu_memory_bytes(0)
        checkpoint_bytes = local_checkpoint_size_bytes(
            args.model_name_or_path
        )
        reserve_bytes = int(args.gpu_memory_reserve_gib * GIB)
        available_cpu_bytes = available_system_memory_bytes()
        checkpoint_text = (
            f"{checkpoint_bytes / GIB:.2f} GiB"
            if checkpoint_bytes is not None
            else "unknown"
        )
        cpu_text = (
            f"{available_cpu_bytes / GIB:.2f} GiB"
            if available_cpu_bytes is not None
            else "unknown"
        )
        print(
            "Memory preflight: "
            f"GPU free={free_gpu_bytes / GIB:.2f} GiB/"
            f"{total_gpu_bytes / GIB:.2f} GiB "
            f"({memory_source}), checkpoint={checkpoint_text}, "
            f"system RAM free={cpu_text}"
        )
        if (
            checkpoint_bytes is not None
            and free_gpu_bytes < checkpoint_bytes + reserve_bytes
            and not args.allow_cpu_offload
        ):
            required_gib = (checkpoint_bytes + reserve_bytes) / GIB
            raise RuntimeError(
                "Insufficient free GPU memory before model loading: "
                f"{free_gpu_bytes / GIB:.2f} GiB free, approximately "
                f"{required_gib:.2f} GiB required "
                f"({checkpoint_bytes / GIB:.2f} GiB checkpoint + "
                f"{args.gpu_memory_reserve_gib:.2f} GiB runtime reserve). "
                "Close GPU-heavy applications and retry."
            )
        usable_gpu_gib = max(
            1, int((free_gpu_bytes - reserve_bytes) // GIB)
        )
        usable_cpu_gib = (
            max(1, int((available_cpu_bytes - 2 * GIB) // GIB))
            if available_cpu_bytes is not None
            else 8
        )
        # 单卡显存不足时允许 Accelerate 按完整模块自动卸载，避免拆开层内张量。
        device_map = "auto" if args.allow_cpu_offload else {"": 0}
        max_memory = {
            0: f"{usable_gpu_gib}GiB",
            "cpu": f"{usable_cpu_gib}GiB",
        }
    print(f"Using device_map={device_map} max_memory={max_memory}")
    original_allocator_warmup = None
    if args.disable_model_load_warmup:
        # Windows WDDM 可能无法申请接近整卡大小的单块临时张量；跳过它不改变模型参数。
        from transformers import modeling_utils

        original_allocator_warmup = modeling_utils.caching_allocator_warmup
        modeling_utils.caching_allocator_warmup = lambda *unused_args, **unused_kwargs: None
    try:
        model = model_cls.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=True,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        )
    finally:
        if original_allocator_warmup is not None:
            modeling_utils.caching_allocator_warmup = original_allocator_warmup
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if args.temperature <= 0:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50

    data_collator = data_collator_factory(
        file_path=params.file_path,
        tokenizer=tokenizer,
        Instruct=params.Instruct,
        end_Instruct=params.end_Instruct,
        assistant_Instruct=params.assistant_Instruct,
        reasoning_connect_1=params.reasoning_connect_1,
        reasoning_connect_2=params.reasoning_connect_2,
        reasoning_connect_3=params.reasoning_connect_3,
        inference_mode=args.inference_mode,
        split_mode=args.split_mode,
        is_training=False,
        pe_cache_start=0,
        is_batch_from_pe_zero=args.batch_from_position_zero,
    )

    dataloader = DataLoader(
        data_collator.dataset_loader(),
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator.collate_fn_inference,
    )

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if prepare_dataloader:
        dataloader = accelerator.prepare(dataloader)
    model.eval()
    model_for_generation = model

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    rows = []
    try:
        dataset_total = len(dataloader)
    except TypeError:
        dataset_total = None
    range_start, range_stop = resolve_sample_range(
        dataset_total,
        getattr(args, "sample_start_index", 0),
        getattr(args, "sample_end_index", None),
        args.limit,
    )
    # islice 会保留原始文件顺序，同时避免对区间外样本执行模型生成。
    selected_batches = itertools.islice(dataloader, range_start, range_stop)
    progress_total = (
        range_stop - range_start if range_stop is not None else None
    )
    progress_bar = tqdm(
        selected_batches,
        total=progress_total,
        desc="Evaluating",
        unit="sample",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )
    running_total = 0
    running_correct = 0
    running_empty = 0
    running_max_token = 0

    for range_offset, batch in enumerate(progress_bar):
        sample_index = range_start + range_offset

        input_ids = batch["source_tokens"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        label_text = tokenizer.decode(labels[0], skip_special_tokens=False)
        online_streams = batch.get("online_edu_boundary_stream")
        online_stream = online_streams[0] if online_streams else None

        latency = {"synchronize_cuda": not args.no_sync_cuda_for_latency}
        generation_kwargs = {
            "input_ids": input_ids.to(accelerator.device),
            "attention_mask": attention_mask.to(accelerator.device),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
            "assistant_token": batch["assistant_token"].to(accelerator.device),
            "end_Instruct": params.end_Instruct,
            "generate_mode": "streaming",
            "tokenizer": tokenizer,
            "pe_cache_length": 0,
            "split_mode": args.split_mode,
            "_lengths": batch["_lengths"],
            "wait_k": 1,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "latency_recorder": latency,
            "force_complete_source": getattr(args, "force_complete_source", False),
            "max_write_tokens_before_read": getattr(
                args, "max_write_tokens_before_read", None
            ),
            "defer_thinking_until_context": getattr(
                args, "defer_thinking_until_context", False
            ),
            "finish_thought_before_read": getattr(
                args, "finish_thought_before_read", False
            ),
            "read_policy": getattr(args, "read_policy", None),
            "initial_source_segments_before_thinking": getattr(
                args, "initial_source_segments_before_thinking", None
            ),
            "force_reflection_after_draft": bool(
                getattr(params, "force_reflection_after_draft", False)
            ),
            "forced_reflection_prompt": getattr(
                params, "forced_reflection_prompt", None
            ),
            "reflection_trigger_token": getattr(
                params, "reflection_trigger_token", "<EOA>"
            ),
            "forced_reflection_min_new_tokens": int(
                getattr(params, "forced_reflection_min_new_tokens", 0)
            ),
        }
        if online_stream is not None:
            generation_kwargs["online_edu_boundary_stream"] = online_stream
        if args.temperature > 0:
            generation_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )

        with torch.inference_mode():
            generated_tokens, generation_lengths = model_for_generation.streaming_generate(**generation_kwargs)

        if isinstance(generation_lengths, torch.Tensor):
            token_length = int(generation_lengths.reshape(-1)[0].item())
        elif isinstance(generation_lengths, list) and generation_lengths:
            token_length = int(generation_lengths[0])
        else:
            token_length = None

        generation = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
        prediction_text = generation
        prediction_scope = "full_generation"
        forced_reflection_enabled = bool(
            getattr(params, "force_reflection_after_draft", False)
        )
        draft_prediction = ""
        if forced_reflection_enabled:
            draft_end_index = latency.get(
                "reflection_prompt_start_output_token_index"
            )
            if draft_end_index is not None:
                draft_text = tokenizer.decode(
                    generated_tokens[0, :int(draft_end_index)],
                    skip_special_tokens=False,
                )
                draft_prediction = extract_boxed_answer(draft_text)
            reflection_content_index = latency.get(
                "reflection_content_start_output_token_index"
            )
            if reflection_content_index is not None:
                prediction_text = tokenizer.decode(
                    generated_tokens[0, int(reflection_content_index):],
                    skip_special_tokens=False,
                )
                prediction_scope = "forced_reflection_only"
            else:
                prediction_text = ""
                prediction_scope = "forced_reflection_missing"
        elif getattr(args, "force_complete_source", False):
            post_input_index = latency.get("input_complete_output_token_index")
            if post_input_index is not None:
                prediction_text = tokenizer.decode(
                    generated_tokens[0, int(post_input_index):],
                    skip_special_tokens=False,
                )
                prediction_scope = "after_input_complete"
        prediction = extract_boxed_answer(prediction_text)
        answer = extract_reference_answer(label_text)
        draft_correct = (
            answers_match(draft_prediction, answer)
            if draft_prediction
            else None
        )
        if prediction and latency.get("answer_time") is None and latency.get("end_time") is not None:
            latency["answer_time"] = latency["end_time"]
            latency["answer_text_at_detection"] = prediction
            latency["answer_time_source"] = "fallback_generation_end"
            latency["answer_latency_s"] = (
                latency["end_time"] - latency["start_time"] if latency.get("start_time") is not None else None
            )
            if latency.get("input_complete_time") is not None:
                latency["answer_after_input_complete_s"] = latency["end_time"] - latency["input_complete_time"]
                latency["model_read_complete_time"] = latency["input_complete_time"]
                latency["model_read_complete_to_answer_s"] = (
                    latency["end_time"] - latency["input_complete_time"]
                )
            if latency.get("prefix_prefill_done_time") is not None:
                latency["answer_after_prefix_s"] = latency["end_time"] - latency["prefix_prefill_done_time"]
            if latency.get("source_start_time") is not None:
                latency["source_start_to_answer_s"] = (
                    latency["end_time"] - latency["source_start_time"]
                )
            if latency.get("source_arrival_complete_time") is not None:
                latency["source_arrival_complete_to_answer_s"] = (
                    latency["end_time"]
                    - latency["source_arrival_complete_time"]
                )
            latency["input_completed"] = latency.get("input_complete_time") is not None
            latency["answer_before_input_complete"] = (
                latency.get("input_complete_time") is None
                or latency["answer_time"] < latency["input_complete_time"]
            )
        original_indices = batch.get("original_index")
        original_index = (
            int(original_indices[0])
            if original_indices is not None
            else sample_index
        )
        row = {
            "index": original_index,
            "subset_position": sample_index,
            "prompt": prompt,
            "generation": generation,
            "prediction": prediction,
            "prediction_scope": prediction_scope,
            "draft_prediction": draft_prediction,
            "draft_correct": draft_correct,
            "answer": answer,
            "correct": answers_match(prediction, answer),
            "token_length": token_length,
            "latency": latency,
            "forced_reflection": {
                "enabled": forced_reflection_enabled,
                "triggered": bool(latency.get("reflection_triggered", False)),
                "completed": bool(latency.get("reflection_completed", False)),
                "trigger_token": getattr(
                    params, "reflection_trigger_token", "<EOA>"
                )
                if forced_reflection_enabled
                else None,
                "prompt": getattr(
                    params, "forced_reflection_prompt", None
                )
                if forced_reflection_enabled
                else None,
                "min_new_tokens": int(
                    getattr(params, "forced_reflection_min_new_tokens", 0)
                )
                if forced_reflection_enabled
                else 0,
            },
        }
        running_total += 1
        if row["correct"]:
            running_correct += 1
        if not prediction:
            running_empty += 1
        if token_length is not None and token_length >= args.max_new_tokens:
            running_max_token += 1
        if accelerator.is_local_main_process:
            progress_bar.set_postfix(
                {
                    "acc": f"{running_correct / running_total:.2%}",
                    "correct": f"{running_correct}/{running_total}",
                    "empty": running_empty,
                    "top": running_max_token,
                }
            )
        if online_stream is not None:
            row["online_edu"] = online_stream.metadata()
            row["prompt"] = prompt + row["online_edu"]["rendered_source_text"]
        else:
            online_edu_metadata = batch.get("online_edu_metadata")
        if online_stream is None and online_edu_metadata:
            row["online_edu"] = online_edu_metadata[0]
        rows.append(row)
        del generated_tokens, generation_lengths, generation_kwargs, input_ids, attention_mask, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    rows = gather_objects(accelerator, rows)

    if accelerator.is_main_process:
        output_prefix = args.output_prefix or f"{Path(args.model_name_or_path).name}_streaming_metrics"
        metrics_path = output_dir / f"{output_prefix}_metrics.json"
        summary_path = output_dir / f"{output_prefix}_summary.json"
        summary = summarize_fn(rows)

        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved per-sample metrics to {metrics_path}")
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
