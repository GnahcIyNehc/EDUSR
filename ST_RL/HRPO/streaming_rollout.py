"""使用 vLLM 执行 AdaSR 逐 EDU 流式 rollout。"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ST_RL.HRPO.streaming_vllm_protocol import encode_streaming_request_id
from ST_RL.EDU_HRPO.counterfactual import (
    build_no_stream_probe_target_segments,
    build_stream_probe_target_segments,
)
from ST_RL.streaming_protocol import (
    STREAM_BOUNDARIES,
    encode_segments,
    flatten_segments,
)
from ST_RL.tokenizer_utils import load_streaming_tokenizer


TORCHRUN_ENV_VARIABLES = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_NAME",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
)

VLLM_WORKER_PORT_BASE = 41000
VLLM_WORKER_PORT_STRIDE = 128
@dataclass
class TrajectoryState:
    """一条轨迹在逐轮 READ-THINK 状态机中的状态。"""

    item_index: int
    completion_index: int
    source_segment_ids: List[List[int]]
    target_prefix_ids: List[int]
    expected_boundaries: List[str]
    target_prefix_segment_lengths: List[int] = field(default_factory=list)
    stage_index: int = 0
    response_ids: List[int] = field(default_factory=list)
    segment_lengths: List[int] = field(default_factory=list)
    stage_finish_reasons: List[str] = field(default_factory=list)
    boundary_matches: List[bool] = field(default_factory=list)
    stage_control_entropies: List[float] = field(default_factory=list)
    stream_probe_text: str = ""
    stream_probe_token_count: int = 0
    stream_probe_finish_reason: str = "not_run"
    stream_probe_truncated: bool = False
    finish_reason: str = ""
    done: bool = False

    @property
    def streaming_stage_count(self) -> int:
        return len(self.expected_boundaries)

    @property
    def is_deep_stage(self) -> bool:
        return self.stage_index >= self.streaming_stage_count


def parse_gpu_ids(value: str) -> List[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one rollout GPU is required.")
    return gpu_ids


def partition_prompt_items(
    items: Sequence[Dict[str, Any]],
    worker_count: int,
) -> List[List[Dict[str, Any]]]:
    """轮询分配 prompt，使长短样本尽量均匀落到各张卡。"""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    partitions: List[List[Dict[str, Any]]] = [[] for _ in range(worker_count)]
    for index, item in enumerate(items):
        copied = dict(item)
        copied["request_position"] = index
        partitions[index % worker_count].append(copied)
    return partitions


def partition_generation_waves(
    states: Sequence[TrajectoryState],
    generation_batch_size: int,
    serial_trajectories: bool = False,
    max_active_trajectories: int = 0,
) -> List[List[TrajectoryState]]:
    """按每个 prompt 的生成序号分波，限制单卡同时活跃的流式轨迹数。"""
    if generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive.")
    if max_active_trajectories < 0:
        raise ValueError("max_active_trajectories must be a nonnegative integer.")
    if not states:
        return []
    if serial_trajectories:
        return [[state] for state in states]
    maximum_completion_index = max(state.completion_index for state in states)
    base_waves = []
    for start in range(0, maximum_completion_index + 1, generation_batch_size):
        end = start + generation_batch_size
        base_waves.append(
            [
                state
                for state in states
                if start <= state.completion_index < end
            ]
        )
    if max_active_trajectories == 0:
        return base_waves

    # 验证集 G=1 时仅按生成序号无法分波，再按轨迹总数限制 FlexAttention 峰值。
    return [
        wave[start : start + max_active_trajectories]
        for wave in base_waves
        for start in range(0, len(wave), max_active_trajectories)
    ]


def retry_gpu_assignments(
    pending_ranks: Sequence[int],
    gpu_ids: Sequence[str],
    attempt: int,
) -> Dict[int, str]:
    """重试时轮换物理 GPU，避开已触发 CUDA 致命错误的上下文和编译缓存。"""
    if attempt <= 0:
        raise ValueError("attempt must be positive.")
    if not gpu_ids:
        raise ValueError("At least one rollout GPU is required.")
    return {
        worker_rank: gpu_ids[(worker_rank + attempt - 1) % len(gpu_ids)]
        for worker_rank in pending_ranks
    }


def build_worker_retry_command(
    base_command: Sequence[str],
    attempt: int,
) -> List[str]:
    """失败后切换为规避 FlexAttention 动态批缺陷的保守生成命令。"""
    if attempt <= 0:
        raise ValueError("attempt must be positive.")
    command = list(base_command)
    if attempt == 1:
        return command

    batch_option = command.index("--rollout-generation-batch-size")
    command[batch_option + 1] = "1"
    if "--enable-prefix-caching" in command:
        command[command.index("--enable-prefix-caching")] = "--no-enable-prefix-caching"
    if "--serial-trajectories" not in command:
        command.append("--serial-trajectories")

    # 首次保守重试保持原随机种子；后续重试换种子，避免确定性复现同一坏形状。
    if attempt >= 3:
        seed_option = command.index("--seed")
        command[seed_option + 1] = str(
            int(command[seed_option + 1]) + (attempt - 2) * 1000003
        )
    return command


def build_worker_environment(gpu_id: str, worker_rank: int = 0) -> Dict[str, str]:
    """隔离单卡 vLLM worker，并启用自定义 FlexAttention 输入路径。"""
    if worker_rank < 0:
        raise ValueError("worker_rank must be nonnegative.")
    environment = os.environ.copy()
    port_base = int(environment.get("ST_RL_VLLM_PORT_BASE", VLLM_WORKER_PORT_BASE))
    worker_port = port_base + worker_rank * VLLM_WORKER_PORT_STRIDE
    if not 1024 <= worker_port <= 65535:
        raise ValueError("vLLM worker port is out of range: %d" % worker_port)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    # 每个并发 worker 使用独立端口段，避免 vLLM 空闲端口探测与实际绑定之间的竞态。
    environment["VLLM_PORT"] = str(worker_port)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    environment["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
    environment["VLLM_USE_V1"] = "1"
    environment.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    environment.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), environment.get("PYTHONPATH", "")) if value
    )
    cache_root = Path(
        environment.get(
            "ST_RL_VLLM_CACHE_ROOT",
            environment.get("VLLM_CACHE_ROOT", str(Path.home() / ".cache" / "vllm")),
        )
    ).expanduser()
    worker_cache = cache_root / ("streaming-worker-gpu-%s" % gpu_id)
    environment["VLLM_CACHE_ROOT"] = str(worker_cache)
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(worker_cache / "torchinductor")
    environment["TRITON_CACHE_DIR"] = str(worker_cache / "triton")
    for variable in TORCHRUN_ENV_VARIABLES:
        environment.pop(variable, None)
    return environment


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False)
    temporary_path.replace(output_path)


def build_hf_overrides(model_path: str) -> Dict[str, Any]:
    """把自定义 HF 架构名映射为 vLLM 原生 Qwen3 架构。"""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return {}
    config = load_json(str(config_path))
    if config.get("model_type") != "qwen3":
        return {}
    if config.get("architectures") == ["Qwen3ForCausalLM"]:
        return {}
    return {"architectures": ["Qwen3ForCausalLM"]}


def _single_token_ids(tokenizer: Any, values: Sequence[str]) -> Tuple[List[int], List[str]]:
    token_ids: List[int] = []
    strings: List[str] = []
    for value in values:
        if not value:
            continue
        encoded = [int(item) for item in tokenizer(value, add_special_tokens=False)["input_ids"]]
        if len(encoded) == 1:
            if encoded[0] not in token_ids:
                token_ids.append(encoded[0])
        elif value not in strings:
            strings.append(value)
    return token_ids, strings


def _ends_with_marker(text: str, markers: Sequence[str]) -> str:
    normalized = str(text or "").rstrip()
    for marker in markers:
        if marker and normalized.endswith(marker):
            return marker
    return ""


def _split_token_segments(
    token_ids: Sequence[int],
    segment_lengths: Sequence[int],
) -> List[List[int]]:
    """按已记录长度恢复不参与采样的 target prefix 分段。"""
    normalized_ids = [int(token_id) for token_id in token_ids]
    normalized_lengths = [int(length) for length in segment_lengths]
    if not normalized_lengths:
        return [normalized_ids]
    if any(length <= 0 for length in normalized_lengths):
        raise ValueError("All target prefix segment lengths must be positive.")
    if sum(normalized_lengths) != len(normalized_ids):
        raise ValueError("Target prefix segment lengths do not sum to the token count.")
    segments = []
    cursor = 0
    for length in normalized_lengths:
        segments.append(normalized_ids[cursor : cursor + length])
        cursor += length
    return segments


def _visible_source_count(state: TrajectoryState) -> int:
    if state.is_deep_stage:
        return len(state.source_segment_ids)
    bootstrap_count = max(len(state.target_prefix_segment_lengths) - 1, 0)
    return min(
        state.stage_index + bootstrap_count + 2,
        len(state.source_segment_ids),
    )


def _build_round_prompt_ids(
    state: TrajectoryState,
) -> Tuple[List[int], List[int], List[int]]:
    """固定完整 source 前缀，只在元数据中逐阶段扩大 target 可见范围。"""
    source_lengths = [len(segment) for segment in state.source_segment_ids]
    source_ids = flatten_segments(state.source_segment_ids)
    target_lengths = (
        list(state.target_prefix_segment_lengths)
        if state.target_prefix_segment_lengths
        else [len(state.target_prefix_ids)]
    ) + list(state.segment_lengths)
    expected_visible_count = min(
        len(target_lengths) + 1,
        len(state.source_segment_ids),
    )
    if _visible_source_count(state) != expected_visible_count:
        raise RuntimeError("Streaming stage and target segment counts differ; source visibility cannot be safely advanced.")
    prompt_ids = source_ids + state.target_prefix_ids + state.response_ids
    return source_lengths, target_lengths, prompt_ids


def _finalize_aborted_state(state: TrajectoryState, finish_reason: str) -> None:
    """补齐零长度阶段，维持 actor 侧固定的分段契约。"""
    state.finish_reason = finish_reason
    while len(state.segment_lengths) < state.streaming_stage_count + 1:
        state.segment_lengths.append(0)
        state.stage_finish_reasons.append("not_run")
    while len(state.stage_control_entropies) < state.streaming_stage_count:
        state.stage_control_entropies.append(0.0)
    state.done = True


def _make_sampling_params(
    args: argparse.Namespace,
    tokenizer: Any,
    state: TrajectoryState,
    maximum_tokens: int,
    seed: int,
):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    if state.is_deep_stage:
        stop_values = ["<EOA>", "<|im_end|>"]
    else:
        # 早到的终止符也必须让 vLLM 停下，随后由状态机判为协议错误。
        stop_values = list(STREAM_BOUNDARIES) + ["<EOA>", "<|im_end|>"]
    if tokenizer.eos_token:
        stop_values.append(str(tokenizer.eos_token))
    stop_token_ids, stop_strings = _single_token_ids(tokenizer, stop_values)
    params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=max(int(maximum_tokens), 1),
        stop=stop_strings,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        seed=int(seed),
        logprobs=(
            20
            if args.counterfactual_probes
            and args.enable_edu_controller
            and not state.is_deep_stage
            else None
        ),
    )
    params.output_kind = RequestOutputKind.FINAL_ONLY
    return params


def _add_round_request(
    llm: Any,
    tokenizer: Any,
    state: TrajectoryState,
    args: argparse.Namespace,
    request_serial: int,
) -> Tuple[str, bool]:
    from vllm.inputs import TokensPrompt

    source_lengths, target_lengths, prompt_ids = _build_round_prompt_ids(state)
    total_room = args.max_model_len - len(prompt_ids)
    total_budget = args.max_new_tokens - len(state.response_ids)
    stage_budget = args.max_deep_tokens if state.is_deep_stage else args.max_streaming_tokens_per_edu
    maximum_tokens = min(total_room, total_budget, stage_budget)
    if maximum_tokens <= 0:
        state.segment_lengths.append(0)
        state.stage_finish_reasons.append("length")
        _finalize_aborted_state(state, "length")
        return "", False

    unique_id = "%d-%d-%d-%d" % (
        state.item_index,
        state.completion_index,
        state.stage_index,
        request_serial,
    )
    request_id = encode_streaming_request_id(unique_id, source_lengths, target_lengths)
    params = _make_sampling_params(
        args,
        tokenizer,
        state,
        maximum_tokens,
        args.seed + request_serial * 17 + state.completion_index,
    )
    llm.llm_engine.add_request(
        request_id,
        TokensPrompt(prompt_token_ids=prompt_ids),
        params,
    )
    return request_id, True


def _consume_stage_output(
    state: TrajectoryState,
    request_output: Any,
    tokenizer: Any,
) -> None:
    completion = request_output.outputs[0]
    token_ids = [int(value) for value in completion.token_ids]
    stage_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    state.response_ids.extend(token_ids)
    state.segment_lengths.append(len(token_ids))

    if state.is_deep_stage:
        terminal = _ends_with_marker(stage_text, ("<EOA>", "<|im_end|>"))
        if terminal:
            state.stage_finish_reasons.append("stop")
            state.finish_reason = "stop"
        else:
            state.stage_finish_reasons.append("length")
            state.finish_reason = "length"
        state.done = True
        return

    entropy = 0.0
    logprob_rows = getattr(completion, "logprobs", None) or []
    if logprob_rows and logprob_rows[0]:
        log_values = []
        for value in logprob_rows[0].values():
            current = getattr(value, "logprob", value)
            try:
                log_values.append(float(current))
            except (TypeError, ValueError):
                continue
        if len(log_values) > 1:
            maximum = max(log_values)
            weights = [math.exp(value - maximum) for value in log_values]
            total = sum(weights)
            probabilities = [value / total for value in weights]
            entropy = -sum(
                probability * math.log(max(probability, 1e-12))
                for probability in probabilities
            ) / math.log(len(probabilities))
    state.stage_control_entropies.append(min(max(entropy, 0.0), 1.0))

    stream_boundary = _ends_with_marker(stage_text, STREAM_BOUNDARIES)
    expected_boundary = state.expected_boundaries[state.stage_index]
    if stream_boundary:
        matched = stream_boundary == expected_boundary
        state.boundary_matches.append(matched)
        state.stage_finish_reasons.append("stop" if matched else "stop_wrong_boundary")
        state.stage_index += 1
        return

    early_terminal = _ends_with_marker(stage_text, ("<EOA>", "<|im_end|>"))
    state.boundary_matches.append(False)
    state.stage_finish_reasons.append("invalid_stop" if early_terminal else "length")
    _finalize_aborted_state(state, "invalid_stop" if early_terminal else "length")


def _build_output_items(
    items: Sequence[Dict[str, Any]],
    states: Sequence[TrajectoryState],
    tokenizer: Any,
    no_stream_probe_by_item: Dict[int, Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    no_stream_probe_by_item = no_stream_probe_by_item or {}
    output_items = []
    for item_index, item in enumerate(items):
        item_states = sorted(
            (state for state in states if state.item_index == item_index),
            key=lambda value: value.completion_index,
        )
        source_segment_ids = item_states[0].source_segment_ids
        target_prefix_ids = item_states[0].target_prefix_ids
        target_prefix_segment_lengths = item_states[0].target_prefix_segment_lengths
        completions = []
        for state in item_states:
            completions.append(
                {
                    "index": state.completion_index,
                    "token_ids": state.response_ids,
                    "text": tokenizer.decode(
                        state.response_ids,
                        skip_special_tokens=False,
                        spaces_between_special_tokens=False,
                    ),
                    "finish_reason": state.finish_reason or "stop",
                    "stage_finish_reasons": state.stage_finish_reasons,
                    "boundary_matches": state.boundary_matches,
                    "stage_control_entropies": state.stage_control_entropies,
                    "response_segment_token_lengths": state.segment_lengths,
                    "stream_probe_text": state.stream_probe_text,
                    "stream_probe_token_count": state.stream_probe_token_count,
                    "stream_probe_finish_reason": state.stream_probe_finish_reason,
                    "stream_probe_truncated": state.stream_probe_truncated,
                }
            )
        no_stream_probe = no_stream_probe_by_item.get(item_index, {})
        output_items.append(
            {
                "request_position": int(item["request_position"]),
                "sample_index": int(item["sample_index"]),
                "source_segments": [str(segment) for segment in item["source_segments"]],
                "source_segment_token_ids": source_segment_ids,
                "target_prefix_token_ids": target_prefix_ids,
                "target_prefix_segment_lengths": target_prefix_segment_lengths,
                "question_segments": int(item["question_segments"]),
                "expected_boundaries": list(item["expected_boundaries"]),
                "prompt_protocol": item.get("prompt_protocol") or {},
                "no_stream_probe_text": str(no_stream_probe.get("text") or ""),
                "no_stream_probe_token_count": int(
                    no_stream_probe.get("token_count", 0)
                ),
                "no_stream_probe_finish_reason": str(
                    no_stream_probe.get("finish_reason") or "not_run"
                ),
                "no_stream_probe_truncated": bool(
                    no_stream_probe.get("truncated", False)
                ),
                "completions": completions,
            }
        )
    return output_items


def _probe_sampling_params(
    tokenizer: Any,
    maximum_tokens: int,
    seed: int,
    stop_at_closing_brace: bool = False,
):
    """构造确定性的短答案或无流式反事实采样参数。"""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    stop_values = ["<EOA>", "<|im_end|>"]
    if tokenizer.eos_token:
        stop_values.append(str(tokenizer.eos_token))
    stop_token_ids, stop_strings = _single_token_ids(tokenizer, stop_values)
    if stop_at_closing_brace and "}" not in stop_strings:
        # 强制保留右花括号，便于奖励侧区分完整答案与截断输出。
        stop_strings.append("}")
    params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max(int(maximum_tokens), 1),
        stop=stop_strings,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        seed=int(seed),
    )
    params.output_kind = RequestOutputKind.FINAL_ONLY
    return params


def _run_counterfactual_probes(
    llm: Any,
    tokenizer: Any,
    items: Sequence[Dict[str, Any]],
    states: Sequence[TrajectoryState],
    args: argparse.Namespace,
) -> Dict[int, Dict[str, Any]]:
    """每条轨迹生成短答案探针，并为每个 prompt 生成一个共享 no-stream 探针。"""
    from vllm.inputs import TokensPrompt

    request_targets: Dict[str, Tuple[str, int, int, str]] = {}
    state_by_key = {
        (state.item_index, state.completion_index): state for state in states
    }
    serial = 0
    no_stream_probe_by_item: Dict[int, Dict[str, Any]] = {}

    def drain_requests() -> None:
        """等待当前探针完成；保守恢复模式用它保证单请求执行。"""
        while request_targets:
            step_outputs = llm.llm_engine.step()
            if not step_outputs and not llm.llm_engine.has_unfinished_requests():
                raise RuntimeError(
                    "vLLM stopped early with %d counterfactual probes pending." % len(request_targets)
                )
            for output in step_outputs:
                if not output.finished:
                    continue
                request_id = str(output.request_id)
                target = request_targets.pop(request_id, None)
                if target is None:
                    raise RuntimeError("vLLM returned an unknown counterfactual request: %s." % request_id)
                kind, item_index, completion_index, answer_prefix = target
                completion = output.outputs[0]
                token_ids = [int(value) for value in completion.token_ids]
                text = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
                stop_reason = str(getattr(completion, "stop_reason", "") or "")
                if answer_prefix and stop_reason == "}" and not text.rstrip().endswith("}"):
                    # 兼容不回传 stop string 的 vLLM 小版本，保持 boxed 结构完整。
                    text += "}"
                finish_reason = str(getattr(completion, "finish_reason", "") or "stop")
                truncated = finish_reason == "length"
                if kind == "stream":
                    state = state_by_key[(item_index, completion_index)]
                    state.stream_probe_text = answer_prefix + text
                    state.stream_probe_token_count = len(token_ids)
                    state.stream_probe_finish_reason = finish_reason
                    state.stream_probe_truncated = truncated
                else:
                    no_stream_probe_by_item[item_index] = {
                        "text": text,
                        "token_count": len(token_ids),
                        "finish_reason": finish_reason,
                        "truncated": truncated,
                    }

    def enqueue(
        kind: str,
        item_index: int,
        completion_index: int,
        source_segment_ids: List[List[int]],
        target_segment_ids: List[List[int]],
        maximum_new_tokens: int,
        answer_prefix: str = "",
    ) -> None:
        nonlocal serial
        source_ids = flatten_segments(source_segment_ids)
        target_ids = flatten_segments(target_segment_ids)
        prompt_ids = source_ids + target_ids
        maximum_tokens = min(
            int(maximum_new_tokens),
            int(args.max_model_len) - len(prompt_ids),
        )
        if maximum_tokens <= 0:
            if kind == "stream":
                state = state_by_key[(item_index, completion_index)]
                state.stream_probe_text = ""
                state.stream_probe_finish_reason = "prompt_length"
            else:
                no_stream_probe_by_item[item_index] = {
                    "text": "",
                    "token_count": 0,
                    "finish_reason": "prompt_length",
                    "truncated": False,
                }
            return
        serial += 1
        unique_id = "probe-%s-%d-%d-%d" % (
            kind,
            item_index,
            completion_index,
            serial,
        )
        request_id = encode_streaming_request_id(
            unique_id,
            [len(segment) for segment in source_segment_ids],
            [len(segment) for segment in target_segment_ids],
        )
        params = _probe_sampling_params(
            tokenizer,
            maximum_tokens,
            args.seed + 700000 + serial * 31,
            stop_at_closing_brace=bool(answer_prefix),
        )
        llm.llm_engine.add_request(
            request_id,
            TokensPrompt(prompt_token_ids=prompt_ids),
            params,
        )
        request_targets[request_id] = (
            kind,
            item_index,
            completion_index,
            answer_prefix,
        )
        if args.serial_trajectories:
            drain_requests()

    states_by_item: Dict[int, List[TrajectoryState]] = {}
    for state in states:
        states_by_item.setdefault(state.item_index, []).append(state)
        streaming_lengths = state.segment_lengths[: state.streaming_stage_count]
        if len(streaming_lengths) != state.streaming_stage_count or any(
            length <= 0 for length in streaming_lengths
        ):
            # 不完整流式轨迹没有可解释的 Q_stream 状态，留空并由奖励侧按失败处理。
            state.stream_probe_text = ""
            state.stream_probe_finish_reason = "invalid_stream_state"
            continue
        streaming_token_count = sum(streaming_lengths)
        streaming_ids = state.response_ids[:streaming_token_count]
        stream_state_segments = []
        offset = 0
        for length in streaming_lengths:
            stream_state_segments.append(streaming_ids[offset : offset + length])
            offset += length
        # 流式视图直接关闭 think，让模型只依据已产生的流式状态回答。
        reasoning_connect_3 = str(items[state.item_index].get("reasoning_connect_3") or "")
        stream_state_text = [tokenizer.decode(segment, skip_special_tokens=False) for segment in stream_state_segments]
        answer_prefix = "\\boxed{" if args.task_type != "generic" else ""
        probe_segments = build_stream_probe_target_segments(
            stream_state_text,
            reasoning_connect_3,
            answer_prefix=answer_prefix,
        )
        target_segments = _split_token_segments(
            state.target_prefix_ids,
            state.target_prefix_segment_lengths,
        )
        target_segments.extend(stream_state_segments)
        target_segments.extend(encode_segments(tokenizer, probe_segments[-1:]))
        enqueue(
            "stream",
            state.item_index,
            state.completion_index,
            state.source_segment_ids,
            target_segments,
            args.stream_probe_max_tokens,
            answer_prefix=answer_prefix,
        )

    for item_index, item in enumerate(items):
        item_states = states_by_item[item_index]
        source_segment_ids = item_states[0].source_segment_ids
        no_stream_marker = "<skip>" if args.enable_edu_controller else "<ignore>"
        # 无流式视图跳过浅层思考后进入原 SFT 的深度推理连接段。
        reasoning_connect_2 = str(item.get("reasoning_connect_2") or "")
        probe_segments = build_no_stream_probe_target_segments(
            item["expected_boundaries"],
            reasoning_connect_2,
            skip_token=no_stream_marker,
        )
        target_segments = _split_token_segments(
            item_states[0].target_prefix_ids,
            item_states[0].target_prefix_segment_lengths,
        ) + encode_segments(
            tokenizer,
            probe_segments,
        )
        enqueue(
            "no_stream",
            item_index,
            -1,
            source_segment_ids,
            target_segments,
            args.no_stream_probe_max_tokens,
        )

    drain_requests()

    return no_stream_probe_by_item


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from vllm import LLM

    request = load_json(args.request_file)
    items = request.get("items") or []
    if not items:
        write_json_atomic(args.output_file, {"items": []})
        return

    tokenizer, tokenizer_info = load_streaming_tokenizer(args.model_path)
    tokenizer_path = args.model_path
    if tokenizer_info.added_tokens:
        runtime_tokenizer_dir = Path(args.output_file).with_name(
            Path(args.output_file).stem + "-tokenizer"
        )
        if runtime_tokenizer_dir.exists():
            shutil.rmtree(runtime_tokenizer_dir)
        tokenizer.save_pretrained(str(runtime_tokenizer_dir))
        tokenizer_path = str(runtime_tokenizer_dir)
    print(
        "streaming_tokenizer_ready source=%s size=%d token_set=%s added=%d ids=%s"
        % (
            args.model_path,
            tokenizer_info.vocab_size,
            tokenizer_info.token_set,
            tokenizer_info.added_tokens,
            json.dumps(tokenizer_info.token_ids, ensure_ascii=True, sort_keys=True),
        ),
        flush=True,
    )
    llm = LLM(
        model=args.model_path,
        tokenizer=tokenizer_path,
        hf_overrides=build_hf_overrides(args.model_path),
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_fraction,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        # source 前缀使用普通因果状态，可在同题 G 条轨迹和不同轮次之间复用。
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=True,
        disable_log_stats=False,
        worker_cls="ST_RL.HRPO.streaming_vllm_worker.StreamingWorker",
        seed=args.seed,
    )

    states: List[TrajectoryState] = []
    for item_index, item in enumerate(items):
        source_segment_ids = encode_segments(tokenizer, [str(value) for value in item["source_segments"]])
        target_prefix = str(item.get("target_prefix") or "")
        if not target_prefix:
            raise ValueError("The rollout request is missing target_prefix; consistency with the SFT prompt template cannot be ensured.")
        prefix_text_segments = [target_prefix] + [
            str(value) for value in item.get("bootstrap_target_segments") or []
        ]
        target_prefix_token_segments = encode_segments(tokenizer, prefix_text_segments)
        target_prefix_ids = flatten_segments(target_prefix_token_segments)
        target_prefix_segment_lengths = [
            len(segment) for segment in target_prefix_token_segments
        ]
        for completion_index in range(args.num_generations):
            states.append(
                TrajectoryState(
                    item_index=item_index,
                    completion_index=completion_index,
                    source_segment_ids=source_segment_ids,
                    target_prefix_ids=target_prefix_ids,
                    expected_boundaries=[str(value) for value in item["expected_boundaries"]],
                    target_prefix_segment_lengths=target_prefix_segment_lengths,
                )
            )

    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    request_serial = 0
    request_to_state: Dict[str, TrajectoryState] = {}

    def enqueue(state: TrajectoryState) -> None:
        nonlocal request_serial
        request_serial += 1
        request_id, added = _add_round_request(
            llm,
            tokenizer,
            state,
            args,
            request_serial,
        )
        if added:
            request_to_state[request_id] = state

    completed_stages = 0
    last_reported_finished = -1
    generation_waves = partition_generation_waves(
        states,
        args.rollout_generation_batch_size,
        serial_trajectories=args.serial_trajectories,
        max_active_trajectories=args.max_active_trajectories,
    )
    for wave_index, wave_states in enumerate(generation_waves, start=1):
        for state in wave_states:
            enqueue(state)
        print(
            "streaming_vllm_generation_wave_start wave=%d/%d trajectories=%d active=%d"
            % (wave_index, len(generation_waves), len(wave_states), len(request_to_state)),
            flush=True,
        )
        while request_to_state:
            step_outputs = llm.llm_engine.step()
            if not step_outputs and not llm.llm_engine.has_unfinished_requests():
                raise RuntimeError(
                    "vLLM stopped early with %d streaming stages pending."
                    % len(request_to_state)
                )
            requeue_states = []
            for output in step_outputs:
                if not output.finished:
                    continue
                request_id = str(output.request_id)
                state = request_to_state.pop(request_id, None)
                if state is None:
                    raise RuntimeError("vLLM returned an unknown streaming request: %s." % request_id)
                _consume_stage_output(state, output, tokenizer)
                completed_stages += 1
                if not state.done:
                    requeue_states.append(state)
            # 某条轨迹到达边界后立即追加下一 EDU，不等待同波中的最长轨迹。
            for state in requeue_states:
                enqueue(state)
            finished = sum(state.done for state in states)
            if finished != last_reported_finished and (
                finished == len(states)
                or completed_stages % max(len(states), 1) == 0
            ):
                last_reported_finished = finished
                print(
                    "streaming_vllm_progress completed_stages=%d active=%d finished=%d/%d"
                    % (completed_stages, len(request_to_state), finished, len(states)),
                    flush=True,
                )
        print(
            "streaming_vllm_generation_wave_complete wave=%d/%d finished=%d/%d"
            % (
                wave_index,
                len(generation_waves),
                sum(state.done for state in states),
                len(states),
            ),
            flush=True,
        )

    unfinished = sum(not state.done for state in states)
    if unfinished:
        raise RuntimeError("The streaming state machine exited with %d unfinished trajectories." % unfinished)

    no_stream_probe_by_item = {}
    if args.counterfactual_probes:
        probe_started_at = time.perf_counter()
        no_stream_probe_by_item = _run_counterfactual_probes(
            llm,
            tokenizer,
            items,
            states,
            args,
        )
        print(
            "counterfactual_probes_complete stream=%d no_stream=%d seconds=%.1f"
            % (
                len(states),
                len(no_stream_probe_by_item),
                time.perf_counter() - probe_started_at,
            ),
            flush=True,
        )
    output_items = _build_output_items(
        items,
        states,
        tokenizer,
        no_stream_probe_by_item=no_stream_probe_by_item,
    )
    elapsed = time.perf_counter() - started_at
    lengths = [len(state.response_ids) for state in states]
    length_sorted = sorted(lengths)
    p95_index = max(math.ceil(len(length_sorted) * 0.95) - 1, 0)
    termination_stats = {
        "trajectories": len(states),
        "stop": sum(state.finish_reason == "stop" for state in states),
        "length": sum(state.finish_reason == "length" for state in states),
        "invalid_stop": sum(state.finish_reason == "invalid_stop" for state in states),
        "average_tokens": sum(lengths) / max(len(lengths), 1),
        "p95_tokens": length_sorted[p95_index] if length_sorted else 0,
        "max_tokens": max(lengths + [0]),
        "boundary_match_count": sum(sum(state.boundary_matches) for state in states),
        "boundary_count": sum(len(state.boundary_matches) for state in states),
    }
    termination_stats["boundary_match_rate"] = termination_stats["boundary_match_count"] / max(
        termination_stats["boundary_count"], 1
    )
    write_json_atomic(
        args.output_file,
        {
            "items": output_items,
            "worker_seconds": elapsed,
            "worker_seed": args.seed,
            "worker_gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "worker_gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            "worker_prefix_caching": bool(args.enable_prefix_caching),
            "worker_serial_trajectories": bool(args.serial_trajectories),
            "worker_generation_batch_size": int(args.rollout_generation_batch_size),
            "termination_stats": termination_stats,
        },
    )
    print(
        "streaming_vllm_worker_complete trajectories=%d seconds=%.1f avg_tokens=%.1f p95_tokens=%d length=%d invalid_stop=%d"
        % (
            len(states),
            elapsed,
            termination_stats["average_tokens"],
            termination_stats["p95_tokens"],
            termination_stats["length"],
            termination_stats["invalid_stop"],
        ),
        flush=True,
    )


def run_coordinator(args: argparse.Namespace) -> None:
    request = load_json(args.request_file)
    items = request.get("items") or []
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    active_worker_count = min(len(gpu_ids), max(len(items), 1))
    partitions = partition_prompt_items(items, active_worker_count)
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    worker_outputs = []
    worker_specs = []
    started_at = time.perf_counter()
    for worker_rank, partition in enumerate(partitions):
        worker_request = work_dir / ("request-%02d.json" % worker_rank)
        worker_output = work_dir / ("output-%02d.json" % worker_rank)
        write_json_atomic(str(worker_request), {"items": partition})
        worker_outputs.append(worker_output)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model-path",
            args.model_path,
            "--request-file",
            str(worker_request),
            "--output-file",
            str(worker_output),
            "--num-generations",
            str(args.num_generations),
            "--rollout-generation-batch-size",
            str(args.rollout_generation_batch_size),
            "--max-active-trajectories",
            str(args.max_active_trajectories),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-streaming-tokens-per-edu",
            str(args.max_streaming_tokens_per_edu),
            "--max-deep-tokens",
            str(args.max_deep_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--enable-prefix-caching"
            if args.enable_prefix_caching
            else "--no-enable-prefix-caching",
            "--gpu-memory-fraction",
            str(args.gpu_memory_fraction),
            "--dtype",
            args.dtype,
            "--task-type",
            args.task_type,
            "--seed",
            str(args.seed + worker_rank * 100000),
        ]
        # 可显式选用已通过 smoke 的串行路径，避免每步先失败再降级。
        if args.serial_trajectories or os.environ.get("ST_RL_SERIAL_TRAJECTORIES", "0") == "1":
            command.append("--serial-trajectories")
        if args.counterfactual_probes:
            command.extend(
                [
                    "--counterfactual-probes",
                    "--stream-probe-max-tokens",
                    str(args.stream_probe_max_tokens),
                    "--no-stream-probe-max-tokens",
                    str(args.no_stream_probe_max_tokens),
                ]
            )
        if args.enable_edu_controller:
            command.append("--enable-edu-controller")
        worker_environment = build_worker_environment(
            gpu_ids[worker_rank],
            worker_rank=worker_rank,
        )
        print(
            "streaming_vllm_worker_start rank=%d gpu=%s prompts=%d trajectories=%d vllm_port=%s"
            % (
                worker_rank,
                gpu_ids[worker_rank],
                len(partition),
                len(partition) * args.num_generations,
                worker_environment["VLLM_PORT"],
            ),
            flush=True,
        )
        worker_specs.append(
            {
                "rank": worker_rank,
                "command": command,
                "output": worker_output,
            }
        )

    max_attempts = max(
        int(os.environ.get("ST_RL_ROLLOUT_WORKER_MAX_ATTEMPTS", "4")),
        1,
    )
    pending_ranks = list(range(len(worker_specs)))
    failures = []
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            retry_delay_seconds = max(
                float(os.environ.get("ST_RL_ROLLOUT_WORKER_RETRY_DELAY_SECONDS", "5")),
                0.0,
            )
            if retry_delay_seconds:
                time.sleep(retry_delay_seconds)
        assigned_gpus = retry_gpu_assignments(pending_ranks, gpu_ids, attempt)
        processes = []
        for worker_rank in pending_ranks:
            spec = worker_specs[worker_rank]
            output_path = spec["output"]
            if output_path.exists():
                output_path.unlink()
            retry_command = build_worker_retry_command(spec["command"], attempt)
            if attempt > 1:
                print(
                    "streaming_vllm_worker_retry rank=%d gpu=%s attempt=%d/%d mode=serial_no_prefix_cache seed=%s"
                    % (
                        worker_rank,
                        assigned_gpus[worker_rank],
                        attempt,
                        max_attempts,
                        retry_command[retry_command.index("--seed") + 1],
                    ),
                    flush=True,
                )
            worker_environment = build_worker_environment(
                assigned_gpus[worker_rank],
                worker_rank=worker_rank,
            )
            processes.append(
                (
                    worker_rank,
                    subprocess.Popen(retry_command, env=worker_environment),
                )
            )
        failures = []
        for worker_rank, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((worker_rank, return_code))
        if not failures:
            break
        pending_ranks = [worker_rank for worker_rank, _ in failures]
    if failures:
        raise RuntimeError(
            "The streaming vLLM worker failed after %d attempts: %s"
            % (max_attempts, failures)
        )

    merged_items = []
    worker_seconds = []
    worker_gpu_peaks = []
    worker_recovery_modes = []
    termination_rows = []
    for output_path in worker_outputs:
        result = load_json(str(output_path))
        merged_items.extend(result.get("items") or [])
        worker_seconds.append(float(result.get("worker_seconds", 0.0)))
        worker_gpu_peaks.append(
            {
                "allocated_gib": float(result.get("worker_gpu_peak_allocated_gib", 0.0)),
                "reserved_gib": float(result.get("worker_gpu_peak_reserved_gib", 0.0)),
            }
        )
        worker_recovery_modes.append(
            {
                "prefix_caching": bool(result.get("worker_prefix_caching", True)),
                "serial_trajectories": bool(result.get("worker_serial_trajectories", False)),
                "generation_batch_size": int(
                    result.get("worker_generation_batch_size", args.rollout_generation_batch_size)
                ),
            }
        )
        termination_rows.append(result.get("termination_stats") or {})
    merged_items.sort(key=lambda value: int(value["request_position"]))
    for item in merged_items:
        if len(item["completions"]) != args.num_generations:
            raise RuntimeError(
                "sample_index=%s generated only %d/%d trajectories."
                % (item["sample_index"], len(item["completions"]), args.num_generations)
            )

    total_trajectories = sum(
        len(item["completions"])
        for item in merged_items
    )
    total_termination_trajectories = sum(
        int(row.get("trajectories", 0)) for row in termination_rows
    )
    termination_stats = {
        key: sum(float(row.get(key, 0.0)) for row in termination_rows)
        for key in ("stop", "length", "invalid_stop")
    }
    termination_stats.update(
        {
            "trajectories": total_termination_trajectories,
            "average_tokens": sum(
                float(row.get("average_tokens", 0.0)) * int(row.get("trajectories", 0))
                for row in termination_rows
            )
            / max(total_termination_trajectories, 1),
            "p95_tokens_max_across_workers": max(
                (int(row.get("p95_tokens", 0)) for row in termination_rows),
                default=0,
            ),
            "max_tokens": max(
                (int(row.get("max_tokens", 0)) for row in termination_rows),
                default=0,
            ),
            "boundary_match_count": sum(
                int(row.get("boundary_match_count", 0)) for row in termination_rows
            ),
            "boundary_count": sum(int(row.get("boundary_count", 0)) for row in termination_rows),
        }
    )
    termination_stats["boundary_match_rate"] = termination_stats[
        "boundary_match_count"
    ] / max(termination_stats["boundary_count"], 1)
    write_json_atomic(
        args.output_file,
        {
            "items": merged_items,
            "rollout_seconds": time.perf_counter() - started_at,
            "worker_seconds": worker_seconds,
            "worker_gpu_peaks": worker_gpu_peaks,
            "worker_recovery_modes": worker_recovery_modes,
            "gpu_ids": gpu_ids[:active_worker_count],
            "rollout_backend": "streaming_vllm_full_source_cached_v2",
            "source_prefill_mode": "full_source_once",
            "visibility_cache_mode": "request_lru",
            "rollout_generation_batch_size": args.rollout_generation_batch_size,
            "termination_stats": termination_stats,
        },
    )
    print(
        "streaming_vllm_rollout_complete prompts=%d trajectories=%d seconds=%.1f stop=%d length=%d invalid_stop=%d"
        % (
            len(merged_items),
            total_trajectories,
            time.perf_counter() - started_at,
            termination_stats["stop"],
            termination_stats["length"],
            termination_stats["invalid_stop"],
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDU-level streaming rollouts with vLLM for AdaSR.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--num-generations", type=int, default=12)
    parser.add_argument("--rollout-generation-batch-size", type=int, default=12)
    parser.add_argument(
        "--max-active-trajectories",
        type=int,
        default=int(os.environ.get("ST_RL_ROLLOUT_MAX_ACTIVE_TRAJECTORIES", "0")),
    )
    parser.add_argument("--serial-trajectories", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-streaming-tokens-per-edu", type=int, default=512)
    parser.add_argument("--max-deep-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.62)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--task-type",
        choices=("generic", "extractive", "classification", "math"),
        default="generic",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counterfactual-probes", action="store_true")
    parser.add_argument("--enable-edu-controller", action="store_true")
    parser.add_argument("--stream-probe-max-tokens", type=int, default=16)
    parser.add_argument("--no-stream-probe-max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_streaming_tokens_per_edu <= 0 or args.max_deep_tokens <= 0:
        raise ValueError("Stage token limits must be positive.")
    if args.max_new_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("Total generation and model lengths must be positive.")
    if args.num_generations <= 0 or args.rollout_generation_batch_size <= 0:
        raise ValueError("Generation count and batch size must be positive.")
    if args.max_active_trajectories < 0:
        raise ValueError("Maximum active trajectories must be a nonnegative integer.")
    if args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError("vLLM batching parameters must be positive.")
    if args.stream_probe_max_tokens <= 0 or args.no_stream_probe_max_tokens <= 0:
        raise ValueError("Counterfactual probe token limits must be positive.")
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0, 1].")
    if not args.worker and not args.work_dir:
        args.work_dir = str(Path(args.output_file).with_suffix("")) + "-workers"
    if args.worker:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
