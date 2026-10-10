"""使用逐 EDU 流式 rollout 和 FSDP actor 训练 AdaSR-HRPO。"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STREAMING_ROOT = ROOT / "StreamingThinker"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from ST_RL.HRPO import HRPOConfig, compute_hrpo_advantages, hrpo_loss
from ST_RL.HRPO.rewards import deep_format_reward, exact_match_accuracy_reward
from ST_RL.HRPO.train_hrpo import (
    binary_group_reward_stats,
    compose_reporting_rewards,
    sft_streaming_format_reward,
)
from ST_RL.common import phase_masks_from_lengths
from ST_RL.epoch_shuffle_sampler import EpochShuffleSampler
from ST_RL.prompt_protocol import PromptProtocol, load_training_prompt_protocol
from ST_RL.streaming_protocol import (
    build_streaming_source,
    compute_streaming_response_statistics,
)
from ST_RL.training_utils import (
    RLSample,
    count_question_stream_segments,
    load_samples,
    parse_hrpo_response,
    set_seed,
)
from ST_RL.tokenizer_utils import load_streaming_tokenizer


@dataclass
class PreparedGroup:
    """一个 prompt 的 G 条轨迹及其 HRPO 奖励信息。"""

    sample_index: int
    source_segment_ids: List[List[int]]
    target_prefix_ids: List[int]
    target_prefix_segment_lengths: List[int]
    response_ids: List[List[int]]
    response_segment_length_rows: List[List[int]]
    response_texts: List[str]
    streaming_format_rewards: List[float]
    deep_format_rewards: List[float]
    accuracy_rewards: List[float]
    segment_length_rows: List[List[int]]
    streaming_token_lengths: List[int]
    deep_lengths: List[int]
    finish_reasons: List[str]
    max_segments: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaSR-HRPO with EDU-level rollouts and FSDP.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--prompt-config", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--boundary-mode", choices=("sentence", "edu"), default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--resume-state-dir", default="")
    parser.add_argument("--resume-state-from", default="")
    parser.add_argument("--resume-state-save-interval", type=int, default=0)
    parser.add_argument("--periodic-model-save-dir", default="")
    parser.add_argument("--periodic-model-save-start-step", type=int, default=0)
    parser.add_argument("--periodic-model-save-interval", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--scheduler-total-steps", type=int, default=0)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=12)
    parser.add_argument("--rollout-generation-batch-size", type=int, default=6)
    parser.add_argument("--rollout-max-active-trajectories", type=int, default=0)
    parser.add_argument("--policy-micro-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-streaming-tokens-per-edu", type=int, default=512)
    parser.add_argument("--max-deep-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--rollout-max-num-seqs", type=int, default=64)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--rollout-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--local-weight", type=float, default=0.05)
    parser.add_argument("--accuracy-weight", type=float, default=1.0)
    parser.add_argument("--format-weight", type=float, default=2.0)
    parser.add_argument("--length-weight", type=float, default=0.05)
    parser.add_argument(
        "--accuracy-reward-mode",
        choices=("boxed", "permissive"),
        default="boxed",
        help="boxed requires a boxed final answer; permissive also accepts a trailing-number fallback.",
    )
    parser.add_argument("--latency-alpha", type=float, default=0.5)
    parser.add_argument("--reward-tau", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rollout-gpu-ids", default="0,1,2,3")
    parser.add_argument("--rollout-gpu-memory-fraction", type=float, default=0.62)
    parser.add_argument("--rollout-python", default=sys.executable)
    parser.add_argument(
        "--rollout-script",
        default=str(Path(__file__).resolve().parent / "streaming_rollout.py"),
    )
    parser.add_argument("--metrics-file", default="")
    parser.add_argument("--compact-log-file", default="")
    parser.add_argument("--validation-file", default="")
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument("--validation-start-step", type=int, default=0)
    parser.add_argument("--skip-initial-validation", action="store_true")
    parser.add_argument("--validation-patience", type=int, default=0)
    parser.add_argument("--validation-min-steps", type=int, default=0)
    parser.add_argument("--validation-metrics-file", default="")
    parser.add_argument("--validation-max-active-trajectories", type=int, default=0)
    parser.add_argument("--best-model-dir", default="")
    parser.add_argument(
        "--best-model-metric",
        choices=("reward", "reward_raw", "answer_accuracy"),
        default="reward_raw",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False)
    temporary_path.replace(path)


def hardlink_or_copy_file(source: str, destination: str) -> str:
    """优先用硬链接保存模型文件，跨文件系统时退化为复制。"""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def save_best_policy_snapshot(
    source_path: Path,
    best_path: Path,
    metadata: Dict[str, Any],
) -> None:
    """原子替换唯一 best 快照，避免按 step 累积完整模型。"""
    source_path = source_path.resolve()
    if not (source_path / "config.json").is_file():
        raise RuntimeError("The best-model source is missing config.json: %s" % source_path)
    next_path = best_path.with_name(best_path.name + "-next")
    if next_path.exists():
        shutil.rmtree(next_path)
    shutil.copytree(
        source_path,
        next_path,
        copy_function=hardlink_or_copy_file,
    )
    write_json_atomic(next_path / "best_metrics.json", metadata)
    if best_path.exists():
        shutil.rmtree(best_path)
    next_path.replace(best_path)


def should_save_periodic_policy(step: int, start_step: int, interval: int) -> bool:
    """判断当前全局 step 是否命中定期完整模型快照。"""
    return (
        start_step > 0
        and interval > 0
        and step >= start_step
        and (step - start_step) % interval == 0
    )


def save_periodic_policy_snapshot(
    source_path: Path,
    snapshot_root: Path,
    completed_step: int,
    metadata: Dict[str, Any],
) -> Path:
    """原子保存指定 step 的完整 Hugging Face 策略，并保留历史快照。"""
    if completed_step <= 0:
        raise ValueError("completed_step must be positive.")
    source_path = source_path.resolve()
    if not (source_path / "config.json").is_file():
        raise RuntimeError("The periodic model snapshot source is missing config.json: %s" % source_path)

    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / ("step-%04d" % completed_step)
    next_path = snapshot_path.with_name(snapshot_path.name + "-next")
    if next_path.exists():
        shutil.rmtree(next_path)
    shutil.copytree(
        source_path,
        next_path,
        copy_function=hardlink_or_copy_file,
    )
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["completed_step"] = completed_step
    write_json_atomic(next_path / "checkpoint_metadata.json", checkpoint_metadata)
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    next_path.replace(snapshot_path)
    return snapshot_path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: str | Path) -> str:
    """流式计算小型配置和数据文件哈希，续训时据此拒绝错配输入。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_resume_checkpoint_path(path: str | Path) -> Path:
    """接受 latest 目录或其父目录，并在原子替换中断时回退到 previous。"""
    requested = Path(path).expanduser().resolve()
    candidates = [requested, requested / "latest"]
    if requested.name == "latest":
        candidates.append(requested.with_name("latest-previous"))
    else:
        candidates.append(requested / "latest-previous")
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("Cannot find a complete resume state: %s" % requested)


def load_resume_manifest(path: str | Path) -> tuple[Path, Dict[str, Any]]:
    checkpoint_path = resolve_resume_checkpoint_path(path)
    manifest = load_json(checkpoint_path / "manifest.json")
    if not manifest.get("complete"):
        raise RuntimeError("Resume state is not marked complete: %s" % checkpoint_path)
    return checkpoint_path, manifest


def _optional_file_fingerprint(path_value: str) -> Dict[str, Any]:
    if not path_value:
        return {"path": "", "sha256": ""}
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return {"path": str(path), "sha256": sha256_file(path)}


def build_resume_fingerprint(
    args: argparse.Namespace,
    protocol: PromptProtocol,
    world_size: int,
) -> Dict[str, Any]:
    """记录会改变优化、rollout 或奖励语义的关键配置。"""
    critical_names = (
        "dataset",
        "boundary_mode",
        "effective_batch_size",
        "num_generations",
        "rollout_generation_batch_size",
        "rollout_prefix_caching",
        "policy_micro_batch_size",
        "max_new_tokens",
        "max_streaming_tokens_per_edu",
        "max_deep_tokens",
        "max_model_len",
        "temperature",
        "top_p",
        "top_k",
        "lr",
        "weight_decay",
        "warmup_ratio",
        "scheduler_total_steps",
        "clip_range",
        "kl_coef",
        "local_weight",
        "accuracy_weight",
        "format_weight",
        "length_weight",
        "accuracy_reward_mode",
        "latency_alpha",
        "reward_tau",
        "grad_clip",
        "seed",
        "gradient_checkpointing",
        "attn_implementation",
        "task_type",
        "generation_tokens_per_second",
        "deep_budget",
        "input_tokens_per_second",
        "stream_probe_max_tokens",
        "no_stream_probe_max_tokens",
        "counterfactual_probes",
        "enable_edu_controller",
        "policy_minibatches",
        "sequence_clip_range_low",
        "sequence_clip_range_high",
        "max_sequence_log_ratio",
        "max_sequence_clip_fraction",
        "validation_interval",
        "validation_start_step",
        "validation_patience",
        "validation_min_steps",
        "best_model_metric",
    )
    critical_args = {
        name: getattr(args, name)
        for name in critical_names
        if hasattr(args, name)
    }
    edu_config = _optional_file_fingerprint(str(getattr(args, "edu_config", "")))
    model_root = Path(args.model_path).expanduser().resolve()
    return {
        "format_version": 2,
        "data_sampler": {
            "method": "epoch_deterministic_shuffle_with_persistent_cursor",
            "state_format_version": 1,
        },
        "world_size": world_size,
        "prompt_protocol_hash": protocol.protocol_hash,
        "prompt_config": _optional_file_fingerprint(str(args.prompt_config)),
        "train_file": _optional_file_fingerprint(str(args.train_file)),
        "validation_file": _optional_file_fingerprint(str(args.validation_file)),
        "edu_config": edu_config,
        "model_config": _optional_file_fingerprint(str(model_root / "config.json")),
        "tokenizer_config": _optional_file_fingerprint(
            str(model_root / "tokenizer_config.json")
        ),
        "critical_args": critical_args,
    }


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    """捕获当前 rank 的全部训练随机数状态。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
    """在所有模型和优化器状态加载完成后恢复随机数状态。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def move_tensors_to_cpu(value: Any) -> Any:
    """递归把优化器状态张量移到 CPU，避免分片保存时额外占用显存。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_cpu(item) for item in value)
    return value


def save_exact_resume_checkpoint(
    *,
    checkpoint_root: Path,
    actor: FSDP,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
    rank: int,
    world_size: int,
    control_group: Any,
    completed_step: int,
    trainer_state: Dict[str, Any],
    fingerprint: Dict[str, Any],
    best_model_dir: Path,
) -> Path:
    """原子保存 actor、优化器、调度器、trainer 状态和每个 rank 的 RNG。"""
    sampler_state = trainer_state.get("data_sampler_state")
    if not isinstance(sampler_state, dict):
        raise ValueError("Exact resume state is missing data_sampler_state.")
    for name in ("epoch", "shuffle_seed", "data_cursor"):
        if name not in trainer_state or name not in sampler_state:
            raise ValueError(f"Exact resume state is missing sampler fields: {name}")
        if int(trainer_state[name]) != int(sampler_state[name]):
            raise ValueError(f"Sampler fields differ while saving exact resume state: {name}")
    try:
        from torch.distributed.checkpoint import save as distributed_checkpoint_save
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        checkpoint_backend = "dcp_get_state_dict"
    except ImportError:
        checkpoint_backend = "fsdp_local_parameter_shards"

    latest_path = checkpoint_root / "latest"
    previous_path = checkpoint_root / "latest-previous"
    staging_path = checkpoint_root / "latest-next"
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if staging_path.exists():
            shutil.rmtree(staging_path)
        staging_path.mkdir(parents=True)
    dist.barrier(group=control_group)

    # 保存操作不应推进训练随机序列，因此结束后把本进程恢复到捕获点。
    rng_state = capture_rng_state(device)
    if checkpoint_backend == "dcp_get_state_dict":
        options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)
        model_state_dict, optimizer_state_dict = get_state_dict(
            actor,
            optimizer,
            options=options,
        )
        distributed_checkpoint_save(
            {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
            },
            checkpoint_id=staging_path / "distributed",
        )
        del model_state_dict, optimizer_state_dict
    else:
        # 旧版 Torch 的 LOCAL_STATE_DICT CPU offload 会破坏 ShardedTensor 的
        # requires_grad 元数据，因此直接保存同 world-size 下的本地扁平参数分片。
        model_parameter_shards = [
            parameter.detach().cpu().clone()
            for parameter in actor.parameters()
        ]
        model_buffers = {
            name: buffer.detach().cpu().clone()
            for name, buffer in actor.named_buffers()
        }
        optimizer_state_dict = move_tensors_to_cpu(optimizer.state_dict())
        torch.save(
            {
                "model_parameter_shards": model_parameter_shards,
                "model_buffers": model_buffers,
                "optimizer": optimizer_state_dict,
            },
            staging_path / ("state_rank_%05d.pt" % rank),
        )
        del model_parameter_shards, model_buffers, optimizer_state_dict
    torch.save(rng_state, staging_path / ("rng_rank_%05d.pt" % rank))
    restore_rng_state(rng_state, device)
    dist.barrier(group=control_group)

    if rank == 0:
        state_to_save = {
            **trainer_state,
            "completed_step": completed_step,
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(state_to_save, staging_path / "trainer_state.pt")
        manifest = {
            "complete": True,
            "format_version": 2,
            "completed_step": completed_step,
            "epoch": int(sampler_state["epoch"]),
            "shuffle_seed": int(sampler_state["shuffle_seed"]),
            "data_cursor": int(sampler_state["data_cursor"]),
            "world_size": world_size,
            "torch_version": torch.__version__,
            "checkpoint_backend": checkpoint_backend,
            "best_model_dir": str(best_model_dir.resolve()),
            "fingerprint": fingerprint,
            "distributed_checkpoint": (
                "distributed" if checkpoint_backend == "dcp_get_state_dict" else ""
            ),
            "rank_state_pattern": "state_rank_%05d.pt",
            "trainer_state": "trainer_state.pt",
            "rng_pattern": "rng_rank_%05d.pt",
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json_atomic(staging_path / "manifest.json", manifest)
        if previous_path.exists():
            shutil.rmtree(previous_path)
        if latest_path.exists():
            latest_path.replace(previous_path)
        try:
            staging_path.replace(latest_path)
        except Exception:
            if previous_path.exists() and not latest_path.exists():
                previous_path.replace(latest_path)
            raise
        if previous_path.exists():
            shutil.rmtree(previous_path)
    dist.barrier(group=control_group)
    return latest_path


def load_exact_resume_checkpoint(
    *,
    checkpoint_path: Path,
    manifest: Dict[str, Any],
    actor: FSDP,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
    rank: int,
    world_size: int,
    control_group: Any,
    fingerprint: Dict[str, Any],
    best_model_dir: Path,
) -> Dict[str, Any]:
    """严格校验运行配置后恢复完整分片训练状态。"""
    if int(manifest.get("world_size", -1)) != world_size:
        raise ValueError(
            "Resume world_size mismatch: %s != %s"
            % (manifest.get("world_size"), world_size)
        )
    saved_fingerprint = manifest.get("fingerprint")
    if saved_fingerprint != fingerprint:
        raise ValueError(
            "Resume configuration, data, or prompt protocol mismatch.\nSaved=%s\nCurrent=%s"
            % (
                json.dumps(saved_fingerprint, ensure_ascii=False, sort_keys=True),
                json.dumps(fingerprint, ensure_ascii=False, sort_keys=True),
            )
        )
    saved_best_model_dir = Path(str(manifest["best_model_dir"])).resolve()
    if saved_best_model_dir != best_model_dir.resolve():
        raise ValueError(
            "Resume must retain the original best_model_dir: %s != %s"
            % (saved_best_model_dir, best_model_dir.resolve())
        )

    checkpoint_backend = str(manifest.get("checkpoint_backend", "dcp_get_state_dict"))
    if checkpoint_backend == "fsdp_local_parameter_shards":
        saved_torch_version = str(manifest.get("torch_version", ""))
        if saved_torch_version and saved_torch_version != torch.__version__:
            raise ValueError(
                "Local FSDP shards require the same Torch version: %s != %s"
                % (saved_torch_version, torch.__version__)
            )
        rank_state_path = checkpoint_path / (
            str(manifest.get("rank_state_pattern", "state_rank_%05d.pt")) % rank
        )
        rank_state = torch.load(rank_state_path, map_location="cpu", weights_only=False)
        parameters = list(actor.parameters())
        saved_parameter_shards = list(rank_state["model_parameter_shards"])
        if len(parameters) != len(saved_parameter_shards):
            raise ValueError(
                "Local parameter shard count mismatch: %d != %d"
                % (len(parameters), len(saved_parameter_shards))
            )
        saved_buffers = dict(rank_state.get("model_buffers") or {})
        current_buffers = dict(actor.named_buffers())
        if set(saved_buffers) != set(current_buffers):
            raise ValueError("Local model buffer names differ from the checkpoint.")
        with torch.no_grad():
            for parameter, saved_parameter in zip(parameters, saved_parameter_shards):
                if parameter.shape != saved_parameter.shape:
                    raise ValueError("Local parameter shard shapes differ from the checkpoint.")
                parameter.copy_(saved_parameter.to(device=parameter.device, dtype=parameter.dtype))
            for name, buffer in current_buffers.items():
                saved_buffer = saved_buffers[name]
                if buffer.shape != saved_buffer.shape:
                    raise ValueError("Model buffer shapes differ from the checkpoint: %s" % name)
                buffer.copy_(saved_buffer.to(device=buffer.device, dtype=buffer.dtype))
        optimizer.load_state_dict(rank_state["optimizer"])
        del rank_state
    elif checkpoint_backend == "dcp_get_state_dict":
        from torch.distributed.checkpoint import load as distributed_checkpoint_load
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
            set_state_dict,
        )

        options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)
        model_state_dict, optimizer_state_dict = get_state_dict(
            actor,
            optimizer,
            options=options,
        )
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }
        distributed_checkpoint_load(
            state_dict,
            checkpoint_id=checkpoint_path / str(manifest["distributed_checkpoint"]),
        )
        set_state_dict(
            actor,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=options,
        )
        del state_dict, model_state_dict, optimizer_state_dict
    else:
        raise ValueError("Unknown resume checkpoint backend: %s" % checkpoint_backend)

    trainer_state = torch.load(
        checkpoint_path / str(manifest["trainer_state"]),
        map_location="cpu",
        weights_only=False,
    )
    scheduler.load_state_dict(trainer_state["scheduler_state_dict"])
    rng_state = torch.load(
        checkpoint_path / (str(manifest["rng_pattern"]) % rank),
        map_location="cpu",
        weights_only=False,
    )
    restore_rng_state(rng_state, device)
    dist.barrier(group=control_group)
    return trainer_state


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return "%dd%02dh%02dm" % (days, hours, minutes)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, secs)
    return "%dm%02ds" % (minutes, secs)


def append_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")


def format_compact_metrics(metrics: Dict[str, Any]) -> str:
    line = (
        "time=%s step=%d loss=%.6f reward=%.6f reward_raw=%.6f "
        "advantage_mean=%.6f advantage_std=%.6f mean_abs_advantage=%.6f "
        "answer_accuracy=%.4f "
        "policy_entropy=%.6f sfmt=%.4f dfmt=%.4f response_len=%.1f "
        "truncation=%.4f grad=%.4f"
        % (
            metrics["time"],
            metrics["step"],
            metrics["loss"],
            metrics["reward"],
            metrics["reward_raw"],
            metrics["advantage_mean"],
            metrics["advantage_std"],
            metrics["mean_abs_advantage"],
            metrics["answer_accuracy"],
            metrics["policy_entropy"],
            metrics["streaming_format_reward"],
            metrics["deep_format_reward"],
            metrics["response_length"],
            metrics["truncation_rate"],
            metrics["grad_norm"],
        )
    )
    if "data_epoch" in metrics:
        line += " data_epoch=%d data_cursor=%d" % (
            metrics["data_epoch"],
            metrics["data_cursor"],
        )
    return line


def append_compact_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(format_compact_metrics(metrics) + "\n")


def pad_segment_lengths(rows: List[List[int]], width: int, device: torch.device) -> torch.Tensor:
    padded = [row[:width] + [0] * max(width - len(row), 0) for row in rows]
    return torch.tensor(padded, dtype=torch.float32, device=device)


def pad_segment_mask(rows: List[List[int]], width: int, device: torch.device) -> torch.Tensor:
    masks = [[True] * min(len(row), width) + [False] * max(width - len(row), 0) for row in rows]
    return torch.tensor(masks, dtype=torch.bool, device=device)


def build_warmup_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    """按论文配置执行线性 warmup，之后保持学习率不变。"""
    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_ratio > 0 and warmup_steps == 0:
        warmup_steps = 1

    def lr_scale(step_index: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(float(step_index + 1) / float(warmup_steps), 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale), warmup_steps


def load_fsdp_model(args: argparse.Namespace, device: torch.device):
    """加载 Qwen3 actor，并按 Transformer block 应用 FSDP FULL_SHARD。"""
    from transformers import AutoConfig
    from Qwen3.qwen_streaming import Qwen3DecoderLayer_stream, Qwen3ForCausalLM_stream

    model_path = args.resume_from_checkpoint or args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer, tokenizer_info = load_streaming_tokenizer(
        model_path,
        config=config,
        padding_side="right",
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            "streaming_tokenizer_ready source=%s size=%d token_set=%s added=%d ids=%s"
            % (
                model_path,
                tokenizer_info.vocab_size,
                tokenizer_info.token_set,
                tokenizer_info.added_tokens,
                json.dumps(tokenizer_info.token_ids, ensure_ascii=True, sort_keys=True),
            ),
            flush=True,
        )
    config._attn_implementation = args.attn_implementation
    actor = Qwen3ForCausalLM_stream.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    embedding_rows = int(actor.get_input_embeddings().weight.shape[0])
    if embedding_rows != len(tokenizer):
        raise RuntimeError(
            "Actor embeddings and streaming tokenizer vocabulary differ: %d != %d."
            % (embedding_rows, len(tokenizer))
        )
    actor.config.pad_token_id = tokenizer.pad_token_id
    actor.config.use_cache = False
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3DecoderLayer_stream},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    actor = FSDP(
        actor,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
    )
    return actor, tokenizer, model_path


def build_rollout_request(
    samples: List[RLSample],
    sample_indices: List[int],
    protocol: PromptProtocol,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """构造 rollout 请求，并在可用时写入精确的输入 token 数。"""
    items = []
    for sample_index in sample_indices:
        source = build_streaming_source(samples[sample_index], protocol)
        bootstrap_segments = list(protocol.rollout_bootstrap_target_segments)
        if len(bootstrap_segments) > len(source.response_boundaries):
            raise ValueError("Rollout bootstrap segment count must not exceed streaming target segment count.")
        for index, segment in enumerate(bootstrap_segments):
            expected_boundary = source.response_boundaries[index]
            if not segment.rstrip().endswith(expected_boundary):
                raise ValueError(
                    "Rollout bootstrap segment %d must end with %s."
                    % (index, expected_boundary)
                )
        bootstrap_count = len(bootstrap_segments)
        item = {
            "sample_index": sample_index,
            "source_segments": source.segments,
            "expected_boundaries": source.response_boundaries[bootstrap_count:],
            "question_segments": max(
                source.question_segments - bootstrap_count,
                0,
            ),
            "target_prefix": protocol.target_prefix,
            "bootstrap_target_segments": bootstrap_segments,
            "reasoning_connect_2": protocol.reasoning_connect_2,
            "reasoning_connect_3": protocol.reasoning_connect_3,
            "prompt_protocol": protocol.to_metadata(),
        }
        if tokenizer is not None:
            prompt_segments = (
                list(source.segments)
                + [protocol.target_prefix]
                + bootstrap_segments
            )
            item["input_token_count"] = sum(
                len(
                    tokenizer(
                        str(segment),
                        add_special_tokens=False,
                    )["input_ids"]
                )
                for segment in prompt_segments
            )
        items.append(item)
    return {"prompt_protocol": protocol.to_metadata(), "items": items}


def run_streaming_rollout(
    args: argparse.Namespace,
    policy_model_path: str,
    request_path: Path,
    output_path: Path,
    worker_dir: Path,
    step: int,
) -> None:
    command = [
        args.rollout_python,
        args.rollout_script,
        "--model-path",
        policy_model_path,
        "--request-file",
        str(request_path),
        "--output-file",
        str(output_path),
        "--work-dir",
        str(worker_dir),
        "--gpu-ids",
        args.rollout_gpu_ids,
        "--num-generations",
        str(args.num_generations),
        "--rollout-generation-batch-size",
        str(args.rollout_generation_batch_size),
        "--max-active-trajectories",
        str(args.rollout_max_active_trajectories),
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
        str(args.rollout_max_num_seqs),
        "--max-num-batched-tokens",
        str(args.rollout_max_num_batched_tokens),
        "--gpu-memory-fraction",
        str(args.rollout_gpu_memory_fraction),
        "--attn-implementation",
        args.attn_implementation,
        "--seed",
        str(args.seed + step * 1000),
    ]
    command.append(
        "--enable-prefix-caching"
        if args.rollout_prefix_caching
        else "--no-enable-prefix-caching"
    )
    task_type = str(getattr(args, "task_type", "generic") or "generic")
    command.extend(["--task-type", task_type])
    if getattr(args, "counterfactual_probes", False):
        command.extend(
            [
                "--counterfactual-probes",
                "--stream-probe-max-tokens",
                str(args.stream_probe_max_tokens),
                "--no-stream-probe-max-tokens",
                str(args.no_stream_probe_max_tokens),
            ]
        )
    if getattr(args, "enable_edu_controller", False):
        command.append("--enable-edu-controller")
    environment = os.environ.copy()
    print(
        "step=%d phase=streaming_rollout_start policy=%s" % (step, policy_model_path),
        flush=True,
    )
    subprocess.run(command, check=True, env=environment)


def evaluate_validation_policy(
    args: argparse.Namespace,
    policy_model_path: str,
    validation_samples: List[RLSample],
    protocol: PromptProtocol,
    tokenizer: Any,
    runtime_dir: Path,
    evaluation_step: int,
) -> Dict[str, Any]:
    """使用确定性单轨迹 rollout 评估独立验证集，不生成反事实探针。"""
    request_path = runtime_dir / "validation_request.json"
    output_path = runtime_dir / "validation_output.json"
    worker_dir = runtime_dir / "validation_workers"
    write_json_atomic(
        request_path,
        build_rollout_request(
            validation_samples,
            list(range(len(validation_samples))),
            protocol,
            tokenizer,
        ),
    )
    if output_path.exists():
        output_path.unlink()

    validation_args = argparse.Namespace(**vars(args))
    validation_args.num_generations = 1
    validation_args.temperature = 0.0
    validation_args.top_p = 1.0
    validation_args.top_k = -1
    validation_args.counterfactual_probes = False
    validation_args.rollout_max_active_trajectories = (
        args.validation_max_active_trajectories
        or args.rollout_max_active_trajectories
    )
    started_at = time.perf_counter()
    run_streaming_rollout(
        validation_args,
        policy_model_path,
        request_path,
        output_path,
        worker_dir,
        evaluation_step,
    )
    rollout_result = load_json(output_path)
    items = rollout_result.get("items") or []
    if len(items) != len(validation_samples):
        raise RuntimeError(
            "Validation rollout prompt count mismatch: %d != %d"
            % (len(items), len(validation_samples))
        )

    accuracy_values: List[float] = []
    streaming_format_values: List[float] = []
    deep_format_values: List[float] = []
    response_lengths: List[int] = []
    truncated_values: List[float] = []
    accuracy_values_by_label: Dict[str, List[float]] = {}
    for item in items:
        sample_index = int(item["sample_index"])
        sample = validation_samples[sample_index]
        completions = item.get("completions") or []
        if len(completions) != 1:
            raise RuntimeError(
                "Validation sample_index=%d has %d trajectories; expected 1."
                % (sample_index, len(completions))
            )
        completion = completions[0]
        text = str(completion.get("text") or "")
        expected_segments = len(item.get("expected_boundaries") or [])
        streaming_segments, deep_text, full_text = parse_hrpo_response(
            text,
            expected_segments,
        )
        if expected_segments == 0:
            stream_format = 1.0
        else:
            required_segments = streaming_segments[:expected_segments]
            stream_format = float(len(streaming_segments) >= expected_segments)
            stream_format *= sft_streaming_format_reward(
                required_segments,
                count_question_stream_segments(sample.question),
            )
        accuracy = exact_match_accuracy_reward(
            full_text,
            sample.answer,
            require_boxed=args.accuracy_reward_mode == "boxed",
        )
        accuracy_values.append(accuracy)
        label = sample.answer.removeprefix("\\boxed{").removesuffix("}")
        accuracy_values_by_label.setdefault(label, []).append(accuracy)
        streaming_format_values.append(float(stream_format))
        deep_format_values.append(float(deep_format_reward(deep_text)))
        response_lengths.append(len(completion.get("token_ids") or []))
        truncated_values.append(float(completion.get("finish_reason") == "length"))

    count = max(len(items), 1)
    answer_accuracy = sum(accuracy_values) / count
    streaming_format = sum(streaming_format_values) / count
    deep_format = sum(deep_format_values) / count
    truncation_rate = sum(truncated_values) / count
    response_length = sum(response_lengths) / count
    return {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_step": evaluation_step,
        "policy_update_step": evaluation_step,
        "validation_sample_count": len(validation_samples),
        "validation_answer_accuracy": answer_accuracy,
        "validation_accuracy_by_label": {
            label: sum(values) / len(values)
            for label, values in sorted(accuracy_values_by_label.items())
        },
        "validation_streaming_format_reward": streaming_format,
        "validation_deep_format_reward": deep_format,
        "validation_format_score": 0.5 * (streaming_format + deep_format),
        "validation_response_length": response_length,
        "validation_truncation_rate": truncation_rate,
        "validation_rollout_seconds": float(rollout_result.get("rollout_seconds", 0.0)),
        "validation_total_seconds": time.perf_counter() - started_at,
        "validation_termination_stats": rollout_result.get("termination_stats") or {},
        "source_policy": str(Path(policy_model_path).resolve()),
    }


def validation_selection_tuple(metrics: Dict[str, Any]) -> tuple[float, ...]:
    """准确率为主指标，其余指标只在准确率相同时打破平局。"""
    return (
        float(metrics["validation_answer_accuracy"]),
        float(metrics["validation_format_score"]),
        -float(metrics["validation_truncation_rate"]),
        -float(metrics["validation_response_length"]),
    )


def should_evaluate_validation(
    step: int,
    scheduler_total_steps: int,
    interval: int,
    start_step: int,
) -> bool:
    """从指定步数开始周期验证，同时保证训练终点一定被评估。"""
    if interval <= 0:
        raise ValueError("validation interval must be positive.")
    if start_step < 0:
        raise ValueError("validation start step must be nonnegative.")
    return step == scheduler_total_steps or (
        step >= start_step and step % interval == 0
    )


def append_validation_compact_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    """把验证结果写入现有精简日志，便于和训练 step 对齐查看。"""
    line = (
        "time=%s validation_step=%d val_accuracy=%.4f val_sfmt=%.4f "
        "val_dfmt=%.4f val_response_len=%.1f val_truncation=%.4f "
        "val_seconds=%.1f is_best=%d\n"
        % (
            metrics["time"],
            metrics["evaluation_step"],
            metrics["validation_answer_accuracy"],
            metrics["validation_streaming_format_reward"],
            metrics["validation_deep_format_reward"],
            metrics["validation_response_length"],
            metrics["validation_truncation_rate"],
            metrics["validation_total_seconds"],
            int(metrics["is_best_model"]),
        )
    )
    with path.open("a", encoding="utf-8") as file:
        file.write(line)


def evaluate_validation_checkpoint(
    args: argparse.Namespace,
    policy_model_path: str,
    validation_samples: List[RLSample],
    protocol: PromptProtocol,
    tokenizer: Any,
    runtime_dir: Path,
    validation_metrics_path: Path,
    compact_metrics_path: Path,
    best_model_dir: Path,
    evaluation_step: int,
    best_tuple: tuple[float, ...],
) -> tuple[Dict[str, Any], tuple[float, ...], bool]:
    """评估当前策略，并在验证排序严格提升时替换唯一 best 快照。"""
    metrics = evaluate_validation_policy(
        args,
        policy_model_path,
        validation_samples,
        protocol,
        tokenizer,
        runtime_dir,
        evaluation_step,
    )
    candidate_tuple = validation_selection_tuple(metrics)
    is_best = candidate_tuple > best_tuple
    metrics["selection_metric"] = "validation_answer_accuracy"
    metrics["selection_tuple"] = list(candidate_tuple)
    metrics["is_best_model"] = is_best
    if is_best:
        best_record = {
            **metrics,
            "selection_score": float(metrics["validation_answer_accuracy"]),
            "evaluated_before_actor_update": False,
        }
        save_best_policy_snapshot(Path(policy_model_path), best_model_dir, best_record)
        best_tuple = candidate_tuple
    append_metrics(validation_metrics_path, metrics)
    append_validation_compact_metrics(compact_metrics_path, metrics)
    print(
        "validation_step=%d accuracy=%.4f sfmt=%.4f dfmt=%.4f "
        "response_len=%.1f truncation=%.4f best=%d seconds=%s"
        % (
            evaluation_step,
            metrics["validation_answer_accuracy"],
            metrics["validation_streaming_format_reward"],
            metrics["validation_deep_format_reward"],
            metrics["validation_response_length"],
            metrics["validation_truncation_rate"],
            int(is_best),
            format_duration(metrics["validation_total_seconds"]),
        ),
        flush=True,
    )
    return metrics, best_tuple, is_best


def prepare_group(
    item: Dict[str, Any],
    samples: List[RLSample],
    tokenizer,
    expected_generations: int,
    accuracy_reward_mode: str = "boxed",
) -> PreparedGroup:
    sample_index = int(item["sample_index"])
    sample = samples[sample_index]
    completions = item.get("completions") or []
    if len(completions) != expected_generations:
        raise RuntimeError(
            "sample_index=%d has %d rollouts; expected %d."
            % (sample_index, len(completions), expected_generations)
        )
    expected_segments = len(item.get("expected_boundaries") or [])
    question_segments = int(item.get("question_segments") or count_question_stream_segments(sample.question))
    streaming_format_rewards = []
    deep_format_rewards = []
    accuracy_rewards = []
    segment_length_rows = []
    streaming_token_lengths = []
    deep_lengths = []
    response_ids = []
    response_segment_length_rows = []
    response_texts = []
    finish_reasons = []

    for completion in completions:
        text = str(completion.get("text") or "")
        streaming_segments, deep_text, full_text = parse_hrpo_response(text, expected_segments)
        required_segments = streaming_segments[:expected_segments] if expected_segments > 0 else streaming_segments
        if expected_segments == 0:
            stream_format = 1.0
        else:
            stream_format = float(len(streaming_segments) >= expected_segments) * sft_streaming_format_reward(
                required_segments,
                question_segments,
            )
        completion_ids = [int(token_id) for token_id in completion["token_ids"]]
        stage_lengths = [int(length) for length in completion["response_segment_token_lengths"]]
        if len(stage_lengths) != expected_segments + 1:
            raise RuntimeError(
                "sample_index=%d has %d target segments; expected %d."
                % (sample_index, len(stage_lengths), expected_segments + 1)
            )
        if sum(stage_lengths) != len(completion_ids):
            raise RuntimeError("Rollout target segment lengths do not match token_ids.")
        segment_lengths = stage_lengths[:expected_segments]
        deep_length = stage_lengths[-1]
        response_ids.append(completion_ids)
        response_segment_length_rows.append(stage_lengths)
        response_texts.append(text)
        streaming_format_rewards.append(stream_format)
        deep_format_rewards.append(deep_format_reward(deep_text))
        accuracy_rewards.append(
            exact_match_accuracy_reward(
                full_text,
                sample.answer,
                require_boxed=accuracy_reward_mode == "boxed",
            )
        )
        segment_length_rows.append(segment_lengths)
        streaming_token_lengths.append(sum(segment_lengths))
        deep_lengths.append(deep_length)
        finish_reasons.append(str(completion.get("finish_reason") or ""))

    return PreparedGroup(
        sample_index=sample_index,
        source_segment_ids=[
            [int(token_id) for token_id in segment]
            for segment in item["source_segment_token_ids"]
        ],
        target_prefix_ids=[int(token_id) for token_id in item["target_prefix_token_ids"]],
        target_prefix_segment_lengths=[
            int(length)
            for length in item.get("target_prefix_segment_lengths")
            or [len(item["target_prefix_token_ids"])]
        ],
        response_ids=response_ids,
        response_segment_length_rows=response_segment_length_rows,
        response_texts=response_texts,
        streaming_format_rewards=streaming_format_rewards,
        deep_format_rewards=deep_format_rewards,
        accuracy_rewards=accuracy_rewards,
        segment_length_rows=segment_length_rows,
        streaming_token_lengths=streaming_token_lengths,
        deep_lengths=deep_lengths,
        finish_reasons=finish_reasons,
        max_segments=max(expected_segments, 1),
    )


def compute_shared_tau(
    groups: List[PreparedGroup],
    device: torch.device,
    configured_tau: float,
) -> float:
    if configured_tau > 0:
        return configured_tau
    local_sum = sum(
        streaming_length + deep_length
        for group in groups
        for streaming_length, deep_length in zip(group.streaming_token_lengths, group.deep_lengths)
    )
    local_count = sum(len(group.response_ids) for group in groups)
    totals = torch.tensor([float(local_sum), float(local_count)], dtype=torch.float64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return max(float((totals[0] / totals[1].clamp_min(1.0)).cpu()), 1.0)


def save_rollout_policy(
    actor: FSDP,
    tokenizer,
    output_path: Path,
    rank: int,
    control_group,
) -> None:
    """原子导出一份完整策略，供下一步 vLLM worker 加载。"""
    next_path = output_path.with_name(output_path.name + "-next")
    if rank == 0:
        if next_path.exists():
            shutil.rmtree(next_path)
        # rollout 已经结束并释放模型文件，先删除旧策略以避免数据盘同时保留两份 4B 权重。
        if output_path.exists():
            shutil.rmtree(output_path)
        next_path.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=control_group)
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(actor, StateDictType.FULL_STATE_DICT, state_config):
        state_dict = actor.state_dict()
    if rank == 0:
        actor.module.save_pretrained(
            str(next_path),
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(str(next_path))
    del state_dict
    dist.barrier(group=control_group)
    if rank == 0:
        next_path.replace(output_path)
    dist.barrier(group=control_group)


def build_config(args: argparse.Namespace, reward_tau: float) -> HRPOConfig:
    return HRPOConfig(
        clip_range=args.clip_range,
        kl_coef=0.0,
        local_weight=args.local_weight,
        accuracy_weight=args.accuracy_weight,
        format_weight=args.format_weight,
        length_weight=args.length_weight,
        latency_alpha=args.latency_alpha,
        reward_tau=reward_tau,
    )


def train_actor_step(
    actor: FSDP,
    tokenizer,
    optimizer,
    groups: List[PreparedGroup],
    args: argparse.Namespace,
    device: torch.device,
    config: HRPOConfig,
) -> Dict[str, float]:
    """按 completion 微批立即反向传播，避免跨微批保留计算图。"""
    actor.train()
    optimizer.zero_grad(set_to_none=True)
    local_rollout_count = sum(len(group.response_ids) for group in groups)
    metric_names = [
        "loss_sum",
        "reward_sum",
        "reward_raw_sum",
        "advantage_sum",
        "advantage_square_sum",
        "advantage_abs_sum",
        "streaming_reward_sum",
        "deep_reward_sum",
        "global_reward_sum",
        "accuracy_sum",
        "streaming_format_sum",
        "deep_format_sum",
        "entropy_sum",
        "entropy_token_count",
        "streaming_length_sum",
        "deep_length_sum",
        "streaming_length_penalty_sum",
        "deep_length_penalty_sum",
        "efficiency_sum",
        "response_token_sum",
        "truncated_sum",
        "rollout_count",
        "accuracy_group_count",
        "accuracy_active_group_count",
        "accuracy_all_correct_group_count",
        "accuracy_all_wrong_group_count",
    ]
    local_metrics = {name: 0.0 for name in metric_names}
    total_chunks = sum(
        (len(group.response_ids) + args.policy_micro_batch_size - 1) // args.policy_micro_batch_size
        for group in groups
    )
    chunk_counter = 0

    for group in groups:
        group_size = len(group.response_ids)
        group_ids = torch.zeros(group_size, dtype=torch.long, device=device)
        segment_lengths = pad_segment_lengths(
            group.segment_length_rows,
            group.max_segments,
            device,
        )
        segment_mask = pad_segment_mask(
            group.segment_length_rows,
            group.max_segments,
            device,
        )
        advantages = compute_hrpo_advantages(
            streaming_format=torch.tensor(group.streaming_format_rewards, dtype=torch.float32, device=device),
            deep_format=torch.tensor(group.deep_format_rewards, dtype=torch.float32, device=device),
            accuracy=torch.tensor(group.accuracy_rewards, dtype=torch.float32, device=device),
            streaming_segment_lengths=segment_lengths,
            streaming_segment_mask=segment_mask,
            deep_lengths=torch.tensor(group.deep_lengths, dtype=torch.float32, device=device),
            group_ids=group_ids,
            config=config,
        )
        reporting_rewards = compose_reporting_rewards(advantages, config)
        advantage_values = reporting_rewards["advantage"]
        local_metrics["reward_sum"] += float(
            reporting_rewards["reward"].sum().detach().cpu()
        )
        local_metrics["reward_raw_sum"] += float(
            reporting_rewards["reward_raw"].sum().detach().cpu()
        )
        local_metrics["advantage_sum"] += float(
            advantage_values.sum().detach().cpu()
        )
        local_metrics["advantage_square_sum"] += float(
            advantage_values.square().sum().detach().cpu()
        )
        local_metrics["advantage_abs_sum"] += float(
            advantage_values.abs().sum().detach().cpu()
        )
        local_metrics["streaming_reward_sum"] += float(
            reporting_rewards["streaming_reward"].sum().detach().cpu()
        )
        local_metrics["deep_reward_sum"] += float(reporting_rewards["deep_reward"].sum().detach().cpu())
        local_metrics["global_reward_sum"] += float(
            reporting_rewards["global_reward"].sum().detach().cpu()
        )
        local_metrics["accuracy_sum"] += sum(group.accuracy_rewards)
        local_metrics["streaming_format_sum"] += sum(group.streaming_format_rewards)
        local_metrics["deep_format_sum"] += sum(group.deep_format_rewards)
        local_metrics["streaming_length_penalty_sum"] += float(
            advantages.components.streaming_length_penalty.sum().detach().cpu()
        )
        local_metrics["deep_length_penalty_sum"] += float(
            advantages.components.deep_length_penalty.sum().detach().cpu()
        )
        local_metrics["efficiency_sum"] += float(
            advantages.components.efficiency.sum().detach().cpu()
        )
        accuracy_group_stats = binary_group_reward_stats(group.accuracy_rewards)
        local_metrics["accuracy_group_count"] += 1.0
        local_metrics["accuracy_active_group_count"] += accuracy_group_stats["active"]
        local_metrics["accuracy_all_correct_group_count"] += accuracy_group_stats["all_correct"]
        local_metrics["accuracy_all_wrong_group_count"] += accuracy_group_stats["all_wrong"]
        local_metrics["truncated_sum"] += sum(reason == "length" for reason in group.finish_reasons)
        local_metrics["rollout_count"] += group_size

        for start in range(0, group_size, args.policy_micro_batch_size):
            end = min(start + args.policy_micro_batch_size, group_size)
            chunk_size = end - start
            chunk_counter += 1
            should_sync = chunk_counter == total_chunks
            sync_context = nullcontext() if should_sync else actor.no_sync()
            with sync_context:
                current_log_probs, response_mask, token_entropy = compute_streaming_response_statistics(
                    model=actor,
                    tokenizer=tokenizer,
                    source_segment_ids=group.source_segment_ids,
                    target_prefix_ids=group.target_prefix_ids,
                    target_prefix_segment_lengths=group.target_prefix_segment_lengths,
                    response_ids=group.response_ids[start:end],
                    response_segment_lengths=group.response_segment_length_rows[start:end],
                    device=device,
                    require_grad=True,
                )
                loss_device = current_log_probs.device
                response_mask = response_mask.to(loss_device)
                token_entropy = token_entropy.to(loss_device)
                stream_lengths = torch.tensor(
                    group.streaming_token_lengths[start:end],
                    dtype=torch.long,
                )
                response_lengths = response_mask.sum(dim=-1).long().cpu()
                stream_lengths = torch.minimum(stream_lengths, response_lengths)
                deep_mask_lengths = response_lengths - stream_lengths
                streaming_mask, deep_mask, mask_from_lengths = phase_masks_from_lengths(
                    stream_lengths.to(loss_device),
                    deep_mask_lengths.to(loss_device),
                    max_length=current_log_probs.shape[1],
                )
                response_mask = response_mask & mask_from_lengths
                loss, loss_metrics = hrpo_loss(
                    log_probs=current_log_probs,
                    old_log_probs=current_log_probs.detach(),
                    response_mask=response_mask,
                    streaming_mask=streaming_mask,
                    deep_mask=deep_mask,
                    streaming_advantages=advantages.streaming[start:end],
                    deep_advantages=advantages.deep[start:end],
                    global_advantages=advantages.global_[start:end],
                    ref_log_probs=None,
                    config=config,
                )
                scaled_loss = loss * (float(chunk_size) / max(local_rollout_count, 1))
                scaled_loss.backward()

            local_metrics["loss_sum"] += float(loss.detach().cpu()) * chunk_size
            entropy_mask = response_mask.to(token_entropy.dtype)
            local_metrics["entropy_sum"] += float((token_entropy * entropy_mask).sum().detach().cpu())
            local_metrics["entropy_token_count"] += int(response_mask.sum().detach().cpu())
            local_metrics["streaming_length_sum"] += float(
                loss_metrics["streaming_len_mean"].detach().cpu()
            ) * chunk_size
            local_metrics["deep_length_sum"] += float(
                loss_metrics["deep_len_mean"].detach().cpu()
            ) * chunk_size
            local_metrics["response_token_sum"] += int(response_mask.sum().detach().cpu())

    grad_norm = actor.clip_grad_norm_(args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metric_tensor = torch.tensor(
        [local_metrics[name] for name in metric_names],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
    reduced = {name: float(metric_tensor[index].cpu()) for index, name in enumerate(metric_names)}
    rollout_count = max(reduced["rollout_count"], 1.0)
    entropy_count = max(reduced["entropy_token_count"], 1.0)
    accuracy_group_count = max(reduced["accuracy_group_count"], 1.0)
    advantage_mean = reduced["advantage_sum"] / rollout_count
    advantage_variance = max(
        reduced["advantage_square_sum"] / rollout_count - advantage_mean**2,
        0.0,
    )
    return {
        "loss": reduced["loss_sum"] / rollout_count,
        "reward": reduced["reward_sum"] / rollout_count,
        "reward_raw": reduced["reward_raw_sum"] / rollout_count,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_variance**0.5,
        "mean_abs_advantage": reduced["advantage_abs_sum"] / rollout_count,
        "streaming_reward": reduced["streaming_reward_sum"] / rollout_count,
        "deep_reward": reduced["deep_reward_sum"] / rollout_count,
        "global_reward": reduced["global_reward_sum"] / rollout_count,
        "answer_accuracy": reduced["accuracy_sum"] / rollout_count,
        "streaming_format_reward": reduced["streaming_format_sum"] / rollout_count,
        "deep_format_reward": reduced["deep_format_sum"] / rollout_count,
        "policy_entropy": reduced["entropy_sum"] / entropy_count,
        "streaming_length": reduced["streaming_length_sum"] / rollout_count,
        "deep_length": reduced["deep_length_sum"] / rollout_count,
        "streaming_length_penalty": reduced["streaming_length_penalty_sum"] / rollout_count,
        "deep_length_penalty": reduced["deep_length_penalty_sum"] / rollout_count,
        "efficiency_reward": reduced["efficiency_sum"] / rollout_count,
        "accuracy_active_group_rate": reduced["accuracy_active_group_count"] / accuracy_group_count,
        "accuracy_all_correct_group_rate": reduced["accuracy_all_correct_group_count"]
        / accuracy_group_count,
        "accuracy_all_wrong_group_rate": reduced["accuracy_all_wrong_group_count"]
        / accuracy_group_count,
        "response_length": reduced["response_token_sum"] / rollout_count,
        "truncation_rate": reduced["truncated_sum"] / rollout_count,
        "rollouts": int(reduced["rollout_count"]),
        "grad_norm": float(grad_norm.detach().cpu()),
    }


def main() -> None:
    args = parse_args()
    prompt_protocol = load_training_prompt_protocol(
        prompt_config=args.prompt_config,
        dataset=args.dataset,
        boundary_mode=args.boundary_mode,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    missing_torchrun_env = [
        name for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE") if name not in os.environ
    ]
    if missing_torchrun_env:
        raise RuntimeError("Launch the FSDP entry point with torchrun.")
    if world_size < 1:
        raise ValueError("FSDP world_size must be positive.")
    if args.kl_coef != 0:
        raise ValueError("This FSDP entry point uses the paper setting kl_coef=0.")
    if args.effective_batch_size % world_size != 0:
        raise ValueError("effective_batch_size must be divisible by FSDP world_size.")
    if args.policy_micro_batch_size <= 0:
        raise ValueError("policy_micro_batch_size must be positive.")
    if args.rollout_max_active_trajectories < 0:
        raise ValueError("rollout_max_active_trajectories must be nonnegative.")
    if args.validation_max_active_trajectories < 0:
        raise ValueError("validation_max_active_trajectories must be nonnegative.")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1].")
    if args.step_offset < 0:
        raise ValueError("step_offset must be nonnegative.")
    if args.resume_from_checkpoint and args.resume_state_from:
        raise ValueError("resume_from_checkpoint and resume_state_from are mutually exclusive.")
    if args.resume_state_save_interval < 0:
        raise ValueError("resume_state_save_interval must be nonnegative.")
    if args.periodic_model_save_start_step < 0:
        raise ValueError("periodic_model_save_start_step must be nonnegative.")
    if args.periodic_model_save_interval < 0:
        raise ValueError("periodic_model_save_interval must be nonnegative.")
    if bool(args.periodic_model_save_start_step) != bool(args.periodic_model_save_interval):
        raise ValueError("Periodic model saving requires start_step and interval to be enabled or disabled together.")
    periodic_model_save_enabled = args.periodic_model_save_interval > 0
    if periodic_model_save_enabled and not args.periodic_model_save_dir:
        raise ValueError("Periodic model saving requires periodic_model_save_dir.")
    if not args.resume_state_from and args.scheduler_total_steps <= 0:
        args.scheduler_total_steps = args.step_offset + args.steps

    validation_enabled = bool(args.validation_file)
    if validation_enabled and args.validation_interval <= 0:
        raise ValueError("validation_interval must be positive when validation_file is provided.")
    if not validation_enabled and (
        args.validation_interval > 0
        or args.validation_start_step > 0
        or args.validation_patience > 0
        or args.validation_min_steps > 0
    ):
        raise ValueError("Validation options require validation_file.")
    if (
        args.validation_start_step < 0
        or args.validation_patience < 0
        or args.validation_min_steps < 0
    ):
        raise ValueError(
            "validation_start_step, validation_patience, and validation_min_steps must be nonnegative."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(hours=6))
    set_seed(args.seed + rank)
    output_dir = Path(args.output_dir)
    runtime_dir = output_dir / "runtime"
    rollout_model_dir = output_dir / "rollout_model"
    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "train_metrics.jsonl"
    compact_metrics_path = (
        Path(args.compact_log_file)
        if args.compact_log_file
        else output_dir / "train_compact.log"
    )
    validation_metrics_path = (
        Path(args.validation_metrics_file)
        if args.validation_metrics_file
        else output_dir / "validation_metrics.jsonl"
    )
    best_model_dir = Path(args.best_model_dir) if args.best_model_dir else output_dir / "best_model"
    periodic_model_save_dir = (
        Path(args.periodic_model_save_dir)
        if periodic_model_save_enabled
        else None
    )
    if rank == 0:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        compact_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if validation_enabled:
            validation_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if periodic_model_save_dir is not None:
            periodic_model_save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=control_group)

    resume_checkpoint_path: Path | None = None
    resume_manifest: Dict[str, Any] = {}
    if args.resume_state_from:
        resume_payload: List[Any] = [None, None]
        if rank == 0:
            resolved_checkpoint, loaded_manifest = load_resume_manifest(
                args.resume_state_from
            )
            resume_payload = [str(resolved_checkpoint), loaded_manifest]
        dist.broadcast_object_list(resume_payload, src=0, group=control_group)
        resume_checkpoint_path = Path(str(resume_payload[0]))
        resume_manifest = dict(resume_payload[1])
        completed_step = int(resume_manifest["completed_step"])
        for name in ("epoch", "shuffle_seed", "data_cursor"):
            if name not in resume_manifest:
                raise ValueError(
                    f"Exact resume manifest is missing sampler field {name}; use a checkpoint in the new format."
                )
        if args.step_offset not in (0, completed_step):
            raise ValueError(
                "step_offset and resume completed_step mismatch: %d != %d"
                % (args.step_offset, completed_step)
            )
        args.step_offset = completed_step
        saved_scheduler_total_steps = int(
            resume_manifest["fingerprint"]["critical_args"]["scheduler_total_steps"]
        )
        if args.scheduler_total_steps <= 0:
            args.scheduler_total_steps = saved_scheduler_total_steps
        elif args.scheduler_total_steps != saved_scheduler_total_steps:
            raise ValueError(
                "scheduler_total_steps and resume mismatch: %d != %d"
                % (args.scheduler_total_steps, saved_scheduler_total_steps)
            )
    if args.scheduler_total_steps <= 0:
        raise ValueError("scheduler_total_steps must be positive.")

    actor, tokenizer, initial_model_path = load_fsdp_model(args, device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, warmup_steps = build_warmup_scheduler(
        optimizer,
        args.scheduler_total_steps,
        args.warmup_ratio,
    )
    resume_fingerprint = build_resume_fingerprint(args, prompt_protocol, world_size)
    resumed_trainer_state: Dict[str, Any] = {}
    if resume_checkpoint_path is not None:
        resumed_trainer_state = load_exact_resume_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            manifest=resume_manifest,
            actor=actor,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            rank=rank,
            world_size=world_size,
            control_group=control_group,
            fingerprint=resume_fingerprint,
            best_model_dir=best_model_dir,
        )
    samples = load_samples(args.train_file)
    validation_samples = load_samples(args.validation_file) if validation_enabled else []
    if len(samples) < args.effective_batch_size:
        raise ValueError("The training dataset is smaller than effective_batch_size.")
    if validation_enabled and not validation_samples:
        raise ValueError("validation_file contains no usable samples.")
    if resumed_trainer_state:
        sampler_state = resumed_trainer_state.get("data_sampler_state")
        if not isinstance(sampler_state, dict):
            raise ValueError(
                "Exact resume checkpoint is missing data_sampler_state, so sample order cannot be guaranteed; "
                "restart training or use a checkpoint containing sampler state."
            )
        data_sampler = EpochShuffleSampler.from_state_dict(
            sampler_state,
            dataset_size=len(samples),
            batch_size=args.effective_batch_size,
            shuffle_seed=args.seed,
        )
        # 顶层副本便于人工审计，必须和采样器完整状态保持一致。
        for name in ("epoch", "shuffle_seed", "data_cursor"):
            if name not in resumed_trainer_state:
                raise ValueError(f"Exact resume checkpoint is missing sampler fields: {name}")
            if int(resumed_trainer_state[name]) != int(sampler_state[name]):
                raise ValueError(f"Exact resume checkpoint sampler fields differ: {name}")
    else:
        data_sampler = EpochShuffleSampler.from_consumed_samples(
            dataset_size=len(samples),
            batch_size=args.effective_batch_size,
            shuffle_seed=args.seed,
            consumed_samples=args.step_offset * args.effective_batch_size,
        )
    expected_consumed_samples = args.step_offset * args.effective_batch_size
    restored_consumed_samples = (
        data_sampler.epoch * len(samples) + data_sampler.data_cursor
    )
    if restored_consumed_samples != expected_consumed_samples:
        raise ValueError(
            "Sampler cursor and completed_step mismatch: checkpoint=%d expected=%d"
            % (restored_consumed_samples, expected_consumed_samples)
        )
    local_prompt_count = args.effective_batch_size // world_size
    if rank == 0:
        print(
            "fsdp_world_size=%d effective_batch_size=%d prompts_per_rank=%d G=%d rollouts_per_step=%d"
            % (
                world_size,
                args.effective_batch_size,
                local_prompt_count,
                args.num_generations,
                args.effective_batch_size * args.num_generations,
            ),
            flush=True,
        )
        print("metrics_file=%s" % metrics_path, flush=True)
        print("compact_log_file=%s" % compact_metrics_path, flush=True)
        print(
            "data_sampler=epoch_shuffle dataset_size=%d batch_size=%d epoch=%d "
            "shuffle_seed=%d data_cursor=%d"
            % (
                len(samples),
                args.effective_batch_size,
                data_sampler.epoch,
                data_sampler.shuffle_seed,
                data_sampler.data_cursor,
            ),
            flush=True,
        )
        if validation_enabled:
            print(
                "validation_file=%s validation_samples=%d interval=%d start_step=%d "
                "patience=%d min_steps=%d"
                % (
                    args.validation_file,
                    len(validation_samples),
                    args.validation_interval,
                    args.validation_start_step,
                    args.validation_patience,
                    args.validation_min_steps,
                ),
                flush=True,
            )
            print("validation_metrics_file=%s" % validation_metrics_path, flush=True)
        print(
            "best_model_dir=%s best_model_metric=%s"
            % (
                best_model_dir,
                "validation_answer_accuracy" if validation_enabled else args.best_model_metric,
            ),
            flush=True,
        )
        print(
            "warmup_ratio=%.4f warmup_steps=%d scheduler_total_steps=%d"
            % (args.warmup_ratio, warmup_steps, args.scheduler_total_steps),
            flush=True,
        )
        print(
            "resume_state_from=%s resume_state_dir=%s resume_state_save_interval=%d"
            % (
                str(resume_checkpoint_path or ""),
                args.resume_state_dir,
                args.resume_state_save_interval,
            ),
            flush=True,
        )
        print(
            "periodic_model_save_dir=%s start_step=%d interval=%d"
            % (
                str(periodic_model_save_dir or ""),
                args.periodic_model_save_start_step,
                args.periodic_model_save_interval,
            ),
            flush=True,
        )
        print(
            "prompt_config=%s prompt_protocol_hash=%s dataset=%s boundary_mode=%s"
            % (
                prompt_protocol.config_path,
                prompt_protocol.protocol_hash,
                prompt_protocol.dataset or args.dataset,
                prompt_protocol.boundary_mode or args.boundary_mode,
            ),
            flush=True,
        )
        print(
            "training_step_range=%d-%d"
            % (args.step_offset + 1, args.step_offset + args.steps),
            flush=True,
        )

    policy_model_path = initial_model_path
    if resumed_trainer_state:
        # DCP 权重恢复后先导出 Hugging Face 策略，供续训第一步 vLLM rollout 使用。
        save_rollout_policy(actor, tokenizer, rollout_model_dir, rank, control_group)
        policy_model_path = str(rollout_model_dir)
    rolling_step_seconds = resumed_trainer_state.get("rolling_step_seconds")
    best_score = float(resumed_trainer_state.get("best_score", float("-inf")))
    best_evaluation_step = int(resumed_trainer_state.get("best_evaluation_step", 0))
    saved_validation_tuple = resumed_trainer_state.get("best_validation_tuple")
    if saved_validation_tuple is None:
        best_validation_tuple: tuple[float, ...] = (
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
        )
    else:
        best_validation_tuple = tuple(float(value) for value in saved_validation_tuple)
    best_validation_accuracy = float(
        resumed_trainer_state.get("best_validation_accuracy", float("-inf"))
    )
    validation_evaluations_without_improvement = int(
        resumed_trainer_state.get("validation_evaluations_without_improvement", 0)
    )
    best_metadata_path = best_model_dir / "best_metrics.json"
    if not resumed_trainer_state and rank == 0 and best_metadata_path.is_file():
        previous_best = load_json(best_metadata_path)
        if validation_enabled and previous_best.get("selection_metric") == "validation_answer_accuracy":
            previous_tuple = previous_best.get("selection_tuple")
            if isinstance(previous_tuple, list) and len(previous_tuple) == 4:
                best_validation_tuple = tuple(float(value) for value in previous_tuple)
            else:
                best_validation_tuple = validation_selection_tuple(previous_best)
            best_validation_accuracy = best_validation_tuple[0]
            best_score = best_validation_tuple[0]
            best_evaluation_step = int(previous_best.get("evaluation_step", 0))
        elif not validation_enabled and previous_best.get("selection_metric") == args.best_model_metric:
            best_score = float(previous_best.get("selection_score", best_score))
            best_evaluation_step = int(previous_best.get("evaluation_step", 0))

    if validation_enabled and not resumed_trainer_state and not getattr(args, "skip_initial_validation", False):
        # step=0 先建立未强化 SFT 基线，允许最终 best 保持为原始模型。
        torch.cuda.empty_cache()
        if rank == 0:
            initial_validation, best_validation_tuple, initial_is_best = (
                evaluate_validation_checkpoint(
                    args,
                    policy_model_path,
                    validation_samples,
                    prompt_protocol,
                    tokenizer,
                    runtime_dir,
                    validation_metrics_path,
                    compact_metrics_path,
                    best_model_dir,
                    0,
                    best_validation_tuple,
                )
            )
            initial_accuracy = float(initial_validation["validation_answer_accuracy"])
            best_validation_accuracy = max(best_validation_accuracy, initial_accuracy)
            best_score = best_validation_tuple[0]
            if initial_is_best:
                best_evaluation_step = 0
        dist.barrier(group=control_group)

    for local_step in range(1, args.steps + 1):
        # 续训使用全局步号，确保数据抽样、rollout 随机种子和日志不会从第一步重复。
        step = args.step_offset + local_step
        step_started_at = time.perf_counter()
        request_path = runtime_dir / "rollout_request.json"
        rollout_path = runtime_dir / "rollout_output.json"
        worker_dir = runtime_dir / "rollout_workers"
        # 所有 rank 同步推进采样状态，rank 0 负责落盘 rollout 请求。
        data_epoch = data_sampler.epoch
        data_cursor_start = data_sampler.data_cursor
        selected_indices = data_sampler.next_batch()
        if rank == 0:
            write_json_atomic(
                request_path,
                build_rollout_request(
                    samples,
                    selected_indices,
                    prompt_protocol,
                    tokenizer,
                ),
            )
            if rollout_path.exists():
                rollout_path.unlink()
        dist.barrier(group=control_group)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if rank == 0:
            run_streaming_rollout(
                args,
                policy_model_path,
                request_path,
                rollout_path,
                worker_dir,
                step,
            )
        dist.barrier(group=control_group)
        rollout_result = load_json(rollout_path)
        rollout_items = rollout_result.get("items") or []
        if len(rollout_items) != args.effective_batch_size:
            raise RuntimeError("Rollout prompt count does not equal effective_batch_size.")
        local_items = rollout_items[rank::world_size]
        local_groups = [
            prepare_group(
                item,
                samples,
                tokenizer,
                args.num_generations,
                args.accuracy_reward_mode,
            )
            for item in local_items
        ]
        reward_tau = compute_shared_tau(local_groups, device, args.reward_tau)
        config = build_config(args, reward_tau)

        update_started_at = time.perf_counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        step_metrics = train_actor_step(
            actor,
            tokenizer,
            optimizer,
            local_groups,
            args,
            device,
            config,
        )
        scheduler.step()
        update_seconds = time.perf_counter() - update_started_at
        update_seconds_tensor = torch.tensor(update_seconds, dtype=torch.float64, device=device)
        dist.all_reduce(update_seconds_tensor, op=dist.ReduceOp.MAX)
        update_seconds = float(update_seconds_tensor.cpu())

        is_best_model = False
        candidate_score = float(step_metrics[args.best_model_metric])
        if not validation_enabled and rank == 0 and candidate_score > best_score:
            best_record = {
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "evaluation_step": step,
                "policy_update_step": step - 1,
                "selection_metric": args.best_model_metric,
                "selection_score": candidate_score,
                "evaluated_before_actor_update": True,
                "source_policy": str(Path(policy_model_path).resolve()),
                "reward": float(step_metrics["reward"]),
                "reward_raw": float(step_metrics["reward_raw"]),
                "answer_accuracy": float(step_metrics["answer_accuracy"]),
                "streaming_format_reward": float(step_metrics["streaming_format_reward"]),
                "deep_format_reward": float(step_metrics["deep_format_reward"]),
                "response_length": float(step_metrics["response_length"]),
                "truncation_rate": float(step_metrics["truncation_rate"]),
            }
            save_best_policy_snapshot(Path(policy_model_path), best_model_dir, best_record)
            best_score = candidate_score
            best_evaluation_step = step
            is_best_model = True

        save_started_at = time.perf_counter()
        save_rollout_policy(actor, tokenizer, rollout_model_dir, rank, control_group)
        save_seconds = time.perf_counter() - save_started_at
        save_seconds_tensor = torch.tensor(save_seconds, dtype=torch.float64, device=device)
        dist.all_reduce(save_seconds_tensor, op=dist.ReduceOp.MAX)
        save_seconds = float(save_seconds_tensor.cpu())
        policy_model_path = str(rollout_model_dir)

        periodic_model_save_seconds = 0.0
        periodic_model_checkpoint = ""
        should_save_periodic_model = should_save_periodic_policy(
            step,
            args.periodic_model_save_start_step,
            args.periodic_model_save_interval,
        )
        if should_save_periodic_model:
            periodic_model_save_started_at = time.perf_counter()
            if rank == 0:
                periodic_model_checkpoint = str(
                    save_periodic_policy_snapshot(
                        source_path=Path(policy_model_path),
                        snapshot_root=periodic_model_save_dir,
                        completed_step=step,
                        metadata={
                            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "policy_update_step": step,
                            "source_policy": str(Path(policy_model_path).resolve()),
                            "dataset": args.dataset,
                            "boundary_mode": args.boundary_mode,
                            "scheduler_total_steps": args.scheduler_total_steps,
                        },
                    )
                )
            dist.barrier(group=control_group)
            periodic_model_save_seconds = time.perf_counter() - periodic_model_save_started_at
            periodic_model_save_seconds_tensor = torch.tensor(
                periodic_model_save_seconds,
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(periodic_model_save_seconds_tensor, op=dist.ReduceOp.MAX)
            periodic_model_save_seconds = float(periodic_model_save_seconds_tensor.cpu())
            if rank == 0:
                print(
                    "periodic_model_checkpoint=%s completed_step=%d save_time=%s"
                    % (
                        periodic_model_checkpoint,
                        step,
                        format_duration(periodic_model_save_seconds),
                    ),
                    flush=True,
                )

        validation_metrics: Dict[str, Any] = {}
        stop_training = False
        should_validate = validation_enabled and should_evaluate_validation(
            step=step,
            scheduler_total_steps=args.scheduler_total_steps,
            interval=args.validation_interval,
            start_step=args.validation_start_step,
        )
        if should_validate:
            torch.cuda.empty_cache()
            if rank == 0:
                validation_metrics, best_validation_tuple, is_best_model = (
                    evaluate_validation_checkpoint(
                        args,
                        policy_model_path,
                        validation_samples,
                        prompt_protocol,
                        tokenizer,
                        runtime_dir,
                        validation_metrics_path,
                        compact_metrics_path,
                        best_model_dir,
                        step,
                        best_validation_tuple,
                    )
                )
                if is_best_model:
                    best_validation_accuracy = max(
                        best_validation_accuracy,
                        float(validation_metrics["validation_answer_accuracy"]),
                    )
                    validation_evaluations_without_improvement = 0
                else:
                    validation_evaluations_without_improvement += 1
                best_score = best_validation_tuple[0]
                if is_best_model:
                    best_evaluation_step = step
                stop_training = (
                    args.validation_patience > 0
                    and step >= args.validation_min_steps
                    and validation_evaluations_without_improvement
                    >= args.validation_patience
                )
            dist.barrier(group=control_group)
            stop_tensor = torch.tensor(int(stop_training), dtype=torch.int32, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(int(stop_tensor.cpu()))

        peak_allocated = torch.tensor(
            torch.cuda.max_memory_allocated(device) / (1024**3),
            dtype=torch.float64,
            device=device,
        )
        peak_reserved = torch.tensor(
            torch.cuda.max_memory_reserved(device) / (1024**3),
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)
        step_seconds = time.perf_counter() - step_started_at
        if rolling_step_seconds is None:
            rolling_step_seconds = step_seconds
        else:
            rolling_step_seconds = 0.8 * rolling_step_seconds + 0.2 * step_seconds

        if rank == 0:
            rollout_seconds = float(rollout_result.get("rollout_seconds", 0.0))
            remaining_seconds = rolling_step_seconds * max(args.steps - local_step, 0)
            metrics = {
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "step": step,
                "local_step": local_step,
                "step_offset": args.step_offset,
                "data_epoch": data_epoch,
                "data_cursor_start": data_cursor_start,
                "data_next_epoch": data_sampler.epoch,
                "data_cursor": data_sampler.data_cursor,
                "shuffle_seed": data_sampler.shuffle_seed,
                **step_metrics,
                "reward_tau": reward_tau,
                "learning_rate": learning_rate_used,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "warmup_ratio": args.warmup_ratio,
                "warmup_steps": warmup_steps,
                "scheduler_total_steps": args.scheduler_total_steps,
                "best_model_metric": (
                    "validation_answer_accuracy" if validation_enabled else args.best_model_metric
                ),
                "best_model_score": best_score,
                "best_model_evaluation_step": best_evaluation_step,
                "is_best_model": is_best_model,
                "validation_enabled": validation_enabled,
                "validation_evaluated": should_validate,
                "validation_start_step": args.validation_start_step,
                "validation_evaluations_without_improvement": (
                    validation_evaluations_without_improvement
                ),
                "early_stopping_triggered": stop_training,
                "effective_batch_size": args.effective_batch_size,
                "num_generations": args.num_generations,
                "policy_micro_batch_size": args.policy_micro_batch_size,
                "rollout_generation_batch_size": args.rollout_generation_batch_size,
                "rollout_gpu_memory_fraction": args.rollout_gpu_memory_fraction,
                "rollout_prefix_caching": args.rollout_prefix_caching,
                "accuracy_reward_mode": args.accuracy_reward_mode,
                "prompt_config": prompt_protocol.config_path,
                "prompt_protocol_hash": prompt_protocol.protocol_hash,
                "rollout_seconds": rollout_seconds,
                "actor_update_seconds": update_seconds,
                "policy_sync_seconds": save_seconds,
                "periodic_model_checkpoint_saved": should_save_periodic_model,
                "periodic_model_checkpoint": periodic_model_checkpoint,
                "periodic_model_save_seconds": periodic_model_save_seconds,
                "step_seconds": step_seconds,
                "eta_seconds": remaining_seconds,
                "actor_gpu_peak_allocated_gib": float(peak_allocated.cpu()),
                "actor_gpu_peak_reserved_gib": float(peak_reserved.cpu()),
                "rollout_worker_gpu_peaks": rollout_result.get("worker_gpu_peaks") or [],
                "rollout_termination_stats": rollout_result.get("termination_stats") or {},
                **{
                    key: value
                    for key, value in validation_metrics.items()
                    if key not in {"time", "is_best_model"}
                },
            }
            append_metrics(metrics_path, metrics)
            append_compact_metrics(compact_metrics_path, metrics)
            print(
                "step=%d loss=%.6f reward=%.6f reward_raw=%.6f advantage_mean=%.6f "
                "advantage_std=%.6f mean_abs_advantage=%.6f answer_accuracy=%.4f "
                "accuracy_active_groups=%.4f policy_entropy=%.6f sfmt=%.4f dfmt=%.4f "
                "response_len=%.1f truncation=%.4f grad=%.4f"
                % (
                    step,
                    metrics["loss"],
                    metrics["reward"],
                    metrics["reward_raw"],
                    metrics["advantage_mean"],
                    metrics["advantage_std"],
                    metrics["mean_abs_advantage"],
                    metrics["answer_accuracy"],
                    metrics["accuracy_active_group_rate"],
                    metrics["policy_entropy"],
                    metrics["streaming_format_reward"],
                    metrics["deep_format_reward"],
                    metrics["response_length"],
                    metrics["truncation_rate"],
                    metrics["grad_norm"],
                ),
                flush=True,
            )
            print(
                "step=%d timing rollout=%s actor=%s sync=%s total=%s eta=%s actor_gpu_peak=%.2f/%.2fGiB"
                % (
                    step,
                    format_duration(rollout_seconds),
                    format_duration(update_seconds),
                    format_duration(save_seconds),
                    format_duration(step_seconds),
                    format_duration(remaining_seconds),
                    metrics["actor_gpu_peak_allocated_gib"],
                    metrics["actor_gpu_peak_reserved_gib"],
                ),
                flush=True,
            )
            termination = metrics["rollout_termination_stats"]
            print(
                "step=%d rollout_termination stop=%d length=%d invalid_stop=%d "
                "avg_tokens=%.1f max_tokens=%d boundary_match=%.4f"
                % (
                    step,
                    int(termination.get("stop", 0)),
                    int(termination.get("length", 0)),
                    int(termination.get("invalid_stop", 0)),
                    float(termination.get("average_tokens", 0.0)),
                    int(termination.get("max_tokens", 0)),
                    float(termination.get("boundary_match_rate", 0.0)),
                ),
                flush=True,
            )

        should_save_resume_state = bool(args.resume_state_dir) and (
            local_step == args.steps
            or stop_training
            or (
                args.resume_state_save_interval > 0
                and step % args.resume_state_save_interval == 0
            )
        )
        if should_save_resume_state:
            checkpoint_started_at = time.perf_counter()
            latest_resume_path = save_exact_resume_checkpoint(
                checkpoint_root=Path(args.resume_state_dir),
                actor=actor,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                rank=rank,
                world_size=world_size,
                control_group=control_group,
                completed_step=step,
                trainer_state={
                    "epoch": data_sampler.epoch,
                    "shuffle_seed": data_sampler.shuffle_seed,
                    "data_cursor": data_sampler.data_cursor,
                    "data_sampler_state": data_sampler.state_dict(),
                    "best_score": best_score,
                    "best_evaluation_step": best_evaluation_step,
                    "best_validation_tuple": list(best_validation_tuple),
                    "best_validation_accuracy": best_validation_accuracy,
                    "validation_evaluations_without_improvement": (
                        validation_evaluations_without_improvement
                    ),
                    "rolling_step_seconds": rolling_step_seconds,
                    "args_snapshot": dict(vars(args)),
                },
                fingerprint=resume_fingerprint,
                best_model_dir=best_model_dir,
            )
            if rank == 0:
                print(
                    "resume_checkpoint=%s completed_step=%d save_time=%s"
                    % (
                        latest_resume_path,
                        step,
                        format_duration(time.perf_counter() - checkpoint_started_at),
                    ),
                    flush=True,
                )

        if stop_training:
            if rank == 0:
                print(
                    "early_stopping step=%d validation_evaluations_without_improvement=%d patience=%d"
                    % (
                        step,
                        validation_evaluations_without_improvement,
                        args.validation_patience,
                    ),
                    flush=True,
                )
            break

    if rank == 0:
        print("final_checkpoint=%s" % rollout_model_dir, flush=True)
        print(
            "best_checkpoint=%s best_model_metric=%s best_model_score=%.6f best_evaluation_step=%d"
            % (
                best_model_dir,
                "validation_answer_accuracy" if validation_enabled else args.best_model_metric,
                best_score,
                best_evaluation_step,
            ),
            flush=True,
        )
        if args.resume_state_dir:
            print(
                "resume_state_latest=%s"
                % (Path(args.resume_state_dir) / "latest"),
                flush=True,
            )
        if periodic_model_save_dir is not None:
            print("periodic_model_checkpoints=%s" % periodic_model_save_dir, flush=True)
    dist.barrier(group=control_group)
    dist.destroy_process_group(control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
