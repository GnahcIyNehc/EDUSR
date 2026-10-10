"""使用三视图归因与暴露成本奖励训练 EDU-BLHRPO。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ST_RL.EDU_HRPO.config import load_config
from ST_RL.EDU_HRPO.fsdp_integration import (
    EDUPreparedGroup,
    prepare_edu_group,
    run_with_edu_advantages,
)
from ST_RL.EDU_HRPO.trainer import EDUBLHRPOTrainerCore
from ST_RL.HRPO import train_hrpo_fsdp as hrpo_runner


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "edu_blhrpo.yaml"
_base_parse_args = hrpo_runner.parse_args
_base_prepare_group = hrpo_runner.prepare_group
_base_train_actor_step = hrpo_runner.train_actor_step
_base_format_compact_metrics = hrpo_runner.format_compact_metrics
_task_type = "classification"
_input_tokens_per_second = 3.33
_max_merge = 3
_enable_edu_controller = False
prepare_edu_group._base_prepare = _base_prepare_group


def parse_args() -> argparse.Namespace:
    """在稳定的 HRPO 参数之外解析 EDU 奖励与探针参数。"""
    global _task_type, _input_tokens_per_second, _max_merge, _enable_edu_controller
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--edu-config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--task-type",
        choices=("extractive", "classification", "math"),
        required=True,
    )
    parser.add_argument("--generation-tokens-per-second", type=float, required=True)
    parser.add_argument("--deep-budget", type=float, required=True)
    parser.add_argument("--input-tokens-per-second", type=float, default=3.33)
    parser.add_argument("--stream-probe-max-tokens", type=int, default=16)
    parser.add_argument("--no-stream-probe-max-tokens", type=int, default=512)
    parser.add_argument(
        "--enable-edu-controller",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    edu_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = _base_parse_args()
    finally:
        sys.argv = original_argv

    for name, value in vars(edu_args).items():
        setattr(args, name.replace("-", "_"), value)
    args.counterfactual_probes = True
    if args.reward_tau <= 0:
        raise ValueError("EDU training requires a calibrated, fixed positive tau via --reward-tau.")
    if args.generation_tokens_per_second <= 0:
        raise ValueError("generation_tokens_per_second must be positive.")
    if args.deep_budget < 0:
        raise ValueError("deep_budget must be nonnegative.")
    if args.input_tokens_per_second <= 0:
        raise ValueError("input_tokens_per_second must be positive.")
    if args.stream_probe_max_tokens <= 0 or args.no_stream_probe_max_tokens <= 0:
        raise ValueError("Counterfactual probe token limits must be positive.")
    if args.format_weight != 2.0:
        raise ValueError("EDU-HRPO requires format_weight=2.0.")
    config = load_config(args.edu_config)
    if args.enable_edu_controller is None:
        args.enable_edu_controller = config.enable_edu_controller
    else:
        config.enable_edu_controller = bool(args.enable_edu_controller)
    if config.rollouts_per_prompt != args.num_generations:
        raise ValueError(
            "EDU configuration G=%d differs from command-line num_generations=%d."
            % (config.rollouts_per_prompt, args.num_generations)
        )
    _task_type = args.task_type
    _input_tokens_per_second = args.input_tokens_per_second
    _max_merge = config.max_merge
    _enable_edu_controller = config.enable_edu_controller
    return args


def prepare_group(
    item: Dict[str, Any],
    samples: List[Any],
    tokenizer: Any,
    expected_generations: int,
    accuracy_reward_mode: str = "boxed",
) -> EDUPreparedGroup:
    """构造基础 actor 批次和对应的 EDU 奖励记录。"""
    return prepare_edu_group(
        item=item,
        samples=samples,
        tokenizer=tokenizer,
        expected_generations=expected_generations,
        accuracy_reward_mode=accuracy_reward_mode,
        task_type=_task_type,
        input_tokens_per_second=_input_tokens_per_second,
        max_merge=_max_merge,
        enable_edu_controller=_enable_edu_controller,
    )


def compute_shared_tau(
    groups: List[EDUPreparedGroup],
    device: torch.device,
    configured_tau: float,
) -> float:
    """EDU 暴露成本的 tau 必须预先校准并在整次训练中冻结。"""
    del groups, device
    if configured_tau <= 0:
        raise ValueError("EDU training requires a fixed reward_tau.")
    return float(configured_tau)


def build_config(args: argparse.Namespace, reward_tau: float):
    """加载论文配置，并注入由硬件和句子基线派生的三个量。"""
    config = load_config(args.edu_config)
    config.clip_range = args.clip_range
    config.kl_coef = args.kl_coef
    config.local_weight = args.local_weight
    config.generation_tokens_per_second = args.generation_tokens_per_second
    config.deep_budget = args.deep_budget
    config.reward_tau = reward_tau
    config.rollouts_per_prompt = args.num_generations
    config.enable_edu_controller = bool(args.enable_edu_controller)
    config.stream_probe_max_tokens = args.stream_probe_max_tokens
    config.no_stream_probe_max_tokens = args.no_stream_probe_max_tokens
    config.validate(require_derived=True)
    return config


def train_actor_step(
    actor: FSDP,
    tokenizer: Any,
    optimizer: Any,
    groups: List[EDUPreparedGroup],
    args: argparse.Namespace,
    device: torch.device,
    config: Any,
) -> Dict[str, float]:
    """复用 FSDP 更新器，但把基础优势替换成 EDU-BLHRPO 优势。"""
    return run_with_edu_advantages(
        train_function=_base_train_actor_step,
        trainer_core_type=EDUBLHRPOTrainerCore,
        actor=actor,
        tokenizer=tokenizer,
        optimizer=optimizer,
        groups=groups,
        args=args,
        device=device,
        config=config,
        advantage_module=hrpo_runner,
    )


def format_compact_metrics(metrics: Dict[str, float]) -> str:
    """在基础精简日志后追加三视图归因和暴露成本指标。"""
    return (
        _base_format_compact_metrics(metrics)
        + " qstream=%.4f qzero=%.4f qvalid=%.4f q0valid=%.4f "
        "qtrunc=%.4f q0trunc=%.4f part=%.4f stream_harm=%.4f "
        "deep_fix=%.4f deep_fix_cond=%.4f deep_regress=%.4f deep_regress_cond=%.4f "
        "hidden=%.1f exposed=%.1f think_rate=%.4f exposure=%.1f"
        % (
            metrics["stream_probe_accuracy"],
            metrics["no_stream_accuracy"],
            metrics["stream_probe_valid_rate"],
            metrics["no_stream_probe_valid_rate"],
            metrics["stream_probe_truncation_rate"],
            metrics["no_stream_probe_truncation_rate"],
            metrics["partial_score"],
            metrics["stream_harm_rate"],
            metrics["deep_correction_rate"],
            metrics["deep_correction_given_stream_wrong"],
            metrics["deep_regression_rate"],
            metrics["deep_regression_given_stream_correct"],
            metrics["hidden_stream_tokens"],
            metrics["exposed_stream_tokens"],
            metrics["think_rate"],
            metrics["exposure_cost"],
        )
    )


def main() -> None:
    """将 EDU 组件注入原训练基础设施并启动独立训练。"""
    hrpo_runner.parse_args = parse_args
    hrpo_runner.prepare_group = prepare_group
    hrpo_runner.compute_shared_tau = compute_shared_tau
    hrpo_runner.build_config = build_config
    hrpo_runner.train_actor_step = train_actor_step
    hrpo_runner.format_compact_metrics = format_compact_metrics
    hrpo_runner.main()


if __name__ == "__main__":
    main()
