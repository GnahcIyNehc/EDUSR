"""可直接运行的 AdaSR-HRPO 训练脚本。"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ST_RL.HRPO import HRPOConfig, compute_hrpo_advantages, hrpo_loss
from ST_RL.HRPO.rewards import deep_format_reward, exact_match_accuracy_reward
from ST_RL.common import phase_masks_from_lengths
from ST_RL.prompt_protocol import PromptProtocol, load_training_prompt_protocol
from ST_RL.training_utils import (
    GeneratedBatch,
    RLSample,
    build_hrpo_prompt,
    compute_response_log_probs_and_entropy_in_batches,
    compute_response_log_probs_in_batches,
    compute_token_lengths,
    count_question_stream_segments,
    generate_batch,
    load_samples,
    mean,
    parse_hrpo_response,
    sample_minibatch,
    save_checkpoint,
    set_seed,
)


@dataclass
class HRPORollout:
    """一个微批生成的轨迹及其原始奖励统计。"""

    generated: GeneratedBatch
    streaming_format_rewards: List[float]
    deep_format_rewards: List[float]
    accuracy_rewards: List[float]
    segment_length_rows: List[List[int]]
    streaming_token_lengths: List[int]
    deep_lengths: List[int]
    max_segments: int


def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal language model with AdaSR HRPO.")
    default_model = Path(__file__).resolve().parents[1] / "model"
    default_output = Path(__file__).resolve().parent / "runs" / "hrpo"
    parser.add_argument("--model-path", default=str(default_model))
    parser.add_argument("--train-file", required=True, help="JSON or JSONL training file.")
    parser.add_argument("--prompt-config", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--boundary-mode", choices=("sentence", "edu"), default="")
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--metrics-file", default="", help="Per-step JSONL metrics file; defaults to the output directory.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100, help="Number of optimizer updates.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts per rollout microbatch.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Rollout microbatches accumulated per parameter update.",
    )
    parser.add_argument("--num-generations", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
    parser.add_argument("--logprob-micro-batch-size", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--debug-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to reduce peak GPU memory.")
    parser.add_argument("--device-map", default="", help="Multi-GPU model device map, such as balanced, auto, or sequential.")
    parser.add_argument("--no-reference", action="store_true", help="Skip loading a reference model even when the KL coefficient is positive.")
    parser.add_argument("--resume-from-checkpoint", default="", help="Load model weights from an existing RL checkpoint to continue training.")
    return parser.parse_args()


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool = False,
    device_map: str = "",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = None
    if device.type == "cuda" and bf16:
        dtype = torch.bfloat16
    elif device.type == "cuda" and fp16:
        dtype = torch.float16
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not device_map:
        model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


def pad_segment_lengths(rows: List[List[int]], width: int) -> torch.Tensor:
    padded = []
    for row in rows:
        padded.append(row[:width] + [0] * max(width - len(row), 0))
    return torch.tensor(padded, dtype=torch.float32)


def pad_segment_mask(rows: List[List[int]], width: int) -> torch.Tensor:
    masks = []
    for row in rows:
        masks.append([True] * min(len(row), width) + [False] * max(width - len(row), 0))
    return torch.tensor(masks, dtype=torch.bool)


def sft_streaming_format_reward(streaming_segments: List[str], question_segments: int) -> float:
    if not streaming_segments:
        return 0.0
    for idx, segment in enumerate(streaming_segments):
        text = str(segment or "").strip()
        expected_token = "<EOQ>" if idx == question_segments - 1 else "<EOT>"
        if not text.endswith(expected_token):
            return 0.0
        content = text[: -len(expected_token)].strip()
        if not content and content != "<ignore>":
            return 0.0
    return 1.0


def collect_hrpo_rollout(
    model,
    tokenizer,
    minibatch: List[RLSample],
    args,
    device: torch.device,
    prompt_protocol: PromptProtocol,
) -> HRPORollout:
    """采样一个 prompt 微批，并在 CPU 侧保留奖励和长度信息。"""
    prompts = [
        build_hrpo_prompt(sample, prompt_protocol) + prompt_protocol.target_prefix
        for sample in minibatch
    ]
    model.eval()
    previous_use_cache = getattr(model.config, "use_cache", None)
    if previous_use_cache is not None:
        model.config.use_cache = True
    try:
        generated = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            num_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache

    expanded_samples = [minibatch[int(group_id)] for group_id in generated.group_ids.tolist()]
    streaming_format_rewards: List[float] = []
    deep_format_rewards: List[float] = []
    accuracy_rewards: List[float] = []
    segment_length_rows: List[List[int]] = []
    streaming_token_lengths: List[int] = []
    deep_lengths: List[int] = []
    expected_segment_counts = [
        count_question_stream_segments(sample.question) + len(sample.context_segments)
        for sample in expanded_samples
    ]
    question_segment_counts = [count_question_stream_segments(sample.question) for sample in expanded_samples]
    max_segments = max(expected_segment_counts + [1])

    for text, sample, expected_segments, question_segments in zip(
        generated.response_texts,
        expanded_samples,
        expected_segment_counts,
        question_segment_counts,
    ):
        streaming_segments, deep_text, full_text = parse_hrpo_response(text, expected_segments)
        required_segments = streaming_segments[:expected_segments] if expected_segments > 0 else streaming_segments
        if expected_segments == 0:
            stream_format = 1.0
        else:
            stream_format = float(len(streaming_segments) >= expected_segments) * sft_streaming_format_reward(
                required_segments,
                question_segments,
            )
        deep_format = deep_format_reward(deep_text)
        accuracy = exact_match_accuracy_reward(
            full_text,
            sample.answer,
            require_boxed=args.accuracy_reward_mode == "boxed",
        )
        segment_lengths = compute_token_lengths(tokenizer, required_segments)
        deep_length = compute_token_lengths(tokenizer, [deep_text])[0] if deep_text else 0

        streaming_format_rewards.append(stream_format)
        deep_format_rewards.append(deep_format)
        accuracy_rewards.append(accuracy)
        segment_length_rows.append(segment_lengths)
        streaming_token_lengths.append(sum(segment_lengths))
        deep_lengths.append(deep_length)

    return HRPORollout(
        generated=generated,
        streaming_format_rewards=streaming_format_rewards,
        deep_format_rewards=deep_format_rewards,
        accuracy_rewards=accuracy_rewards,
        segment_length_rows=segment_length_rows,
        streaming_token_lengths=streaming_token_lengths,
        deep_lengths=deep_lengths,
        max_segments=max_segments,
    )


def compose_reporting_rewards(advantages, config: HRPOConfig) -> Dict[str, torch.Tensor]:
    """按 HRPO 的局部与全局权重组合仅用于监控的标量奖励。"""
    components = advantages.components
    stream_valid = (components.streaming_format > 0.5).to(components.streaming_length_penalty.dtype)
    deep_valid = (components.deep_format > 0.5).to(components.deep_length_penalty.dtype)
    streaming_reward = (
        config.format_weight * components.streaming_format
        + config.length_weight * components.streaming_length_penalty * stream_valid
    )
    deep_reward = (
        config.format_weight * components.deep_format
        + config.length_weight * components.deep_length_penalty * deep_valid
    )
    global_reward = (
        config.accuracy_weight * components.accuracy
        + config.length_weight * components.efficiency
    )
    total_reward = global_reward + config.local_weight * (streaming_reward + deep_reward)
    reward_upper_bound = (
        config.accuracy_weight
        + config.length_weight
        + 2.0 * config.local_weight * config.format_weight
    )
    normalized_reward = total_reward / max(reward_upper_bound, config.normalize_eps)
    combined_advantage = advantages.global_ + config.local_weight * (
        advantages.streaming + advantages.deep
    )
    return {
        "reward": normalized_reward,
        "reward_raw": total_reward,
        "reward_normalized": normalized_reward,
        "advantage": combined_advantage,
        "streaming_reward": streaming_reward,
        "deep_reward": deep_reward,
        "global_reward": global_reward,
    }


def binary_group_reward_stats(values: Sequence[float]) -> Dict[str, float]:
    """统计一个二值奖励组是否提供组内相对优势信号。"""
    if not values:
        return {"active": 0.0, "all_correct": 0.0, "all_wrong": 0.0}
    minimum = min(values)
    maximum = max(values)
    return {
        "active": float(minimum < maximum),
        "all_correct": float(minimum >= 0.5),
        "all_wrong": float(maximum < 0.5),
    }


def append_metrics(metrics_path: Path, metrics: Dict[str, float]) -> None:
    """将单个优化步的指标追加到 JSONL 文件。"""
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def main():
    args = parse_args()
    prompt_protocol = load_training_prompt_protocol(
        prompt_config=args.prompt_config,
        dataset=args.dataset,
        boundary_mode=args.boundary_mode,
    )
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive.")
    if args.num_generations <= 1:
        raise ValueError("HRPO requires num_generations > 1 for group-relative normalization.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(args.train_file, max_samples=args.max_samples or None)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    if len(samples) < effective_batch_size:
        raise ValueError(
            "Training sample count %d is smaller than effective batch size %d." % (len(samples), effective_batch_size)
        )

    effective_model_path = args.resume_from_checkpoint or args.model_path
    print("initial_model=%s" % effective_model_path, flush=True)
    print(
        "batch_size=%d gradient_accumulation_steps=%d effective_batch_size=%d num_generations=%d rollouts_per_step=%d"
        % (
            args.batch_size,
            args.gradient_accumulation_steps,
            effective_batch_size,
            args.num_generations,
            effective_batch_size * args.num_generations,
        ),
        flush=True,
    )
    model, tokenizer = load_model_and_tokenizer(
        effective_model_path,
        device,
        args.bf16,
        args.fp16,
        args.gradient_checkpointing,
        args.device_map,
    )
    ref_model = None
    if args.kl_coef > 0 and not args.no_reference:
        ref_model, _ = load_model_and_tokenizer(
            effective_model_path,
            device,
            args.bf16,
            args.fp16,
            device_map=args.device_map,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_config = HRPOConfig(
        clip_range=args.clip_range,
        kl_coef=args.kl_coef if ref_model is not None else 0.0,
        local_weight=args.local_weight,
        accuracy_weight=args.accuracy_weight,
        format_weight=args.format_weight,
        length_weight=args.length_weight,
        latency_alpha=args.latency_alpha,
        reward_tau=args.reward_tau if args.reward_tau > 0 else None,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "train_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    print("metrics_file=%s" % metrics_path, flush=True)

    for step in range(1, args.steps + 1):
        step_started_at = time.perf_counter()
        step_samples = sample_minibatch(samples, effective_batch_size)
        rollouts: List[HRPORollout] = []
        rollout_accuracy_values: List[float] = []
        progress_every = max(args.gradient_accumulation_steps // 4, 1)
        for accumulation_index in range(args.gradient_accumulation_steps):
            start = accumulation_index * args.batch_size
            end = start + args.batch_size
            rollout = collect_hrpo_rollout(
                model,
                tokenizer,
                step_samples[start:end],
                args,
                device,
                prompt_protocol,
            )
            rollouts.append(rollout)
            rollout_accuracy_values.extend(rollout.accuracy_rewards)
            completed = accumulation_index + 1
            if completed == 1 or completed % progress_every == 0 or completed == args.gradient_accumulation_steps:
                print(
                    "step=%d rollout_progress=%d/%d sampled_prompts=%d/%d running_answer_accuracy=%.4f"
                    % (
                        step,
                        completed,
                        args.gradient_accumulation_steps,
                        completed * args.batch_size,
                        effective_batch_size,
                        mean(rollout_accuracy_values),
                    ),
                    flush=True,
                )

        all_total_lengths = [
            streaming_length + deep_length
            for rollout in rollouts
            for streaming_length, deep_length in zip(rollout.streaming_token_lengths, rollout.deep_lengths)
        ]
        step_tau = args.reward_tau if args.reward_tau > 0 else max(mean(all_total_lengths), 1.0)
        config = replace(base_config, reward_tau=step_tau)

        metric_sums: Dict[str, float] = {
            "loss": 0.0,
            "reward": 0.0,
            "reward_raw": 0.0,
            "advantage": 0.0,
            "advantage_square": 0.0,
            "advantage_abs": 0.0,
            "streaming_reward": 0.0,
            "deep_reward": 0.0,
            "global_reward": 0.0,
            "answer_accuracy": 0.0,
            "streaming_format_reward": 0.0,
            "deep_format_reward": 0.0,
            "kl": 0.0,
            "streaming_length": 0.0,
            "deep_length": 0.0,
            "streaming_length_penalty": 0.0,
            "deep_length_penalty": 0.0,
            "efficiency_reward": 0.0,
        }
        total_rollouts = 0
        accuracy_group_count = 0
        accuracy_active_group_count = 0.0
        accuracy_all_correct_group_count = 0.0
        accuracy_all_wrong_group_count = 0.0
        entropy_sum = 0.0
        entropy_token_count = 0
        optimizer.zero_grad(set_to_none=True)

        for rollout in rollouts:
            generated = rollout.generated
            model.eval()
            old_log_probs, response_mask = compute_response_log_probs_in_batches(
                model,
                tokenizer,
                generated.prompt_ids,
                generated.response_ids,
                device,
                require_grad=False,
                micro_batch_size=args.logprob_micro_batch_size,
            )
            ref_log_probs = None
            if ref_model is not None:
                ref_log_probs, _ = compute_response_log_probs_in_batches(
                    ref_model,
                    tokenizer,
                    generated.prompt_ids,
                    generated.response_ids,
                    device,
                    require_grad=False,
                    micro_batch_size=args.logprob_micro_batch_size,
                )

            model.train()
            current_log_probs, response_mask, token_entropy = compute_response_log_probs_and_entropy_in_batches(
                model,
                tokenizer,
                generated.prompt_ids,
                generated.response_ids,
                device,
                require_grad=True,
                micro_batch_size=args.logprob_micro_batch_size,
            )
            loss_device = current_log_probs.device
            old_log_probs = old_log_probs.to(loss_device)
            response_mask = response_mask.to(loss_device)
            token_entropy = token_entropy.to(loss_device)
            if ref_log_probs is not None:
                ref_log_probs = ref_log_probs.to(loss_device)

            response_lengths = response_mask.sum(dim=-1).long().cpu()
            stream_lengths_tensor = torch.tensor(rollout.streaming_token_lengths, dtype=torch.long)
            stream_lengths_tensor = torch.minimum(stream_lengths_tensor, response_lengths)
            deep_mask_lengths = response_lengths - stream_lengths_tensor
            streaming_mask, deep_mask, response_mask_from_lengths = phase_masks_from_lengths(
                stream_lengths_tensor.to(loss_device),
                deep_mask_lengths.to(loss_device),
                max_length=current_log_probs.shape[1],
            )
            response_mask = response_mask & response_mask_from_lengths

            segment_lengths_tensor = pad_segment_lengths(
                rollout.segment_length_rows,
                rollout.max_segments,
            ).to(loss_device)
            segment_mask_tensor = pad_segment_mask(
                rollout.segment_length_rows,
                rollout.max_segments,
            ).to(loss_device)
            deep_lengths_tensor = torch.tensor(rollout.deep_lengths, dtype=torch.float32, device=loss_device)
            advantages = compute_hrpo_advantages(
                streaming_format=torch.tensor(
                    rollout.streaming_format_rewards,
                    dtype=torch.float32,
                    device=loss_device,
                ),
                deep_format=torch.tensor(
                    rollout.deep_format_rewards,
                    dtype=torch.float32,
                    device=loss_device,
                ),
                accuracy=torch.tensor(
                    rollout.accuracy_rewards,
                    dtype=torch.float32,
                    device=loss_device,
                ),
                streaming_segment_lengths=segment_lengths_tensor,
                streaming_segment_mask=segment_mask_tensor,
                deep_lengths=deep_lengths_tensor,
                group_ids=generated.group_ids.to(loss_device),
                config=config,
            )
            loss, loss_metrics = hrpo_loss(
                log_probs=current_log_probs,
                old_log_probs=old_log_probs,
                response_mask=response_mask,
                streaming_mask=streaming_mask,
                deep_mask=deep_mask,
                streaming_advantages=advantages.streaming,
                deep_advantages=advantages.deep,
                global_advantages=advantages.global_,
                ref_log_probs=ref_log_probs,
                config=config,
            )
            (loss / args.gradient_accumulation_steps).backward()

            reporting_rewards = compose_reporting_rewards(advantages, config)
            rollout_count = len(generated.response_ids)
            total_rollouts += rollout_count
            metric_sums["loss"] += _as_float(loss) * rollout_count
            advantage_values = reporting_rewards["advantage"]
            metric_sums["reward"] += _as_float(reporting_rewards["reward"].sum())
            metric_sums["reward_raw"] += _as_float(reporting_rewards["reward_raw"].sum())
            metric_sums["advantage"] += _as_float(advantage_values.sum())
            metric_sums["advantage_square"] += _as_float(advantage_values.square().sum())
            metric_sums["advantage_abs"] += _as_float(advantage_values.abs().sum())
            metric_sums["streaming_reward"] += _as_float(reporting_rewards["streaming_reward"].sum())
            metric_sums["deep_reward"] += _as_float(reporting_rewards["deep_reward"].sum())
            metric_sums["global_reward"] += _as_float(reporting_rewards["global_reward"].sum())
            metric_sums["answer_accuracy"] += sum(rollout.accuracy_rewards)
            metric_sums["streaming_format_reward"] += sum(rollout.streaming_format_rewards)
            metric_sums["deep_format_reward"] += sum(rollout.deep_format_rewards)
            metric_sums["kl"] += _as_float(loss_metrics["kl"]) * rollout_count
            metric_sums["streaming_length"] += _as_float(
                loss_metrics["streaming_len_mean"]
            ) * rollout_count
            metric_sums["deep_length"] += _as_float(loss_metrics["deep_len_mean"]) * rollout_count
            metric_sums["streaming_length_penalty"] += _as_float(
                advantages.components.streaming_length_penalty.sum()
            )
            metric_sums["deep_length_penalty"] += _as_float(
                advantages.components.deep_length_penalty.sum()
            )
            metric_sums["efficiency_reward"] += _as_float(advantages.components.efficiency.sum())
            group_ids = generated.group_ids.tolist()
            for group_id in sorted(set(group_ids)):
                group_values = [
                    value
                    for value, current_group_id in zip(rollout.accuracy_rewards, group_ids)
                    if current_group_id == group_id
                ]
                group_stats = binary_group_reward_stats(group_values)
                accuracy_group_count += 1
                accuracy_active_group_count += group_stats["active"]
                accuracy_all_correct_group_count += group_stats["all_correct"]
                accuracy_all_wrong_group_count += group_stats["all_wrong"]
            entropy_sum += _as_float((token_entropy * response_mask.to(token_entropy.dtype)).sum())
            entropy_token_count += int(response_mask.sum().detach().cpu())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        advantage_mean = metric_sums["advantage"] / total_rollouts
        advantage_variance = max(
            metric_sums["advantage_square"] / total_rollouts - advantage_mean**2,
            0.0,
        )
        metrics = {
            "step": step,
            "loss": metric_sums["loss"] / total_rollouts,
            "reward": metric_sums["reward"] / total_rollouts,
            "reward_raw": metric_sums["reward_raw"] / total_rollouts,
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_variance**0.5,
            "mean_abs_advantage": metric_sums["advantage_abs"] / total_rollouts,
            "answer_accuracy": metric_sums["answer_accuracy"] / total_rollouts,
            "policy_entropy": entropy_sum / max(entropy_token_count, 1),
            "streaming_reward": metric_sums["streaming_reward"] / total_rollouts,
            "deep_reward": metric_sums["deep_reward"] / total_rollouts,
            "global_reward": metric_sums["global_reward"] / total_rollouts,
            "streaming_format_reward": metric_sums["streaming_format_reward"] / total_rollouts,
            "deep_format_reward": metric_sums["deep_format_reward"] / total_rollouts,
            "kl": metric_sums["kl"] / total_rollouts,
            "streaming_length": metric_sums["streaming_length"] / total_rollouts,
            "deep_length": metric_sums["deep_length"] / total_rollouts,
            "streaming_length_penalty": metric_sums["streaming_length_penalty"] / total_rollouts,
            "deep_length_penalty": metric_sums["deep_length_penalty"] / total_rollouts,
            "efficiency_reward": metric_sums["efficiency_reward"] / total_rollouts,
            "accuracy_active_group_rate": accuracy_active_group_count / max(accuracy_group_count, 1),
            "accuracy_all_correct_group_rate": accuracy_all_correct_group_count
            / max(accuracy_group_count, 1),
            "accuracy_all_wrong_group_rate": accuracy_all_wrong_group_count
            / max(accuracy_group_count, 1),
            "accuracy_reward_mode": args.accuracy_reward_mode,
            "reward_tau": step_tau,
            "grad_norm": _as_float(grad_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "num_generations": args.num_generations,
            "rollouts": total_rollouts,
            "entropy_token_count": entropy_token_count,
            "elapsed_seconds": time.perf_counter() - step_started_at,
        }
        append_metrics(metrics_path, metrics)

        if step % args.log_every == 0:
            print(
                "step=%d loss=%.6f reward=%.6f reward_raw=%.6f advantage_mean=%.6f "
                "advantage_std=%.6f mean_abs_advantage=%.6f answer_accuracy=%.4f "
                "accuracy_active_groups=%.4f policy_entropy=%.6f sfmt=%.4f dfmt=%.4f "
                "kl=%.6f slen=%.2f dlen=%.2f tau=%.2f grad=%.4f time=%.1fs"
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
                    metrics["kl"],
                    metrics["streaming_length"],
                    metrics["deep_length"],
                    metrics["reward_tau"],
                    metrics["grad_norm"],
                    metrics["elapsed_seconds"],
                ),
                flush=True,
            )
            for sample_idx, text in enumerate(
                rollouts[0].generated.response_texts[: max(args.debug_samples, 0)]
            ):
                preview = text.replace("\n", "\\n")[:1000]
                print("debug_sample[%d]=%s" % (sample_idx, preview), flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            path = save_checkpoint(model, tokenizer, args.output_dir, step)
            print("saved=%s" % path, flush=True)

    final_path = save_checkpoint(model, tokenizer, args.output_dir, args.steps)
    print("final_checkpoint=%s" % final_path, flush=True)


if __name__ == "__main__":
    main()
