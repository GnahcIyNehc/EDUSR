"""把 EDU 三视图奖励接入现有 vLLM/FSDP 训练基础设施。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type

import torch
import torch.distributed as dist

from ST_RL.EDU_HRPO.controller import ControlAction, EDUBoundaryEvent, EDUGroupRecord
from ST_RL.EDU_HRPO.rewards import EDUBLHRPOAdvantages
from ST_RL.EDU_HRPO.rollout import (
    DeepRecord,
    EDURolloutRecord,
    ProbeRecord,
    ScoreRecord,
)
from ST_RL.EDU_HRPO.task_scorers import (
    ClassificationScorer,
    ExtractiveQAScorer,
    MathTaskScorer,
)
from ST_RL.EDU_HRPO.trainer import reward_batch_from_records
from ST_RL.HRPO import train_hrpo_fsdp as hrpo_runner
from ST_RL.training_utils import RLSample, parse_hrpo_response


BOUNDARY_RE = re.compile(r"(?:<EOT>|<EOQ>|<EOP>)\s*$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class EDUPreparedGroup(hrpo_runner.PreparedGroup):
    """在基础 actor 批次旁保存可审计的 EDU 奖励记录。"""

    records: List[EDURolloutRecord]


@dataclass
class EDUAdvantagesAdapter:
    """为既有 HRPO 更新器提供兼容字段，同时保留 EDU 原始分量。"""

    streaming: torch.Tensor
    deep: torch.Tensor
    global_: torch.Tensor
    components: Any
    edu_advantages: EDUBLHRPOAdvantages
    advantage: torch.Tensor
    reward_raw: torch.Tensor
    streaming_reward: torch.Tensor
    deep_reward: torch.Tensor
    global_reward: torch.Tensor
    reward_batch: Any


def build_task_scorer(task_type: str):
    """根据独立数据集选择确定性评分器。"""
    if task_type == "extractive":
        return ExtractiveQAScorer()
    if task_type == "classification":
        return ClassificationScorer()
    if task_type == "math":
        return MathTaskScorer(require_boxed=True)
    raise ValueError("Unknown task_type: %s" % task_type)


def _lexical_relevance(question: str, edu_text: str) -> float:
    """在没有独立句向量服务时使用冻结的词项重叠分数分配 EDU 预算。"""
    question_tokens = set(WORD_RE.findall(str(question).lower()))
    edu_tokens = set(WORD_RE.findall(str(edu_text).lower()))
    if not question_tokens or not edu_tokens:
        return 0.0
    return min(len(question_tokens & edu_tokens) / float(len(edu_tokens)), 1.0)


def _split_stage_text(text: str) -> Tuple[ControlAction, str]:
    """解析兼容 SFT 的 ignore 以及新增 skip/defer/think 控制前缀。"""
    content = BOUNDARY_RE.sub("", str(text or "")).strip()
    lowered = content.lower()
    if lowered.startswith("<defer>"):
        return ControlAction.DEFER, content[len("<defer>") :].strip()
    if lowered.startswith("<skip>"):
        return ControlAction.SKIP, content[len("<skip>") :].strip()
    if lowered.startswith("<ignore>"):
        return ControlAction.SKIP, content[len("<ignore>") :].strip()
    if lowered.startswith("<think>"):
        content = content[len("<think>") :].strip()
        if content.lower().endswith("</think>"):
            content = content[: -len("</think>")].strip()
    return ControlAction.THINK, content


def _split_sft_stage_text(text: str) -> Tuple[ControlAction, str]:
    """固定分组模式只识别原 SFT 的 ignore，不启用新增控制动作。"""

    content = BOUNDARY_RE.sub("", str(text or "")).strip()
    if content.lower().startswith("<ignore>"):
        return ControlAction.SKIP, ""
    return ControlAction.THINK, content


def _build_events(
    item: Dict[str, Any],
    input_tokens_per_second: float,
) -> List[EDUBoundaryEvent]:
    source_segments = [str(value) for value in item["source_segments"]][1:]
    source_token_ids = item["source_segment_token_ids"][1:]
    if len(source_segments) != len(source_token_ids):
        raise RuntimeError("Source text and token segment counts differ.")
    elapsed = 0.0
    boundary_after = 0
    events = []
    for edu_id, (text, token_ids) in enumerate(zip(source_segments, source_token_ids)):
        token_count = len(token_ids)
        boundary_after += token_count
        elapsed += token_count / float(input_tokens_per_second)
        events.append(
            EDUBoundaryEvent(
                edu_id=edu_id,
                text=text,
                input_token_count=token_count,
                boundary_after=boundary_after,
                emitted_at=elapsed,
                visible_at=elapsed,
            )
        )
    return events


def _build_dynamic_groups(
    events: Sequence[EDUBoundaryEvent],
    stage_texts: Sequence[str],
    control_entropies: Sequence[float],
    tokenizer: Any,
    question: str,
    max_merge: int,
) -> List[EDUGroupRecord]:
    """把逐边界输出还原成 skip/defer/think 动态 EDU 组。"""
    if len(events) != len(stage_texts):
        raise RuntimeError("EDU event and streaming response segment counts differ.")
    if len(control_entropies) != len(stage_texts):
        raise RuntimeError("Control entropy and streaming response segment counts differ.")
    groups = []
    pending = []
    for event_index, (event, stage_text) in enumerate(zip(events, stage_texts)):
        pending.append(event)
        action, thought = _split_stage_text(stage_text)
        forced = action == ControlAction.DEFER and len(pending) >= max_merge
        if action == ControlAction.DEFER and not forced:
            continue
        if forced:
            action = ControlAction.SKIP
            thought = ""
        if action == ControlAction.SKIP:
            thought = ""
        thought_tokens = len(
            tokenizer(thought, add_special_tokens=False)["input_ids"]
        ) if thought else 0
        boundary_time = pending[-1].emitted_at
        next_boundary_time = (
            events[event_index + 1].emitted_at
            if event_index + 1 < len(events)
            else boundary_time
        )
        relevance = max(
            (_lexical_relevance(question, value.text) for value in pending),
            default=0.0,
        )
        groups.append(
            EDUGroupRecord(
                edu_ids=[value.edu_id for value in pending],
                action=action,
                input_token_count=sum(value.input_token_count for value in pending),
                boundary_time=boundary_time,
                next_boundary_time=next_boundary_time,
                thought=thought,
                token_count=thought_tokens,
                control_entropy=float(control_entropies[event_index]),
                relevance=relevance,
                forced_by_merge_limit=forced,
            )
        )
        pending = []
    if pending:
        boundary_time = pending[-1].emitted_at
        groups.append(
            EDUGroupRecord(
                edu_ids=[value.edu_id for value in pending],
                action=ControlAction.SKIP,
                input_token_count=sum(value.input_token_count for value in pending),
                boundary_time=boundary_time,
                next_boundary_time=boundary_time,
                relevance=max(
                    (_lexical_relevance(question, value.text) for value in pending),
                    default=0.0,
                ),
                forced_by_merge_limit=True,
            )
        )
    return groups


def _build_fixed_groups(
    events: Sequence[EDUBoundaryEvent],
    stage_texts: Sequence[str],
    tokenizer: Any,
    question: str,
) -> List[EDUGroupRecord]:
    """按原 SFT 协议固定为一 EDU 一组，不解析 defer 或应用 M_merge。"""

    if len(events) != len(stage_texts):
        raise RuntimeError("EDU event and streaming response segment counts differ.")
    groups = []
    for event_index, (event, stage_text) in enumerate(zip(events, stage_texts)):
        action, thought = _split_sft_stage_text(stage_text)
        thought_tokens = (
            len(tokenizer(thought, add_special_tokens=False)["input_ids"])
            if thought
            else 0
        )
        next_boundary_time = (
            events[event_index + 1].emitted_at
            if event_index + 1 < len(events)
            else event.emitted_at
        )
        groups.append(
            EDUGroupRecord(
                edu_ids=[event.edu_id],
                action=action,
                input_token_count=event.input_token_count,
                boundary_time=event.emitted_at,
                next_boundary_time=next_boundary_time,
                thought=thought,
                token_count=thought_tokens,
                control_entropy=0.0,
                relevance=_lexical_relevance(question, event.text),
                forced_by_merge_limit=False,
            )
        )
    return groups


def prepare_edu_group(
    item: Dict[str, Any],
    samples: List[RLSample],
    tokenizer: Any,
    expected_generations: int,
    accuracy_reward_mode: str,
    task_type: str,
    input_tokens_per_second: float,
    max_merge: int,
    enable_edu_controller: bool = False,
) -> EDUPreparedGroup:
    """复用基础协议校验，并补齐 EDU 三视图奖励记录。"""
    base_group = hrpo_runner.PreparedGroup
    original_prepare = getattr(prepare_edu_group, "_base_prepare", None)
    if original_prepare is None:
        raise RuntimeError("prepare_edu_group has not been bound to the base prepare_group.")
    prepared = original_prepare(
        item,
        samples,
        tokenizer,
        expected_generations,
        accuracy_reward_mode,
    )
    if not isinstance(prepared, base_group):
        raise TypeError("The base prepare_group returned an invalid type.")

    sample = samples[prepared.sample_index]
    scorer = build_task_scorer(task_type)
    events = _build_events(item, input_tokens_per_second)
    no_stream_text = str(item.get("no_stream_probe_text") or "")
    no_stream_score = scorer.score(no_stream_text, sample.answer)
    no_stream_finish_reason = str(item.get("no_stream_probe_finish_reason") or "")
    no_stream_truncated = bool(
        item.get("no_stream_probe_truncated", no_stream_finish_reason == "length")
    )
    no_stream_valid = bool(scorer.is_valid_probe(no_stream_text))
    no_stream_token_count = int(item.get("no_stream_probe_token_count", 0))
    records = []
    expected_segments = len(item.get("expected_boundaries") or [])
    for completion_index, completion in enumerate(item.get("completions") or []):
        lengths = [int(value) for value in completion["response_segment_token_lengths"]]
        token_ids = [int(value) for value in completion["token_ids"]]
        stage_texts = []
        offset = 0
        for length in lengths[:expected_segments]:
            stage_texts.append(
                tokenizer.decode(
                    token_ids[offset : offset + length],
                    skip_special_tokens=False,
                )
            )
            offset += length
        if enable_edu_controller:
            groups = _build_dynamic_groups(
                events,
                stage_texts,
                [
                    float(value)
                    for value in completion.get(
                        "stage_control_entropies",
                        [0.0] * expected_segments,
                    )[:expected_segments]
                ],
                tokenizer,
                sample.question,
                max_merge,
            )
        else:
            groups = _build_fixed_groups(
                events,
                stage_texts,
                tokenizer,
                sample.question,
            )
        stream_probe_text = str(completion.get("stream_probe_text") or "")
        stream_probe_score = scorer.score(stream_probe_text, sample.answer)
        stream_probe_finish_reason = str(
            completion.get("stream_probe_finish_reason") or ""
        )
        stream_probe_truncated = bool(
            completion.get(
                "stream_probe_truncated",
                stream_probe_finish_reason == "length",
            )
        )
        stream_probe_valid = bool(scorer.is_valid_probe(stream_probe_text))
        stream_probe_token_count = int(
            completion.get("stream_probe_token_count", 0)
        )
        streaming_segments, deep_text, full_text = parse_hrpo_response(
            prepared.response_texts[completion_index],
            expected_segments,
        )
        del streaming_segments
        final_score = scorer.score(full_text, sample.answer)
        record = EDURolloutRecord(
            prompt_id="sample-%d" % prepared.sample_index,
            full_input="".join(str(value) for value in item["source_segments"]),
            answer=sample.answer,
            edu_events=list(events),
            stream_groups=groups,
            stream_probe=ProbeRecord(
                answer=stream_probe_text,
                exact=stream_probe_score.exact,
                part=stream_probe_score.part,
                valid=stream_probe_valid,
                finish_reason=stream_probe_finish_reason,
                token_count=stream_probe_token_count,
                truncated=stream_probe_truncated,
            ),
            no_stream_probe=ProbeRecord(
                answer=no_stream_text,
                exact=no_stream_score.exact,
                part=no_stream_score.part,
                valid=no_stream_valid,
                finish_reason=no_stream_finish_reason,
                token_count=no_stream_token_count,
                truncated=no_stream_truncated,
            ),
            deep=DeepRecord(
                text=deep_text,
                token_count=prepared.deep_lengths[completion_index],
            ),
            final_answer=full_text,
            scores=ScoreRecord(
                exact=final_score.exact,
                part=final_score.part,
                stream_format=prepared.streaming_format_rewards[completion_index],
                deep_format=prepared.deep_format_rewards[completion_index],
                stream_verifier=0.0,
            ),
        )
        record.validate()
        records.append(record)
    return EDUPreparedGroup(**vars(prepared), records=records)


def build_advantages_adapter(
    group: EDUPreparedGroup,
    config: Any,
    device: torch.device,
    trainer_core_type: Type[Any],
) -> EDUAdvantagesAdapter:
    """计算 EDU 奖励并适配既有更新器需要的兼容字段。"""
    reward_batch = reward_batch_from_records(
        group.records,
        device=device,
        expected_rollouts_per_prompt=config.rollouts_per_prompt,
    )
    core = trainer_core_type(config)
    advantages, _ = core.compute_advantages(reward_batch)
    components = advantages.components
    streaming_reward = (
        float(config.format_weight) * reward_batch.stream_format
        + float(config.beta_s) * components.stream_length_reward
        + float(config.gamma_attr_s) * components.stream_attribution
    )
    deep_reward = (
        float(config.format_weight) * reward_batch.deep_format
        + float(config.beta_d) * components.deep_length_reward
        + float(config.gamma_attr_d) * components.deep_attribution
    )
    global_reward = (
        float(config.exact_weight) * reward_batch.exact
        + float(config.gamma_part) * reward_batch.part
        + float(config.beta_eff) * components.efficiency_reward
    )
    compatible_components = SimpleNamespace(
        streaming_length_penalty=components.stream_length_reward,
        deep_length_penalty=components.deep_length_reward,
        efficiency=components.efficiency_reward,
    )
    return EDUAdvantagesAdapter(
        streaming=advantages.streaming,
        deep=advantages.deep,
        global_=advantages.global_,
        components=compatible_components,
        edu_advantages=advantages,
        advantage=advantages.streaming + advantages.deep + advantages.global_,
        reward_raw=streaming_reward + deep_reward + global_reward,
        streaming_reward=streaming_reward,
        deep_reward=deep_reward,
        global_reward=global_reward,
        reward_batch=reward_batch,
    )


def compose_reporting_rewards(adapter: EDUAdvantagesAdapter, config: Any) -> Dict[str, torch.Tensor]:
    """返回 EDU 原始总奖励与组内相对优势，供日志和 best 选择使用。"""
    del config
    return {
        "reward": adapter.reward_raw,
        "reward_raw": adapter.reward_raw,
        "advantage": adapter.advantage,
        "streaming_reward": adapter.streaming_reward,
        "deep_reward": adapter.deep_reward,
        "global_reward": adapter.global_reward,
    }


def _reduce_edu_diagnostics(
    adapters: Sequence[EDUAdvantagesAdapter],
    device: torch.device,
) -> Dict[str, float]:
    names = (
        "partial_score",
        "stream_probe_accuracy",
        "no_stream_accuracy",
        "stream_probe_valid_rate",
        "no_stream_probe_valid_rate",
        "stream_probe_truncation_rate",
        "no_stream_probe_truncation_rate",
        "control_entropy",
        "stream_harm_rate",
        "deep_regression_rate",
        "deep_correction_rate",
        "deep_regression_given_stream_correct",
        "deep_correction_given_stream_wrong",
        "stream_final_both_correct_rate",
        "stream_final_both_wrong_rate",
        "stream_causal_help",
        "stream_causal_harm",
        "stream_attribution",
        "deep_attribution",
        "correction_burden",
        "hidden_stream_tokens",
        "exposed_stream_tokens",
        "think_rate",
        "stream_length_reward_edu",
        "deep_length_reward_edu",
        "exposure_cost",
    )
    numerators = {name: 0.0 for name in names}
    denominators = {name: 0.0 for name in names}

    def add(name: str, numerator: torch.Tensor, denominator: torch.Tensor) -> None:
        numerators[name] += float(numerator.detach().double().cpu())
        denominators[name] += float(denominator.detach().double().cpu())

    for adapter in adapters:
        batch = adapter.reward_batch
        components = adapter.edu_advantages.components
        rollout_count = torch.tensor(
            float(batch.exact.numel()),
            dtype=torch.float32,
            device=device,
        )
        stream_valid = batch.stream_probe_valid > 0.5
        no_stream_valid = batch.no_stream_probe_valid > 0.5
        both_valid = stream_valid & no_stream_valid
        stream_correct = batch.stream_probe_exact > 0.5
        final_correct = batch.exact > 0.5
        deep_regression = stream_correct & ~final_correct
        deep_correction = ~stream_correct & final_correct

        add("partial_score", batch.part.sum(), rollout_count)
        add(
            "stream_probe_accuracy",
            (batch.stream_probe_exact * stream_valid.float()).sum(),
            stream_valid.float().sum(),
        )
        add(
            "no_stream_accuracy",
            (batch.no_stream_exact * no_stream_valid.float()).sum(),
            no_stream_valid.float().sum(),
        )
        add("stream_probe_valid_rate", stream_valid.float().sum(), rollout_count)
        add("no_stream_probe_valid_rate", no_stream_valid.float().sum(), rollout_count)
        add(
            "stream_probe_truncation_rate",
            batch.stream_probe_truncated.float().sum(),
            rollout_count,
        )
        add(
            "no_stream_probe_truncation_rate",
            batch.no_stream_probe_truncated.float().sum(),
            rollout_count,
        )
        add(
            "control_entropy",
            (batch.control_uncertainty * batch.group_mask).sum(),
            batch.group_mask.float().sum(),
        )
        add(
            "stream_harm_rate",
            ((components.stream_causal_harm > 1e-8) & both_valid).float().sum(),
            both_valid.float().sum(),
        )
        add(
            "deep_regression_rate",
            (deep_regression & stream_valid).float().sum(),
            stream_valid.float().sum(),
        )
        add(
            "deep_correction_rate",
            (deep_correction & stream_valid).float().sum(),
            stream_valid.float().sum(),
        )
        add(
            "deep_regression_given_stream_correct",
            (deep_regression & stream_valid).float().sum(),
            (stream_valid & stream_correct).float().sum(),
        )
        add(
            "deep_correction_given_stream_wrong",
            (deep_correction & stream_valid).float().sum(),
            (stream_valid & ~stream_correct).float().sum(),
        )
        add(
            "stream_final_both_correct_rate",
            (stream_correct & final_correct & stream_valid).float().sum(),
            stream_valid.float().sum(),
        )
        add(
            "stream_final_both_wrong_rate",
            (~stream_correct & ~final_correct & stream_valid).float().sum(),
            stream_valid.float().sum(),
        )
        add(
            "stream_causal_help",
            (components.stream_causal_help * both_valid.float()).sum(),
            both_valid.float().sum(),
        )
        add(
            "stream_causal_harm",
            (components.stream_causal_harm * both_valid.float()).sum(),
            both_valid.float().sum(),
        )
        for name, value in (
            ("stream_attribution", components.stream_attribution),
            ("deep_attribution", components.deep_attribution),
            ("correction_burden", components.correction_burden),
        ):
            add(
                name,
                (value * stream_valid.float()).sum(),
                stream_valid.float().sum(),
            )
        for name, value in (
            ("hidden_stream_tokens", components.latency.hidden_token_count),
            ("exposed_stream_tokens", components.latency.exposed_token_count),
            ("think_rate", components.latency.think_rate),
            ("stream_length_reward_edu", components.stream_length_reward),
            ("deep_length_reward_edu", components.deep_length_reward),
            ("exposure_cost", components.latency.exposure_cost),
        ):
            add(name, value.sum(), rollout_count)

    packed = []
    for name in names:
        packed.extend((numerators[name], denominators[name]))
    values = torch.tensor(packed, dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    reduced = {}
    for index, name in enumerate(names):
        numerator = float(values[index * 2].cpu())
        denominator = float(values[index * 2 + 1].cpu())
        reduced[name] = numerator / max(denominator, 1.0)
    return reduced


def run_with_edu_advantages(
    train_function: Callable[..., Dict[str, float]],
    trainer_core_type: Type[Any],
    actor: Any,
    tokenizer: Any,
    optimizer: Any,
    groups: List[EDUPreparedGroup],
    args: Any,
    device: torch.device,
    config: Any,
    advantage_module: Any,
) -> Dict[str, float]:
    """临时注入 EDU 优势，调用成熟的 HRPO actor 更新实现。"""
    adapters = [
        build_advantages_adapter(group, config, device, trainer_core_type)
        for group in groups
    ]
    queue = list(adapters)

    def next_advantages(**kwargs):
        del kwargs
        if not queue:
            raise RuntimeError("The EDU advantage queue is exhausted.")
        return queue.pop(0)

    original_advantage_function = advantage_module.compute_hrpo_advantages
    original_reporting_function = hrpo_runner.compose_reporting_rewards
    advantage_module.compute_hrpo_advantages = next_advantages
    hrpo_runner.compose_reporting_rewards = compose_reporting_rewards
    try:
        metrics = train_function(
            actor,
            tokenizer,
            optimizer,
            groups,
            args,
            device,
            config,
        )
    finally:
        advantage_module.compute_hrpo_advantages = original_advantage_function
        hrpo_runner.compose_reporting_rewards = original_reporting_function
    if queue:
        raise RuntimeError("Unconsumed EDU advantage batches remain.")
    metrics.update(_reduce_edu_diagnostics(adapters, device))
    return metrics
