"""AdaSR 流式输入、分段位置和策略统计的共享协议。"""

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from ST_RL.training_utils import RLSample, SFT_STREAMING_INSTRUCTION

if TYPE_CHECKING:
    from ST_RL.prompt_protocol import PromptProtocol

SFT_SOURCE_PREFIX = SFT_STREAMING_INSTRUCTION + "<SEP>"
# 从本地生成器完成首个 <SEP> 后的状态开始 rollout，保持 target 分段与 SFT 完全对齐。
SFT_TARGET_PREFIX = "<|im_start|>assistant\n<think>\n<SEP>"
SFT_END_INSTRUCTION = "<|im_end|>\n"
STREAM_BOUNDARIES = ("<EOT>", "<EOQ>", "<EOP>")


@dataclass(frozen=True)
class StreamingSource:
    """一条样本的源流，以及每个真实输入 EDU 对应的目标边界。"""

    segments: List[str]
    response_boundaries: List[str]
    question_segments: int


@dataclass(frozen=True)
class PackedStreamingBatch:
    """供流式 actor 前向使用的右侧补齐批次。"""

    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    position_ids: List[List[int]]
    lengths: List[Dict[str, Any]]
    response_starts: List[int]
    response_lengths: List[int]


def _split_with_markers(text: str, marker_pattern: str) -> List[str]:
    pattern = re.compile(r"(.*?(?:%s))" % marker_pattern, re.DOTALL)
    chunks = pattern.findall(text or "")
    consumed = sum(len(chunk) for chunk in chunks)
    tail = (text or "")[consumed:]
    if tail:
        chunks.append(tail)
    return [chunk for chunk in chunks if chunk]


def split_question_edus(question: str) -> List[str]:
    """保留问题边界；没有显式边界时把整个问题视为一个最终 EDU。"""
    chunks = _split_with_markers(str(question or ""), r"<EOE>|<EOS>|<EOQ>")
    if not chunks:
        chunks = [str(question or "")]
    if not chunks[-1].endswith("<EOQ>"):
        chunks[-1] += "<EOQ>"
    return chunks


def split_context_edus(sample: RLSample) -> List[str]:
    """优先使用数据集里的原始带边界 context，避免丢失 SFT 的到达顺序。"""
    instruct_context = sample.raw.get("Instruct_Context") or {}
    raw_context = instruct_context.get("context") if isinstance(instruct_context, dict) else None
    if raw_context:
        chunks = _split_with_markers(str(raw_context), r"<EOE>|<EOS>")
        if chunks:
            return chunks
    return ["%s<EOS>" % segment for segment in sample.context_segments]


def build_streaming_source(
    sample: RLSample,
    protocol: Optional["PromptProtocol"] = None,
) -> StreamingSource:
    """按句子模型 SFT 的格式构造前缀、问题 EDU 和上下文 EDU。"""
    question_edus = split_question_edus(sample.question)
    context_edus = split_context_edus(sample)
    source_prefix = protocol.source_prefix if protocol is not None else SFT_SOURCE_PREFIX
    end_instruction = protocol.end_instruction if protocol is not None else SFT_END_INSTRUCTION
    source_segments = [source_prefix] + question_edus + context_edus
    source_segments[-1] += end_instruction
    response_boundaries = [
        "<EOQ>" if index == len(question_edus) - 1 else "<EOT>"
        for index in range(len(question_edus))
    ]
    response_boundaries.extend("<EOT>" for _ in context_edus)
    return StreamingSource(
        segments=source_segments,
        response_boundaries=response_boundaries,
        question_segments=len(question_edus),
    )


def encode_segments(tokenizer: Any, segments: Sequence[str]) -> List[List[int]]:
    """按 SFT 的整段编码结果切回 EDU，避免 BPE 跨边界合并造成错位。"""
    texts = [str(segment) for segment in segments]
    if not texts:
        return []

    full_text = "".join(texts)
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        full_ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = encoded.get("offset_mapping")
    except (KeyError, TypeError, ValueError, NotImplementedError):
        offsets = None

    # 慢 tokenizer 不提供 offset 时保留旧路径；Qwen 的 fast tokenizer 会走上面的严格路径。
    if offsets is None or len(offsets) != len(full_ids):
        return [
            [int(token_id) for token_id in tokenizer(text, add_special_tokens=False)["input_ids"]]
            for text in texts
        ]

    boundaries = [0]
    for text in texts:
        boundaries.append(boundaries[-1] + len(text))
    split_ids: List[List[int]] = [[] for _ in texts]
    for token_id, offset in zip(full_ids, offsets):
        start, end = int(offset[0]), int(offset[1])
        # 零长度 offset 只可能来自 tokenizer 的特殊行为，按当前位置归属即可。
        position = bisect_right(boundaries, start) - 1
        position = min(max(position, 0), len(texts) - 1)
        split_ids[position].append(token_id)

    if any(not token_ids for token_ids in split_ids):
        raise ValueError("Whole-text tokenization fully merged an EDU into a neighbor; no valid streaming boundary can be constructed.")
    return split_ids


def flatten_segments(segments: Sequence[Sequence[int]]) -> List[int]:
    return [int(token_id) for segment in segments for token_id in segment]


def pack_streaming_batch(
    source_segment_ids: Sequence[Sequence[int]],
    target_prefix_ids: Sequence[int],
    response_ids: Sequence[Sequence[int]],
    response_segment_lengths: Sequence[Sequence[int]],
    pad_token_id: int,
    target_prefix_segment_lengths: Optional[Sequence[int]] = None,
) -> PackedStreamingBatch:
    """把完整源流与多条目标轨迹打包，并让目标位置从零重新编号。"""
    if not response_ids:
        raise ValueError("response_ids must not be empty.")
    if len(response_ids) != len(response_segment_lengths):
        raise ValueError("response_ids and response_segment_lengths have different counts.")
    source_segments = [[int(token_id) for token_id in segment] for segment in source_segment_ids]
    if not source_segments or any(not segment for segment in source_segments):
        raise ValueError("Each source segment must contain at least one token.")
    source_ids = flatten_segments(source_segments)
    source_lengths = [len(segment) for segment in source_segments]
    prefix_ids = [int(token_id) for token_id in target_prefix_ids]
    if not prefix_ids:
        raise ValueError("target_prefix_ids must not be empty.")

    prefix_segment_lengths = [
        int(length)
        for length in (target_prefix_segment_lengths or [len(prefix_ids)])
    ]
    if any(length <= 0 for length in prefix_segment_lengths):
        raise ValueError("Every target prefix segment length must be positive.")
    if sum(prefix_segment_lengths) != len(prefix_ids):
        raise ValueError("Target prefix segment lengths must sum to the prefix token count.")

    rows: List[List[int]] = []
    target_lengths_rows: List[List[int]] = []
    response_lengths: List[int] = []
    for row_ids, row_segment_lengths in zip(response_ids, response_segment_lengths):
        normalized_ids = [int(token_id) for token_id in row_ids]
        normalized_lengths = [int(length) for length in row_segment_lengths]
        if any(length < 0 for length in normalized_lengths):
            raise ValueError("Target segment lengths must be nonnegative.")
        if sum(normalized_lengths) != len(normalized_ids):
            raise ValueError("Target segment lengths must sum to the response token count.")
        rows.append(source_ids + prefix_ids + normalized_ids)
        target_lengths_rows.append(prefix_segment_lengths + normalized_lengths)
        response_lengths.append(len(normalized_ids))

    max_length = max(len(row) for row in rows)
    padded_rows: List[List[int]] = []
    masks: List[List[int]] = []
    positions: List[List[int]] = []
    lengths: List[Dict[str, Any]] = []
    response_start = len(source_ids) + len(prefix_ids)
    for row, target_lengths in zip(rows, target_lengths_rows):
        padding = max_length - len(row)
        padded_rows.append(row + [int(pad_token_id)] * padding)
        masks.append([1] * len(row) + [0] * padding)
        positions.append(list(range(len(source_ids))) + list(range(max_length - len(source_ids))))
        lengths.append(
            {
                "source_token_len": len(source_ids),
                "source_seg_len": list(source_lengths),
                "target_token_len": sum(target_lengths),
                "target_seg_len": target_lengths,
                "input_batch_len": max_length,
                "prefix_source_segments": 1,
            }
        )
    return PackedStreamingBatch(
        input_ids=padded_rows,
        attention_mask=masks,
        position_ids=positions,
        lengths=lengths,
        response_starts=[response_start] * len(rows),
        response_lengths=response_lengths,
    )


def build_streaming_visibility(
    source_segment_lengths: Sequence[int],
    target_segment_lengths: Sequence[int],
    padded_length: int = 0,
) -> torch.Tensor:
    """生成与本地 Qwen 流式模型等价的布尔可见性矩阵，主要用于验收测试。"""
    source_lengths = [int(length) for length in source_segment_lengths]
    target_lengths = [int(length) for length in target_segment_lengths]
    total_length = sum(source_lengths) + sum(target_lengths)
    matrix_size = max(int(padded_length), total_length)
    visible = torch.tril(torch.ones((matrix_size, matrix_size), dtype=torch.bool))
    if total_length < matrix_size:
        visible[total_length:, :] = False

    source_total = sum(source_lengths)
    target_cursor = source_total
    for index, segment_length in enumerate(target_lengths):
        source_visible = sum(source_lengths[: index + 1])
        if source_visible < source_total:
            visible[target_cursor : target_cursor + segment_length, source_visible:source_total] = False
        target_cursor += segment_length

    # 自回归 logits 位于目标 token 的前一位，因此源可见性向前平移一位。
    target_total = sum(target_lengths)
    if target_total > 1:
        visible[source_total : source_total + target_total - 1, :source_total] = visible[
            source_total + 1 : source_total + target_total, :source_total
        ].clone()
    return visible


def compute_streaming_response_statistics(
    model: Any,
    tokenizer: Any,
    source_segment_ids: Sequence[Sequence[int]],
    target_prefix_ids: Sequence[int],
    response_ids: Sequence[Sequence[int]],
    response_segment_lengths: Sequence[Sequence[int]],
    device: torch.device,
    require_grad: bool,
    target_prefix_segment_lengths: Optional[Sequence[int]] = None,
    compute_entropy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用论文流式掩码计算回复 log-prob、有效掩码和策略熵。"""
    packed = pack_streaming_batch(
        source_segment_ids=source_segment_ids,
        target_prefix_ids=target_prefix_ids,
        response_ids=response_ids,
        response_segment_lengths=response_segment_lengths,
        pad_token_id=tokenizer.pad_token_id,
        target_prefix_segment_lengths=target_prefix_segment_lengths,
    )
    input_ids = torch.tensor(packed.input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(packed.attention_mask, dtype=torch.long, device=device)
    position_ids = torch.tensor(packed.position_ids, dtype=torch.long, device=device)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            training_mode="streaming",
            split_mode="sentence",
            is_training=True,
            _lengths=packed.lengths,
        )
        logits = outputs.logits
        max_response_length = max(packed.response_lengths + [1])
        log_probs_out = torch.zeros(
            (len(response_ids), max_response_length),
            dtype=torch.float32,
            device=logits.device,
        )
        response_mask = torch.zeros_like(log_probs_out, dtype=torch.bool)
        entropy_out = torch.zeros_like(log_probs_out)
        for row_index, (start, response_length) in enumerate(
            zip(packed.response_starts, packed.response_lengths)
        ):
            if response_length <= 0:
                continue
            token_logits = logits[row_index, start - 1 : start + response_length - 1, :].float()
            target_ids = torch.tensor(
                response_ids[row_index],
                dtype=torch.long,
                device=token_logits.device,
            )
            token_log_probs = torch.log_softmax(token_logits, dim=-1)
            log_probs_out[row_index, :response_length] = token_log_probs.gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            response_mask[row_index, :response_length] = True
            # 更新阶段不消费策略熵，跳过全词表临时张量；默认保持原统计行为。
            if compute_entropy:
                detached = token_log_probs.detach()
                entropy_out[row_index, :response_length] = -(detached.exp() * detached).sum(dim=-1)
    return log_probs_out, response_mask, entropy_out
