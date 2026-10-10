"""流式 vLLM 请求元数据与分组位置编码工具。"""

import base64
import json
from typing import Dict, List, Sequence


REQUEST_PREFIX = "st-v1."


def encode_streaming_request_id(
    unique_id: str,
    source_segment_lengths: Sequence[int],
    target_segment_lengths: Sequence[int],
) -> str:
    """把流式分段信息编码进 vLLM request_id，供 GPU worker 构造输入。"""
    source_lengths = [int(value) for value in source_segment_lengths]
    target_lengths = [int(value) for value in target_segment_lengths]
    if not source_lengths or any(value <= 0 for value in source_lengths):
        raise ValueError("All source_segment_lengths must be positive.")
    if not target_lengths or any(value < 0 for value in target_lengths):
        raise ValueError("target_segment_lengths must be nonempty and nonnegative.")
    payload = json.dumps(
        {"s": source_lengths, "t": target_lengths},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return "%s%s.%s" % (REQUEST_PREFIX, encoded, str(unique_id))


def decode_streaming_request_id(request_id: str) -> Dict[str, List[int]]:
    """解析流式 request_id；非流式请求直接抛出 ValueError。"""
    value = str(request_id)
    if not value.startswith(REQUEST_PREFIX):
        raise ValueError("Not a streaming request_id.")
    encoded = value[len(REQUEST_PREFIX) :].split(".", 1)[0]
    encoded += "=" * ((4 - len(encoded) % 4) % 4)
    payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("ascii"))
    source_lengths = [int(item) for item in payload["s"]]
    target_lengths = [int(item) for item in payload["t"]]
    if not source_lengths or any(item <= 0 for item in source_lengths):
        raise ValueError("Invalid source segment lengths in the streaming request.")
    if not target_lengths or any(item < 0 for item in target_lengths):
        raise ValueError("Invalid target segment lengths in the streaming request.")
    return {"source_segment_lengths": source_lengths, "target_segment_lengths": target_lengths}


def build_group_position_ids(physical_positions: Sequence[int], source_total: int) -> List[int]:
    """source 与 target 使用两个分别从零开始的位置计数器。"""
    source_total = int(source_total)
    if source_total <= 0:
        raise ValueError("source_total must be positive.")
    return [
        int(position) if int(position) < source_total else int(position) - source_total
        for position in physical_positions
    ]


def build_target_source_visibility(
    source_segment_lengths: Sequence[int],
    target_segment_lengths: Sequence[int],
    maximum_position: int,
) -> List[int]:
    """返回每个物理 query 位置可见的 source token 前缀长度。"""
    source_lengths = [int(value) for value in source_segment_lengths]
    target_lengths = [int(value) for value in target_segment_lengths]
    source_total = sum(source_lengths)
    maximum_position = int(maximum_position)
    if maximum_position <= 0:
        raise ValueError("maximum_position must be positive.")

    visibility = [source_total] * maximum_position
    target_visibility: List[int] = []
    for segment_index, segment_length in enumerate(target_lengths):
        visible_segments = min(segment_index + 1, len(source_lengths))
        visible_source = sum(source_lengths[:visible_segments])
        target_visibility.extend([visible_source] * max(segment_length, 0))

    # source 始终完整预填充；当前待生成段只解锁下一个已经到达的 EDU。
    current_visible_segments = min(len(target_lengths) + 1, len(source_lengths))
    current_visible_source = sum(source_lengths[:current_visible_segments])

    # 自回归 logits 位于待预测 token 的前一位置，所以把 source 可见性向前平移一位。
    for target_offset in range(max(0, maximum_position - source_total)):
        physical_position = source_total + target_offset
        next_target_offset = target_offset + 1
        if next_target_offset < len(target_visibility):
            visibility[physical_position] = target_visibility[next_target_offset]
        else:
            # 历史 target 结束后只允许读取当前轮已经到达的 source。
            visibility[physical_position] = current_visible_source
    return visibility
