"""缓存流式 vLLM 请求对应的 source 可见性向量。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ST_RL.HRPO.streaming_vllm_protocol import (
    build_target_source_visibility,
    decode_streaming_request_id,
)


@dataclass(frozen=True)
class CachedStreamingVisibility:
    """一条流式请求可复用的紧凑掩码元数据。"""

    source_total: int
    visibility: np.ndarray


class StreamingVisibilityCache:
    """按 request_id 缓存 visibility，并通过 LRU 限制常驻内存。"""

    def __init__(self, max_entries: int) -> None:
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive.")
        self.max_entries = int(max_entries)
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[
            Tuple[str, int], CachedStreamingVisibility
        ] = OrderedDict()

    def get(self, request_id: str, maximum_position: int) -> CachedStreamingVisibility:
        """返回请求对应的 visibility；首次访问时完成解码和构造。"""
        maximum_position = int(maximum_position)
        if maximum_position <= 0:
            raise ValueError("maximum_position must be positive.")
        key = (str(request_id), maximum_position)
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            self.hits += 1
            return cached

        metadata = decode_streaming_request_id(str(request_id))
        source_lengths = metadata["source_segment_lengths"]
        row = np.asarray(
            build_target_source_visibility(
                source_lengths,
                metadata["target_segment_lengths"],
                maximum_position,
            ),
            dtype=np.int32,
        )
        row.setflags(write=False)
        cached = CachedStreamingVisibility(
            source_total=sum(source_lengths),
            visibility=row,
        )
        self._entries[key] = cached
        self.misses += 1
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return cached

    def __len__(self) -> int:
        return len(self._entries)
