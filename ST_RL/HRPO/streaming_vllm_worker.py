"""为 vLLM 注入 StreamingThinker 分组位置和严格流式注意力掩码。"""

import logging
import os
from typing import Any

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask
from ST_RL.HRPO.chunked_block_mask import create_chunked_block_mask

from ST_RL.HRPO.streaming_vllm_protocol import (
    build_group_position_ids,
)
from ST_RL.HRPO.streaming_visibility_cache import StreamingVisibilityCache


def _configure_dynamo_limits() -> None:
    """允许流式批处理中不同活跃序列数各自缓存 FlexAttention 图。"""
    recompile_limit = max(int(os.environ.get("ST_RL_DYNAMO_RECOMPILE_LIMIT", "128")), 8)
    accumulated_limit = max(
        int(os.environ.get("ST_RL_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT", "1024")),
        recompile_limit,
    )
    torch._dynamo.config.recompile_limit = recompile_limit
    torch._dynamo.config.cache_size_limit = recompile_limit
    torch._dynamo.config.accumulated_recompile_limit = accumulated_limit
    torch._dynamo.config.accumulated_cache_size_limit = accumulated_limit


def _configure_inductor_for_streaming() -> None:
    """动态批形状使用编译内核，但跳过不稳定的 CUDAGraph 重录制。"""
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None


_configure_dynamo_limits()
_configure_inductor_for_streaming()


def _load_vllm_types():
    from vllm.v1.attention.backends.flex_attention import (
        FlexAttentionMetadata,
        create_block_mask_compiled,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    # 动态图跳过是预期路径，只屏蔽其逐图 perf hint，真实异常仍会正常记录。
    logging.getLogger("torch._inductor.cudagraph_utils.__perf_hints").setLevel(logging.ERROR)
    return create_block_mask_compiled, FlexAttentionMetadata, GPUModelRunner, Worker


create_block_mask_compiled, FlexAttentionMetadata, GPUModelRunner, Worker = _load_vllm_types()
_BLOCK_MASK_MODE_LOGGED = False


def _build_placeholder_block_mask(*args, **kwargs):
    """为即将被流式 mask 覆盖的 vLLM 默认 mask 创建最小占位对象。"""
    device = kwargs.get("device")
    block_size = kwargs.get("BLOCK_SIZE", (128, 128))

    def causal_single_token(batch, head, query_index, kv_index):
        return query_index >= kv_index

    return create_block_mask(
        causal_single_token,
        None,
        None,
        1,
        1,
        device=device,
        BLOCK_SIZE=block_size,
    )


def use_compiled_block_mask() -> bool:
    """按环境变量选择 block mask 构建器，默认保留既有编译路径。"""
    value = os.environ.get("ST_RL_COMPILE_BLOCK_MASK", "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("ST_RL_COMPILE_BLOCK_MASK must be 0/1 or boolean text.")


class StreamingGPUModelRunner(GPUModelRunner):
    """保留 vLLM 的物理 KV 页表，只替换传给 Qwen3 的逻辑位置和可见性。"""

    def _get_streaming_visibility_cache(
        self,
        request_capacity: int,
    ) -> StreamingVisibilityCache:
        """惰性创建缓存，避免依赖 vLLM 内部初始化顺序。"""
        max_entries = max(int(request_capacity) * 4, 64)
        cache = getattr(self, "_streaming_visibility_cache", None)
        if cache is None or cache.max_entries != max_entries:
            cache = StreamingVisibilityCache(max_entries=max_entries)
            self._streaming_visibility_cache = cache
        return cache

    def _get_streaming_cpu_buffers(
        self,
        request_capacity: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """复用固定容量 CPU 缓冲，减少每个 decode 轮次的分配。"""
        expected_shape = (int(request_capacity), int(self.max_model_len))
        source_totals = getattr(self, "_streaming_source_totals_cpu", None)
        visibility = getattr(self, "_streaming_visibility_cpu", None)
        if source_totals is None or source_totals.shape != (request_capacity,):
            source_totals = np.zeros(request_capacity, dtype=np.int64)
            self._streaming_source_totals_cpu = source_totals
        if visibility is None or visibility.shape != expected_shape:
            visibility = np.zeros(expected_shape, dtype=np.int32)
            self._streaming_visibility_cpu = visibility
        return source_totals, visibility

    def _prepare_inputs(self, scheduler_output):
        compile_mask = use_compiled_block_mask()
        if compile_mask:
            prepared = super()._prepare_inputs(scheduler_output)
        else:
            # 真实流式请求会在本方法后半段安装最终 mask；先用最小占位对象跳过
            # vLLM 默认全尺寸 mask，避免重复构造和 compiled Triton 非法访存。
            import vllm.v1.attention.backends.flex_attention as flex_attention_module

            pending_req_ids = list(self.input_batch.req_ids)
            is_streaming_batch = bool(pending_req_ids) and all(
                isinstance(request_id, str) and request_id.startswith("st-v1.")
                for request_id in pending_req_ids
            )
            original_builder = flex_attention_module.create_block_mask_compiled
            flex_attention_module.create_block_mask_compiled = (
                _build_placeholder_block_mask
                if is_streaming_batch
                else create_block_mask
            )
            try:
                prepared = super()._prepare_inputs(scheduler_output)
            finally:
                flex_attention_module.create_block_mask_compiled = original_builder
        total_tokens = int(scheduler_output.total_num_scheduled_tokens)
        req_ids = list(self.input_batch.req_ids)
        scheduled = np.array(
            [scheduler_output.num_scheduled_tokens[request_id] for request_id in req_ids],
            dtype=np.int32,
        )
        req_indices = np.repeat(np.arange(len(req_ids), dtype=np.int32), scheduled)

        request_capacity = int(self.max_num_reqs)
        if len(req_ids) > request_capacity:
            raise RuntimeError(
                "Active streaming requests %d exceed fixed capacity %d."
                % (len(req_ids), request_capacity)
            )
        visibility_cache = self._get_streaming_visibility_cache(request_capacity)
        try:
            visibility_rows = [
                visibility_cache.get(request_id, self.max_model_len)
                for request_id in req_ids
            ]
        except (KeyError, TypeError, ValueError):
            # vLLM 初始化和 profile 使用普通虚拟请求，维持默认因果注意力即可。
            return prepared

        source_totals = np.array(
            [row.source_total for row in visibility_rows],
            dtype=np.int64,
        )
        physical_positions = self.positions.np[:total_tokens].astype(np.int64, copy=True)
        logical_positions = np.empty_like(physical_positions)
        for request_index, source_total in enumerate(source_totals):
            selected = req_indices == request_index
            logical_positions[selected] = np.asarray(
                build_group_position_ids(physical_positions[selected], int(source_total)),
                dtype=np.int64,
            )
        self.positions.np[:total_tokens] = logical_positions
        self.positions.copy_to_gpu(total_tokens)

        # 固定请求维度，避免活跃请求减少时 FlexAttention 为不同闭包形状反复编译内核。
        source_totals_cpu, visibility_cpu = self._get_streaming_cpu_buffers(
            request_capacity
        )
        source_totals_cpu[: len(source_totals)] = source_totals
        source_totals_cpu[len(source_totals) :] = 0
        source_totals_gpu = torch.as_tensor(
            source_totals_cpu,
            dtype=torch.long,
            device=self.device,
        )
        for request_index, row in enumerate(visibility_rows):
            visibility_cpu[request_index] = row.visibility
        visibility_cpu[len(visibility_rows) :] = 0
        visibility_gpu = torch.as_tensor(visibility_cpu, dtype=torch.int32, device=self.device)

        attention_metadata = prepared[0]
        installed = set()
        for layer_metadata in attention_metadata.values():
            if not isinstance(layer_metadata, FlexAttentionMetadata):
                raise RuntimeError(
                    "Streaming vLLM rollouts require VLLM_ATTENTION_BACKEND=FLEX_ATTENTION."
                )
            if id(layer_metadata) in installed:
                continue
            installed.add(id(layer_metadata))
            self._install_streaming_mask(
                layer_metadata,
                source_totals_gpu,
                visibility_gpu,
            )
        return prepared

    @staticmethod
    def _install_streaming_mask(
        metadata: Any,
        source_totals: torch.Tensor,
        visibility: torch.Tensor,
    ) -> None:
        """在 paged KV 的物理索引上构造论文所述 block-diagonal 可见性。"""

        # 在 FlexAttention 闭包外修复请求索引，避免把动态页表展开成复杂 Triton 图。
        request_limit = int(metadata.num_reqs)
        safe_doc_ids = metadata.doc_ids.clamp(min=0, max=request_limit - 1)

        def streaming_mask_mod(batch, head, query_index, physical_kv_index):
            # FlexAttention 按块扫描时包含 padding；任何页表访问前都必须保护索引。
            query_is_valid = (query_index >= 0) & (query_index < metadata.num_actual_tokens)
            kv_is_valid = (physical_kv_index >= 0) & (
                physical_kv_index < metadata.total_cache_tokens
            )
            safe_query_index = query_index.clamp(min=0, max=metadata.num_actual_tokens - 1)
            safe_kv_index = physical_kv_index.clamp(min=0, max=metadata.total_cache_tokens - 1)
            is_valid, logical_query, logical_kv = metadata._convert_physical_to_logical(
                safe_doc_ids,
                safe_query_index,
                safe_kv_index,
            )
            raw_request_index = metadata.doc_ids[safe_query_index].long()
            request_is_valid = (raw_request_index >= 0) & (raw_request_index < request_limit)
            request_index = safe_doc_ids[safe_query_index].long()
            source_total = source_totals[request_index]
            safe_query = logical_query.clamp(min=0, max=visibility.shape[1] - 1).long()
            visible_source = visibility[request_index, safe_query]
            causal = logical_query >= logical_kv
            target_query = logical_query >= source_total
            target_visibility = (logical_kv >= source_total) | (logical_kv < visible_source)
            allowed = (~target_query) | target_visibility
            return query_is_valid & kv_is_valid & is_valid & request_is_valid & causal & allowed

        metadata.mask_mod = streaming_mask_mod
        # Ada/消费级 GPU 上编译 create_block_mask 可能非法访存，可按任务切换 eager 路径。
        compile_mask = use_compiled_block_mask()
        block_mask_builder = create_block_mask_compiled if compile_mask else create_chunked_block_mask
        global _BLOCK_MASK_MODE_LOGGED
        if not _BLOCK_MASK_MODE_LOGGED:
            logging.getLogger(__name__).warning(
                "ST_RL FlexAttention block mask builder=%s",
                "compiled" if compile_mask else "eager_chunked",
            )
            _BLOCK_MASK_MODE_LOGGED = True
        metadata.block_mask = block_mask_builder(
            streaming_mask_mod,
            None,
            None,
            metadata.num_actual_tokens,
            metadata.total_cache_tokens,
            device=metadata.block_table.device,
            BLOCK_SIZE=(metadata.q_block_size, metadata.kv_block_size),
        )


class StreamingWorker(Worker):
    """通过官方 worker_cls 扩展点替换 GPUModelRunner。"""

    def init_device(self):
        import vllm.v1.worker.gpu_worker as gpu_worker_module

        original_runner = gpu_worker_module.GPUModelRunner
        gpu_worker_module.GPUModelRunner = StreamingGPUModelRunner
        try:
            super().init_device()
        finally:
            gpu_worker_module.GPUModelRunner = original_runner
