"""按 query 块构造精确的稀疏 mask，限制 eager 中间张量峰值。"""

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def create_chunked_block_mask(
    mask_mod, B, H, Q_LEN, KV_LEN, device="cuda", BLOCK_SIZE=(128, 128)
):
    """逐 query 块归约，仅拼接稀疏元数据，不拼接完整 token 掩码。"""
    block_size = (BLOCK_SIZE, BLOCK_SIZE) if isinstance(BLOCK_SIZE, int) else BLOCK_SIZE
    query_block = block_size[0]
    if Q_LEN <= query_block:
        return create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=device, BLOCK_SIZE=block_size)
    chunks = []
    for offset in range(0, Q_LEN, query_block):
        # 偏移仅在构建局部块时生效；最终注意力使用原始全局坐标函数。
        def shifted_mask(batch, head, query, key):
            return mask_mod(batch, head, query + offset, key)

        chunks.append(create_block_mask(
            shifted_mask, B, H, min(query_block, Q_LEN - offset), KV_LEN,
            device=device, BLOCK_SIZE=block_size,
        ))
    return BlockMask.from_kv_blocks(
        torch.cat([chunk.kv_num_blocks for chunk in chunks], dim=-1),
        torch.cat([chunk.kv_indices for chunk in chunks], dim=-2),
        torch.cat([chunk.full_kv_num_blocks for chunk in chunks], dim=-1),
        torch.cat([chunk.full_kv_indices for chunk in chunks], dim=-2),
        BLOCK_SIZE=block_size, mask_mod=mask_mod, seq_lengths=(Q_LEN, KV_LEN),
    )
