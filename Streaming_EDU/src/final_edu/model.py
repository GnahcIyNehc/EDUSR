from __future__ import annotations

import torch
from torch import nn


class StreamingTransformerEDU(nn.Module):
    """Frozen inference architecture used by the bundled checkpoint."""

    def __init__(
        self,
        vocab_size: int,
        char_buckets: int = 2048,
        token_dim: int = 160,
        shape_dim: int = 16,
        cue_dim: int = 0,
        char_dim: int = 48,
        stream_dim: int = 0,
        model_dim: int = 256,
        layers: int = 4,
        heads: int = 4,
        ff_dim: int = 768,
        dropout: float = 0.15,
        max_len: int = 256,
        shape_vocab_size: int = 9,
        use_shape: bool = True,
        use_cue: bool = False,
        use_char: bool = True,
        use_stream_feats: bool = False,
        use_causal_conv: bool = False,
        conv_kernel: int = 3,
        conv_dilation: int = 1,
        use_prefix_summary: bool = False,
        recent_summary_window: int = 0,
        use_ema_summary: bool = False,
        ema_decay: float = 0.85,
        aux_start: bool = False,
        aux_dist: bool = False,
        dist_classes: int = 9,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.shape_emb = (
            nn.Embedding(shape_vocab_size, shape_dim, padding_idx=0)
            if use_shape and shape_dim > 0
            else None
        )
        self.cue_emb = nn.Embedding(6, cue_dim, padding_idx=0) if use_cue and cue_dim > 0 else None
        self.char_emb = nn.Embedding(char_buckets + 1, char_dim, padding_idx=0) if use_char and char_dim > 0 else None
        self.stream_embs = (
            nn.ModuleList(
                [
                    nn.Embedding(32, stream_dim),
                    nn.Embedding(17, stream_dim),
                    nn.Embedding(9, stream_dim),
                ]
            )
            if use_stream_feats and stream_dim > 0
            else None
        )

        in_dim = token_dim
        in_dim += shape_dim if self.shape_emb is not None else 0
        in_dim += cue_dim if self.cue_emb is not None else 0
        in_dim += char_dim if self.char_emb is not None else 0
        in_dim += stream_dim if self.stream_embs is not None else 0

        self.proj = nn.Linear(in_dim, model_dim)
        self.pos_emb = nn.Embedding(max_len, model_dim)
        self.causal_conv = (
            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, conv_kernel, groups=model_dim),
                nn.GELU(),
                nn.Conv1d(model_dim, model_dim, 1),
                nn.Dropout(dropout),
            )
            if use_causal_conv
            else None
        )
        self.prefix_proj = nn.Linear(model_dim * 2, model_dim) if use_prefix_summary else None

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, 1))
        self.start_out = (
            nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, 1))
            if aux_start
            else None
        )
        self.dist_out = (
            nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, dist_classes))
            if aux_dist
            else None
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        shape_ids: torch.Tensor,
        char_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        cue_ids: torch.Tensor | None = None,
        stream_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        parts = [self.token_emb(token_ids)]
        if self.shape_emb is not None:
            parts.append(self.shape_emb(shape_ids))
        if self.cue_emb is not None:
            parts.append(self.cue_emb(cue_ids if cue_ids is not None else torch.zeros_like(token_ids)))
        if self.char_emb is not None:
            parts.append(self.char_emb(char_ids).mean(dim=2))
        if self.stream_embs is not None:
            if stream_ids is None:
                stream_ids = torch.zeros((*token_ids.shape, 3), dtype=token_ids.dtype, device=token_ids.device)
            parts.append(torch.stack([emb(stream_ids[..., i]) for i, emb in enumerate(self.stream_embs)]).sum(0))

        x = self.proj(torch.cat(parts, dim=-1))
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        if self.causal_conv is not None:
            kernel = self.causal_conv[0].kernel_size[0]
            conv_input = torch.nn.functional.pad(x.transpose(1, 2), (kernel - 1, 0))
            x = x + self.causal_conv(conv_input).transpose(1, 2)

        if self.prefix_proj is not None:
            mask = padding_mask.unsqueeze(-1).to(x.dtype)
            prefix = (x * mask).cumsum(dim=1) / mask.cumsum(dim=1).clamp_min(1.0)
            x = x + self.prefix_proj(torch.cat([x, prefix], dim=-1))

        hidden = self.encoder(x, mask=attn_mask, src_key_padding_mask=~padding_mask)
        outputs = [self.out(hidden).squeeze(-1)]
        if self.start_out is not None:
            outputs.append(self.start_out(hidden).squeeze(-1))
        if self.dist_out is not None:
            outputs.append(self.dist_out(hidden))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def lookahead_mask(seq_len: int, lookahead: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(seq_len, device=device).unsqueeze(1)
    col = torch.arange(seq_len, device=device).unsqueeze(0)
    return col > row + lookahead
