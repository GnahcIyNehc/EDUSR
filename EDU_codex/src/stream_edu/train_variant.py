from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    DECIMAL_POINT,
    NONTERMINAL_PERIOD,
    build_vocab,
    collate_sequences,
    encode_sequence_docs,
    make_biomedical_parenthetical_augmentation,
    make_decimal_augmentation,
    make_period_augmentation,
    read_split,
)
from .metrics import prf, tune_threshold
from .seq_model import StreamingTransformerEDU, lookahead_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/gum_edu"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments"))
    p.add_argument("--lookahead", type=int, default=1)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--decimal-augment-count", type=int, default=0)
    p.add_argument("--decimal-negative-weight", type=float, default=1.0)
    p.add_argument("--decimal-only-data", action="store_true")
    p.add_argument("--decimal-embedding-only", action="store_true")
    p.add_argument("--period-augment-count", type=int, default=0)
    p.add_argument("--period-negative-weight", type=float, default=1.0)
    p.add_argument("--period-only-data", action="store_true")
    p.add_argument("--period-embedding-only", action="store_true")
    p.add_argument("--biomedical-parenthetical-augment-count", type=int, default=0)
    p.add_argument("--biomedical-parenthetical-only-data", action="store_true")
    p.add_argument("--hard-negative-weight", type=float, default=1.0)
    p.add_argument("--boundary-head-only", action="store_true")
    p.add_argument("--init-distill-weight", type=float, default=0.0)
    p.add_argument("--token-dim", type=int, default=160)
    p.add_argument("--shape-dim", type=int, default=16)
    p.add_argument("--cue-dim", type=int, default=0)
    p.add_argument("--char-dim", type=int, default=48)
    p.add_argument("--stream-dim", type=int, default=0)
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--causal-conv", action="store_true")
    p.add_argument("--conv-kernel", type=int, default=3)
    p.add_argument("--conv-dilation", type=int, default=1)
    p.add_argument("--prefix-summary", action="store_true")
    p.add_argument("--recent-summary-window", type=int, default=0)
    p.add_argument("--ema-summary", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.85)
    p.add_argument("--char-buckets", type=int, default=2048)
    p.add_argument("--char-slots", type=int, default=8)
    p.add_argument("--no-shape", action="store_true")
    p.add_argument("--use-cue", action="store_true")
    p.add_argument("--use-stream-feats", action="store_true")
    p.add_argument("--no-char", action="store_true")
    p.add_argument("--aux-start-weight", type=float, default=0.0)
    p.add_argument("--aux-dist-weight", type=float, default=0.0)
    p.add_argument("--rule-distill-weight", type=float, default=0.0)
    p.add_argument("--token-dropout", type=float, default=0.0)
    p.add_argument("--soft-dice-weight", type=float, default=0.0)
    p.add_argument("--adjacent-boundary-penalty", type=float, default=0.0)
    p.add_argument("--boundary-margin-weight", type=float, default=0.0)
    p.add_argument("--boundary-margin", type=float, default=1.0)
    p.add_argument("--near-boundary-weight", type=float, default=1.0)
    p.add_argument("--near-boundary-window", type=int, default=1)
    p.add_argument("--count-loss-weight", type=float, default=0.0)
    p.add_argument("--decode-min-gap", type=int, default=0)
    p.add_argument("--force-final-boundary", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--final-loss-weight", type=float, default=1.0)
    p.add_argument("--train-lookahead-min", type=int, default=-1)
    p.add_argument("--train-lookahead-max", type=int, default=-1)
    p.add_argument("--lookahead-consistency-weight", type=float, default=0.0)
    p.add_argument("--consistency-lookahead", type=int, default=-1)
    p.add_argument("--future-distill-weight", type=float, default=0.0)
    p.add_argument("--teacher-lookahead", type=int, default=-1)
    p.add_argument("--stability-loss-weight", type=float, default=0.0)
    p.add_argument("--local-contrast-weight", type=float, default=0.0)
    p.add_argument("--local-contrast-window", type=int, default=2)
    p.add_argument("--confidence-loss-weight", type=float, default=0.0)
    p.add_argument("--brier-weight", type=float, default=0.0)
    p.add_argument("--ramp-loss-weight", type=float, default=0.0)
    p.add_argument("--ramp-tau", type=float, default=2.0)
    p.add_argument("--trigger-shift-weight", type=float, default=0.0)
    p.add_argument("--trigger-shift", type=int, default=1)
    p.add_argument("--post-boundary-suppress-weight", type=float, default=0.0)
    p.add_argument("--post-boundary-window", type=int, default=1)
    p.add_argument("--hazard-loss-weight", type=float, default=0.0)
    p.add_argument("--hazard-center", type=float, default=8.0)
    p.add_argument("--hazard-scale", type=float, default=2.5)
    p.add_argument("--boundary-window-loss-weight", type=float, default=0.0)
    p.add_argument("--boundary-window", type=int, default=1)
    p.add_argument("--anti-delay-weight", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--pct-start", type=float, default=0.12)
    return p.parse_args()


def torch_batch(batch: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.from_numpy(batch["tokens"]).to(device),
        "shapes": torch.from_numpy(batch["shapes"]).to(device),
        "cues": torch.from_numpy(batch["cues"]).to(device),
        "stream_feats": torch.from_numpy(batch["stream_feats"]).to(device),
        "chars": torch.from_numpy(batch["chars"]).to(device),
        "labels": torch.from_numpy(batch["labels"]).to(device),
        "starts": torch.from_numpy(batch["starts"]).to(device),
        "dist": torch.from_numpy(batch["dist"]).to(device),
        "age": torch.from_numpy(batch["age"]).to(device),
        "decimal": torch.from_numpy(batch["decimal"]).to(device),
        "nonterminal_period": torch.from_numpy(batch["nonterminal_period"]).to(device),
        "rules": torch.from_numpy(batch["rules"]).to(device),
        "hard_negative": torch.from_numpy(batch["hard_negative"]).to(device),
        "mask": torch.from_numpy(batch["mask"]).to(device),
        "final": torch.from_numpy(batch["final"]).to(device),
    }


def weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    label_smoothing: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = labels
    if label_smoothing > 0:
        targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    if focal_gamma > 0:
        prob = torch.sigmoid(logits)
        p_t = prob * labels + (1.0 - prob) * (1.0 - labels)
        loss = loss * (1.0 - p_t).pow(focal_gamma)
    if sample_weight is None:
        return loss[mask].mean()
    weighted = loss * sample_weight
    return weighted[mask].sum() / sample_weight[mask].sum().clamp_min(1.0)


@torch.no_grad()
def predict(model, loader, lookahead: int, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    labels = []
    finals = []
    for raw in loader:
        b = torch_batch(raw, device)
        am = lookahead_mask(b["tokens"].shape[1], lookahead, b["tokens"].device)
        out = model(b["tokens"], b["shapes"], b["chars"], b["mask"], am, b["cues"], b["stream_feats"])
        logits = out[0] if isinstance(out, tuple) else out
        valid = b["mask"]
        scores.append(torch.sigmoid(logits[valid]).cpu().numpy())
        labels.append(b["labels"][valid].cpu().numpy())
        finals.append(b["final"][valid].cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(finals).astype(bool)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_docs = read_split("train", args.data_root)
    dev_docs = read_split("dev", args.data_root)
    init_checkpoint = (
        torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.init_checkpoint is not None
        else None
    )
    vocab = (
        dict(init_checkpoint["vocab"])
        if init_checkpoint is not None
        else build_vocab(train_docs, min_freq=1, max_size=60000)
    )
    if args.decimal_augment_count > 0 and DECIMAL_POINT not in vocab:
        vocab[DECIMAL_POINT] = len(vocab)
    if args.period_augment_count > 0 and NONTERMINAL_PERIOD not in vocab:
        vocab[NONTERMINAL_PERIOD] = len(vocab)
    decimal_docs = make_decimal_augmentation(args.decimal_augment_count, args.seed + 10_000)
    period_docs = make_period_augmentation(args.period_augment_count, args.seed + 20_000)
    biomedical_docs = make_biomedical_parenthetical_augmentation(
        args.biomedical_parenthetical_augment_count,
        args.seed + 30_000,
    )
    if args.period_only_data:
        if not period_docs:
            raise ValueError("--period-only-data requires --period-augment-count > 0")
        train_docs = period_docs
    elif args.decimal_only_data:
        if not decimal_docs:
            raise ValueError("--decimal-only-data requires --decimal-augment-count > 0")
        train_docs = decimal_docs
    elif args.biomedical_parenthetical_only_data:
        if not biomedical_docs:
            raise ValueError(
                "--biomedical-parenthetical-only-data requires "
                "--biomedical-parenthetical-augment-count > 0"
            )
        train_docs = biomedical_docs
    else:
        train_docs.extend(decimal_docs)
        train_docs.extend(period_docs)
        train_docs.extend(biomedical_docs)
    train_data = encode_sequence_docs(train_docs, vocab, args.char_buckets, args.char_slots)
    dev_data = encode_sequence_docs(dev_docs, vocab, args.char_buckets, args.char_slots)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_sequences)

    model_config = {
        "char_buckets": args.char_buckets,
        "token_dim": args.token_dim,
        "shape_dim": args.shape_dim,
        "shape_vocab_size": 11,
        "cue_dim": args.cue_dim,
        "char_dim": args.char_dim,
        "stream_dim": args.stream_dim,
        "model_dim": args.model_dim,
        "layers": args.layers,
        "heads": args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "use_causal_conv": args.causal_conv,
        "conv_kernel": args.conv_kernel,
        "conv_dilation": args.conv_dilation,
        "use_prefix_summary": args.prefix_summary,
        "recent_summary_window": args.recent_summary_window,
        "use_ema_summary": args.ema_summary,
        "ema_decay": args.ema_decay,
        "use_shape": not args.no_shape,
        "use_cue": args.use_cue,
        "use_char": not args.no_char,
        "use_stream_feats": args.use_stream_feats,
        "aux_start": args.aux_start_weight > 0,
        "aux_dist": args.aux_dist_weight > 0,
    }
    model = StreamingTransformerEDU(len(vocab), **model_config).to(args.device)
    if init_checkpoint is not None:
        source = init_checkpoint["model_state"]
        target = model.state_dict()
        loaded = {}
        for key, value in source.items():
            if key not in target:
                continue
            if value.shape == target[key].shape:
                loaded[key] = value
                continue
            if key == "token_emb.weight" and value.ndim == 2 and value.shape[1] == target[key].shape[1]:
                expanded = target[key].clone()
                expanded[: value.shape[0]] = value
                source_token_id = init_checkpoint["vocab"].get(
                    DECIMAL_POINT,
                    init_checkpoint["vocab"].get("."),
                )
                if source_token_id is not None:
                    for row in range(value.shape[0], expanded.shape[0]):
                        expanded[row] = value[source_token_id]
                loaded[key] = expanded
                continue
            if key == "shape_emb.weight" and value.ndim == 2 and value.shape[1] == target[key].shape[1]:
                expanded = target[key].clone()
                expanded[: value.shape[0]] = value
                for row in range(value.shape[0], expanded.shape[0]):
                    expanded[row] = value[min(9, value.shape[0] - 1)]
                loaded[key] = expanded
        missing, unexpected = model.load_state_dict(loaded, strict=False)
        print(
            json.dumps(
                {
                    "initialized_from": str(args.init_checkpoint),
                    "loaded_tensors": len(loaded),
                    "missing_tensors": missing,
                    "unexpected_tensors": unexpected,
                },
                ensure_ascii=False,
            )
        )
    if args.decimal_embedding_only:
        if init_checkpoint is None:
            raise ValueError("--decimal-embedding-only requires --init-checkpoint")
        decimal_id = vocab[DECIMAL_POINT]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.token_emb.weight.requires_grad_(True)
        if model.shape_emb is None:
            raise ValueError("--decimal-embedding-only requires shape embeddings")
        model.shape_emb.weight.requires_grad_(True)

        token_mask = torch.zeros_like(model.token_emb.weight)
        token_mask[decimal_id] = 1.0
        shape_mask = torch.zeros_like(model.shape_emb.weight)
        shape_mask[9] = 1.0
        model.token_emb.weight.register_hook(lambda grad: grad * token_mask)
        model.shape_emb.weight.register_hook(lambda grad: grad * shape_mask)
    if args.period_embedding_only:
        if init_checkpoint is None:
            raise ValueError("--period-embedding-only requires --init-checkpoint")
        period_id = vocab[NONTERMINAL_PERIOD]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.token_emb.weight.requires_grad_(True)
        if model.shape_emb is None:
            raise ValueError("--period-embedding-only requires shape embeddings")
        model.shape_emb.weight.requires_grad_(True)
        token_mask = torch.zeros_like(model.token_emb.weight)
        token_mask[period_id] = 1.0
        shape_mask = torch.zeros_like(model.shape_emb.weight)
        shape_mask[10] = 1.0
        model.token_emb.weight.register_hook(lambda grad: grad * token_mask)
        model.shape_emb.weight.register_hook(lambda grad: grad * shape_mask)
    if args.boundary_head_only:
        if init_checkpoint is None:
            raise ValueError("--boundary-head-only requires --init-checkpoint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.out.parameters():
            parameter.requires_grad_(True)
    init_teacher = None
    if args.init_distill_weight > 0:
        if init_checkpoint is None:
            raise ValueError("--init-distill-weight requires --init-checkpoint")
        init_teacher = StreamingTransformerEDU(len(vocab), **model_config).to(args.device)
        init_teacher.load_state_dict(model.state_dict())
        init_teacher.eval()
        for parameter in init_teacher.parameters():
            parameter.requires_grad_(False)
    n_pos = sum(float(item["labels"].sum()) for item in train_data)
    n_tok = sum(float(len(item["labels"])) for item in train_data)
    end_pos_weight = torch.tensor([(n_tok - n_pos) / max(n_pos, 1.0)], device=args.device)
    n_start = sum(float(item["starts"].sum()) for item in train_data)
    start_pos_weight = torch.tensor([(n_tok - n_start) / max(n_start, 1.0)], device=args.device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=args.epochs * len(loader),
        pct_start=args.pct_start,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))

    best_state = None
    best_row = None
    best_internal = -1.0
    history = []
    t0 = time.perf_counter()
    train_lookahead_min = args.lookahead if args.train_lookahead_min < 0 else args.train_lookahead_min
    train_lookahead_max = args.lookahead if args.train_lookahead_max < 0 else args.train_lookahead_max
    if train_lookahead_min > train_lookahead_max:
        raise ValueError("--train-lookahead-min cannot be greater than --train-lookahead-max")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for raw in tqdm(loader, desc=f"{args.name} epoch {epoch}", leave=False):
            b = torch_batch(raw, args.device)
            train_lookahead = random.randint(train_lookahead_min, train_lookahead_max)
            am = lookahead_mask(b["tokens"].shape[1], train_lookahead, b["tokens"].device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                token_ids = b["tokens"]
                if args.token_dropout > 0:
                    drop = (torch.rand_like(token_ids, dtype=torch.float32) < args.token_dropout) & b["mask"] & (token_ids != 0)
                    token_ids = token_ids.masked_fill(drop, 1)
                out = model(token_ids, b["shapes"], b["chars"], b["mask"], am, b["cues"], b["stream_feats"])
                if isinstance(out, tuple):
                    end_logits = out[0]
                    start_logits = out[1] if args.aux_start_weight > 0 else None
                    dist_logits = out[-1] if args.aux_dist_weight > 0 else None
                else:
                    end_logits, start_logits, dist_logits = out, None, None
                end_sample_weight = torch.ones_like(b["labels"])
                if args.final_loss_weight != 1.0:
                    end_sample_weight = end_sample_weight.masked_fill(b["final"], args.final_loss_weight)
                if args.near_boundary_weight != 1.0:
                    near = (b["dist"] <= args.near_boundary_window) & b["mask"]
                    end_sample_weight = end_sample_weight.masked_fill(near, args.near_boundary_weight)
                if args.decimal_negative_weight != 1.0:
                    end_sample_weight = end_sample_weight.masked_fill(
                        b["decimal"],
                        args.decimal_negative_weight,
                    )
                if args.period_negative_weight != 1.0:
                    end_sample_weight = end_sample_weight.masked_fill(
                        b["nonterminal_period"],
                        args.period_negative_weight,
                    )
                if args.hard_negative_weight != 1.0:
                    end_sample_weight = end_sample_weight.masked_fill(
                        b["hard_negative"],
                        args.hard_negative_weight,
                    )
                loss = weighted_loss(
                    end_logits,
                    b["labels"],
                    b["mask"],
                    end_pos_weight,
                    args.focal_gamma,
                    args.label_smoothing,
                    end_sample_weight,
                )
                if start_logits is not None:
                    start_loss = weighted_loss(start_logits, b["starts"], b["mask"], start_pos_weight, 0.0, 0.0)
                    loss = loss + args.aux_start_weight * start_loss
                if dist_logits is not None:
                    dist_loss = torch.nn.functional.cross_entropy(
                        dist_logits.reshape(-1, dist_logits.shape[-1]),
                        b["dist"].reshape(-1),
                        reduction="none",
                    ).reshape_as(b["labels"])
                    loss = loss + args.aux_dist_weight * dist_loss[b["mask"]].mean()
                if args.rule_distill_weight > 0:
                    rule_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        end_logits,
                        b["rules"],
                        reduction="none",
                    )
                    loss = loss + args.rule_distill_weight * rule_loss[b["mask"]].mean()
                probs = torch.sigmoid(end_logits)
                if init_teacher is not None:
                    with torch.no_grad():
                        teacher_out = init_teacher(
                            b["tokens"],
                            b["shapes"],
                            b["chars"],
                            b["mask"],
                            am,
                            b["cues"],
                            b["stream_feats"],
                        )
                        teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
                        teacher_probs = torch.sigmoid(teacher_logits)
                    distill_mask = b["mask"] & ~b["hard_negative"]
                    if distill_mask.any():
                        init_distill = torch.nn.functional.binary_cross_entropy_with_logits(
                            end_logits,
                            teacher_probs,
                            reduction="none",
                        )
                        loss = loss + args.init_distill_weight * init_distill[distill_mask].mean()
                if args.soft_dice_weight > 0:
                    p = probs[b["mask"]]
                    y = b["labels"][b["mask"]]
                    dice = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)
                    loss = loss + args.soft_dice_weight * dice
                if args.adjacent_boundary_penalty > 0:
                    pair_mask = b["mask"][:, 1:] & b["mask"][:, :-1]
                    adjacent = probs[:, 1:] * probs[:, :-1]
                    loss = loss + args.adjacent_boundary_penalty * adjacent[pair_mask].mean()
                if args.boundary_margin_weight > 0:
                    gold = (b["labels"] > 0.5) & b["mask"]
                    margins = []
                    prev_valid = gold[:, 1:] & b["mask"][:, :-1]
                    if prev_valid.any():
                        margins.append(torch.relu(args.boundary_margin - end_logits[:, 1:] + end_logits[:, :-1])[prev_valid].mean())
                    next_valid = gold[:, :-1] & b["mask"][:, 1:]
                    if next_valid.any():
                        margins.append(torch.relu(args.boundary_margin - end_logits[:, :-1] + end_logits[:, 1:])[next_valid].mean())
                    if margins:
                        loss = loss + args.boundary_margin_weight * torch.stack(margins).mean()
                if args.count_loss_weight > 0:
                    lengths = b["mask"].sum(dim=1).clamp_min(1).to(probs.dtype)
                    pred_counts = (probs * b["mask"].to(probs.dtype)).sum(dim=1) / lengths
                    gold_counts = (b["labels"] * b["mask"].to(probs.dtype)).sum(dim=1) / lengths
                    loss = loss + args.count_loss_weight * torch.nn.functional.mse_loss(pred_counts, gold_counts)
                if args.stability_loss_weight > 0:
                    pair_mask = (
                        b["mask"][:, 1:]
                        & b["mask"][:, :-1]
                        & (b["labels"][:, 1:] < 0.5)
                        & (b["labels"][:, :-1] < 0.5)
                    )
                    if pair_mask.any():
                        smooth = torch.nn.functional.mse_loss(probs[:, 1:][pair_mask], probs[:, :-1][pair_mask])
                        loss = loss + args.stability_loss_weight * smooth
                if args.local_contrast_weight > 0:
                    gold = (b["labels"] > 0.5) & b["mask"]
                    contrast_terms = []
                    for shift in range(1, args.local_contrast_window + 1):
                        left_valid = gold[:, shift:] & b["mask"][:, :-shift] & (b["labels"][:, :-shift] < 0.5)
                        if left_valid.any():
                            contrast_terms.append(
                                torch.relu(
                                    args.boundary_margin
                                    - end_logits[:, shift:]
                                    + end_logits[:, :-shift]
                                )[left_valid].mean()
                            )
                        right_valid = gold[:, :-shift] & b["mask"][:, shift:] & (b["labels"][:, shift:] < 0.5)
                        if right_valid.any():
                            contrast_terms.append(
                                torch.relu(
                                    args.boundary_margin
                                    - end_logits[:, :-shift]
                                    + end_logits[:, shift:]
                                )[right_valid].mean()
                            )
                    if contrast_terms:
                        loss = loss + args.local_contrast_weight * torch.stack(contrast_terms).mean()
                if args.confidence_loss_weight > 0:
                    confidence = (probs * (1.0 - probs))[b["mask"]].mean()
                    loss = loss + args.confidence_loss_weight * confidence
                if args.brier_weight > 0:
                    brier = torch.nn.functional.mse_loss(probs[b["mask"]], b["labels"][b["mask"]])
                    loss = loss + args.brier_weight * brier
                if args.ramp_loss_weight > 0:
                    ramp_target = torch.exp(-b["dist"].to(probs.dtype) / max(args.ramp_tau, 1e-6))
                    ramp = torch.nn.functional.mse_loss(probs[b["mask"]], ramp_target[b["mask"]])
                    loss = loss + args.ramp_loss_weight * ramp
                if args.hazard_loss_weight > 0:
                    age = b["age"].to(probs.dtype)
                    hazard_target = torch.sigmoid((age - args.hazard_center) / max(args.hazard_scale, 1e-6))
                    hazard = torch.nn.functional.mse_loss(probs[b["mask"]], hazard_target[b["mask"]])
                    loss = loss + args.hazard_loss_weight * hazard
                if args.trigger_shift_weight > 0 and args.trigger_shift > 0:
                    shifted = torch.zeros_like(b["labels"])
                    shift = args.trigger_shift
                    shifted[:, :-shift] = torch.maximum(shifted[:, :-shift], b["labels"][:, shift:])
                    trigger_mask = b["mask"] & (b["labels"] < 0.5)
                    if trigger_mask.any():
                        trigger_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            end_logits,
                            shifted,
                            reduction="none",
                        )
                        loss = loss + args.trigger_shift_weight * trigger_loss[trigger_mask].mean()
                if args.post_boundary_suppress_weight > 0:
                    suppress = torch.zeros_like(b["labels"], dtype=torch.bool)
                    for shift in range(1, args.post_boundary_window + 1):
                        suppress[:, shift:] |= (b["labels"][:, :-shift] > 0.5) & b["mask"][:, shift:]
                    suppress &= b["mask"] & (b["labels"] < 0.5)
                    if suppress.any():
                        suppress_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            end_logits[suppress],
                            torch.zeros_like(end_logits[suppress]),
                        )
                        loss = loss + args.post_boundary_suppress_weight * suppress_loss
                if args.boundary_window_loss_weight > 0 and args.boundary_window > 0:
                    gold = (b["labels"] > 0.5) & b["mask"]
                    window_terms = []
                    for shift in range(0, args.boundary_window + 1):
                        valid = gold[:, shift:] if shift else gold
                        logits_slice = end_logits[:, :-shift] if shift else end_logits
                        if valid.any():
                            window_terms.append(torch.nn.functional.softplus(-logits_slice[valid]).mean())
                    if window_terms:
                        loss = loss + args.boundary_window_loss_weight * torch.stack(window_terms).mean()
                if args.anti_delay_weight > 0:
                    gold = (b["labels"] > 0.5) & b["mask"]
                    anti_terms = []
                    for shift in (1, 2):
                        late = gold[:, :-shift] & b["mask"][:, shift:]
                        if late.any():
                            anti_terms.append(torch.nn.functional.softplus(end_logits[:, shift:][late]).mean())
                    if anti_terms:
                        loss = loss + args.anti_delay_weight * torch.stack(anti_terms).mean()
                if args.future_distill_weight > 0:
                    teacher_lookahead = args.teacher_lookahead if args.teacher_lookahead >= 0 else max(train_lookahead, args.lookahead)
                    if teacher_lookahead != train_lookahead:
                        teacher_mask = lookahead_mask(b["tokens"].shape[1], teacher_lookahead, b["tokens"].device)
                        with torch.no_grad():
                            teacher_out = model(
                                token_ids,
                                b["shapes"],
                                b["chars"],
                                b["mask"],
                                teacher_mask,
                                b["cues"],
                                b["stream_feats"],
                            )
                            teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
                            teacher_probs = torch.sigmoid(teacher_logits).detach()
                        distill = torch.nn.functional.binary_cross_entropy_with_logits(
                            end_logits,
                            teacher_probs,
                            reduction="none",
                        )
                        loss = loss + args.future_distill_weight * distill[b["mask"]].mean()
                if args.lookahead_consistency_weight > 0:
                    consistency_lookahead = (
                        args.consistency_lookahead if args.consistency_lookahead >= 0 else max(0, train_lookahead - 1)
                    )
                    if consistency_lookahead != train_lookahead:
                        consistency_mask = lookahead_mask(b["tokens"].shape[1], consistency_lookahead, b["tokens"].device)
                        consistency_out = model(
                            token_ids,
                            b["shapes"],
                            b["chars"],
                            b["mask"],
                            consistency_mask,
                            b["cues"],
                            b["stream_feats"],
                        )
                        consistency_logits = consistency_out[0] if isinstance(consistency_out, tuple) else consistency_out
                        consistency_probs = torch.sigmoid(consistency_logits)
                        consistency = torch.nn.functional.mse_loss(
                            consistency_probs[b["mask"]],
                            probs.detach()[b["mask"]],
                        )
                        loss = loss + args.lookahead_consistency_weight * consistency
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = int(b["mask"].sum().item())
            total_loss += float(loss.item()) * n
            total_tokens += n

        scores, gold, final = predict(model, dev_loader, args.lookahead, args.device)
        threshold, dev_internal = tune_threshold(scores, gold, ~final)
        dev_all = prf(scores >= threshold, gold)
        row = {
            "epoch": epoch,
            "loss": total_loss / total_tokens,
            "threshold": threshold,
            "dev_internal": dev_internal,
            "dev_all": dev_all,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if (
            dev_internal["f1"] > best_internal
            or ((args.decimal_embedding_only or args.period_embedding_only) and epoch == args.epochs)
        ):
            best_internal = dev_internal["f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_row = row

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.pt"
    torch.save(
        {
            "model_state": best_state,
            "vocab": vocab,
            "lookahead": args.lookahead,
            "best_dev": best_row,
            "history": history,
            "train_seconds": time.perf_counter() - t0,
            "model": "StreamingTransformerEDU",
            "model_config": model_config,
            "char_buckets": args.char_buckets,
            "char_slots": args.char_slots,
            "train_args": vars(args),
            "decode_min_gap": args.decode_min_gap,
            "force_final_boundary": args.force_final_boundary,
        },
        out,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
