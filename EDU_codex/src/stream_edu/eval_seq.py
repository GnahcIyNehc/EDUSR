from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    collate_sequences,
    encode_sequence_docs,
    is_decimal_point,
    is_likely_sentence_period,
    is_nonterminal_period,
    read_split,
)
from .metrics import prf
from .seq_model import StreamingTransformerEDU, lookahead_mask


def decode_doc(scores: np.ndarray, threshold: float, min_gap: int, force_final: bool) -> np.ndarray:
    pred = scores >= threshold
    if min_gap > 0:
        decoded = np.zeros_like(pred, dtype=bool)
        last_boundary = -10_000
        for i, is_boundary in enumerate(pred):
            if is_boundary and i - last_boundary > min_gap:
                decoded[i] = True
                last_boundary = i
        pred = decoded
    if force_final and len(pred):
        pred[-1] = True
    return pred


def first_sentence_complete_stats(docs, speech_tokens_per_sec: float) -> dict[str, float]:
    waits = [1000.0 * len(doc.tokens) / speech_tokens_per_sec for doc in docs]
    waits_a = np.asarray(waits, dtype=np.float64)
    return {
        "mean_ms": float(waits_a.mean()),
        "median_ms": float(np.median(waits_a)),
        "p95_ms": float(np.percentile(waits_a, 95)),
    }


def first_edu_wait_stats(
    doc_scores: list[np.ndarray],
    threshold: float,
    lookahead: int,
    speech_tokens_per_sec: float,
    processing_us_per_token: float,
    min_gap: int = 0,
    force_final: bool = False,
) -> dict[str, float]:
    waits = []
    for scores in doc_scores:
        pred = decode_doc(scores, threshold, min_gap, force_final)
        hits = np.flatnonzero(pred)
        first_boundary = int(hits[0]) if len(hits) else len(scores) - 1
        observed_token = min(first_boundary + lookahead, len(scores) - 1)
        # Token index is 0-based; receiving token 0 already takes one token time.
        speech_ms = 1000.0 * (observed_token + 1) / speech_tokens_per_sec
        waits.append(speech_ms + processing_us_per_token / 1000.0)
    waits_a = np.asarray(waits, dtype=np.float64)
    return {
        "mean_ms": float(waits_a.mean()),
        "median_ms": float(np.median(waits_a)),
        "p95_ms": float(np.percentile(waits_a, 95)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/gum_edu"))
    p.add_argument("--split", default="test", choices=["dev", "test", "test2"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--speech-tokens-per-sec", type=float, default=2.5)
    p.add_argument("--period-guard", action="store_true")
    return p.parse_args()


def torch_batch(batch: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.from_numpy(batch["tokens"]).to(device),
        "shapes": torch.from_numpy(batch["shapes"]).to(device),
        "cues": torch.from_numpy(batch["cues"]).to(device),
        "stream_feats": torch.from_numpy(batch["stream_feats"]).to(device),
        "chars": torch.from_numpy(batch["chars"]).to(device),
        "labels": torch.from_numpy(batch["labels"]).to(device),
        "mask": torch.from_numpy(batch["mask"]).to(device),
        "final": torch.from_numpy(batch["final"]).to(device),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    lookahead = int(ckpt["lookahead"])
    char_buckets = int(ckpt.get("char_buckets", ckpt.get("model_config", {}).get("char_buckets", 2048)))
    char_slots = int(ckpt.get("char_slots", 8))
    docs = read_split(args.split, args.data_root)
    data = encode_sequence_docs(docs, vocab, char_buckets, char_slots)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequences)
    model = StreamingTransformerEDU(len(vocab), **ckpt.get("model_config", {})).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scores = []
    labels = []
    finals = []
    doc_scores = []
    doc_labels = []
    doc_finals = []
    t0 = time.perf_counter()
    for raw in loader:
        b = torch_batch(raw, args.device)
        am = lookahead_mask(b["tokens"].shape[1], lookahead, b["tokens"].device)
        out = model(b["tokens"], b["shapes"], b["chars"], b["mask"], am, b["cues"], b["stream_feats"])
        logits = out[0] if isinstance(out, tuple) else out
        valid = b["mask"]
        scores.append(torch.sigmoid(logits[valid]).cpu().numpy())
        labels.append(b["labels"][valid].cpu().numpy())
        finals.append(b["final"][valid].cpu().numpy())
        probs = torch.sigmoid(logits).cpu().numpy()
        lengths = b["mask"].sum(dim=1).cpu().numpy().astype(int)
        for row, n in zip(probs, lengths):
            doc_scores.append(row[:n])
        label_rows = b["labels"].cpu().numpy()
        final_rows = b["final"].cpu().numpy()
        for row, final_row, n in zip(label_rows, final_rows, lengths):
            doc_labels.append(row[:n])
            doc_finals.append(final_row[:n])
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    scores_a = np.concatenate(scores)
    labels_a = np.concatenate(labels)
    final_a = np.concatenate(finals).astype(bool)
    threshold = float(ckpt["best_dev"]["threshold"])
    if args.period_guard:
        for doc, scores in zip(docs, doc_scores):
            for index in range(len(doc.tokens)):
                if is_decimal_point(doc.tokens, index) or is_nonterminal_period(doc.tokens, index):
                    scores[index] = 0.0
                elif is_likely_sentence_period(doc.tokens, index):
                    scores[index] = 1.0
        scores_a = np.concatenate(doc_scores)
    min_gap = int(ckpt.get("decode_min_gap", ckpt.get("train_args", {}).get("decode_min_gap", 0)))
    force_final = bool(ckpt.get("force_final_boundary", ckpt.get("train_args", {}).get("force_final_boundary", False)))
    if min_gap > 0 or force_final:
        pred = np.concatenate([decode_doc(s, threshold, min_gap, force_final) for s in doc_scores])
        labels_a = np.concatenate(doc_labels)
        final_a = np.concatenate(doc_finals).astype(bool)
    else:
        pred = scores_a >= threshold
    proc_us = elapsed * 1e6 / len(labels_a)
    speech_delay_ms = 1000.0 * lookahead / args.speech_tokens_per_sec
    first_wait = first_edu_wait_stats(
        doc_scores,
        threshold,
        lookahead,
        args.speech_tokens_per_sec,
        proc_us,
        min_gap,
        force_final,
    )
    first_sentence_complete = first_sentence_complete_stats(docs, args.speech_tokens_per_sec)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "lookahead_tokens": lookahead,
        "speech_tokens_per_sec": args.speech_tokens_per_sec,
        "speech_delay_ms": speech_delay_ms,
        "batch_processing_us_per_token": proc_us,
        "decision_latency_ms_estimate": speech_delay_ms + proc_us / 1000.0,
        "first_predicted_edu_wait_ms": first_wait,
        "first_sentence_complete_input_ms": first_sentence_complete,
        "decode_min_gap": min_gap,
        "force_final_boundary": force_final,
        "threshold": threshold,
        "period_guard": args.period_guard,
        "metrics_all": prf(pred, labels_a),
        "metrics_internal": prf(pred[~final_a], labels_a[~final_a]),
        "tokens": int(len(labels_a)),
        "gold_boundaries": int(labels_a.sum()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
