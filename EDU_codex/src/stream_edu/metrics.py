from __future__ import annotations

import numpy as np


def prf(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gold = gold.astype(bool)
    tp = int(np.logical_and(pred, gold).sum())
    fp = int(np.logical_and(pred, ~gold).sum())
    fn = int(np.logical_and(~pred, gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def tune_threshold(scores: np.ndarray, gold: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
    if mask is None:
        mask = np.ones_like(gold, dtype=bool)
    best_t = 0.5
    best = {"f1": -1.0}
    for t in np.linspace(0.05, 0.95, 91):
        m = prf(scores[mask] >= t, gold[mask])
        if m["f1"] > best["f1"]:
            best_t = float(t)
            best = m
    return best_t, best

