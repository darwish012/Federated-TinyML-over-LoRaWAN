"""Dataset loading and evaluation helpers shared by offline and serial tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

WINDOW_SIZE = 512
NUM_FEATURES = 12
BIN_DIR = Path(__file__).parent / "cwru_bin"
FEATURE_NAMES = (
    "rms", "peak_to_peak", "kurtosis", "skewness", "crest", "clearance",
    "shape", "impulse", "mean", "stddev", "maximum", "minimum",
)


def load_normalization(bin_dir: Path = BIN_DIR) -> tuple[np.ndarray, np.ndarray]:
    path = bin_dir / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare_cwru.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    means = np.asarray(payload["means"], dtype=np.float32)
    stds = np.asarray(payload["stds"], dtype=np.float32)
    if means.shape != (NUM_FEATURES,) or stds.shape != (NUM_FEATURES,):
        raise ValueError("normalization.json must contain 12 means and stds")
    return means, stds


def extract_raw_features(window: Sequence[float] | np.ndarray) -> np.ndarray:
    """Feature order and population moments used by the ESP32 preprocessor."""
    x = np.asarray(window, dtype=np.float32)
    if x.shape != (WINDOW_SIZE,):
        raise ValueError(f"expected a {WINDOW_SIZE}-sample block, got {x.shape}")
    mean = np.mean(x)
    maximum = np.max(x)
    minimum = np.min(x)
    centered = x - mean
    absolute = np.abs(x)
    rms = np.sqrt(np.mean(x * x))
    stddev = np.sqrt(np.mean(centered * centered))
    abs_mean = np.mean(absolute)
    max_abs = np.max(absolute)
    if stddev > 1e-6:
        skewness = np.mean(centered**3) / (stddev**3)
        kurtosis = np.mean(centered**4) / (stddev**4)
    else:
        skewness = kurtosis = 0.0
    squared_sqrt_mean = np.mean(np.sqrt(absolute)) ** 2
    return np.asarray(
        [
            rms,
            maximum - minimum,
            kurtosis,
            skewness,
            max_abs / rms if rms > 1e-6 else 0.0,
            max_abs / squared_sqrt_mean if squared_sqrt_mean > 1e-6 else 0.0,
            rms / abs_mean if abs_mean > 1e-6 else 0.0,
            max_abs / abs_mean if abs_mean > 1e-6 else 0.0,
            mean,
            stddev,
            maximum,
            minimum,
        ],
        dtype=np.float32,
    )


def load_blocks(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size == 0 or values.size % WINDOW_SIZE:
        raise ValueError(
            f"{path} has {values.size} floats; expected complete independent {WINDOW_SIZE}-float blocks"
        )
    return values.reshape(-1, WINDOW_SIZE)


def feature_matrix(path: Path, normalized: bool = True, bin_dir: Path = BIN_DIR) -> np.ndarray:
    raw = np.vstack([extract_raw_features(block) for block in load_blocks(path)])
    if not normalized:
        return raw
    means, stds = load_normalization(bin_dir)
    return ((raw - means) / stds).astype(np.float32)


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["load"] = int(row["load"])
        row["severity"] = float(row["severity"])
        row["start_sample"] = int(row["start_sample"])
        row["label"] = int(row["label"])
        row["binary_window_index"] = int(row["binary_window_index"])
    return rows


def load_domain(client_slug: str, bin_dir: Path = BIN_DIR) -> dict[str, object]:
    domain_dir = bin_dir / client_slug
    rows = load_manifest(domain_dir / "manifest.csv")
    parts: dict[str, object] = {"rows": rows, "client": client_slug}
    for split in ("train", "calibration", "val", "test"):
        parts[split] = feature_matrix(domain_dir / f"raw_{split}.bin", True, bin_dir)
        split_rows = [row for row in rows if row["split"] == split]
        if len(split_rows) != len(parts[split]):
            raise ValueError(
                f"{client_slug} {split}: manifest has {len(split_rows)} rows but binary has {len(parts[split])} blocks"
            )
        parts[f"{split}_rows"] = split_rows
    parts["labels"] = np.asarray([row["label"] for row in parts["test_rows"]], dtype=np.int8)
    parts["val_labels"] = np.asarray([row["label"] for row in parts["val_rows"]], dtype=np.int8)
    return parts


def threshold_from_calibration(scores: Sequence[float], sigma: float = 3.0) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 2:
        raise ValueError("at least two calibration scores are required")
    return float(values.mean() + sigma * values.std(ddof=1))


def classification_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if truth.size != values.size:
        raise ValueError(f"label/score mismatch: {truth.size} vs {values.size}")
    predicted = (values > threshold).astype(np.int8)
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))
    if positives and negatives:
        # Mann-Whitney form of ROC-AUC, with average ranks for tied scores.
        order = np.argsort(values, kind="mergesort")
        sorted_scores = values[order]
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        rank_sum = float(np.sum(ranks[truth == 1]))
        roc_auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    else:
        roc_auc = float("nan")

    if positives:
        descending = np.argsort(-values, kind="mergesort")
        ordered_truth = truth[descending]
        cumulative_tp = np.cumsum(ordered_truth == 1)
        cumulative_count = np.arange(1, truth.size + 1)
        precision_curve = cumulative_tp / cumulative_count
        # Average precision is the step-wise area under the PR curve.
        pr_auc = float(np.sum(precision_curve[ordered_truth == 1]) / positives)
    else:
        pr_auc = float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
    }


def evaluate_score_function(
    calibration_features: np.ndarray,
    test_features: np.ndarray,
    labels: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float | int]:
    calibration_scores = np.asarray(scorer(calibration_features), dtype=np.float64)
    test_scores = np.asarray(scorer(test_features), dtype=np.float64)
    threshold = threshold_from_calibration(calibration_scores)
    result = classification_metrics(labels, test_scores, threshold)
    normal = test_scores[labels == 0]
    faults = test_scores[labels == 1]
    result["normal_mean"] = float(np.mean(normal)) if normal.size else float("nan")
    result["fault_mean"] = float(np.mean(faults)) if faults.size else float("nan")
    result["separation"] = (
        float(np.mean(faults) / np.mean(normal)) if normal.size and faults.size and np.mean(normal) > 0 else float("nan")
    )
    return result


def per_condition_metrics(
    rows: list[dict[str, object]], labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = f"{row['sensor']}|load={row['load']}|{row['fault_type']}|severity={row['severity']}"
        groups.setdefault(key, []).append(index)
    return {
        key: classification_metrics(labels[indices], scores[indices], threshold)
        for key, indices in sorted(groups.items())
    }
