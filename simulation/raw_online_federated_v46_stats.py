"""V46 full-statistics helpers; pure numerics, no file access."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(y) != len(s) or not np.any(y):
        return None
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    tp = np.cumsum(ranked)
    precision = tp / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / np.sum(ranked))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    p = np.asarray(y_pred, dtype=bool).reshape(-1)
    if len(y) != len(p):
        raise ValueError("truth/prediction length mismatch")
    tp = int(np.sum(y & p))
    tn = int(np.sum(~y & ~p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = None if precision is None or recall is None or precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    balanced = None if recall is None or specificity is None else float((recall + specificity) / 2)
    return {
        "n": int(len(y)), "positives": int(np.sum(y)), "negatives": int(np.sum(~y)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _safe_div(tp + tn, len(y)), "balanced_accuracy": balanced,
        "specificity": specificity, "precision": precision, "recall": recall, "f1": f1,
        "auprc": average_precision(y, p.astype(float) if score is None else np.asarray(score, dtype=float)),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def event_summary(rows: Sequence[dict]) -> dict:
    total = len(rows)
    detected = sum(bool(row["preterminal_detected"]) for row in rows)
    alarmed = sum(bool(row["ever_alarmed"]) for row in rows)
    leads = np.asarray([row["lead_acquisitions"] for row in rows if row.get("lead_acquisitions") is not None], dtype=float)
    fractions = np.asarray([row["lead_fraction"] for row in rows if row.get("lead_fraction") is not None], dtype=float)
    low, high = wilson_interval(detected, total)
    return {
        "machines": total, "preterminal_detected": detected,
        "preterminal_detection_rate": _safe_div(detected, total),
        "preterminal_detection_ci_low": low, "preterminal_detection_ci_high": high,
        "ever_alarmed": alarmed, "missed_or_postterminal": total - detected,
        "lead_n": int(len(leads)),
        "lead_acquisitions_median": float(np.median(leads)) if len(leads) else None,
        "lead_acquisitions_q1": float(np.quantile(leads, 0.25)) if len(leads) else None,
        "lead_acquisitions_q3": float(np.quantile(leads, 0.75)) if len(leads) else None,
        "lead_fraction_median": float(np.median(fractions)) if len(fractions) else None,
        "guard_exceedances": int(sum(int(row.get("guard_exceedances", 0)) for row in rows)),
        "guard_acquisitions": int(sum(int(row.get("guard_count", 0)) for row in rows)),
        "guard_exceedance_rate": _safe_div(
            sum(int(row.get("guard_exceedances", 0)) for row in rows),
            sum(int(row.get("guard_count", 0)) for row in rows),
        ),
    }


def event_summary_by_dataset(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    return [{"dataset": dataset, **event_summary(items)} for dataset, items in sorted(grouped.items())]


def timeline_statistics(timelines: Sequence[dict]) -> dict:
    if not timelines:
        return {"pooled": None, "macro": None, "machines": 0, "streams": 0}
    pooled_y = np.concatenate([np.asarray(row["truth"], dtype=bool) for row in timelines])
    pooled_p = np.concatenate([np.asarray(row["prediction"], dtype=bool) for row in timelines])
    pooled_s = np.concatenate([np.asarray(row["score"], dtype=float) for row in timelines])
    per_machine = [binary_metrics(row["truth"], row["prediction"], row["score"]) for row in timelines]
    keys = ("accuracy", "balanced_accuracy", "specificity", "precision", "recall", "f1", "auprc")
    macro = {}
    for key in keys:
        values = [row[key] for row in per_machine if row[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    return {
        "pooled": binary_metrics(pooled_y, pooled_p, pooled_s),
        "macro": macro,
        "machines": len(timelines),
        "streams": int(sum(int(row.get("stream_count", 1)) for row in timelines)),
        "per_machine": [
            {"machine_id": timeline["machine_id"], "dataset": timeline["dataset"], **metrics}
            for timeline, metrics in zip(timelines, per_machine)
        ],
    }


def clustered_bootstrap_timelines(
    timelines: Sequence[dict], draws: int, seed: int, estimand: str = "pooled"
) -> dict[str, dict[str, float | None]]:
    keys = ("accuracy", "balanced_accuracy", "specificity", "precision", "recall", "f1", "auprc")
    if not timelines:
        return {key: {"estimate": None, "ci_low": None, "ci_high": None} for key in keys}
    full = timeline_statistics(timelines)
    observed = full[estimand]
    rng = np.random.default_rng(int(seed))
    n_machines = len(timelines)
    indices = rng.integers(0, n_machines, size=(int(draws), n_machines))
    samples: dict[str, np.ndarray] = {}
    if estimand == "macro":
        per_machine = full["per_machine"]
        for key in keys:
            values = np.asarray([np.nan if row[key] is None else row[key] for row in per_machine], dtype=float)
            with np.errstate(invalid="ignore"):
                samples[key] = np.nanmean(values[indices], axis=1)
    else:
        counts = []
        for timeline in timelines:
            metric = binary_metrics(timeline["truth"], timeline["prediction"], timeline["score"])
            counts.append([metric["tp"], metric["tn"], metric["fp"], metric["fn"]])
        counts_array = np.asarray(counts, dtype=float)
        weights = np.zeros((int(draws), n_machines), dtype=float)
        for row_index in range(int(draws)):
            weights[row_index] = np.bincount(indices[row_index], minlength=n_machines)
        totals = weights @ counts_array
        tp, tn, fp, fn = (totals[:, index] for index in range(4))
        total = tp + tn + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            accuracy = (tp + tn) / total
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            balanced = (specificity + recall) / 2.0
            f1 = 2.0 * precision * recall / (precision + recall)
        samples.update({
            "accuracy": accuracy, "balanced_accuracy": balanced,
            "specificity": specificity, "precision": precision,
            "recall": recall, "f1": f1,
        })
        # Exact pooled AUPRC under a cluster resample is more expensive because
        # machine multiplicities change ranking weights. Use up to 1,000 exact
        # cluster draws while the confusion-derived metrics retain all draws.
        ap_draws = min(int(draws), 1000)
        all_scores = np.concatenate([np.asarray(row["score"], dtype=float) for row in timelines])
        all_truth = np.concatenate([np.asarray(row["truth"], dtype=bool) for row in timelines])
        machine_index = np.concatenate([
            np.full(len(np.asarray(row["truth"])), index, dtype=int) for index, row in enumerate(timelines)
        ])
        order = np.argsort(-all_scores, kind="mergesort")
        truth_ordered = all_truth[order].astype(float)
        machine_ordered = machine_index[order]
        ap_values = []
        for draw_index in range(ap_draws):
            point_weight = weights[draw_index, machine_ordered]
            positives = float(np.sum(truth_ordered * point_weight))
            if positives <= 0:
                continue
            cumulative_tp = np.cumsum(truth_ordered * point_weight)
            cumulative_n = np.cumsum(point_weight)
            precision_rank = np.divide(cumulative_tp, cumulative_n, out=np.zeros_like(cumulative_tp), where=cumulative_n > 0)
            ap_values.append(float(np.sum(precision_rank * truth_ordered * point_weight) / positives))
        samples["auprc"] = np.asarray(ap_values, dtype=float)
    result = {}
    for key in keys:
        values = np.asarray(samples.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        result[key] = {
            "estimate": observed[key],
            "ci_low": float(np.quantile(values, 0.025)) if len(values) else None,
            "ci_high": float(np.quantile(values, 0.975)) if len(values) else None,
            "bootstrap_draws": int(len(values)),
        }
    return result


def onset_event(first_alarm: int, onset: float | None, terminal: int, n: int, tolerance: float) -> str:
    if onset is None:
        return "onset_unavailable"
    if first_alarm < 0:
        return "missed"
    adjusted = float(onset) - float(tolerance) * max(float(n - 1 - onset), 0.0)
    if first_alarm < adjusted:
        return "premature"
    if first_alarm < terminal:
        return "valid"
    return "late"


def onset_tolerance_panel(rows: Sequence[dict], tolerances: Sequence[float] = (0.0, 0.05, 0.10, 0.20)) -> list[dict]:
    output = []
    for tolerance in tolerances:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[onset_event(int(row["first_alarm"]), row.get("onset"), int(row["terminal"]), int(row["n"]), float(tolerance))] += 1
        output.append({"tolerance": float(tolerance), **{key: counts.get(key, 0) for key in ("premature", "valid", "late", "missed", "onset_unavailable")}})
    return output


__all__ = [
    "average_precision", "binary_metrics", "clustered_bootstrap_timelines",
    "event_summary", "event_summary_by_dataset", "onset_event",
    "onset_tolerance_panel", "timeline_statistics", "wilson_interval",
]
