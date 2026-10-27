"""V92 source-only nested cross-dataset federated causal risk experiment.

V92 trains a tiny logistic risk layer by equal-client full-batch FedAvg.  Every
outer dataset is scored by a model that never sees that dataset.  Its alarm
threshold and persistence are selected from inner leave-one-dataset-out
predictions made only inside the outer training pool.

This is deliberately development evidence.  B07, the reserved LDM cohort, and
all target artifacts are forbidden dependencies.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

import raw_online_federated_v57_policy as v57
import raw_online_federated_v62 as v62
import raw_online_federated_v65 as v65


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v92_protocol.json"
FORBIDDEN_PATH_TOKENS = (
    "b07", "ldm_reserved_cohort",
)


class IntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlarmCandidate:
    threshold: float
    k: int
    window: int

    @property
    def name(self) -> str:
        return f"z{self.threshold:g}__{self.k}of{self.window}"


@dataclass(frozen=True)
class FedLogistic:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def root_for(run_id: str) -> Path:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not run_id or any(char not in allowed for char in run_id):
        raise ValueError("invalid run-id")
    return RESULTS / f"raw_online_federated_v92_{run_id}"


def dataset_of(device: dict[str, Any]) -> str:
    return str(device["machine_id"]).split("::", 1)[0]


def is_control(device: dict[str, Any]) -> bool:
    return device["truth_kind"] == "healthy_control"


def is_lifecycle(device: dict[str, Any]) -> bool:
    return device["tier"] in {"historical_run_to_failure", "ldm_upb_complete_lifecycle"}


def is_evaluable(device: dict[str, Any]) -> bool:
    if not is_lifecycle(device) or device["truth"] is None:
        return False
    terminal = int(math.floor(0.9 * int(device["n"])))
    return int(device["truth"]) < terminal and int(device["n"]) > v57.OP_START


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    center = np.median(array, axis=0)
    mad = 1.4826 * np.median(np.abs(array - center), axis=0)
    std = np.std(array, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-6), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return center, scale


def _causal_change(values: np.ndarray, recent: int, prior: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    output = np.zeros(len(x), dtype=np.float64)
    total = int(recent) + int(prior)
    for index in range(total - 1, len(x)):
        now = x[index - recent + 1:index + 1]
        before = x[index - total + 1:index - recent + 1]
        output[index] = float(np.median(now) - np.median(before))
    return output


def _rolling_fraction(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    cumulative = np.concatenate(([0.0], np.cumsum(x)))
    output = np.zeros(len(x), dtype=np.float64)
    for index in range(len(x)):
        start = max(0, index - int(window) + 1)
        output[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return output


def causal_feature_matrix(device: dict[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return label-blind, backward-only features for one machine."""
    streams = [np.asarray(stream, dtype=np.float64) for stream in device["feature_streams"]]
    arrays = device["v65_arrays"]
    n = int(device["n"])
    if not streams or any(stream.shape != (n, 100) for stream in streams):
        raise IntegrityError(f"invalid feature stream: {device['machine_id']}")

    per_stream: list[np.ndarray] = []
    for stream in streams:
        center, scale = _robust_center_scale(stream[v57.SCORE_CAL])
        absolute = np.abs((stream - center) / scale)
        logged = np.log1p(absolute)
        global_health = np.mean(np.sort(logged, axis=1)[:, -10:], axis=1)
        impulse_health = np.mean(
            np.sort(logged[:, v57.IMPULSE_INDICES], axis=1)[:, -10:], axis=1
        )
        rms_health = np.max(logged[:, np.asarray([0, 32, 64])], axis=1)
        profile_health = np.mean(np.sort(logged[:, :96], axis=1)[:, -8:], axis=1)
        per_stream.append(np.stack((global_health, impulse_health, rms_health, profile_health), axis=1))
    level = np.max(np.stack(per_stream, axis=0), axis=0)

    base_names = ("global_level", "impulse_level", "rms_level", "profile_level")
    names = list(base_names)
    columns = [level[:, index] for index in range(level.shape[1])]
    for recent, prior in ((4, 16), (8, 32), (16, 64)):
        for index, base_name in enumerate(base_names):
            columns.append(_causal_change(level[:, index], recent, prior))
            names.append(f"{base_name}_change_{recent}_{prior}")

    ae = np.asarray(arrays["ae"], dtype=np.float64).reshape(-1)
    trend = np.asarray(arrays["trend"], dtype=np.float64).reshape(-1)
    trend = np.where(np.isfinite(trend), trend, 0.0)
    reference = np.asarray(arrays["reference"], dtype=np.float64).reshape(-1)
    watch = np.asarray(arrays["raw_watch"], dtype=np.float64).reshape(-1)
    columns.extend((ae, trend, reference, watch, _rolling_fraction(reference, 10), _rolling_fraction(watch, 10)))
    names.extend(("ae", "impulse_trend", "reference", "watch", "reference_fraction10", "watch_fraction10"))
    matrix = np.stack(columns, axis=1)
    if matrix.shape[0] != n or not np.all(np.isfinite(matrix)):
        raise IntegrityError(f"non-finite causal matrix: {device['machine_id']}")
    return matrix, tuple(names)


def _even_sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if len(values) <= int(maximum):
        return values
    positions = np.linspace(0, len(values) - 1, int(maximum), dtype=np.int64)
    return values[positions]


def client_training_rows(
    device: dict[str, Any], matrix: np.ndarray, maximum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(v57.OP_START)
    indices = np.arange(start, int(device["n"]), dtype=np.int64)
    if is_control(device):
        selected = _even_sample(indices, 2 * int(maximum))
        return matrix[selected], np.zeros(len(selected), dtype=np.float64), np.ones(len(selected), dtype=np.float64)
    if device["truth"] is None:
        return np.empty((0, matrix.shape[1])), np.empty(0), np.empty(0)
    onset = int(device["truth"])
    negative = _even_sample(indices[indices < onset], int(maximum))
    positive = _even_sample(indices[indices >= onset], int(maximum))
    selected = np.concatenate((negative, positive))
    labels = np.concatenate((np.zeros(len(negative)), np.ones(len(positive))))
    if not len(selected):
        return np.empty((0, matrix.shape[1])), np.empty(0), np.empty(0)
    weights = np.ones(len(selected), dtype=np.float64)
    if len(negative) and len(positive):
        weights[:len(negative)] = 0.5 / len(negative)
        weights[len(negative):] = 0.5 / len(positive)
    else:
        weights[:] = 1.0 / len(weights)
    return matrix[selected], labels, weights


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_fed_logistic(
    devices: list[dict[str, Any]], matrices: dict[str, np.ndarray], spec: dict[str, Any],
) -> FedLogistic:
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    maximum = int(spec["maximum_samples_per_class_per_device"])
    for device in devices:
        values = client_training_rows(device, matrices[device["machine_id"]], maximum)
        if len(values[1]):
            rows.append(values)
    if len(rows) < 2 or not any(np.any(labels == 1) for _, labels, _ in rows):
        raise IntegrityError("insufficient federated training clients")

    all_x = np.concatenate([values for values, _, _ in rows], axis=0)
    mean = np.mean(all_x, axis=0)
    scale = np.std(all_x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    standardized = [((values - mean) / scale, labels, weights) for values, labels, weights in rows]

    weight = np.zeros(all_x.shape[1], dtype=np.float64)
    bias = 0.0
    learning_rate = float(spec["learning_rate"])
    l2 = float(spec["l2"])
    for _ in range(int(spec["rounds"])):
        local_weights: list[np.ndarray] = []
        local_biases: list[float] = []
        for values, labels, sample_weight in standardized:
            local_weight = weight.copy()
            local_bias = float(bias)
            normalizer = float(np.sum(sample_weight))
            for _ in range(int(spec["local_epochs"])):
                probability = _sigmoid(values @ local_weight + local_bias)
                residual = (probability - labels) * sample_weight
                gradient = values.T @ residual / normalizer + l2 * local_weight
                bias_gradient = float(np.sum(residual) / normalizer)
                local_weight -= learning_rate * gradient
                local_bias -= learning_rate * bias_gradient
            local_weights.append(local_weight)
            local_biases.append(local_bias)
        weight = np.mean(np.stack(local_weights), axis=0)
        bias = float(np.mean(local_biases))
    return FedLogistic(mean=mean, scale=scale, weight=weight, bias=bias)


def calibrated_risk(model: FedLogistic, matrix: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(matrix, dtype=np.float64) - model.mean) / model.scale
    logit = standardized @ model.weight + float(model.bias)
    calibration = logit[v57.SCORE_CAL]
    center = float(np.median(calibration))
    mad = float(1.4826 * np.median(np.abs(calibration - center)))
    std = float(np.std(calibration))
    scale = mad if np.isfinite(mad) and mad > 1e-6 else std if np.isfinite(std) and std > 1e-6 else 1.0
    return (logit - center) / scale


def first_alarm(score: np.ndarray, candidate: AlarmCandidate) -> int:
    raw = np.isfinite(score) & (np.asarray(score) >= float(candidate.threshold))
    raw[:min(v57.OP_START, len(raw))] = False
    alarm = v57.persistent_latch(raw, k=int(candidate.k), window=int(candidate.window))
    return v57.first_alarm(alarm)


def alarm_grid(spec: dict[str, Any]) -> list[AlarmCandidate]:
    rule = spec["alarm_candidates"]
    return [
        AlarmCandidate(float(threshold), int(k), int(window))
        for threshold in rule["thresholds"]
        for k, window in rule["persistence"]
    ]


def event(first: int, device: dict[str, Any]) -> str:
    return v62._outcome(int(first), device)


def summarize_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    outcomes = [str(row["outcome"]) for row in rows]
    return {name: outcomes.count(name) for name in sorted(set(outcomes))}


def select_alarm_candidate(
    devices: list[dict[str, Any]], scores: dict[str, np.ndarray], spec: dict[str, Any],
) -> tuple[AlarmCandidate, list[dict[str, Any]]]:
    evaluated: list[dict[str, Any]] = []
    for candidate in alarm_grid(spec):
        rows = [
            {"device": d["machine_id"], "outcome": event(first_alarm(scores[d["machine_id"]], candidate), d)}
            for d in devices
        ]
        counts = summarize_predictions(rows)
        lifecycle = sum(1 for d in devices if is_evaluable(d))
        valid = sum(row["outcome"] == "valid" for row, d in zip(rows, devices) if is_evaluable(d))
        premature = sum(row["outcome"] == "premature" for row, d in zip(rows, devices) if is_evaluable(d))
        controls = sum(row["outcome"] == "false_alarm" for row, d in zip(rows, devices) if is_control(d))
        safe = premature <= 1 and controls == 0
        evaluated.append({
            "candidate": candidate, "safe": safe, "valid": valid,
            "eligible": lifecycle, "premature": premature, "controls": controls,
            "missed": counts.get("missed", 0) + counts.get("blind", 0),
            "late": counts.get("late", 0),
        })
    safe = [row for row in evaluated if row["safe"]]
    pool = safe if safe else evaluated
    selected = min(pool, key=lambda row: (
        not row["safe"], -row["valid"], row["premature"], row["missed"], row["late"],
        -(row["candidate"].k / row["candidate"].window), -row["candidate"].threshold,
    ))
    return selected["candidate"], evaluated


def nested_run(devices: list[dict[str, Any]], matrices: dict[str, np.ndarray], spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = sorted({dataset_of(device) for device in devices})
    outer_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for held_out in datasets:
        outer_train = [device for device in devices if dataset_of(device) != held_out]
        outer_test = [device for device in devices if dataset_of(device) == held_out]
        inner_scores: dict[str, np.ndarray] = {}
        inner_datasets = sorted({dataset_of(device) for device in outer_train})
        for inner_held_out in inner_datasets:
            inner_train = [device for device in outer_train if dataset_of(device) != inner_held_out]
            inner_test = [device for device in outer_train if dataset_of(device) == inner_held_out]
            model = fit_fed_logistic(inner_train, matrices, spec)
            for device in inner_test:
                inner_scores[device["machine_id"]] = calibrated_risk(model, matrices[device["machine_id"]])
        if set(inner_scores) != {device["machine_id"] for device in outer_train}:
            raise IntegrityError(f"incomplete inner predictions for {held_out}")
        candidate, evaluated = select_alarm_candidate(outer_train, inner_scores, spec)
        chosen = next(row for row in evaluated if row["candidate"] == candidate)
        selection_rows.append({
            "outer_held_out_dataset": held_out, "selected_candidate": candidate.name,
            "inner_safe": chosen["safe"], "inner_valid": chosen["valid"],
            "inner_eligible": chosen["eligible"], "inner_premature": chosen["premature"],
            "inner_control_alarms": chosen["controls"], "inner_missed_or_blind": chosen["missed"],
            "inner_late": chosen["late"],
        })
        outer_model = fit_fed_logistic(outer_train, matrices, spec)
        for device in outer_test:
            score = calibrated_risk(outer_model, matrices[device["machine_id"]])
            first = first_alarm(score, candidate)
            outer_rows.append({
                "outer_held_out_dataset": held_out, "machine_id": device["machine_id"],
                "tier": device["tier"], "n": int(device["n"]), "truth_kind": device["truth_kind"],
                "truth": "" if device["truth"] is None else int(device["truth"]),
                "eligible": is_evaluable(device), "candidate": candidate.name,
                "first_alarm": first, "outcome": event(first, device),
            })
    if len(outer_rows) != len(devices) or len({row["machine_id"] for row in outer_rows}) != len(devices):
        raise IntegrityError("outer predictions do not cover every device exactly once")
    return outer_rows, selection_rows


def gate(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    historical = [row for row in rows if row["tier"] == "historical_run_to_failure" and row["eligible"]]
    controls = [row for row in rows if row["truth_kind"] == "healthy_control"]
    valid = sum(row["outcome"] == "valid" for row in historical)
    premature = sum(row["outcome"] == "premature" for row in historical)
    control_alarms = sum(row["outcome"] == "false_alarm" for row in controls)
    expected = protocol["cohort"]
    limits = protocol["primary_gate"]
    checks = {
        "historical_denominator_locked": len(historical) == int(expected["historical_eligible"]),
        "historical_valid_min": valid >= int(limits["historical_valid_min"]),
        "historical_valid_rate_strict": (valid / len(historical)) > float(limits["historical_valid_rate_strictly_greater_than"]),
        "historical_premature_max": premature <= int(limits["historical_premature_max"]),
        "healthy_control_alarms_max": control_alarms <= int(limits["healthy_control_alarms_max"]),
        "all_outer_devices_reported": len(rows) == int(expected["expected_devices"]),
        "no_target_dependency": True,
    }
    summary = {
        "historical_eligible": len(historical), "historical_valid": valid,
        "historical_valid_rate": valid / len(historical), "historical_premature": premature,
        "historical_late": sum(row["outcome"] == "late" for row in historical),
        "historical_missed": sum(row["outcome"] in {"missed", "blind"} for row in historical),
        "healthy_controls": len(controls), "healthy_control_alarms": control_alarms,
    }
    return all(checks.values()), {"checks": checks, "summary": summary}


def all_phase(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise IntegrityError(f"results root is not empty: {root}")
    protocol = read_json(PROTOCOL)
    if any(protocol["cohort"].get(name) is not False for name in ("b07_used", "reserved_ldm_used", "target_data_used")):
        raise IntegrityError("protocol does not seal targets")
    devices = v65.prepare_devices()
    if len(devices) != int(protocol["cohort"]["expected_devices"]):
        raise IntegrityError("source cohort size drift")
    for device in devices:
        dependency_paths = list(device.get("feature_paths", ())) + [device.get("prediction_path")]
        for path in dependency_paths:
            if path is None:
                continue
            lowered = str(path).lower()
            if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
                raise IntegrityError(f"forbidden target dependency: {path}")

    matrices: dict[str, np.ndarray] = {}
    feature_names: tuple[str, ...] | None = None
    for device in devices:
        matrix, names = causal_feature_matrix(device)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise IntegrityError("feature schema drift")
        matrices[device["machine_id"]] = matrix

    rows, selections = nested_run(devices, matrices, protocol["design"])
    passed, result = gate(rows, protocol)
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "v92_outer_ledger.csv", rows)
    write_rows(root / "v92_inner_selections.csv", selections)
    write_json(root / "v92_gate.json", {"passed": passed, **result})
    dependencies = {
        "protocol": PROTOCOL,
        "driver": Path(__file__).resolve(),
        "v65_lock": v65.RESULTS / "raw_online_federated_v65_frozen1" / "v65_development_lock.json",
    }
    write_json(root / "v92_dependency_lock.json", {
        "classification": protocol["classification"],
        "b07_used": False, "reserved_ldm_used": False, "target_data_used": False,
        "feature_names": list(feature_names or ()),
        "dependency_sha256": {name: sha256(path) for name, path in dependencies.items()},
    })
    summary = result["summary"]
    report = [
        "# V92 nested cross-dataset federated causal risk result", "",
        "**PRIMARY GATE PASSED.**" if passed else "**PRIMARY GATE FAILED -- SCIENTIFIC SAFETY STOP.**", "",
        f"Historical held-out datasets: {summary['historical_valid']}/{summary['historical_eligible']} valid "
        f"({100.0 * summary['historical_valid_rate']:.2f}%), {summary['historical_premature']} premature, "
        f"{summary['historical_late']} late, {summary['historical_missed']} missed/blind.",
        f"Healthy controls: {summary['healthy_control_alarms']}/{summary['healthy_controls']} alarmed.", "",
        "Every row was produced by a model and alarm candidate selected without its outer dataset. "
        "This remains nested SOURCE development evidence, not independent validation. B07 and the reserved LDM cohort were not accessed.", "",
    ]
    (root / "V92_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(
        f"V92 {'PASS' if passed else 'STOP'}: valid={summary['historical_valid']}/{summary['historical_eligible']} "
        f"premature={summary['historical_premature']} controls={summary['healthy_control_alarms']}/{summary['healthy_controls']}",
        flush=True,
    )
    if not passed:
        raise IntegrityError("V92 primary >80% gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("all",))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        all_phase(root_for(args.run_id))
    except Exception as exc:
        print(f"V92 FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
