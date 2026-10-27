"""Causal analytical-baseline benchmark for the frozen V64 detector.

The benchmark uses the same 66 burned development devices and the exact V64
timing/safety gates. Device calibration is chronological and label blind. B07
is neither required nor accessed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import raw_online_federated_v45_cohort as cohort
import raw_online_federated_v57_policy as v57
import raw_online_federated_v62 as v62


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v64_baseline_protocol.json"
V64_ROOT = RESULTS / "raw_online_federated_v64_frozen1"
V63_ROOT = RESULTS / "raw_online_federated_v63_frozen1"
V52_ROOT = RESULTS / "raw_online_federated_v52r1_corrected1" / "feature_streams"
V58_ROOT = RESULTS / "raw_online_federated_v58_frozen1" / "feature_streams"
V61_ROOT = RESULTS / "raw_online_federated_v61_frozen1" / "feature_streams"

RMS_INDICES = np.asarray([0, 32, 64], dtype=np.int64)
PROFILE_INDICES = np.arange(96, dtype=np.int64)
SAFETY_CHECKS = (
    "historical_premature_max", "hit_false_alarms_max",
    "ldm_premature_max", "paderborn_false_alarms_max",
)


class IntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    family: str
    candidate_id: str
    threshold: float | None = None
    k: int | None = None
    window: int | None = None
    components: int | None = None
    alpha: float | None = None
    drift: float | None = None
    limit: float | None = None


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
    return RESULTS / f"raw_online_federated_v64_baselines_{run_id}"


def candidate_grid(spec: dict | None = None) -> list[Candidate]:
    families = (spec or read_json(PROTOCOL))["families"]
    result: list[Candidate] = []
    for family in ("rms_z", "impulse_z", "profile_z", "impulse_trend", "federated_ae_only"):
        rule = families[family]
        for threshold in rule["thresholds"]:
            for k, window in rule["persistence"]:
                name = f"{family}__t{threshold:g}__k{k}of{window}"
                result.append(Candidate(family, name, float(threshold), int(k), int(window)))
    rule = families["local_pca"]
    for components in rule["components"]:
        for threshold in rule["thresholds"]:
            for k, window in rule["persistence"]:
                name = f"local_pca{components}__t{threshold:g}__k{k}of{window}"
                result.append(Candidate("local_pca", name, float(threshold), int(k), int(window), int(components)))
    rule = families["rms_ewma"]
    k, window = rule["persistence"]
    for alpha in rule["alphas"]:
        for threshold in rule["thresholds"]:
            name = f"rms_ewma__a{alpha:g}__t{threshold:g}__k{k}of{window}"
            result.append(Candidate("rms_ewma", name, float(threshold), int(k), int(window), alpha=float(alpha)))
    rule = families["rms_cusum"]
    for drift in rule["drifts"]:
        for limit in rule["limits"]:
            name = f"rms_cusum__d{drift:g}__h{limit:g}"
            result.append(Candidate("rms_cusum", name, drift=float(drift), limit=float(limit)))
    if len({item.candidate_id for item in result}) != len(result):
        raise IntegrityError("duplicate candidate id")
    return result


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    center = np.median(array, axis=0)
    mad = 1.4826 * np.median(np.abs(array - center), axis=0)
    std = np.std(array, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center, scale


def _level_score(values: np.ndarray, indices: np.ndarray, *, absolute: bool) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)[:, indices]
    center, scale = _robust_center_scale(selected[20:84])
    z = (selected - center) / scale
    if absolute:
        z = np.abs(z)
    return np.max(z, axis=1)


def _pca_scores(values: np.ndarray, components: tuple[int, ...]) -> dict[int, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    center, scale = _robust_center_scale(array[:50])
    z = (array - center) / scale
    fit_center = np.mean(z[:50], axis=0)
    centered = z - fit_center
    _, _, vt = np.linalg.svd(centered[:50], full_matrices=False)
    result: dict[int, np.ndarray] = {}
    for count in components:
        basis = vt[:count]
        residual = centered - (centered @ basis.T) @ basis
        raw = np.log1p(np.mean(residual * residual, axis=1))
        score_center, score_scale = _robust_center_scale(raw[50:84, None])
        result[count] = (raw - float(score_center[0])) / float(score_scale[0])
    return result


def _fuse_max(parts: list[np.ndarray], n: int) -> np.ndarray:
    if not parts:
        raise ValueError("empty score parts")
    return np.max(np.stack([np.asarray(part[:n], dtype=np.float64) for part in parts]), axis=0)


def score_device(feature_streams: list[np.ndarray], ae: np.ndarray, pca_components: tuple[int, ...]) -> dict[str, np.ndarray]:
    n = min([len(stream) for stream in feature_streams] + [len(ae)])
    if n < v57.OP_START:
        raise ValueError("baseline device is too short")
    rms = [_level_score(stream[:n], RMS_INDICES, absolute=False) for stream in feature_streams]
    impulse = [_level_score(stream[:n], v57.IMPULSE_INDICES, absolute=True) for stream in feature_streams]
    profile = [_level_score(stream[:n], PROFILE_INDICES, absolute=True) for stream in feature_streams]
    trends = [v57.impulse_trend(stream[:n]) for stream in feature_streams]
    pca_by_stream = [_pca_scores(stream[:n], pca_components) for stream in feature_streams]
    result = {
        "rms_z": _fuse_max(rms, n),
        "impulse_z": _fuse_max(impulse, n),
        "profile_z": _fuse_max(profile, n),
        "impulse_trend": _fuse_max(trends, n),
        "federated_ae_only": np.asarray(ae[:n], dtype=np.float64),
    }
    for count in pca_components:
        result[f"local_pca_{count}"] = _fuse_max([item[count] for item in pca_by_stream], n)
    return result


def _persistent_first(score: np.ndarray, threshold: float, k: int, window: int) -> int:
    raw = np.isfinite(score) & (np.asarray(score) >= float(threshold))
    raw[:min(v57.OP_START, len(raw))] = False
    trailing = np.convolve(raw.astype(np.int16), np.ones(int(window), dtype=np.int16), mode="full")[:len(raw)]
    positions = np.flatnonzero(trailing >= int(k))
    return int(positions[0]) if len(positions) else -1


def candidate_first(scores: dict[str, np.ndarray], candidate: Candidate) -> int:
    if candidate.family == "local_pca":
        score = scores[f"local_pca_{candidate.components}"]
    else:
        score = scores["rms_z"] if candidate.family in ("rms_ewma", "rms_cusum") else scores[candidate.family]
    if candidate.family == "rms_ewma":
        smooth = np.zeros(len(score), dtype=np.float64)
        for index in range(v57.OP_START, len(score)):
            previous = smooth[index - 1] if index else 0.0
            smooth[index] = candidate.alpha * max(0.0, float(score[index])) + (1.0 - candidate.alpha) * previous
        return _persistent_first(smooth, candidate.threshold, candidate.k, candidate.window)
    if candidate.family == "rms_cusum":
        cumulative = 0.0
        for index in range(v57.OP_START, len(score)):
            cumulative = max(0.0, cumulative + max(0.0, float(score[index])) - candidate.drift)
            if cumulative >= candidate.limit:
                return index
        return -1
    return _persistent_first(score, candidate.threshold, candidate.k, candidate.window)


def _feature_path(machine_id: str) -> Path:
    bearing = machine_id.split("::")[-1]
    if bearing in {"B01", "B02", "B04", "B05"}:
        root = V52_ROOT
    elif bearing in {"B08", "B10", "B11", "B17"}:
        root = V58_ROOT
    elif bearing in {"B09", "B16"}:
        root = V61_ROOT
    else:
        raise KeyError(machine_id)
    return root / f"ldm_upb__{bearing}.npz"


def load_devices() -> list[dict[str, Any]]:
    devices = v62._development_devices()  # burned cohort and frozen timing labels
    registry = cohort.stream_registry()
    by_id = {row["client_id"]: row for row in registry}
    groups = cohort.group_by_experiment(list(by_id))
    for device in devices:
        if device["tier"] == "ldm_upb_complete_lifecycle":
            path = _feature_path(device["machine_id"])
            with np.load(path, allow_pickle=False) as archive:
                streams = [np.asarray(archive["features"], dtype=np.float64)]
            feature_paths = [path]
        else:
            members = groups[device["machine_id"]]
            streams = [np.asarray(cohort.load_stream(by_id[member])[0], dtype=np.float64) for member in members]
            feature_paths = [Path(by_id[member]["path"]) for member in members]
        n = min([len(stream) for stream in streams] + [device["n"]])
        if n != device["n"]:
            raise IntegrityError(f"feature/prediction length mismatch: {device['machine_id']} {n} != {device['n']}")
        device["feature_streams"] = [stream[:n] for stream in streams]
        device["feature_paths"] = feature_paths
    for bearing in ("K003", "K004", "K005", "K006"):
        path = V63_ROOT / "paderborn_predictions" / f"paderborn__{bearing}.npz"
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float64)
            ae = np.asarray(archive["ae_continuous_score"], dtype=np.float64)
        devices.append({
            "tier": "paderborn_healthy_snapshot", "evidence_status": "burned_development_v64",
            "machine_id": f"paderborn::{bearing}", "role": "healthy_snapshot", "n": len(features),
            "truth_kind": "healthy_control", "truth": None, "structurally_blind": False,
            "feature_streams": [features], "feature_paths": [path],
            "arrays": {"ae": ae}, "prediction_path": path,
        })
    if len(devices) != 66 or len({device["machine_id"] for device in devices}) != 66:
        raise IntegrityError("baseline cohort is not 66 unique devices")
    return devices


def _counts(devices: list[dict[str, Any]], firsts: dict[str, int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for tier in ("historical_run_to_failure", "hit_verified_healthy", "ldm_upb_complete_lifecycle", "paderborn_healthy_snapshot"):
        outcomes = [v62._outcome(firsts[device["machine_id"]], device) for device in devices if device["tier"] == tier]
        result[tier] = {name: outcomes.count(name) for name in sorted(set(outcomes))}
        result[tier]["devices"] = len(outcomes)
    return result


def gate(summary: dict[str, dict[str, int]], spec: dict | None = None) -> tuple[bool, dict[str, bool]]:
    limits = (spec or read_json(PROTOCOL))["v64_equivalence_gates"]
    h = summary["historical_run_to_failure"]
    c = summary["hit_verified_healthy"]
    l = summary["ldm_upb_complete_lifecycle"]
    p = summary["paderborn_healthy_snapshot"]
    checks = {
        "historical_valid_min": h.get("valid", 0) >= limits["historical_valid_min"],
        "historical_premature_max": h.get("premature", 0) <= limits["historical_premature_max"],
        "historical_missed_max": h.get("missed", 0) <= limits["historical_missed_max"],
        "hit_false_alarms_max": c.get("false_alarm", 0) <= limits["hit_false_alarms_max"],
        "ldm_valid_min": l.get("valid", 0) >= limits["ldm_valid_min"],
        "ldm_premature_max": l.get("premature", 0) <= limits["ldm_premature_max"],
        "ldm_missed_max": l.get("missed", 0) <= limits["ldm_missed_max"],
        "paderborn_false_alarms_max": p.get("false_alarm", 0) <= limits["paderborn_false_alarms_max"],
    }
    return all(checks.values()), checks


def _selection_key(item: dict[str, Any]) -> tuple:
    h = item["summary"]["historical_run_to_failure"]
    l = item["summary"]["ldm_upb_complete_lifecycle"]
    failed = item["failed_checks"]
    safety_failed = sum(name in failed for name in SAFETY_CHECKS)
    return (
        len(failed), safety_failed, -l.get("valid", 0), -h.get("valid", 0),
        h.get("missed", 0), h.get("late", 0), item["candidate"].candidate_id,
    )


def _flat_summary(candidate: Candidate, summary: dict, passed: bool, checks: dict) -> dict[str, Any]:
    h, c, l, p = (summary[name] for name in (
        "historical_run_to_failure", "hit_verified_healthy",
        "ldm_upb_complete_lifecycle", "paderborn_healthy_snapshot",
    ))
    return {
        "family": candidate.family, "candidate": candidate.candidate_id,
        "threshold": candidate.threshold, "k": candidate.k, "window": candidate.window,
        "components": candidate.components, "alpha": candidate.alpha,
        "drift": candidate.drift, "limit": candidate.limit,
        "historical_valid": h.get("valid", 0), "historical_premature": h.get("premature", 0),
        "historical_late": h.get("late", 0), "historical_missed": h.get("missed", 0),
        "historical_blind": h.get("blind", 0), "historical_unavailable": h.get("unavailable", 0),
        "hit_false_alarms": c.get("false_alarm", 0),
        "ldm_valid": l.get("valid", 0), "ldm_premature": l.get("premature", 0),
        "ldm_missed": l.get("missed", 0), "paderborn_false_alarms": p.get("false_alarm", 0),
        "all_v64_gates_pass": passed,
        "safety_admissible": all(checks[name] for name in SAFETY_CHECKS),
        "failed_checks": ";".join(name for name, ok in checks.items() if not ok),
    }


def _v64_rows() -> dict[str, dict[str, str]]:
    path = V64_ROOT / "v64_full_device_statistics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 66:
        raise IntegrityError("V64 frozen ledger does not contain 66 devices")
    return {row["machine_id"]: row for row in rows}


def all_phase(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise IntegrityError(f"results root is not empty: {root}")
    spec = read_json(PROTOCOL)
    if spec["cohort"].get("b07_opened") is not False:
        raise IntegrityError("protocol must seal B07")
    candidates = candidate_grid(spec)
    devices = load_devices()
    v64_rows = _v64_rows()
    pca_components = tuple(int(value) for value in spec["families"]["local_pca"]["components"])

    score_cache: dict[str, dict[str, np.ndarray]] = {}
    for index, device in enumerate(devices, start=1):
        score_cache[device["machine_id"]] = score_device(
            device["feature_streams"], device["arrays"]["ae"], pca_components,
        )
        if index % 10 == 0:
            print(f"V64 BASELINES scoring features: {index}/66", flush=True)

    evaluated: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        firsts = {device["machine_id"]: candidate_first(score_cache[device["machine_id"]], candidate) for device in devices}
        summary = _counts(devices, firsts)
        passed, checks = gate(summary, spec)
        item = {"candidate": candidate, "firsts": firsts, "summary": summary, "passed": passed,
                "checks": checks, "failed_checks": {name for name, ok in checks.items() if not ok}}
        evaluated.append(item)
        sweep_rows.append(_flat_summary(candidate, summary, passed, checks))
        if index % 50 == 0:
            print(f"V64 BASELINES evaluating candidates: {index}/{len(candidates)}", flush=True)

    selected: dict[str, dict[str, Any]] = {}
    family_rows: list[dict[str, Any]] = []
    for family in spec["families"]:
        members = [item for item in evaluated if item["candidate"].family == family]
        passing = [item for item in members if item["passed"]]
        safe = [item for item in members if all(item["checks"][name] for name in SAFETY_CHECKS)]
        chosen = min(passing or safe or members, key=_selection_key)
        selected[family] = chosen
        row = _flat_summary(chosen["candidate"], chosen["summary"], chosen["passed"], chosen["checks"])
        row["family_candidates"] = len(members)
        row["family_passing_candidates"] = len(passing)
        row["selection_status"] = "full_gate_pass" if passing else ("safe_diagnostic" if safe else "unsafe_diagnostic")
        family_rows.append(row)

    device_rows: list[dict[str, Any]] = []
    for device in devices:
        machine_id = device["machine_id"]
        source_hashes = [sha256(path) for path in device["feature_paths"]]
        row: dict[str, Any] = {
            "tier": device["tier"], "evidence_status": "burned_development_v64_baseline_benchmark",
            "machine_id": machine_id, "n": device["n"], "truth_kind": device["truth_kind"],
            "truth": "" if device["truth"] is None else device["truth"],
            "v64_first": v64_rows[machine_id]["v64_first"], "v64_outcome": v64_rows[machine_id]["v64_outcome"],
            "feature_source_sha256": ";".join(source_hashes),
        }
        for family, item in selected.items():
            first = item["firsts"][machine_id]
            row[f"{family}_candidate"] = item["candidate"].candidate_id
            row[f"{family}_first"] = first
            row[f"{family}_outcome"] = v62._outcome(first, device)
        device_rows.append(row)

    analytical = spec["selection"]["analytical_families"]
    analytical_passes = [family for family in analytical if selected[family]["passed"]]
    ae_only_pass = selected["federated_ae_only"]["passed"]
    conclusion = (
        "At least one causal analytical family matched every V64 development gate; the burned cohort does not establish that the autoencoder is necessary."
        if analytical_passes else
        "No tested causal analytical family matched every V64 development gate; within this burned cohort, the frozen hybrid has incremental empirical value over these bounded baselines."
    )
    conclusion += (
        " Federated-AE-only also passed, so the impulse corroboration is not required to meet these development gates."
        if ae_only_pass else
        " Federated-AE-only did not pass, so the result supports the hybrid evidence structure rather than AE evidence alone."
    )

    dependencies: dict[str, Path] = {
        "protocol": PROTOCOL, "driver": Path(__file__).resolve(),
        "v64_lock": V64_ROOT / "v64_development_lock.json",
        "v64_full_statistics": V64_ROOT / "v64_full_device_statistics.csv",
    }
    for device in devices:
        for number, path in enumerate(device["feature_paths"], start=1):
            dependencies[f"feature::{device['machine_id']}::{number}"] = path

    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "v64_baseline_candidate_sweep.csv", sweep_rows)
    write_rows(root / "v64_baseline_family_summary.csv", family_rows)
    write_rows(root / "v64_baseline_full_device_statistics.csv", device_rows)
    lock = {
        "status": "benchmark_complete", "development_only": True, "devices_reported": len(device_rows),
        "candidate_count": len(candidates), "families": len(selected),
        "analytical_families_passing_all_v64_gates": analytical_passes,
        "federated_ae_only_passed_all_v64_gates": ae_only_pass,
        "selected_candidates": {family: item["candidate"].candidate_id for family, item in selected.items()},
        "conclusion": conclusion, "b07_opened": False,
        "dependency_sha256": {name: sha256(path) for name, path in dependencies.items()},
    }
    write_json(root / "v64_baseline_benchmark_lock.json", lock)

    def table_row(label: str, row: dict[str, Any]) -> str:
        return (f"| {label} | {row['historical_valid']} | {row['historical_premature']} | "
                f"{row['historical_late']} | {row['historical_missed']} | {row['historical_blind']} | "
                f"{row['hit_false_alarms']} | {row['ldm_valid']} | {row['ldm_premature']} | "
                f"{row['ldm_missed']} | {row['paderborn_false_alarms']} | {row['all_v64_gates_pass']} |")

    v64_reference = {
        "historical_valid": 26, "historical_premature": 1, "historical_late": 10,
        "historical_missed": 5, "historical_blind": 3, "hit_false_alarms": 0,
        "ldm_valid": 9, "ldm_premature": 1, "ldm_missed": 0,
        "paderborn_false_alarms": 0, "all_v64_gates_pass": True,
    }
    lines = [
        "# V64 causal analytical-baseline benchmark", "",
        "**BURNED-DATA DEVELOPMENT BENCHMARK COMPLETE. B07 REMAINS SEALED.**", "",
        conclusion, "",
        "| detector family | hist valid | hist premature | hist late | hist missed | hist blind | HIT FA | LDM valid | LDM premature | LDM missed | Paderborn FA | all V64 gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        table_row("V64 frozen", v64_reference),
    ]
    for row in family_rows:
        lines.append(table_row(f"{row['family']} ({row['selection_status']})", row))
    lines += [
        "", "## Interpretation", "",
        "All baseline calibration used only chronological prefixes and no device outcome labels. Candidate selection did use the burned cohort outcomes, exactly as a retrospective development comparison; therefore the table is not independent validation.", "",
        "The analytical families never use AE scores or V48 alarms. `federated_ae_only` is an attribution ablation, not an analytical baseline.", "",
        "This experiment does **not** prove a federated-learning advantage because it does not compare the frozen federated AE with otherwise identical local-only and centralized AEs. That requires a separate training-topology ablation.", "",
        "Paderborn K003-K006 are burned V64 safety controls consisting of short healthy snapshots. B07 was not accessed.", "",
        "Artifacts: `v64_baseline_candidate_sweep.csv`, `v64_baseline_family_summary.csv`, `v64_baseline_full_device_statistics.csv`, and `v64_baseline_benchmark_lock.json`.", "",
    ]
    (root / "V64_BASELINE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"V64 BASELINES COMPLETE: devices=66 candidates={len(candidates)} analytical_passes={analytical_passes} ae_only_pass={ae_only_pass}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all",), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        all_phase(root_for(args.run_id))
    except Exception as exc:
        print(f"V64 BASELINES FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
