"""Formal V65 burned-data replay with full 66-device statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import raw_online_federated_v57_policy as v57
import raw_online_federated_v62 as v62
import raw_online_federated_v63_policy as v63
import raw_online_federated_v64_policy as v64
import raw_online_federated_v65_policy as policy
from raw_online_federated_v64_baselines import load_devices


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v65_protocol.json"
V63_ROOT = RESULTS / "raw_online_federated_v63_frozen1"
V64_ROOT = RESULTS / "raw_online_federated_v64_frozen1"


class IntegrityError(RuntimeError):
    pass


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
    return RESULTS / f"raw_online_federated_v65_{run_id}"


def frozen_candidates() -> tuple[v63.Candidate, v64.Candidate]:
    lock63 = read_json(V63_ROOT / "v63_development_lock.json")
    lock64 = read_json(V64_ROOT / "v64_development_lock.json")
    if lock63.get("status") != "development_passed_policy_frozen":
        raise IntegrityError("V63 is not frozen")
    if lock64.get("status") != "development_passed_policy_frozen":
        raise IntegrityError("V64 is not frozen")
    return v63.Candidate(**lock63["selected_candidate"]), v64.Candidate(**lock64["selected_candidate"])


def first_alarm(values: np.ndarray) -> int:
    positions = np.flatnonzero(np.asarray(values, dtype=bool))
    return int(positions[0]) if len(positions) else -1


def prepare_devices() -> list[dict[str, Any]]:
    devices = load_devices()
    candidate63, candidate64 = frozen_candidates()
    for device in devices:
        if device["tier"] == "paderborn_healthy_snapshot":
            with np.load(device["prediction_path"], allow_pickle=False) as archive:
                raw = np.asarray(archive["v57_raw_watch_evidence"], dtype=bool)
                alarm63 = np.asarray(archive["v63_alarm"], dtype=bool)
                reference = np.asarray(archive["v48_alarm"], dtype=bool)
                collapse = bool(archive["collapse_route"])
                clip = float(archive["clip_fraction"])
                trend = np.asarray(archive["impulse_trend"], dtype=np.float64)
                ae = np.asarray(archive["ae_continuous_score"], dtype=np.float64)
        else:
            arrays = device["arrays"]
            with np.load(device["prediction_path"], allow_pickle=False) as archive:
                reference = np.asarray(archive["v48_alarm"], dtype=bool)
            raw = np.asarray(arrays["raw_watch"], dtype=bool)
            collapse = bool(arrays["collapse_route"])
            clip = float(arrays["clip_fraction"])
            trend = np.asarray(arrays["trend"], dtype=np.float64)
            ae = np.asarray(arrays["ae"], dtype=np.float64)
            alarm63 = v63.candidate_alarm(
                trend, ae, raw, arrays["v57_final"], collapse, clip, candidate63,
            )[0]
        base = v64.candidate_alarm(alarm63, raw, reference, collapse, candidate64)
        n = int(device["n"])
        if not all(len(item) == n for item in (raw, reference, trend, ae, base)):
            raise IntegrityError(f"array length mismatch: {device['machine_id']}")
        device["v65_arrays"] = {
            "raw_watch": raw, "reference": reference, "trend": trend, "ae": ae,
            "collapse_route": collapse, "clip_fraction": clip, "v64_alarm": base,
        }
    if len(devices) != 66 or len({item["machine_id"] for item in devices}) != 66:
        raise IntegrityError("V65 requires 66 unique burned development devices")
    return devices


def counts(devices: list[dict[str, Any]], candidate: policy.Candidate) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for tier in (
        "historical_run_to_failure", "hit_verified_healthy",
        "ldm_upb_complete_lifecycle", "paderborn_healthy_snapshot",
    ):
        outcomes = []
        for device in devices:
            if device["tier"] != tier:
                continue
            arrays = device["v65_arrays"]
            alarm = policy.candidate_alarm(
                arrays["v64_alarm"], arrays["raw_watch"], arrays["collapse_route"],
                arrays["clip_fraction"], candidate,
            )
            outcomes.append(v62._outcome(first_alarm(alarm), device))
        result[tier] = {name: outcomes.count(name) for name in sorted(set(outcomes))}
        result[tier]["devices"] = len(outcomes)
    return result


def gate(summary: dict[str, dict[str, int]], spec: dict | None = None) -> tuple[bool, dict[str, bool]]:
    limits = (spec or read_json(PROTOCOL))["development_gates"]
    historical = summary["historical_run_to_failure"]
    hit = summary["hit_verified_healthy"]
    ldm = summary["ldm_upb_complete_lifecycle"]
    paderborn = summary["paderborn_healthy_snapshot"]
    checks = {
        "historical_valid_min": historical.get("valid", 0) >= limits["historical_valid_min"],
        "historical_premature_max": historical.get("premature", 0) <= limits["historical_premature_max"],
        "historical_missed_max": historical.get("missed", 0) <= limits["historical_missed_max"],
        "historical_late_max": historical.get("late", 0) <= limits["historical_late_max"],
        "hit_false_alarms_max": hit.get("false_alarm", 0) <= limits["hit_false_alarms_max"],
        "ldm_valid_min": ldm.get("valid", 0) >= limits["ldm_valid_min"],
        "ldm_premature_max": ldm.get("premature", 0) <= limits["ldm_premature_max"],
        "ldm_missed_max": ldm.get("missed", 0) <= limits["ldm_missed_max"],
        "paderborn_false_alarms_max": paderborn.get("false_alarm", 0) <= limits["paderborn_false_alarms_max"],
    }
    return all(checks.values()), checks


def selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    candidate: policy.Candidate = item["candidate"]
    historical = item["summary"]["historical_run_to_failure"]
    return (
        -historical.get("valid", 0), historical.get("premature", 0),
        historical.get("missed", 0), historical.get("late", 0),
        -candidate.clip_min, -(candidate.watch_k / candidate.watch_window), candidate.name,
    )


def flat_row(candidate: policy.Candidate, summary: dict, passed: bool, checks: dict) -> dict[str, Any]:
    historical, hit, ldm, paderborn = (summary[name] for name in (
        "historical_run_to_failure", "hit_verified_healthy",
        "ldm_upb_complete_lifecycle", "paderborn_healthy_snapshot",
    ))
    return {
        "candidate": candidate.name, "clip_min": candidate.clip_min,
        "watch_k": candidate.watch_k, "watch_window": candidate.watch_window,
        "historical_valid": historical.get("valid", 0),
        "historical_premature": historical.get("premature", 0),
        "historical_late": historical.get("late", 0),
        "historical_missed": historical.get("missed", 0),
        "historical_blind": historical.get("blind", 0),
        "historical_unavailable": historical.get("unavailable", 0),
        "hit_false_alarms": hit.get("false_alarm", 0),
        "ldm_valid": ldm.get("valid", 0), "ldm_premature": ldm.get("premature", 0),
        "ldm_missed": ldm.get("missed", 0),
        "paderborn_false_alarms": paderborn.get("false_alarm", 0),
        "passed": passed,
        "failed_checks": ";".join(name for name, ok in checks.items() if not ok),
    }


def bottleneck(device: dict[str, Any]) -> str:
    arrays = device["v65_arrays"]
    if int(device["n"]) < v57.RECENT + v57.PRIOR:
        return "structurally_insufficient_impulse_history"
    raw_first = first_alarm(arrays["raw_watch"])
    reference_first = first_alarm(arrays["reference"])
    if raw_first < 0:
        return "no_persistent_AE_plus_impulse_WATCH"
    if reference_first < 0:
        return "V48_authorization_absent"
    if reference_first >= int(np.floor(0.9 * int(device["n"]))):
        return "V48_authorization_near_terminal"
    return "WATCH_and_V48_not_timely_aligned"


def all_phase(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise IntegrityError(f"results root is not empty: {root}")
    spec = read_json(PROTOCOL)
    if spec["cohort"].get("b07_used") is not False:
        raise IntegrityError("V65 protocol must exclude B07")
    devices = prepare_devices()
    candidates = policy.candidate_grid()
    evaluated: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    for candidate in candidates:
        summary = counts(devices, candidate)
        passed, checks = gate(summary, spec)
        evaluated.append({"candidate": candidate, "summary": summary, "passed": passed, "checks": checks})
        sweep.append(flat_row(candidate, summary, passed, checks))
    passing = [item for item in evaluated if item["passed"]]
    selected = min(passing, key=selection_key) if passing else None

    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "v65_candidate_sweep.csv", sweep)
    if selected is None:
        write_json(root / "v65_development_lock.json", {
            "status": "safety_stop", "candidate_count": len(candidates),
            "passing_candidates": 0, "b07_used": False,
        })
        raise IntegrityError("V65 safety stop: no candidate passed")

    v64_rows = {
        row["machine_id"]: row for row in csv.DictReader(
            (V64_ROOT / "v64_full_device_statistics.csv").open(newline="", encoding="utf-8")
        )
    }
    full_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    candidate = selected["candidate"]
    for device in devices:
        arrays = device["v65_arrays"]
        alarm = policy.candidate_alarm(
            arrays["v64_alarm"], arrays["raw_watch"], arrays["collapse_route"],
            arrays["clip_fraction"], candidate,
        )
        first64, first65 = first_alarm(arrays["v64_alarm"]), first_alarm(alarm)
        outcome64 = v62._outcome(first64, device)
        outcome65 = v62._outcome(first65, device)
        full_rows.append({
            "tier": device["tier"], "evidence_status": "burned_development_v65",
            "machine_id": device["machine_id"], "n": device["n"],
            "truth_kind": device["truth_kind"],
            "truth": "" if device["truth"] is None else device["truth"],
            "clip_fraction": arrays["clip_fraction"],
            "collapse_route": arrays["collapse_route"],
            "v64_first": first64, "v64_outcome": outcome64,
            "v65_first": first65, "v65_outcome": outcome65,
            "changed": outcome64 != outcome65 or first64 != first65,
        })
        if outcome64 in ("late", "missed"):
            finite = arrays["trend"][np.isfinite(arrays["trend"])]
            diagnostics.append({
                "machine_id": device["machine_id"], "n": device["n"],
                "truth": device["truth"], "v64_outcome": outcome64, "v64_first": first64,
                "v65_outcome": outcome65, "v65_first": first65,
                "clip_fraction": arrays["clip_fraction"],
                "collapse_route": arrays["collapse_route"],
                "raw_watch_first": first_alarm(arrays["raw_watch"]),
                "raw_watch_acquisitions": int(np.sum(arrays["raw_watch"])),
                "v48_first": first_alarm(arrays["reference"]),
                "v48_true_acquisitions": int(np.sum(arrays["reference"])),
                "max_impulse_trend": float(np.max(finite)) if len(finite) else None,
                "ae_floor_acquisitions": int(np.sum(arrays["ae"] >= v57.AE_FLOOR)),
                "bottleneck": bottleneck(device),
            })
    write_rows(root / "v65_full_device_statistics.csv", full_rows)
    write_rows(root / "v65_late_missed_route_diagnostics.csv", diagnostics)

    dependencies = {
        "protocol": PROTOCOL, "policy": HERE / "raw_online_federated_v65_policy.py",
        "driver": Path(__file__).resolve(), "v63_lock": V63_ROOT / "v63_development_lock.json",
        "v64_lock": V64_ROOT / "v64_development_lock.json",
        "v64_statistics": V64_ROOT / "v64_full_device_statistics.csv",
    }
    lock = {
        "status": "development_passed_policy_frozen", "classification": spec["classification"],
        "candidate_count": len(candidates), "passing_candidates": len(passing),
        "selected_candidate": candidate.__dict__, "selected_summary": selected["summary"],
        "devices_reported": len(full_rows), "late_missed_diagnostics": len(diagnostics),
        "b07_used": False, "b07_dependency_present": False,
        "dependency_sha256": {name: sha256(path) for name, path in dependencies.items()},
    }
    write_json(root / "v65_development_lock.json", lock)

    historical = selected["summary"]["historical_run_to_failure"]
    hit = selected["summary"]["hit_verified_healthy"]
    ldm = selected["summary"]["ldm_upb_complete_lifecycle"]
    paderborn = selected["summary"]["paderborn_healthy_snapshot"]
    changed = [row for row in full_rows if row["changed"]]
    lines = [
        "# V65 calibration-eligible WATCH maturation", "",
        "**RETROSPECTIVE BURNED-DATA DEVELOPMENT GATE PASSED.**", "",
        f"Frozen candidate: `{candidate.name}`. {len(passing)}/{len(candidates)} candidates passed.", "",
        "| tier | valid | premature | late | missed | blind | false alarms |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| historical 50 | {historical.get('valid',0)} | {historical.get('premature',0)} | {historical.get('late',0)} | {historical.get('missed',0)} | {historical.get('blind',0)} | n/a |",
        f"| HIT 2 | n/a | n/a | n/a | n/a | n/a | {hit.get('false_alarm',0)} |",
        f"| LDM 10 | {ldm.get('valid',0)} | {ldm.get('premature',0)} | n/a | {ldm.get('missed',0)} | 0 | n/a |",
        f"| Paderborn 4 | n/a | n/a | n/a | n/a | n/a | {paderborn.get('false_alarm',0)} |", "",
        "## Changes from frozen V64", "",
        *[f"- `{row['machine_id']}`: {row['v64_outcome']} at {row['v64_first']} -> {row['v65_outcome']} at {row['v65_first']}." for row in changed], "",
        "## Claim boundary", "",
        "V65 was designed and selected on the burned 66-device cohort after examining V64 failures. It is not independent validation. B07 was not used and cannot validate V65 because its V64 result is already known. A future untouched positive lifecycle is required.", "",
    ]
    (root / "V65_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"V65 COMPLETE: selected={candidate.name} passing={len(passing)}/{len(candidates)} devices={len(full_rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("all",))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        all_phase(root_for(args.run_id))
    except Exception as exc:
        print(f"V65 FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
