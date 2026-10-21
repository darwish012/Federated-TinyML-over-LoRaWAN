"""V63 adaptive, confidence-gated rescue study with held-out Paderborn controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from scipy.io import loadmat

import raw_online_federated_v12 as v12
import raw_online_federated_v45_grid as v45grid
import raw_online_federated_v52 as v52
import raw_online_federated_v57_policy as v57
import raw_online_federated_v60_analysis as timing
import raw_online_federated_v62 as v62study
import raw_online_federated_v63_policy as policy


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v63_protocol.json"
V62_LOCK = RESULTS / "raw_online_federated_v62_frozen1" / "v62_development_lock.json"
PADERBORN = Path(r"E:\masters\datasets\paderborn\archive.zip")


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def root_for(run_id: str) -> Path:
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("invalid run-id")
    return RESULTS / f"raw_online_federated_v63_{run_id}"


def dependencies() -> dict[str, Path]:
    result = {
        "protocol": PROTOCOL,
        "policy": HERE / "raw_online_federated_v63_policy.py",
        "driver": HERE / "raw_online_federated_v63.py",
        "v62_development_lock": V62_LOCK,
    }
    result.update({f"upstream_{name}": path for name, path in v62study.dependencies().items()})
    return result


def audit_phase(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise IntegrityError(f"results root is not empty: {root}")
    missing = [str(path) for path in dependencies().values() if not path.is_file()]
    if missing:
        raise IntegrityError(f"missing dependencies: {missing}")
    if read_json(V62_LOCK)["status"] != "safety_stop_no_candidate_passed":
        raise IntegrityError("V63 requires the locked V62 negative result")
    if len(policy.candidate_grid()) != 810:
        raise IntegrityError("V63 candidate grid changed")
    write_json(root / "v63_audit_lock.json", {
        "dependency_sha256": {name: sha256(path) for name, path in dependencies().items()},
        "paderborn_metadata_opened": False,
        "paderborn_signal_opened": False,
        "b07_opened": False,
    })
    print(f"V63 AUDIT ok: {root}", flush=True)


def require_audit(root: Path) -> dict:
    lock = read_json(root / "v63_audit_lock.json")
    for name, path in dependencies().items():
        if sha256(path) != lock["dependency_sha256"][name]:
            raise IntegrityError(f"dependency changed after audit: {name}")
    return lock


def alarm_for(device: dict, candidate: policy.Candidate | None) -> tuple[np.ndarray, float | None]:
    arrays = device["arrays"]
    if candidate is None:
        return arrays["v57_final"], None
    return policy.candidate_alarm(
        arrays["trend"], arrays["ae"], arrays["raw_watch"], arrays["v57_final"],
        arrays["collapse_route"], arrays["clip_fraction"], candidate,
    )


def counts(devices: list[dict], candidate: policy.Candidate) -> dict:
    result = {}
    for tier in ("historical_run_to_failure", "hit_verified_healthy", "ldm_upb_complete_lifecycle"):
        selected = [d for d in devices if d["tier"] == tier]
        outcomes = []
        for device in selected:
            first = v57.first_alarm(alarm_for(device, candidate)[0])
            outcomes.append(v62study._outcome(first, device))
        result[tier] = {name: outcomes.count(name) for name in sorted(set(outcomes))}
        result[tier]["devices"] = len(selected)
    b09 = next(d for d in devices if d["machine_id"] == "ldm_upb::B09")
    first = v57.first_alarm(alarm_for(b09, candidate)[0])
    result["b09_first"] = first
    result["b09_outcome"] = v62study._outcome(first, b09)
    return result


def gate(summary: dict) -> tuple[bool, dict[str, bool]]:
    g = read_json(PROTOCOL)["development_gates"]
    h, c, l = (summary[name] for name in ("historical_run_to_failure", "hit_verified_healthy", "ldm_upb_complete_lifecycle"))
    checks = {
        "b09_valid": summary["b09_outcome"] == "valid",
        "historical_valid_min": h.get("valid", 0) >= g["historical_valid_min"],
        "historical_premature_max": h.get("premature", 0) <= g["historical_premature_max"],
        "hit_healthy_alarms_max": c.get("false_alarm", 0) <= g["hit_healthy_alarms_max"],
        "ldm_valid_min": l.get("valid", 0) >= g["ldm_valid_min"],
        "ldm_premature_max": l.get("premature", 0) <= g["ldm_premature_max"],
        "ldm_missed_max": l.get("missed", 0) <= g["ldm_missed_max"],
    }
    return all(checks.values()), checks


def selection_key(item) -> tuple:
    candidate, summary, _, _ = item
    h, l = summary["historical_run_to_failure"], summary["ldm_upb_complete_lifecycle"]
    return (
        -l.get("valid", 0), -h.get("valid", 0),
        -candidate.rescue_clip_min, -candidate.threshold_floor,
        -candidate.calibration_margin, -candidate.calibration_quantile,
        -candidate.impulse_hits, -(candidate.ae_hits / candidate.history),
        -candidate.calibration_length, candidate.name,
    )


def develop_phase(root: Path) -> policy.Candidate | None:
    audit = require_audit(root)
    devices = v62study._development_devices()
    evaluated = []
    sweep_rows = []
    for candidate in policy.candidate_grid():
        summary = counts(devices, candidate)
        passed, checks = gate(summary)
        evaluated.append((candidate, summary, passed, checks))
        h, c, l = (summary[name] for name in ("historical_run_to_failure", "hit_verified_healthy", "ldm_upb_complete_lifecycle"))
        sweep_rows.append({
            "candidate": candidate.name, **candidate.__dict__,
            "historical_valid": h.get("valid", 0), "historical_premature": h.get("premature", 0),
            "historical_late": h.get("late", 0), "historical_missed": h.get("missed", 0),
            "hit_false_alarms": c.get("false_alarm", 0),
            "ldm_valid": l.get("valid", 0), "ldm_premature": l.get("premature", 0),
            "ldm_missed": l.get("missed", 0), "b09_first": summary["b09_first"],
            "all_development_gates_pass": passed,
            "failed_checks": ";".join(name for name, ok in checks.items() if not ok),
        })
    write_rows(root / "v63_candidate_sweep.csv", sweep_rows)
    passing = [item for item in evaluated if item[2]]
    selected = min(passing, key=selection_key) if passing else None
    device_rows = []
    for device in devices:
        base_first = v57.first_alarm(alarm_for(device, None)[0])
        candidate_alarm, threshold = (alarm_for(device, selected[0]) if selected else (np.zeros(device["n"], bool), None))
        candidate_first = v57.first_alarm(candidate_alarm)
        arrays = device["arrays"]
        finite_trend = arrays["trend"][np.isfinite(arrays["trend"])]
        device_rows.append({
            "tier": device["tier"], "evidence_status": device["evidence_status"],
            "machine_id": device["machine_id"], "role": device["role"], "n": device["n"],
            "truth_kind": device["truth_kind"], "truth": "" if device["truth"] is None else device["truth"],
            "clip_fraction": arrays["clip_fraction"], "collapse_route": arrays["collapse_route"],
            "adaptive_threshold": "" if threshold is None else threshold,
            "finite_trend_values": len(finite_trend),
            "max_impulse_trend": "" if len(finite_trend) == 0 else float(np.max(finite_trend)),
            "ae_above_floor_count": int(np.sum(arrays["ae"] >= v57.AE_FLOOR)),
            "ae_above_floor_fraction": float(np.mean(arrays["ae"] >= v57.AE_FLOOR)),
            "v57_first": base_first, "v57_outcome": v62study._outcome(base_first, device),
            "v63_first": candidate_first, "v63_outcome": "not_selected" if selected is None else v62study._outcome(candidate_first, device),
            "source_prediction_sha256": sha256(device["prediction_path"]),
        })
    write_rows(root / "v63_full_device_statistics.csv", device_rows)
    payload = {
        "status": "development_passed_policy_frozen" if selected else "safety_stop_no_candidate_passed",
        "candidate_count": len(evaluated), "passing_candidate_count": len(passing),
        "development_devices": len(devices),
        "selected_candidate": None if selected is None else selected[0].__dict__,
        "selected_candidate_name": None if selected is None else selected[0].name,
        "selected_summary": None if selected is None else selected[1],
        "protocol_sha256": sha256(PROTOCOL), "policy_sha256": sha256(HERE / "raw_online_federated_v63_policy.py"),
        "audit_lock_sha256": sha256(root / "v63_audit_lock.json"),
        "paderborn_opened": False, "b07_opened": False,
    }
    write_json(root / "v63_development_lock.json", payload)
    print(f"V63 DEVELOP: candidates={len(evaluated)} passing={len(passing)} status={payload['status']}", flush=True)
    if selected:
        print(f"V63 FROZEN CANDIDATE: {selected[0].name}", flush=True)
        return selected[0]
    print("V63 SAFETY STOP: Paderborn and B07 remain unopened.", flush=True)
    return None


def require_development(root: Path) -> tuple[dict, policy.Candidate]:
    lock = read_json(root / "v63_development_lock.json")
    if lock["status"] != "development_passed_policy_frozen":
        raise IntegrityError("V63 development gate did not pass")
    if sha256(PROTOCOL) != lock["protocol_sha256"] or sha256(HERE / "raw_online_federated_v63_policy.py") != lock["policy_sha256"]:
        raise IntegrityError("V63 policy or protocol changed after freeze")
    return lock, policy.Candidate(**lock["selected_candidate"])


def _paderborn_entries(archive: zipfile.ZipFile, bearing: str) -> list[str]:
    pattern = re.compile(rf"^{bearing}/(N[^/]+_{bearing}_(\d+)\.mat)$", re.I)
    found = []
    for name in archive.namelist():
        match = pattern.fullmatch(name)
        if match:
            prefix = match.group(1).rsplit(f"_{bearing}_", 1)[0]
            found.append((prefix, int(match.group(2)), name))
    found.sort()
    if len(found) != 80:
        raise IntegrityError(f"expected 80 MAT files for {bearing}, found {len(found)}")
    return [item[2] for item in found]


def _paderborn_signal(raw: bytes) -> np.ndarray:
    mat = loadmat(io.BytesIO(raw))
    keys = [key for key in mat if not key.startswith("__")]
    if len(keys) != 1:
        raise IntegrityError("Paderborn MAT has ambiguous root structure")
    struct = mat[keys[0]][0, 0]
    for element in struct["Y"][0]:
        name = str(element["Name"][0]) if len(element["Name"]) else ""
        if name == "vibration_1":
            signal = np.asarray(element["Data"]).reshape(-1).astype(np.float32)
            if len(signal) < 1024 or not np.all(np.isfinite(signal)):
                raise IntegrityError("invalid Paderborn vibration signal")
            return signal
    raise IntegrityError("Paderborn vibration_1 channel not found")


def _paderborn_features(archive: zipfile.ZipFile, bearing: str) -> np.ndarray:
    rows = []
    for index, entry in enumerate(_paderborn_entries(archive, bearing), start=1):
        signal = _paderborn_signal(archive.read(entry))
        midpoint = len(signal) // 2
        for half in (signal[:midpoint], signal[midpoint:]):
            feature = v12.local_preprocess_current_acquisition(half.reshape(1, -1))
            if feature.shape != (100,) or not np.all(np.isfinite(feature)):
                raise IntegrityError(f"invalid Paderborn feature: {bearing} {entry}")
            rows.append(np.asarray(feature, dtype=np.float32))
        if index % 20 == 0:
            print(f"V63 PADERBORN {bearing}: {index}/80 MAT files", flush=True)
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (160, 100):
        raise IntegrityError(f"unexpected Paderborn feature shape: {bearing} {values.shape}")
    return values


def paderborn_phase(root: Path) -> None:
    lock, candidate = require_development(root)
    if not PADERBORN.is_file():
        raise IntegrityError(f"missing Paderborn archive: {PADERBORN}")
    bundle = v45grid.load_bundle("all_source")
    result_rows, ledger_rows = [], []
    prediction_dir = root / "paderborn_predictions"; prediction_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PADERBORN) as archive:
        for bearing in ("K003", "K004", "K005", "K006"):
            values = _paderborn_features(archive, bearing)
            prepared = v52._prepare(values, f"paderborn::{bearing}", bundle)
            ae = v52.machine.fuse_machine([v52.ablation.subset_evidence(prepared, "AE")], "bearing_latch_then_any")
            reference = v52.machine.fuse_machine([v52.ablation.subset_evidence(prepared, "AE_PCA_SVDD")], "bearing_latch_then_any")
            # The frozen machine fusion contract names the continuous maximum
            # across bearing streams simply ``score`` (V58/V61 use the same
            # field).  This adapter has one bearing stream per Paderborn unit.
            ae_score = np.asarray(ae["score"], dtype=np.float64)
            v48_alarm = np.asarray(reference["alarm"], dtype=bool)
            trace = v57.evaluate_machine([values], ae_score, v48_alarm)
            alarm, adaptive = policy.candidate_alarm(
                trace.impulse_trend, ae_score, trace.raw_watch_evidence, trace.final_alarm,
                trace.collapse_route, trace.clip_fraction, candidate,
            )
            first = v57.first_alarm(alarm)
            prediction = prediction_dir / f"paderborn__{bearing}.npz"
            np.savez_compressed(
                prediction, features=values, impulse_trend=trace.impulse_trend,
                ae_continuous_score=ae_score, v48_alarm=v48_alarm,
                v57_raw_watch_evidence=trace.raw_watch_evidence,
                v57_alarm=trace.final_alarm, v63_alarm=alarm,
                clip_fraction=np.asarray(trace.clip_fraction), collapse_route=np.asarray(trace.collapse_route),
                adaptive_threshold=np.asarray(np.nan if adaptive is None else adaptive),
            )
            row = {
                "tier": "paderborn_held_out_healthy_snapshot", "evidence_status": "held_out_once_after_v63_freeze",
                "machine_id": f"paderborn::{bearing}", "role": "healthy_snapshot_stress",
                "n": 160, "truth_kind": "healthy_snapshot", "truth": "healthy",
                "clip_fraction": trace.clip_fraction, "collapse_route": trace.collapse_route,
                "adaptive_threshold": "" if adaptive is None else adaptive,
                "finite_trend_values": int(np.sum(np.isfinite(trace.impulse_trend))),
                "max_impulse_trend": float(np.max(trace.impulse_trend[np.isfinite(trace.impulse_trend)])),
                "ae_above_floor_count": int(np.sum(ae_score >= v57.AE_FLOOR)),
                "ae_above_floor_fraction": float(np.mean(ae_score >= v57.AE_FLOOR)),
                "v57_first": v57.first_alarm(trace.final_alarm),
                "v57_outcome": "false_alarm" if v57.first_alarm(trace.final_alarm) >= 0 else "control_ok",
                "v63_first": first, "v63_outcome": "false_alarm" if first >= 0 else "control_ok",
                "source_prediction_sha256": sha256(prediction),
            }
            ledger_rows.append(row)
            result_rows.append({"bearing": bearing, "n": 160, "v63_first": first, "outcome": row["v63_outcome"], "prediction_sha256": row["source_prediction_sha256"]})
            print(f"V63 PADERBORN {bearing}: first={first} outcome={row['v63_outcome']}", flush=True)
    existing = []
    with (root / "v63_full_device_statistics.csv").open(newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    write_rows(root / "v63_full_device_statistics.csv", existing + ledger_rows)
    false_alarms = sum(row["outcome"] == "false_alarm" for row in result_rows)
    passed = false_alarms == 0
    paderborn_lock = {
        "passed": passed, "kind": "held_out_healthy_snapshot_gate",
        "bearings": 4, "false_alarms": false_alarms, "results": result_rows,
        "candidate_name": candidate.name, "development_lock_sha256": sha256(root / "v63_development_lock.json"),
        "archive_bytes": PADERBORN.stat().st_size,
        "claim_boundary": "short healthy snapshots in deterministic pseudo-sequence; not machine-life specificity",
        "b07_opened": False,
    }
    write_json(root / "v63_paderborn_lock.json", paderborn_lock)
    print(f"V63 PADERBORN GATE: {'PASSED' if passed else 'FAILED'} false_alarms={false_alarms}/4", flush=True)


def report_phase(root: Path) -> None:
    development = read_json(root / "v63_development_lock.json")
    paderborn = read_json(root / "v63_paderborn_lock.json") if (root / "v63_paderborn_lock.json").is_file() else None
    s = development["selected_summary"]
    lines = [
        "# V63 adaptive confidence-gated rescue", "",
        f"**DEVELOPMENT GATE {'PASSED' if development['status'] == 'development_passed_policy_frozen' else 'FAILED'}**: "
        f"{development['passing_candidate_count']}/{development['candidate_count']} candidates passed.", "",
        f"Frozen candidate: `{development['selected_candidate_name']}`.", "",
        "## Burned development results", "",
        "| tier | valid | premature | late | missed | blind | false alarms |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| historical 50 | {s['historical_run_to_failure'].get('valid',0)} | {s['historical_run_to_failure'].get('premature',0)} | {s['historical_run_to_failure'].get('late',0)} | {s['historical_run_to_failure'].get('missed',0)} | {s['historical_run_to_failure'].get('blind',0)} | n/a |",
        f"| HIT healthy 2 | n/a | n/a | n/a | n/a | n/a | {s['hit_verified_healthy'].get('false_alarm',0)} |",
        f"| LDM-UPB 10 | {s['ldm_upb_complete_lifecycle'].get('valid',0)} | {s['ldm_upb_complete_lifecycle'].get('premature',0)} | n/a | {s['ldm_upb_complete_lifecycle'].get('missed',0)} | 0 | n/a |", "",
        f"B09: valid at acquisition {s['b09_first']}.", "",
    ]
    if paderborn:
        lines += [
            "## Held-out Paderborn healthy-snapshot gate", "",
            f"**{'PASSED' if paderborn['passed'] else 'FAILED'}**: {paderborn['false_alarms']}/4 bearings alarmed.", "",
            "These are short snapshot stress tests, not healthy machine-life experiments.", "",
        ]
    lines += [
        "## Claim boundary", "",
        "V63 is developed on the now-burned 62-device cohort. Passing that gate is not validation. The Paderborn tier is held-out but consists of synthetic ordering of short healthy snapshots. B07 remains sealed and is required for a new positive-lifecycle result.", "",
    ]
    (root / "V63_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"V63 REPORT: {root / 'V63_RESULTS.md'}", flush=True)


def all_phase(root: Path) -> None:
    audit_phase(root)
    selected = develop_phase(root)
    if selected is None:
        report_phase(root); return
    paderborn_phase(root)
    report_phase(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("audit", "develop", "paderborn", "report", "all"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); root = root_for(args.run_id)
    try:
        {"audit": audit_phase, "develop": develop_phase, "paderborn": paderborn_phase, "report": report_phase, "all": all_phase}[args.phase](root)
    except Exception as exc:
        print(f"V63 {args.phase.upper()} FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
