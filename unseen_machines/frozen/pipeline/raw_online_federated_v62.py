"""V62 fail-closed accumulated-corroboration development study.

The default ``all`` phase audits locked inputs, evaluates the predeclared
candidate grid on burned data, writes full per-device statistics, and stops if
no candidate passes every gate.  Held-out Paderborn controls and B07 are never
opened after a failed development gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import beta

import raw_online_federated_v57_policy as v57
import raw_online_federated_v60_analysis as timing
import raw_online_federated_v62_policy as policy


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v62_protocol.json"
V57_ROOT = RESULTS / "raw_online_federated_v57_frozen1"
V58_ROOT = RESULTS / "raw_online_federated_v58_frozen1"
V60_ROOT = RESULTS / "raw_online_federated_v60_frozen2"
V61_ROOT = RESULTS / "raw_online_federated_v61_frozen1"
V52R1_ROOT = RESULTS / "raw_online_federated_v52r1_corrected1"


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def root_for(run_id: str) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
        raise ValueError("run-id must contain only letters, digits, underscore, or hyphen")
    return RESULTS / f"raw_online_federated_v62_{run_id}"


def dependencies() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL,
        "policy": HERE / "raw_online_federated_v62_policy.py",
        "v57_policy": HERE / "raw_online_federated_v57_policy.py",
        "v57_gate": V57_ROOT / "v57_gate.json",
        "v57_ledger": V57_ROOT / "v57_prediction_ledger.csv",
        "v57_machine_results": V57_ROOT / "v57_historical_machine_results.csv",
        "v58_gate": V58_ROOT / "v58_gate.json",
        "v58_results": V58_ROOT / "v58_machine_results.csv",
        "v60_gate": V60_ROOT / "v60_gate.json",
        "v60_eligibility": V60_ROOT / "v60_phase1_eligibility.csv",
        "v61_gate": V61_ROOT / "v61_gate.json",
        "v61_results": V61_ROOT / "v61_machine_results.csv",
        "v52r1_results": V52R1_ROOT / "v52r1_machine_results.csv",
    }


def audit_phase(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "v62_audit_lock.json"
    if lock.exists():
        raise IntegrityError(f"audit already exists: {lock}")
    deps = dependencies()
    missing = [str(path) for path in deps.values() if not path.is_file()]
    if missing:
        raise IntegrityError(f"missing dependencies: {missing}")
    for name in ("v57_gate", "v58_gate", "v60_gate", "v61_gate"):
        gate = read_json(deps[name])
        if not bool(gate.get("passed") or gate.get("development_gate_passed") or gate.get("all_checks_passed")):
            raise IntegrityError(f"upstream gate is not passed: {name}")
    spec = read_json(PROTOCOL)
    if spec["development_devices"]["total_reported_in_full_ledger"] != 62:
        raise IntegrityError("protocol development denominator changed")
    write_json(lock, {
        "kind": "v62_burned_development_input_lock",
        "dependency_sha256": {name: sha256(path) for name, path in deps.items()},
        "candidate_count": len(policy.candidate_grid()),
        "paderborn_opened": False,
        "b07_opened": False,
    })
    print(f"V62 AUDIT ok: {root}", flush=True)


def _require_audit(root: Path) -> dict:
    path = root / "v62_audit_lock.json"
    if not path.is_file():
        raise IntegrityError("run --phase audit first")
    lock = read_json(path)
    for name, dependency in dependencies().items():
        if sha256(dependency) != lock["dependency_sha256"][name]:
            raise IntegrityError(f"dependency changed after audit: {name}")
    return lock


def _prediction_path(machine_id: str, panel_by_machine: dict[str, str]) -> Path:
    bearing = machine_id.split("::")[-1]
    if bearing in {"B08", "B10", "B11", "B17"}:
        return V58_ROOT / "machine_predictions" / f"ldm_upb__{bearing}.npz"
    if bearing in {"B09", "B16"}:
        return V61_ROOT / "machine_predictions" / f"ldm_upb__{bearing}.npz"
    panel = panel_by_machine[machine_id]
    return V57_ROOT / "machine_predictions" / f"{panel}__{machine_id.replace('::', '__')}.npz"


def _load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        final_key = "final_alarm" if "final_alarm" in archive else "v57_alarm"
        return {
            "trend": np.asarray(archive["impulse_trend"], dtype=np.float64),
            "ae": np.asarray(archive["ae_continuous_score"], dtype=np.float64),
            "raw_watch": np.asarray(archive["raw_watch_evidence"], dtype=bool),
            "v57_final": np.asarray(archive[final_key], dtype=bool),
            "clip_fraction": float(archive["clip_fraction"]),
            "collapse_route": bool(archive["collapse_route"]),
        }


def _development_devices() -> list[dict[str, Any]]:
    ledger = read_rows(V57_ROOT / "v57_prediction_ledger.csv")
    panel_by_machine = {row["machine_id"]: row["panel"] for row in ledger}
    devices: list[dict[str, Any]] = []
    for row in read_rows(V60_ROOT / "v60_phase1_eligibility.csv"):
        devices.append({
            "tier": "historical_run_to_failure",
            "evidence_status": "burned_development",
            "machine_id": row["machine_id"],
            "role": "run_to_failure",
            "n": int(row["n"]),
            "truth_kind": "onset",
            "truth": None if not row["onset"] else float(row["onset"]),
            "structurally_blind": row["structurally_blind"].lower() == "true",
        })
    for machine_id in ("hit::data1", "hit::data2"):
        source = next(row for row in ledger if row["machine_id"] == machine_id)
        devices.append({
            "tier": "hit_verified_healthy",
            "evidence_status": "burned_development_control",
            "machine_id": machine_id,
            "role": "verified_healthy",
            "n": int(source["n"]),
            "truth_kind": "healthy_control",
            "truth": None,
            "structurally_blind": False,
        })
    transition_rows: list[dict[str, str]] = []
    transition_rows.extend(read_rows(V52R1_ROOT / "v52r1_machine_results.csv"))
    transition_rows.extend(read_rows(V58_ROOT / "v58_machine_results.csv"))
    transition_rows.extend(read_rows(V61_ROOT / "v61_machine_results.csv"))
    for row in transition_rows:
        machine_id = row.get("machine_id") or f"ldm_upb::{row['machine']}"
        truth = row.get("official_transition") or row.get("published_transition")
        devices.append({
            "tier": "ldm_upb_complete_lifecycle",
            "evidence_status": "burned_development_for_v62",
            "machine_id": machine_id,
            "role": "run_to_failure",
            "n": int(row["n"]),
            "truth_kind": "published_transition",
            "truth": float(truth),
            "structurally_blind": False,
        })
    if len(devices) != 62 or len({row["machine_id"] for row in devices}) != 62:
        raise IntegrityError(f"V62 expected 62 unique development devices, found {len(devices)}")
    for device in devices:
        device["prediction_path"] = _prediction_path(device["machine_id"], panel_by_machine)
        if not device["prediction_path"].is_file():
            raise IntegrityError(f"missing prediction: {device['prediction_path']}")
        device["arrays"] = _load_prediction(device["prediction_path"])
    return devices


def _outcome(first: int, device: dict[str, Any]) -> str:
    if device["truth_kind"] == "healthy_control":
        return timing.classify_healthy_control(first)
    if device["truth_kind"] == "published_transition":
        return timing.classify_transition(first, device["truth"], device["n"])
    outcome = timing.classify_onset(first, device["truth"], device["n"])
    # Match V60's correction exactly: only a genuine miss is recoded as
    # structural blindness.  A machine without an onset remains unavailable.
    return "blind" if device["structurally_blind"] and outcome == "missed" else outcome


def _alarm_for(device: dict[str, Any], candidate: policy.Candidate | None) -> np.ndarray:
    arrays = device["arrays"]
    if candidate is None:
        return arrays["v57_final"]
    return policy.candidate_alarm(
        arrays["trend"], arrays["ae"], arrays["raw_watch"], arrays["v57_final"],
        arrays["collapse_route"], candidate,
    )


def _counts(devices: list[dict[str, Any]], candidate: policy.Candidate | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tier in ("historical_run_to_failure", "hit_verified_healthy", "ldm_upb_complete_lifecycle"):
        selected = [device for device in devices if device["tier"] == tier]
        outcomes = [_outcome(v57.first_alarm(_alarm_for(device, candidate)), device) for device in selected]
        result[tier] = {name: outcomes.count(name) for name in sorted(set(outcomes))}
        result[tier]["devices"] = len(selected)
    b09 = next(device for device in devices if device["machine_id"] == "ldm_upb::B09")
    b09_first = v57.first_alarm(_alarm_for(b09, candidate))
    result["b09_first"] = b09_first
    result["b09_outcome"] = _outcome(b09_first, b09)
    return result


def _gate(summary: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    spec = read_json(PROTOCOL)["selection_gates"]
    historical = summary["historical_run_to_failure"]
    healthy = summary["hit_verified_healthy"]
    ldm = summary["ldm_upb_complete_lifecycle"]
    checks = {
        "b09_valid": summary["b09_outcome"] == "valid",
        "historical_valid_min": historical.get("valid", 0) >= int(spec["historical_valid_min"]),
        "historical_premature_max": historical.get("premature", 0) <= int(spec["historical_premature_max"]),
        "hit_healthy_alarms_max": healthy.get("false_alarm", 0) <= int(spec["hit_healthy_alarms_max"]),
        "ldm_valid_min": ldm.get("valid", 0) >= int(spec["ldm_valid_min"]),
        "ldm_premature_max": ldm.get("premature", 0) <= int(spec["ldm_premature_max"]),
    }
    return all(checks.values()), checks


def _rank(summary: dict[str, Any]) -> tuple:
    h = summary["historical_run_to_failure"]
    c = summary["hit_verified_healthy"]
    l = summary["ldm_upb_complete_lifecycle"]
    return (
        c.get("false_alarm", 0),
        max(0, h.get("premature", 0) - 1),
        max(0, l.get("premature", 0) - 1),
        -l.get("valid", 0),
        -h.get("valid", 0),
        h.get("missed", 0) + h.get("blind", 0),
    )


def _interval(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "proportion": float("nan"), "low": float("nan"), "high": float("nan")}
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return {"successes": successes, "total": total, "proportion": successes / total, "low": low, "high": high}


def develop_phase(root: Path) -> None:
    audit = _require_audit(root)
    if (root / "v62_development_lock.json").exists():
        raise IntegrityError("development phase already completed")
    devices = _development_devices()
    baseline = _counts(devices, None)
    summaries: list[tuple[policy.Candidate, dict[str, Any], bool, dict[str, bool]]] = []
    candidate_rows: list[dict[str, Any]] = []
    for candidate in policy.candidate_grid():
        summary = _counts(devices, candidate)
        passed, checks = _gate(summary)
        summaries.append((candidate, summary, passed, checks))
        h, c, l = (summary[name] for name in ("historical_run_to_failure", "hit_verified_healthy", "ldm_upb_complete_lifecycle"))
        candidate_rows.append({
            "candidate": candidate.name,
            "trend_threshold": candidate.trend_threshold,
            "history": candidate.history,
            "impulse_hits": candidate.impulse_hits,
            "ae_hits": candidate.ae_hits,
            "historical_valid": h.get("valid", 0),
            "historical_premature": h.get("premature", 0),
            "historical_late": h.get("late", 0),
            "historical_missed": h.get("missed", 0),
            "historical_blind": h.get("blind", 0),
            "hit_control_alarms": c.get("false_alarm", 0),
            "ldm_valid": l.get("valid", 0),
            "ldm_premature": l.get("premature", 0),
            "ldm_missed": l.get("missed", 0),
            "b09_first": summary["b09_first"],
            "b09_outcome": summary["b09_outcome"],
            "all_selection_gates_pass": passed,
            "failed_checks": ";".join(name for name, ok in checks.items() if not ok),
        })
    write_rows(root / "v62_candidate_sweep.csv", candidate_rows)
    passing = [item for item in summaries if item[2]]
    selected = min(passing, key=lambda item: (_rank(item[1]), item[0].name)) if passing else None
    b09_recovering = [item for item in summaries if item[1]["b09_outcome"] == "valid"]
    diagnostic = min(b09_recovering, key=lambda item: (_rank(item[1]), item[0].name)) if b09_recovering else None
    reported = selected or diagnostic
    device_rows: list[dict[str, Any]] = []
    for device in devices:
        arrays = device["arrays"]
        base_first = v57.first_alarm(arrays["v57_final"])
        candidate_first = -1 if reported is None else v57.first_alarm(_alarm_for(device, reported[0]))
        finite = arrays["trend"][np.isfinite(arrays["trend"])]
        ae_active = np.isfinite(arrays["ae"]) & (arrays["ae"] >= v57.AE_FLOOR)
        device_rows.append({
            "tier": device["tier"],
            "evidence_status": device["evidence_status"],
            "machine_id": device["machine_id"],
            "role": device["role"],
            "n": device["n"],
            "truth_kind": device["truth_kind"],
            "truth": "" if device["truth"] is None else device["truth"],
            "structurally_blind": device["structurally_blind"],
            "clip_fraction": arrays["clip_fraction"],
            "collapse_route": arrays["collapse_route"],
            "finite_trend_values": len(finite),
            "max_impulse_trend": "" if len(finite) == 0 else float(np.max(finite)),
            "ae_above_floor_count": int(np.sum(ae_active)),
            "ae_above_floor_fraction": float(np.mean(ae_active)),
            "v57_first": base_first,
            "v57_outcome": _outcome(base_first, device),
            "reported_candidate": "" if reported is None else reported[0].name,
            "candidate_is_selected": selected is not None,
            "candidate_first": candidate_first,
            "candidate_outcome": "not_available" if reported is None else _outcome(candidate_first, device),
            "prediction_sha256": sha256(device["prediction_path"]),
        })
    write_rows(root / "v62_full_device_statistics.csv", device_rows)
    baseline_h = baseline["historical_run_to_failure"]
    baseline_c = baseline["hit_verified_healthy"]
    baseline_l = baseline["ldm_upb_complete_lifecycle"]
    result = {
        "status": "candidate_frozen" if selected else "safety_stop_no_candidate_passed",
        "candidate_count": len(summaries),
        "passing_candidate_count": len(passing),
        "development_device_count": len(devices),
        "baseline_v57": baseline,
        "selected_candidate": None if selected is None else selected[0].__dict__,
        "selected_summary": None if selected is None else selected[1],
        "best_b09_recovering_diagnostic": None if diagnostic is None else {
            "candidate": diagnostic[0].__dict__, "summary": diagnostic[1], "checks": diagnostic[3],
        },
        "intervals": {
            "v57_historical_valid": _interval(baseline_h.get("valid", 0), 42),
            "v57_hit_alarm_free": _interval(2 - baseline_c.get("false_alarm", 0), 2),
            "v57_ldm_valid": _interval(baseline_l.get("valid", 0), 10),
        },
        "paderborn_opened": False,
        "b07_opened": False,
        "audit_lock_sha256": sha256(root / "v62_audit_lock.json"),
    }
    write_json(root / "v62_development_lock.json", result)
    lines = [
        "# V62 accumulated-corroboration development result", "",
        f"**{'CANDIDATE FROZEN' if selected else 'SAFETY STOP: NO CANDIDATE PASSED'}.**", "",
        "This is a burned-data development study, not independent validation. The full CSV reports every one of the 62 devices used for policy selection in separate evidence tiers.", "",
        "## V57 baseline on the V62 development universe", "",
        "| tier | devices | valid | premature | late | missed | blind | control alarms |", "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| historical run-to-failure | 50 | {baseline_h.get('valid',0)} | {baseline_h.get('premature',0)} | {baseline_h.get('late',0)} | {baseline_h.get('missed',0)} | {baseline_h.get('blind',0)} | n/a |",
        f"| HIT verified healthy | 2 | n/a | n/a | n/a | n/a | n/a | {baseline_c.get('false_alarm',0)} |",
        f"| LDM-UPB complete lifecycles | 10 | {baseline_l.get('valid',0)} | {baseline_l.get('premature',0)} | n/a | {baseline_l.get('missed',0)} | 0 | n/a |", "",
        f"Candidate configurations tested: **{len(summaries)}**; all-gate passes: **{len(passing)}**.", "",
    ]
    if selected:
        lines += [f"Selected and frozen: `{selected[0].name}`.", "", "The held-out Paderborn phase is now eligible, but remains a healthy-snapshot stress test only."]
    else:
        lines += [
            "No B09-recovering accumulated-evidence rule preserved all predeclared historical, healthy-control, and LDM timing gates.", "",
            "Per protocol, K003-K006 were not opened and B07 was not acquired or opened. V57 remains the deployed thesis policy; V62 is a reportable negative development result.",
        ]
        if diagnostic:
            d = diagnostic[1]
            lines += ["", "## Least-violating B09-recovering diagnostic (not selected)", "", f"`{diagnostic[0].name}`: {json.dumps(d, sort_keys=True)}"]
    lines += ["", "Artifacts: `v62_candidate_sweep.csv`, `v62_full_device_statistics.csv`, and `v62_development_lock.json`.", ""]
    (root / "V62_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"V62 DEVELOP complete: status={result['status']} devices=62 candidates={len(summaries)}", flush=True)
    if not selected:
        print("V62 SAFETY STOP: K003-K006 and B07 remain unopened.", flush=True)


def all_phase(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise IntegrityError(f"results root is not empty: {root}")
    audit_phase(root)
    develop_phase(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("audit", "develop", "all"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = root_for(args.run_id)
    try:
        {"audit": audit_phase, "develop": develop_phase, "all": all_phase}[args.phase](root)
    except Exception as exc:
        print(f"V62 {args.phase.upper()} FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
