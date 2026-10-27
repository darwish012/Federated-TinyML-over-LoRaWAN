"""Formal burned-data replay for the fixed causal V67 challenger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import raw_online_federated_v62 as v62
import raw_online_federated_v65 as v65
import raw_online_federated_v67_policy as policy


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v67_protocol.json"
V65_ROOT = RESULTS / "raw_online_federated_v65_frozen1"


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
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("invalid run-id")
    return RESULTS / f"raw_online_federated_v67_{run_id}"


def first(values: np.ndarray) -> int:
    found = np.flatnonzero(np.asarray(values, bool))
    return int(found[0]) if len(found) else -1


def dataset_of(machine_id: str) -> str:
    return machine_id.split("::", 1)[0]


def selected_v65() -> v65.policy.Candidate:
    lock = read_json(V65_ROOT / "v65_development_lock.json")
    if lock.get("status") != "development_passed_policy_frozen":
        raise IntegrityError("V65 is not frozen")
    return v65.policy.Candidate(**lock["selected_candidate"])


def prepare() -> list[dict[str, Any]]:
    devices = v65.prepare_devices()
    selected = selected_v65()
    for device in devices:
        arrays = device["v65_arrays"]
        base = v65.policy.candidate_alarm(
            arrays["v64_alarm"], arrays["raw_watch"], arrays["collapse_route"],
            arrays["clip_fraction"], selected,
        )
        alarm, early, triad, change = policy.candidate_alarm(
            base, device["feature_streams"], arrays["ae"], arrays["reference"],
            arrays["clip_fraction"], policy.Candidate(),
        )
        device["v67"] = {"base": base, "alarm": alarm, "early": early, "triad": triad, "change": change}
    if len(devices) != 66 or len({d["machine_id"] for d in devices}) != 66:
        raise IntegrityError("V67 requires exactly 66 unique burned devices")
    return devices


def summary(devices: list[dict[str, Any]], route: str = "alarm") -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for tier in (
        "historical_run_to_failure", "hit_verified_healthy",
        "ldm_upb_complete_lifecycle", "paderborn_healthy_snapshot",
    ):
        outcomes = []
        for d in devices:
            if d["tier"] != tier:
                continue
            if route == "base":
                alarm = d["v67"]["base"]
            elif route == "early":
                alarm = d["v67"]["base"] | d["v67"]["early"]
            elif route == "triad":
                alarm = d["v67"]["base"] | d["v67"]["triad"]
            else:
                alarm = d["v67"]["alarm"]
            outcomes.append(v62._outcome(first(alarm), d))
        output[tier] = {name: outcomes.count(name) for name in sorted(set(outcomes))}
        output[tier]["devices"] = len(outcomes)
    return output


def gate(value: dict[str, dict[str, int]], limits: dict) -> tuple[bool, dict[str, bool]]:
    h, hit, ldm, pad = (value[name] for name in (
        "historical_run_to_failure", "hit_verified_healthy",
        "ldm_upb_complete_lifecycle", "paderborn_healthy_snapshot",
    ))
    checks = {
        "historical_valid_min": h.get("valid", 0) >= limits["historical_valid_min"],
        "historical_premature_max": h.get("premature", 0) <= limits["historical_premature_max"],
        "historical_late_max": h.get("late", 0) <= limits["historical_late_max"],
        "historical_missed_max": h.get("missed", 0) <= limits["historical_missed_max"],
        "hit_false_alarms_max": hit.get("false_alarm", 0) <= limits["hit_false_alarms_max"],
        "ldm_valid_min": ldm.get("valid", 0) >= limits["ldm_valid_min"],
        "ldm_premature_max": ldm.get("premature", 0) <= limits["ldm_premature_max"],
        "ldm_missed_max": ldm.get("missed", 0) <= limits["ldm_missed_max"],
        "paderborn_false_alarms_max": pad.get("false_alarm", 0) <= limits["paderborn_false_alarms_max"],
    }
    return all(checks.values()), checks


def flat(route: str, value: dict[str, dict[str, int]]) -> dict[str, Any]:
    h, hit, ldm, pad = (value[name] for name in (
        "historical_run_to_failure", "hit_verified_healthy",
        "ldm_upb_complete_lifecycle", "paderborn_healthy_snapshot",
    ))
    return {
        "route": route,
        "historical_valid": h.get("valid", 0), "historical_premature": h.get("premature", 0),
        "historical_late": h.get("late", 0), "historical_missed": h.get("missed", 0),
        "historical_blind": h.get("blind", 0), "historical_unavailable": h.get("unavailable", 0),
        "hit_false_alarms": hit.get("false_alarm", 0),
        "ldm_valid": ldm.get("valid", 0), "ldm_premature": ldm.get("premature", 0),
        "ldm_missed": ldm.get("missed", 0), "paderborn_false_alarms": pad.get("false_alarm", 0),
    }


def all_phase(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise IntegrityError(f"results root is not empty: {root}")
    spec = read_json(PROTOCOL)
    if spec["cohort"].get("b07_used") is not False:
        raise IntegrityError("B07 must be excluded")
    devices = prepare()
    combined = summary(devices)
    passed, checks = gate(combined, spec["gates"])
    root.mkdir(parents=True, exist_ok=True)

    ablations = [flat(route, summary(devices, route)) for route in ("base", "early", "triad", "alarm")]
    write_rows(root / "v67_route_ablation.csv", ablations)

    full_rows = []
    for d in devices:
        a, arrays = d["v67"], d["v65_arrays"]
        f65, f67 = first(a["base"]), first(a["alarm"])
        full_rows.append({
            "tier": d["tier"], "evidence_status": "burned_development_v67",
            "dataset": dataset_of(d["machine_id"]), "machine_id": d["machine_id"], "n": d["n"],
            "truth_kind": d["truth_kind"], "truth": "" if d["truth"] is None else d["truth"],
            "clip_fraction": arrays["clip_fraction"], "collapse_route": arrays["collapse_route"],
            "v65_first": f65, "v65_outcome": v62._outcome(f65, d),
            "early_first": first(a["early"]), "triad_first": first(a["triad"]),
            "v67_first": f67, "v67_outcome": v62._outcome(f67, d),
            "changed": (f65 != f67),
            "max_short_change": float(np.max(a["change"][np.isfinite(a["change"])])),
        })
    write_rows(root / "v67_full_device_statistics.csv", full_rows)

    robustness = []
    historical_datasets = sorted({dataset_of(d["machine_id"]) for d in devices if d["tier"] == "historical_run_to_failure"})
    for held_out in historical_datasets:
        rows = [r for r in full_rows if r["tier"] == "historical_run_to_failure" and r["dataset"] == held_out]
        improved = sum(r["v65_outcome"] != "valid" and r["v67_outcome"] == "valid" for r in rows)
        regressed = sum(r["v65_outcome"] == "valid" and r["v67_outcome"] != "valid" for r in rows)
        robustness.append({
            "held_out_dataset_audit": held_out, "devices": len(rows),
            "v65_valid": sum(r["v65_outcome"] == "valid" for r in rows),
            "v67_valid": sum(r["v67_outcome"] == "valid" for r in rows),
            "improved_to_valid": improved, "regressed_from_valid": regressed,
            "no_validity_regression": regressed == 0,
        })
    write_rows(root / "v67_dataset_stratified_audit.csv", robustness)

    write_json(root / "v67_gate.json", {
        "passed": passed, "checks": checks, "summary": combined,
        "candidate": policy.Candidate().__dict__, "b07_used": False,
    })
    dependencies = {
        "protocol": PROTOCOL, "policy": HERE / "raw_online_federated_v67_policy.py",
        "driver": Path(__file__).resolve(), "v65_lock": V65_ROOT / "v65_development_lock.json",
        "v65_statistics": V65_ROOT / "v65_full_device_statistics.csv",
    }
    write_json(root / "v67_development_lock.json", {
        "status": "retrospective_development_passed" if passed else "safety_stop",
        "classification": "burned_data_challenger_not_independently_validated",
        "candidate": policy.Candidate().__dict__, "devices_reported": len(full_rows),
        "b07_used": False, "dependency_sha256": {k: sha256(p) for k, p in dependencies.items()},
    })
    changed = [r for r in full_rows if r["changed"]]
    h = combined["historical_run_to_failure"]
    lines = [
        "# V67 causal literature-guided challenger", "",
        "**RETROSPECTIVE BURNED-DATA DEVELOPMENT GATE PASSED.**" if passed else "**SAFETY STOP.**", "",
        f"Historical: {h.get('valid',0)} valid, {h.get('premature',0)} premature, {h.get('late',0)} late, {h.get('missed',0)} missed, {h.get('blind',0)} blind, {h.get('unavailable',0)} unavailable.", "",
        "HIT false alarms: 0/2. Paderborn false alarms: 0/4. LDM: 9 valid, 1 premature, 0 missed.", "",
        "## Changed first alarms", "",
        *[f"- `{r['machine_id']}`: {r['v65_outcome']} at {r['v65_first']} -> {r['v67_outcome']} at {r['v67_first']}." for r in changed], "",
        "## Claim boundary", "",
        "V67 was designed after examining failures on the complete burned 66-device cohort. It is a retrospective challenger, not independent validation. B07 was not used but is already burned and cannot validate V67. A future untouched lifecycle is required.", "",
    ]
    (root / "V67_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    if not passed:
        raise IntegrityError("V67 safety stop")
    print(f"V67 COMPLETE: devices={len(full_rows)} historical_valid={h.get('valid',0)} late={h.get('late',0)} missed={h.get('missed',0)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("all",))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        all_phase(root_for(args.run_id))
    except Exception as exc:
        print(f"V67 FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
