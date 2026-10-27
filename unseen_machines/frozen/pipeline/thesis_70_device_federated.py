"""One frozen 70-logical-client healthy-only FedAdam trial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import ldm_reserved_cohort_external as ldm
import raw_online_federated_v67 as v67
import raw_online_federated_v92 as v92
import thesis_participating_node_scaling as common


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "thesis_70_device_federated_protocol.json"
RESULTS = HERE / "results"
LDM_ROOT = RESULTS / "ldm_reserved_cohort_external_frozen1"
TRANSITIONS = HERE / "ldm_reserved_cohort_transitions_after_lock.json"
ADDED = ("B06", "B13", "B14", "B15")


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    if rows:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.write_text("", encoding="utf-8")
    temporary.replace(path)


def root_for(run_id: str) -> Path:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not run_id or any(char not in allowed for char in run_id):
        raise IntegrityError("invalid run id")
    return RESULTS / f"thesis_70_device_federated_{run_id}"


@dataclass(frozen=True)
class MachinePartition:
    machine_id: str
    dataset: str
    truth_kind: str
    truth: int | None
    train: np.ndarray
    calibration: np.ndarray
    healthy_test: np.ndarray
    unhealthy_test: np.ndarray
    unscored_test: np.ndarray
    train_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    healthy_test_indices: tuple[int, ...]
    unhealthy_test_indices: tuple[int, ...]
    unscored_test_indices: tuple[int, ...]


def _added_ldm_device(machine: str, transitions: dict[str, int]) -> dict[str, Any]:
    feature_path = LDM_ROOT / "features" / f"ldm_upb__{machine}.npz"
    prediction_path = LDM_ROOT / "predictions" / f"ldm_upb__{machine}.npz"
    if not feature_path.is_file() or not prediction_path.is_file():
        raise IntegrityError(f"missing locked LDM artifact for {machine}")
    with np.load(feature_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float64)
    with np.load(prediction_path, allow_pickle=False) as archive:
        ae = np.asarray(archive["ae_continuous_score"], dtype=np.float64)
        reference = np.asarray(archive["v48_alarm"], dtype=bool)
        trend = np.asarray(archive["impulse_trend"], dtype=np.float64)
        watch = np.asarray(archive["raw_watch_evidence"], dtype=bool)
        clip = float(archive["clip_fraction"])
    return {
        "machine_id": f"ldm_upb::{machine}",
        "tier": "ldm_upb_complete_lifecycle",
        "truth_kind": "published_transition",
        "truth": int(transitions[machine]),
        "n": len(features),
        "feature_streams": [features],
        "v65_arrays": {
            "ae": ae,
            "trend": trend,
            "reference": reference,
            "raw_watch": watch,
            "clip_fraction": clip,
        },
    }


def load_70(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], tuple[str, ...], dict[str, str]]:
    source = v67.prepare()
    if len(source) != int(protocol["cohort"]["v67_devices"]):
        raise IntegrityError("V67 device count drift")
    transition_payload = json.loads(TRANSITIONS.read_text(encoding="utf-8"))
    transitions = {key: int(value) for key, value in transition_payload["transitions"].items()}
    added = [_added_ldm_device(machine, transitions) for machine in ADDED]
    devices = source + added
    if len(devices) != int(protocol["cohort"]["total_logical_clients"]):
        raise IntegrityError("70-client cohort barrier failed")
    ids = [str(device["machine_id"]) for device in devices]
    if len(set(ids)) != len(ids) or any(identifier.endswith("::B07") for identifier in ids):
        raise IntegrityError("duplicate device or incompatible B07 in cohort")
    matrices: dict[str, np.ndarray] = {}
    feature_names: tuple[str, ...] | None = None
    for device in devices:
        matrix, names = v92.causal_feature_matrix(device)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise IntegrityError(f"feature schema drift for {device['machine_id']}")
        matrices[str(device["machine_id"])] = np.asarray(matrix, dtype=np.float64)
    if len(feature_names or ()) != int(protocol["cohort"]["expected_feature_schema_width"]):
        raise IntegrityError("feature width drift")
    artifact_hashes = {"transitions": sha256(TRANSITIONS)}
    for machine in ADDED:
        artifact_hashes[f"features_{machine}"] = sha256(LDM_ROOT / "features" / f"ldm_upb__{machine}.npz")
        artifact_hashes[f"prediction_{machine}"] = sha256(LDM_ROOT / "predictions" / f"ldm_upb__{machine}.npz")
    return devices, matrices, feature_names or (), artifact_hashes


def _three_way(indices: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    if len(values) < 3 * minimum:
        raise IntegrityError(f"only {len(values)} healthy commissioning rows")
    train_end = max(minimum, int(math.floor(0.60 * len(values))))
    calibration_end = max(train_end + minimum, int(math.floor(0.80 * len(values))))
    calibration_end = min(calibration_end, len(values) - minimum)
    return values[:train_end], values[train_end:calibration_end], values[calibration_end:]


def prepare_partitions(
    devices: list[dict[str, Any]], matrices: dict[str, np.ndarray], protocol: dict[str, Any],
) -> tuple[dict[str, MachinePartition], list[dict[str, Any]]]:
    minimum = int(protocol["partition"]["minimum_each_healthy_partition"])
    partitions: dict[str, MachinePartition] = {}
    audit: list[dict[str, Any]] = []
    for device in devices:
        machine_id = str(device["machine_id"])
        matrix = matrices[machine_id]
        n = len(matrix)
        truth = None if device.get("truth") is None else int(device["truth"])
        truth_kind = str(device.get("truth_kind", "unknown"))
        if truth is not None:
            healthy = np.arange(0, max(0, min(truth, n)), dtype=np.int64)
            unhealthy = np.arange(max(0, min(truth, n)), n, dtype=np.int64)
            unscored = np.empty(0, dtype=np.int64)
        elif truth_kind == "healthy_control":
            healthy = np.arange(n, dtype=np.int64)
            unhealthy = np.empty(0, dtype=np.int64)
            unscored = np.empty(0, dtype=np.int64)
        else:
            commissioning_end = max(3 * minimum, int(math.floor(0.50 * n)))
            commissioning_end = min(commissioning_end, n - 1)
            healthy = np.arange(commissioning_end, dtype=np.int64)
            unhealthy = np.empty(0, dtype=np.int64)
            unscored = np.arange(commissioning_end, n, dtype=np.int64)
        train_idx, calibration_idx, healthy_test_idx = _three_way(healthy, minimum)
        calibration_raw = matrix[calibration_idx]
        center = np.median(calibration_raw, axis=0)
        mad = 1.4826 * np.median(np.abs(calibration_raw - center), axis=0)
        std = np.std(calibration_raw, axis=0)
        scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, np.where(np.isfinite(std) & (std > 1e-8), std, 1.0))
        normalize = lambda index: np.asarray((matrix[index] - center) / scale, dtype=np.float64)
        groups = [set(train_idx), set(calibration_idx), set(healthy_test_idx), set(unhealthy), set(unscored)]
        if any(groups[left] & groups[right] for left in range(5) for right in range(left + 1, 5)):
            raise IntegrityError(f"partition overlap for {machine_id}")
        partition = MachinePartition(
            machine_id=machine_id,
            dataset=machine_id.split("::", 1)[0],
            truth_kind=truth_kind,
            truth=truth,
            train=normalize(train_idx),
            calibration=normalize(calibration_idx),
            healthy_test=normalize(healthy_test_idx),
            unhealthy_test=normalize(unhealthy),
            unscored_test=normalize(unscored),
            train_indices=tuple(map(int, train_idx)),
            calibration_indices=tuple(map(int, calibration_idx)),
            healthy_test_indices=tuple(map(int, healthy_test_idx)),
            unhealthy_test_indices=tuple(map(int, unhealthy)),
            unscored_test_indices=tuple(map(int, unscored)),
        )
        partitions[machine_id] = partition
        audit.append({
            "machine_id": machine_id,
            "dataset": partition.dataset,
            "truth_kind": truth_kind,
            "truth_available": truth is not None,
            "healthy_train_rows": len(train_idx),
            "healthy_calibration_rows": len(calibration_idx),
            "healthy_test_rows": len(healthy_test_idx),
            "unhealthy_test_rows": len(unhealthy),
            "unscored_prediction_rows": len(unscored),
            "train_fault_rows": 0,
            "calibration_fault_rows": 0,
            "partition_overlap": 0,
            "status": "PASS",
        })
    return partitions, audit


def evaluate(model: common.TinyAE, partition: MachinePartition, protocol: dict[str, Any]) -> dict[str, Any]:
    alarm = protocol["alarm"]
    threshold = float(np.quantile(model.scores(partition.calibration), float(alarm["threshold_quantile"])))
    required = int(alarm["persistence_required"])
    window = int(alarm["persistence_window"])
    healthy_scores = model.scores(partition.healthy_test)
    healthy_alarm, healthy_first = common.persistent_alarm(healthy_scores, threshold, required, window)
    unhealthy_scores = model.scores(partition.unhealthy_test) if len(partition.unhealthy_test) else np.empty(0)
    fault_alarm, fault_first = common.persistent_alarm(unhealthy_scores, threshold, required, window)
    unscored_scores = model.scores(partition.unscored_test) if len(partition.unscored_test) else np.empty(0)
    unscored_alarm, unscored_first = common.persistent_alarm(unscored_scores, threshold, required, window)
    if partition.truth is not None:
        scored = True
        machine_success = (not healthy_alarm) and fault_alarm
        expected = "fault"
    elif partition.truth_kind == "healthy_control":
        scored = True
        machine_success = not healthy_alarm
        expected = "healthy_control"
    else:
        scored = False
        machine_success = False
        expected = "outcome_unavailable"
    healthy_raw = int(np.sum(healthy_scores > threshold))
    fault_raw = int(np.sum(unhealthy_scores > threshold))
    return {
        "machine_id": partition.machine_id,
        "dataset": partition.dataset,
        "truth_kind": partition.truth_kind,
        "expected": expected,
        "scored": scored,
        "machine_success": machine_success,
        "threshold": threshold,
        "healthy_test_rows": len(healthy_scores),
        "healthy_raw_alarm_rows": healthy_raw,
        "healthy_persistent_alarm": healthy_alarm,
        "healthy_first_persistent_alarm": "" if healthy_first is None else healthy_first,
        "unhealthy_test_rows": len(unhealthy_scores),
        "unhealthy_raw_alarm_rows": fault_raw,
        "fault_detected": fault_alarm,
        "first_unhealthy_persistent_alarm_delay": "" if fault_first is None else fault_first,
        "unscored_test_rows": len(unscored_scores),
        "unscored_persistent_alarm": unscored_alarm,
        "unscored_first_persistent_alarm": "" if unscored_first is None else unscored_first,
        "healthy_window_specificity": 1.0 - healthy_raw / len(healthy_scores),
        "unhealthy_window_recall": "" if not len(unhealthy_scores) else fault_raw / len(unhealthy_scores),
    }


def run(run_id: str) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    root = root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "execution_lock.json"
    driver_hash = sha256(Path(__file__).resolve())
    protocol_hash = sha256(PROTOCOL)
    common_hash = sha256(Path(common.__file__).resolve())
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("complete"):
            raise IntegrityError("frozen run already complete")
        if existing.get("driver_sha256") != driver_hash or existing.get("protocol_sha256") != protocol_hash:
            raise IntegrityError("execution dependency drift")
    devices, matrices, feature_names, artifact_hashes = load_70(protocol)
    partitions, audit = prepare_partitions(devices, matrices, protocol)
    clients = tuple(sorted(partitions))
    lock = {
        "schema": "thesis_70_device_federated_execution_lock_v1",
        "classification": protocol["classification"],
        "driver_sha256": driver_hash,
        "protocol_sha256": protocol_hash,
        "common_trainer_sha256": common_hash,
        "artifact_sha256": artifact_hashes,
        "logical_clients": len(clients),
        "federated_rounds": int(protocol["model"]["rounds"]),
        "complete": False,
    }
    write_json(lock_path, lock)
    write_csv(root / "partition_audit.csv", audit)
    print("training one FedAdam model with 70 logical clients for 12 rounds", flush=True)
    model = common.fit_federated(partitions, clients, "fedadam", protocol)
    rows = [evaluate(model, partitions[client], protocol) for client in clients]
    write_csv(root / "device_results.csv", rows)
    scored = [row for row in rows if row["scored"]]
    faults = [row for row in rows if row["expected"] == "fault"]
    controls = [row for row in rows if row["expected"] == "healthy_control"]
    unknown = [row for row in rows if row["expected"] == "outcome_unavailable"]
    successes = sum(bool(row["machine_success"]) for row in scored)
    detections = sum(bool(row["fault_detected"]) for row in faults)
    quiet = sum(not bool(row["healthy_persistent_alarm"]) for row in scored)
    fragments = int(protocol["communication_screen"]["fragments_per_update"])
    global_fragments = int(protocol["communication_screen"]["fragments_per_global_model"])
    rounds = int(protocol["model"]["rounds"])
    summary = {
        "logical_clients_trained": len(clients),
        "federated_rounds": rounds,
        "optimizer": "fedadam",
        "scored_devices": len(scored),
        "outcome_unavailable_devices": len(unknown),
        "strict_machine_successes": successes,
        "strict_machine_success_rate_scored": successes / len(scored),
        "strict_machine_success_rate_conservative_full_70": successes / len(rows),
        "known_fault_devices": len(faults),
        "faults_detected": detections,
        "fault_detection_rate": detections / len(faults),
        "scored_devices_without_healthy_alarm": quiet,
        "healthy_specificity_rate": quiet / len(scored),
        "healthy_controls": len(controls),
        "healthy_control_false_alarms": sum(bool(row["healthy_persistent_alarm"]) for row in controls),
        "uplink_update_fragments": len(clients) * rounds * fragments,
        "downlink_global_unicast_fragments": len(clients) * rounds * global_fragments,
        "total_fragments": len(clients) * rounds * (fragments + global_fragments),
        "physical_lilygo_boards": int(protocol["communication_screen"]["physical_lilygo_boards"]),
        "logical_clients_are_multiplexed": True,
    }
    write_json(root / "summary.json", summary)
    np.savez_compressed(
        root / "fedadam_70_client_model.npz",
        w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2,
        feature_names=np.asarray(feature_names), clients=np.asarray(clients),
    )
    report = [
        "# One 70-logical-client FedAdam trial", "",
        f"- Logical clients trained: **{len(clients)}**",
        f"- Federated rounds: **{rounds}**",
        f"- Scored devices: **{len(scored)}/70**; outcome unavailable: **{len(unknown)}**",
        f"- Strict machine success: **{successes}/{len(scored)} ({100 * summary['strict_machine_success_rate_scored']:.2f}%)**",
        f"- Conservative full-70 success: **{successes}/70 ({100 * summary['strict_machine_success_rate_conservative_full_70']:.2f}%)**",
        f"- Fault detection: **{detections}/{len(faults)} ({100 * summary['fault_detection_rate']:.2f}%)**",
        f"- Healthy-period specificity: **{quiet}/{len(scored)} ({100 * summary['healthy_specificity_rate']:.2f}%)**",
        f"- Healthy-control false alarms: **{summary['healthy_control_false_alarms']}/{len(controls)}**", "",
        "## Communication screen", "",
        f"The 12-round strict unicast screen requires {summary['uplink_update_fragments']} uplink update fragments and {summary['downlink_global_unicast_fragments']} downlink global-model fragments ({summary['total_fragments']} total). Seventy logical clients are multiplexed over the ten available boards; this is not a 70-radio hardware result.", "",
        "## Claim boundary", "",
        "This is a retrospective participating-client trial on opened data, not independent external validation. Five run-to-failure devices without accepted onset references are predicted and retained in the 70-client federation but cannot be placed in the scored accuracy denominator.", "",
    ]
    (root / "THESIS_70_DEVICE_FEDERATED_RESULT.md").write_text("\n".join(report), encoding="utf-8")
    lock.update({"complete": True, "scored_devices": len(scored), "device_rows": len(rows), "summary_sha256": sha256(root / "summary.json"), "model_sha256": sha256(root / "fedadam_70_client_model.npz")})
    write_json(lock_path, lock)
    print(json.dumps(summary, indent=2), flush=True)


def verify(run_id: str) -> None:
    root = root_for(run_id)
    lock = json.loads((root / "execution_lock.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((root / "device_results.csv").open(encoding="utf-8")))
    audit = list(csv.DictReader((root / "partition_audit.csv").open(encoding="utf-8")))
    checks = {
        "complete": lock.get("complete") is True,
        "driver_hash": lock.get("driver_sha256") == sha256(Path(__file__).resolve()),
        "protocol_hash": lock.get("protocol_sha256") == sha256(PROTOCOL),
        "common_hash": lock.get("common_trainer_sha256") == sha256(Path(common.__file__).resolve()),
        "summary_hash": lock.get("summary_sha256") == sha256(root / "summary.json"),
        "model_hash": lock.get("model_sha256") == sha256(root / "fedadam_70_client_model.npz"),
        "clients_70": summary.get("logical_clients_trained") == 70 and len(rows) == 70 and len(audit) == 70,
        "rounds_12": summary.get("federated_rounds") == 12,
        "audit_pass": all(row["status"] == "PASS" and row["train_fault_rows"] == "0" and row["calibration_fault_rows"] == "0" for row in audit),
        "five_unscored": summary.get("outcome_unavailable_devices") == 5,
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise IntegrityError("verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "verify"), required=True)
    parser.add_argument("--run-id", default="frozen1")
    args = parser.parse_args()
    if args.phase == "run":
        run(args.run_id)
    else:
        verify(args.run_id)


if __name__ == "__main__":
    main()
