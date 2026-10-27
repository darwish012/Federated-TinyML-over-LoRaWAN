"""Run the complete 2,4,...,70 healthy-only FedAdam scaling curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

import thesis_70_device_federated as endpoint
import thesis_participating_node_scaling as common


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "thesis_70_node_scaling_protocol.json"
TRAINING_PROTOCOL = HERE / "thesis_70_device_federated_protocol.json"
RESULTS = HERE / "results"


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def root_for(run_id: str) -> Path:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not run_id or any(char not in allowed for char in run_id):
        raise IntegrityError("invalid run id")
    return RESULTS / f"thesis_70_node_scaling_{run_id}"


def subset_design(device_ids: tuple[str, ...], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    if len(device_ids) != 70:
        raise IntegrityError("subset design requires exactly 70 devices")
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(itertools.combinations(device_ids, 2)):
        rows.append({
            "logical_clients": 2,
            "subset_id": f"n02_pair_{index:04d}",
            "design": "exhaustive_pair",
            "clients": tuple(pair),
        })
    if len(rows) != math.comb(70, 2):
        raise IntegrityError("exhaustive pair barrier failed")
    rng = np.random.default_rng(int(protocol["subset_design"]["design_seed"]))
    for n in range(4, 70, 2):
        permutation = list(rng.permutation(device_ids))
        pairs = [(permutation[2 * index], permutation[2 * index + 1]) for index in range(35)]
        width = n // 2
        subsets: list[tuple[str, ...]] = []
        for offset in range(35):
            selected = [client for step in range(width) for client in pairs[(offset + step) % 35]]
            subsets.append(tuple(sorted(selected)))
        if len(set(subsets)) != 35:
            raise IntegrityError(f"balanced subset uniqueness failed for N={n}")
        appearances = {device_id: sum(device_id in subset for subset in subsets) for device_id in device_ids}
        if set(appearances.values()) != {width}:
            raise IntegrityError(f"balanced incidence failed for N={n}")
        for index, subset in enumerate(subsets):
            rows.append({
                "logical_clients": n,
                "subset_id": f"n{n:02d}_balanced_{index:02d}",
                "design": "balanced_cyclic_pairs",
                "clients": subset,
            })
    rows.append({
        "logical_clients": 70,
        "subset_id": "n70_all",
        "design": "all_devices",
        "clients": device_ids,
    })
    if len(rows) != int(protocol["subset_design"]["expected_total_configurations"]):
        raise IntegrityError("total configuration barrier failed")
    return rows


def configuration_metrics(
    device_rows: list[dict[str, Any]], clients: tuple[str, ...], n: int,
    subset_id: str, design: str,
) -> dict[str, Any]:
    selected = set(clients)
    scored = [row for row in device_rows if bool(row["scored"])]
    faults = [row for row in device_rows if row["expected"] == "fault"]
    trained_scored = [row for row in scored if row["machine_id"] in selected]
    untrained_scored = [row for row in scored if row["machine_id"] not in selected]
    trained_faults = [row for row in faults if row["machine_id"] in selected]
    untrained_faults = [row for row in faults if row["machine_id"] not in selected]

    def rate(rows: list[dict[str, Any]], field: str, *, invert: bool = False) -> float | str:
        if not rows:
            return ""
        values = [not bool(row[field]) if invert else bool(row[field]) for row in rows]
        return sum(values) / len(values)

    return {
        "logical_clients": n,
        "subset_id": subset_id,
        "design": design,
        "training_clients": "|".join(clients),
        "trained_scored_devices": len(trained_scored),
        "untrained_scored_devices": len(untrained_scored),
        "all_scored_devices": len(scored),
        "trained_strict_success_rate": rate(trained_scored, "machine_success"),
        "untrained_strict_success_rate": rate(untrained_scored, "machine_success"),
        "all_strict_successes": sum(bool(row["machine_success"]) for row in scored),
        "all_strict_success_rate": rate(scored, "machine_success"),
        "conservative_full_70_success_rate": sum(bool(row["machine_success"]) for row in scored) / 70,
        "trained_known_faults": len(trained_faults),
        "untrained_known_faults": len(untrained_faults),
        "all_known_faults": len(faults),
        "trained_fault_detection_rate": rate(trained_faults, "fault_detected"),
        "untrained_fault_detection_rate": rate(untrained_faults, "fault_detected"),
        "all_fault_detections": sum(bool(row["fault_detected"]) for row in faults),
        "all_fault_detection_rate": rate(faults, "fault_detected"),
        "trained_healthy_specificity_rate": rate(trained_scored, "healthy_persistent_alarm", invert=True),
        "untrained_healthy_specificity_rate": rate(untrained_scored, "healthy_persistent_alarm", invert=True),
        "all_healthy_specificity_rate": rate(scored, "healthy_persistent_alarm", invert=True),
    }


def summarize(configurations: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rate_fields = (
        "trained_strict_success_rate", "untrained_strict_success_rate", "all_strict_success_rate",
        "conservative_full_70_success_rate", "trained_fault_detection_rate",
        "untrained_fault_detection_rate", "all_fault_detection_rate",
        "trained_healthy_specificity_rate", "untrained_healthy_specificity_rate",
        "all_healthy_specificity_rate",
    )
    for n in protocol["cohort"]["node_ladder"]:
        selected = [row for row in configurations if int(row["logical_clients"]) == int(n)]
        summary: dict[str, Any] = {
            "logical_clients": int(n),
            "configurations": len(selected),
            "all_evaluation_devices": 70,
            "all_scored_devices": 65,
            "all_known_fault_devices": 59,
        }
        for field in rate_fields:
            values = [float(row[field]) for row in selected if row[field] != ""]
            summary[f"mean_{field}"] = statistics.fmean(values) if values else ""
            summary[f"std_{field}"] = statistics.pstdev(values) if len(values) > 1 else 0.0 if values else ""
            summary[f"minimum_{field}"] = min(values) if values else ""
            summary[f"maximum_{field}"] = max(values) if values else ""
        rows.append(summary)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def communication_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    comm = protocol["communication"]
    rounds = int(comm["rounds"])
    per_client = int(comm["fragments_per_update"]) + int(comm["fragments_per_global_model"])
    return [{
        "logical_clients": int(n),
        "rounds": rounds,
        "uplink_fragments": int(n) * rounds * int(comm["fragments_per_update"]),
        "downlink_unicast_fragments": int(n) * rounds * int(comm["fragments_per_global_model"]),
        "total_fragments": int(n) * rounds * per_client,
        "simultaneous_physical_boards_required": int(n),
        "fits_ten_physical_boards_simultaneously": int(n) <= int(comm["physical_lilygo_boards"]),
    } for n in protocol["cohort"]["node_ladder"]]


def render_report(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Complete 2-to-70 participating-client FedAdam scaling", "",
        "Every configuration trains for 12 rounds on healthy-only client partitions and is evaluated on the same 70 devices. Scored accuracy uses 65 devices; five onset-unavailable devices remain prediction-only.", "",
        "| training clients | configurations | trained strict success | all-65 strict success | all-59 fault detection | all-65 healthy specificity |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def percent(field: str) -> str:
            value = row[field]
            return "NA" if value == "" else f"{100 * float(value):.2f}%"
        lines.append(
            f"| {row['logical_clients']} | {row['configurations']} | "
            f"{percent('mean_trained_strict_success_rate')} | "
            f"{percent('mean_all_strict_success_rate')} | "
            f"{percent('mean_all_fault_detection_rate')} | "
            f"{percent('mean_all_healthy_specificity_rate')} |"
        )
    lines.extend([
        "", "## Interpretation rules", "",
        "- N=2 contains all 2,415 unordered pairs.",
        "- Each N=4–68 contains 35 balanced subsets; every device appears exactly N/2 times at that node count.",
        "- N=70 is the single complete cohort.",
        "- Subset ranges measure sensitivity to client composition; they are not confidence intervals and configurations are not independent machines.",
        "- The N=70 endpoint was known before this full curve was launched, so this is descriptive retrospective evidence.", "",
    ])
    return "\n".join(lines)


def run(run_id: str) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    training_protocol = json.loads(TRAINING_PROTOCOL.read_text(encoding="utf-8"))
    root = root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    driver_hash = sha256(Path(__file__).resolve())
    protocol_hash = sha256(PROTOCOL)
    training_hash = sha256(TRAINING_PROTOCOL)
    endpoint_hash = sha256(Path(endpoint.__file__).resolve())
    common_hash = sha256(Path(common.__file__).resolve())
    lock_path = root / "execution_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = (driver_hash, protocol_hash, training_hash, endpoint_hash, common_hash)
        actual = tuple(previous.get(key) for key in ("driver_sha256", "protocol_sha256", "training_protocol_sha256", "endpoint_loader_sha256", "common_trainer_sha256"))
        if actual != expected:
            raise IntegrityError("execution dependency hash drift")
        if previous.get("complete"):
            raise IntegrityError("frozen scaling run already complete")
    devices, matrices, feature_names, artifact_hashes = endpoint.load_70(training_protocol)
    partitions, partition_audit = endpoint.prepare_partitions(devices, matrices, training_protocol)
    device_ids = tuple(sorted(partitions))
    design = subset_design(device_ids, protocol)
    lock = {
        "schema": "thesis_70_node_scaling_execution_lock_v1",
        "classification": protocol["classification"],
        "driver_sha256": driver_hash,
        "protocol_sha256": protocol_hash,
        "training_protocol_sha256": training_hash,
        "endpoint_loader_sha256": endpoint_hash,
        "common_trainer_sha256": common_hash,
        "artifact_sha256": artifact_hashes,
        "device_count": len(device_ids),
        "configuration_count": len(design),
        "complete": False,
    }
    atomic_json(lock_path, lock)
    endpoint.write_csv(root / "partition_audit.csv", partition_audit)
    write_csv(root / "subset_manifest.csv", [{
        "logical_clients": row["logical_clients"],
        "subset_id": row["subset_id"],
        "design": row["design"],
        "training_clients": "|".join(row["clients"]),
    } for row in design])
    checkpoint_root = root / "checkpoints"
    for index, subset in enumerate(design, start=1):
        checkpoint = checkpoint_root / f"{subset['subset_id']}.json"
        if checkpoint.exists():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("driver_sha256") != driver_hash or payload.get("protocol_sha256") != protocol_hash:
                raise IntegrityError(f"checkpoint hash drift: {checkpoint}")
        else:
            model = common.fit_federated(partitions, subset["clients"], "fedadam", training_protocol)
            device_rows = [endpoint.evaluate(model, partitions[device_id], training_protocol) for device_id in device_ids]
            metrics = configuration_metrics(device_rows, subset["clients"], int(subset["logical_clients"]), str(subset["subset_id"]), str(subset["design"]))
            atomic_json(checkpoint, {
                "schema": "thesis_70_node_scaling_checkpoint_v1",
                "driver_sha256": driver_hash,
                "protocol_sha256": protocol_hash,
                "configuration": metrics,
                "devices": device_rows,
            })
        if index == 1 or index % 50 == 0 or index == len(design):
            print(f"completed configurations {index}/{len(design)}", flush=True)

    configurations: list[dict[str, Any]] = []
    device_results: list[dict[str, Any]] = []
    for subset in design:
        payload = json.loads((checkpoint_root / f"{subset['subset_id']}.json").read_text(encoding="utf-8"))
        configurations.append(payload["configuration"])
        selected = set(subset["clients"])
        for row in payload["devices"]:
            device_results.append({
                "logical_clients": subset["logical_clients"],
                "subset_id": subset["subset_id"],
                "training_client": row["machine_id"] in selected,
                **row,
            })
    summary = summarize(configurations, protocol)
    write_csv(root / "configuration_metrics.csv", configurations)
    write_csv(root / "device_results.csv", device_results)
    write_csv(root / "node_scaling_summary.csv", summary)
    write_csv(root / "communication_screen.csv", communication_rows(protocol))
    (root / "THESIS_70_NODE_SCALING_RESULTS.md").write_text(render_report(summary), encoding="utf-8")
    lock.update({
        "complete": True,
        "configuration_rows": len(configurations),
        "device_result_rows": len(device_results),
        "summary_sha256": sha256(root / "node_scaling_summary.csv"),
    })
    atomic_json(lock_path, lock)
    print(f"COMPLETE: {len(configurations)} configurations; {len(device_results)} device evaluations", flush=True)


def verify(run_id: str) -> None:
    root = root_for(run_id)
    lock = json.loads((root / "execution_lock.json").read_text(encoding="utf-8"))
    checks = {
        "complete": lock.get("complete") is True,
        "driver_hash": lock.get("driver_sha256") == sha256(Path(__file__).resolve()),
        "protocol_hash": lock.get("protocol_sha256") == sha256(PROTOCOL),
        "training_protocol_hash": lock.get("training_protocol_sha256") == sha256(TRAINING_PROTOCOL),
        "endpoint_hash": lock.get("endpoint_loader_sha256") == sha256(Path(endpoint.__file__).resolve()),
        "common_hash": lock.get("common_trainer_sha256") == sha256(Path(common.__file__).resolve()),
        "configurations": lock.get("configuration_rows") == 3571,
        "device_results": lock.get("device_result_rows") == 3571 * 70,
        "summary_hash": lock.get("summary_sha256") == sha256(root / "node_scaling_summary.csv"),
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
