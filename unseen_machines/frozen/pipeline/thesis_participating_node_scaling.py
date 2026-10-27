"""Healthy-only participating-client node scaling over all 14 available sites.

FedAdam is primary and FedAvg is a fixed comparator.  Every two-client pair is
evaluated.  Larger node counts use a deterministic balanced design, and every
trained model is evaluated on the same fourteen devices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import realtime_federated_simulation as realtime


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "thesis_participating_node_scaling_protocol.json"
RESULTS = HERE / "results"


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extras)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def result_root(run_id: str) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
        raise IntegrityError("run id must contain only letters, numbers, underscore or hyphen")
    return RESULTS / f"thesis_participating_node_scaling_{run_id}"


@dataclass(frozen=True)
class DevicePartition:
    device_id: str
    dataset: str
    logical_client: str
    train: np.ndarray
    calibration: np.ndarray
    healthy_test: np.ndarray
    unhealthy_by_source: tuple[tuple[str, np.ndarray], ...]
    healthy_by_source: tuple[tuple[str, np.ndarray], ...]
    mean: np.ndarray
    scale: np.ndarray
    train_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    healthy_test_ids: tuple[str, ...]
    unhealthy_ids: tuple[str, ...]


@dataclass
class TinyAE:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def arrays(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def copy(self) -> "TinyAE":
        return TinyAE(*(value.copy() for value in self.arrays()))

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        hidden = np.tanh(values @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def scores(self, values: np.ndarray) -> np.ndarray:
        error = self.reconstruct(values) - values
        return np.mean(error * error, axis=1)


def initial_model(protocol: dict[str, Any]) -> TinyAE:
    model = protocol["model"]
    width = int(model["input_features"])
    hidden = int(model["hidden_units"])
    rng = np.random.default_rng(int(model["initialization_seed"]))
    w1 = rng.normal(0.0, math.sqrt(2.0 / (width + hidden)), (width, hidden))
    w2 = rng.normal(0.0, math.sqrt(2.0 / (width + hidden)), (hidden, width))
    result = TinyAE(w1, np.zeros(hidden), w2, np.zeros(width))
    if sum(value.size for value in result.arrays()) != int(model["trainable_parameters"]):
        raise IntegrityError("trainable parameter count drift")
    return result


def local_step(model: TinyAE, values: np.ndarray, protocol: dict[str, Any]) -> TinyAE:
    result = model.copy()
    spec = protocol["model"]
    rate = float(spec["local_learning_rate"])
    l2 = float(spec["l2"])
    normalizer = float(max(values.shape[0] * values.shape[1], 1))
    for _ in range(int(spec["local_epochs"])):
        hidden = np.tanh(values @ result.w1 + result.b1)
        reconstruction = hidden @ result.w2 + result.b2
        output_gradient = 2.0 * (reconstruction - values) / normalizer
        grad_w2 = hidden.T @ output_gradient + l2 * result.w2
        grad_b2 = np.sum(output_gradient, axis=0)
        hidden_gradient = (output_gradient @ result.w2.T) * (1.0 - hidden * hidden)
        grad_w1 = values.T @ hidden_gradient + l2 * result.w1
        grad_b1 = np.sum(hidden_gradient, axis=0)
        result.w1 -= rate * grad_w1
        result.b1 -= rate * grad_b1
        result.w2 -= rate * grad_w2
        result.b2 -= rate * grad_b2
    return result


def fit_federated(
    partitions: dict[str, DevicePartition], clients: tuple[str, ...], optimizer: str,
    protocol: dict[str, Any],
) -> TinyAE:
    if optimizer not in {"fedavg", "fedadam"}:
        raise IntegrityError(f"unsupported optimizer {optimizer}")
    global_model = initial_model(protocol)
    moments = [np.zeros_like(value) for value in global_model.arrays()]
    variances = [np.zeros_like(value) for value in global_model.arrays()]
    spec = protocol["model"]
    beta1 = float(spec["fedadam_beta1"])
    beta2 = float(spec["fedadam_beta2"])
    eta = float(spec["fedadam_server_learning_rate"])
    tau = float(spec["fedadam_tau"])
    for round_index in range(1, int(spec["rounds"]) + 1):
        deltas: list[list[np.ndarray]] = []
        for client in clients:
            local = local_step(global_model, partitions[client].train, protocol)
            deltas.append([
                local_value - global_value
                for local_value, global_value in zip(local.arrays(), global_model.arrays())
            ])
        average = [
            np.mean(np.stack([client_delta[index] for client_delta in deltas]), axis=0)
            for index in range(4)
        ]
        if optimizer == "fedavg":
            for target, delta in zip(global_model.arrays(), average):
                target += delta
        else:
            for index, (target, delta) in enumerate(zip(global_model.arrays(), average)):
                moments[index] = beta1 * moments[index] + (1.0 - beta1) * delta
                variances[index] = beta2 * variances[index] + (1.0 - beta2) * delta * delta
                m_hat = moments[index] / (1.0 - beta1 ** round_index)
                v_hat = variances[index] / (1.0 - beta2 ** round_index)
                target += eta * m_hat / (np.sqrt(v_hat) + tau)
        if not all(np.all(np.isfinite(value)) for value in global_model.arrays()):
            raise IntegrityError(f"non-finite model for {optimizer} clients={clients}")
    return global_model


def _hash_rank(salt: str, device_id: str, window_id: str) -> str:
    return hashlib.sha256(f"{salt}|{device_id}|{window_id}".encode("utf-8")).hexdigest()


def _source_arrays(
    windows: Iterable[Any], mean: np.ndarray, scale: np.ndarray, *, lifecycle: bool,
) -> tuple[tuple[str, np.ndarray], ...]:
    groups: dict[str, list[Any]] = {}
    for window in windows:
        key = "complete_lifecycle" if lifecycle else str(window.source_file)
        groups.setdefault(key, []).append(window)
    output: list[tuple[str, np.ndarray]] = []
    for source, values in sorted(groups.items()):
        ordered = sorted(values, key=realtime._window_sort_key)
        matrix = np.asarray([(item.raw_features - mean) / scale for item in ordered], dtype=np.float64)
        output.append((source, matrix))
    return tuple(output)


def prepare_partitions(protocol: dict[str, Any]) -> tuple[dict[str, DevicePartition], list[dict[str, Any]]]:
    config = realtime.RealtimeConfig(
        num_devices=14,
        include_ldm_clients=True,
        ldm_split_strategy=str(protocol["cohort"]["ldm_split_strategy"]),
    )
    shards, _, leakage = realtime.load_device_study(config)
    if len(shards) != int(protocol["cohort"]["expected_devices"]):
        raise IntegrityError("device count drift")
    if any(row["status"] != "PASS" for row in leakage):
        raise IntegrityError("upstream window/source leakage audit failed")
    split = protocol["partition"]
    partitions: dict[str, DevicePartition] = {}
    audit: list[dict[str, Any]] = []
    for device_id, shard in sorted(shards.items()):
        stream = next(iter(shard.streams.values()))
        unique: dict[str, Any] = {}
        for window in list(stream.train) + list(stream.operation) + list(stream.server_test):
            unique.setdefault(str(window.window_id), window)
        windows = list(unique.values())
        healthy = [window for window in windows if int(window.label) == 0]
        unhealthy = [window for window in windows if int(window.label) == 1]
        ranked = sorted(
            healthy,
            key=lambda item: _hash_rank(str(split["split_salt"]), device_id, str(item.window_id)),
        )
        calibration_count = max(int(split["minimum_calibration_windows"]), int(math.floor(len(ranked) * float(split["healthy_calibration_fraction"]))))
        test_count = max(int(split["minimum_healthy_test_windows"]), int(math.floor(len(ranked) * float(split["healthy_test_fraction"]))))
        if calibration_count + test_count >= len(ranked):
            raise IntegrityError(f"insufficient healthy windows for {device_id}")
        calibration_windows = ranked[:calibration_count]
        healthy_test_windows = ranked[calibration_count:calibration_count + test_count]
        train_windows = ranked[calibration_count + test_count:]
        groups = [
            {str(item.window_id) for item in train_windows},
            {str(item.window_id) for item in calibration_windows},
            {str(item.window_id) for item in healthy_test_windows},
            {str(item.window_id) for item in unhealthy},
        ]
        if any(groups[left] & groups[right] for left in range(4) for right in range(left + 1, 4)):
            raise IntegrityError(f"partition overlap for {device_id}")
        calibration_raw = np.asarray([item.raw_features for item in calibration_windows], dtype=np.float64)
        mean = np.mean(calibration_raw, axis=0)
        scale = np.std(calibration_raw, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        train = np.asarray([(item.raw_features - mean) / scale for item in train_windows], dtype=np.float64)
        calibration = (calibration_raw - mean) / scale
        healthy_test = np.asarray([(item.raw_features - mean) / scale for item in healthy_test_windows], dtype=np.float64)
        partition = DevicePartition(
            device_id=device_id,
            dataset=str(stream.dataset),
            logical_client=str(stream.logical_client),
            train=train,
            calibration=calibration,
            healthy_test=healthy_test,
            unhealthy_by_source=_source_arrays(
                unhealthy, mean, scale, lifecycle=str(stream.dataset) == realtime.LDM_DATASET,
            ),
            healthy_by_source=_source_arrays(
                healthy_test_windows, mean, scale, lifecycle=str(stream.dataset) == realtime.LDM_DATASET,
            ),
            mean=mean,
            scale=scale,
            train_ids=tuple(sorted(groups[0])),
            calibration_ids=tuple(sorted(groups[1])),
            healthy_test_ids=tuple(sorted(groups[2])),
            unhealthy_ids=tuple(sorted(groups[3])),
        )
        partitions[device_id] = partition
        audit.append({
            "device_id": device_id,
            "dataset": partition.dataset,
            "logical_client": partition.logical_client,
            "healthy_train_windows": len(partition.train_ids),
            "healthy_calibration_windows": len(partition.calibration_ids),
            "healthy_test_windows": len(partition.healthy_test_ids),
            "unhealthy_test_windows": len(partition.unhealthy_ids),
            "train_fault_windows": 0,
            "calibration_fault_windows": 0,
            "unhealthy_windows_omitted_from_test": 0,
            "pairwise_overlap_windows": 0,
            "status": "PASS",
        })
    return partitions, audit


def subset_design(device_ids: tuple[str, ...], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    if len(device_ids) != int(protocol["cohort"]["expected_devices"]):
        raise IntegrityError("subset design device count drift")
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(itertools.combinations(device_ids, 2)):
        rows.append({"logical_clients": 2, "subset_id": f"n02_pair_{index:03d}", "design": "exhaustive", "clients": tuple(pair)})
    if len([row for row in rows if row["logical_clients"] == 2]) != int(protocol["subset_design"]["expected_two_device_combinations"]):
        raise IntegrityError("two-device combination count drift")
    rng = np.random.default_rng(int(protocol["subset_design"]["design_seed"]))
    for n in (4, 6, 8, 10, 12):
        seen: set[tuple[str, ...]] = set()
        sequence = 0
        for _ in range(2):
            block: list[tuple[str, ...]] = []
            for _attempt in range(1000):
                permutation = list(rng.permutation(device_ids))
                candidate_block = [
                    tuple(sorted(permutation[(offset + step) % len(device_ids)] for step in range(n)))
                    for offset in range(len(device_ids))
                ]
                if len(set(candidate_block)) == len(device_ids) and not (set(candidate_block) & seen):
                    block = candidate_block
                    break
            if not block:
                raise IntegrityError(f"could not construct balanced unique subset block for N={n}")
            for subset in block:
                seen.add(subset)
                rows.append({"logical_clients": n, "subset_id": f"n{n:02d}_balanced_{sequence:03d}", "design": "balanced_cyclic", "clients": subset})
                sequence += 1
    rows.append({"logical_clients": 14, "subset_id": "n14_all", "design": "all_devices", "clients": tuple(device_ids)})
    return rows


def persistent_alarm(scores: np.ndarray, threshold: float, required: int, window: int) -> tuple[bool, int | None]:
    raw = np.asarray(scores) > float(threshold)
    if len(raw) < window:
        return False, None
    counts = np.convolve(raw.astype(np.int16), np.ones(window, dtype=np.int16), mode="valid")
    hits = np.flatnonzero(counts >= required)
    if not len(hits):
        return False, None
    return True, int(hits[0] + window - 1)


def evaluate_device(model: TinyAE, partition: DevicePartition, protocol: dict[str, Any]) -> dict[str, Any]:
    alarm = protocol["alarm"]
    threshold = float(np.quantile(model.scores(partition.calibration), float(alarm["threshold_quantile"])))
    required = int(alarm["persistence_required"])
    window = int(alarm["persistence_window"])
    healthy_raw = 0
    healthy_total = 0
    healthy_alarm = False
    for _, values in partition.healthy_by_source:
        scores = model.scores(values)
        healthy_raw += int(np.sum(scores > threshold))
        healthy_total += len(scores)
        source_alarm, _ = persistent_alarm(scores, threshold, required, window)
        healthy_alarm = healthy_alarm or source_alarm
    fault_raw = 0
    fault_total = 0
    fault_alarm = False
    first_delays: list[int] = []
    for _, values in partition.unhealthy_by_source:
        scores = model.scores(values)
        fault_raw += int(np.sum(scores > threshold))
        fault_total += len(scores)
        source_alarm, delay = persistent_alarm(scores, threshold, required, window)
        fault_alarm = fault_alarm or source_alarm
        if delay is not None:
            first_delays.append(delay)
    tn = healthy_total - healthy_raw
    tp = fault_raw
    fp = healthy_raw
    fn = fault_total - fault_raw
    recall = tp / fault_total if fault_total else 0.0
    specificity = tn / healthy_total if healthy_total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "device_id": partition.device_id,
        "dataset": partition.dataset,
        "logical_client": partition.logical_client,
        "threshold": threshold,
        "healthy_test_windows": healthy_total,
        "unhealthy_test_windows": fault_total,
        "healthy_false_positive_windows": fp,
        "unhealthy_true_positive_windows": tp,
        "healthy_alarm": healthy_alarm,
        "fault_detected": fault_alarm,
        "machine_success": (not healthy_alarm) and fault_alarm,
        "window_specificity": specificity,
        "window_recall": recall,
        "window_balanced_accuracy": 0.5 * (specificity + recall),
        "window_f1": f1,
        "first_unhealthy_persistent_alarm_delay": min(first_delays) if first_delays else "",
    }


def communication_row(n: int, protocol: dict[str, Any]) -> dict[str, Any]:
    model = protocol["model"]
    comm = protocol["communication_screen"]
    rounds = int(model["rounds"])
    update = int(comm["fragments_per_update"])
    downlink = int(comm["fragments_per_global_model"])
    return {
        "logical_clients": n,
        "rounds": rounds,
        "uplink_fragments": n * rounds * update,
        "downlink_unicast_fragments": n * rounds * downlink,
        "total_fragments": n * rounds * (update + downlink),
        "application_parameter_bytes": n * rounds * 2 * int(comm["serialized_parameter_bytes"]),
        "physical_boards_required_if_simultaneous": n,
        "fits_ten_physical_boards_simultaneously": n <= int(comm["physical_lilygo_boards"]),
    }


def summarize(config_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for optimizer in protocol["model"]["optimizers"]:
        for n in protocol["cohort"]["node_ladder"]:
            rows = [row for row in config_rows if row["optimizer"] == optimizer and row["logical_clients"] == n]
            rates = [float(row["machine_success_rate"]) for row in rows]
            detections = [float(row["machine_detection_rate"]) for row in rows]
            specificities = [float(row["machine_healthy_specificity_rate"]) for row in rows]
            output.append({
                "optimizer": optimizer,
                "logical_clients": n,
                "configurations": len(rows),
                "evaluation_devices_per_configuration": 14,
                "mean_machine_success_rate": statistics.fmean(rates),
                "std_machine_success_rate": statistics.pstdev(rates) if len(rates) > 1 else 0.0,
                "minimum_machine_success_rate": min(rates),
                "median_machine_success_rate": statistics.median(rates),
                "maximum_machine_success_rate": max(rates),
                "mean_machine_detection_rate": statistics.fmean(detections),
                "mean_machine_healthy_specificity_rate": statistics.fmean(specificities),
                "mean_window_balanced_accuracy": statistics.fmean(float(row["mean_window_balanced_accuracy"]) for row in rows),
                "mean_window_f1": statistics.fmean(float(row["mean_window_f1"]) for row in rows),
            })
    return output


def render_report(summary: list[dict[str, Any]], protocol: dict[str, Any]) -> str:
    lines = [
        "# Participating-client node-scaling experiment", "",
        "Every two-device pair was fitted. Intermediate node counts use 28 deterministic balanced subsets; the final point uses all 14 devices. Every configuration is evaluated on the same 14-device cohort.", "",
        "Training and calibration use healthy windows only. All unhealthy windows and a disjoint healthy holdout are test-only.", "",
        "| optimizer | training devices | configurations | mean machine success | range | mean detection | mean healthy specificity | window balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['optimizer']} | {row['logical_clients']} | {row['configurations']} | "
            f"{100 * row['mean_machine_success_rate']:.2f}% | "
            f"{100 * row['minimum_machine_success_rate']:.2f}–{100 * row['maximum_machine_success_rate']:.2f}% | "
            f"{100 * row['mean_machine_detection_rate']:.2f}% | "
            f"{100 * row['mean_machine_healthy_specificity_rate']:.2f}% | "
            f"{100 * row['mean_window_balanced_accuracy']:.2f}% |"
        )
    lines.extend([
        "", "## Claim boundary", "",
        "This is a retrospective participating-client engineering benchmark. The machines and outcomes were previously opened, so these results are not independent external validation. Client-subset configurations quantify sensitivity and do not multiply the independent machine denominator.", "",
        "The communication table is an analytical strict Class-A unicast screen at 200 application bytes per fragment. Counts above ten require virtual logical clients because only ten physical LilyGO boards are available.", "",
    ])
    return "\n".join(lines)


def execute(run_id: str, limit: int | None = None) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_first_execution":
        raise IntegrityError("protocol status drift")
    root = result_root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    driver_hash = sha256(Path(__file__).resolve())
    protocol_hash = sha256(PROTOCOL)
    lock_path = root / "execution_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["driver_sha256"] != driver_hash or lock["protocol_sha256"] != protocol_hash:
            raise IntegrityError("execution dependency hash drift")
    partitions, partition_audit = prepare_partitions(protocol)
    device_ids = tuple(sorted(partitions))
    design = subset_design(device_ids, protocol)
    lock = {
        "schema": "thesis_participating_node_scaling_execution_lock_v1",
        "classification": protocol["classification"],
        "driver_sha256": driver_hash,
        "protocol_sha256": protocol_hash,
        "upstream_runner_sha256": sha256(Path(realtime.__file__).resolve()),
        "device_count": len(device_ids),
        "subset_count": len(design),
        "two_device_subset_count": sum(row["logical_clients"] == 2 for row in design),
        "optimizers": protocol["model"]["optimizers"],
        "complete": False,
    }
    atomic_json(lock_path, lock)
    write_csv(root / "partition_audit.csv", partition_audit)
    write_csv(root / "subset_manifest.csv", [
        {**{key: value for key, value in row.items() if key != "clients"}, "training_clients": "|".join(row["clients"])}
        for row in design
    ])
    chosen = design if limit is None else design[:limit]
    checkpoint_root = root / "checkpoints"
    completed_configurations = 0
    for subset_index, subset in enumerate(chosen, start=1):
        for optimizer in protocol["model"]["optimizers"]:
            checkpoint = checkpoint_root / optimizer / f"{subset['subset_id']}.json"
            if checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("protocol_sha256") == protocol_hash and payload.get("driver_sha256") == driver_hash:
                    continue
                raise IntegrityError(f"checkpoint hash drift: {checkpoint}")
            model = fit_federated(partitions, subset["clients"], optimizer, protocol)
            devices = [evaluate_device(model, partitions[device_id], protocol) for device_id in device_ids]
            successes = sum(bool(row["machine_success"]) for row in devices)
            detections = sum(bool(row["fault_detected"]) for row in devices)
            healthy_specific = sum(not bool(row["healthy_alarm"]) for row in devices)
            configuration = {
                "optimizer": optimizer,
                "logical_clients": subset["logical_clients"],
                "subset_id": subset["subset_id"],
                "design": subset["design"],
                "training_clients": "|".join(subset["clients"]),
                "evaluation_devices": len(devices),
                "machine_successes": successes,
                "machine_success_rate": successes / len(devices),
                "machine_detections": detections,
                "machine_detection_rate": detections / len(devices),
                "machine_healthy_specific": healthy_specific,
                "machine_healthy_specificity_rate": healthy_specific / len(devices),
                "mean_window_balanced_accuracy": statistics.fmean(float(row["window_balanced_accuracy"]) for row in devices),
                "mean_window_f1": statistics.fmean(float(row["window_f1"]) for row in devices),
            }
            atomic_json(checkpoint, {
                "schema": "thesis_participating_node_scaling_checkpoint_v1",
                "protocol_sha256": protocol_hash,
                "driver_sha256": driver_hash,
                "configuration": configuration,
                "devices": devices,
            })
        completed_configurations += 1
        if subset_index == 1 or subset_index % 10 == 0 or subset_index == len(chosen):
            print(f"completed subsets {subset_index}/{len(chosen)}", flush=True)
    if limit is not None:
        print(f"limited execution complete: {completed_configurations} subsets", flush=True)
        return
    config_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    for subset in design:
        for optimizer in protocol["model"]["optimizers"]:
            checkpoint = checkpoint_root / optimizer / f"{subset['subset_id']}.json"
            if not checkpoint.exists():
                raise IntegrityError(f"missing checkpoint: {checkpoint}")
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            config_rows.append(payload["configuration"])
            for row in payload["devices"]:
                device_rows.append({
                    "optimizer": optimizer,
                    "logical_clients": subset["logical_clients"],
                    "subset_id": subset["subset_id"],
                    "training_client": row["device_id"] in subset["clients"],
                    **row,
                })
    summary = summarize(config_rows, protocol)
    write_csv(root / "configuration_metrics.csv", config_rows)
    write_csv(root / "device_metrics.csv", device_rows)
    write_csv(root / "node_scaling_summary.csv", summary)
    write_csv(root / "communication_screen.csv", [communication_row(int(n), protocol) for n in protocol["cohort"]["node_ladder"]])
    (root / "THESIS_PARTICIPATING_NODE_SCALING_RESULTS.md").write_text(render_report(summary, protocol), encoding="utf-8")
    lock["complete"] = True
    lock["configuration_results"] = len(config_rows)
    lock["device_results"] = len(device_rows)
    atomic_json(lock_path, lock)
    print(f"COMPLETE: {len(config_rows)} optimizer/configurations; {len(device_rows)} device evaluations", flush=True)


def verify(run_id: str) -> None:
    root = result_root(run_id)
    lock = json.loads((root / "execution_lock.json").read_text(encoding="utf-8"))
    checks = {
        "complete": lock.get("complete") is True,
        "driver_hash": lock.get("driver_sha256") == sha256(Path(__file__).resolve()),
        "protocol_hash": lock.get("protocol_sha256") == sha256(PROTOCOL),
        "upstream_hash": lock.get("upstream_runner_sha256") == sha256(Path(realtime.__file__).resolve()),
        "two_device_pairs": int(lock.get("two_device_subset_count", -1)) == 91,
        "configuration_results": int(lock.get("configuration_results", -1)) == 2 * 232,
        "device_results": int(lock.get("device_results", -1)) == 2 * 232 * 14,
        "partition_audit": all(row["status"] == "PASS" for row in csv.DictReader((root / "partition_audit.csv").open(encoding="utf-8"))),
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise IntegrityError("verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "verify"), required=True)
    parser.add_argument("--run-id", default="frozen1")
    parser.add_argument("--limit", type=int, help="development smoke limit; incomplete and not reportable")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.phase == "run":
        execute(args.run_id, args.limit)
    else:
        if args.limit is not None:
            parser.error("--limit is valid only for --phase run")
        verify(args.run_id)


if __name__ == "__main__":
    main()
