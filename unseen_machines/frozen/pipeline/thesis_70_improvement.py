"""Healthy-only correction and fixed calibration ablations for 70-client scaling."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np

import raw_online_federated_v64_baselines as raw
import thesis_70_device_federated as old
import thesis_70_node_scaling as scaling
import thesis_participating_node_scaling as trainer

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "thesis_70_improvement_protocol.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_array(values):
    import hashlib
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def load_raw_devices():
    devices = raw.load_devices()
    transitions = read_json(old.TRANSITIONS)["transitions"]
    for machine in old.ADDED:
        path = old.LDM_ROOT / "features" / f"ldm_upb__{machine}.npz"
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float64)
        devices.append({"machine_id": f"ldm_upb::{machine}", "tier": "ldm_upb_complete_lifecycle",
                        "truth_kind": "published_transition", "truth": int(transitions[machine]),
                        "n": len(features), "feature_streams": [features], "feature_paths": [path]})
    assert len(devices) == len({d["machine_id"] for d in devices}) == 70
    return devices


def physical_matrix(streams, training_indices, indices):
    """No truth, test-derived statistics, or learned upstream model input."""
    output = []
    for stream in streams:
        values = np.asarray(stream, dtype=np.float64)[:, indices]
        fit = values[np.asarray(training_indices)]
        center = np.median(fit, axis=0)
        mad = 1.4826 * np.median(np.abs(fit - center), axis=0)
        scale = np.maximum(mad, np.std(fit, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        output.append(np.abs(np.arcsinh((values - center) / scale)))
    matrix = np.max(np.stack(output), axis=0)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite physical features")
    return matrix


def prepare(devices, training_protocol, protocol):
    # The established helper supplies indices only. Its normalized dummy values
    # are discarded, so its calibration-fitted transform is not reused.
    dummy = {d["machine_id"]: np.zeros((d["n"], 22)) for d in devices}
    layout, audit = old.prepare_partitions(devices, dummy, training_protocol)
    matrices, partitions = {}, {}
    fields = ("train", "calibration", "healthy_test", "unhealthy_test", "unscored_test")
    for device in devices:
        key = device["machine_id"]
        part = layout[key]
        values = physical_matrix(device["feature_streams"], part.train_indices,
                                 protocol["features"]["raw_feature_indices"])
        matrices[key] = values
        partitions[key] = replace(part, **{field: values[list(getattr(part, field + "_indices"))]
                                           for field in fields})
    for row in audit:
        p = partitions[row["machine_id"]]
        row["training_transform_rows"] = len(p.train_indices)
        row["training_health_verified"] = p.truth is not None or p.truth_kind == "healthy_control"
        row["commissioning_basis"] = "known_healthy_partition" if row["training_health_verified"] else "assumed_first_half_unverified"
        row["train_fault_rows"] = 0 if row["training_health_verified"] else "unknown"
        row["calibration_fault_rows"] = 0 if row["training_health_verified"] else "unknown"
    return partitions, matrices, audit


def event_statistic(scores, k, window):
    scores = np.asarray(scores, dtype=float)
    if not 1 <= k <= window:
        raise ValueError("invalid persistence")
    result = np.full(len(scores), -np.inf)
    if len(scores) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(scores, window)
        result[window - 1:] = np.partition(windows, window - k, axis=1)[:, window - k]
    return result


def threshold_from_calibration(scores, arm, protocol):
    values = np.asarray(scores, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("invalid calibration scores")
    if arm["threshold"] == "point_quantile":
        return float(np.quantile(values, protocol["threshold"]["point_quantile"]))
    statistic = event_statistic(values, arm["k"], arm["window"])
    statistic = statistic[np.isfinite(statistic)]
    if not len(statistic):
        statistic = values
    spread = 1.4826 * np.median(np.abs(statistic - np.median(statistic)))
    return float(np.max(statistic) + protocol["threshold"]["spread_multiplier"] * spread)


def evaluate_scores(scores, part, arm, protocol):
    cal_idx = np.asarray(part.calibration_indices)
    threshold = threshold_from_calibration(scores[cal_idx], arm, protocol)
    # Prediction runs continuously. The boundary uses commissioning metadata;
    # truth is consulted only to classify the completed prediction array.
    start = int(cal_idx[0])
    statistic = event_statistic(scores[start:], arm["k"], arm["window"])
    flags = np.zeros(len(scores), dtype=bool)
    flags[start:] = statistic > threshold
    flags[:int(cal_idx[-1]) + 1] = False
    healthy = np.asarray(part.healthy_test_indices, dtype=int)
    faulty = np.asarray(part.unhealthy_test_indices, dtype=int)
    unknown = np.asarray(part.unscored_test_indices, dtype=int)
    first = np.flatnonzero(flags)
    healthy_hits, fault_hits = np.flatnonzero(flags[healthy]), np.flatnonzero(flags[faulty])
    h_alarm, f_alarm = bool(len(healthy_hits)), bool(len(fault_hits))
    scored = part.truth is not None or part.truth_kind == "healthy_control"
    expected = "fault" if part.truth is not None else "healthy_control" if scored else "outcome_unavailable"
    return {"machine_id": part.machine_id, "dataset": part.dataset, "truth_kind": part.truth_kind,
            "expected": expected, "scored": scored, "machine_success": scored and not h_alarm and (f_alarm or expected == "healthy_control"),
            "threshold": threshold, "healthy_persistent_alarm": h_alarm, "fault_detected": f_alarm,
            "first_alarm_index": int(first[0]) if len(first) else "",
            "first_unhealthy_persistent_alarm_delay": int(fault_hits[0]) if f_alarm else "",
            "healthy_test_rows": len(healthy), "unhealthy_test_rows": len(faulty),
            "healthy_window_specificity": float(np.mean(scores[healthy] <= threshold)),
            "unhealthy_window_recall": float(np.mean(scores[faulty] > threshold)) if len(faulty) else "",
            "unscored_test_rows": len(unknown), "unscored_persistent_alarm": bool(np.any(flags[unknown])),
            "training_health_verified": part.truth is not None or part.truth_kind == "healthy_control"}


def dependencies():
    paths = [PROTOCOL, Path(__file__), old.PROTOCOL, scaling.PROTOCOL,
             Path(old.__file__), Path(trainer.__file__), Path(scaling.__file__), Path(raw.__file__)]
    return {str(p.resolve()): old.sha256(p) for p in paths}


def save_model(path, model):
    np.savez_compressed(path, **{name: value for name, value in zip(("w1", "b1", "w2", "b2"), model.arrays())})


def metrics(rows, clients, n, subset_id, design):
    result = scaling.configuration_metrics(rows, clients, n, subset_id, design)
    controls = [r for r in rows if r["expected"] == "healthy_control"]
    faults = [r for r in rows if r["expected"] == "fault"]
    scored = [r for r in rows if r["scored"]]
    result.update(healthy_control_alarms=sum(r["healthy_persistent_alarm"] for r in controls),
                  healthy_controls=len(controls),
                  all_healthy_alarm_devices=sum(r["healthy_persistent_alarm"] for r in scored),
                  fault_delay_median=float(np.median([r["first_unhealthy_persistent_alarm_delay"] for r in faults if r["fault_detected"]])))
    return result


def run(phase, run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("bad run id")
    protocol, train_spec, scale_spec = read_json(PROTOCOL), read_json(old.PROTOCOL), read_json(scaling.PROTOCOL)
    root = HERE / "results" / f"thesis_70_improvement_{run_id}"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "execution_lock.json"
    deps = dependencies()
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock["dependency_sha256"] != deps:
            raise ValueError("dependency drift")
    else:
        lock = {"classification": protocol["classification"], "dependency_sha256": deps, "complete_endpoint": False, "complete_scaling": False}
        old.write_json(lock_path, lock)
    if lock.get("complete_" + phase):
        raise ValueError("phase already complete")
    print("Loading existing raw feature caches; no downloads", flush=True)
    devices = load_raw_devices()
    partitions, matrices, audit = prepare(devices, train_spec, protocol)
    clients = tuple(sorted(partitions))
    input_hashes = {d["machine_id"]: {"streams": [digest_array(x) for x in d["feature_streams"]],
                                    "truth": d["truth"], "truth_kind": d["truth_kind"], "n": d["n"]}
                    for d in devices}
    if "input_arrays" in lock and lock["input_arrays"] != input_hashes:
        raise ValueError("raw input drift")
    lock["input_arrays"] = input_hashes
    old.write_json(lock_path, lock)
    old.write_csv(root / "partition_audit.csv", audit)
    if phase == "endpoint":
        print("Training corrected 70-client FedAdam model, 12 rounds", flush=True)
        model = trainer.fit_federated(partitions, clients, "fedadam", train_spec)
        save_model(root / "corrected_model.npz", model)
        scores = {key: model.scores(matrices[key]) for key in clients}
        summaries, all_rows = [], []
        for arm in protocol["arms"]:
            rows = [evaluate_scores(scores[key], partitions[key], arm, protocol) for key in clients]
            summary = metrics(rows, clients, 70, arm["name"], "fixed_ablation")
            summary["primary"] = arm["name"] == protocol["primary_arm"]
            summaries.append(summary)
            all_rows.extend({"arm": arm["name"], **r} for r in rows)
            print(f"{arm['name']}: success={summary['all_strict_successes']}/65 detection={summary['all_fault_detections']}/59 healthy alarms={summary['all_healthy_alarm_devices']}/65 controls={summary['healthy_control_alarms']}/6", flush=True)
        old.write_csv(root / "endpoint_ablation_metrics.csv", summaries)
        old.write_csv(root / "endpoint_device_results.csv", all_rows)
        old.write_json(root / "endpoint_summary.json", summaries)
        lock["complete_endpoint"] = True
    else:
        arm = next(a for a in protocol["arms"] if a["name"] == protocol["primary_arm"])
        design = scaling.subset_design(clients, scale_spec)
        config_rows, device_rows = [], []
        checkpoint_root = root / "checkpoints"
        for i, subset in enumerate(design, 1):
            path = checkpoint_root / (subset["subset_id"] + ".json")
            if path.exists():
                payload = read_json(path)
                if payload["dependencies"] != deps or payload["clients"] != list(subset["clients"]):
                    raise ValueError("checkpoint drift")
            else:
                model = trainer.fit_federated(partitions, subset["clients"], "fedadam", train_spec)
                rows = [evaluate_scores(model.scores(matrices[key]), partitions[key], arm, protocol) for key in clients]
                payload = {"dependencies": deps, "clients": list(subset["clients"]), "devices": rows,
                           "configuration": metrics(rows, subset["clients"], subset["logical_clients"], subset["subset_id"], subset["design"])}
                old.write_json(path, payload)
            config_rows.append(payload["configuration"])
            device_rows.extend({"logical_clients": subset["logical_clients"], "subset_id": subset["subset_id"],
                                "training_client": r["machine_id"] in subset["clients"], **r} for r in payload["devices"])
            if i == 1 or i % 100 == 0 or i == len(design):
                print(f"Completed {i}/{len(design)} configurations", flush=True)
        summary = scaling.summarize(config_rows, scale_spec)
        old.write_csv(root / "configuration_metrics.csv", config_rows)
        old.write_csv(root / "device_results.csv", device_rows)
        old.write_csv(root / "node_scaling_summary.csv", summary)
        lock.update(complete_scaling=True, configurations=len(config_rows), device_rows=len(device_rows))
    lock["output_sha256"] = {p.name: old.sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "execution_lock.json"}
    old.write_json(lock_path, lock)


def verify(run_id):
    root = HERE / "results" / f"thesis_70_improvement_{run_id}"
    lock = read_json(root / "execution_lock.json")
    if lock["dependency_sha256"] != dependencies():
        raise ValueError("dependency drift")
    for name, digest in lock["output_sha256"].items():
        if old.sha256(root / name) != digest:
            raise ValueError(f"output drift: {name}")
    if lock.get("complete_scaling"):
        with (root / "configuration_metrics.csv").open(newline="", encoding="utf-8") as f:
            configs = list(csv.DictReader(f))
        with (root / "device_results.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(configs) == 3571 and len(rows) == 3571 * 70
        assert {int(r["logical_clients"]) for r in configs} == set(range(2, 71, 2))
        counts = Counter(r["subset_id"] for r in rows)
        assert set(counts) == {c["subset_id"] for c in configs} and set(counts.values()) == {70}
    print(json.dumps({"hashes_pass": True, "endpoint_complete": lock["complete_endpoint"], "scaling_complete": lock["complete_scaling"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("endpoint", "scaling", "verify"), required=True)
    parser.add_argument("--run-id", default="dev1")
    args = parser.parse_args()
    verify(args.run_id) if args.phase == "verify" else run(args.phase, args.run_id)
