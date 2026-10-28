"""Isolated causal healthy-only study with LDM-UPB removed everywhere.

The original 2026-09-08 study is intentionally not imported as a runner and
its frozen output is never written.  Manifest metadata is filtered before any
selected device cache is opened.  H4 and H12 models are retrained from the
selected clients, so no model trained on the original 70-client cohort is
reused.
"""
from __future__ import annotations
import csv, hashlib, json, os, pickle, sys
from pathlib import Path
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = HERE.parent / "healthy_only_suite_20260907"
ORIGINAL = HERE.parent / "healthy_only_causal_20260908"
OUT = ROOT / "results/healthy_only_causal_no_ldm_20260922"
CACHE = ROOT / "results/healthy_only_suite_20260907/cache"
OLD = ROOT / "results/healthy_only_suite_20260907"
PROTOCOL = HERE / "protocol.json"
EXCLUDED_DATASET = "ldm_upb"
RULES = ("global_h4", "global_h12", "personalized_h12", "svdd_h12", "pca", "feature_envelope",
         "pca_and_global_h4", "pca_and_global_h12", "pca_and_personalized_h12", "pca_and_svdd_h12",
         "pca_and_global_h4_and_envelope", "pca_and_global_h12_and_envelope",
         "pca_and_personalized_h12_and_envelope", "pca_and_svdd_h12_and_envelope")

sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(SUITE))
import wider_models as wider  # noqa: E402
import cadence  # noqa: E402
import run_suite as old_suite  # noqa: E402
import models as old_models  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_devices(manifest):
    """Select non-LDM metadata before any device cache is loaded."""
    devices = sorted((d for d in manifest["devices"] if d.get("dataset") != EXCLUDED_DATASET),
                     key=lambda d: d["machine_id"])
    if any(d.get("dataset") == EXCLUDED_DATASET or d["machine_id"].startswith("ldm_upb::") for d in devices):
        raise RuntimeError("LDM-UPB device survived exclusion")
    if len(devices) != 54 or len({d["machine_id"] for d in devices}) != 54:
        raise RuntimeError("non-LDM cohort identity count mismatch")
    faults = sum(d.get("truth") is not None for d in devices)
    controls = sum(d.get("truth_kind") == "healthy_control" for d in devices)
    unknown = sum(d.get("truth") is None and d.get("truth_kind") != "healthy_control" for d in devices)
    if (faults, controls, unknown) != (43, 6, 5):
        raise RuntimeError("non-LDM cohort scored/fault/control/unknown counts mismatch")
    return devices


def verify_selected_cache_integrity(manifest, devices):
    """Verify cache provenance and only the selected non-LDM cache payloads."""
    for path, expected in manifest.get("config", {}).get("provenance_sha256", {}).items():
        if sha(path) != expected:
            raise RuntimeError(f"cache provenance drift: {path}")
    expected_files = {str(Path(path).resolve()): digest for path, digest in manifest["files"].items()}
    selected = {str(Path(d["path"]).resolve()) for d in devices}
    if len(selected) != 54 or any("ldm_upb" in path.lower() for path in selected):
        raise RuntimeError("selected cache-file set is not the fixed non-LDM cohort")
    missing = selected - set(expected_files)
    if missing:
        raise RuntimeError(f"selected cache files absent from manifest: {sorted(missing)}")
    for path in sorted(selected):
        if sha(path) != expected_files[path]:
            raise RuntimeError(f"selected cache hash drift: {path}")


def _selected_paths(manifest, devices):
    """Dependency paths include only selected cache files, never LDM cache files."""
    selected = {str(Path(d["path"]).resolve()) for d in devices}
    if any("ldm_upb" in p.lower() for p in selected):
        raise RuntimeError("LDM path entered selected dependency list")
    paths = [PROTOCOL, HERE / "run_study.py", HERE / "test_study.py",
             ORIGINAL / "wider_models.py", ORIGINAL / "cadence.py",
             SUITE / "run_suite.py", SUITE / "models.py", SUITE / "calibration.py", CACHE / "manifest.json"]
    paths += [Path(p) for p in manifest.get("config", {}).get("provenance_sha256", {})]
    paths += [Path(p) for p in sorted(selected)]
    return paths


def combine(flags, rule):
    if rule in ("global_h4", "global_h12", "personalized_h12", "svdd_h12", "pca", "feature_envelope"):
        return np.asarray(flags[rule], bool)
    base = rule.removeprefix("pca_and_")
    env = base.endswith("_and_envelope")
    base = base.removesuffix("_and_envelope")
    out = np.asarray(flags["pca"], bool) & np.asarray(flags[base], bool)
    return out & np.asarray(flags["feature_envelope"], bool) if env else out


def evaluate(meta, arrays, flags, metadata=None):
    h, f, u = [np.asarray(arrays[k], dtype=int) for k in
               ("healthy_test_indices", "unhealthy_test_indices", "unscored_test_indices")]
    flags = np.asarray(flags, bool)
    ha = bool(np.any(flags[h])); fh = np.flatnonzero(flags[f]); first = np.flatnonzero(flags)
    expected = "fault" if meta["truth"] is not None else ("healthy_control" if meta["truth_kind"] == "healthy_control" else "unknown")
    return {"machine_id": meta["machine_id"], "dataset": meta["dataset"], "expected": expected,
            "scored": expected != "unknown", "strict_success": expected != "unknown" and not ha and (bool(len(fh)) or expected == "healthy_control"),
            "healthy_alarm": ha, "fault_detected": bool(len(fh)),
            "fault_delay_acquisitions": int(fh[0]) if len(fh) else None,
            "first_alarm_index": int(first[0]) if len(first) else None,
            "unknown_alarm": bool(np.any(flags[u])), "training_health_verified": meta["training_health_verified"],
            "cadence_metadata": metadata or {}}


def summarize(rows):
    scored = [r for r in rows if r["scored"]]
    faults = [r for r in scored if r["expected"] == "fault"]
    controls = [r for r in scored if r["expected"] == "healthy_control"]
    delays = [r["fault_delay_acquisitions"] for r in faults if r["fault_detected"]]
    n = len(scored); strict = sum(r["strict_success"] for r in scored)
    return {"strict_successes": strict, "scored": n, "strict_percent": 100 * strict / max(1, n),
            "faults_detected": sum(r["fault_detected"] for r in faults), "faults": len(faults),
            "healthy_alarm_devices": sum(r["healthy_alarm"] for r in scored),
            "control_alarms": sum(r["healthy_alarm"] for r in controls), "controls": len(controls),
            "fault_delay_median": float(np.median(delays)) if delays else None}


def _state(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _write_state(path, state):
    with Path(path).open("wb") as handle:
        pickle.dump(state, handle, protocol=5)


def run():
    spec = read(PROTOCOL)
    manifest = read(CACHE / "manifest.json")
    devices = select_devices(manifest)  # must precede every np.load below
    verify_selected_cache_integrity(manifest, devices)
    paths = _selected_paths(manifest, devices)
    deps = {str(Path(p).resolve()): sha(p) for p in paths}
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "study_lock.json"
    if lock.exists() and read(lock)["dependencies"] != deps:
        raise RuntimeError("dependency drift; refusing resume")
    if not lock.exists():
        lock.write_text(json.dumps({"dependencies": deps, "classification": spec["classification"]}, indent=2) + "\n", encoding="utf-8")

    arrays, clients = {}, {}
    for meta in devices:
        # This is the first cache load, after select_devices has proven no LDM rows remain.
        with np.load(meta["path"], allow_pickle=False) as data:
            arrays[meta["machine_id"]] = {k: data[k].copy() for k in data.files}
        clients[meta["machine_id"]] = arrays[meta["machine_id"]]["x"][arrays[meta["machine_id"]]["train_indices"].astype(int)]
    if any(mid.startswith("ldm_upb::") for mid in clients):
        raise RuntimeError("LDM-UPB entered training clients")
    if len(clients) != 54:
        raise RuntimeError("training client count mismatch")

    states = {}
    variants = ("global_h4",) + tuple(spec["new_models"])
    for seed in spec["seeds"]:
        for variant in variants:
            p = OUT / f"{variant}_seed{seed}.pkl"
            cp = OUT / f"{variant}_seed{seed}_checkpoint.json"
            if p.exists():
                if not cp.exists() or read(cp).get("input_dependencies") != deps:
                    raise RuntimeError(f"state checkpoint drift {variant} seed{seed}")
                if read(cp).get("state_sha256") != sha(p):
                    raise RuntimeError(f"saved model hash drift {variant} seed{seed}")
                states[(variant, seed)] = _state(p)
            else:
                states[(variant, seed)] = old_models.fit(clients, "global_ae", seed) if variant == "global_h4" else wider.fit(clients, variant, seed)
                if variant == "global_h4":
                    states[(variant, seed)]["protocol"] = spec
                _write_state(p, states[(variant, seed)])
            comm = {"training": "fresh non-LDM global H4"} if variant == "global_h4" else wider.communication(states[(variant, seed)])
            checkpoint = {"variant": variant, "seed": int(seed), "clients": len(clients), "input_dependencies": deps,
                          "state_sha256": sha(p), "communication": comm,
                          "mean_training_reconstruction_mse": float(np.mean([np.mean(old_models.reconstruction_scores(states[(variant, seed)], mid, clients[mid])) if variant == "global_h4" else np.mean(wider.reconstruction_scores(states[(variant, seed)], mid, clients[mid])) for mid in clients]))}
            cp.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
            print(f"ready {variant} seed{seed}", flush=True)

    artifact_manifest = {str(OUT / f"{v}_seed{s}.pkl"): sha(OUT / f"{v}_seed{s}.pkl") for s in spec["seeds"] for v in variants}
    artifact_manifest.update({str(OUT / f"{v}_seed{s}_checkpoint.json"): sha(OUT / f"{v}_seed{s}_checkpoint.json") for s in spec["seeds"] for v in variants})
    rows = []
    for seed in spec["seeds"]:
        for meta in devices:
            mid = meta["machine_id"]; a = arrays[mid]; x = a["x"]; ti = a["train_indices"].astype(int); ci = a["calibration_indices"].astype(int)
            scores = {"global_h4": old_models.score(states[("global_h4", seed)], mid, x),
                      "global_h12": wider.score(states[("global_h12", seed)], mid, x),
                      "personalized_h12": wider.score(states[("personalized_h12", seed)], mid, x),
                      "svdd_h12": wider.score(states[("svdd_h12", seed)], mid, x),
                      "pca": old_suite.reference_scores("local_pca", x, x[ti], None)}
            scores["feature_envelope"] = np.max(x / np.maximum(np.max(x[np.concatenate((ti, ci))], axis=0), 1e-8), axis=1)
            if any(not np.all(np.isfinite(v)) for v in scores.values()):
                raise RuntimeError(f"nonfinite scores {mid}")
            for mode in ("dense", "history_phase"):
                flag, cadmeta = {}, {}
                for key, value in scores.items():
                    mult = 1.25 if key == "feature_envelope" else 2.0
                    if mode == "dense":
                        flag[key] = cadence.dense_flags(value, ti, ci, 21, 13, mult); cadmeta[key] = {"stride": 1, "decision_count": len(value)}
                    else:
                        q = cadence.phase_vote(value, ti, ci, cadence.stride_from_history(ti, ci, 100), 21, 13, mult)
                        flag[key] = q["flags"]; cadmeta[key] = q["metadata"]; cadmeta[key]["decision_count"] = int(sum(q["metadata"]["phase_decisioncounts"]))
                for rule in RULES:
                    rows.append({"rule": rule, "cadence": mode, "seed": seed, **evaluate(meta, a, combine(flag, rule), {"mode": mode, "routes": cadmeta})})
        print(f"completed seed {seed}", flush=True)

    flat = []
    for seed in spec["seeds"]:
        for mode in ("dense", "history_phase"):
            for rule in RULES:
                rr = [r for r in rows if r["seed"] == seed and r["cadence"] == mode and r["rule"] == rule]
                if len(rr) != 54 or sum(r["scored"] for r in rr) != 49 or sum(r["expected"] == "fault" for r in rr) != 43 or sum(r["expected"] == "healthy_control" for r in rr) != 6:
                    raise RuntimeError(f"config cohort mismatch {seed} {mode} {rule}")
                if any(r["dataset"] == EXCLUDED_DATASET for r in rr):
                    raise RuntimeError("LDM-UPB result survived scoring exclusion")
                flat.append({"rule": rule, "cadence": mode, "seed": seed, "metrics": summarize(rr),
                             "per_dataset": {d: summarize([r for r in rr if r["dataset"] == d]) for d in sorted({r["dataset"] for r in rr})}, "devices": rr})
    if len(flat) != 84:
        raise RuntimeError("configuration count mismatch")
    primary = [summarize([r for r in rows if r["seed"] == s and r["rule"] == spec["primary"]["rule"] and r["cadence"] == spec["primary"]["cadence"]]) for s in spec["seeds"]]
    goal = {"passes": all(x["strict_successes"] >= 45 and x["faults_detected"] >= 41 and x["control_alarms"] == 0 for x in primary), "primary": spec["primary"], "per_seed": primary}
    goal["all_fixed_pairs"] = [{"rule": rule, "cadence": mode, "passes": all(c["metrics"]["strict_successes"] >= 45 and c["metrics"]["faults_detected"] >= 41 and c["metrics"]["control_alarms"] == 0 for c in flat if c["rule"] == rule and c["cadence"] == mode)} for rule in RULES for mode in ("dense", "history_phase")]
    summary = {"complete": True, "classification": spec["classification"], "excluded_dataset": EXCLUDED_DATASET,
               "cohort": spec["cohort"], "configuration_count": len(flat), "dependencies": deps,
               "artifact_hashes": artifact_manifest, "goal": goal, "results": flat}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rule", "cadence", "seed"] + list(flat[0]["metrics"])); writer.writeheader()
        for item in flat: writer.writerow({"rule": item["rule"], "cadence": item["cadence"], "seed": item["seed"], **item["metrics"]})
    report = ["# Causal cadence and wider representation study without LDM-UPB", "", spec["classification"], "", "| Seed | Primary strict | Detected | Healthy alarms | Controls |", "|---:|---:|---:|---:|---:|"]
    report += [f"| {seed} | {metric['strict_successes']}/49 | {metric['faults_detected']}/43 | {metric['healthy_alarm_devices']} | {metric['control_alarms']}/6 |" for seed, metric in zip(spec["seeds"], primary)]
    report += ["", f"Goal gate: {'PASS' if goal['passes'] else 'FAIL'} (fixed primary across all three seeds).", "All 84 configurations are retained in summary.json; no winner is selected.", "", "LDM-UPB was excluded before np.load and is absent from training, calibration, scoring, and every dependency cache-file list."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    run()
