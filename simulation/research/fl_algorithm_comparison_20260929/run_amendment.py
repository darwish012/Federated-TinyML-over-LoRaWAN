"""Amendment 1: round-budget descriptive study (Part A) and depth-on-ldm_upb
falsification test (Part B). See amendment1.json for the specification.

Reuses fl.py (unedited) and the LDM-free study's noldm.select_devices /
verify_selected_cache_integrity / evaluate / summarize exactly as
run_study.py does, plus cadence.dense_flags. Does not edit fl.py,
run_study.py, or any file outside this folder / the amendment1 results
folder.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, pickle, sys, time
from pathlib import Path
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOLDM = HERE.parent / "healthy_only_causal_no_ldm_20260922"
SAVED_B = ROOT / "results/healthy_only_causal_20260908"
CACHE = ROOT / "results/healthy_only_suite_20260907/cache"
OUT = ROOT / "results/fl_algorithm_comparison_20260929/amendment1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_causal_20260908"))
import fl  # noqa: E402
import cadence  # noqa: E402

_spec = importlib.util.spec_from_file_location("noldm_study", NOLDM / "run_study.py")
noldm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(noldm)

ARCH = {"h12": [22, 12, 22], "deep": [22, 16, 8, 16, 22]}
SEEDS = [17070, 17071, 17072]
ARMS = [("h12", "fedavg"), ("h12", "fedadam"), ("deep", "fedavg"), ("deep", "fedadam")]
BUDGETS = [4, 6, 8, 10, 12, 15]
EPS_LIST = [0.01, 0.02, 0.05]


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def evaluate_params(params, devices, arrays):
    """Same pattern as run_study.py's evaluate_params: score -> dense_flags -> noldm.evaluate/summarize."""
    rows = []
    for meta in devices:
        a = arrays[meta["machine_id"]]
        x = a["x"]; ti = a["train_indices"].astype(int); ci = a["calibration_indices"].astype(int)
        s = fl.score(params, x)
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"nonfinite score {meta['machine_id']}")
        flags = cadence.dense_flags(s, ti, ci, 21, 13, 2.0)
        rows.append(noldm.evaluate(meta, a, flags))
    return noldm.summarize(rows), rows


# --------------------------------------------------------------------------
# PART A: LDM-free 54-client cohort, round-budget study
# --------------------------------------------------------------------------
def part_a():
    print("=== PART A: LDM-free 54-client round-budget study ===", flush=True)
    manifest = read(CACHE / "manifest.json")
    devices = noldm.select_devices(manifest)            # before any np.load
    noldm.verify_selected_cache_integrity(manifest, devices)

    arrays, train, calib = {}, {}, {}
    for meta in devices:
        with np.load(meta["path"], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        mid = meta["machine_id"]
        arrays[mid] = a
        train[mid] = a["x"][a["train_indices"].astype(int)]
        calib[mid] = a["x"][a["calibration_indices"].astype(int)]
    assert len(train) == 54 and not any(m.startswith("ldm_upb::") for m in train)

    rows_out = []
    consistency = {}

    for arch, alg in ARMS:
        layers = ARCH[arch]
        frag = fl.fragments(layers)
        for seed in SEEDS:
            t0 = time.time()
            s = fl.Server(alg, fl.init(layers, seed))
            losses, history = [], []
            for r in range(1, 16):
                p = s.round(train)
                history.append(p)
                L = float(np.mean([np.mean(fl.score(p, calib[m])) for m in calib]))
                if not np.isfinite(L):
                    raise RuntimeError(f"diverged {arch} {alg} seed {seed} round {r}")
                losses.append(L)

            for b in BUDGETS:
                metrics, _ = evaluate_params(history[b - 1], devices, arrays)
                rows_out.append({
                    "part": "A", "arch": arch, "algorithm": alg, "seed": seed, "rule": f"budget_{b}",
                    "epsilon": None, "round": b, "stop_reason": None,
                    "fragments_per_client": 2 * frag * b,
                    "strict_successes": metrics["strict_successes"], "scored": metrics["scored"],
                    "faults_detected": metrics["faults_detected"], "faults": metrics["faults"],
                    "healthy_alarm_devices": metrics["healthy_alarm_devices"],
                    "control_alarms": metrics["control_alarms"], "controls": metrics["controls"],
                })
                if arch == "h12" and alg == "fedadam" and b == 12:
                    consistency.setdefault("h12_fedadam_budget12", {})[seed] = metrics["strict_successes"]

            for eps in EPS_LIST:
                t_stop, why = fl.stop_round(losses, eps, 3, 15)
                metrics, _ = evaluate_params(history[t_stop - 1], devices, arrays)
                rows_out.append({
                    "part": "A", "arch": arch, "algorithm": alg, "seed": seed, "rule": f"capped_eps{eps}",
                    "epsilon": eps, "round": t_stop, "stop_reason": why,
                    "fragments_per_client": 2 * frag * t_stop,
                    "strict_successes": metrics["strict_successes"], "scored": metrics["scored"],
                    "faults_detected": metrics["faults_detected"], "faults": metrics["faults"],
                    "healthy_alarm_devices": metrics["healthy_alarm_devices"],
                    "control_alarms": metrics["control_alarms"], "controls": metrics["controls"],
                })

            print(f"A seed {seed} {arch}_{alg}: budgets {BUDGETS} + capped eps{EPS_LIST} done "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    expect = {17070: 45, 17071: 44, 17072: 44}
    got = consistency.get("h12_fedadam_budget12", {})
    for seed, exp in expect.items():
        if got.get(seed) != exp:
            raise SystemExit(
                f"CONSISTENCY CHECK FAILED: (h12, fedadam) @ budget 12, seed {seed}: "
                f"strict={got.get(seed)} != expected {exp} (primary-study reproduction). Aborting.")
    print(f"Part A consistency check PASSED: h12/fedadam @ budget 12 -> "
          f"{[got[s] for s in SEEDS]} (expected [45, 44, 44])", flush=True)

    return rows_out


# --------------------------------------------------------------------------
# PART B: full 70-client cohort including ldm_upb
# --------------------------------------------------------------------------
def part_b():
    print("\n=== PART B: full 70-client cohort including ldm_upb ===", flush=True)
    manifest = read(CACHE / "manifest.json")
    devices_all = sorted(manifest["devices"], key=lambda d: d["machine_id"])
    ids = [d["machine_id"] for d in devices_all]
    if len(devices_all) != 70 or len(set(ids)) != 70:
        raise SystemExit("70-device count/uniqueness mismatch")
    faults = sum(d.get("truth") is not None for d in devices_all)
    controls = sum(d.get("truth_kind") == "healthy_control" for d in devices_all)
    unknown = sum(d.get("truth") is None and d.get("truth_kind") != "healthy_control" for d in devices_all)
    if (faults, controls, unknown) != (59, 6, 5):
        raise SystemExit(f"70-cohort truth-count mismatch: faults={faults} controls={controls} unknown={unknown}")

    # verify each device file's sha256 against manifest['files'] before loading
    expected_files = {str(Path(p).resolve()): digest for p, digest in manifest["files"].items()}
    for meta in devices_all:
        rp = str(Path(meta["path"]).resolve())
        if rp not in expected_files:
            raise SystemExit(f"device file missing from manifest['files']: {rp}")
        if sha(meta["path"]) != expected_files[rp]:
            raise SystemExit(f"cache hash drift: {rp}")
    print(f"verified sha256 for all {len(devices_all)} device files against manifest['files']", flush=True)

    arrays_all, train_all = {}, {}
    for meta in devices_all:            # sorted machine_id order -> dict insertion order fixed
        with np.load(meta["path"], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        mid = meta["machine_id"]
        arrays_all[mid] = a
        train_all[mid] = a["x"][a["train_indices"].astype(int)]
    if len(train_all) != 70:
        raise SystemExit("70-client training dict count mismatch")

    ldm_devices = [d for d in devices_all if d["dataset"] == "ldm_upb"]
    nonldm_devices = [d for d in devices_all if d["dataset"] != "ldm_upb"]
    if len(ldm_devices) != 16 or len(nonldm_devices) != 54:
        raise SystemExit(f"ldm_upb/non-LDM split mismatch: ldm={len(ldm_devices)} nonldm={len(nonldm_devices)}")

    def eval_all(params):
        metrics, rows = evaluate_params(params, devices_all, arrays_all)
        ldm_rows = [r for r in rows if r["dataset"] == "ldm_upb"]
        nonldm_rows = [r for r in rows if r["dataset"] != "ldm_upb"]
        return metrics, noldm.summarize(ldm_rows), noldm.summarize(nonldm_rows), rows

    # ---------------- REPLICATION GATE (must pass before any new outcome) ----------------
    gate = []
    expect_overall = {17070: 52, 17071: 51, 17072: 51}
    for seed in SEEDS:
        s = fl.Server("fedadam", fl.init(ARCH["h12"], seed))
        for _ in range(12):
            s.round(train_all)
        with (SAVED_B / f"global_h12_seed{seed}.pkl").open("rb") as f:
            ref = pickle.load(f)["global"]
        maxdiff = max(float(np.max(np.abs(np.asarray(u) - np.asarray(v)))) for u, v in zip(s.p, ref))
        metrics, ldm_metrics, nonldm_metrics, _ = eval_all(s.p)
        ok = (maxdiff <= 1e-10 and metrics["strict_successes"] == expect_overall[seed]
              and ldm_metrics["strict_successes"] == 7)
        gate.append({
            "seed": seed, "max_abs_weight_diff": maxdiff,
            "strict_overall": metrics["strict_successes"], "scored_overall": metrics["scored"],
            "expected_overall": expect_overall[seed],
            "strict_ldm_upb": ldm_metrics["strict_successes"], "scored_ldm_upb": ldm_metrics["scored"],
            "expected_ldm_upb": 7, "pass": ok,
        })
        print(f"gate-B seed {seed}: maxdiff {maxdiff:.2e}  strict {metrics['strict_successes']}/65 "
              f"(expect {expect_overall[seed]})  ldm_upb {ldm_metrics['strict_successes']}/16 (expect 7)  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    if not all(g["pass"] for g in gate):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "GATE_B_FAILED.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print("REPLICATION GATE B FAILED - Part B aborted, nothing else written for Part B", flush=True)
        return {"gate": gate, "passed": False}, [], []

    # ---------------- Part B arms at exactly 12 rounds ----------------
    rows_out = []
    ldm_device_rows = []
    for arch, alg in ARMS:
        layers = ARCH[arch]
        frag = fl.fragments(layers)
        for seed in SEEDS:
            t0 = time.time()
            s = fl.Server(alg, fl.init(layers, seed))
            p = None
            for _ in range(12):
                p = s.round(train_all)
            metrics, ldm_metrics, nonldm_metrics, rows = eval_all(p)
            rows_out.append({
                "part": "B", "arch": arch, "algorithm": alg, "seed": seed, "round": 12,
                "fragments_per_client": 2 * frag * 12,
                "strict_overall": metrics["strict_successes"], "scored_overall": metrics["scored"],
                "strict_ldm_upb": ldm_metrics["strict_successes"], "scored_ldm_upb": ldm_metrics["scored"],
                "strict_nonldm": nonldm_metrics["strict_successes"], "scored_nonldm": nonldm_metrics["scored"],
                "faults_detected": metrics["faults_detected"], "faults": metrics["faults"],
                "healthy_alarm_devices": metrics["healthy_alarm_devices"],
                "control_alarms": metrics["control_alarms"], "controls": metrics["controls"],
            })
            for r in rows:
                if r["dataset"] == "ldm_upb":
                    ldm_device_rows.append({
                        "arch": arch, "algorithm": alg, "seed": seed, "machine_id": r["machine_id"],
                        "strict_success": r["strict_success"], "healthy_alarm": r["healthy_alarm"],
                        "fault_detected": r["fault_detected"],
                    })
            print(f"B seed {seed} {arch}_{alg}: overall {metrics['strict_successes']}/65  "
                  f"ldm_upb {ldm_metrics['strict_successes']}/16  nonldm {nonldm_metrics['strict_successes']}/49  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    # ---------------- decision rule from amendment1.json part_B, applied literally ----------------
    def per_seed(arch, alg, key):
        return [next(r for r in rows_out if r["arch"] == arch and r["algorithm"] == alg and r["seed"] == s)[key]
                for s in SEEDS]

    decisions = {}
    for alg in ("fedavg", "fedadam"):
        deep_ldm = per_seed("deep", alg, "strict_ldm_upb")
        h12_ldm = per_seed("h12", alg, "strict_ldm_upb")
        deep_nonldm = per_seed("deep", alg, "strict_nonldm")
        h12_nonldm = per_seed("h12", alg, "strict_nonldm")
        ge_all = all(d >= h for d, h in zip(deep_ldm, h12_ldm))
        gt_two = sum(d > h for d, h in zip(deep_ldm, h12_ldm)) >= 2
        no_big_loss = all((h - d) <= 1 for d, h in zip(deep_nonldm, h12_nonldm))
        decisions[alg] = {
            "deep_helps_on_ldm_upb": bool(ge_all and gt_two and no_big_loss),
            "deep_ldm_upb_strict_per_seed": deep_ldm, "h12_ldm_upb_strict_per_seed": h12_ldm,
            "deep_nonldm_strict_per_seed": deep_nonldm, "h12_nonldm_strict_per_seed": h12_nonldm,
        }

    return {"gate": gate, "passed": True, "decisions": decisions}, rows_out, ldm_device_rows


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    a_rows = part_a()
    b_summary, b_rows, ldm_device_rows = part_b()

    summary = {
        "complete": True,
        "spec_file": "amendment1.json",
        "spec_sha256": sha(HERE / "amendment1.json"),
        "dependencies": {str(p): sha(p) for p in (
            HERE / "amendment1.json", HERE / "fl.py", HERE / "run_study.py", HERE / "run_amendment.py",
            NOLDM / "run_study.py", CACHE / "manifest.json",
        )},
        "part_a": {"consistency_check": "h12/fedadam @ budget 12 == [45, 44, 44] PASSED", "rows": a_rows},
        "part_b": b_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    a_fields = ["part", "arch", "algorithm", "seed", "rule", "epsilon", "round", "stop_reason",
                "fragments_per_client", "strict_successes", "scored", "faults_detected", "faults",
                "healthy_alarm_devices", "control_alarms", "controls"]
    with (OUT / "part_a.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=a_fields, restval="")
        w.writeheader()
        for r in a_rows:
            w.writerow(r)

    b_fields = ["part", "arch", "algorithm", "seed", "round", "fragments_per_client",
                "strict_overall", "scored_overall", "strict_ldm_upb", "scored_ldm_upb",
                "strict_nonldm", "scored_nonldm", "faults_detected", "faults",
                "healthy_alarm_devices", "control_alarms", "controls"]
    with (OUT / "part_b.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=b_fields, restval="")
        w.writeheader()
        for r in b_rows:
            w.writerow(r)

    if ldm_device_rows:
        ldm_fields = ["arch", "algorithm", "seed", "machine_id", "strict_success", "healthy_alarm", "fault_detected"]
        with (OUT / "ldm_upb_device_rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ldm_fields, restval="")
            w.writeheader()
            for r in ldm_device_rows:
                w.writerow(r)

    print(f"\nwritten {OUT}  total time {time.time() - t_start:.0f}s", flush=True)
    return summary


if __name__ == "__main__":
    run()
