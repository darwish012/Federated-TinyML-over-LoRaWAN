"""FedAvg / FedAdam / FedProx under one label-free stopping rule, plus a depth arm.

Order of operations is fixed:
  1. select the LDM-free cohort from metadata BEFORE any cache load
  2. REPLICATION GATE: fedadam h12 @ round 12 must reproduce the saved
     LDM-free weights and 45/44/44; abort otherwise
  3. only then train and evaluate the new arms
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
SAVED = ROOT / "results/healthy_only_causal_no_ldm_20260922"
CACHE = ROOT / "results/healthy_only_suite_20260907/cache"
OUT = ROOT / "results/fl_algorithm_comparison_20260929"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_causal_20260908"))
import fl  # noqa: E402
import cadence  # noqa: E402

_spec = importlib.util.spec_from_file_location("noldm_study", NOLDM / "run_study.py")
noldm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(noldm)

ARCH = {"h12": [22, 12, 22], "deep": [22, 16, 8, 16, 22]}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arms(spec):
    out = []
    for alg in ("fedavg", "fedadam", "fedprox"):
        for arch in ("h12", "deep"):
            out.append((arch, alg, 0.01 if alg == "fedprox" else 0.0, True))
    for mu in (0.001, 0.1, 1.0):
        out.append(("h12", "fedprox", mu, False))   # sensitivity only
    return out


def arm_name(arch, alg, mu):
    return f"{arch}_{alg}" + (f"_mu{mu:g}" if alg == "fedprox" else "")


def evaluate_params(params, devices, arrays):
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


def run():
    spec = read(HERE / "protocol.json")
    stop_cfg = spec["stopping_rule"]
    EPS, P, MAXR = stop_cfg["epsilon"], stop_cfg["patience_P"], stop_cfg["max_rounds"]
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

    OUT.mkdir(parents=True, exist_ok=True)
    deps = {str(p): sha(p) for p in (HERE / "protocol.json", HERE / "fl.py", HERE / "run_study.py",
                                     HERE / "test_fl.py", NOLDM / "run_study.py", CACHE / "manifest.json")}

    # ---------------- 2. REPLICATION GATE ----------------
    gate = []
    for seed, expect in zip(spec["seeds"], (45, 44, 44)):
        s = fl.Server("fedadam", fl.init(ARCH["h12"], seed))
        for _ in range(12):
            s.round(train)
        with (SAVED / f"global_h12_seed{seed}.pkl").open("rb") as f:
            ref = pickle.load(f)["global"]
        maxdiff = max(float(np.max(np.abs(np.asarray(u) - np.asarray(v)))) for u, v in zip(s.p, ref))
        metrics, _ = evaluate_params(s.p, devices, arrays)
        ok = maxdiff <= 1e-10 and metrics["strict_successes"] == expect
        gate.append({"seed": seed, "max_abs_weight_diff": maxdiff, "strict": metrics["strict_successes"],
                     "expected": expect, "pass": ok})
        print(f"gate seed {seed}: maxdiff {maxdiff:.2e}  strict {metrics['strict_successes']}/49 "
              f"(expect {expect})  {'PASS' if ok else 'FAIL'}", flush=True)
    if not all(g["pass"] for g in gate):
        (OUT / "GATE_FAILED.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        raise SystemExit("REPLICATION GATE FAILED - no new outcome computed")

    # ---------------- 3. new arms ----------------
    results, curves = [], []
    for seed in spec["seeds"]:
        for arch, alg, mu, primary in arms(spec):
            t0 = time.time()
            s = fl.Server(alg, fl.init(ARCH[arch], seed), mu=mu)
            losses, history = [], []
            for r in range(1, MAXR + 1):
                p = s.round(train)
                history.append(p)
                L = float(np.mean([np.mean(fl.score(p, calib[m])) for m in calib]))
                if not np.isfinite(L):
                    raise RuntimeError(f"diverged {arch} {alg} seed {seed} round {r}")
                losses.append(L)
                curves.append({"seed": seed, "arm": arm_name(arch, alg, mu), "round": r, "val_loss": L})
            t_stop, why = fl.stop_round(losses, EPS, P, MAXR)
            frag = fl.fragments(ARCH[arch])
            entry = {"seed": seed, "arch": arch, "algorithm": alg, "mu": mu, "primary": primary,
                     "arm": arm_name(arch, alg, mu), "parameters": fl.n_params(ARCH[arch]),
                     "fragments_per_direction": frag, "stop_round": t_stop, "stop_reason": why,
                     "val_loss_at_stop": losses[t_stop - 1], "val_loss_at_12": losses[11],
                     "val_loss_at_60": losses[-1], "checkpoints": {}}
            for label, rnd in (("stop", t_stop), ("r12", 12), ("r60", MAXR)):
                m, _ = evaluate_params(history[rnd - 1], devices, arrays)
                entry["checkpoints"][label] = {"round": rnd, "fragments_per_client": 2 * frag * rnd, **m}
            results.append(entry)
            c = entry["checkpoints"]
            print(f"seed {seed} {entry['arm']:<20} stop {t_stop:>2} ({why})  "
                  f"strict stop {c['stop']['strict_successes']}/49  r12 {c['r12']['strict_successes']}  "
                  f"r60 {c['r60']['strict_successes']}  [{time.time() - t0:.0f}s]", flush=True)

    # ---------------- decision rules, applied mechanically ----------------
    def per_seed(arm, ck="stop", key="strict_successes"):
        return [next(e for e in results if e["seed"] == s and e["arm"] == arm)["checkpoints"][ck][key]
                for s in spec["seeds"]]

    def rounds(arm):
        return [next(e for e in results if e["seed"] == s and e["arm"] == arm)["stop_round"] for s in spec["seeds"]]

    def better(a, b):
        x, y = per_seed(a), per_seed(b)
        return all(u >= v for u, v in zip(x, y)) and sum(u > v for u, v in zip(x, y)) >= 2

    def cheaper(a, b):
        return all(u < v for u, v in zip(rounds(a), rounds(b)))

    decisions = {}
    for arch in ("h12", "deep"):
        names = [arm_name(arch, a, 0.01 if a == "fedprox" else 0.0) for a in ("fedavg", "fedadam", "fedprox")]
        pair = {}
        for a in names:
            for b in names:
                if a != b:
                    pair[f"{a} vs {b}"] = {"better_for_detection": better(a, b), "cheaper": cheaper(a, b)}
        decisions[arch] = pair
    depth = {}
    for alg in ("fedavg", "fedadam", "fedprox"):
        mu = 0.01 if alg == "fedprox" else 0.0
        depth[alg] = {"deep_better_than_h12": better(arm_name("deep", alg, mu), arm_name("h12", alg, mu))}
    decisions["depth"] = depth

    summary = {"complete": True, "classification": spec["classification"], "protocol_sha256": sha(HERE / "protocol.json"),
               "dependencies": deps, "replication_gate": gate, "results": results, "decisions": decisions}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (OUT / "curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "round", "val_loss"]); w.writeheader(); w.writerows(curves)
    print("\nwritten", OUT)
    return summary


if __name__ == "__main__":
    run()
