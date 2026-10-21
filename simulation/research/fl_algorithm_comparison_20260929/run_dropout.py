"""Client dropout / packet-loss robustness experiment. See dropout.json.

Order of operations:
  1. select the LDM-free cohort from metadata BEFORE any cache load
     (noldm.select_devices), then verify cache integrity.
  2. S0 GATE: with all-True uplink/downlink masks, dropout_fl.train_with_loss
     must reproduce fl.Server bit-exactly for 12 rounds, and strict must be
     45/44/44 (fedadam) / 46/44/43 (fedavg) for seeds 17070/17071/17072.
     Abort otherwise - no scenario is run.
  3. only then run scenarios S1 (uplink iid loss), S2 (both directions iid
     loss), S3 (persistent non-participation), 5 repetitions per training
     seed each, and apply the predeclared robustness criterion.

Reuses fl.py (init/local_train/score/constants), dropout_fl.py (the dropout
training mechanics), cadence.dense_flags, and the noldm helpers
(select_devices/verify_selected_cache_integrity/evaluate/summarize) exactly
as run_study.py / run_amendment2.py do. Does not edit any existing file.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOLDM = HERE.parent / "healthy_only_causal_no_ldm_20260922"
CACHE = ROOT / "results/healthy_only_suite_20260907/cache"
OUT = ROOT / "results/fl_algorithm_comparison_20260929/dropout"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_causal_20260908"))
import fl  # noqa: E402
import cadence  # noqa: E402
import dropout_fl as dfl  # noqa: E402

_spec = importlib.util.spec_from_file_location("noldm_dropout", NOLDM / "run_study.py")
noldm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(noldm)

LAYERS = [22, 12, 22]
SEEDS = [17070, 17071, 17072]
ALGORITHMS = ["fedadam", "fedavg"]
ROUNDS = 12
REPS = 5
S0_EXPECTED = {"fedadam": [45, 44, 44], "fedavg": [46, 44, 43]}
SCENARIO_INDEX = {"S1": 1, "S2": 2, "S3": 3}
LEVELS = {"S1": dfl.S1_LEVELS, "S2": dfl.S2_LEVELS, "S3": dfl.S3_LEVELS}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cohort():
    manifest = read(CACHE / "manifest.json")
    devices = noldm.select_devices(manifest)            # before any np.load
    noldm.verify_selected_cache_integrity(manifest, devices)
    arrays, train = {}, {}
    for meta in devices:
        with np.load(meta["path"], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        mid = meta["machine_id"]
        arrays[mid] = a
        train[mid] = a["x"][a["train_indices"].astype(int)]
    assert len(train) == 54 and not any(m.startswith("ldm_upb::") for m in train)
    return devices, arrays, train


def evaluate_with_copies(client_finals, devices, arrays):
    """Score every device with ITS OWN final params (own_copy for clients,
    or the server global if a shared dict is passed for all mids)."""
    rows = []
    for meta in devices:
        mid = meta["machine_id"]
        a = arrays[mid]
        x = a["x"]
        ti = a["train_indices"].astype(int)
        ci = a["calibration_indices"].astype(int)
        p = client_finals[mid]
        s = fl.score(p, x)
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"nonfinite score {mid}")
        flags = cadence.dense_flags(s, ti, ci, 21, 13, 2.0)
        rows.append(noldm.evaluate(meta, a, flags))
    return noldm.summarize(rows), rows


# ---------------------------------------------------------------------------
# S0 gate.
# ---------------------------------------------------------------------------
def run_s0_gate(devices, arrays, train):
    client_ids = list(train.keys())
    n = len(client_ids)
    gate = []
    s0_strict = {}
    for algorithm in ALGORITHMS:
        for seed, expected in zip(SEEDS, S0_EXPECTED[algorithm]):
            uplink = np.ones((ROUNDS, n), dtype=bool)
            downlink = np.ones((ROUNDS, n), dtype=bool)
            server, finals = dfl.train_with_loss(train, LAYERS, seed, algorithm, ROUNDS, uplink, downlink)

            ref = fl.Server(algorithm, fl.init(LAYERS, seed))
            for _ in range(ROUNDS):
                ref.round(train)
            bit_exact = all(np.array_equal(a, b) for a, b in zip(server, ref.p))
            finals_match_server = all(
                all(np.array_equal(a, b) for a, b in zip(finals[cid], server)) for cid in client_ids
            )

            metrics, _ = evaluate_with_copies(finals, devices, arrays)
            strict = metrics["strict_successes"]
            ok = bit_exact and finals_match_server and strict == expected
            gate.append({"algorithm": algorithm, "seed": seed, "bit_exact": bit_exact,
                         "finals_match_server": finals_match_server,
                         "strict": strict, "expected": expected, "pass": ok})
            s0_strict[(algorithm, seed)] = strict
            print(f"S0 gate {algorithm:>8} seed {seed}: bit_exact={bit_exact} "
                  f"strict={strict}/49 (expect {expected})  {'PASS' if ok else 'FAIL'}", flush=True)
    if not all(g["pass"] for g in gate):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "S0_GATE_FAILED.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        raise SystemExit("S0 GATE FAILED - aborting, no scenario computed")
    return gate, s0_strict


# ---------------------------------------------------------------------------
# One (scenario, level, seed, rep, algorithm) repetition.
# ---------------------------------------------------------------------------
def run_repetition(devices, arrays, train, algorithm, seed, uplink, downlink, never_participating=None):
    server, finals = dfl.train_with_loss(train, LAYERS, seed, algorithm, ROUNDS, uplink, downlink)
    metrics, rows = evaluate_with_copies(finals, devices, arrays)
    mean_uplinks = float(np.mean(uplink.sum(axis=1)))
    stale_scored = 0
    for meta, row in zip(devices, rows):
        if not row["scored"]:
            continue
        mid = meta["machine_id"]
        if not all(np.array_equal(a, b) for a, b in zip(finals[mid], server)):
            stale_scored += 1
    result = {"strict": metrics["strict_successes"], "scored": metrics["scored"],
              "faults_detected": metrics["faults_detected"], "faults": metrics["faults"],
              "healthy_alarm_devices": metrics["healthy_alarm_devices"],
              "control_alarms": metrics["control_alarms"], "controls": metrics["controls"],
              "mean_uplinks_per_round": mean_uplinks, "stale_scored_devices": stale_scored}
    if never_participating is not None:
        part_rows = [r for r, meta in zip(rows, devices) if r["scored"] and meta["machine_id"] not in never_participating]
        never_rows = [r for r, meta in zip(rows, devices) if r["scored"] and meta["machine_id"] in never_participating]
        result["strict_participating"] = sum(r["strict_success"] for r in part_rows)
        result["scored_participating"] = len(part_rows)
        result["strict_never_participating"] = sum(r["strict_success"] for r in never_rows)
        result["scored_never_participating"] = len(never_rows)
    return result


# ---------------------------------------------------------------------------
# Scenario runners.
# ---------------------------------------------------------------------------
def run_scenarios(devices, arrays, train):
    client_ids = list(train.keys())
    n = len(client_ids)
    cells = []

    t0 = time.time()
    for level_idx, q in enumerate(dfl.S1_LEVELS):
        for seed in SEEDS:
            for rep in range(REPS):
                r = dfl.rep_rng(seed, SCENARIO_INDEX["S1"], rep)
                lvl_rngs = dfl.level_rngs(r, len(dfl.S1_LEVELS))
                uplink, downlink = dfl.make_s1_masks(lvl_rngs[level_idx], ROUNDS, n, q)
                for algorithm in ALGORITHMS:
                    res = run_repetition(devices, arrays, train, algorithm, seed, uplink, downlink)
                    res.update({"scenario": "S1", "level": q, "seed": seed, "rep": rep, "algorithm": algorithm})
                    cells.append(res)
    print(f"S1 done [{time.time() - t0:.0f}s]", flush=True)

    t0 = time.time()
    for level_idx, q in enumerate(dfl.S2_LEVELS):
        for seed in SEEDS:
            for rep in range(REPS):
                r = dfl.rep_rng(seed, SCENARIO_INDEX["S2"], rep)
                lvl_rngs = dfl.level_rngs(r, len(dfl.S2_LEVELS))
                uplink, downlink = dfl.make_s2_masks(lvl_rngs[level_idx], ROUNDS, n, q)
                for algorithm in ALGORITHMS:
                    res = run_repetition(devices, arrays, train, algorithm, seed, uplink, downlink)
                    res.update({"scenario": "S2", "level": q, "seed": seed, "rep": rep, "algorithm": algorithm})
                    cells.append(res)
    print(f"S2 done [{time.time() - t0:.0f}s]", flush=True)

    t0 = time.time()
    for level_idx, f in enumerate(dfl.S3_LEVELS):
        for seed in SEEDS:
            for rep in range(REPS):
                r = dfl.rep_rng(seed, SCENARIO_INDEX["S3"], rep)
                lvl_rngs = dfl.level_rngs(r, len(dfl.S3_LEVELS))
                never = dfl.select_never_participating(devices, f, lvl_rngs[level_idx])
                uplink, downlink = dfl.make_s3_masks(ROUNDS, client_ids, never)
                for algorithm in ALGORITHMS:
                    res = run_repetition(devices, arrays, train, algorithm, seed, uplink, downlink,
                                          never_participating=never)
                    res.update({"scenario": "S3", "level": f, "seed": seed, "rep": rep, "algorithm": algorithm})
                    cells.append(res)
    print(f"S3 done [{time.time() - t0:.0f}s]", flush=True)

    return cells


# ---------------------------------------------------------------------------
# Robustness criterion (robustness_criterion_fixed_now), applied literally.
# ---------------------------------------------------------------------------
def evaluate_criterion(scenario, level, algorithm, groups, s0_strict):
    per_seed_detail = []
    seeds_ok = []
    for seed in SEEDS:
        reps = groups[(scenario, level, algorithm, seed)]
        strict_vals = [r["strict"] for r in reps]
        mean_strict = float(np.mean(strict_vals))
        s0 = s0_strict[(algorithm, seed)]
        mean_diff = abs(mean_strict - s0)
        worst_diff = max(abs(v - s0) for v in strict_vals)
        within_1 = mean_diff <= 1
        within_2 = worst_diff <= 2
        seed_ok = within_1 and within_2
        seeds_ok.append(seed_ok)
        per_seed_detail.append({"seed": seed, "s0_strict": s0, "mean_strict": mean_strict,
                                 "mean_abs_diff": mean_diff, "worst_abs_diff": worst_diff,
                                 "within_1_mean": within_1, "within_2_worst": within_2, "seed_tolerated": seed_ok})
    base_tolerated = all(seeds_ok)

    participation_gap_pp = None
    participation_within_10pp = None
    if scenario == "S3":
        all_reps = [r for seed in SEEDS for r in groups[(scenario, level, algorithm, seed)]]
        strict_p = sum(r["strict_participating"] for r in all_reps)
        scored_p = sum(r["scored_participating"] for r in all_reps)
        strict_n = sum(r["strict_never_participating"] for r in all_reps)
        scored_n = sum(r["scored_never_participating"] for r in all_reps)
        if scored_p and scored_n:
            rate_p = strict_p / scored_p
            rate_n = strict_n / scored_n
            participation_gap_pp = abs(rate_p - rate_n) * 100.0
            participation_within_10pp = participation_gap_pp <= 10.0

    if scenario == "S3":
        tolerated = base_tolerated and bool(participation_within_10pp)
    else:
        tolerated = base_tolerated
    return {"per_seed": per_seed_detail, "base_tolerated": base_tolerated,
            "participation_gap_pp": participation_gap_pp,
            "participation_within_10pp": participation_within_10pp, "tolerated": tolerated}


# ---------------------------------------------------------------------------
# Report/output assembly.
# ---------------------------------------------------------------------------
def build_report(cells):
    groups = defaultdict(list)
    for c in cells:
        groups[(c["scenario"], c["level"], c["algorithm"], c["seed"])].append(c)

    report = {"S1": {}, "S2": {}, "S3": {}}
    for scenario in ("S1", "S2", "S3"):
        for level in LEVELS[scenario]:
            report[scenario][str(level)] = {}
            for algorithm in ALGORITHMS:
                per_seed = {}
                for seed in SEEDS:
                    reps = groups[(scenario, level, algorithm, seed)]
                    strict_vals = [r["strict"] for r in reps]
                    entry = {
                        "mean_strict": float(np.mean(strict_vals)),
                        "min_strict": int(min(strict_vals)),
                        "max_strict": int(max(strict_vals)),
                        "faults_detected_mean": float(np.mean([r["faults_detected"] for r in reps])),
                        "healthy_alarm_devices_mean": float(np.mean([r["healthy_alarm_devices"] for r in reps])),
                        "control_alarms_mean": float(np.mean([r["control_alarms"] for r in reps])),
                        "mean_uplinks_per_round": float(np.mean([r["mean_uplinks_per_round"] for r in reps])),
                    }
                    if scenario == "S2":
                        entry["stale_scored_devices_mean"] = float(np.mean([r["stale_scored_devices"] for r in reps]))
                    if scenario == "S3":
                        entry["strict_participating_total"] = sum(r["strict_participating"] for r in reps)
                        entry["scored_participating_total"] = sum(r["scored_participating"] for r in reps)
                        entry["strict_never_participating_total"] = sum(r["strict_never_participating"] for r in reps)
                        entry["scored_never_participating_total"] = sum(r["scored_never_participating"] for r in reps)
                    per_seed[str(seed)] = entry
                report[scenario][str(level)][algorithm] = per_seed
    return report, groups


def build_criterion(groups, s0_strict):
    criterion = {"S1": {}, "S2": {}, "S3": {}}
    rows = []
    for scenario in ("S1", "S2", "S3"):
        for level in LEVELS[scenario]:
            criterion[scenario][str(level)] = {}
            for algorithm in ALGORITHMS:
                res = evaluate_criterion(scenario, level, algorithm, groups, s0_strict)
                criterion[scenario][str(level)][algorithm] = res
                for sd in res["per_seed"]:
                    rows.append({"scenario": scenario, "level": level, "algorithm": algorithm, "row_type": "seed",
                                 "seed": sd["seed"], "s0_strict": sd["s0_strict"], "mean_strict": sd["mean_strict"],
                                 "mean_abs_diff": sd["mean_abs_diff"], "worst_abs_diff": sd["worst_abs_diff"],
                                 "within_1_mean": sd["within_1_mean"], "within_2_worst": sd["within_2_worst"],
                                 "seed_tolerated": sd["seed_tolerated"], "participation_gap_pp": "",
                                 "participation_within_10pp": "", "tolerated": ""})
                rows.append({"scenario": scenario, "level": level, "algorithm": algorithm, "row_type": "level_summary",
                             "seed": "ALL", "s0_strict": "", "mean_strict": "", "mean_abs_diff": "",
                             "worst_abs_diff": "", "within_1_mean": "", "within_2_worst": "", "seed_tolerated": "",
                             "participation_gap_pp": res["participation_gap_pp"] if res["participation_gap_pp"] is not None else "",
                             "participation_within_10pp": res["participation_within_10pp"] if res["participation_within_10pp"] is not None else "",
                             "tolerated": res["tolerated"]})
    return criterion, rows


def run():
    t_start = time.time()
    devices, arrays, train = load_cohort()

    gate, s0_strict = run_s0_gate(devices, arrays, train)

    cells = run_scenarios(devices, arrays, train)

    report, groups = build_report(cells)
    criterion, criterion_rows = build_criterion(groups, s0_strict)

    OUT.mkdir(parents=True, exist_ok=True)

    deps = {str(p): sha(p) for p in (
        HERE / "dropout.json", HERE / "dropout_fl.py", HERE / "run_dropout.py", HERE / "test_dropout.py",
        HERE / "fl.py", HERE.parent / "healthy_only_causal_20260908" / "cadence.py",
        NOLDM / "run_study.py", CACHE / "manifest.json",
    )}

    summary = {
        "complete": True,
        "study": "dropout.json",
        "cohort": {"clients": 54, "scored": 49, "faults": 43, "controls": 6},
        "algorithms": ALGORITHMS,
        "rounds": ROUNDS,
        "training_seeds": SEEDS,
        "dropout_repetitions_per_seed": REPS,
        "dropout_rng": "numpy default_rng(1000*training_seed + 100*scenario_index + repetition); "
                       "S1=1, S2=2, S3=3; within one repetition's generator, one independent child "
                       "Generator per loss level is drawn via Generator.spawn(n_levels) (NumPy SeedSequence "
                       "spawning), in the fixed level order S1/S2 q in (0.1,0.2,0.3,0.5), S3 f in (0.1,0.2,0.3), "
                       "so a level's masks never depend on another level's draws.",
        "s1_levels": list(dfl.S1_LEVELS), "s2_levels": list(dfl.S2_LEVELS), "s3_levels": list(dfl.S3_LEVELS),
        "dependencies": deps,
        "s0_gate": gate,
        "s0_strict": {f"{alg}_{seed}": s0_strict[(alg, seed)] for alg in ALGORITHMS for seed in SEEDS},
        "report": report,
        "criterion": criterion,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    cell_fields = ["scenario", "level", "algorithm", "seed", "rep", "strict", "scored", "faults_detected", "faults",
                   "healthy_alarm_devices", "control_alarms", "controls", "mean_uplinks_per_round",
                   "stale_scored_devices", "strict_participating", "scored_participating",
                   "strict_never_participating", "scored_never_participating"]
    with (OUT / "cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cell_fields, restval="")
        w.writeheader()
        w.writerows(cells)

    criterion_fields = ["scenario", "level", "algorithm", "row_type", "seed", "s0_strict", "mean_strict",
                        "mean_abs_diff", "worst_abs_diff", "within_1_mean", "within_2_worst", "seed_tolerated",
                        "participation_gap_pp", "participation_within_10pp", "tolerated"]
    with (OUT / "criterion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=criterion_fields, restval="")
        w.writeheader()
        w.writerows(criterion_rows)

    print(f"\nwritten {OUT}  total time {time.time() - t_start:.0f}s", flush=True)
    return summary


if __name__ == "__main__":
    run()
