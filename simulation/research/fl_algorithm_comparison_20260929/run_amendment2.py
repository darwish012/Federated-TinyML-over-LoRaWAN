"""Amendment 2: label-free, detector-aligned generalised stopping rule study
(Parts C, D, E). See amendment2.json for the specification.

Reuses fl.py, cadence.py, and the LDM-free noldm helpers (select_devices,
verify_selected_cache_integrity, evaluate, summarize) exactly as run_study.py
and run_amendment.py do. Does not edit fl.py, run_study.py, run_amendment.py,
any test file, or any protocol file.

Design: the expensive part (training + per-round scoring/flagging/evaluation)
is done exactly once per (arch, algorithm, seed), for every round 1..15.
Parts C, D and E are pure lookups/aggregations over that one pass.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOLDM = HERE.parent / "healthy_only_causal_no_ldm_20260922"
CACHE = ROOT / "results/healthy_only_suite_20260907/cache"
CADENCE_SRC = HERE.parent / "healthy_only_causal_20260908" / "cadence.py"
OUT = ROOT / "results/fl_algorithm_comparison_20260929/amendment2"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_causal_20260908"))
import fl  # noqa: E402
import cadence  # noqa: E402

_spec = importlib.util.spec_from_file_location("noldm_study_amend2", NOLDM / "run_study.py")
noldm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(noldm)

ARCH = {"h12": [22, 12, 22]}
SEEDS = [17070, 17071, 17072]
ARMS = [("h12", "fedadam"), ("h12", "fedavg")]
MAX_ROUNDS = 15
DATASETS = ["dlr", "ferrara", "hit", "ims", "kaist", "paderborn", "pronostia", "unsw", "xjtu"]
WINDOW, K, MULT = 21, 13, 2.0

D1_CONSTANTS = [0.995, 0.99, 0.999]
D2_CONSTANTS = [0.02, 0.01, 0.05]
D3_CONSTANTS = [0.02, 0.01, 0.05]
FAMILIES = {"D1": D1_CONSTANTS, "D2": D2_CONSTANTS, "D3": D3_CONSTANTS}
PRIMARY = {"D1": 0.995, "D2": 0.02, "D3": 0.02}
REPORT_ROUNDS = [1, 2, 3, 4, 6, 8, 10, 12, 15]

SANITY = {
    ("h12", "fedadam", 12): [45, 44, 44],
    ("h12", "fedadam", 4): [46, 44, 43],
    ("h12", "fedavg", 12): [46, 44, 43],
}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm_name(arch, alg):
    return f"{arch}_{alg}"


# --------------------------------------------------------------------------
# Spearman rank correlation (NumPy only, ties handled by average rank).
# --------------------------------------------------------------------------
def rankdata_avg(x):
    """1-based ranks with ties resolved to the average rank, matching
    scipy.stats.rankdata(method='average')."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("spearman: arrays must be nonempty and equal length")
    ra, rb = rankdata_avg(a), rankdata_avg(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.sum(ra * rb) / denom)


# --------------------------------------------------------------------------
# Reference replication of cadence.dense_flags' own alarm reference.
# --------------------------------------------------------------------------
def reference(score, ti, ci):
    """ref(client, t) exactly as cadence.dense_flags computes it: max finite
    value of the trailing 13-of-21 order statistic over the commissioning
    history (train+cal, chronological), with dense_flags' own fallback."""
    hist = np.concatenate((ti, ci))
    refstat = cadence._rolling(score[hist], WINDOW, K)
    finite = refstat[np.isfinite(refstat)]
    if len(finite):
        return float(np.max(finite))
    return float(np.quantile(score[ci], 0.995))


def verify_reference(score, ti, ci, ref, flags):
    """ASSERT: recomputing flags from `ref` (rolling statistic over the FULL
    score > multiplier*ref, suppressed through max(ci) inclusive) reproduces
    dense_flags' own output exactly. Raises on mismatch."""
    threshold = MULT * ref
    stat = cadence._rolling(score, WINDOW, K)
    recomputed = stat > threshold
    recomputed[: int(np.max(ci)) + 1] = False
    if not np.array_equal(recomputed, np.asarray(flags, bool)):
        raise RuntimeError("reference() does not reproduce cadence.dense_flags output")


# --------------------------------------------------------------------------
# Cohort loading (identical pattern to run_study.py / run_amendment.py).
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Expensive part, done once: train 15 rounds, score/flag/evaluate every
# round for every device, verifying the reference helper each time.
# --------------------------------------------------------------------------
def evaluate_round(params, devices, arrays):
    rows, comm, ref_map, cal_median = [], {}, {}, {}
    for meta in devices:
        mid = meta["machine_id"]
        a = arrays[mid]
        x = a["x"]
        ti = a["train_indices"].astype(int)
        ci = a["calibration_indices"].astype(int)
        s = fl.score(params, x)
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"nonfinite score {mid}")
        flags = cadence.dense_flags(s, ti, ci, WINDOW, K, MULT)
        ref = reference(s, ti, ci)
        verify_reference(s, ti, ci, ref, flags)
        rows.append(noldm.evaluate(meta, a, flags))
        comm[mid] = s[np.concatenate((ti, ci))]
        ref_map[mid] = ref
        med = float(np.median(s[ci]))
        if med <= 0 or not np.isfinite(med):
            raise RuntimeError(f"non-positive/non-finite calibration median {mid}: {med}")
        cal_median[mid] = med
    return rows, comm, ref_map, cal_median


def train_and_collect(arch, alg, seed, devices, arrays, train):
    layers = ARCH[arch]
    s = fl.Server(alg, fl.init(layers, seed))
    per_round = []
    for t in range(1, MAX_ROUNDS + 1):
        p = s.round(train)
        rows, comm, ref_map, cal_median = evaluate_round(p, devices, arrays)
        per_round.append({"round": t, "params": p, "rows": rows, "comm": comm,
                           "ref": ref_map, "cal_median": cal_median})
    return per_round


def metrics_at_round(per_round, t):
    rd = per_round[t - 1]
    overall = noldm.summarize(rd["rows"])
    per_ds = {ds: noldm.summarize([r for r in rd["rows"] if r["dataset"] == ds]) for ds in DATASETS}
    return overall, per_ds


# --------------------------------------------------------------------------
# Part D signals.
# --------------------------------------------------------------------------
def compute_signals(per_round):
    """round(2..15) -> {'rho': .., 'delta': .., 'param_change': ..}."""
    signals = {}
    for i in range(1, len(per_round)):
        prev, cur = per_round[i - 1], per_round[i]
        t = cur["round"]
        mids = list(cur["comm"].keys())
        rhos = [spearman(cur["comm"][mid], prev["comm"][mid]) for mid in mids]
        rho_t = float(np.mean(rhos))

        rel = []
        for mid in mids:
            h_cur = cur["ref"][mid] / cur["cal_median"][mid]
            h_prev = prev["ref"][mid] / prev["cal_median"][mid]
            if h_prev == 0 or not np.isfinite(h_prev) or not np.isfinite(h_cur):
                raise RuntimeError(f"D2 headroom non-finite/zero for {mid} at round {t}")
            rel.append(abs(h_cur - h_prev) / abs(h_prev))
        delta_t = float(np.median(rel))

        num = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(cur["params"], prev["params"]))))
        den = float(np.sqrt(sum(np.sum(b ** 2) for b in prev["params"])))
        if den == 0 or not np.isfinite(den):
            raise RuntimeError(f"D3 denominator non-finite/zero at round {t}")
        param_change_t = num / den

        if not (np.isfinite(rho_t) and np.isfinite(delta_t) and np.isfinite(param_change_t)):
            raise RuntimeError(f"non-finite Part D signal at round {t}")
        signals[t] = {"rho": rho_t, "delta": delta_t, "param_change": param_change_t}
    return signals


def criterion(family, constant, sig):
    if family == "D1":
        return sig["rho"] >= constant
    if family == "D2":
        return sig["delta"] < constant
    if family == "D3":
        return sig["param_change"] < constant
    raise ValueError(family)


def rule_stop(family, constant, signals):
    """Criterion must hold at t-1 AND t (signals exist for t>=2); earliest
    stop t=3. If never met by round 15, stop at 15 ('cap')."""
    for t in range(3, MAX_ROUNDS + 1):
        if t - 1 in signals and t in signals:
            if criterion(family, constant, signals[t - 1]) and criterion(family, constant, signals[t]):
                return t, "rule"
    return MAX_ROUNDS, "cap"


# --------------------------------------------------------------------------
# Sanity asserts.
# --------------------------------------------------------------------------
def sanity_checks(training):
    checks = {}
    for (arch, alg, t), expected in SANITY.items():
        got = []
        for seed in SEEDS:
            overall, _ = metrics_at_round(training[(arch, alg, seed)], t)
            got.append(int(overall["strict_successes"]))
        checks[f"{arch}_{alg}_round{t}"] = {"got": got, "expected": expected, "pass": got == expected}
    failures = {k: v for k, v in checks.items() if not v["pass"]}
    if failures:
        raise SystemExit("SANITY ASSERT FAILED: " + json.dumps(failures, indent=2))
    return checks


# --------------------------------------------------------------------------
# Part C.
# --------------------------------------------------------------------------
def part_c(training):
    rows = []
    for (arch, alg, seed), per_round in training.items():
        for rd in per_round:
            for ds in DATASETS:
                m = noldm.summarize([r for r in rd["rows"] if r["dataset"] == ds])
                rows.append({"arch": arch, "algorithm": alg, "seed": seed, "round": rd["round"], "dataset": ds,
                             "strict_successes": m["strict_successes"], "scored": m["scored"],
                             "faults_detected": m["faults_detected"], "faults": m["faults"],
                             "healthy_alarm_devices": m["healthy_alarm_devices"],
                             "control_alarms": m["control_alarms"], "controls": m["controls"]})
    return rows


# --------------------------------------------------------------------------
# Part D.
# --------------------------------------------------------------------------
def part_d(training, signals_by_key):
    rows = []
    table = {}   # (arch, alg, family, constant, seed) -> {"stop_round","stop_reason","overall","per_dataset"}
    for (arch, alg, seed), per_round in training.items():
        sig = signals_by_key[(arch, alg, seed)]
        frag = fl.fragments(ARCH[arch])
        for family, constants in FAMILIES.items():
            for c in constants:
                t_stop, why = rule_stop(family, c, sig)
                overall, per_ds = metrics_at_round(per_round, t_stop)
                table[(arch, alg, family, c, seed)] = {"stop_round": t_stop, "stop_reason": why,
                                                        "overall": overall, "per_dataset": per_ds}
                row = {"arch": arch, "algorithm": alg, "seed": seed, "family": family, "constant": c,
                       "is_primary": c == PRIMARY[family], "stop_round": t_stop, "stop_reason": why,
                       "strict_successes": overall["strict_successes"], "scored": overall["scored"],
                       "faults_detected": overall["faults_detected"], "faults": overall["faults"],
                       "healthy_alarm_devices": overall["healthy_alarm_devices"],
                       "control_alarms": overall["control_alarms"], "controls": overall["controls"],
                       "fragments_per_client": 2 * frag * t_stop}
                for ds in DATASETS:
                    row[f"{ds}_strict"] = per_ds[ds]["strict_successes"]
                rows.append(row)
    return rows, table


# --------------------------------------------------------------------------
# Part E (LODO).
# --------------------------------------------------------------------------
def select_constant(arch, alg, family, constants, held_out, table):
    scored = []
    for c in constants:
        total_not_d = 0
        stop_rounds = []
        for seed in SEEDS:
            entry = table[(arch, alg, family, c, seed)]
            total_not_d += entry["overall"]["strict_successes"] - entry["per_dataset"][held_out]["strict_successes"]
            stop_rounds.append(entry["stop_round"])
        mean_stop = float(np.mean(stop_rounds))
        is_primary = (c == PRIMARY[family])
        # sort key: maximise total_not_d, then minimise mean_stop, then prefer primary
        scored.append((-total_not_d, mean_stop, 0 if is_primary else 1, c))
    scored.sort(key=lambda z: (z[0], z[1], z[2]))
    return scored[0][3]


def part_e(training, table):
    rows = []
    summary = {}
    for arch, alg in ARMS:
        fixed4 = [metrics_at_round(training[(arch, alg, s)], 4)[0]["strict_successes"] for s in SEEDS]
        fixed12 = [metrics_at_round(training[(arch, alg, s)], 12)[0]["strict_successes"] for s in SEEDS]
        rows.append({"arch": arch, "algorithm": alg, "family": "", "row_type": "fixed4_pooled",
                     "held_out_dataset": "", "chosen_constant": "",
                     "strict_seed17070": fixed4[0], "strict_seed17071": fixed4[1], "strict_seed17072": fixed4[2],
                     "generalises_vs_fixed12": ""})
        rows.append({"arch": arch, "algorithm": alg, "family": "", "row_type": "fixed12_pooled",
                     "held_out_dataset": "", "chosen_constant": "",
                     "strict_seed17070": fixed12[0], "strict_seed17071": fixed12[1], "strict_seed17072": fixed12[2],
                     "generalises_vs_fixed12": ""})

        arm_summary = {"fixed4_pooled": fixed4, "fixed12_pooled": fixed12, "families": {}}
        for family, constants in FAMILIES.items():
            chosen_per_d = {}
            pooled_lodo = [0, 0, 0]
            for d in DATASETS:
                c = select_constant(arch, alg, family, constants, d, table)
                chosen_per_d[d] = c
                strict_d = [table[(arch, alg, family, c, seed)]["per_dataset"][d]["strict_successes"] for seed in SEEDS]
                for i in range(3):
                    pooled_lodo[i] += strict_d[i]
                rows.append({"arch": arch, "algorithm": alg, "family": family, "row_type": "held_out_choice",
                             "held_out_dataset": d, "chosen_constant": c,
                             "strict_seed17070": strict_d[0], "strict_seed17071": strict_d[1],
                             "strict_seed17072": strict_d[2], "generalises_vs_fixed12": ""})

            rows.append({"arch": arch, "algorithm": alg, "family": family, "row_type": "pooled_lodo",
                         "held_out_dataset": "", "chosen_constant": "",
                         "strict_seed17070": pooled_lodo[0], "strict_seed17071": pooled_lodo[1],
                         "strict_seed17072": pooled_lodo[2], "generalises_vs_fixed12": ""})

            primary_c = PRIMARY[family]
            primary_entries = [table[(arch, alg, family, primary_c, seed)] for seed in SEEDS]
            primary_strict = [e["overall"]["strict_successes"] for e in primary_entries]
            primary_stop = [e["stop_round"] for e in primary_entries]
            generalises = (all(abs(primary_strict[i] - fixed12[i]) <= 1 for i in range(3))
                           and all(r <= 12 for r in primary_stop))
            rows.append({"arch": arch, "algorithm": alg, "family": family, "row_type": "primary_no_tuning",
                         "held_out_dataset": "", "chosen_constant": primary_c,
                         "strict_seed17070": primary_strict[0], "strict_seed17071": primary_strict[1],
                         "strict_seed17072": primary_strict[2], "generalises_vs_fixed12": generalises})

            arm_summary["families"][family] = {
                "chosen_constant_per_heldout": chosen_per_d,
                "pooled_lodo_strict_per_seed": pooled_lodo,
                "primary_pooled_strict_per_seed": primary_strict,
                "primary_stop_round_per_seed": primary_stop,
                "generalises_vs_fixed12": generalises,
            }
        summary[arm_name(arch, alg)] = arm_summary
    return rows, summary


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
def run():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    devices, arrays, train = load_cohort()

    training = {}
    for arch, alg in ARMS:
        for seed in SEEDS:
            t0 = time.time()
            training[(arch, alg, seed)] = train_and_collect(arch, alg, seed, devices, arrays, train)
            print(f"trained+scored {arch}_{alg} seed {seed}: 15 rounds x 54 devices "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    checks = sanity_checks(training)
    for k, v in checks.items():
        print(f"sanity {k}: {v['got']} (expect {v['expected']}) {'PASS' if v['pass'] else 'FAIL'}", flush=True)

    signals_by_key = {key: compute_signals(pr) for key, pr in training.items()}

    c_rows = part_c(training)
    d_rows, d_table = part_d(training, signals_by_key)
    e_rows, e_summary = part_e(training, d_table)

    signal_rows = []
    for (arch, alg, seed), sig in signals_by_key.items():
        for t in sorted(sig):
            signal_rows.append({"arch": arch, "algorithm": alg, "seed": seed, "round": t,
                                 "rho_t": sig[t]["rho"], "delta_t": sig[t]["delta"],
                                 "param_change_t": sig[t]["param_change"]})

    deps = {str(p): sha(p) for p in (HERE / "amendment2.json", HERE / "fl.py", HERE / "run_study.py",
                                     HERE / "run_amendment2.py", NOLDM / "run_study.py", CADENCE_SRC,
                                     CACHE / "manifest.json")}

    part_d_summary = {}
    for (arch, alg, family, c, seed), entry in d_table.items():
        part_d_summary.setdefault(arm_name(arch, alg), {}).setdefault(family, {}).setdefault(str(c), {})[str(seed)] = {
            "stop_round": entry["stop_round"], "stop_reason": entry["stop_reason"],
            "strict_successes": entry["overall"]["strict_successes"], "scored": entry["overall"]["scored"],
        }

    summary = {
        "complete": True,
        "amendment": 2,
        "spec_sha256": sha(HERE / "amendment2.json"),
        "dependencies": deps,
        "cohort": {"clients": 54, "scored": 49, "datasets": DATASETS},
        "arms": [arm_name(a, b) for a, b in ARMS],
        "seeds": SEEDS,
        "max_rounds": MAX_ROUNDS,
        "primary_constants": PRIMARY,
        "sensitivity_constants": FAMILIES,
        "sanity_checks": checks,
        "part_c_report_rounds": REPORT_ROUNDS,
        "part_d": part_d_summary,
        "part_e": e_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with (OUT / "part_c_per_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "algorithm", "seed", "round", "dataset", "strict_successes",
                                          "scored", "faults_detected", "faults", "healthy_alarm_devices",
                                          "control_alarms", "controls"])
        w.writeheader(); w.writerows(c_rows)

    d_fields = ["arch", "algorithm", "seed", "family", "constant", "is_primary", "stop_round", "stop_reason",
                "strict_successes", "scored", "faults_detected", "faults", "healthy_alarm_devices",
                "control_alarms", "controls", "fragments_per_client"] + [f"{ds}_strict" for ds in DATASETS]
    with (OUT / "part_d_rules.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=d_fields)
        w.writeheader(); w.writerows(d_rows)

    with (OUT / "part_e_lodo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "algorithm", "family", "row_type", "held_out_dataset",
                                          "chosen_constant", "strict_seed17070", "strict_seed17071",
                                          "strict_seed17072", "generalises_vs_fixed12"])
        w.writeheader(); w.writerows(e_rows)

    with (OUT / "signals.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "algorithm", "seed", "round", "rho_t", "delta_t", "param_change_t"])
        w.writeheader(); w.writerows(signal_rows)

    print(f"\nwritten {OUT}  total time {time.time() - t_start:.0f}s", flush=True)
    return summary


if __name__ == "__main__":
    run()
