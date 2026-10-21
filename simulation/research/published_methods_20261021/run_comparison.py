"""Saved reconstruction of memory appendices 09d/09e. Run with python -I -B.

Only protocol.json's fixed choices are implemented. Historical missing settings
are explicit assumptions, not claims about the original papers or lost runs.
No downloads, H12 training, writes to existing studies, or source modifications.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
import warnings

sys.dont_write_bytecode = True
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import scipy
from scipy.signal import hilbert
from scipy.stats import genextreme
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "results/published_methods_20261021"
MANIFEST = ROOT / "results/healthy_only_suite_20260907/cache/manifest.json"
STATES = ROOT / "results/healthy_only_causal_no_ldm_20260922"
SPEC = json.loads((HERE / "protocol.json").read_text())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CADENCE_PATH = ROOT / "research/healthy_only_causal_20260908/cadence.py"
WIDER_PATH = ROOT / "research/healthy_only_causal_20260908/wider_models.py"
cadence = module("saved_cadence", CADENCE_PATH)
wider = module("saved_wider", WIDER_PATH)


def load_arrays(meta):
    with np.load(meta["path"], allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def history(a):
    return np.concatenate((a["train_indices"], a["calibration_indices"])).astype(int)


def point_flags(scores, threshold, a, persist=False):
    flags = np.asarray(scores > threshold, bool)
    if persist:
        counts = np.convolve(flags.astype(int), np.ones(21, int), mode="full")[:len(flags)]
        flags = counts >= 13
        flags[:20] = False
    flags[:int(a["calibration_indices"][-1]) + 1] = False
    return flags


def wang(scores, a, sigma, persist=False):
    hi = history(a)
    bi = hi[:100 if len(hi) >= 100 else min(30, len(hi))]
    threshold = float(np.mean(scores[bi]) + sigma * np.std(scores[bi], ddof=0))
    return point_flags(scores, threshold, a, persist), {"threshold": threshold, "baseline_indices": bi.tolist()}


def evaluate(meta, a, flags):
    h, f = a["healthy_test_indices"].astype(int), a["unhealthy_test_indices"].astype(int)
    ha, fd = bool(np.any(flags[h])), bool(np.any(flags[f]))
    control = meta["truth_kind"] == "healthy_control"
    assert meta["truth"] is not None or control
    hits = np.flatnonzero(flags)
    return {"machine_id": meta["machine_id"], "dataset": meta["dataset"],
            "expected": "healthy_control" if control else "fault", "strict_success": not ha and (fd or control),
            "healthy_alarm": ha, "healthy_alarm_acquisitions": int(np.sum(flags[h])),
            "healthy_test_rows": len(h), "fault_test_rows": len(f), "fault_detected": fd,
            "first_alarm_index": int(hits[0]) if len(hits) else None}


def metrics(rows):
    return {"strict_successes": sum(r["strict_success"] for r in rows), "scored": len(rows),
            "faults_detected": sum(r["fault_detected"] for r in rows if r["expected"] == "fault"),
            "faults": sum(r["expected"] == "fault" for r in rows),
            "healthy_alarm_devices": sum(r["healthy_alarm"] for r in rows),
            "healthy_alarm_acquisitions": sum(r["healthy_alarm_acquisitions"] for r in rows),
            "controls": sum(r["expected"] == "healthy_control" for r in rows),
            "control_alarms": sum(r["healthy_alarm"] for r in rows if r["expected"] == "healthy_control")}


def gev(scores, a, regimes, seed):
    hi = history(a)
    labels = np.zeros(len(scores), int)
    if regimes > 1:
        km = KMeans(n_clusters=regimes, n_init=10, random_state=seed).fit(a["x"][hi])
        labels = km.predict(a["x"])
    thresholds = np.zeros(len(scores), float)
    records = []
    for regime in range(regimes):
        baseline = scores[hi[labels[hi] == regime]]
        assert len(baseline)
        maxima = np.array([np.max(baseline[i:i+10]) for i in range(0, len(baseline), 10)])
        reason, params, warns = None, None, []
        if len(maxima) < 3 or np.ptp(maxima) <= 1e-12:
            reason = "fewer than 3 or degenerate block maxima"
        else:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                try:
                    params = [float(v) for v in genextreme.fit(maxima)]
                    threshold = float(genextreme.ppf(.99, *params))
                    if not np.isfinite(threshold) or threshold < 0:
                        reason = "invalid fitted return level"
                except (ValueError, FloatingPointError) as exc:
                    reason = str(exc)
            warns = [str(w.message) for w in ws]
        if reason:
            threshold = float(np.quantile(baseline, .995))
        thresholds[labels == regime] = threshold
        records.append({"regime": regime, "baseline_n": len(baseline), "block_maxima": maxima.tolist(),
                        "fit_params": params, "threshold": threshold, "fallback": reason, "warnings": warns})
    return thresholds, {"regimes": records}


def onset(scores, a, seed):
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(scores[:, None])
    flips = np.flatnonzero((labels != labels[0]) & (np.arange(len(scores)) >= 10))
    flags = np.zeros(len(scores), bool)
    if len(flips):
        flags[int(flips[0]):] = True
    flags[:int(a["calibration_indices"][-1])+1] = False
    return flags, {"onset_index": int(flips[0]) if len(flips) else None, "tolerance": 10,
                   "uses_full_trajectory": True}


def add_cell(cells, part, front, rule, seed, rows):
    cells.append({"part": part, "front_end": front, "rule": rule, "seed": seed,
                  "metrics": metrics(rows), "devices": rows})


def natural(p):
    return [int(v) if v.isdigit() else v for v in re.split(r"(\d+)", p.name)]


def raw_files(meta):
    ds, unit = meta["machine_id"].split("::")
    data = REPO / "datasets"
    if ds == "pronostia":
        base = data / "pronostia/phm-ieee-2012-data-challenge-dataset-master"
        dirs = [base / group / unit for group in ("Learning_set", "Full_Test_Set")]
        dirs = [p for p in dirs if p.is_dir()]
        assert len(dirs) == 1, dirs
        return sorted(dirs[0].glob("acc_*.csv"), key=natural)
    assert ds == "xjtu"
    dirs = []
    for group in ("xjtu_raw", "xjtu_reserved_raw", "xjtu_prospective_raw"):
        dirs += [p for p in (data / "xjtu" / group).rglob(unit) if p.is_dir() and next(p.glob("*.csv"), None)]
    assert len(dirs) == 1, dirs
    return sorted(dirs[0].glob("*.csv"), key=natural)


def read_raw(p, ds):
    payload = p.read_bytes()
    text = payload.decode("ascii")
    if ds == "pronostia":
        v = np.fromstring(text.replace(",", " ").replace(";", " "), sep=" ")
        assert v.size == 2560*6, p
        v = v.reshape(-1, 6)[:, 4:6].T
    else:
        text = text.split("\n", 1)[1]
        v = np.fromstring(text.replace(",", " "), sep=" ")
        assert v.size == 32768*2, p
        v = v.reshape(-1, 2).T
    return v, hashlib.sha256(payload).hexdigest()


def loges(channels):
    x = np.asarray(channels, float)
    x = x - np.mean(x, axis=1, keepdims=True)
    env = np.abs(hilbert(x, axis=1))
    env -= np.mean(env, axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(env, axis=1))
    freq = np.fft.rfftfreq(x.shape[1], 1/25600)
    spec = spec[:, freq <= 5000]
    edges = np.linspace(0, spec.shape[1], 257, dtype=int)
    bins = np.stack([np.mean(spec[:, edges[i]:edges[i+1]], axis=1) for i in range(256)], axis=1)
    return np.max(np.log1p(bins), axis=0)


def build_loges(meta, a, dependencies):
    mid = meta["machine_id"]
    p = OUT / "loges" / (mid.replace("::", "__") + ".npz")
    source = p.with_suffix(".sources.json")
    key = sha(HERE / "protocol.json")
    if p.exists() and source.exists():
        rec = read(source)
        assert rec["protocol_sha256"] == key and rec["cache_sha256"] == sha(p)
        # Revalidate every raw input on resume, not only the generated matrix.
        files = raw_files(meta)
        assert [str(f.resolve()) for f in files] == [r["path"] for r in rec["files"]]
        assert all(sha(r["path"]) == r["sha256"] for r in rec["files"])
        with np.load(p) as z:
            matrix = z["loges"].copy()
        dependencies[str(source)] = sha(source)
        dependencies[str(p)] = sha(p)
        return matrix
    files = raw_files(meta)
    assert len(files) == len(a["x"]), (mid, len(files), len(a["x"]))
    rows, src = [], []
    for f in files:
        x, digest = read_raw(f, meta["dataset"])
        rows.append(loges(x))
        src.append({"path": str(f.resolve()), "sha256": digest})
    matrix = np.asarray(rows)
    assert matrix.shape == (len(a["x"]), 256) and np.all(np.isfinite(matrix))
    p.parent.mkdir(exist_ok=True)
    np.savez_compressed(p, loges=matrix)
    write(source, {"machine_id": mid, "protocol_sha256": key, "cache_sha256": sha(p), "files": src,
                   "raw_count": len(files), "cached_count": len(a["x"]), "length_mismatch": 0})
    dependencies[str(source)] = sha(source)
    dependencies[str(p)] = sha(p)
    print(f"raw logES {mid}: {len(files)} acquisitions", flush=True)
    return matrix


def ae_scores(matrix, a, seed):
    train = matrix[a["train_indices"].astype(int)]
    center, scale = np.mean(train, axis=0), np.std(train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    x = (matrix-center)/scale
    train = x[a["train_indices"].astype(int)]
    rng = np.random.default_rng(seed)
    sd = np.sqrt(2./288)
    weights = [rng.normal(0, sd, (256, 32)), np.zeros(32), rng.normal(0, sd, (32, 256)), np.zeros(256)]
    m, v = [np.zeros_like(q) for q in weights], [np.zeros_like(q) for q in weights]
    losses = []
    for epoch in range(1, 201):
        w1, b1, w2, b2 = weights
        hidden = np.tanh(train@w1+b1)
        diff = hidden@w2+b2-train
        losses.append(float(np.mean(diff**2)))
        dy = 2*diff/train.size
        dz = (dy@w2.T)*(1-hidden**2)
        grads = [train.T@dz, np.sum(dz, axis=0), hidden.T@dy, np.sum(dy, axis=0)]
        for i, g in enumerate(grads):
            m[i] = .9*m[i]+.1*g
            v[i] = .999*v[i]+.001*g*g
            weights[i] -= .001*(m[i]/(1-.9**epoch))/(np.sqrt(v[i]/(1-.999**epoch))+1e-8)
    w1, b1, w2, b2 = weights
    scores = np.mean((np.tanh(x@w1+b1)@w2+b2-x)**2, axis=1)
    assert np.all(np.isfinite(scores))
    return scores, {"w1": w1, "b1": b1, "w2": w2, "b2": b2,
                    "normalisation_center": center, "normalisation_scale": scale,
                    "loss_before_each_epoch": np.asarray(losses)}


def search_originals():
    terms = ("loges", "wang", "toshkova", "javanmardi", "sigma", "published", "gev")
    content_terms = ("genextreme", "toshkova", "javanmardi", "loges")
    bases = [ROOT/"tmp", ROOT/"research", ROOT/"results", REPO/"collab"]
    matches, scanned = [], 0
    for base in bases:
        for dp, ds, fs in os.walk(base):
            ds[:] = [d for d in ds if d not in {".git", "node_modules", "__pycache__", ".venv", "venv"}
                     and (Path(dp)/d).resolve() not in {HERE, OUT}]
            for f in fs:
                p = Path(dp)/f
                scanned += 1
                name_hit = any(t in f.lower() for t in terms)
                content_hit = []
                if p.suffix == ".py" and p.stat().st_size < 2000000:
                    txt = p.read_text(encoding="utf-8", errors="replace").lower()
                    content_hit = [t for t in content_terms if t in txt]
                if name_hit or content_hit:
                    matches.append({"path": str(p.resolve()), "filename_match": name_hit,
                                    "content_terms": content_hit, "sha256": sha(p)})
    return {"roots": [str(p.resolve()) for p in bases], "files_scanned": scanned, "matches": matches,
            "original_09d_09e_found": False,
            "assessment": "Only older v67 logES PCA probes found: 128 bins, first 2048 samples, Hann window, PCA and fixed commissioning; not appendix 09e. No original 09d/09e runner or output identified; probes deliberately not rerun because they differ and write outside authorized directories."}


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h12-only", action="store_true", help="Save a provisional 09d result; full run required for final summary")
    args = ap.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    deps = {str(p): sha(p) for p in [HERE/"protocol.json", HERE/"run_comparison.py", CADENCE_PATH, WIDER_PATH, MANIFEST,
                                    ROOT/"memory.txt", REPO/"reviews/verification_2026-10-14.md", REPO/"paper/RESULTS_TRACEABILITY.md"]}
    search = search_originals()
    write(OUT/"original_search.json", search)
    manifest = read(MANIFEST)
    all_meta = sorted([m for m in manifest["devices"] if m["dataset"] != "ldm_upb"], key=lambda m:m["machine_id"])
    assert len(all_meta) == 54
    devices = [m for m in all_meta if m["truth"] is not None or m["truth_kind"] == "healthy_control"]
    assert len(devices) == 49 and sum(m["truth_kind"] == "healthy_control" for m in devices) == 6
    arrays = {}
    for meta in devices:
        p = meta["path"]
        assert sha(p) == manifest["files"][p], p
        deps[str(p)] = sha(p)
        arrays[meta["machine_id"]] = load_arrays(meta)
    subset = [m for m in devices if m["dataset"] in {"pronostia", "xjtu"}]
    assert len(subset) == 25 and all(m["truth"] is not None for m in subset)
    cells, score_paths = [], []
    frozen = read(STATES/"summary.json")
    frozen_by_seed = {c["seed"]: c for c in frozen["results"] if c["rule"] == "global_h12" and c["cadence"] == "dense"}
    deps[str(STATES/"summary.json")] = sha(STATES/"summary.json")
    (OUT/"scores").mkdir(exist_ok=True)
    for seed in SPEC["seeds"]:
        cp, sp = STATES/f"global_h12_seed{seed}_checkpoint.json", STATES/f"global_h12_seed{seed}.pkl"
        assert sha(sp) == read(cp)["state_sha256"]
        deps[str(cp)], deps[str(sp)] = sha(cp), sha(sp)
        with sp.open("rb") as f:
            state = pickle.load(f)
        assert len(state["clients"]) == 54 and not any(s.startswith("ldm_upb::") for s in state["clients"])
        rule_rows, flags_saved, scores_saved = {}, {}, {}
        for meta in devices:
            mid = meta["machine_id"]
            a = arrays[mid]
            scores = wider.score(state, mid, a["x"])
            scores_saved[mid.replace("::", "__")] = scores
            outputs = {}
            for sigma in (3, 4, 6):
                for persist in (False, True):
                    name = f"wang_{sigma}sigma" + ("_p13of21" if persist else "")
                    outputs[name] = wang(scores, a, sigma, persist)
            for regimes in (2, 1):
                thresholds, details = gev(scores, a, regimes, seed)
                for persist in (False, True):
                    name = ("gev_regime2" if regimes == 2 else "gev_no_clustering") + ("_p13of21" if persist else "")
                    outputs[name] = point_flags(scores, thresholds, a, persist), details
            outputs["javanmardi_2means"] = onset(scores, a, seed)
            outputs["project"] = cadence.dense_flags(scores, a["train_indices"], a["calibration_indices"]), {}
            for name, (flags, details) in outputs.items():
                row = {**evaluate(meta, a, flags), "fit": details}
                rule_rows.setdefault(name, []).append(row)
                flags_saved[mid.replace("::", "__")+"__"+name] = flags
            expected = next(r for r in frozen_by_seed[seed]["devices"] if r["machine_id"] == mid)
            actual = rule_rows["project"][-1]
            assert all(actual[k] == expected[k] for k in ("strict_success", "healthy_alarm", "fault_detected", "first_alarm_index")), (seed, mid)
        p = OUT/"scores"/f"h12_seed{seed}.npz"
        np.savez_compressed(p, **scores_saved)
        score_paths.append(str(p))
        np.savez_compressed(OUT/"scores"/f"h12_flags_seed{seed}.npz", **flags_saved)
        for rule, rows in rule_rows.items():
            add_cell(cells, "09d", "22feat_h12", rule, seed, rows)
            if rule in {"wang_3sigma", "project"}:
                add_cell(cells, "09e", "22feat_h12", rule, seed, [r for r in rows if r["dataset"] in {"pronostia", "xjtu"}])
        print("H12", seed, {k: metrics(v)["strict_successes"] for k, v in rule_rows.items()}, flush=True)
    write(OUT/"h12_provisional.json", {"complete": False, "cells": cells})
    if args.h12_only:
        return
    logrows = {(seed, rule): [] for seed in SPEC["seeds"] for rule in ("wang_3sigma", "project")}
    for meta in subset:
        mid = meta["machine_id"]
        a = arrays[mid]
        matrix = build_loges(meta, a, deps)
        for seed in SPEC["seeds"]:
            scores, state = ae_scores(matrix, a, seed)
            p = OUT/"scores"/(mid.replace("::", "__")+f"_loges_seed{seed}.npz")
            flags = {"wang_3sigma": wang(scores, a, 3)[0],
                     "project": cadence.dense_flags(scores, a["train_indices"], a["calibration_indices"])}
            np.savez_compressed(p, scores=scores, **state, **{f"flags_{k}": v for k,v in flags.items()})
            score_paths.append(str(p))
            for rule, values in flags.items():
                logrows[seed, rule].append(evaluate(meta, a, values))
            print("logES AE", mid, seed, {k: logrows[seed,k][-1]["strict_success"] for k in flags}, flush=True)
    for (seed, rule), rows in logrows.items():
        add_cell(cells, "09e", "loges_ae", rule, seed, rows)
    targets = {"09d/22feat_h12/wang_3sigma":16, "09d/22feat_h12/wang_3sigma_p13of21":39,
               "09d/22feat_h12/wang_4sigma_p13of21":42, "09d/22feat_h12/wang_6sigma_p13of21":39,
               "09d/22feat_h12/wang_4sigma":18, "09d/22feat_h12/wang_6sigma":23,
               "09d/22feat_h12/gev_regime2":20, "09d/22feat_h12/gev_no_clustering":20,
               "09d/22feat_h12/gev_regime2_p13of21":34, "09d/22feat_h12/javanmardi_2means":23,
               "09d/22feat_h12/project":45, "09e/loges_ae/wang_3sigma":0,
               "09e/loges_ae/project":20, "09e/22feat_h12/wang_3sigma":4, "09e/22feat_h12/project":23}
    comparison = []
    for c in cells:
        key = "/".join(c[k] for k in ("part", "front_end", "rule"))
        if key in targets:
            comparison.append({"cell": key, "seed": c["seed"], "logged": targets[key],
                               "new": c["metrics"]["strict_successes"],
                               "matches": c["metrics"]["strict_successes"] == targets[key]})
    elapsed = time.time()-started
    outputs = {str(p): sha(p) for p in OUT.rglob("*.npz")}
    summary = {"complete": True, "configuration": SPEC, "runtime_seconds": elapsed,
               "python": sys.version, "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "source_search": search, "dependencies_sha256": deps, "artifacts_sha256": outputs,
               "cohort": [{"machine_id": m["machine_id"], "dataset": m["dataset"], "truth": m["truth"], "truth_kind": m["truth_kind"], "counts": m["counts"]} for m in devices],
               "headline_device_parity_passed": True, "cells": cells, "logged_comparison": comparison,
               "historical_exact_reproduction_established": False,
               "reproduction_note": "Original scripts and settings missing; numerical matches alone do not establish exact historical reproduction. New frozen headline excludes LDM from model training; September 9 predated September 22 exclusion. Missing implementation choices are fixed openly in protocol.json; no outcome tuning performed."}
    write(OUT/"summary.json", summary)
    with (OUT/"counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["part", "front_end", "rule", "seed", *cells[0]["metrics"]])
        writer.writeheader()
        for c in cells:
            writer.writerow({k:c[k] for k in ("part", "front_end", "rule", "seed")} | c["metrics"])
    report(summary)
    print(f"COMPLETE {elapsed:.1f} seconds: {OUT/'summary.json'}", flush=True)


def report(s):
    lines = ["# Published decision rules on the development cohort", "",
             "This is a saved, deterministic reconstruction of memory appendices 2026-09-09d/e using the CURRENT frozen non-LDM headline detector. It is not an exact recovery of the lost September runners and is not a replication of the full published methods.", "",
             "## Reproduction status", "", s["reproduction_note"], "",
             "The historical single counts have no recorded seed. Every requested seed is reported below; no best seed is selected. The project reference matches the frozen summary for every scored device's strict outcome, healthy alarm, detection and first alarm index.", "",
             "| Part | Front end | Rule | Seed | Strict | Detected | Healthy-FA devices | Control alarms | Logged strict | Match |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    lookup = {(r["cell"],r["seed"]):r for r in s["logged_comparison"]}
    for c in s["cells"]:
        m = c["metrics"]
        comp = lookup.get(("/".join(c[k] for k in ("part","front_end","rule")), c["seed"]))
        logged = comp["logged"] if comp else "NR"
        match = ("yes" if comp["matches"] else "no") if comp else "-"
        lines.append(f'| {c["part"]} | {c["front_end"]} | {c["rule"]} | {c["seed"]} | {m["strict_successes"]}/{m["scored"]} | {m["faults_detected"]}/{m["faults"]} | {m["healthy_alarm_devices"]}/{m["scored"]} | {m["control_alarms"]}/{m["controls"]} | {logged} | {match} |')
    lines += ["", "## Method and unresolved historical settings", "",
              "Code and all fixed choices are in ../../research/published_methods_20261021/{run_comparison.py,protocol.json}. H12 scores are recomputed from saved global_h12_seed17070/71/72.pkl in results/healthy_only_causal_no_ldm_20260922 using the existing wider_models.score; H12 is never retrained. The fl_algorithm_comparison_20260929 report independently records bit-exact round-12 parity. The 49 scored devices are 43 fault lifecycles and six designated healthy controls; four controls are Paderborn snapshots, not whole-life controls. Five unknown-label clients are excluded from scoring, but remain part of the original 54-client H12 checkpoint. LDM is absent from the checkpoint client lists and evaluated caches.", "",
              "All cache partitions are reused unchanged. Strict success means no alarm anywhere in held-out healthy_test_indices and at least one alarm in unhealthy_test_indices, or silence for a control. Fault detection and healthy false alarms are separate, overlapping properties. Boundaries are proxy labels used to choose healthy partitions: this is not label-free training partition selection. Alarms are suppressed through calibration and persistence spans the continuous timeline without a fault-boundary reset.", "",
              "Wang-style rules use H12 reconstruction MSE (or logES AE MSE), mean + 3/4/6 population standard deviations of the first 100 training+calibration acquisitions, or first min(30, history length) if shorter. Giving thresholds persistence uses a full 21-acquisition window with at least 13 exceedances. The project rule directly calls cadence.dense_flags: twice the maximum historical rolling 13th-largest score, with the 0.995 calibration-quantile short-history fallback.", "",
              "Toshkova-style reconstruction fits two KMeans regimes to healthy 22-feature inputs, assigns every acquisition to a regime, takes blocks of 10 chronological baseline scores within regime, and uses scipy GEV MLE's 0.99 quantile. An unclustered version is also run. Short/degenerate baselines and invalid fitted levels use an explicitly recorded empirical 0.995 baseline-quantile fallback; these fallback cases are not GEV MLE estimates. Block size, return level, clustering input and fallback are absent from the appendix, so exact reproduction of 20/49 or 34/49 cannot be claimed. All fitted maxima, parameters, levels, warnings and fallbacks are saved per device.", "",
              "Javanmardi-style reconstruction applies retrospective full-trajectory 2-means to scalar H12 MSE, assumes the first label is healthy, and holds an alarm from the first different label at or after Tc=10. Tc=10, scalar input and held-alarm semantics are reconstruction assumptions; the appendix only specifies a separate Tc=50 sensitivity. It is not the published Spectrum-Principal-Energy-Vector front end. Future acquisitions affect its clusters, unlike the causal project alarm. The transition detector lacks an explicit abstention model and is task-mismatched to healthy-control silence; the historical log reports alarms on 6/6 controls. The new table reports actual control alarms rather than presuming that count.", "",
              "09e uses the 25 scored PRONOSTIA+XJTU lifecycles (no controls), retaining their original train/cal/healthy-test/fault-test indices. Raw file order is natural numeric order and is checked against cached acquisition count. Only the scored subset is built, rather than the log's 30 devices including unscored clients. Each raw source has a SHA-256 in loges/*.sources.json. No raw archive or data download is used.", "",
              "Each waveform channel is demeaned, Hilbert transformed, envelope-demeaned, transformed by unnormalised rFFT, restricted to 0..5000 Hz at 25.6 kHz, mean-pooled into 256 equal index bins, and log1p transformed. Channel spectra are combined by an elementwise maximum. The per-device 256-32-256 tanh/linear AE fits ONLY that device's healthy training rows, with training-only per-bin mean/std scaling, full-batch Adam for 200 epochs at 0.001 and each of the three seeds. Saved state includes weights, scaling and loss history. Channel aggregation, normalisation, exact binning, optimiser, training duration and seed were not recorded in 09e; these choices were frozen before obtaining results. This is a plain AE instead of the published VAE. Therefore rebuilt logES is not a full SimUFD replication.", "",
              "## Interpretation and caveats", "",
              "Numerical mismatches are retained, not tuned away. Candidate causes are the current no-LDM-trained H12 checkpoint (the September comparison predates its training), unspecified threshold baseline details, GEV and onset settings, and unspecified logES AE training/normalisation/channel choices. These are identifiable sources of ambiguity, not measured causal attributions for any particular difference. Without the original scripts, a unique explanation is unavailable. Do not replace the historical numbers with a claim of exact reproduction; cite this report as the new fully specified comparison.", "",
              "The 09d experiment compares adapted DECISION RULES on the project's HI, not complete published front ends. The 09e factorial rebuilds the documented logES operations but substitutes an AE for VAE and makes explicit implementation assumptions. Persistence is a generic temporal mechanism; equalising it is essential to interpretation. The project's configuration was selected across roughly 35 mechanisms on these devices, giving it a substantial home advantage over fixed published rules. The 4-sigma fairness arm is a sensitivity, not the published 3-sigma constant. No independent held-out dataset is introduced here.", "",
              "These device-level strict metrics differ from each original paper's within-unit protocol, operating conditions, datasets and false-alarm accounting. A healthy-period false alarm anywhere can fail an otherwise detected lifecycle. In particular, 09e's held-out healthy periods belong to fault lifecycles and are not healthy control machines. Any headline should include fault detection and healthy-period false-alarm counts, not strict success alone. The historical 24/25 detections and 24/25 healthy-FA devices for logES 3-sigma are reference claims; compare them with this run's saved counts rather than assuming they transfer.", "",
              "## Artifacts and rerun", "",
              "Run from any directory: `python -I -B E:/masters/autoencoder_cwru_release/full_pipeline_simulation/research/published_methods_20261021/run_comparison.py`. Run the saved-artifact check with the adjacent `test_summary.py` using the same flags. Imports disable bytecode writes and NumPy/BLAS threads are capped at one. No H12 training or original experiment output is invoked. Original search details are in original_search.json; the only related scripts identified are tmp/v67_loges_probe.py and tmp/v67_loges_paderborn.py, which differ materially and were not rerun.", "",
              "summary.json contains configuration, dependency hashes, per-seed/device outcomes, fits and logged-count comparisons. counts.csv is a compact export. scores/*.npz retains H12 scores, all H12 alarm vectors, logES model parameters and both logES alarm vectors. loges/*.npz retains computed representations; source manifests hash every raw file. No raw waveforms are copied.", "",
              f'Runtime: {s["runtime_seconds"]:.2f} seconds for the full successful invocation, including discovery, H12 inference, raw processing and per-device AE training. Versions: Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}.', ""]
    (OUT/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
