"""SEALED STAGE-2 RUNNER. Do not execute during stage 1.

Imports the paper's feature, normalizer, partition, onset, model and alarm code.
Only dataset adapters, integrity/gating, and reporting are implemented here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


class StopRun(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def verify_protocol(protocol_path):
    root = Path(protocol_path).resolve().parent
    if root != Path(r"D:\external_unseen").resolve():
        raise StopRun("This runner may execute only after delivery to D:\\external_unseen.")
    checks = {}
    for line in (root / "PROTOCOL_SHA256.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            checks[name.lstrip(" *")] = digest
    if set(checks) != {"PROTOCOL.md", "protocol.json"}:
        raise StopRun("Protocol seal must name exactly PROTOCOL.md and protocol.json")
    for name, digest in checks.items():
        if sha(root / name) != digest:
            raise StopRun(f"Protocol seal broken: {name}")
    p = read_json(protocol_path)
    if Path(p["output_root"]).resolve() != root / "results":
        raise StopRun("Output root differs from the frozen D: destination")
    for rel, digest in p["frozen_files_sha256"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise StopRun(f"Frozen artifact drift: {rel}")
    if sha(root / "download_sha256.txt") != p["download_manifest_sha256"]:
        raise StopRun("Download manifest drift")
    return root, p


def time_metadata(root, p):
    """Parse ONLY column 1. Never parse speed or any vibration field here."""
    g = p["snd"]["identity_gate"]
    out = {"schema": "unseen_time_identity_gate_v1", "units": [], "reasons": [],
           "mapping_status": "ASSUMED_MAPPING_NOT_CONTRADICTED_NOT_VERIFIED"}
    for unit in p["snd"]["units"]:
        times = []
        try:
            with Path(unit["file"]).open("rb") as f:
                for rownum, line in enumerate(f, 1):
                    # Prefix only: the remaining raw fields are neither decoded nor parsed.
                    prefix = line.split(b",", 1)[0].lstrip(b"\xef\xbb\xbf")
                    t = float(prefix)
                    if not math.isfinite(t):
                        raise ValueError(f"nonfinite time at row {rownum}")
                    times.append(t)
            if len(times) < 2:
                raise ValueError("fewer than two time rows")
            gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
            bad = [i + 2 for i, d in enumerate(gaps) if d <= 0]
            median_gap = statistics.median([d for d in gaps if d > 0]) if any(d > 0 for d in gaps) else None
            cut = max(g["gap_min_years"], g["gap_median_multiplier"] * (median_gap or 0))
            gap_rows = [{"before_csv_row_1based": i + 1, "after_csv_row_1based": i + 2,
                         "before_years": times[i], "after_years": times[i + 1], "gap_years": d}
                        for i, d in enumerate(gaps) if d > cut]
            signatures = {}
            for event in g["event_years"]:
                near = [q for q in gap_rows if q["before_years"] <= event + g["event_tolerance_years"]
                        and q["after_years"] >= event - g["event_tolerance_years"]]
                signatures[str(event)] = near
            m = {"turbine": unit["turbine"], "file": unit["file"], "rows": len(times),
                 "time_min_years": min(times), "time_max_years": max(times),
                 "first_years": times[0], "last_years": times[-1],
                 "median_positive_gap_years": median_gap, "gap_cutoff_years": cut,
                 "large_gaps": gap_rows, "nonincreasing_at_rows_1based": bad,
                 "event_gap_signatures": signatures}
            out["units"].append(m)
            if bad:
                out["reasons"].append(f"Turbine {unit['turbine']}: time reset, duplicate or reversal; chronology/mapping unresolved")
            if not (times[0] < min(g["event_years"]) and times[-1] > max(g["event_years"])):
                out["reasons"].append(f"Turbine {unit['turbine']}: time does not bracket both documented events")
        except (ValueError, OSError) as e:
            out["reasons"].append(f"Turbine {unit['turbine']}: invalid time metadata: {e}")
    if len(out["units"]) == 6:
        both = [u["turbine"] for u in out["units"] if all(u["event_gap_signatures"].values())]
        if both and 5 not in both:
            out["reasons"].append(f"Event-gap evidence points to turbine(s) {both}, but not assumed Turbine 5; stop, do not remap")
    if out["reasons"]:
        out["mapping_status"] = "STOP_METADATA_CONTRADICTION_OR_UNRESOLVED_CHRONOLOGY"
    out["caveat"] = "Times and gaps cannot establish turbine identity. Absence of event gaps is not proof against the assumed mapping. No automatic permutation or cutoff adjustment is allowed."
    return out


def frozen_imports(root, p):
    frozen = root / "frozen" / "pipeline"
    sys.path[:0] = [str(frozen / "research/healthy_only_causal_20260908"),
                    str(frozen / "research/healthy_only_suite_20260907"), str(frozen)]
    names = ["raw_online_federated_v12", "wider_models", "cadence", "data_cache",
             "thesis_70_device_federated", "frozen_v19_analytic_onset", "frozen_v19_eda_features"]
    modules = {name: importlib.import_module(name) for name in names}
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(frozen):
            raise StopRun(f"Import escaped frozen pipeline: {name}")
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(Path(p["source_pipeline_root"]).parent):
            raise StopRun(f"Import used live E: source: {name}")
        if filename and Path(filename).resolve().is_relative_to(root / "frozen"):
            rel = Path(filename).resolve().relative_to(root).as_posix()
            if rel not in p["frozen_files_sha256"]:
                raise StopRun(f"Unsealed local import: {name}, {rel}")
    if modules["data_cache"].FEATURE_INDICES.tolist() != p["features"]["indices22"]:
        raise StopRun("Feature selection drift")
    return modules


def check_data_integrity(p):
    for item in p["target_files"]:
        if sha(item["path"]) != item["sha256"]:
            raise StopRun(f"Target file drift: {item['path']}")


def load_models(root, p, np):
    states = {}
    for item in p["models"]:
        with (root / item["frozen_relative_path"]).open("rb") as f:
            s = pickle.load(f)  # trusted local checkpoint, already SHA-256 checked
        if s["seed"] != item["seed"] or s["variant"] != "global_h12" or len(s["clients"]) != 54:
            raise StopRun("Checkpoint identity mismatch")
        if len(set(s["clients"])) != 54 or any(c.split("::")[0] not in p["training"]["datasets"] for c in s["clients"]):
            raise StopRun("Checkpoint cohort mismatch")
        shapes = [list(a.shape) for a in s["global"]]
        if shapes != [[22, 12], [12], [12, 22], [22]] or sum(a.size for a in s["global"]) != 562:
            raise StopRun("Checkpoint architecture mismatch")
        states[item["seed"]] = s
    return states


def snd_features(unit, p, mods, np):
    """One CSV row is one acquisition. Exclude replacement rows before parsing vibration."""
    v12 = mods["raw_online_federated_v12"]
    eda = mods["frozen_v19_eda_features"]
    rows, times, rms, features, onset = [], [], [], [], []
    raw_count = excluded_replacement = 0
    with Path(unit["file"]).open("r", encoding="utf-8-sig", newline="") as f:
        for rownum, row in enumerate(csv.reader(f), 1):
            raw_count += 1
            if len(row) != p["snd"]["columns"]:
                raise StopRun(f"SND schema mismatch in {unit['file']} row {rownum}: {len(row)} fields")
            t = float(row[0])
            if unit["turbine"] == 5 and t >= p["snd"]["bearing_replacement_years"]:
                excluded_replacement += 1
                continue
            signal = np.asarray(row[2:], dtype=np.float64)
            if len(signal) != 16384 or not np.all(np.isfinite(signal)):
                raise StopRun(f"Invalid vibration row in {unit['file']} row {rownum}")
            # The only additional numeric calculation is the authors' prespecified operating gate.
            r = float(np.sqrt(np.mean(signal * signal)))
            channels = signal.reshape(1, -1)
            features.append(v12.local_preprocess_current_acquisition(channels))
            if unit["turbine"] == 5:
                onset.append(eda.acquisition_features(channels))
            rows.append(rownum)
            times.append(t)
            rms.append(r)
    return {"features": np.asarray(features, dtype=np.float32), "onset_features": onset,
            "times": np.asarray(times), "csv_rows": np.asarray(rows), "rms": np.asarray(rms),
            "raw_rows": raw_count, "excluded_replacement_rows": excluded_replacement}


def make_snd_partitions(data, unit, arm, p, mods, np, pd):
    keep = data["rms"] > p["snd"]["rms_gate_g"] if arm == "authors_rms_gt_0p5g" else np.ones(len(data["times"]), bool)
    positions = np.flatnonzero(keep)
    if not len(positions):
        raise StopRun(f"No eligible records: Turbine {unit['turbine']}, {arm}")
    onset_info = None
    onset_index = None
    if unit["turbine"] == 5:
        frame = pd.DataFrame([data["onset_features"][int(i)] for i in positions])
        onset_info = mods["frozen_v19_analytic_onset"].estimate(frame)
        onset_index = onset_info.get("onset_index")
        if onset_index is None:
            raise StopRun(f"No historical onset for Turbine 5 ({arm}): {onset_info['analysis_status']}; no substitute onset or strict denominator change allowed")
    healthy_end = len(positions) if onset_index is None else int(onset_index)
    ti, ci, hi = mods["thesis_70_device_federated"]._three_way(
        np.arange(healthy_end, dtype=np.int64), p["partitions"]["minimum_each_healthy_partition"])
    fi = np.arange(healthy_end, len(positions), dtype=np.int64)
    if unit["turbine"] == 5 and not len(fi):
        raise StopRun("No fault test rows")
    return positions, ti, ci, hi, fi, onset_info


def score_unit(mid, features100, ti, ci, hi, fi, coordinate, source_rows,
               states, mods, p, np, results_root, labels, extra):
    # EXACT training-only normalization and imported final-global inference.
    x = mods["data_cache"].physical_matrix([features100], ti)
    outputs = []
    for seed in p["seeds"]:
        scores = mods["wider_models"].score(states[seed], mid, x)
        flags = mods["cadence"].dense_flags(scores, ti, ci, p["alarm"]["m"],
                                          p["alarm"]["k"], p["alarm"]["alpha"])
        hist = np.concatenate((ti, ci))
        hist_stat = mods["cadence"]._rolling(scores[hist], p["alarm"]["m"], p["alarm"]["k"])
        finite = hist_stat[np.isfinite(hist_stat)]
        reference = float(np.max(finite)) if len(finite) else float(np.quantile(scores[ci], p["alarm"]["fallback_quantile"]))
        first = np.flatnonzero(flags)
        first_index = int(first[0]) if len(first) else None
        fault_hits = np.flatnonzero(flags[fi])
        healthy_alarm = bool(np.any(flags[hi]))
        endpoint = labels["endpoint"]
        result = {**labels, "unit": mid, "seed": seed, "n_rows": len(x),
                  "training_rows": len(ti), "calibration_rows": len(ci),
                  "healthy_test_rows": len(hi), "fault_test_rows": len(fi),
                  "healthy_alarm": healthy_alarm if endpoint in ("fault", "control") else None,
                  "fault_detected": bool(len(fault_hits)) if endpoint == "fault" else None,
                  "strict_success": (not healthy_alarm and (bool(len(fault_hits)) if endpoint == "fault" else True)) if endpoint in ("fault", "control") else None,
                  "first_alarm_index_0based": first_index,
                  "first_alarm_source_row_or_day_1based": int(source_rows[first_index]) if first_index is not None else None,
                  "first_alarm_coordinate": float(coordinate[first_index]) if first_index is not None else None,
                  "reference": reference, "threshold": p["alarm"]["alpha"] * reference,
                  "threshold_method": "history_max_persistent" if len(finite) else "calibration_point_quantile_0.995",
                  **extra}
        if labels["dataset"] == "SND" and endpoint == "fault":
            t = result["first_alarm_coordinate"]
            result["first_alarm_minus_replacement_years"] = t - p["snd"]["bearing_replacement_years"] if t is not None else None
            result["lead_to_replacement_years"] = p["snd"]["bearing_replacement_years"] - t if t is not None else None
            result["first_alarm_before_estimated_onset"] = bool(first_index < len(ti) + len(ci) + len(hi)) if first_index is not None else None
        if labels["dataset"] == "MathWorks_HSS":
            day = result["first_alarm_source_row_or_day_1based"]
            result["first_alarm_day"] = day
            result["precedes_failure_day50"] = bool(day < 50) if day is not None else False
            result["lead_in_record_days_to_day50"] = 50 - day if day is not None else None
            if result["threshold_method"] != "calibration_point_quantile_0.995":
                raise StopRun("HSS must use the frozen short-history fallback")
        safe_mid = mid.replace("::", "__")
        ti_set, ci_set, hi_set, fi_set = set(ti), set(ci), set(hi), set(fi)
        score_rows = [{"index_0based": i, "source_row_or_day_1based": int(source_rows[i]),
                       "coordinate": float(coordinate[i]), "partition": "training" if i in ti_set else "calibration" if i in ci_set else "healthy_test" if i in hi_set else "fault_test" if i in fi_set else "test",
                       "score": float(scores[i]), "alarm": bool(flags[i])}
                      for i in range(len(x))]
        write_csv(results_root / "scores" / f"{safe_mid}_seed{seed}.csv", score_rows)
        outputs.append(result)
    return outputs


def report(results_root, p, rows, environment, completed):
    primary, sensitivity, hss = {}, {}, []
    for arm, target in [(p["snd"]["arms"][0], primary), (p["snd"]["arms"][1], sensitivity)]:
        for seed in p["seeds"]:
            selected = [r for r in rows if r["dataset"] == "SND" and r["arm"] == arm and r["seed"] == seed]
            target[str(seed)] = {"strict_successes": sum(r["strict_success"] for r in selected),
                                 "observed_units": len(selected), "registered_units": 6,
                                 "complete": len(selected) == 6,
                                 "turbine5": next((r for r in selected if r["endpoint"] == "fault"), None)}
        target["total"] = {"strict_successes": sum(target[str(s)]["strict_successes"] for s in p["seeds"]),
                           "observed_unit_seed_evaluations": sum(target[str(s)]["observed_units"] for s in p["seeds"]),
                           "registered_unit_seed_evaluations": 18,
                           "independent_machines": 6, "independent_farms": 1}
    hss = [r for r in rows if r["dataset"] == "MathWorks_HSS"]
    summary = {"complete": completed, "protocol_id": p["protocol_id"], "primary": primary,
               "sensitivity_no_operating_filter": sensitivity, "secondary_hss": hss,
               "environment": environment,
               "mapping_status": "ASSUMED_MAPPING_NOT_CONTRADICTED_NOT_VERIFIED",
               "assumptions": {"filename_mapping": p["snd"]["mapping"],
                               "replacement_rule": p["snd"]["replacement_rule"],
                               "onset": "retrospective feature departure, not independent damage truth",
                               "hss_failure": "day50 terminal proxy; exact failure time unknown"},
               "interpretation": "Seed repetitions and six turbines in one farm are correlated. SND onset is a retrospective vibration departure proxy; strict success is conditional on that partition. HSS has no strict score. Incomplete outputs cannot be quoted as complete results."}
    write_json(results_root / "summary.json", summary)
    write_csv(results_root / "per_unit.csv", rows)
    text = ["# Frozen unseen-machine evaluation", "", f"Complete: {completed}", "",
            "| SND arm | Seed | Strict successes | Registered units | Processed units |",
            "|---|---:|---:|---:|---:|"]
    for arm, target in [("Primary RMS > 0.5 g", primary), ("Sensitivity no filter", sensitivity)]:
        for seed in p["seeds"]:
            r = target[str(seed)]
            text.append(f"| {arm} | {seed} | {r['strict_successes']} | 6 | {r['observed_units']} |")
        text.append(f"| {arm} pooled unit-seed evaluations | all | {target['total']['strict_successes']} | 18 | {target['total']['observed_unit_seed_evaluations']} |")
    text += ["", "Turbine 5: first alarm minus replacement time is negative for an early warning; every analysed row precedes the fixed 1.2-year cutoff.",
             "", "| Turbine 5 arm | Seed | First alarm years | Alarm minus1.2 years | Lead to replacement years | Healthy alarm | Fault detected |", "|---|---:|---:|---:|---:|---|---|"]
    for r in rows:
        if r["dataset"] == "SND" and r["endpoint"] == "fault":
            text.append(f"| {r['arm']} | {r['seed']} | {r['first_alarm_coordinate']} | {r['first_alarm_minus_replacement_years']} | {r['lead_to_replacement_years']} | {r['healthy_alarm']} | {r['fault_detected']} |")
    text += ["", "Filename-to-turbine mapping remains ASSUMED, not publisher-verified. The time-only check found no preregistered contradiction. Replacement is a nominal approximate 1.2-year cutoff.",
             "", "Turbine 2's documented early electrical sensor issue was retained, with no targeted exclusion or correction.",
             "", "| HSS seed | First alarm record-day | Precedes day-50 endpoint |", "|---:|---:|---|"]
    for r in hss:
        text.append(f"| {r['seed']} | {r['first_alarm_day']} | {r['precedes_failure_day50']} |")
    text += ["", "HSS days are ordered record-days (both March 17 records retained); actual filename timestamps are in hss_record_days.csv. Failure by day 50 is a terminal proxy, not a precisely timestamped failure event.",
             "", summary["interpretation"], "", "See metadata_identity_check.json, partition_onset_audit.json, source/model hashes in protocol.json, and execution_environment.json for provenance."]
    (results_root / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--metadata-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root, p = verify_protocol(args.protocol)
    results_root = Path(p["output_root"])
    results_root.mkdir(exist_ok=True)
    if (results_root / "ANALYSIS_STARTED.json").exists():
        raise StopRun("Analysis has already started: no automatic rerun, overwrite, rescue or resume")
    gate = time_metadata(root, p)
    previous_gate = results_root / "metadata_identity_check.json"
    if previous_gate.exists() and read_json(previous_gate) != gate:
        raise StopRun("Metadata gate changed since its first execution")
    write_json(previous_gate, gate)
    if gate["reasons"]:
        (results_root / "REPORT.md").write_text("# STOP before vibration analysis\n\n" + "\n".join(gate["reasons"]) + "\n", encoding="utf-8")
        with (root / "DEVIATIONS.md").open("a", encoding="utf-8") as f:
            f.write(f"\n- {datetime.now(timezone.utc).isoformat()} METADATA STOP: {'; '.join(gate['reasons'])}. No vibration fields parsed.\n")
        raise StopRun("Metadata identity/chronology gate failed; do not analyse vibration")
    if args.metadata_only:
        print("Metadata gate written. Mapping remains an assumption; no vibration fields were parsed.")
        return
    # Hash opaque source bytes only after the time-only gate. No sample statistics.
    check_data_integrity(p)
    mods = frozen_imports(root, p)
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat, whosmat
    import importlib.metadata
    environment = {"python": sys.version, "utc": datetime.now(timezone.utc).isoformat(),
                   "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn"]},
                   "threads": {name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
                   "protocol_json_sha256": sha(root / "protocol.json")}
    write_json(results_root / "execution_environment.json", environment)
    states = load_models(root, p, np)
    (results_root / "scores").mkdir(exist_ok=True)
    write_json(results_root / "ANALYSIS_STARTED.json", environment)
    rows, audits = [], []
    try:
        # SND first. Each arm uses exactly its own eligible rows and its own onset/partitions.
        for unit in p["snd"]["units"]:
            data = snd_features(unit, p, mods, np)
            for arm in p["snd"]["arms"]:
                pos, ti, ci, hi, fi, onset = make_snd_partitions(data, unit, arm, p, mods, np, pd)
                audit = {"turbine": unit["turbine"], "arm": arm, "raw_rows": data["raw_rows"],
                         "excluded_replacement_rows": data["excluded_replacement_rows"],
                         "excluded_operating_filter_rows": len(data["times"]) - len(pos),
                         "onset": onset, "onset_years": float(data["times"][pos[int(onset["onset_index"])]]) if onset else None,
                         "partitions_csv_rows_1based": {name: data["csv_rows"][pos[idx]].tolist() for name, idx in [("train", ti), ("calibration", ci), ("healthy_test", hi), ("fault_test", fi)]}}
                audits.append(audit)
                write_json(results_root / "partition_onset_audit.json", audits)
                mid = f"SND::Turbine{unit['turbine']}::{arm}"
                rows.extend(score_unit(mid, data["features"][pos], ti, ci, hi, fi,
                                       data["times"][pos], data["csv_rows"][pos], states, mods, p, np, results_root,
                                       {"dataset": "SND", "arm": arm, "turbine": unit["turbine"], "endpoint": "fault" if unit["turbine"] == 5 else "control"},
                                       {"estimated_onset_years": audit["onset_years"], "bearing_replacement_years": 1.2 if unit["turbine"] == 5 else None}))
                report(results_root, p, rows, environment, False)
            del data
        # Secondary case: select ONLY vibration; tach is never loaded or used.
        feature_rows, days = [], []
        for day, item in enumerate(p["hss"]["records"], 1):
            path = Path(item["path"])
            header = {name: (shape, dtype) for name, shape, dtype in whosmat(path)}
            expected = tuple(item["vibration_shape"])
            if "vibration" not in header or header["vibration"][0] != expected:
                raise StopRun(f"HSS MAT schema drift: {path.name}")
            signal = np.asarray(loadmat(path, variable_names=["vibration"])["vibration"], dtype=np.float64)
            if signal.ndim != 2 or min(signal.shape) != 1 or not np.all(np.isfinite(signal)):
                raise StopRun(f"HSS requires a finite single-channel vector: {path.name}")
            feature_rows.append(mods["raw_online_federated_v12"].local_preprocess_current_acquisition(signal.reshape(1, -1)))
            days.append({"record_day_1based": day, "filename": path.name, "timestamp_utc_from_filename": item["timestamp_utc"]})
        if len(days) != 50:
            raise StopRun("HSS must contain all 50 records")
        write_csv(results_root / "hss_record_days.csv", days)
        train_days = p["hss"]["train_days_inclusive"]
        cal_days = p["hss"]["calibration_days_inclusive"]
        ti = np.arange(train_days[0] - 1, train_days[1], dtype=np.int64)
        ci = np.arange(cal_days[0] - 1, cal_days[1], dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        rows.extend(score_unit("MathWorks_HSS::bearing", np.asarray(feature_rows, dtype=np.float32),
                               ti, ci, empty, empty, np.arange(1, 51), np.arange(1, 51),
                               states, mods, p, np, results_root,
                               {"dataset": "MathWorks_HSS", "arm": "secondary", "turbine": None, "endpoint": "case_study"},
                               {"test_start_record_day": 21, "test_end_record_day": 50}))
        report(results_root, p, rows, environment, True)
        write_json(results_root / "COMPLETE.json", {"utc": datetime.now(timezone.utc).isoformat(), "unit_seed_rows": len(rows)})
    except Exception as e:
        report(results_root, p, rows, environment, False)
        write_json(results_root / "STOP.json", {"exception": type(e).__name__, "reason": str(e), "completed_unit_seed_rows": len(rows)})
        with (root / "DEVIATIONS.md").open("a", encoding="utf-8") as f:
            f.write(f"\n- {datetime.now(timezone.utc).isoformat()} STOP: {type(e).__name__}: {e}. No automated rerun authorized.\n")
        with (results_root / "REPORT.md").open("a", encoding="utf-8") as f:
            f.write(f"\nSTOP: {type(e).__name__}: {e}\n")
        raise


if __name__ == "__main__":
    main()
