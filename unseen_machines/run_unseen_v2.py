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


class NotEvaluableStop(StopRun):
    def __init__(self, reason, eligible_rows, onset=None):
        super().__init__("Not evaluable; STOP: " + reason)
        self.eligible_rows = eligible_rows
        self.onset = onset


ARM_A = "authors_rms_gt_0p5g"
ARM_B = "variable_speed_operating_window"
SENSITIVITY = "no_operating_filter"
PARTITION_NAMES = ("train", "calibration", "healthy_test", "fault_test")


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
    if set(checks) != {"PROTOCOL.md", "protocol.json", "PROTOCOL_v1.md", "protocol_v1.json"}:
        raise StopRun("Protocol seal must name current v2 and preserved v1 MD/JSON")
    for name, digest in checks.items():
        if sha(root / name) != digest:
            raise StopRun(f"Protocol seal broken: {name}")
    p = read_json(protocol_path)
    if p.get("version") != 2 or p.get("amendment", {}).get("number") != 1:
        raise StopRun("This runner requires frozen protocol v2 / Amendment 1")
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
            if unit["turbine"] == 5:
                event2 = unit.get("event") == "event2"
                in_segment = (p["snd"]["bearing_replacement_years"] < t < p["snd"]["gearbox_replacement_years"]) if event2 else t < p["snd"]["bearing_replacement_years"]
                if not in_segment:
                    excluded_replacement += 1
                    continue
            signal = np.asarray(row[2:], dtype=np.float64)
            if len(signal) != 16384 or not np.all(np.isfinite(signal)):
                raise StopRun(f"Invalid vibration row in {unit['file']} row {rownum}")
            # Arm A eligibility; speed is read separately for Arm B inclusion only.
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
    if arm not in (ARM_A, SENSITIVITY):
        raise StopRun("Arm B must inherit Arm A onset and partitions")
    keep = data["rms"] > p["snd"]["rms_gate_g"] if arm == ARM_A else np.ones(len(data["times"]), bool)
    positions = np.flatnonzero(keep)
    if not len(positions):
        raise NotEvaluableStop(f"No eligible records: Turbine {unit['turbine']}, {arm}", 0)
    onset_info = None
    onset_index = None
    if unit["turbine"] == 5:
        frame = pd.DataFrame([data["onset_features"][int(i)] for i in positions])
        onset_info = mods["frozen_v19_analytic_onset"].estimate(frame)
        onset_index = onset_info.get("onset_index")
        if onset_index is None:
            raise NotEvaluableStop(f"No historical onset for Turbine 5 {unit.get('event', 'event1')} ({arm}): {onset_info['analysis_status']}; no substitute onset or strict denominator change allowed", len(positions), onset_info)
    healthy_end = len(positions) if onset_index is None else int(onset_index)
    if healthy_end < 3 * p["partitions"]["minimum_each_healthy_partition"]:
        raise NotEvaluableStop(f"Only {healthy_end} pre-onset/healthy rows: Turbine {unit['turbine']}, {arm}", len(positions), onset_info)
    ti, ci, hi = mods["thesis_70_device_federated"]._three_way(
        np.arange(healthy_end, dtype=np.int64), p["partitions"]["minimum_each_healthy_partition"])
    fi = np.arange(healthy_end, len(positions), dtype=np.int64)
    if unit["turbine"] == 5 and not len(fi):
        raise StopRun("No fault test rows")
    return positions, ti, ci, hi, fi, onset_info


def read_snd_speeds(unit, source_rows):
    """Interpret column 2 only for the requested Arm A source rows; no signal parsing."""
    requested = {int(r) for r in source_rows}
    speeds = {}
    with Path(unit["file"]).open("r", encoding="utf-8-sig", newline="") as f:
        for rownum, line in enumerate(f, 1):
            if rownum not in requested:
                continue
            fields = line.split(",", 2)
            speed = float(fields[1])
            if not math.isfinite(speed):
                raise StopRun(f"Nonfinite speed: Turbine {unit['turbine']}, row {rownum}")
            speeds[rownum] = speed
    if set(speeds) != requested:
        raise StopRun("Missing requested Arm A speed rows")
    return speeds


def make_speed_window(data, unit, base, p, np):
    """Filter fixed Arm A memberships, then remap indices onto the kept causal stream."""
    pos, ti, ci, hi, fi, onset = base
    training_source_rows = data["csv_rows"][pos[ti]]
    training_speeds = read_snd_speeds(unit, training_source_rows)
    low, high = np.percentile([training_speeds[int(r)] for r in training_source_rows],
                              [25, 75], method="linear")
    # No calibration/test speed contributes to these frozen window bounds.
    other_rows = data["csv_rows"][pos[np.concatenate((ci, hi, fi))]]
    speeds = {**training_speeds, **read_snd_speeds(unit, other_rows)}
    speed = np.asarray([speeds[int(r)] for r in data["csv_rows"][pos]], dtype=float)
    kept_base_indices = np.flatnonzero((speed >= low) & (speed <= high))
    parts = tuple(np.flatnonzero(np.isin(kept_base_indices, idx)) for idx in (ti, ci, hi, fi))
    counts = dict(zip(PARTITION_NAMES, map(len, parts)))
    minimum = p["partitions"]["minimum_each_healthy_partition"]
    insufficient = [f"{name} keeps {counts[name]} < {minimum}" for name in PARTITION_NAMES[:3]
                    if counts[name] < minimum]
    if unit["turbine"] == 5 and counts["fault_test"] < 1:
        insufficient.append("fault_test keeps 0 < 1")
    window = {"speed_unit": "cycles/min", "p25": float(low), "p75": float(high),
              "quantile_method": "linear", "inclusive": True,
              "fitted_from": "Arm A training rows only", "kept_partition_counts": counts,
              "base_partition_counts": dict(zip(PARTITION_NAMES, map(len, (ti, ci, hi, fi)))),
              "status": "not evaluable" if insufficient else "evaluable",
              "reason": "; ".join(insufficient) if insufficient else None}
    return (pos[kept_base_indices], *parts, onset), window


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
                  "evaluation_status": "evaluable",
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
            replacement = labels["replacement_years"]
            result["first_alarm_minus_replacement_years"] = t - replacement if t is not None else None
            result["lead_to_replacement_years"] = replacement - t if t is not None else None
            result["lead_time_days"] = (replacement - t) * 365.25 if t is not None else None
            result["lead_time_basis"] = "days (365.25 per source year)"
            result["first_alarm_in_fault_period"] = bool(first_index in fi) if first_index is not None else False
            result["first_alarm_before_estimated_onset"] = bool(first_index < len(ti) + len(ci) + len(hi)) if first_index is not None else None
        if labels["dataset"] == "MathWorks_HSS":
            day = result["first_alarm_source_row_or_day_1based"]
            result["first_alarm_day"] = day
            result["precedes_failure_day50"] = bool(day < 50) if day is not None else False
            result["lead_in_record_days_to_day50"] = 50 - day if day is not None else None
            result["lead_time_days"] = result["lead_in_record_days_to_day50"]
            result["lead_time_basis"] = "record-days"
            result["first_alarm_in_fault_period"] = None  # v1 has no HSS fault partition
            result["fault_period_status"] = "not defined in v1; day-50 terminal proxy only"
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
    arms = {}
    for arm in (ARM_A, ARM_B, SENSITIVITY):
        target = {}
        for seed in p["seeds"]:
            selected = [r for r in rows if r["dataset"] == "SND" and r["population"] == "primary"
                        and r["arm"] == arm and r["seed"] == seed]
            evaluable = [r for r in selected if r["evaluation_status"] == "evaluable"]
            target[str(seed)] = {"strict_successes": sum(r["strict_success"] is True for r in selected),
                                 "observed_units": len(selected), "evaluable_units": len(evaluable),
                                 "not_evaluable_units": len(selected) - len(evaluable), "registered_units": 6,
                                 "complete": len(selected) == 6 and len(evaluable) == 6,
                                 "controls_silent": sum(r["turbine"] != 5 and r["strict_success"] is True for r in selected),
                                 "registered_controls": 5,
                                 "turbine5": next((r for r in selected if r["turbine"] == 5), None)}
        target["total"] = {"strict_successes": sum(target[str(s)]["strict_successes"] for s in p["seeds"]),
                           "observed_unit_seed_evaluations": sum(target[str(s)]["observed_units"] for s in p["seeds"]),
                           "evaluable_unit_seed_evaluations": sum(target[str(s)]["evaluable_units"] for s in p["seeds"]),
                           "registered_unit_seed_evaluations": 18,
                           "independent_machines": 6, "independent_farms": 1}
        arms[arm] = target
    hss = [r for r in rows if r["dataset"] == "MathWorks_HSS"]
    event2 = [r for r in rows if r.get("event") == "event2"]
    summary = {"complete": completed, "protocol_id": p["protocol_id"], "protocol_version": 2,
               "primary": arms[ARM_A], "primary_arms": {a: arms[a] for a in (ARM_A, ARM_B)},
               "sensitivity_no_operating_filter": arms[SENSITIVITY],
               "secondary_event2": event2, "secondary_hss": hss, "environment": environment,
               "mapping_status": "ASSUMED_MAPPING_NOT_CONTRADICTED_NOT_VERIFIED",
               "assumptions": {"filename_mapping": p["snd"]["mapping"],
                               "replacement_rule": p["snd"]["replacement_rule"],
                               "onset": "retrospective feature departure, not independent damage truth",
                               "hss_failure": "day50 terminal proxy; exact failure time unknown"},
               "interpretation": "Seed repetitions and six turbines in one farm are correlated. SND onset is a retrospective vibration departure proxy; strict success is conditional on that partition. HSS has no strict score or fault partition. Not-evaluable units remain in registered denominators; their outcomes are null."}
    write_json(results_root / "summary.json", summary)
    write_csv(results_root / "per_unit.csv", rows)

    def cell(row, key):
        if row is None:
            return "not processed"
        if row.get("evaluation_status") == "not evaluable":
            return "not evaluable"
        value = row.get(key)
        return "null" if value is None else str(value)

    names = {ARM_A: "A: RMS > 0.5 g", ARM_B: "B: variable-speed operating window", SENSITIVITY: "No-filter sensitivity"}
    text = ["# Frozen unseen-machine evaluation", "", f"Complete: {completed}; protocol v2, Amendment 1.", "",
            "| Seed | Arm | Strict successes /6 | T5 event-1 fault detected | Event-1 lead (days) | First alarm in fault period | Controls silent /5 | Evaluable /6 | Processed /6 |",
            "|---:|---|---:|---|---:|---|---:|---:|---:|"]
    def primary_line(arm, seed):
        r = arms[arm][str(seed)]
        t5 = r["turbine5"]
        return f"| {seed} | {names[arm]} | {r['strict_successes']}/6 | {cell(t5, 'fault_detected')} | {cell(t5, 'lead_time_days')} | {cell(t5, 'first_alarm_in_fault_period')} | {r['controls_silent']}/5 | {r['evaluable_units']}/6 | {r['observed_units']}/6 |"
    for seed in p["seeds"]:
        for arm in (ARM_A, ARM_B):
            text.append(primary_line(arm, seed))
    text += ["", "| Secondary unit | Seed | Arm | Status / applicability | Strict success | Healthy alarm | Fault detected | First alarm (years / record-day) | Lead (days / record-days) | First alarm in fault period | Before day 50 |",
             "|---|---:|---|---|---|---|---|---:|---:|---|---|"]
    for dataset, event, label in (("SND", "event2", "T5 event 2: planet bearing / gearbox"),
                                  ("MathWorks_HSS", None, "HSS")):
        for seed in p["seeds"]:
            for arm in (ARM_A, ARM_B):
                r = next((r for r in rows if r["dataset"] == dataset and r.get("event") == event
                          and r["seed"] == seed and r["arm"] == arm), None)
                status = "not processed" if r is None else r.get("arm_applicability", r["evaluation_status"])
                strict = "not applicable" if dataset == "MathWorks_HSS" else cell(r, "strict_success")
                alarm_key = "first_alarm_day" if dataset == "MathWorks_HSS" else "first_alarm_coordinate"
                fault_period = "not defined" if dataset == "MathWorks_HSS" else cell(r, "first_alarm_in_fault_period")
                before = cell(r, "precedes_failure_day50") if dataset == "MathWorks_HSS" else "not applicable"
                text.append(f"| {label} | {seed} | {names[arm]} | {status} | {strict} | {cell(r, 'healthy_alarm')} | {cell(r, 'fault_detected')} | {cell(r, alarm_key)} | {cell(r, 'lead_time_days')} | {fault_period} | {before} |")
    text += ["", "Event 2 begins with a new output-shaft bearing. The accelerometer is on the output-shaft bearing housing, not on the failed planet bearing; event 2 is a harder, indirect test. It is separate from the six-turbine primary denominator.",
             "", "HSS Arm B: not applicable (no speed column); its results are identical to Arm A. Tach-derived speed is prohibited. HSS has no defined healthy/fault partition; first-alarm membership in a fault period is not applicable.",
             "", "## No-filter sensitivity", "",
             "| Seed | Arm | Strict successes /6 | T5 event-1 fault detected | Event-1 lead (days) | First alarm in fault period | Controls silent /5 | Evaluable /6 | Processed /6 |",
             "|---:|---|---:|---|---:|---|---:|---:|---:|"]
    for seed in p["seeds"]:
        text.append(primary_line(SENSITIVITY, seed))
    text += ["", "Pooled counts are repeated machine-seed evaluations, not independent machines:", ""]
    for arm in (ARM_A, ARM_B, SENSITIVITY):
        total = arms[arm]["total"]
        text.append(f"- {names[arm]}: {total['strict_successes']}/18 strict successes; {total['evaluable_unit_seed_evaluations']}/18 evaluable; {total['observed_unit_seed_evaluations']}/18 processed.")
    text += ["", "## Partition and onset audit", "",
             "Arm B inherits Arm A onset and partition boundaries. Its scaler and threshold use kept commissioning rows; persistence advances over kept rows only. No onset or partition is re-estimated for B.", "",
             "| Unit | Arm | Status | Onset (years) | Speed P25 / P75 (cycles/min) | Train kept/base | Calibration kept/base | Healthy test kept/base | Fault kept/base | Reason |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---|"]
    audit_path = results_root / "partition_onset_audit.json"
    for audit in read_json(audit_path) if audit_path.exists() else []:
        window = audit.get("operating_window") or {}
        kept = audit["partition_counts"]
        base = window.get("base_partition_counts", kept)
        counts = [f"{kept[n]}/{base[n]}" for n in PARTITION_NAMES]
        bounds = f"{window['p25']} / {window['p75']}" if window else "not applicable"
        text.append(f"| {audit['unit_label']} | {names[audit['arm']]} | {audit['evaluation_status']} | {audit['onset_years']} | {bounds} | " + " | ".join(counts) + f" | {window.get('reason') or audit.get('reason') or ''} |")
    text += ["", "Every turbine/seed outcome and first-alarm source row is in per_unit.csv. Source-row memberships, Arm A boundary rows, onset diagnostics and exclusion counts are in partition_onset_audit.json. Scores retain source rows, times and fixed partition labels.",
             "", "Lead times include premature healthy-test alarms, which cause strict failure. SND lead is (nominal replacement year - first alarm year) × 365.25. No alarm gives null timing and false detection. HSS lead is 50 minus the first alarm record-day.",
             "", "## Limitations", "",
             "Filename-to-turbine mapping remains ASSUMED, not publisher-verified. The time-only check found no registered contradiction. Nominal replacement times of 1.2 and 2.0 years are approximate.",
             "", "Turbine 2's documented early electrical sensor issue was retained; it is an electrical measurement issue, not a known bearing failure.",
             "", "HSS days are ordered record-days, including both March 17 records. Filename timestamps are in hss_record_days.csv. Day 50 is a terminal proxy, not an exact failure timestamp; commissioning health is assumed, and this is one small case study.",
             "", summary["interpretation"],
             "", "Healthy/fault labels use proxy-partition assumptions. Physical damage onset is unknown. Earlier development selection bias remains. No unseen unit trains or updates the global models; no extra external-data tuning is authorized.",
             "", "Source/model hashes are in protocol.json; environment and time-only evidence are in execution_environment.json and metadata_identity_check.json."]
    (results_root / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def process_snd_unit(unit, arms, p, mods, np, pd, states, results_root, rows, audits, environment):
    data = snd_features(unit, p, mods, np)
    base = None
    event = unit.get("event")
    unit_label = f"Turbine {unit['turbine']}" + (f" {event}" if event else "")
    for arm in arms:
        window = None
        if arm == ARM_B:
            if base is None:
                raise StopRun("Arm A must precede Arm B")
            partition, window = make_speed_window(data, unit, base, p, np)
        else:
            try:
                partition = make_snd_partitions(data, unit, arm, p, mods, np, pd)
            except NotEvaluableStop as exc:
                # Preserve the v1 global stop, while explicitly reporting the failed unit.
                audits.append({"unit_label": unit_label, "turbine": unit["turbine"], "event": event,
                               "arm": arm, "evaluation_status": "not evaluable", "reason": str(exc),
                               "onset": exc.onset, "onset_years": None,
                               "partition_counts": dict.fromkeys(PARTITION_NAMES),
                               "eligible_rows": exc.eligible_rows})
                write_json(results_root / "partition_onset_audit.json", audits)
                for seed in p["seeds"]:
                    rows.append({"dataset": "SND", "arm": arm, "turbine": unit["turbine"],
                                 "event": event, "population": "secondary" if event == "event2" else "primary",
                                 "endpoint": "fault" if unit["turbine"] == 5 else "control",
                                 "seed": seed, "unit": f"SND::{unit_label}::{arm}",
                                 "evaluation_status": "not evaluable", "not_evaluable_reason": str(exc),
                                 "n_rows": exc.eligible_rows, "healthy_alarm": None,
                                 "fault_detected": None, "strict_success": None})
                raise
            if arm == ARM_A:
                base = partition
        pos, ti, ci, hi, fi, onset = partition
        # The onset index belongs to the unfiltered parent arm, never to B's kept rows.
        onset_parent = base if arm == ARM_B else partition
        onset_pos = onset_parent[0][int(onset["onset_index"])] if onset else None
        status = window["status"] if window else "evaluable"
        parent_parts = dict(zip(PARTITION_NAMES, onset_parent[1:5]))
        parent_rows = data["csv_rows"][onset_parent[0]]
        audit = {"unit_label": unit_label, "turbine": unit["turbine"], "event": event,
                 "population": "secondary" if event == "event2" else "primary", "arm": arm,
                 "raw_rows": data["raw_rows"], "excluded_outside_segment_rows": data["excluded_replacement_rows"],
                 "excluded_operating_filter_rows": len(data["times"]) - len(pos),
                 "evaluation_status": status, "operating_window": window,
                 "onset": onset, "onset_years": float(data["times"][onset_pos]) if onset_pos is not None else None,
                 "onset_csv_row_1based": int(data["csv_rows"][onset_pos]) if onset_pos is not None else None,
                 "fixed_parent_partition_boundary_rows": {
                     n: [int(parent_rows[idx[0]]), int(parent_rows[idx[-1]])] if len(idx) else []
                     for n, idx in parent_parts.items()},
                 "partition_counts": dict(zip(PARTITION_NAMES, map(len, (ti, ci, hi, fi)))),
                 "partitions_csv_rows_1based": {n: data["csv_rows"][pos[idx]].tolist()
                                               for n, idx in zip(PARTITION_NAMES, (ti, ci, hi, fi))}}
        audits.append(audit)
        write_json(results_root / "partition_onset_audit.json", audits)
        labels = {"dataset": "SND", "arm": arm, "turbine": unit["turbine"], "event": event,
                  "population": audit["population"], "endpoint": "fault" if unit["turbine"] == 5 else "control",
                  "replacement_years": (p["snd"]["gearbox_replacement_years"] if event == "event2" else p["snd"]["bearing_replacement_years"]) if unit["turbine"] == 5 else None}
        extra = {"estimated_onset_years": audit["onset_years"],
                 "speed_window_p25": window["p25"] if window else None,
                 "speed_window_p75": window["p75"] if window else None,
                 "not_evaluable_reason": window["reason"] if window else None}
        mid = f"SND::Turbine{unit['turbine']}" + (f"::{event}" if event else "") + f"::{arm}"
        if status == "not evaluable":
            for seed in p["seeds"]:
                rows.append({**labels, **extra, "unit": mid, "seed": seed, "evaluation_status": status,
                             "n_rows": len(pos), "training_rows": len(ti), "calibration_rows": len(ci),
                             "healthy_test_rows": len(hi), "fault_test_rows": len(fi),
                             **{k: None for k in ("healthy_alarm", "fault_detected", "strict_success",
                                "first_alarm_index_0based", "first_alarm_source_row_or_day_1based",
                                "first_alarm_coordinate", "first_alarm_minus_replacement_years",
                                "lead_to_replacement_years", "lead_time_days", "first_alarm_in_fault_period",
                                "first_alarm_before_estimated_onset", "reference", "threshold", "threshold_method")}})
        else:
            rows.extend(score_unit(mid, data["features"][pos], ti, ci, hi, fi,
                                   data["times"][pos], data["csv_rows"][pos], states, mods, p, np, results_root,
                                   labels, extra))
        report(results_root, p, rows, environment, False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--metadata-only", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
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
        # Primary: all six turbines; event 1 for T5. A precedes its derived B arm.
        for registered_unit in p["snd"]["units"]:
            unit = dict(registered_unit)
            if unit["turbine"] == 5:
                unit["event"] = "event1"
            process_snd_unit(unit, p["snd"]["arms"], p, mods, np, pd, states,
                             results_root, rows, audits, environment)
        # Event 2 is a separate unseen unit with its own onset and commissioning.
        unit = {**next(u for u in p["snd"]["units"] if u["turbine"] == 5), "event": "event2"}
        process_snd_unit(unit, (ARM_A, ARM_B), p, mods, np, pd, states,
                         results_root, rows, audits, environment)
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
        for arm in (ARM_A, ARM_B):
            rows.extend(score_unit(f"MathWorks_HSS::bearing::{arm}", np.asarray(feature_rows, dtype=np.float32),
                                   ti, ci, empty, empty, np.arange(1, 51), np.arange(1, 51),
                                   states, mods, p, np, results_root,
                                   {"dataset": "MathWorks_HSS", "arm": arm, "turbine": None,
                                    "population": "secondary", "endpoint": "case_study"},
                                   {"test_start_record_day": 21, "test_end_record_day": 50,
                                    "arm_applicability": "not applicable (no speed column)" if arm == ARM_B else "applicable"}))
        if len(rows) != p["amendment"]["expected_output_rows"]["total"]:
            raise StopRun("Output count differs from the frozen 66-row design")
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
