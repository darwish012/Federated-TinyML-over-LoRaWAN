"""V42 score-source phase: legs, gate, reports, figures.

This is Section 5 of V42_PREREGISTRATION.md. It is invoked as a separate
script (mirroring V32's driver/score split) so that
``raw_online_federated_v42.py`` never has to import this module and vice
versa never becomes circular.

Structural requirement (carried from V25/V31/V38, restated in
V42_PREREGISTRATION.md SS5): the complete report, every table, and every
figure are written BEFORE the gate is evaluated. Gate failure exits 2 with
the report already on disk; only an integrity failure (a changed frozen
dependency, a broken hash chain) exits 3, and only via
raw_online_federated_v42.IntegrityError.

Onset is opened here, and only here, after the source-prediction lock
already exists -- the predictions this script scores were produced blind to
onset and to device status (raw_online_federated_v42.py never imports the
evidence CSV). Everything under "PRIMARY" below is computed WITHOUT onset;
onset is read for the SECONDARY table only, exactly as
V42_PREREGISTRATION.md SS1 requires.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import raw_online_federated_v42 as v42

HERE = v42.HERE
sha256 = v42.sha256
read_json = v42.read_json
write_json = v42.write_json
write_csv = v42.write_csv
IntegrityError = v42.IntegrityError

SECONDARY_CAVEAT = (
    "premature/valid/late/missed against the analytical onset reference is a SECONDARY, "
    "proxy-dependent endpoint (onset is not identifiable from archived data; memory.txt "
    "Appendices 2026-07-27B/C). The premature fraction moves from 23% to 97% purely with the "
    "commissioning rule (Appendix 2026-07-27B); these counts must never be read as physical "
    "ground truth."
)


# --------------------------------------------------------------------------
# Loading and merging (onset opened here, post-lock)
# --------------------------------------------------------------------------
def _load_predictions(root: Path) -> pd.DataFrame:
    v42.require_source_prediction_lock(root)
    frame = pd.read_csv(root / "v42_source_predictions_summary.csv")
    frame["calibration_label"] = frame["calibration_label"].astype(str)
    frame["guard_alarm"] = frame["guard_alarm"].astype(bool)
    return frame


def _load_evidence() -> pd.DataFrame:
    spec = v42.specification()
    path = v42.HERE / spec["source_evidence_csv"]
    evidence = pd.read_csv(path).set_index("client_id")
    return evidence


def classify_event(status: str, first_alarm: int, acquisitions: int, onset_value: float) -> dict:
    """Pure, directly-testable classification for one (device, operating point).

    Exactly mirrors raw_online_federated_v41.score_source's own convention
    -- required for the G5 reproduction check
    (:func:`arm_a_reproduction_check`) to be meaningful at all: the default
    ``event`` is "not_lifecycle", overridden ONLY for ``status == "failed"``
    with an available onset (-> missed/premature/valid/late) or
    ``status == "healthy"`` (-> control_alarm/control_clean). Censored,
    fixed_fault, and failed-without-onset devices are therefore
    "not_lifecycle", exactly matching V41R1's own locked per-device CSV --
    NOT a separate "onset_unavailable" or "control_*" label for those cases.

    ``ever_alarmed`` and ``preterminal_detected`` are onset-free (PRIMARY);
    ``event`` uses onset and is SECONDARY only.
    """
    terminal = int(math.floor(0.9 * int(acquisitions)))
    first_alarm = int(first_alarm)
    ever_alarmed = bool(first_alarm >= 0)
    preterminal_detected = bool(0 <= first_alarm < terminal)
    onset_available = status == "failed" and pd.notna(onset_value)
    event = "not_lifecycle"
    if onset_available:
        onset = int(onset_value)
        event = (
            "missed"
            if first_alarm < 0
            else "premature"
            if first_alarm < onset
            else "valid"
            if first_alarm < terminal
            else "late"
        )
    elif status == "healthy":
        event = "control_alarm" if ever_alarmed else "control_clean"
    return {
        "terminal": terminal,
        "ever_alarmed": ever_alarmed,
        "preterminal_detected": preterminal_detected,
        "onset_available": bool(onset_available),
        "event": event,
    }


def build_events(root: Path) -> pd.DataFrame:
    """Merge locked predictions with evidence (status/EoL/onset), post-lock.

    Per-row classification is delegated to :func:`classify_event`.
    """
    predictions = _load_predictions(root)
    evidence = _load_evidence()
    rows = []
    for record in predictions.to_dict("records"):
        client_id = record["client_id"]
        if client_id not in evidence.index:
            raise IntegrityError(f"no evidence row for {client_id}")
        meta = evidence.loc[client_id]
        status = str(meta["status"])
        acquisitions = int(meta["acquisitions"])
        onset_value = meta.get("first_sustained_onset_acquisition", np.nan)
        classified = classify_event(status, record["first_alarm"], acquisitions, onset_value)
        rows.append({**record, "status": status, "acquisitions_evidence": acquisitions, **classified})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# PRIMARY endpoints (onset-independent) -- P1, P2, P3, and the G4 control rate
# --------------------------------------------------------------------------
def primary_table(events: pd.DataFrame) -> pd.DataFrame:
    """G2 erratum (V42_PREREGISTRATION.md SS5): two guard figures are
    reported, not one.

    ``guard_latch_false_alarm_rate`` is the ORIGINAL, WITHDRAWN definition
    (device-level mean of ``guard_alarm``, the 4-of-7 latch outcome within
    the guard). It is structurally 0.0 for every arm/calibration combination
    -- guard_length=4 < persistence_window=7 means the 4-of-7 condition can
    never even be evaluated within the guard -- and is kept in the table
    ONLY so it can never again be cited as evidence without this label.

    ``guard_exceedance_rate`` is the REDEFINED G2: the pooled per-acquisition
    rate at which the pre-persistence ensemble condition (raw, before any
    4-of-7 requirement) is met within the guard -- ``sum(guard_exceedances)
    / sum(guard_count)`` across every eligible device at this operating
    point. This quantity CAN vary with calibration size and limit rule and
    is what the gate's G2 condition actually uses.
    """
    rows = []
    for (arm, label), group in events.groupby(["arm", "calibration_label"]):
        healthy = group[group["status"] == "healthy"]
        controls = group[group["status"].isin(["healthy", "censored"])]
        failed = group[group["status"] == "failed"]
        guard_latch_rate = float(group["guard_alarm"].mean()) if len(group) else float("nan")
        total_guard_acquisitions = int(group["guard_count"].sum())
        total_guard_exceedances = int(group["guard_exceedances"].sum())
        guard_exceedance_rate = (
            float(total_guard_exceedances) / total_guard_acquisitions if total_guard_acquisitions else float("nan")
        )
        rows.append(
            {
                "arm": arm,
                "calibration_label": label,
                "healthy_control_alarm_rate": float(healthy["ever_alarmed"].mean()) if len(healthy) else float("nan"),
                "healthy_control_devices": int(len(healthy)),
                "guard_latch_false_alarm_rate": guard_latch_rate,
                "guard_latch_false_alarm_rate_is_vacuous": True,
                "guard_exceedance_rate": guard_exceedance_rate,
                "guard_exceedances": total_guard_exceedances,
                "guard_acquisitions": total_guard_acquisitions,
                "guard_eligible_devices": int(len(group)),
                "preterminal_detection_rate": float(failed["preterminal_detected"].mean()) if len(failed) else float("nan"),
                "failed_eligible_devices": int(len(failed)),
                "device_level_false_alarm_rate_controls": float(controls["ever_alarmed"].mean()) if len(controls) else float("nan"),
                "control_eligible_devices": int(len(controls)),
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "calibration_label"]).reset_index(drop=True)


def secondary_table(events: pd.DataFrame) -> pd.DataFrame:
    """Premature/valid/late/missed among failed, onset-available devices.

    Within ``failed = group[group["status"] == "failed"]``, ``event ==
    "not_lifecycle"`` can only mean "failed but onset unavailable" (healthy
    and censored rows are already excluded by the status filter) -- see
    build_events for why "not_lifecycle" is the shared default rather than a
    dedicated "onset_unavailable" label.
    """
    rows = []
    for (arm, label), group in events.groupby(["arm", "calibration_label"]):
        failed = group[group["status"] == "failed"]
        onset_available_count = int(failed["onset_available"].sum())
        counts = failed["event"].value_counts()
        rows.append(
            {
                "arm": arm,
                "calibration_label": label,
                "failed_eligible_devices": int(len(failed)),
                "onset_available_devices": onset_available_count,
                "premature": int(counts.get("premature", 0)),
                "valid": int(counts.get("valid", 0)),
                "late": int(counts.get("late", 0)),
                "missed": int(counts.get("missed", 0)),
                "onset_unavailable": int(len(failed) - onset_available_count),
                "premature_fraction_of_onset_available": (
                    float(counts.get("premature", 0)) / max(onset_available_count, 1)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "calibration_label"]).reset_index(drop=True)


def eligibility_summary(root: Path) -> pd.DataFrame:
    ledger = pd.read_csv(root / "v42_eligibility_ledger.csv")
    rows = []
    for label, group in ledger.groupby("calibration_label"):
        eligible = group[group["eligible"]]
        ineligible = group[~group["eligible"]]
        rows.append(
            {
                "calibration_label": label,
                "eligible_devices": int(len(eligible)),
                "ineligible_devices": int(len(ineligible)),
                "ineligible_client_ids": ";".join(sorted(ineligible["client_id"])),
                "ineligible_datasets": ";".join(sorted(set(ineligible["dataset"]))),
            }
        )
    order = {"8": 0, "16": 1, "32": 2, "64": 3, "128": 4, "self_starting": 5}
    return pd.DataFrame(rows).sort_values("calibration_label", key=lambda s: s.map(order)).reset_index(drop=True)


# --------------------------------------------------------------------------
# G5: arm A must reproduce V41R1's locked per-device outcomes exactly
# --------------------------------------------------------------------------
def arm_a_reproduction_check(events: pd.DataFrame) -> dict:
    locked = pd.read_csv(v42.V41R1_ROOT / "v41_source_per_device.csv").set_index("client_id")
    computed = events[(events["arm"] == "A_V41R1_BASELINE") & (events["calibration_label"] == "8")].set_index(
        "client_id"
    )
    mismatches = []
    if set(locked.index) != set(computed.index):
        mismatches.append({"client_id": "__roster__", "detail": "device set differs from locked V41R1 roster"})
    for client_id in sorted(set(locked.index) & set(computed.index)):
        expected_first = int(locked.loc[client_id, "first_alarm"])
        expected_event = str(locked.loc[client_id, "event"])
        got_first = int(computed.loc[client_id, "first_alarm"])
        got_event = str(computed.loc[client_id, "event"])
        if expected_first != got_first or expected_event != got_event:
            mismatches.append(
                {
                    "client_id": client_id,
                    "expected_first_alarm": expected_first,
                    "got_first_alarm": got_first,
                    "expected_event": expected_event,
                    "got_event": got_event,
                }
            )
    return {
        "checked_devices": len(set(locked.index) & set(computed.index)),
        "locked_devices": len(locked),
        "computed_devices": len(computed),
        "mismatches": mismatches,
        "reproduced_exactly": len(mismatches) == 0 and len(locked) == len(computed) == 40,
    }


# --------------------------------------------------------------------------
# Gate (V42_PREREGISTRATION.md SS5)
# --------------------------------------------------------------------------
def evaluate_gate(primary: pd.DataFrame, arm_a: dict) -> dict:
    """G2 uses the redefined per-acquisition guard exceedance rate (SS5 "G2
    erratum"), not the withdrawn latch-based figure -- which is still
    surfaced in ``values`` for visibility but never gates on. G4 is the
    load-bearing specificity condition: it is evaluated over full control
    records (not just a 4-acquisition guard) and can genuinely vary; a
    reader must not weight G2 and G4 equally.
    """
    spec = v42.specification()["gate"]
    op = spec["primary_operating_point"]
    row = primary[(primary["arm"] == op["arm"]) & (primary["calibration_label"] == str(op["calibration_points"]))]
    if len(row) != 1:
        raise IntegrityError("primary operating point row missing from primary table")
    row = row.iloc[0]
    values = {
        "G1_healthy_control_alarm_rate": float(row["healthy_control_alarm_rate"]),
        "G2_guard_exceedance_rate": float(row["guard_exceedance_rate"]),
        "G2_guard_latch_false_alarm_rate_WITHDRAWN_VACUOUS": float(row["guard_latch_false_alarm_rate"]),
        "G3_preterminal_detection_rate": float(row["preterminal_detection_rate"]),
        "G4_device_level_false_alarm_rate_controls": float(row["device_level_false_alarm_rate_controls"]),
    }
    conditions = {
        "G1": values["G1_healthy_control_alarm_rate"] <= float(spec["G1_healthy_control_alarm_rate_max"]),
        "G2": values["G2_guard_exceedance_rate"] <= float(spec["G2_guard_per_acquisition_exceedance_rate_max"]),
        "G3": values["G3_preterminal_detection_rate"] >= float(spec["G3_preterminal_detection_rate_min"]),
        "G4": values["G4_device_level_false_alarm_rate_controls"] <= float(spec["G4_device_level_false_alarm_probability_max_on_controls"]),
        "G5": bool(arm_a["reproduced_exactly"]),
    }
    return {
        "primary_operating_point": op,
        "values": values,
        "load_bearing_specificity_gate": "G4",
        "denominators": {
            "healthy_control_devices": int(row["healthy_control_devices"]),
            "guard_eligible_devices": int(row["guard_eligible_devices"]),
            "guard_acquisitions": int(row["guard_acquisitions"]),
            "failed_eligible_devices": int(row["failed_eligible_devices"]),
            "control_eligible_devices": int(row["control_eligible_devices"]),
        },
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _figures(root: Path, primary: pd.DataFrame, events: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    order = {"8": 8, "16": 16, "32": 32, "64": 64, "128": 128, "self_starting": 200}
    plotted = primary.copy()
    plotted["x"] = plotted["calibration_label"].map(order)
    plotted = plotted.sort_values("x")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arm, group in plotted.groupby("arm"):
        axes[0].plot(group["x"], 100 * group["healthy_control_alarm_rate"], marker="o", label=arm)
        axes[1].plot(group["x"], 100 * group["preterminal_detection_rate"], marker="o", label=arm)
    axes[0].set_xlabel("calibration points (self_starting plotted at 200)")
    axes[0].set_ylabel("healthy-control alarm rate (%)")
    axes[0].set_title("P1: healthy-control alarms")
    axes[1].set_xlabel("calibration points (self_starting plotted at 200)")
    axes[1].set_ylabel("preterminal detection rate (%)")
    axes[1].set_title("P3: preterminal detection")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("V42 trade curve: specificity (P1) vs sensitivity (P3), per arm and calibration size")
    fig.tight_layout()
    path = figures_dir / "01_trade_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig2, ax2 = plt.subplots(1, 1, figsize=(11, 4.5))
    for arm, group in plotted.groupby("arm"):
        ax2.plot(group["x"], 100 * group["device_level_false_alarm_rate_controls"], marker="s", label=arm)
    ax2.axhline(5.0, color="red", linestyle="--", linewidth=1, label="5% EPC target")
    ax2.set_xlabel("calibration points (self_starting plotted at 200)")
    ax2.set_ylabel("device-level false-alarm rate on controls (%)")
    ax2.set_title("G4: device-level false-alarm rate on healthy + censored controls")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    path2 = figures_dir / "02_device_level_false_alarm.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    paths.append(path2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(11, 5))
    op_events = events[events["arm"].isin(["A_V41R1_BASELINE", "B_CALIB_SWEEP", "C_EPC_ADJUSTED"])].copy()
    op_events["x"] = op_events["calibration_label"].map(order)
    alarmed = op_events[op_events["first_alarm"] >= 0]
    for arm, group in alarmed.groupby("arm"):
        ax3.scatter(group["x"], group["first_alarm"], s=10, alpha=0.5, label=arm)
    ax3.axhline(57, color="red", linestyle="--", linewidth=1, label="V41R1 acquisition-57 cluster")
    ax3.set_xlabel("calibration points")
    ax3.set_ylabel("first-alarm acquisition index")
    ax3.set_title("Per-device first alarm vs calibration size")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    path3 = figures_dir / "03_first_alarm_vs_calibration.png"
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    paths.append(path3)

    return paths


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _pct(value: float) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{100.0 * value:.2f}%"


def _write_report(root: Path, payload: dict) -> None:
    lines = [
        "# V42 source-only results: calibration sample size and EPC-adjusted limits",
        "",
        "V42 reuses V41R1's frozen streams, trained models, scores, and V29b triad unchanged. "
        "No retraining occurred. All four arms are source-only; no target data was ever opened.",
        "",
        f"Arm A reproduces V41R1's locked per-device outcomes exactly: **{payload['arm_a']['reproduced_exactly']}** "
        f"({payload['arm_a']['checked_devices']}/{payload['arm_a']['locked_devices']} devices checked).",
        "",
        "## Section 0: null design simulation (freshly re-run, not copied from the preregistration)",
        "",
        payload["design_markdown"],
        "",
        "## Section 4: eligibility ledger",
        "",
        "Every table below states its own eligible denominator; larger calibration windows "
        "exclude short devices (XJTU is expected ineligible at most sizes -- a documented "
        "deployment limitation, never counted as success).",
        "",
        "| Calibration | Eligible | Ineligible | Ineligible datasets |",
        "|---|---:|---:|---|",
    ]
    for row in payload["eligibility"]:
        lines.append(
            f"| {row['calibration_label']} | {row['eligible_devices']} | {row['ineligible_devices']} | {row['ineligible_datasets'] or '-'} |"
        )
    guard_check = payload["guard_structural_check"]["fixed_window_guard"]
    lines += [
        "",
        "## G2 erratum: the latch-based guard rate is structurally vacuous",
        "",
        f"guard_length={guard_check['guard_length']}, persistence_window={guard_check['persistence_window']} "
        f"=> complete 7-windows available in the guard = {guard_check['complete_windows']}. Because that count is "
        "<= 0, the 4-of-7 persistence condition can never be evaluated within the guard at all, for any arm, "
        "any calibration size, any threshold -- a reported latch-based guard false-alarm rate of 0% is "
        "**arithmetically forced, not measured**. This is the same defect in V41R1's own \"guard was already "
        "clean\" finding (recorded in memory.txt as a PASSED source-gate condition) and in V19's XJTU split "
        "(guard 0 acquisitions, rule 10-of-12) -- both vacuous by the same arithmetic, never validated safety.",
        "",
        "**G2 is therefore redefined** as the per-acquisition exceedance rate within the guard (the fraction of "
        "guard acquisitions where the pre-persistence ensemble condition is met, pooled across eligible devices), "
        "which CAN vary with calibration size and limit rule. The withdrawn latch-based figure is still reported "
        "below, in its own column, explicitly labelled vacuous, so it is never again cited as evidence.",
        "",
        "**G4 (device-level false-alarm rate on full healthy+censored control records) is the load-bearing "
        "specificity gate**, not G2 -- G4 is evaluated over each control device's entire operational record and "
        "can genuinely vary; G2's near-zero values below are expected and must not be read as comparable evidence "
        "of specificity.",
        "",
        "## PRIMARY endpoints (onset-independent: P1 healthy-control alarm rate, "
        "P2 guard exceedance rate [see erratum above], P3 preterminal detection using verified end-of-life, "
        "G4 device-level false-alarm rate on full control records)",
        "",
        "| Arm | Calibration | P1 healthy-control alarms | (n) | P2 guard exceedance rate | (guard acq) | "
        "P2 guard LATCH rate (VACUOUS) | P3 preterminal detection | (n failed) | G4 control FA rate | (n controls) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["primary"]:
        lines.append(
            f"| {row['arm']} | {row['calibration_label']} | {_pct(row['healthy_control_alarm_rate'])} | {row['healthy_control_devices']} | "
            f"{_pct(row['guard_exceedance_rate'])} | {row['guard_acquisitions']} | "
            f"{_pct(row['guard_latch_false_alarm_rate'])} (vacuous) | "
            f"{_pct(row['preterminal_detection_rate'])} | {row['failed_eligible_devices']} | "
            f"{_pct(row['device_level_false_alarm_rate_controls'])} | {row['control_eligible_devices']} |"
        )
    lines += [
        "",
        "## SECONDARY endpoint: premature/valid/late/missed against the analytical onset proxy",
        "",
        SECONDARY_CAVEAT,
        "",
        "| Arm | Calibration | Failed (eligible) | Premature | Valid | Late | Missed | Onset unavailable |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["secondary"]:
        lines.append(
            f"| {row['arm']} | {row['calibration_label']} | {row['failed_eligible_devices']} | {row['premature']} | "
            f"{row['valid']} | {row['late']} | {row['missed']} | {row['onset_unavailable']} |"
        )
    lines += [
        "",
        "## Predeclared gate",
        "",
        f"Primary operating point: arm **{payload['gate']['primary_operating_point']['arm']}**, "
        f"calibration **{payload['gate']['primary_operating_point']['calibration_points']}**, "
        f"p = {payload['gate']['primary_operating_point']['p']}.",
        "",
        "| Condition | Value | Requirement | Passed |",
        "|---|---:|---|:---:|",
        f"| G1 healthy-control alarm rate | {_pct(payload['gate']['values']['G1_healthy_control_alarm_rate'])} | = 0% | {payload['gate']['conditions']['G1']} |",
        f"| G2 guard exceedance rate (per-acquisition; redefined) | {_pct(payload['gate']['values']['G2_guard_exceedance_rate'])} | <= 5% | {payload['gate']['conditions']['G2']} |",
        f"| G2 latch-based guard rate (withdrawn, vacuous, informational only) | {_pct(payload['gate']['values']['G2_guard_latch_false_alarm_rate_WITHDRAWN_VACUOUS'])} | not gated | n/a |",
        f"| G3 preterminal detection | {_pct(payload['gate']['values']['G3_preterminal_detection_rate'])} | >= 90% | {payload['gate']['conditions']['G3']} |",
        f"| **G4 device-level FA rate on controls (LOAD-BEARING specificity gate)** | {_pct(payload['gate']['values']['G4_device_level_false_alarm_rate_controls'])} | <= 5% | {payload['gate']['conditions']['G4']} |",
        f"| G5 arm A reproduces V41R1 exactly | {payload['arm_a']['reproduced_exactly']} | True | {payload['gate']['conditions']['G5']} |",
        "",
        f"**GATE {'PASSED' if payload['gate']['passed'] else 'FAILED'}.** "
        "G3 is the constraint that makes this honest: if specificity is bought by losing "
        "preterminal detection, the gate fails regardless of G1/G2/G4. The trade curve above, "
        "not this single operating point, is the headline deliverable. **G4, not G2, is the "
        "load-bearing specificity condition** -- G2 is expected to sit near zero for structural "
        "reasons unrelated to calibration quality (see the erratum above) and must not be read "
        "as evidence with the same weight as G4.",
        "",
        "## Audit-phase structural check",
        "",
        f"`guard_length >= persistence_window` was asserted during the audit phase and recorded in "
        f"`v42_audit_lock.json`: fixed-window guard complete_windows = "
        f"{guard_check['complete_windows']}, self-starting guard complete_windows = "
        f"{payload['guard_structural_check']['self_starting_guard']['complete_windows']}. "
        f"`latch_based_guard_metrics_vacuous` = **{payload['guard_structural_check']['latch_based_guard_metrics_vacuous']}** "
        "for this run. The check fails loudly (a printed warning, not a silent zero) whenever it holds.",
    ]
    (root / "V42_SOURCE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Phase entry point
# --------------------------------------------------------------------------
def score_source(root: Path) -> None:
    v42.require_source_prediction_lock(root)
    audit_lock = read_json(root / "v42_audit_lock.json")
    guard_structural_check = audit_lock["guard_structural_check"]
    events = build_events(root)  # onset opened here, post-lock
    primary = primary_table(events)
    secondary = secondary_table(events)
    eligibility = eligibility_summary(root)
    arm_a = arm_a_reproduction_check(events)
    gate = evaluate_gate(primary, arm_a)

    design_table = read_json(root / "v42_design_table.json")
    design_rows = design_table["rows"]
    m_values = sorted({r["calibration_points"] for r in design_rows})
    z_values = sorted({r["threshold_z"] for r in design_rows})
    by_key = {(r["calibration_points"], r["threshold_z"]): r["false_alarm_percentage"] for r in design_rows}
    design_lines = ["| calibration points | " + " | ".join(f"z>={z:g}" for z in z_values) + " |"]
    design_lines.append("|---:|" + "---:|" * len(z_values))
    for m in m_values:
        design_lines.append("| " + str(m) + " | " + " | ".join(f"{by_key[(m, z)]:.2f}%" for z in z_values) + " |")

    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(tables_dir / "v42_events.csv", index=False)
    primary.to_csv(tables_dir / "v42_primary_endpoints.csv", index=False)
    secondary.to_csv(tables_dir / "v42_secondary_events.csv", index=False)
    eligibility.to_csv(tables_dir / "v42_eligibility_summary.csv", index=False)
    write_json(tables_dir / "v42_arm_a_reproduction.json", arm_a)

    figure_paths = _figures(root, primary, events)

    payload = {
        "experiment": "V42",
        "arm_a": arm_a,
        "primary": primary.to_dict("records"),
        "secondary": secondary.to_dict("records"),
        "eligibility": eligibility.to_dict("records"),
        "gate": gate,
        "design_markdown": "\n".join(design_lines),
        "secondary_caveat": SECONDARY_CAVEAT,
        "guard_structural_check": guard_structural_check,
        "figures": [str(path.relative_to(root)) for path in figure_paths],
    }
    write_json(root / "v42_final_metrics.json", payload)
    _write_report(root, payload)

    write_json(
        root / "v42_gate.json",
        {
            "prediction_lock_sha256": sha256(root / "v42_source_prediction_lock.json"),
            "final_metrics_sha256": sha256(root / "v42_final_metrics.json"),
            "report_sha256": sha256(root / "V42_SOURCE_RESULTS.md"),
            **gate,
        },
    )
    print(f"V42 SCORE-SOURCE report: {root / 'V42_SOURCE_RESULTS.md'}", flush=True)
    if not gate["passed"]:
        print("V42 GATE FAILED: report and figures were written before this exit.", flush=True)
        sys.exit(2)
    print("V42 GATE PASSED.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True, choices=("score-source",))
    args = parser.parse_args()
    root = v42.root_for(args.run_id)
    try:
        score_source(root)
    except IntegrityError as error:
        print(f"V42 INTEGRITY FAILURE (score-source): {error}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
