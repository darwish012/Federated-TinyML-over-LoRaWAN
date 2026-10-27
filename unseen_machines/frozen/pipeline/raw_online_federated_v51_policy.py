"""Pure V51 threshold/persistence policy and leakage-safe outer selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import raw_online_federated_v45_cohort as cohort
import raw_online_federated_v45_grid as v45grid
import raw_online_federated_v46_stats as stats

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "raw_online_federated_v51_protocol.json"
REFERENCE = "reference_epc_triad_4of7"


def specification() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def candidate_names() -> tuple[str, ...]:
    return tuple(specification()["candidate_order"])


def candidate_config(name: str) -> dict:
    return dict(specification()["candidates"][name])


def apply_policy(score: np.ndarray, op_start: int, *, threshold: float, k: int, window: int, holdoff: int) -> dict[str, np.ndarray | int]:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    start = int(op_start)
    if start < 0 or start >= len(values):
        raise ValueError("invalid operation start")
    if not (0 < int(k) <= int(window)) or int(holdoff) < 0:
        raise ValueError("invalid persistence or holdoff")
    operation = values[start:]
    raw = operation > float(threshold)
    raw[:min(int(holdoff), len(raw))] = False
    latched = np.asarray(v45grid.k_of_m_latch(raw, int(k), int(window)), dtype=bool)
    full_alarm = np.zeros(len(values), dtype=bool)
    full_alarm[start:] = latched
    full_score = np.zeros(len(values), dtype=np.float64)
    full_score[start:] = operation - float(threshold)
    positions = np.flatnonzero(full_alarm)
    return {"alarm": full_alarm, "score": full_score, "raw": raw, "first_alarm": int(positions[0]) if len(positions) else -1}


def summarize(first_alarm: int, *, n: int, op_start: int, early_fraction: float = 0.10) -> dict:
    first, total, start = int(first_alarm), int(n), int(op_start)
    terminal = int(np.floor(0.9 * total))
    early_stop = start + int(np.floor(float(early_fraction) * max(terminal - start, 1)))
    return {
        "n": total, "op_start": start, "terminal": terminal, "first_alarm": first,
        "ever_alarmed": bool(first >= 0), "preterminal_detected": bool(0 <= first < terminal),
        "early_operation_alarm": bool(0 <= first < early_stop),
        "lead_acquisitions": int(total - 1 - first) if first >= 0 else None,
        "lead_fraction": float((total - 1 - first) / max(total - 1, 1)) if first >= 0 else None,
    }


def _timing(rows: Sequence[dict], onsets: Mapping[str, float | None]) -> dict[str, int]:
    output = {"valid": 0, "premature": 0, "late": 0, "missed": 0, "unavailable": 0}
    for row in rows:
        onset = onsets.get(str(row["machine_id"]))
        if onset is None:
            output["unavailable"] += 1
        else:
            output[stats.onset_event(int(row["first_alarm"]), float(onset), int(row["terminal"]), int(row["n"]), 0.0)] += 1
    return output


def select_candidate(records: Sequence[dict], heldout_dataset: str, development_onsets: Mapping[str, float | None]) -> dict:
    names = candidate_names()
    development = [row for row in records if str(row["dataset"]) != str(heldout_dataset)]
    forbidden = {str(row["machine_id"]) for row in records if str(row["dataset"]) == str(heldout_dataset)}
    if any(machine in forbidden for machine in development_onsets):
        raise ValueError("held-out onset leaked into V51 selection")
    indexed = {name: {str(row["machine_id"]): row for row in development if row["candidate"] == name} for name in names}
    reference = indexed[REFERENCE]
    rtf_ids = sorted(key for key, row in reference.items() if row["role"] == cohort.RUN_TO_FAILURE)
    healthy_ids = sorted(key for key, row in reference.items() if row["role"] == cohort.HEALTHY_CONTROL)
    reference_timing = _timing([reference[key] for key in rtf_ids], development_onsets)
    reference_healthy = sum(bool(reference[key]["ever_alarmed"]) for key in healthy_ids)
    candidates = []
    for order, name in enumerate(names):
        current = indexed[name]
        timing = _timing([current[key] for key in rtf_ids], development_onsets)
        healthy = sum(bool(current[key]["ever_alarmed"]) for key in healthy_ids)
        lost = sum(bool(reference[key]["preterminal_detected"]) and not bool(current[key]["preterminal_detected"]) for key in rtf_ids)
        preterminal = sum(bool(current[key]["preterminal_detected"]) for key in rtf_ids)
        early = sum(bool(current[key]["early_operation_alarm"]) for key in rtf_ids)
        cfg = candidate_config(name)
        qualifies = name == REFERENCE or bool(healthy <= reference_healthy and lost == 0 and timing["valid"] >= reference_timing["valid"] and timing["premature"] <= reference_timing["premature"])
        candidates.append({
            "candidate": name, "order": order, "qualifies": qualifies,
            "heldout_dataset": heldout_dataset,
            "development_datasets": sorted({str(row["dataset"]) for row in development}),
            "development_rtf_machines": len(rtf_ids), "development_healthy_machines": len(healthy_ids),
            "healthy_alarms": healthy, "reference_healthy_alarms": reference_healthy,
            "lost_reference_preterminal": lost, "preterminal_detected": preterminal,
            "early_operation_alarms": early, "strict_timing": timing,
            "reference_strict_timing": reference_timing, "holdoff": int(cfg.get("holdoff", 0)),
        })
    feasible = [row for row in candidates if row["qualifies"]]
    feasible.sort(key=lambda row: (-row["strict_timing"]["valid"], row["strict_timing"]["premature"], row["early_operation_alarms"], -row["preterminal_detected"], row["holdoff"], row["order"]))
    return {"heldout_dataset": heldout_dataset, "development_datasets": sorted({str(row["dataset"]) for row in development}), "selected_candidate": feasible[0]["candidate"], "candidates": candidates}


__all__ = ["REFERENCE", "apply_policy", "candidate_config", "candidate_names", "select_candidate", "specification", "summarize"]
