"""V45 Section 2/9: the 96-stream / 80-machine-life-experiment cohort roster.

Pure structural bookkeeping -- dataset/unit/lane/role/path, and the
``experiment_id`` grouping function every later V45 module must use. Touches
no score, no model, no label, and no onset value; safe to import from every
V45 phase, including phases that must run before the nested-fold selection
locks (V45_PREREGISTRATION.md Section 1's onset ban).

Reused, never reimplemented:

    raw_online_federated_v43_cohort.experiment_id/dataset_of/
        group_by_experiment/validate_no_partial_experiment  the IMS
        simultaneous-recording grouping rule and the pseudo-replication check
    raw_online_federated_v30_data.target_devices          the 35-device V30
        target roster (dlr/kaist/unsw/ottawa)
    raw_online_federated_v41r1_data.target_devices        the 4-device DLR
        Tests 2/5 roster V44 already opened and sealed
    raw_online_federated_v41_data.load_feature_stream      the
        features/acquisition_index npz schema shared by the historical
        streams and the V44 DLR Tests 2/5 streams
    raw_online_federated_v30_data.load_feature_stream      the
        features/context/tokens npz schema used by the V30 target streams

The historical 57-stream roster (dataset/unit/lane) is read from
``raw_online_federated_v12_protocol.json``'s ``train_clients``/``test_clients``
keys -- the same frozen file ``raw_online_federated_v41.source_roster`` already
reads for the 40-device ``train_clients`` half; V45 additionally reads the
17-device ``test_clients`` half (already present in that file, never written
by V45, and containing no onset/label/status value -- only dataset/unit
identity) so its cohort covers the full 57 historical streams the
preregistration's Section 2 table requires, not only the 40-device source
roster V41R1/V42/V43/V44 restricted themselves to.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import raw_online_federated_v30_data as v30data
import raw_online_federated_v41_data as v41data
import raw_online_federated_v41r1_data as r1data
import raw_online_federated_v43_cohort as base_cohort

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
V12_PROTOCOL = HERE / "raw_online_federated_v12_protocol.json"
V19_ROOT = RESULTS / "raw_online_federated_v19" / "feature_streams"
V30_ROOT = RESULTS / "raw_online_federated_v30_prereg1" / "target_streams"
V44_FROZEN4_ROOT = RESULTS / "raw_online_federated_v44_frozen4" / "target_streams"

# --------------------------------------------------------------------------
# Roles (V45_PREREGISTRATION.md Section 2 table, rightmost column)
# --------------------------------------------------------------------------
RUN_TO_FAILURE = "run_to_failure"
HEALTHY_CONTROL = "s1_verified_healthy"
FIXED_FAULT_DIAGNOSTIC = "hit_fixed_fault_diagnostic"
STAGED_STATE_DIAGNOSTIC = "ottawa_staged_state_diagnostic"

# Datasets that were part of the original 40-device V41R1/V29b source-training
# roster, hence have a leave-one-dataset-out fold model
# (models/fold_without_{dataset}.npz) available and REQUIRED to avoid a
# device informing its own model. Every other dataset in this cohort (never
# part of that 40-device training roster) uses the all_source bundle -- exactly
# V44's own convention for DLR Tests 2/5, generalised here to every dataset V44
# never needed to know about (ldm_upb, the V30 dlr/kaist/unsw/ottawa streams).
FOLD_TRAINED_DATASETS: frozenset[str] = frozenset({"ferrara", "xjtu", "pronostia", "ims", "hit"})

# Bookkeeping constants the module self-verifies against
# (V45_PREREGISTRATION.md Section 2, memory.txt Appendix 2026-08-05B).
TOTAL_STREAMS = 96
TOTAL_EXPERIMENTS = 80
RUN_TO_FAILURE_STREAMS = 71
RUN_TO_FAILURE_EXPERIMENTS = 55
HEALTHY_STREAMS = 2
HEALTHY_EXPERIMENTS = 2
FIXED_FAULT_STREAMS = 3
OTTAWA_STREAMS = 20

_DLR_UNIT_RE = re.compile(r"^Test_(\d+)_B\d+$")


def _read_v12_protocol() -> dict:
    return json.loads(V12_PROTOCOL.read_text(encoding="utf-8"))


def historical_roster() -> list[tuple[str, str, str]]:
    """All 57 historical (dataset, unit, lane) triples.

    ``lane`` is ``"source_train"`` or ``"heldout_test"`` -- the subdirectory
    under ``V19_ROOT`` the stream's npz file lives in. Reads
    ``raw_online_federated_v12_protocol.json``'s ``train_clients`` (the same
    40-device roster ``raw_online_federated_v41.source_roster`` reads) AND
    ``test_clients`` (the 17-device heldout roster nothing before V45 reads as
    a roster) -- both are dataset/unit identity only, no onset/label/status.
    """
    protocol = _read_v12_protocol()
    rows: list[tuple[str, str, str]] = []
    for lane, key in (("source_train", "train_clients"), ("heldout_test", "test_clients")):
        for dataset, units in protocol[key].items():
            for unit in units:
                rows.append((dataset, unit, lane))
    if len(rows) != 57 or len({f"{d}::{u}" for d, u, _ in rows}) != 57:
        raise RuntimeError("V45 historical roster must be exactly 57 unique streams")
    return rows


def _role_for_historical(dataset: str, unit: str) -> str:
    if dataset != "hit":
        return RUN_TO_FAILURE
    return HEALTHY_CONTROL if unit in ("data1", "data2") else FIXED_FAULT_DIAGNOSTIC


def stream_registry() -> list[dict]:
    """The full 96-stream roster (V45_PREREGISTRATION.md Section 2 table).

    Each row: ``client_id``, ``dataset``, ``unit``, ``lane``
    (``source_train``/``heldout_test``/``v30_target``/``v44_target``),
    ``role``, ``path`` (absolute), ``experiment_id``, ``bundle_dataset_key``
    (which fold-excluded or all_source model bundle scores this device, never
    the device's own dataset when that dataset was part of the 40-device
    source-training roster).
    """
    rows: list[dict] = []

    for dataset, unit, lane in historical_roster():
        client_id = f"{dataset}::{unit}"
        rows.append(
            {
                "client_id": client_id,
                "dataset": dataset,
                "unit": unit,
                "lane": lane,
                "role": _role_for_historical(dataset, unit),
                "path": V19_ROOT / lane / f"{dataset}__{unit}.npz",
                "experiment_id": experiment_id(client_id),
                "bundle_dataset_key": dataset if dataset in FOLD_TRAINED_DATASETS else "all_source",
            }
        )

    for device in v30data.target_devices():
        client_id = device.client_id
        role = STAGED_STATE_DIAGNOSTIC if device.dataset == "ottawa" else RUN_TO_FAILURE
        rows.append(
            {
                "client_id": client_id,
                "dataset": device.dataset,
                "unit": device.unit,
                "lane": "v30_target",
                "role": role,
                "path": V30_ROOT / device.dataset / f"{device.unit}.npz",
                "experiment_id": experiment_id(client_id),
                "bundle_dataset_key": device.dataset if device.dataset in FOLD_TRAINED_DATASETS else "all_source",
            }
        )

    for device in r1data.target_devices():
        client_id = device.client_id
        rows.append(
            {
                "client_id": client_id,
                "dataset": device.dataset,
                "unit": device.unit,
                "lane": "v44_target",
                "role": RUN_TO_FAILURE,
                "path": V44_FROZEN4_ROOT / f"{device.dataset}__{device.unit}.npz",
                "experiment_id": experiment_id(client_id),
                "bundle_dataset_key": device.dataset if device.dataset in FOLD_TRAINED_DATASETS else "all_source",
            }
        )

    _validate_registry(rows)
    return rows


def _validate_registry(rows: list[dict]) -> None:
    client_ids = [row["client_id"] for row in rows]
    if len(client_ids) != TOTAL_STREAMS or len(set(client_ids)) != TOTAL_STREAMS:
        raise RuntimeError(f"V45 cohort must contain exactly {TOTAL_STREAMS} unique streams, got {len(client_ids)}")
    experiments = {row["experiment_id"] for row in rows}
    if len(experiments) != TOTAL_EXPERIMENTS:
        raise RuntimeError(f"V45 cohort must contain exactly {TOTAL_EXPERIMENTS} machine-life experiments, got {len(experiments)}")

    def _count(role: str) -> tuple[int, int]:
        members = [row for row in rows if row["role"] == role]
        return len(members), len({row["experiment_id"] for row in members})

    rtf_streams, rtf_exp = _count(RUN_TO_FAILURE)
    if rtf_streams != RUN_TO_FAILURE_STREAMS or rtf_exp != RUN_TO_FAILURE_EXPERIMENTS:
        raise RuntimeError(f"V45 run-to-failure role must be {RUN_TO_FAILURE_STREAMS}/{RUN_TO_FAILURE_EXPERIMENTS}, got {rtf_streams}/{rtf_exp}")
    healthy_streams, healthy_exp = _count(HEALTHY_CONTROL)
    if healthy_streams != HEALTHY_STREAMS or healthy_exp != HEALTHY_EXPERIMENTS:
        raise RuntimeError(f"V45 S1 healthy role must be {HEALTHY_STREAMS}/{HEALTHY_EXPERIMENTS}, got {healthy_streams}/{healthy_exp}")
    fixed_streams, _ = _count(FIXED_FAULT_DIAGNOSTIC)
    if fixed_streams != FIXED_FAULT_STREAMS:
        raise RuntimeError(f"V45 HIT fixed-fault role must be {FIXED_FAULT_STREAMS} streams, got {fixed_streams}")
    ottawa_streams, ottawa_exp = _count(STAGED_STATE_DIAGNOSTIC)
    if ottawa_streams != OTTAWA_STREAMS or ottawa_exp != OTTAWA_STREAMS:
        raise RuntimeError(f"V45 Ottawa role must be {OTTAWA_STREAMS}/{OTTAWA_STREAMS}, got {ottawa_streams}/{ottawa_exp}")


# --------------------------------------------------------------------------
# experiment_id -- generalizes raw_online_federated_v43_cohort.experiment_id
# with a DLR special case (SPEC-AMBIGUITY: not a V43 dataset, so V43's own
# function does not know about it), following the exact precedent already
# established by raw_online_federated_v44_prospective.experiment_id (same
# "group dlr::Test_{n}_B{bearing} by dlr::Test_{n}" rule, for the same reason
# -- DLR Tests each record two simultaneous bearings on one rig, exactly the
# IMS pattern V43 already handles). raw_online_federated_v43_cohort.py itself
# is never edited (out of scope).
# --------------------------------------------------------------------------
def experiment_id(client_id: str) -> str:
    dataset, sep, unit = client_id.partition("::")
    if not sep:
        raise ValueError(f"malformed client_id (expected 'dataset::unit'): {client_id!r}")
    if dataset != "dlr":
        return base_cohort.experiment_id(client_id)
    match = _DLR_UNIT_RE.match(unit)
    if not match:
        raise ValueError(f"malformed DLR unit (expected 'Test_{{n}}_B{{bearing}}'): {client_id!r}")
    return f"dlr::Test_{match.group(1)}"


def dataset_of(client_id: str) -> str:
    return base_cohort.dataset_of(client_id)


def group_by_experiment(client_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for client_id in client_ids:
        groups.setdefault(experiment_id(client_id), []).append(client_id)
    return groups


def validate_no_partial_experiment(selected: set[str] | list[str], universe: list[str]) -> bool:
    selected_set = set(selected)
    for members in group_by_experiment(list(universe)).values():
        present = [client_id for client_id in members if client_id in selected_set]
        if present and len(present) != len(members):
            return False
    return True


# --------------------------------------------------------------------------
# Stream loading -- dispatches on lane, reusing the two existing npz schemas
# outright. Never opens onset or status evidence.
# --------------------------------------------------------------------------
def load_stream(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """(values, indices) for one cohort row, dispatched on ``row['lane']``."""
    path: Path = row["path"]
    if row["lane"] in ("source_train", "heldout_test", "v44_target"):
        return v41data.load_feature_stream(path)
    if row["lane"] == "v30_target":
        features, _context, _tokens = v30data.load_feature_stream(path)
        indices = np.arange(len(features), dtype=np.int32)
        return features, indices
    raise ValueError(f"unknown V45 stream lane: {row['lane']!r}")


# --------------------------------------------------------------------------
# Section 9 deliverable: cohort table with both stream and machine-life-
# experiment denominators (G6), by dataset and by role.
# --------------------------------------------------------------------------
def cohort_table(rows: list[dict] | None = None) -> list[dict]:
    rows = rows if rows is not None else stream_registry()
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    table = []
    for dataset in sorted(by_dataset):
        members = by_dataset[dataset]
        experiments = {member["experiment_id"] for member in members}
        roles = sorted({member["role"] for member in members})
        table.append(
            {
                "dataset": dataset,
                "streams": len(members),
                "machine_life_experiments": len(experiments),
                "roles": roles,
                "cohort_size_statement": f"{len(members)} bearing-level or recording-unit streams corresponding to {len(experiments)} machine-life experiments",
            }
        )
    return table


def role_table(rows: list[dict] | None = None) -> list[dict]:
    rows = rows if rows is not None else stream_registry()
    by_role: dict[str, list[dict]] = {}
    for row in rows:
        by_role.setdefault(row["role"], []).append(row)
    table = []
    for role in sorted(by_role):
        members = by_role[role]
        experiments = {member["experiment_id"] for member in members}
        table.append(
            {
                "role": role,
                "streams": len(members),
                "machine_life_experiments": len(experiments),
                "cohort_size_statement": f"{len(members)} bearing-level or recording-unit streams corresponding to {len(experiments)} machine-life experiments",
            }
        )
    return table


__all__ = [
    "FIXED_FAULT_DIAGNOSTIC",
    "FOLD_TRAINED_DATASETS",
    "HEALTHY_CONTROL",
    "RUN_TO_FAILURE",
    "STAGED_STATE_DIAGNOSTIC",
    "TOTAL_EXPERIMENTS",
    "TOTAL_STREAMS",
    "V19_ROOT",
    "V30_ROOT",
    "V44_FROZEN4_ROOT",
    "cohort_table",
    "dataset_of",
    "experiment_id",
    "group_by_experiment",
    "historical_roster",
    "load_stream",
    "role_table",
    "stream_registry",
    "validate_no_partial_experiment",
]
