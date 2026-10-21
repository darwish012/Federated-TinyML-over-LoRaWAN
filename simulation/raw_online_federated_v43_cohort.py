"""V43 terminology, stream -> machine-life-experiment grouping, and the cohort table.

Implements V43_PREREGISTRATION.md Section 1: the machine-life experiment (not
the bearing stream) is the primary independent statistical unit (SS1.2), and
the mandatory cohort table (SS1.3). This module is pure structural bookkeeping
-- dataset names, stream/experiment counts, and the ``client_id ->
experiment_id`` grouping function -- and touches no scores, no model, no
label, and no onset value. It is safe to import from every V43 phase,
including phases that must run before the prediction lock.

Terminology (SS1.1), used consistently by every other V43 module:

    BEARING / UNIT STREAM     one recorded signal channel or bearing history
    FEDERATED CLIENT          one logical or physical endpoint in training
    MACHINE-LIFE EXPERIMENT   one independent run-to-failure or test-assembly
                               experiment
    PHYSICAL RIG              laboratory hardware, reused across experiments

Pseudo-replication rule (SS1.2): bearings recorded simultaneously in the same
IMS test stay grouped in every partition, bootstrap resample and prospective
split. The 12 IMS streams are 3 independent units, not 12. This module
supplies the single ``experiment_id`` function every later V43 module
(``raw_online_federated_v43_stats.py`` in particular) must use so that rule is
enforced in exactly one place.
"""
from __future__ import annotations

from typing import Mapping

# --------------------------------------------------------------------------
# SS1.2: verified historical counts (memory.txt Appendix 2026-08-05,
# arithmetic checked 2026-08-05). These are NOT read from any file -- they are
# the frozen, hand-verified ground truth this module's grouping function must
# reproduce exactly (see raw_online_federated_v43_cohort_tests.py).
# --------------------------------------------------------------------------
HISTORICAL_STREAMS: dict[str, int] = {
    "ferrara": 6,
    "xjtu": 15,
    "pronostia": 17,
    "ims": 12,
    "hit": 5,
    "ldm_upb": 2,
}
HISTORICAL_EXPERIMENTS: dict[str, int] = {
    "ferrara": 6,
    "xjtu": 15,
    "pronostia": 17,
    "ims": 3,
    "hit": 5,
    "ldm_upb": 2,
}
HISTORICAL_TOTAL_STREAMS = sum(HISTORICAL_STREAMS.values())
HISTORICAL_TOTAL_EXPERIMENTS = sum(HISTORICAL_EXPERIMENTS.values())
assert HISTORICAL_TOTAL_STREAMS == 57
assert HISTORICAL_TOTAL_EXPERIMENTS == 48

# Bearings/channels per machine-life experiment, when more than one shares a
# rig. Absent here (== 1) for every dataset except IMS.
BEARINGS_PER_EXPERIMENT: dict[str, int] = {"ims": 4}

# SS1.3 cohort-table columns that do not depend on any stream length or
# status value -- static, dataset-level facts.
SIMULTANEOUS_RECORDING: dict[str, bool] = {
    "ferrara": False,
    "xjtu": False,
    "pronostia": False,
    "ims": True,
    "hit": False,
    "ldm_upb": False,
}
PHYSICAL_RIG_REUSED: dict[str, bool] = {
    # IMS: the same test stand recorded three separate runs (Sets 1-3), each
    # a distinct machine-life experiment, so the rig is reused across
    # experiments within the dataset.
    "ferrara": False,
    "xjtu": False,
    "pronostia": False,
    "ims": True,
    "hit": False,
    "ldm_upb": False,
}
ONSET_TERMINAL_HEALTHY_DOCUMENTED: dict[str, str] = {
    # Free-text summary, not a boolean -- datasets mix run-to-failure
    # (terminal condition documented by construction: the recording stops at
    # failure) with fixed-fault-state transfer (HIT) and censored histories
    # (some IMS bearings). See raw_online_federated_v43_cohort_tests.py for
    # the source evidence this text is checked against.
    "ferrara": "run-to-failure; terminal documented; no documented-healthy device",
    "xjtu": "run-to-failure; terminal documented; no documented-healthy device",
    "pronostia": "run-to-failure; terminal documented; no documented-healthy device",
    "ims": "run-to-failure; terminal documented for 3/4 bearings per set, 1/4 censored per set; no documented-healthy device",
    "hit": "fixed-fault-state transfer, not a lifecycle; 2/5 documented healthy, 3/5 documented fixed-fault",
    "ldm_upb": "run-to-failure; terminal documented; no documented-healthy device",
}

DATASETS: tuple[str, ...] = ("ferrara", "xjtu", "pronostia", "ims", "hit", "ldm_upb")


# --------------------------------------------------------------------------
# The grouping function every other V43 module must import and use.
# --------------------------------------------------------------------------
def experiment_id(client_id: str) -> str:
    """Map one bearing/unit-stream client_id to its machine-life-experiment id.

    Every dataset except IMS records one bearing per independent experiment,
    so the experiment id is the client id itself. IMS records four bearings
    per simultaneous run on one shaft/housing under one radial load
    (``ims::R{set}_B{bearing}``); the experiment id groups by the run/set
    prefix only (``ims::R{set}``), so all bearings from one IMS set are
    always the same experiment.
    """
    if "::" not in client_id:
        raise ValueError(f"malformed client_id (expected 'dataset::unit'): {client_id!r}")
    dataset, unit = client_id.split("::", 1)
    if dataset == "ims":
        if "_B" not in unit:
            raise ValueError(f"malformed IMS unit (expected 'R{{n}}_B{{m}}'): {client_id!r}")
        run_id = unit.split("_B", 1)[0]
        return f"ims::{run_id}"
    return client_id


def group_by_experiment(client_ids: list[str]) -> dict[str, list[str]]:
    """Group a list of client_ids by their experiment_id, preserving order."""
    groups: dict[str, list[str]] = {}
    for client_id in client_ids:
        groups.setdefault(experiment_id(client_id), []).append(client_id)
    return groups


def dataset_of(client_id: str) -> str:
    return client_id.split("::", 1)[0]


# --------------------------------------------------------------------------
# SS1.3 mandatory cohort table
# --------------------------------------------------------------------------
def cohort_table(all_client_ids: list[str] | None = None) -> list[dict]:
    """Per-dataset row: streams; experiments; bearings/channels per experiment;
    proposed physical nodes; proposed logical clients; simultaneous recording;
    onset/terminal/healthy documentation status; physical rig reuse.

    ``all_client_ids`` is optional -- when supplied (the full 57-stream
    historical roster, or any subset such as the 40-device source roster),
    per-dataset stream/experiment counts are recomputed from that exact list
    via :func:`experiment_id` (so the table is internally consistent with
    whatever roster produced predictions); the module-level
    ``HISTORICAL_STREAMS``/``HISTORICAL_EXPERIMENTS`` counts are always
    reported alongside as the frozen full-cohort reference. When omitted, the
    table reports the frozen full historical cohort only.
    """
    rows = []
    by_dataset: dict[str, list[str]] | None = None
    if all_client_ids is not None:
        by_dataset = {}
        for client_id in all_client_ids:
            by_dataset.setdefault(dataset_of(client_id), []).append(client_id)
    for dataset in DATASETS:
        streams_full = HISTORICAL_STREAMS[dataset]
        experiments_full = HISTORICAL_EXPERIMENTS[dataset]
        row = {
            "dataset": dataset,
            "signal_streams": streams_full,
            "machine_life_experiments": experiments_full,
            "bearings_per_experiment": BEARINGS_PER_EXPERIMENT.get(dataset, 1),
            "proposed_physical_federated_nodes": experiments_full,
            "proposed_logical_clients": streams_full,
            "recorded_simultaneously": SIMULTANEOUS_RECORDING[dataset],
            "onset_terminal_healthy_documentation": ONSET_TERMINAL_HEALTHY_DOCUMENTED[dataset],
            "physical_rig_reused": PHYSICAL_RIG_REUSED[dataset],
            "cohort_size_statement": (
                f"{streams_full} bearing-level or recording-unit streams corresponding to "
                f"{experiments_full} machine-life experiments"
            ),
        }
        if by_dataset is not None:
            present = by_dataset.get(dataset, [])
            groups = group_by_experiment(present)
            row["streams_in_supplied_roster"] = len(present)
            row["experiments_in_supplied_roster"] = len(groups)
        rows.append(row)
    return rows


def cohort_totals(rows: list[dict]) -> dict:
    total_streams = sum(int(row["signal_streams"]) for row in rows)
    total_experiments = sum(int(row["machine_life_experiments"]) for row in rows)
    totals = {
        "signal_streams": total_streams,
        "machine_life_experiments": total_experiments,
        "cohort_size_statement": (
            f"{total_streams} bearing-level or recording-unit streams corresponding to "
            f"approximately {total_experiments} machine-life experiments"
        ),
    }
    if all("streams_in_supplied_roster" in row for row in rows):
        totals["streams_in_supplied_roster"] = sum(int(row["streams_in_supplied_roster"]) for row in rows)
        totals["experiments_in_supplied_roster"] = sum(int(row["experiments_in_supplied_roster"]) for row in rows)
    return totals


def validate_no_partial_experiment(
    selected_client_ids: set[str] | list[str],
    universe_client_ids: list[str],
) -> bool:
    """True iff ``selected_client_ids`` never contains part, but not all, of
    any experiment defined over ``universe_client_ids``.

    This is the reusable structural check behind gate G2 (SS5) and is used
    both by ``raw_online_federated_v43_stats.py``'s clustered bootstrap (every
    resample must satisfy this) and directly by
    ``raw_online_federated_v43_cohort_tests.py``.
    """
    selected = set(selected_client_ids)
    universe_groups = group_by_experiment(list(universe_client_ids))
    for members in universe_groups.values():
        present = [client_id for client_id in members if client_id in selected]
        if present and len(present) != len(members):
            return False
    return True


__all__ = [
    "BEARINGS_PER_EXPERIMENT",
    "DATASETS",
    "HISTORICAL_EXPERIMENTS",
    "HISTORICAL_STREAMS",
    "HISTORICAL_TOTAL_EXPERIMENTS",
    "HISTORICAL_TOTAL_STREAMS",
    "ONSET_TERMINAL_HEALTHY_DOCUMENTED",
    "PHYSICAL_RIG_REUSED",
    "SIMULTANEOUS_RECORDING",
    "cohort_table",
    "cohort_totals",
    "dataset_of",
    "experiment_id",
    "group_by_experiment",
    "validate_no_partial_experiment",
]
