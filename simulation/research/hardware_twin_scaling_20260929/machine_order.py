"""Nested machine order and board assignment for protocol.json.

Implements protocol.json's "machine_order" and "board_assignment" fields
exactly:

  machine_order.construction: "rng = numpy default_rng(20260929); shuffle
    machines within each dataset with rng (datasets in sorted name order,
    machines sorted by machine_id before shuffling); then interleave
    round-robin over datasets in sorted name order, skipping exhausted
    datasets"

  board_assignment: "machine at position k (0-based) in machine_order runs
    on board (k mod 10) + 1, for every level. A machine never changes
    board ... Level N activates the first N machines; boards used =
    min(N, 10)."

Devices are the LDM-free 54-client cohort exactly as
research/healthy_only_causal_no_ldm_20260922/run_study.select_devices
returns it (sorted by machine_id, filtered to non-ldm_upb, 49 scored / 43
faults / 6 controls / 5 unknown asserted there).

Does not edit run_study.py or any other existing file; loads it by path via
importlib, the same pattern research/simulation_completion_20260929/
run_int8.py already uses for the same module.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # .../full_pipeline_simulation
NOLDM_DIR = HERE.parent / "healthy_only_causal_no_ldm_20260922"
CACHE = ROOT / "results" / "healthy_only_suite_20260907" / "cache"

SEED = 20260929
LEVELS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54]
N_BOARDS = 10


def _load_noldm():
    """Load run_study.py by path (not edited, not imported as a package)."""
    spec = importlib.util.spec_from_file_location("hwtwin_noldm", NOLDM_DIR / "run_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_devices():
    """Return (devices, noldm_module, manifest). devices: the 54 LDM-free
    device metadata dicts, sorted by machine_id (select_devices' own order;
    NOT the machine_order permutation -- that is built by build_order)."""
    noldm = _load_noldm()
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    devices = noldm.select_devices(manifest)
    noldm.verify_selected_cache_integrity(manifest, devices)
    return devices, noldm, manifest


def build_order(devices, seed=SEED):
    """Construct the one fixed nested order of `devices` per protocol.json's
    machine_order.construction, verbatim."""
    rng = np.random.default_rng(seed)
    by_dataset: dict[str, list] = {}
    for d in devices:
        by_dataset.setdefault(d["dataset"], []).append(d)
    dataset_names = sorted(by_dataset)

    queues = {}
    for name in dataset_names:
        machines = sorted(by_dataset[name], key=lambda d: d["machine_id"])
        rng.shuffle(machines)  # in place, Fisher-Yates, one shared rng
        queues[name] = machines

    order = []
    pos = {name: 0 for name in dataset_names}
    while True:
        progressed = False
        for name in dataset_names:
            i = pos[name]
            if i < len(queues[name]):
                order.append(queues[name][i])
                pos[name] += 1
                progressed = True
        if not progressed:
            break

    if len(order) != len(devices) or len({d["machine_id"] for d in order}) != len(devices):
        raise RuntimeError("machine_order construction lost or duplicated a device")
    return order


def board_of_position(position: int) -> int:
    """Board assignment: position k (0-based) -> board (k mod 10) + 1."""
    return (position % N_BOARDS) + 1


def build_board_assignment(order):
    """{machine_id: board_number} for every machine in `order` (1..10)."""
    return {d["machine_id"]: board_of_position(k) for k, d in enumerate(order)}


def boards_used(level: int) -> int:
    return min(level, N_BOARDS)


def machines_on_board(order, board: int, level: int | None = None):
    """Machines assigned to `board`, restricted to the first `level`
    machines of `order` if `level` is given (else the full order -- i.e.
    every machine that board will ever host, 5 or 6 of the 54)."""
    prefix = order if level is None else order[:level]
    return [d for k, d in enumerate(prefix) if board_of_position(k) == board]


def composition_row(prefix):
    """scored/faults/controls/unknown counts for a prefix of `order`."""
    faults = sum(1 for d in prefix if d.get("truth") is not None)
    controls = sum(1 for d in prefix if d.get("truth_kind") == "healthy_control")
    scored = faults + controls
    unknown = len(prefix) - scored
    return {"machines": len(prefix), "scored": scored, "faults": faults,
            "controls": controls, "unknown": unknown}


def composition_table(order, levels=LEVELS):
    """Per-level composition table: scored, faults, healthy controls (and
    unknown / boards_used), for `levels`, each a prefix of `order`."""
    rows = []
    for n in levels:
        row = {"level": n, "boards_used": boards_used(n)}
        row.update(composition_row(order[:n]))
        rows.append(row)
    return rows


def build_all(seed=SEED, levels=LEVELS):
    """Convenience: devices, order, board assignment, composition table."""
    devices, noldm, manifest = load_devices()
    order = build_order(devices, seed=seed)
    assignment = build_board_assignment(order)
    comp = composition_table(order, levels=levels)
    return {
        "devices": devices, "noldm": noldm, "manifest": manifest,
        "order": order, "board_assignment": assignment, "composition": comp,
    }


if __name__ == "__main__":
    info = build_all()
    print(f"order: {len(info['order'])} machines, "
          f"{len(set(info['board_assignment'].values()))} boards")
    for row in info["composition"]:
        print(row)
