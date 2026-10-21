"""Study geometry for the v3 experiment: levels, boards, machine order, FL modules.

Kept apart from the controller so the controller can be tested with fakes and
so the one place that touches the research folder is obvious.

The research code is imported UNCHANGED via sys.path:

    research/hardware_twin_scaling_20260929/twin_server.py   FedAdam aggregation
    research/hardware_twin_scaling_20260929/machine_order_v2.py  (machine order)
    research/fl_algorithm_comparison_20260929/fl.py          fl.init

Nothing here edits those folders, and the paths are APPENDED to sys.path so
this server's own ``protocol``/``controller`` modules can never be shadowed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

from protocol_v3 import (
    HARDWARE_LEVELS,
    N_BOARDS,
    board_of_position,
    level_index_of,
)

SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_RESEARCH_ROOT = (
    SERVER_DIR.parents[1] / "autoencoder_cwru_release" / "full_pipeline_simulation" / "research"
)
TWIN_DIR_NAME = "hardware_twin_scaling_20260929"
FL_DIR_NAME = "fl_algorithm_comparison_20260929"
SEED_ORDER = 20260929  # protocol.json machine_order.construction


def resolve_research_root(configured: str | Path | None) -> Path:
    root = Path(configured) if configured else DEFAULT_RESEARCH_ROOT
    if not root.is_absolute():
        root = (SERVER_DIR / root).resolve()
    return root


def _append_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.append(text)


def load_fl_modules(research_root: str | Path | None = None):
    """Import (twin_server, fl) from the research folder, unchanged."""
    root = resolve_research_root(research_root)
    twin_dir = root / TWIN_DIR_NAME
    fl_dir = root / FL_DIR_NAME
    for directory in (twin_dir, fl_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"research folder {directory} not found; set [v3] research_root in config.toml"
            )
    _append_path(twin_dir)
    _append_path(fl_dir)
    return importlib.import_module("twin_server"), importlib.import_module("fl")


# ---------------------------------------------------------------------------
# Machine order (amendment2_int16_raw_order.json "machine_order_v2")
# ---------------------------------------------------------------------------
def load_amendment2(research_root: str | Path | None = None) -> dict:
    path = resolve_research_root(research_root) / TWIN_DIR_NAME / "amendment2_int16_raw_order.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_order_standalone(amendment: dict, seed: int = SEED_ORDER) -> list[str]:
    """The amendment's construction, implemented from its text.

    protocol.json machine_order.construction: rng = default_rng(20260929);
    datasets in sorted name order; machines sorted by machine_id, then shuffled
    within their dataset with the one shared rng; then a round-robin interleave
    over datasets in sorted name order, skipping exhausted datasets. Applied to
    the 41 eligible machines (positions 0-40), then again with a FRESH rng to
    the 13 ineligible ones (positions 41-53).

    Used when machine_order_v2.py cannot be imported, and cross-checked against
    it in tests/test_study_plan.py. Needs numpy for the shuffle so it draws the
    same permutation as the reference.
    """
    import numpy as np

    def construct(machine_ids: list[str]) -> list[str]:
        rng = np.random.default_rng(seed)
        by_dataset: dict[str, list[str]] = {}
        for machine_id in machine_ids:
            by_dataset.setdefault(machine_id.split("::", 1)[0], []).append(machine_id)
        names = sorted(by_dataset)
        queues = {}
        for name in names:
            machines = sorted(by_dataset[name])
            rng.shuffle(machines)
            queues[name] = machines
        order: list[str] = []
        position = {name: 0 for name in names}
        while True:
            progressed = False
            for name in names:
                i = position[name]
                if i < len(queues[name]):
                    order.append(queues[name][i])
                    position[name] += 1
                    progressed = True
            if not progressed:
                break
        return order

    eligible = list(amendment["hardware_eligibility"]["eligible_41"])
    ineligible = list(amendment["hardware_eligibility"]["ineligible_13"])
    if len(eligible) != 41 or len(ineligible) != 13 or set(eligible) & set(ineligible):
        raise RuntimeError("amendment eligible/ineligible lists are not a 41/13 partition")
    order = construct(eligible) + construct(ineligible)
    if len(order) != 54 or len(set(order)) != 54:
        raise RuntimeError("machine order lost or duplicated a machine")
    return order


def order_from_research_module(research_root: str | Path | None = None) -> list[str]:
    """machine_order_v2.build_order_v2 driven by stub devices.

    machine_order.build_order only reads ``machine_id`` and ``dataset``, so the
    committed order can be reproduced without loading any dataset cache.
    """
    load_fl_modules(research_root)  # puts the twin folder on sys.path
    module = importlib.import_module("machine_order_v2")
    amendment = module.load_amendment()
    ids = list(amendment["hardware_eligibility"]["eligible_41"]) + list(
        amendment["hardware_eligibility"]["ineligible_13"]
    )
    devices = [{"machine_id": m, "dataset": m.split("::", 1)[0]} for m in sorted(ids)]
    return [d["machine_id"] for d in module.build_order_v2(devices, amendment)]


def load_machine_order(research_root: str | Path | None = None) -> tuple[list[str], str]:
    """(machine ids by position, source). Prefers the research module."""
    try:
        return order_from_research_module(research_root), "machine_order_v2"
    except Exception as exc:  # noqa: BLE001 - any import/data problem falls back
        fallback = build_order_standalone(load_amendment2(research_root))
        return fallback, f"amendment2 construction (machine_order_v2 unavailable: {exc})"


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------
def validate_levels(levels) -> list[int]:
    levels = [int(v) for v in levels]
    if not levels:
        raise ValueError("levels must not be empty")
    if levels != sorted(set(levels)):
        raise ValueError("levels must be strictly increasing")
    for level in levels:
        level_index_of(level)  # raises for anything outside the canonical list
    return levels


def active_positions(level: int) -> tuple[int, ...]:
    """PROTOCOL_V3: at level N the active machines are positions 0..N-1."""
    return tuple(range(level))


def boards_for_level(level: int) -> tuple[int, ...]:
    """Boards owning at least one active position (1-based, ascending)."""
    return tuple(sorted({board_of_position(p) for p in active_positions(level)}))


def positions_on_board(board: int, level: int) -> tuple[int, ...]:
    return tuple(p for p in active_positions(level) if board_of_position(p) == board)


def max_positions_per_board(level: int) -> int:
    return max(len(positions_on_board(b, level)) for b in range(1, N_BOARDS + 1))


__all__ = [
    "HARDWARE_LEVELS", "active_positions", "boards_for_level", "build_order_standalone",
    "load_amendment2", "load_fl_modules", "load_machine_order", "positions_on_board",
    "resolve_research_root", "validate_levels", "max_positions_per_board",
]
