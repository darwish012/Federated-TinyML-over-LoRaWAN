"""machine_order_v2 -- amendment2_int16_raw_order.json ("machine_order_v2").

  construction: apply protocol.json machine_order.construction (rng
    default_rng(20260929), datasets in sorted name order, machines sorted
    by machine_id then shuffled within dataset, round-robin interleave) to
    the 41 eligible machines -> positions 0-40; then apply the same
    construction with a FRESH default_rng(20260929) to the 13 ineligible
    machines -> positions 41-53.
  board_assignment: unchanged, board = (position mod 10) + 1.
  levels: unchanged 4, 8, ... 52, 54; hardware levels 4-28.

The eligible / ineligible lists are frozen in the amendment file and are
READ from it (never recomputed), then checked to partition the 54-device
LDM-free cohort exactly. The construction itself is machine_order.build_order
(unmodified, imported), which builds a fresh rng from `seed` on every call --
exactly the "fresh default_rng(20260929)" the amendment asks for.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import machine_order as mo

HERE = Path(__file__).resolve().parent
AMENDMENT2 = HERE / "amendment2_int16_raw_order.json"
SEED = mo.SEED
LEVELS = mo.LEVELS
N_BOARDS = mo.N_BOARDS
HARDWARE_LEVELS = (4, 8, 12, 16, 20, 24, 28)
N_ELIGIBLE = 41
HARDWARE_POSITIONS = 28  # positions 0..27 -> levels 4-28


def load_amendment() -> dict:
    return json.loads(AMENDMENT2.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_order_v2(devices, amendment: dict | None = None, seed: int = SEED):
    """devices: the 54 LDM-free device dicts (machine_order.load_devices()).
    Returns the v2 order (list of 54 device dicts, positions 0-53)."""
    amendment = amendment if amendment is not None else load_amendment()
    eligible_ids = list(amendment["hardware_eligibility"]["eligible_41"])
    ineligible_ids = list(amendment["hardware_eligibility"]["ineligible_13"])
    if len(eligible_ids) != N_ELIGIBLE or len(ineligible_ids) != 13:
        raise RuntimeError("amendment eligible/ineligible list sizes are not 41/13")
    if set(eligible_ids) & set(ineligible_ids):
        raise RuntimeError("eligible and ineligible lists overlap")
    by_id = {d["machine_id"]: d for d in devices}
    if set(by_id) != set(eligible_ids) | set(ineligible_ids) or len(by_id) != 54:
        raise RuntimeError("amendment lists do not partition the 54-device cohort exactly")
    eligible = [by_id[m] for m in eligible_ids]
    ineligible = [by_id[m] for m in ineligible_ids]
    order = mo.build_order(eligible, seed=seed) + mo.build_order(ineligible, seed=seed)
    if len(order) != 54 or len({d["machine_id"] for d in order}) != 54:
        raise RuntimeError("machine_order_v2 lost or duplicated a device")
    return order


def build_all_v2(seed: int = SEED, levels=LEVELS):
    devices, noldm, manifest = mo.load_devices()
    order = build_order_v2(devices, seed=seed)
    assignment = mo.build_board_assignment(order)
    return {
        "devices": devices, "noldm": noldm, "manifest": manifest, "order": order,
        "board_assignment": assignment,
        "composition": mo.composition_table(order, levels=levels),
    }


def is_hardware_position(position: int) -> bool:
    return position < HARDWARE_POSITIONS


if __name__ == "__main__":
    info = build_all_v2()
    for k, d in enumerate(info["order"]):
        print(f"{k:2d} board {mo.board_of_position(k):2d} {d['machine_id']}")
    for row in info["composition"]:
        print(row)
