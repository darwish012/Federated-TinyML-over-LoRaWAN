"""Run plan for protocol v4: planned phases or an elastic run (PROTOCOL_V4.md section 4).

*planned*: a list of phases, each with ``positions`` (an explicit set or "first N")
and ``rounds`` (default 12). The thesis sweep is 7 phases, first 4, 8, ..., 28,
12 rounds each, and is built from machine_order_v2 exactly like v3's study_plan:
position p is the machine at index p of that order, "first N" = positions 0..N-1.

*elastic*: one (or more) phases of R rounds; the participants are whoever is
ACTIVE, so a phase carries no position list (``positions is None``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from protocol_v4 import MASK_WINDOW, MAX_PHASE_INDEX, MAX_POSITION, Participation

MODES = ("planned", "elastic")
THESIS_LEVELS = (4, 8, 12, 16, 20, 24, 28)
DEFAULT_ROUNDS = 12


@dataclass(frozen=True)
class PhaseSpec:
    index: int
    name: str
    positions: tuple | None        # None = elastic (any position of an ACTIVE board)
    rounds: int

    def covers(self, position: int) -> bool:
        return self.positions is None or position in self.positions

    def participation(self) -> Participation:
        """What every GLOBAL of this phase multicasts (rev 4.3)."""
        return Participation.everyone() if self.positions is None else Participation.of_positions(self.positions)

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "rounds": self.rounds,
                "positions": None if self.positions is None else list(self.positions)}

    @classmethod
    def from_dict(cls, d: dict) -> "PhaseSpec":
        pos = d["positions"]
        return cls(int(d["index"]), str(d["name"]), None if pos is None else tuple(pos),
                   int(d["rounds"]))


@dataclass(frozen=True)
class RunPlan:
    mode: str
    phases: tuple
    seed: int
    machine_order: tuple = ()      # machine id by position (planned runs; may be empty)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        if not self.phases:
            raise ValueError("a run plan needs at least one phase")
        if len(self.phases) > MAX_PHASE_INDEX + 1:
            raise ValueError(f"at most {MAX_PHASE_INDEX + 1} phases are allowed")
        for i, phase in enumerate(self.phases):
            if phase.index != i:
                raise ValueError("phase indices must be 0, 1, 2, ... in order")

    def phase(self, index: int) -> PhaseSpec:
        return self.phases[index]

    def all_positions(self) -> tuple:
        """Union of every planned position (empty for an elastic plan)."""
        out: set = set()
        for phase in self.phases:
            out |= set(phase.positions or ())
        return tuple(sorted(out))

    def machine_of(self, position: int) -> str | None:
        if 0 <= position < len(self.machine_order):
            return self.machine_order[position]
        return None

    def to_dict(self) -> dict:
        return {"mode": self.mode, "seed": self.seed,
                "phases": [p.to_dict() for p in self.phases],
                "machine_order": list(self.machine_order)}

    @classmethod
    def from_dict(cls, d: dict) -> "RunPlan":
        return cls(str(d["mode"]), tuple(PhaseSpec.from_dict(p) for p in d["phases"]),
                   int(d["seed"]), tuple(d.get("machine_order", ())))


# ---------------------------------------------------------------------------
# Parsing the [v4] phase list
# ---------------------------------------------------------------------------
def parse_phase_specs(raw: Iterable, default_rounds: int,
                      machine_count: int | None = None) -> tuple[list, list]:
    """Turn the config's ``phases`` into PhaseSpec objects.

    Each entry is an int N ("first N positions") or a table with exactly one of
    ``first`` (N) / ``positions`` (list) and optional ``rounds`` and ``name``.
    Returns (specs, error strings) so the caller can report every problem at once.
    """
    errors: list[str] = []
    specs: list[PhaseSpec] = []
    for i, entry in enumerate(list(raw)):
        where = f"v4.phases[{i}]"
        rounds = default_rounds
        name = None
        positions: tuple | None = None
        if isinstance(entry, bool):
            errors.append(f"{where}: a phase must be an integer or a table, not a boolean")
            continue
        if isinstance(entry, int):
            first = entry
            positions = tuple(range(first)) if first >= 1 else None
            if first < 1:
                errors.append(f"{where}: 'first N' needs N >= 1")
            name = f"first{first}"
        elif isinstance(entry, dict):
            unknown = set(entry) - {"first", "positions", "rounds", "name"}
            if unknown:
                errors.append(f"{where}: unknown keys {sorted(unknown)}")
            if ("first" in entry) == ("positions" in entry):
                errors.append(f"{where}: give exactly one of 'first' or 'positions'")
                continue
            if "first" in entry:
                first = entry["first"]
                if not isinstance(first, int) or isinstance(first, bool) or first < 1:
                    errors.append(f"{where}.first must be an integer >= 1")
                    continue
                positions = tuple(range(first))
                name = str(entry.get("name", f"first{first}"))
            else:
                listed = entry["positions"]
                if (not isinstance(listed, list) or not listed
                        or any(not isinstance(p, int) or isinstance(p, bool) for p in listed)):
                    errors.append(f"{where}.positions must be a non-empty list of integers")
                    continue
                if len(set(listed)) != len(listed):
                    errors.append(f"{where}.positions lists a position twice")
                    continue
                positions = tuple(sorted(listed))
                name = str(entry.get("name", f"set{len(positions)}"))
            if "rounds" in entry:
                rounds = entry["rounds"]
        else:
            errors.append(f"{where}: must be an integer N or a table, not {type(entry).__name__}")
            continue
        if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 255:
            errors.append(f"{where}: rounds must be an integer in 1..255")
            continue
        if positions is not None:
            if any(p < 0 or p > MAX_POSITION for p in positions):
                errors.append(f"{where}: positions must be in 0..{MAX_POSITION}")
                continue
            if positions and positions[-1] - positions[0] >= MASK_WINDOW:
                errors.append(
                    f"{where}: positions {positions[0]}..{positions[-1]} do not fit one "
                    f"{MASK_WINDOW}-position window (rev 4.3 participation mask); split the phase "
                    "or use mode = \"elastic\"")
                continue
            if machine_count is not None and positions and max(positions) >= machine_count:
                errors.append(
                    f"{where}: position {max(positions)} has no machine in the machine order "
                    f"({machine_count} machines)")
                continue
        specs.append(PhaseSpec(len(specs), name or f"phase{len(specs)}", positions, rounds))
    return specs, errors


def planned_plan(raw_phases, seed: int, default_rounds: int = DEFAULT_ROUNDS,
                 machine_order=None) -> RunPlan:
    order = tuple(machine_order or ())
    specs, errors = parse_phase_specs(raw_phases, default_rounds, len(order) or None)
    if errors:
        raise ValueError("; ".join(errors))
    return RunPlan("planned", tuple(specs), int(seed), order)


def thesis_plan(seed: int, rounds: int = DEFAULT_ROUNDS, machine_order=None) -> RunPlan:
    """The thesis sweep: 7 phases, first 4, 8, ..., 28 positions, 12 rounds each."""
    return planned_plan(list(THESIS_LEVELS), seed, rounds, machine_order)


def elastic_plan(seed: int, phases: int = 1, rounds: int = DEFAULT_ROUNDS) -> RunPlan:
    if not 1 <= phases <= MAX_PHASE_INDEX + 1:
        raise ValueError(f"elastic phases must be 1..{MAX_PHASE_INDEX + 1}")
    if not 1 <= rounds <= 255:
        raise ValueError("rounds must be 1..255")
    return RunPlan("elastic",
                   tuple(PhaseSpec(i, f"elastic{i}", None, rounds) for i in range(phases)),
                   int(seed))


def positions_without_board(plan: RunPlan, hosted: Iterable[int]) -> dict:
    """Planned positions that no bound board hosts, per phase (empty for elastic)."""
    hosted_set = set(hosted)
    out = {}
    for phase in plan.phases:
        if phase.positions is None:
            continue
        missing = [p for p in phase.positions if p not in hosted_set]
        if missing:
            out[phase.index] = missing
    return out
