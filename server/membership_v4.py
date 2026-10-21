"""Board membership for protocol v4 (PROTOCOL_V4.md section 4).

Per DevEUI:

    ROSTERED -> BOUND     first valid HELLO: board_id <-> DevEUI, positions recorded
    BOUND    -> ACTIVE    acknowledged a GLOBAL of the current phase (or delivered an UPDATE)
    ACTIVE   -> SUSPECT   missed ``suspect_after_rounds`` rounds in a row
    SUSPECT  -> DEPARTED  missed ``depart_after_rounds`` rounds in a row
    SUSPECT  -> ACTIVE    delivered again or acknowledged a GLOBAL
    DEPARTED -> BOUND     spoke again; re-admitted at the next round boundary

A HELLO whose board_id is bound to another DevEUI, or whose positions collide with
another board's, is rejected. This module is pure state; the controller decides
when transitions happen and feeds the events to the logs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from protocol_v4 import NO_SLOT, BoardHello


class State(str, Enum):
    ROSTERED = "ROSTERED"
    BOUND = "BOUND"
    ACTIVE = "ACTIVE"
    SUSPECT = "SUSPECT"
    DEPARTED = "DEPARTED"


@dataclass
class Member:
    dev_eui: str
    device_id: str = ""
    state: State = State.ROSTERED
    board_id: int | None = None
    positions: tuple = ()
    fw_version: int = 0
    flags: int = 0
    counted: bool = False
    hello_count: int = 0
    flash_run_id: int = 0
    misses: int = 0                 # consecutive rounds without a delivered UPDATE
    ready: bool = False             # in Class C and listening
    ready_slot: int = NO_SLOT       # awaited slot of its last CLASS_C_READY
    ready_since: float = 0.0
    last_heard_t: float = 0.0
    held_final_phases: set = field(default_factory=set)   # phases whose final GLOBAL it acked
    counters: Counter = field(default_factory=Counter)

    @property
    def name(self) -> str:
        return self.device_id or self.dev_eui

    def to_dict(self) -> dict:
        return {
            "dev_eui": self.dev_eui, "device_id": self.device_id, "state": self.state.value,
            "board_id": self.board_id, "positions": list(self.positions),
            "fw_version": self.fw_version, "flags": self.flags, "counted": self.counted,
            "hello_count": self.hello_count, "flash_run_id": self.flash_run_id,
            "misses": self.misses, "held_final_phases": sorted(self.held_final_phases),
            "counters": dict(self.counters),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Member":
        return cls(
            dev_eui=d["dev_eui"], device_id=d.get("device_id", ""), state=State(d["state"]),
            board_id=d.get("board_id"), positions=tuple(d.get("positions", ())),
            fw_version=d.get("fw_version", 0), flags=d.get("flags", 0),
            counted=d.get("counted", False), hello_count=d.get("hello_count", 0),
            flash_run_id=d.get("flash_run_id", 0), misses=d.get("misses", 0),
            held_final_phases=set(d.get("held_final_phases", ())),
            counters=Counter(d.get("counters", {})),
        )


EventSink = Callable[[dict], None]


class Membership:
    """Roster-aware registry of boards: binding, collision checks, state changes."""

    def __init__(self, is_rostered: Callable[[str], bool] | None, emit: EventSink,
                 clock: Callable[[], float]) -> None:
        self._is_rostered = is_rostered
        self._emit = emit
        self._clock = clock
        self.members: dict[str, Member] = {}
        self.board_index: dict[int, str] = {}
        self.position_owner: dict[int, str] = {}

    # -- lookups -----------------------------------------------------------
    def get(self, dev_eui: str) -> Member | None:
        return self.members.get(dev_eui)

    def authorised(self, dev_eui: str) -> bool:
        return self._is_rostered is None or self._is_rostered(dev_eui)

    def in_state(self, *states: State) -> list[Member]:
        return [m for m in self.members.values() if m.state in states]

    def counts(self) -> dict:
        counter = Counter(m.state.value for m in self.members.values())
        return {s.value: int(counter.get(s.value, 0)) for s in State}

    def hosted_positions(self, include_departed: bool = False) -> set:
        return {
            p for m in self.members.values()
            if include_departed or m.state is not State.DEPARTED for p in m.positions
        }

    # -- transitions ---------------------------------------------------------
    def set_state(self, member: Member, new: State, reason: str, **context) -> bool:
        old = member.state
        if old is new:
            return False
        member.state = new
        self._emit({
            "dev_eui": member.dev_eui, "board_id": member.board_id,
            "from": old.value, "to": new.value, "reason": reason, **context,
        })
        return True

    def ensure_rostered(self, dev_eui: str, device_id: str) -> Member:
        member = self.members.get(dev_eui)
        if member is None:
            member = Member(dev_eui=dev_eui, device_id=device_id)
            self.members[dev_eui] = member
        return member

    def seed_roster(self, dev_euis, device_id_of: Callable[[str], str]) -> None:
        """Create ROSTERED members (never heard) for every rostered DevEUI."""
        for dev_eui in dev_euis:
            self.ensure_rostered(dev_eui, device_id_of(dev_eui))

    # -- HELLO -----------------------------------------------------------------
    def bind(self, dev_eui: str, device_id: str, hello: BoardHello) -> tuple[Member | None, dict | None]:
        """Apply a valid HELLO. Returns (member, None) or (None, rejection details)."""
        if not self.authorised(dev_eui):
            return None, {"code": "not_rostered", "reason": "DevEUI is not in the roster"}
        if hello.board_id == 0xFFFF:
            return None, {"code": "board_id_invalid", "reason": "board_id 0xFFFF is reserved"}
        member = self.members.get(dev_eui)
        if member is not None and member.board_id is not None and member.board_id != hello.board_id:
            return None, {
                "code": "board_id_changed", "bound_as": member.board_id,
                "now_claims": hello.board_id,
                "reason": "board.txt on this card changed; a DevEUI keeps its board_id for the run"}
        holder = self.board_index.get(hello.board_id)
        if holder is not None and holder != dev_eui:
            return None, {"code": "board_id_conflict", "board_id": hello.board_id,
                          "bound_to": holder,
                          "reason": "a second DevEUI claims a bound board_id"}
        collisions = {}
        for position in hello.positions:
            owner = self.position_owner.get(position)
            if owner is not None and owner != dev_eui:
                collisions[position] = owner
        if collisions:
            return None, {"code": "position_collision",
                          "positions": {str(p): o for p, o in sorted(collisions.items())},
                          "reason": "positions are already hosted by another bound board"}
        if (member is not None and member.positions and tuple(hello.positions) != member.positions
                and member.state in (State.ACTIVE, State.SUSPECT)):
            return None, {"code": "positions_changed_while_active",
                          "bound": list(member.positions), "now_claims": list(hello.positions),
                          "reason": "a participating board cannot change its machines mid-run"}

        member = self.ensure_rostered(dev_eui, device_id)
        # release positions this board no longer claims (only possible while not participating)
        for position in member.positions:
            if position not in hello.positions and self.position_owner.get(position) == dev_eui:
                del self.position_owner[position]
        member.device_id = device_id
        member.board_id = hello.board_id
        member.positions = tuple(hello.positions)
        member.fw_version = hello.fw_version
        member.flags = hello.flags
        member.counted = hello.counted
        member.flash_run_id = hello.stored_run_id
        member.hello_count += 1
        member.last_heard_t = self._clock()
        self.board_index[hello.board_id] = dev_eui
        for position in hello.positions:
            self.position_owner[position] = dev_eui
        if member.state is State.ROSTERED:
            self.set_state(member, State.BOUND, "first_hello", positions=list(member.positions))
        elif member.state is State.DEPARTED:
            self.set_state(member, State.BOUND, "spoke_again_hello")
            member.misses = 0
        return member, None

    # -- checkpoint ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"members": [m.to_dict() for m in self.members.values()]}

    def restore(self, data: dict) -> None:
        self.members.clear()
        self.board_index.clear()
        self.position_owner.clear()
        for entry in data["members"]:
            member = Member.from_dict(entry)
            self.members[member.dev_eui] = member
            if member.board_id is not None:
                self.board_index[member.board_id] = member.dev_eui
            for position in member.positions:
                self.position_owner[position] = member.dev_eui
