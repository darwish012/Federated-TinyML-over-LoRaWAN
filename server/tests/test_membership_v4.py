"""Membership state machine: binding, collisions, departure and return, joins and leaves,
and the 50-board elastic scenario."""
from __future__ import annotations

import numpy as np

import protocol_v4 as p4
from controller_v4 import Stage
from membership_v4 import Membership, State
from plan_v4 import elastic_plan, planned_plan
from v4_ref import assert_matches_twin
from v4_sim import GROUP, HELLO_OK, RUN_ID, Sim, dev_eui_for


def hello(board_id, positions, flags=HELLO_OK, stored=0, fw=1):
    return p4.BoardHello(board_id, fw, flags, stored, tuple(positions))


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, e):
        self.events.append(e)


def make(rostered=None):
    rec = Recorder()
    m = Membership(None if rostered is None else (lambda d: d in rostered), rec, lambda: 5.0)
    return m, rec


# ---------------------------------------------------------------------------
# Unit: the registry
# ---------------------------------------------------------------------------
def test_rostered_to_bound_on_the_first_valid_hello_and_the_binding_is_recorded():
    m, rec = make()
    m.seed_roster(["A" * 16], lambda d: "eui-" + d.lower())
    assert m.get("A" * 16).state is State.ROSTERED and m.counts()["ROSTERED"] == 1
    member, rejection = m.bind("A" * 16, "eui-a", hello(7, [3, 9]))
    assert rejection is None and member.state is State.BOUND
    assert (member.board_id, member.positions, member.counted, member.hello_count) == (7, (3, 9), True, 1)
    assert m.board_index == {7: "A" * 16} and m.position_owner == {3: "A" * 16, 9: "A" * 16}
    assert [(e["from"], e["to"], e["reason"]) for e in rec.events] == [("ROSTERED", "BOUND", "first_hello")]


def test_an_unrostered_devEUI_is_rejected_and_never_becomes_a_member():
    m, rec = make(rostered={"A" * 16})
    member, rejection = m.bind("B" * 16, "eui-b", hello(1, [0]))
    assert member is None and rejection["code"] == "not_rostered"
    assert "B" * 16 not in m.members and not rec.events


def test_board_id_is_bound_to_one_deveui_and_a_deveui_keeps_its_board_id():
    m, _ = make()
    m.bind("A" * 16, "a", hello(1, [0]))
    member, rejection = m.bind("B" * 16, "b", hello(1, [5]))
    assert member is None and rejection["code"] == "board_id_conflict" and rejection["bound_to"] == "A" * 16
    member, rejection = m.bind("A" * 16, "a", hello(2, [0]))
    assert member is None and rejection["code"] == "board_id_changed"
    member, rejection = m.bind("C" * 16, "c", hello(0xFFFF, [6]))
    assert rejection["code"] == "board_id_invalid"


def test_position_collisions_are_rejected_and_nothing_is_recorded_for_the_loser():
    m, rec = make()
    m.bind("A" * 16, "a", hello(1, [0, 1]))
    member, rejection = m.bind("B" * 16, "b", hello(2, [1, 2]))
    assert member is None and rejection["code"] == "position_collision"
    assert rejection["positions"] == {"1": "A" * 16}
    assert "B" * 16 not in m.members and 2 not in m.board_index and 2 not in m.position_owner
    assert len(rec.events) == 1


def test_a_hello_that_changes_positions_is_accepted_only_while_not_participating():
    m, _ = make()
    member, _ = m.bind("A" * 16, "a", hello(1, [0, 1]))
    member2, rejection = m.bind("A" * 16, "a", hello(1, [1, 4]))
    assert rejection is None and member2.positions == (1, 4)
    assert 0 not in m.position_owner and m.position_owner[4] == "A" * 16      # released and claimed
    m.set_state(member, State.ACTIVE, "test")
    _, rejection = m.bind("A" * 16, "a", hello(1, [7]))
    assert rejection["code"] == "positions_changed_while_active"
    same, rejection = m.bind("A" * 16, "a", hello(1, [1, 4]))
    assert rejection is None and same.hello_count == 3 and same.state is State.ACTIVE   # a reboot


def test_a_departed_board_that_says_hello_re_enters_bound():
    m, rec = make()
    member, _ = m.bind("A" * 16, "a", hello(1, [0]))
    m.set_state(member, State.DEPARTED, "missed_rounds")
    member.misses = 5
    again, rejection = m.bind("A" * 16, "a", hello(1, [0]))
    assert rejection is None and again.state is State.BOUND and again.misses == 0
    assert [e["to"] for e in rec.events] == ["BOUND", "DEPARTED", "BOUND"]
    assert rec.events[-1]["reason"] == "spoke_again_hello"


def test_departed_boards_do_not_hold_positions_against_counting_but_do_collide():
    m, _ = make()
    member, _ = m.bind("A" * 16, "a", hello(1, [0]))
    m.set_state(member, State.DEPARTED, "missed_rounds")
    assert m.hosted_positions() == set() and m.hosted_positions(include_departed=True) == {0}
    _, rejection = m.bind("B" * 16, "b", hello(2, [0]))
    assert rejection["code"] == "position_collision"            # a replacement must not silently take its machines


def test_member_round_trips_through_a_checkpoint():
    m, _ = make()
    member, _ = m.bind("A" * 16, "a", hello(1, [0, 5], flags=HELLO_OK))
    member.counters["uplinks"] = 7
    member.held_final_phases.add(2)
    member.misses = 3
    m.set_state(member, State.SUSPECT, "missed_rounds")
    other, _ = make()
    other.restore(m.to_dict())
    got = other.get("A" * 16)
    assert (got.state, got.board_id, got.positions, got.misses, got.held_final_phases) == (
        State.SUSPECT, 1, (0, 5), 3, {2})
    assert got.counters["uplinks"] == 7 and other.board_index == {1: "A" * 16}
    assert other.position_owner == {0: "A" * 16, 5: "A" * 16}


# ---------------------------------------------------------------------------
# Through the controller
# ---------------------------------------------------------------------------
def test_position_collision_through_the_controller_is_rejected_and_logged():
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0, 1), 2: (1, 2)}, min_boards=1)
    c = sim.controller
    sim.boards[1].hello()
    sim.boards[2].hello()                           # claims position 1 too
    rejected = [x for e, _, x in sim.events if e == "hello_rejected"]
    assert len(rejected) == 1 and rejected[0]["code"] == "position_collision"
    assert c.membership.get(dev_eui_for(2)) is None
    assert c.counters["hello_rejected_position_collision"] == 1
    # the rejected board's uplinks are ignored, it is never a participant
    sim.boards[2].ready(p4.NO_SLOT)                 # (hello() already sent one READY: both ignored)
    assert c.counters["ignored_device_without_hello"] == 2
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert c.stage is Stage.DONE
    assert sim.round_records(0)[0]["expected_positions"] == [0, 1]


def test_a_second_deveui_cannot_take_over_a_bound_board_id_and_hello_flags_are_checked():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,)}, min_boards=1)
    c = sim.controller
    sim.boards[1].hello()
    sim.controller.handle_uplink("eui-70b3d57ed0000fff", 10, p4.encode_board_hello(1, 1, HELLO_OK, 0, (9,)))
    assert [x["code"] for e, _, x in sim.events if e == "hello_rejected"] == ["board_id_conflict"]
    # more positions than max_machines_per_board is a protocol violation, not a member
    positions = tuple(range(20, 29))
    assert len(positions) == 9
    c.handle_uplink("eui-70b3d57ed0000fff", 10, p4.encode_frame(
        p4.MSG_BOARD_HELLO, p4.NO_SLOT, 0, 1,
        b"\x05\x00\x01\x03\x00\x00\x00\x00\x09" + b"".join(p.to_bytes(2, "little") for p in positions)))
    assert c.counters["frames_rejected"] >= 1 and c.membership.get("70B3D57ED0000FFF") is None


def test_max_machines_per_board_is_enforced():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0, 1, 2)}, min_boards=1, max_machines_per_board=2)
    sim.boards[1].hello()
    assert sim.controller.counters["frames_rejected"] == 1
    assert sim.controller.membership.get(dev_eui_for(1)) is None


def test_the_roster_authorises_members_only():
    allowed = {dev_eui_for(1), dev_eui_for(2)}
    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,), 3: (2,)}, min_boards=2,
              is_rostered=lambda d: d in allowed)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE
    assert c.counters["hello_rejected_not_rostered"] == 1
    assert {m.dev_eui for m in c.membership.members.values()} == allowed
    assert sim.round_records(0)[0]["expected_positions"] == [0, 1]


def test_rostered_but_silent_devices_are_listed_as_rostered():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,)})
    sim.controller.sync_roster([dev_eui_for(1), dev_eui_for(9)], lambda d: "eui-" + d.lower())
    sim.run_to_done()
    counts = sim.controller.membership.counts()
    assert counts["ROSTERED"] == 1 and counts["ACTIVE"] == 1


def test_a_rebooted_active_board_is_not_counted_twice_and_stays_active():
    sim = Sim(elastic_plan(17070, 1, 6), boards={1: (0,), 2: (1,)})
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    board = sim.boards[2]                                     # reboot: HELLO again
    board.uplink(p4.encode_board_hello(2, 1, HELLO_OK, board.stored_run_id, board.positions))
    member = c.membership.get(dev_eui_for(2))
    assert member.hello_count == 2 and member.state is State.ACTIVE
    assert not member.ready                                   # it lost its Class C state
    board.ready(p4.NO_SLOT)
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert c.stage is Stage.DONE and "board_rebooted_mid_run" in [e for e, _, _ in sim.events]
    assert c.membership.counts()["ACTIVE"] == 2
    assert_matches_twin(sim, 0)


# ---------------------------------------------------------------------------
# A board departs and returns
# ---------------------------------------------------------------------------
def departure_params():
    return dict(round_timeout_seconds=60.0, suspect_after_rounds=1, depart_after_rounds=2,
                ready_wait_seconds=20.0, multicast_report_window_seconds=20.0)


def test_a_board_departs_and_returns_and_is_readmitted_at_a_round_boundary():
    sim = Sim(elastic_plan(17070, 1, 12), boards={b: (b - 1,) for b in range(1, 5)}, **departure_params())
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    sim.boards[3].power_off()
    member = c.membership.get(dev_eui_for(3))
    sim.run_until(lambda: member.state is State.DEPARTED, max_steps=3000)
    closed_when_gone = len(sim.out.records("round"))
    sim.run_until(lambda: len(sim.out.records("round")) >= closed_when_gone + 2)
    sim.boards[3].power_on()                                  # HELLO: spoke again
    assert member.state is State.BOUND and member.misses == 0
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=20000)
    assert c.stage is Stage.DONE and member.state is State.ACTIVE
    transitions = [(e["from"], e["to"]) for e in sim.out.membership_events if e["dev_eui"] == dev_eui_for(3)]
    assert transitions == [("ROSTERED", "BOUND"), ("BOUND", "ACTIVE"), ("ACTIVE", "SUSPECT"),
                           ("SUSPECT", "DEPARTED"), ("DEPARTED", "BOUND"), ("BOUND", "ACTIVE")]
    recs = sim.round_records(0)
    by_round = {r["round"]: r for r in recs}
    timeouts = [r["round"] for r in recs if r["close_reason"] == "timeout"]
    died = timeouts[0]                                        # the board trained one more round before it died
    assert timeouts == [died] and died >= 3                   # only the round it died in waited for it
    assert by_round[died]["participants"] == [0, 1, 2, 3] and by_round[died]["missing_positions"] == [2]
    assert by_round[died]["left_boards"] == [3]               # ACTIVE -> SUSPECT at that round's close
    nxt = by_round[died + 1]
    assert nxt["participants"] == [0, 1, 3] and nxt["left_positions"] == [2]
    assert nxt["left_boards"] == [3]                          # SUSPECT -> DEPARTED, nobody waited
    rejoin = [r for r in recs if r["joined_positions"] == [2]]
    assert len(rejoin) == 1 and len([r for r in recs if 3 in r["joined_boards"]]) == 2
    assert all(r["arrived_positions"] == [0, 1, 3] for r in recs if died < r["round"] < rejoin[0]["round"])
    assert recs[-1]["participants"] == [0, 1, 2, 3] and recs[-1]["participating_boards"] == [1, 2, 3, 4]
    # the departed board's positions are absent in the masks (false/false) and present again after
    masks = sim.masks(0)
    col = masks["positions"].index(2)
    assert [row[col] for row in masks["uplink"]][died - 1:rejoin[0]["round"] - 1] == [False] * (rejoin[0]["round"] - died)
    assert masks["uplink"][-1][col] and masks["downlink"][-1][col]
    assert_matches_twin(sim, 0)


def test_a_suspect_board_is_not_waited_for_and_recovers_when_it_delivers_again():
    sim = Sim(elastic_plan(17070, 1, 10), boards={b: (b - 1,) for b in range(1, 4)},
              round_timeout_seconds=60.0, suspect_after_rounds=1, depart_after_rounds=5,
              ready_wait_seconds=20.0)
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    sim.boards[2].power_off()
    member = c.membership.get(dev_eui_for(2))
    sim.run_until(lambda: member.state is State.SUSPECT, max_steps=3000)
    t = sim.clock.now
    sim.run_until(lambda: len(sim.out.records("round")) >= 6)
    assert sim.clock.now - t < 200                             # rounds close promptly: nobody waits for it
    sim.boards[2].power_on(hello=False)                         # the radio is back, no reboot
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=20000)
    assert member.state is State.ACTIVE and member.misses == 0
    assert_matches_twin(sim, 0)


# ---------------------------------------------------------------------------
# Joins at round boundaries
# ---------------------------------------------------------------------------
def test_a_late_board_joins_at_the_next_round_boundary():
    sim = Sim(elastic_plan(17070, 1, 8), boards={1: (0,), 2: (1,), 3: (2,)}, min_boards=2)
    c = sim.controller
    sim.start(only={1, 2})
    sim.run_until(lambda: len(sim.out.records("round")) >= 3)
    sim.boards[3].hello()                                      # mid-phase
    assert c.membership.get(dev_eui_for(3)).state is State.BOUND
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    recs = sim.round_records(0)
    first = min(r["round"] for r in recs if 3 in r["participating_boards"])
    assert first > 3 and all(3 in r["participating_boards"] for r in recs if r["round"] >= first)
    assert [r["expected_positions"] for r in recs if r["round"] < first] == [[0, 1]] * (first - 1)
    assert sim.controller.membership.get(dev_eui_for(3)).state is State.ACTIVE
    assert 2 in sim.masks(0)["positions"]
    assert_matches_twin(sim, 0)


def test_a_board_holding_another_run_is_sent_adopt_and_joins_mid_phase():
    """HELLO with a different non-zero stored_run_id -> RUN_CONTROL ADOPT (3.4), then it joins (5.1)."""
    sim = Sim(elastic_plan(17070, 1, 8), boards={1: (0,), 2: (1,), 3: (2,)},
              stored_runs={3: 0x0BADF00D}, min_boards=2)
    c = sim.controller
    sim.start(only={1, 2})
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    adopts_before = c.counters["adopt_commands"]
    assert adopts_before == 0                                  # nobody held another run at the start
    board = sim.boards[3]
    board.hello()
    assert c.membership.get(dev_eui_for(3)).flash_run_id == 0x0BADF00D
    assert "board_holds_other_run" in [e for e, _, _ in sim.events]
    sim.run_until(lambda: sim.adopts_heard[3] >= 1, max_steps=500)       # it was off the air at run start
    assert c.counters["adopt_commands"] == adopts_before + 1
    assert board.stored_run_id == RUN_ID and board.adoptions == 1
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    recs = sim.round_records(0)
    first = min(r["round"] for r in recs if 2 in r["participants"])
    assert first >= 3 and all(2 in r["participants"] for r in recs if r["round"] >= first)
    assert_matches_twin(sim, 0)


def test_without_adopt_a_foreign_run_board_waits_for_a_phase_start():
    """The contract's safety rule: a GLOBAL of another run with round > 0 is ignored; round 0 adopts."""
    sim = Sim(elastic_plan(17070, 2, 4), boards={1: (0,), 2: (1,), 3: (2,)},
              stored_runs={3: 0x0BADF00D}, min_boards=2)
    c = sim.controller
    board = sim.boards[3]
    board.receive_run_control = lambda frame: None             # it never hears ADOPT
    sim.start(only={1, 2})
    sim.run_until(lambda: len(sim.out.records("round")) >= 1)
    board.hello()
    member = c.membership.get(dev_eui_for(3))
    sim.run_until(lambda: len(sim.out.records("round")) >= 4)
    assert member.state is State.BOUND and board.stored_run_id == 0x0BADF00D
    assert all(r["participants"] == [0, 1] for r in sim.round_records(0))
    assert not any(r["close_reason"] == "timeout" for r in sim.round_records(0))   # nobody waited for it
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert member.state is State.ACTIVE                         # round 0 of phase 1 adopts the run
    assert [r["participants"] for r in sim.round_records(1)] == [[0, 1, 2]] * 4
    assert_matches_twin(sim, 0)
    assert_matches_twin(sim, 1)


def test_adopt_is_multicast_at_run_start_before_the_first_global_and_coalesced():
    """Only when a bound board still reports another run: otherwise the answers to an ADOPT (a HELLO
    from every board) would collide with the first GLOBAL."""
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,), 2: (1,), 3: (2,), 4: (3,)}, min_boards=2,
              stored_runs={3: 0x11, 4: 0x22})
    c = sim.controller
    sim.start(only={1, 2, 3})
    sim.run_until(lambda: len(sim.published) >= 2)
    first, second = sim.published[0], sim.published[1]
    f1, f2 = p4.decode_frame(first[1]), p4.decode_frame(second[1])
    assert first[0] == GROUP and f1.message_type == p4.MSG_RUN_CONTROL
    assert p4.decode_run_control(f1).command == p4.CMD_ADOPT and p4.decode_run_control(f1).run_id == RUN_ID
    assert f2.message_type == p4.MSG_GLOBAL_FRAGMENT and f2.round_id == 1
    # two boards with other runs say HELLO in the same tick: one ADOPT serves both
    sim.run_until(lambda: sim.adopts_heard[3] >= 1, max_steps=500)
    sim.run_until(lambda: sim.boards[3].stored_run_id == RUN_ID)
    sim.boards[4].hello()
    n = c.counters["adopt_commands"]
    sim.run_until(lambda: c.counters["adopt_commands"] > n, max_steps=500)
    sim.run_until(lambda: sim.adopts_heard[4] >= 1, max_steps=500)
    assert c.counters["adopt_commands"] == n + 1
    assert sim.boards[3].stored_run_id == RUN_ID == sim.boards[4].stored_run_id
    # a board that already holds this run never triggers an ADOPT
    sim.boards[1].uplink(p4.encode_board_hello(1, 1, HELLO_OK, RUN_ID, (0,)))
    assert 1 not in c._adopt_waiting


def test_a_run_complete_board_accepts_any_global_without_adopt():
    sim = Sim(elastic_plan(17070, 1, 6), boards={1: (0,), 2: (1,)}, min_boards=1,
              stored_runs={2: 0x0BADF00D})
    sim.boards[2].run_complete = True
    sim.boards[2].flags |= p4.HELLO_RUN_COMPLETE
    sim.boards[2].receive_run_control = lambda frame: None
    sim.start(only={1})
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    sim.boards[2].hello()
    sim.run_until(lambda: sim.boards[2].stored_run_id == RUN_ID, max_steps=500)    # adopts on a mid-phase GLOBAL
    sim.run_until(lambda: sim.controller.stage in (Stage.DONE, Stage.FAILED))
    assert any(1 in rec["participants"] for rec in sim.round_records(0))      # position 1 took part

def test_planned_boards_hosting_no_position_of_a_phase_become_active_when_a_later_phase_needs_them():
    sim = Sim(planned_plan([2, 4], 17070, 3), boards={b: (b - 1,) for b in range(1, 5)})
    c = sim.controller
    sim.run_to_done()
    assert c.stage is Stage.DONE
    p0 = sim.round_records(0)
    assert p0[0]["participating_boards"] == [1, 2] and p0[0]["joined_boards"] == [1, 2]
    p1 = sim.round_records(1)
    assert p1[0]["participating_boards"] == [1, 2, 3, 4] and sorted(p1[0]["joined_boards"]) == [3, 4]
    assert [r["expected_positions"] for r in p1] == [[0, 1, 2, 3]] * 3
    assert_matches_twin(sim, 0)
    assert_matches_twin(sim, 1)


# ---------------------------------------------------------------------------
# 50 virtual boards, elastic, joins and leaves mid-phase
# ---------------------------------------------------------------------------
def fifty_boards():
    boards = {}
    for b in range(1, 51):
        boards[b] = (b - 1,) if not 11 <= b <= 20 else (b - 1, 100 + b)     # boards 11..20 host two machines
    return boards


def test_fifty_boards_elastic_run_with_joins_and_leaves_mid_phase():
    rounds = 6
    foreign = {b: 0x0BADF000 + b for b in range(41, 51)}      # boards 41..50 still hold some old run
    sim = Sim(elastic_plan(17070, 2, rounds), boards=fifty_boards(), stored_runs=foreign, min_boards=20,
              round_timeout_seconds=100.0, suspect_after_rounds=1, depart_after_rounds=2,
              ready_wait_seconds=30.0, multicast_report_window_seconds=30.0, post_run_collect_seconds=200.0)
    c = sim.controller
    closed = lambda: len(sim.out.records("round"))             # noqa: E731
    sim.start(only=set(range(1, 31)))
    sim.run_until(lambda: closed() >= 2)
    assert c.stage in (Stage.DOWNLINK, Stage.UPLOAD)
    for b in range(31, 51):
        sim.boards[b].hello()                                   # 20 boards join mid-phase
    sim.run_until(lambda: closed() >= 3)
    for b in (5, 6, 7):
        sim.boards[b].power_off()                               # three boards leave
    sim.run_until(lambda: closed() >= rounds + 2)               # into phase 1
    assert c.membership.get(dev_eui_for(5)).state is State.DEPARTED
    sim.boards[7].power_on()                                    # one of them returns
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=60000)
    assert c.stage is Stage.DONE and not c.failed

    p0, p1 = sim.round_records(0), sim.round_records(1)
    assert len(p0) == len(p1) == rounds
    # phase 0: the 30 starters first; the 20 mid-phase joiners (10 fresh boards that adopt any GLOBAL,
    # 10 holding an old run that adopt after RUN_CONTROL ADOPT) join at round boundaries, never at round 1
    assert set(range(1, 31)) <= set(p0[0]["participating_boards"])
    assert not set(range(31, 51)) & set(p0[0]["participating_boards"])
    joined_p0 = {b for r in p0 for b in r["joined_boards"]}
    assert set(range(31, 51)) <= joined_p0
    assert all(sim.boards[b].adoptions == 1 and sim.adopts_heard[b] >= 1 for b in range(41, 51))   # via ADOPT
    assert all(sim.boards[b].adoptions == 1 for b in range(31, 41))     # fresh boards adopt on their first GLOBAL
    assert all(sim.boards[b].stored_run_id == sim.run_id for b in range(31, 51))
    assert c.counters["adopt_commands"] >= 1                            # the foreign HELLOs
    left = {b for r in p0 + p1 for b in r["left_boards"]}
    assert {5, 6, 7} <= left
    # phase 1 starts from fl.init again with every non-departed board
    assert set(range(31, 51)) <= set(p1[0]["participating_boards"])
    assert not {5, 6} & set(p1[0]["participating_boards"]) and not {5, 6} & set(p1[-1]["participating_boards"])
    assert 7 in p1[-1]["participating_boards"]                  # re-admitted
    states = {b: c.membership.get(dev_eui_for(b)).state for b in range(1, 51)}
    assert states[5] is State.DEPARTED and states[6] is State.DEPARTED and states[7] is State.ACTIVE
    assert all(states[b] is State.ACTIVE for b in range(1, 51) if b not in (5, 6))
    # positions seen include the two-machine boards' second positions; masks are aligned to them
    masks = sim.masks(1)
    # (boards 5 and 6 had departed: their positions were never seen in phase 1)
    assert masks["positions"] == sorted((set(range(50)) - {4, 5}) | {100 + b for b in range(11, 21)})
    assert {4, 5} <= set(sim.masks(0)["positions"])
    assert all(len(row) == len(masks["positions"]) for row in masks["uplink"])
    # a position is absent (false / false) before its board joined, present after
    m0 = sim.masks(0)                                           # board 31 hosts position 30
    col0 = m0["positions"].index(30)
    first_row = next(r for r, row in enumerate(m0["uplink"]) if row[col0])
    assert first_row >= 2 and not any(m0["uplink"][r][col0] or m0["downlink"][r][col0] for r in range(first_row - 1))
    assert all(m0["base_slot"][r][col0] is None for r in range(first_row))
    # every position of a participating board arrived in a round where nobody timed out
    full = [r for r in p1 if r["close_reason"] == "all_delivered"]
    assert full and all(len(r["arrived_positions"]) == len(r["expected_positions"]) for r in full)
    for phase in (0, 1):
        server = assert_matches_twin(sim, phase)
        for ours, theirs in zip(c.phases[phase].server.p, server.p):
            assert np.array_equal(ours, theirs)
    events = {(e["from"], e["to"]) for e in sim.out.membership_events}
    assert {("ROSTERED", "BOUND"), ("BOUND", "ACTIVE"), ("ACTIVE", "SUSPECT"),
            ("SUSPECT", "DEPARTED"), ("DEPARTED", "BOUND")} <= events