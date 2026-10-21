"""Liveness: a silent board never delays the run, and re-enters the gate as soon as it speaks."""
from __future__ import annotations

import json

import pytest

import config_v4 as cv
import protocol_v4 as p4
from controller_v4 import ControllerParams, Stage
from membership_v4 import State
from plan_v4 import elastic_plan, thesis_plan
from run_outputs_v4 import RunOutputs, load_checkpoint
from study_plan import load_machine_order
from v4_ref import assert_matches_twin
from v4_sim import Sim, dev_eui_for, thesis_boards

LIVE = 60.0                     # liveness_seconds for the simulated runs (a round takes tens of seconds)
OLD = 1e9                       # liveness effectively off: the behaviour before the fix
DIES_AT, RETURNS_AT = 30, 62    # closed-round counts: phase 2 round 7 .. phase 5 round 3 (12 rounds per phase)


def sweep(liveness, dead=None, plan=None, sim_cls=Sim, dies_at=DIES_AT, **kwargs):
    """10-board planned thesis sweep; board ``dead`` (default none) dies mid-phase 2 and returns in phase 5."""
    plan = plan or thesis_plan(17070, 12, load_machine_order()[0])
    sim = sim_cls(plan, boards=thesis_boards(10, 28), liveness_seconds=liveness, round_timeout_seconds=600.0,
                  post_run_collect_seconds=300.0, **kwargs)
    closed = lambda: len(sim.out.records("round"))                   # noqa: E731
    sim.start()
    if dead is not None:
        sim.run_until(lambda: closed() >= dies_at, max_steps=200000)
        sim.boards[dead].power_off()
        sim.run_until(lambda: closed() >= RETURNS_AT, max_steps=400000)
        sim.boards[dead].power_on()
    sim.run_until(lambda: sim.controller.stage in (Stage.DONE, Stage.FAILED), max_steps=400000)
    assert sim.controller.stage is Stage.DONE, sim.controller.failure_reason
    return sim


def duration(sim):
    return sim.clock.now - 1000.0


@pytest.fixture(scope="module")
def timings():
    baseline = sweep(LIVE)
    before = sweep(OLD, dead=2)
    after = sweep(LIVE, dead=2)
    return baseline, before, after


def test_a_dead_board_costs_no_more_than_the_guard_times(timings):
    baseline, before, after = timings
    print(f"\nsimulated duration: no dead board {duration(baseline):.0f} s | dead board, liveness off (before) "
          f"{duration(before):.0f} s | dead board, liveness {LIVE:.0f} s (after) {duration(after):.0f} s")
    assert duration(before) > duration(baseline) + 1000                  # the problem was real
    assert duration(after) < duration(baseline) + 150                    # within a small margin of the healthy run
    assert duration(after) < duration(before) / 2
    # no round of the dead stretch closed by timeout once the board was recognised as silent
    recs = after.round_records(2) + after.round_records(3) + after.round_records(4)
    assert sum(1 for r in recs if r["close_reason"] == "timeout") <= 1
    assert "board_not_gated_silent" in after.names() and "board_dropped_from_gate" in after.names()


def test_phase_starts_do_not_wait_for_the_dead_board():
    """The board dies two rounds before a phase ends and is still ACTIVE at the next phase start
    (suspect_after_rounds is raised so only liveness can save the start)."""
    slow = dict(suspect_after_rounds=6, depart_after_rounds=8)
    baseline = sweep(LIVE, **slow)
    before = sweep(OLD, dead=2, dies_at=46, **slow)
    after = sweep(LIVE, dead=2, dies_at=46, **slow)
    start = lambda sim, phase: next(r for r in sim.out.records("phase_start")        # noqa: E731
                                    if r["phase_index"] == phase)["delivery_duration_s"]
    print(f"\nphase 4 start delivery: no dead board {start(baseline, 4):.0f} s | before {start(before, 4):.0f} s "
          f"| after {start(after, 4):.0f} s; whole run {duration(baseline):.0f} / {duration(before):.0f} / "
          f"{duration(after):.0f} s")
    assert start(before, 4) > start(baseline, 4) + 100                   # ready_wait + repair windows
    assert start(after, 4) <= start(baseline, 4) + 60
    assert duration(after) < duration(baseline) + 250 < duration(before)
    ng = [x for e, _, x in after.events if e == "board_not_gated_silent"]
    assert ng and all(x["board_id"] == 2 and x["silent_seconds"] > LIVE for x in ng)
    assert len({(x["board_id"], x["slot"]) for x in ng}) == len(ng)      # one event per (board, slot)

def test_the_round_records_name_the_boards_that_were_not_gated(timings):
    after = timings[2]
    flagged = [r for r in after.out.records("round") if r["not_gated_boards"]]
    assert flagged and all(r["not_gated_boards"] == [2] for r in flagged)
    assert all("not_gated_boards" in r for r in after.out.records("round"))
    # masks keep their meaning: the dead board's positions are false / false while it is silent
    masks = after.masks(3)
    col = masks["positions"].index(1)
    assert not any(row[col] for row in masks["uplink"]) and not any(row[col] for row in masks["downlink"])
    assert not any(r["arrived_positions"] and 1 in r["arrived_positions"] for r in after.round_records(3))


def test_the_returning_board_is_regated_and_its_machines_rejoin(timings):
    after = timings[2]
    c = after.controller
    assert c.membership.get(dev_eui_for(2)).state is State.ACTIVE
    late = after.round_records(6)
    assert all({1, 11, 21} <= set(r["participants"]) for r in late)
    assert all({1, 11, 21} <= set(r["arrived_positions"]) for r in late[2:])
    masks = after.masks(6)
    col = masks["positions"].index(1)
    assert masks["uplink"][-1][col] and masks["downlink"][-1][col]


def test_the_twin_comparison_still_passes_on_a_run_with_a_silent_board(tmp_path):
    ls = pytest.importorskip("v4_learning_sim")
    try:
        data, evaluator = ls.load_machine_data()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"research data unavailable: {exc}")
    out = RunOutputs(tmp_path)
    sim = sweep(LIVE, dead=2, sim_cls=ls.LearningSim, data=data, outputs=out)
    out.close()
    report = ls.cmp.compare(tmp_path, data, evaluator, allow_absent=True)      # its positions are absent in phases 3-4
    assert report["pass"], json.dumps({k: v["criteria"] for k, v in report["phases"].items()})
    gone = report["phases"]["3"]["membership"]["1"]
    assert gone["uplink_rounds"] == 0 and 1 in report["phases"]["3"]["absent_planned_positions"]
    assert report["phases"]["6"]["membership"]["1"]["uplink_rounds"] == 12
    assert sim.controller.counters["frames_rejected"] == 0


# ---------------------------------------------------------------------------
# Mechanics, one at a time
# ---------------------------------------------------------------------------
def started(n_boards=4, rounds=12, **kwargs):
    sim = Sim(elastic_plan(17070, 1, rounds), boards={b: (b - 1,) for b in range(1, n_boards + 1)},
              liveness_seconds=LIVE, round_timeout_seconds=600.0, **kwargs)
    sim.start()
    return sim, sim.controller


def test_any_valid_uplink_makes_a_board_live_again():
    sim, c = started()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    member = c.membership.get(dev_eui_for(3))
    member.last_heard_t -= 10 * LIVE
    assert not c._is_live(member)
    frames = {
        "ready": p4.encode_class_c_ready(sim.run_id, p4.NO_SLOT),
        "status": p4.encode_global_status(sim.run_id, 1, 0b111),
        "alarm": p4.encode_alarm(sim.run_id, 2, 0, 1, 1.0, 1.0),
        "self_check": p4.encode_self_check(2, 4, 0.1, 0.1),
        "replay_done": p4.encode_replay_done(sim.run_id, 2, 0, 5, 0),
        "update_fragment": p4.encode_frame(p4.MSG_UPDATE_FRAGMENT, 1, 0, 3, b"\x02\x00" + bytes(211)),
        "hello": p4.encode_board_hello(3, 1, 3, sim.run_id, (2,)),
    }
    for name, frame in frames.items():
        member.last_heard_t -= 10 * LIVE
        sim.boards[3].uplink(frame)
        assert c._is_live(member), name
    sim.boards[3].uplink(b"\x00")                                 # a frame that does not decode is not a sign of life
    member.last_heard_t -= 10 * LIVE
    sim.boards[3].uplink(b"\xf1\x04\x00")
    assert not c._is_live(member)


def test_a_required_board_that_goes_silent_during_a_downlink_is_dropped_and_re_added_when_it_speaks():
    sim, c = started()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    # force a fresh downlink in which board 3 is silent
    sim.boards[3].power_off()
    sim.run_until(lambda: c.stage is Stage.DOWNLINK and c._g.armed or c._g.ever_armed and c.stage is Stage.DOWNLINK,
                  max_steps=5000)
    g = c._g
    member = c.membership.get(dev_eui_for(3))
    assert member.dev_eui in g.required or member.dev_eui in g.dropped
    member.last_heard_t -= 1.5 * LIVE                              # silent, but not yet SUSPECT (2 x liveness)
    c.tick()
    assert member.dev_eui in g.dropped and member.dev_eui not in g.required
    assert member.dev_eui not in g.board_missing and "board_dropped_from_gate" in sim.names()
    sim.boards[3].powered = True
    sim.boards[3].ready(p4.NO_SLOT)                                # it speaks again
    assert member.dev_eui in g.required and member.dev_eui not in g.dropped
    assert g.board_missing[member.dev_eui] == 0b111 and "board_regated" in sim.names()


def test_silence_for_twice_the_liveness_makes_an_active_board_suspect():
    sim, c = started()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    member = c.membership.get(dev_eui_for(2))
    assert member.state is State.ACTIVE
    member.last_heard_t -= 2 * LIVE - 5                             # silent, but not yet beyond 2 x liveness
    c.tick()
    assert member.state is State.ACTIVE
    member.last_heard_t -= 10
    c.tick()
    assert member.state is State.SUSPECT
    event = [e for e in sim.out.membership_events if e["dev_eui"] == member.dev_eui][-1]
    assert event["reason"] == "silence" and event["to"] == "SUSPECT" and event["silent_seconds"] > 2 * LIVE


def test_a_board_that_goes_silent_and_comes_back_is_regated_and_rejoins():
    sim, c = started(rounds=14)
    closed = lambda: len(sim.out.records("round"))                  # noqa: E731
    sim.run_until(lambda: closed() >= 3)
    sim.boards[3].power_off()
    sim.run_until(lambda: "board_not_gated_silent" in sim.names(), max_steps=20000)
    mark = closed()
    t0 = sim.clock.now
    sim.run_until(lambda: closed() >= mark + 3)
    assert sim.clock.now - t0 < 400                                  # three rounds without waiting for it
    silent = [r for r in sim.out.records("round") if r["not_gated_boards"] == [3]]
    assert silent
    sim.boards[3].power_on(hello=False)                              # the radio is back, no reboot
    before = len([e for e in sim.names() if e == "board_not_gated_silent"])
    sim.run_until(lambda: c.stage in (Stage.COLLECTING, Stage.ENDING, Stage.DONE, Stage.FAILED))
    member = c.membership.get(dev_eui_for(3))
    assert member.state is State.ACTIVE and c._is_live(member)
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    recs = sim.round_records(0)
    assert 2 in recs[-1]["participants"] and 2 in recs[-1]["arrived_positions"]
    rejoin = next(r for r in recs if r["round"] > silent[-1]["round"] and 2 in r["arrived_positions"])
    # after its first uplink it is gated again: no new "not gated" events for later slots
    assert len([e for e in sim.names() if e == "board_not_gated_silent"]) <= before + 1
    assert rejoin["round"] <= silent[-1]["round"] + 4
    assert_matches_twin(sim, 0)


def test_a_silent_board_still_receives_the_multicast_and_may_deliver_late():
    """Not gated is not excluded: its UPDATE for the current round is accepted if it arrives."""
    sim, c = started(n_boards=3)
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    board = sim.boards[3]
    member = c.membership.get(dev_eui_for(3))
    member.last_heard_t -= 1.5 * LIVE                                # silent in the controller's eyes only
    sim.run_until(lambda: len(sim.out.records("round")) >= 6)
    last = sim.round_records(0)[-1]
    assert 2 in last["arrived_positions"]                            # it kept uploading and was aggregated
    assert board.held_slot > 0 and c._is_live(member)


# ---------------------------------------------------------------------------
# Resume with a silent board
# ---------------------------------------------------------------------------
def test_resume_treats_every_board_as_heard_then_stops_waiting_for_the_silent_one(tmp_path):
    out = RunOutputs(tmp_path)
    sim = Sim(elastic_plan(17070, 1, 12), boards={b: (b - 1,) for b in range(1, 5)}, outputs=out,
              liveness_seconds=LIVE, round_timeout_seconds=600.0, post_run_collect_seconds=200.0)
    sim.start()
    closed = lambda: len(sim.out.records("round"))                   # noqa: E731
    sim.run_until(lambda: closed() >= 3)
    sim.boards[3].power_off()
    sim.run_until(lambda: closed() >= 4)
    for _ in range(int(LIVE) + 30):
        sim.step()                                                   # the dead board is silent for > liveness
    sim.controller.checkpoint_now()
    ck = load_checkpoint(tmp_path)
    ages = {m["dev_eui"]: m["silent_seconds_at_checkpoint"] for m in ck["membership"]["members"]}
    assert ages[dev_eui_for(3)] > LIVE and ages[dev_eui_for(1)] < LIVE      # persisted for the operator
    out.close()

    out2 = RunOutputs(tmp_path)
    sim.out = out2
    c = sim.resume_controller(ck, outputs=out2)
    now = sim.clock.now
    assert all(m.last_heard_t == now and c._is_live(m, now) for m in c.membership.members.values())
    resumed_at = now
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=100000)
    assert c.stage is Stage.DONE
    names = sim.names()
    assert "board_dropped_from_gate" in names or "board_not_gated_silent" in names
    assert c.membership.get(dev_eui_for(3)).state in (State.SUSPECT, State.DEPARTED)
    # 9 more rounds without the dead board: nowhere near 9 x round_timeout
    assert sim.clock.now - resumed_at < 2000
    out2.close()
    rounds = [json.loads(s) for s in (tmp_path / "delivery_log.jsonl").read_text().splitlines()]
    rounds = [r for r in rounds if r["record"] == "round"]
    assert [r["round"] for r in rounds] == list(range(1, 13))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def config(**v4):
    return {"tts": {"application_id": "a"},
            "experiment": {"multicast_device_id": "g", "multicast_gateway_ids": ["gw"]}, "v4": dict(v4)}


def test_liveness_seconds_config():
    assert cv.parse_config(config(), cv.SERVER_DIR).params.liveness_seconds == 600.0
    assert ControllerParams().liveness_seconds == 600.0
    assert cv.parse_config(config(liveness_seconds=90), cv.SERVER_DIR).params.liveness_seconds == 90.0
    assert cv.parse_config(config(liveness_seconds=45.5), cv.SERVER_DIR).params.liveness_seconds == 45.5
    for bad in (0, -1, "600", True, None):
        if bad is None:
            continue
        with pytest.raises(cv.ConfigError, match="liveness_seconds"):
            cv.parse_config(config(liveness_seconds=bad), cv.SERVER_DIR)
    assert "liveness_seconds" in cv.V4_KEYS
