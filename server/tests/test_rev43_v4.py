"""Protocol rev 4.3 on the server: participation mask, REPLAY_DONE with first_alarm_index,
repeated HELLOs, the ENDING stage and prompt GLOBAL repair (PROTOCOL_V4 3.2-3.4, 4, 5)."""
from __future__ import annotations

import json
import struct

import protocol_v4 as p4
from controller_v4 import Stage
from membership_v4 import State
from plan_v4 import elastic_plan, planned_plan
from run_outputs_v4 import RunOutputs, load_checkpoint
from v4_ref import assert_matches_twin
from v4_sim import HELLO_OK, RUN_ID, Sim, dev_eui_for, device_id_for, thesis_boards

BOARDS4 = thesis_boards(4, 4)                       # board b hosts position b - 1
OLD_BEHAVIOUR = dict(legacy_rev42=True, downlink_guard_seconds=0.0, prompt_repair=False)


# ---------------------------------------------------------------------------
# a. participation mask
# ---------------------------------------------------------------------------
def test_participation_is_logged_for_every_global_and_every_phase():
    sim = Sim(planned_plan([2, 4], 17070, 3), boards=BOARDS4)
    sim.run_to_done()
    for row in sim.out.global_models:
        part = row["participation"]
        want = [0, 1] if row["phase_index"] == 0 else [0, 1, 2, 3]
        assert part == {"all": False, "mask_base": 0, "positions": want}
        assert row["global_bytes"] == 624 and len(row["codes"]) == 562       # codes/scales/crc unchanged
    for phase in sim.out.replay_masks["phases"]:
        assert phase["participation"]["positions"] == ([0, 1] if phase["phase_index"] == 0 else [0, 1, 2, 3])


def test_an_elastic_phase_sets_the_all_positions_bit():
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,), 2: (1,)})
    sim.run_to_done()
    assert all(r["participation"]["all"] for r in sim.out.global_models)
    assert sim.masks(0)["participation"]["all"] is True
    first = next(m for d, f, c in sim.published if d != "" for m in [p4.decode_frame(f)]
                 if m.message_type == p4.MSG_GLOBAL_FRAGMENT)
    assert first.fragment_count == 3


def test_a_planned_board_trains_and_uploads_only_the_positions_of_the_phase():
    """Boards host every position they can; the GLOBAL's mask says which of them take part."""
    boards = {1: (0, 3), 2: (1, 2)}
    sim = Sim(planned_plan([2, 4], 17070, 3), boards=boards)
    sim.run_to_done()
    c = sim.controller
    assert c.counters["updates_ignored_not_in_phase"] == 0
    first_phase = [r for r in sim.round_records(0)]
    assert all(r["expected_positions"] == [0, 1] and r["arrived_positions"] == [0, 1] for r in first_phase)
    # rev 4.2 behaviour: the same boards upload positions 2 and 3 in phase 0 too, and they are dropped
    old = Sim(planned_plan([2, 4], 17070, 3), boards=boards, legacy_rev42=True)
    old.run_to_done()
    assert old.controller.counters["updates_ignored_not_in_phase"] > 0


def test_resume_keeps_the_participation_of_every_phase(tmp_path):
    out = RunOutputs(tmp_path)
    sim = Sim(planned_plan([2, 4], 17070, 4), boards=BOARDS4, outputs=out)
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 5)
    out.close()
    checkpoint = load_checkpoint(tmp_path)
    out2 = RunOutputs(tmp_path)
    sim.out = out2
    c = sim.resume_controller(checkpoint, outputs=out2)
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=80000)
    out2.close()
    assert c.stage is Stage.DONE
    rows = [json.loads(s) for s in (tmp_path / "global_models.jsonl").read_text().splitlines()]
    assert sorted({r["slot"] for r in rows}) == list(range(1, 11))
    for row in rows:
        want = [0, 1] if row["phase_index"] == 0 else [0, 1, 2, 3]
        assert row["participation"]["positions"] == want and not row["participation"]["all"]
    masks = json.loads((tmp_path / "replay_masks.json").read_text())
    assert [p["participation"]["positions"] for p in masks["phases"]] == [[0, 1], [0, 1, 2, 3]]


# ---------------------------------------------------------------------------
# b. REPLAY_DONE carries first_alarm_index
# ---------------------------------------------------------------------------
def replay_sim(**kw):
    return Sim(elastic_plan(17070, 2, 2), boards={1: (0,), 2: (1,), 3: (2,)}, **kw)


def test_a_lost_alarm_is_recovered_from_replay_done():
    sim = replay_sim(alarm_for=lambda position, phase: position == 1 and phase == 0,
                     alarm_drop=lambda board, position, phase: True)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE
    rows = [a for a in sim.out.alarms if (a["position"], a["phase_index"]) == (1, 0)]
    assert len(rows) == 1
    assert rows[0]["source"] == "replay_done" and rows[0]["score"] == "" and rows[0]["threshold"] == ""
    assert rows[0]["acquisition_index"] == 1234 + 1
    done = next(d for d in sim.out.replay_dones if (d["position"], d["phase_index"]) == (1, 0))
    assert done["alarm_latched"] == 1 and done["first_alarm_index"] == 1235
    clean = next(d for d in sim.out.replay_dones if (d["position"], d["phase_index"]) == (0, 0))
    assert clean["alarm_latched"] == 0 and clean["first_alarm_index"] == p4.NO_ALARM_INDEX
    assert c.counters["alarms_from_replay_done"] == 1 and "protocol_violation" not in sim.names()
    assert sim.out.replay_summary["positions"]["1"]["status"] == "alarm"


def test_an_alarm_that_arrives_keeps_its_score_and_replay_done_adds_no_second_row():
    sim = replay_sim(alarm_for=lambda position, phase: position == 1 and phase == 0)
    sim.run_to_done()
    rows = [a for a in sim.out.alarms if (a["position"], a["phase_index"]) == (1, 0)]
    assert len(rows) == 1 and rows[0]["source"] == "alarm" and rows[0]["score"] != ""
    assert sim.controller.counters["alarms_from_replay_done"] == 0


def violations(sim):
    return [x for e, _, x in sim.events if e == "protocol_violation"]


def test_replay_done_that_disagrees_with_itself_or_with_the_alarm_is_a_protocol_violation():
    sim = replay_sim()
    sim.run_to_collecting()
    c = sim.controller
    dev = device_id_for(1)
    # latched = 1 without an index
    c.handle_uplink(dev, 10, p4.encode_replay_done(RUN_ID, 0, 0, 100, 1, p4.NO_ALARM_INDEX))
    # latched = 0 with an index
    c.handle_uplink(dev, 10, p4.encode_replay_done(RUN_ID, 0, 1, 100, 0, 77))
    kinds = [v["kind"] for v in violations(sim)]
    assert kinds.count("replay_done_latched_vs_index") == 2
    assert c.counters["protocol_violations"] >= 2
    # an ALARM whose index differs from the REPLAY_DONE's
    c.handle_uplink(device_id_for(2), 10, p4.encode_alarm(RUN_ID, 1, 0, 500, 9.0, 3.0))
    c.handle_uplink(device_id_for(2), 10, p4.encode_replay_done(RUN_ID, 1, 0, 100, 1, 501))
    assert "alarm_index_vs_replay_done" in [v["kind"] for v in violations(sim)] or \
        "alarm_index_differs" in [v["kind"] for v in violations(sim)]


def test_the_12_byte_replay_done_of_rev_4_2_is_rejected():
    sim = replay_sim()
    sim.run_to_collecting()
    c = sim.controller
    before = c.counters["frames_rejected"]
    body = struct.pack("<IHBIB", RUN_ID, 0, 0, 100, 0)
    c.handle_uplink(device_id_for(1), 10, p4.encode_frame(p4.MSG_REPLAY_DONE, p4.NO_SLOT, 0, 1, body))
    assert c.counters["frames_rejected"] == before + 1


# ---------------------------------------------------------------------------
# c. a repeated identical HELLO is liveness only
# ---------------------------------------------------------------------------
def identical_hello(board):
    return p4.encode_board_hello(board.board_id, 1, board.flags, board.stored_run_id, board.positions)


def test_a_repeated_hello_of_a_bound_board_changes_nothing():
    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,)}, min_boards=2)
    c = sim.controller
    sim.boards[1].hello()
    member = c.membership.get(dev_eui_for(1))
    assert member.state is State.BOUND and member.hello_count == 1
    ready_before = member.ready
    for _ in range(3):
        sim.boards[1].uplink(identical_hello(sim.boards[1]))
    assert member.state is State.BOUND and member.hello_count == 1 and member.ready == ready_before
    assert c.counters["hello_repeats"] == 3 and sim.names().count("board_hello") == 1
    assert [e for e, _, _ in sim.events].count("membership") == 1          # the first bind only


def test_a_repeated_hello_with_a_stored_run_changes_nothing_in_any_stage():
    sim = Sim(elastic_plan(17070, 1, 8), boards={1: (0,), 2: (1,), 3: (2,)}, min_boards=3,
              stored_runs={1: RUN_ID, 2: RUN_ID, 3: RUN_ID}, duty_cycle=True)
    c = sim.controller
    sim.start()
    stages = set()
    for k in range(60):
        for _ in range(25):
            sim.step()
        if c.stage in (Stage.DONE, Stage.FAILED):
            break
        stages.add(c.stage)
        for b in (1, 2, 3):
            sim.boards[b].uplink(identical_hello(sim.boards[b]))
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=80000)
    assert c.stage is Stage.DONE and len(stages) >= 2                        # several stages were exercised
    assert [c.membership.get(dev_eui_for(b)).hello_count for b in (1, 2, 3)] == [1, 1, 1]
    assert "board_rebooted_mid_run" not in sim.names() and c.membership.counts()["ACTIVE"] == 3
    assert c.counters["hello_repeats"] >= 6
    assert_matches_twin(sim, 0)


def test_a_repeated_hello_of_an_active_board_in_a_phase_keeps_it_active_and_ready():
    sim = Sim(elastic_plan(17070, 1, 10), boards={1: (0,), 2: (1,)}, stored_runs={1: RUN_ID, 2: RUN_ID},
              duty_cycle=True)
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    member = c.membership.get(dev_eui_for(2))
    assert member.state is State.ACTIVE
    sim.run_until(lambda: member.ready)
    sim.boards[2].uplink(identical_hello(sim.boards[2]))
    sim.step()
    sim.run_until(lambda: c.counters["hello_repeats"] >= 1)
    assert member.state is State.ACTIVE and member.hello_count == 1 and member.ready


def test_a_hello_with_any_other_content_is_a_rebind():
    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,)}, min_boards=2)
    c = sim.controller
    sim.start()
    member = c.membership.get(dev_eui_for(1))
    sim.boards[1].uplink(p4.encode_board_hello(1, 2, HELLO_OK, 0, (0,)))         # new firmware version
    assert member.hello_count == 2 and member.fw_version == 2


# ---------------------------------------------------------------------------
# d. the ENDING stage
# ---------------------------------------------------------------------------
def end_sim(**kw):
    return Sim(elastic_plan(17070, 1, 2), boards={1: (0,), 2: (1,), 3: (2,)}, duty_cycle=True, **kw)


def test_end_waits_for_the_guard_reaches_every_board_and_silence_ends_the_run():
    sim = end_sim()
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE
    assert c.counters["end_sends"] == 1 and len(sim.end_heard) == 3
    assert all(not talking for _, talking in sim.end_in_window)               # nobody was in its RX windows
    done = [x for e, _, x in sim.events if e == "run_done"][0]
    assert done["end_unconfirmed"] == [] and done["ending_seconds"] >= c.params.end_quiet_seconds
    assert c.summary()["end_unconfirmed"] == []
    end_t = [t for command, t in sim.run_control_times if command == p4.CMD_END][0]
    assert end_t >= max(b.last_uplink_t for b in sim.boards.values()) + c.params.downlink_guard_seconds - 1.0


def test_old_behaviour_multicasts_end_into_the_rx_windows_and_never_reaches_a_board():
    """downlink_guard_seconds = 0: END goes out the instant the last uplink is heard, and every
    re-send is triggered by an uplink, i.e. lands inside that board's windows again (testbed bug 3)."""
    sim = end_sim(downlink_guard_seconds=0.0, end_linger_seconds=900.0)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and sim.end_heard == []
    assert sim.missed_in_window["run_control"] >= 3
    assert c.end_unconfirmed == [1, 2, 3] and c.counters["end_sends"] >= 2


def test_a_board_that_missed_end_is_sent_another_after_its_next_uplink_rate_limited():
    sim = end_sim()
    board = sim.boards[2]
    real = board.receive_run_control
    state = {"dropped": 0}

    def lose_first_end(frame):
        rc = p4.decode_run_control(p4.decode_frame(frame))
        if rc.command == p4.CMD_END and state["dropped"] == 0:
            state["dropped"] = 1
            return
        real(frame)
    board.receive_run_control = lose_first_end
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and state["dropped"] == 1
    assert c.counters["end_sends"] == 2 and len(sim.end_heard) == 3 and c.end_unconfirmed == []
    ends = [t for command, t in sim.run_control_times if command == p4.CMD_END]
    assert ends[1] - ends[0] >= c.params.end_repeat_seconds


def test_end_is_resent_at_most_once_per_end_repeat_seconds_and_linger_lists_the_talkers():
    sim = end_sim(end_quiet_seconds=420.0, end_linger_seconds=700.0, end_repeat_seconds=60.0)
    for b in (1, 2):
        sim.boards[b].receive_run_control = (lambda real: lambda frame: (
            None if p4.decode_run_control(p4.decode_frame(frame)).command == p4.CMD_END else real(frame)))(
                sim.boards[b].receive_run_control)
    sim.start()
    sim.run_until(lambda: sim.controller.stage is Stage.ENDING, max_steps=60000)
    while sim.controller.stage is Stage.ENDING:
        for _ in range(10):
            sim.step()
        for b in (1, 2):
            sim.boards[b].uplink(p4.encode_class_c_ready(RUN_ID, p4.NO_SLOT))      # keep talking
        assert sim.clock.now < 1000.0 + 20000
    c = sim.controller
    assert c.stage is Stage.DONE and c.end_unconfirmed == [1, 2]
    ends = [t for command, t in sim.run_control_times if command == p4.CMD_END]
    assert len(ends) >= 3 and all(b - a >= 59.0 for a, b in zip(ends, ends[1:]))
    assert json.loads(json.dumps(c.summary()))["end_unconfirmed"] == [1, 2]
    assert [x for e, _, x in sim.events if e == "run_done"][0]["ending_seconds"] <= 710.0


def test_resume_in_ending_finishes_the_run(tmp_path):
    out = RunOutputs(tmp_path)
    sim = end_sim(outputs=out)
    sim.start()
    sim.run_until(lambda: sim.controller.stage is Stage.ENDING, max_steps=60000)
    for _ in range(20):
        sim.step()
    sim.controller.checkpoint_now()
    out.close()
    checkpoint = load_checkpoint(tmp_path)
    assert checkpoint["stage"] == "ending"
    out2 = RunOutputs(tmp_path)
    sim.out = out2
    c = sim.resume_controller(checkpoint, outputs=out2)
    assert c.stage is Stage.ENDING
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=80000)
    out2.close()
    assert c.stage is Stage.DONE and c.counters["end_sends"] >= 1
    assert json.loads((tmp_path / "summary.json").read_text()).get("end_unconfirmed", []) == [] \
        if (tmp_path / "summary.json").exists() else True


# ---------------------------------------------------------------------------
# e. radio guard and prompt GLOBAL repair
# ---------------------------------------------------------------------------
def test_the_first_multicast_of_a_round_never_reaches_a_board_inside_its_windows():
    sim = Sim(planned_plan([4], 17070, 6), boards=BOARDS4, duty_cycle=True)
    sim.run_to_done()
    assert sim.controller.stage is Stage.DONE and sim.window_hits == []


def waiting_for_board_2(sim):
    return any(e == "multicast_awaiting_reports" and x["outstanding"] == [2]
               and sim.controller._g is not None and sim.controller._g.slot == 2
               for e, _, x in sim.events)


def test_a_ready_that_awaits_the_current_global_triggers_a_repair_a_guard_later():
    """A board that missed the GLOBAL inside its RX windows announces CLASS_C_READY (awaiting that slot)
    when they close: the server re-multicasts `downlink_guard_seconds` after that uplink, not on a timer."""
    def lose_slot_2(board, slot, fragment, count):
        return board == 2 and slot == 2 and count == 1

    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,), 3: (2,)}, downlink_drop=lose_slot_2,
              multicast_report_window_seconds=300.0)
    c = sim.controller
    sim.start()
    sim.run_until(lambda: waiting_for_board_2(sim))
    n_events = len(sim.events)
    t_ready = sim.clock.now
    sim.boards[2].ready(2)
    sim.run_until(lambda: c.counters["global_prompt_repairs"] >= 1, max_steps=300)
    t_repair = sim.event_times("multicast_prompt_repair")[0]
    assert t_repair - t_ready <= c.params.downlink_guard_seconds + 3.0
    assert len([x for e, _, x in sim.events[n_events:] if e == "multicast_prompt_repair"]) == 1
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert c.stage is Stage.DONE and "multicast_repair" not in sim.names()      # the timer path never fired
    assert_matches_twin(sim, 0)


def test_a_status_with_missing_fragments_triggers_a_repair_without_the_report_window_timer():
    def lose_first_fragment(board, slot, fragment, count):
        return board == 2 and slot == 2 and fragment == 0 and count == 1

    kw = dict(downlink_drop=lose_first_fragment, multicast_report_window_seconds=300.0, idle_timeout=20.0)
    plan = elastic_plan(17070, 1, 4)
    new = Sim(plan, boards={1: (0,), 2: (1,), 3: (2,)}, **kw)
    new.run_to_done()
    old = Sim(plan, boards={1: (0,), 2: (1,), 3: (2,)}, prompt_repair=False, **kw)
    old.run_to_done()
    assert new.controller.stage is Stage.DONE and old.controller.stage is Stage.DONE
    repairs = [x for e, _, x in new.events if e == "multicast_prompt_repair"]
    assert repairs and repairs[0]["union_mask"] == "01" and repairs[0]["fragments"] == [0]
    assert old.controller.counters["global_prompt_repairs"] == 0
    assert new.t_collecting + 100.0 < old.t_collecting          # the old server sat out the 300 s window
    assert_matches_twin(new, 0)
