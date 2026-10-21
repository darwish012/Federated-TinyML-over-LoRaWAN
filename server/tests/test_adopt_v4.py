"""Rev 4.3 RUN_CONTROL ADOPT (PROTOCOL_V4 3.4): a board that completed an earlier run is silent in
Class C and only answers an ADOPT, with a HELLO; the server multicasts ADOPT while WAITING_START and
whenever a rostered but unbound DevEUI speaks (its HELLO was lost)."""
from __future__ import annotations

import protocol_v4 as p4
from controller_v4 import Stage
from membership_v4 import State
from plan_v4 import elastic_plan
from v4_ref import assert_matches_twin
from v4_sim import RUN_ID, Sim, dev_eui_for

OLD_RUN = 0x0D1DD00D
BOARDS3 = {1: (0,), 2: (1,), 3: (2,)}


def adopt_times(sim):
    return [t for command, t in sim.run_control_times if command == p4.CMD_ADOPT]


# ---------------------------------------------------------------------------
# WAITING_START: silent boards of an earlier run
# ---------------------------------------------------------------------------
def test_a_silent_board_of_a_previous_run_adopts_hellos_and_the_new_run_starts():
    sim = Sim(elastic_plan(17070, 1, 3), boards=BOARDS3, silent={3}, min_boards=3)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and not c.failed
    board = sim.boards[3]
    assert board.adoptions == 1 and sim.adopt_hellos[3] == 1 and board.stored_run_id == RUN_ID
    assert sim.run_control_times[0][0] == p4.CMD_ADOPT
    assert sim.run_control_times[0][1] - 1000.0 <= 3.0                    # as soon as the run exists
    assert c.membership.get(dev_eui_for(3)).hello_count == 1
    assert "experiment_started" in sim.names() and any(2 in r["participants"] for r in sim.round_records(0))   # it trained
    assert_matches_twin(sim, 0)


def test_old_boards_do_not_answer_an_adopt_so_the_run_never_starts():
    """rev 4.2 behaviour (no HELLO on ADOPT): the silent board is never bound and the run waits."""
    sim = Sim(elastic_plan(17070, 1, 3), boards=BOARDS3, silent={3}, min_boards=3, legacy_rev42=True)
    sim.start()
    for _ in range(700):
        sim.step()
    assert sim.controller.stage is Stage.WAITING_START
    assert sim.adopts_heard[3] >= 3 and sim.adopt_hellos[3] == 0


def test_every_board_of_the_previous_run_is_silent_and_a_new_run_adopts_them_all():
    """smoke test -> full sweep: a new server run with a new run_id finds three completed boards."""
    sim = Sim(elastic_plan(17070, 1, 3), boards=BOARDS3, silent={1, 2, 3}, min_boards=3,
              stored_runs={1: OLD_RUN, 2: OLD_RUN, 3: OLD_RUN}, run_id=RUN_ID + 1)
    sim.run_to_done()
    assert sim.controller.stage is Stage.DONE
    assert all(b.adoptions == 1 and b.stored_run_id == RUN_ID + 1 for b in sim.boards.values())
    assert_matches_twin(sim, 0)


def test_a_completed_board_ignores_an_adopt_of_its_own_run():
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,)}, min_boards=1)
    board = sim.boards[1]
    sim.start()
    board.ended = board.run_complete = True                     # it got END
    board.stored_run_id = RUN_ID
    for _ in range(100):
        sim.step()                                                # past its RX windows
    before = board.hello_count
    board.receive_run_control(p4.encode_run_control(RUN_ID, p4.CMD_ADOPT))
    assert board.hello_count == before and sim.adopt_hellos[1] == 0


def test_adopt_is_repeated_every_period_while_waiting_and_never_after_the_start():
    sim = Sim(elastic_plan(17070, 1, 3), boards=BOARDS3, silent={3}, min_boards=3,
              start_after_seconds=500.0, adopt_period_seconds=120.0)
    sim.boards[3].receive_run_control = lambda frame: None        # never hears it: stays silent
    sim.start()
    sim.run_until(lambda: sim.controller.stage is not Stage.WAITING_START, max_steps=1000)
    t_start = sim.clock.now
    sim.run_until(lambda: sim.controller.stage in (Stage.DONE, Stage.FAILED), max_steps=20000)
    times = sim.event_times("run_control_adopt_queued", why="waiting_start")
    assert len(times) >= 4
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(119.0 <= g <= 123.0 for g in gaps), gaps
    assert all(t <= t_start for t in times)                       # none once the run is running


def test_adopt_period_is_configurable():
    sim = Sim(elastic_plan(17070, 1, 3), boards=BOARDS3, silent={3}, min_boards=3,
              adopt_period_seconds=40.0)
    sim.boards[3].receive_run_control = lambda frame: None
    sim.start()
    for _ in range(300):
        sim.step()
    times = sim.event_times("run_control_adopt_queued", why="waiting_start")
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert len(gaps) >= 6 and all(39.0 <= g <= 43.0 for g in gaps), gaps


def test_adopt_downlinks_are_one_at_a_time_and_respect_the_unconfirmed_send():
    """The server never has two downlinks in flight (the gateway is half duplex and does the duty-cycle
    accounting); an ADOPT whose down/sent never arrives is retried only after the in-flight timeout."""
    sim = Sim(elastic_plan(17070, 1, 3), boards=BOARDS3, silent={3}, min_boards=3, legacy_rev42=True,
              adopt_period_seconds=30.0, sent_drop=lambda d, f, n: True, inflight_downlink_timeout_seconds=100.0)
    sim.start()
    for _ in range(450):
        sim.step()
    adopts = [p for p in sim.published if p4.decode_frame(p[1]).message_type == p4.MSG_RUN_CONTROL]
    assert 3 <= len(adopts) <= 5                                  # one per 100 s timeout, not one per 30 s period


# ---------------------------------------------------------------------------
# Any stage: an unbound DevEUI that speaks gets an ADOPT
# ---------------------------------------------------------------------------
def lose_first_hello(board_id=3):
    return lambda board, n: board == board_id and n == 1


def test_a_lost_hello_is_recovered_by_adopt_within_a_period_and_the_board_joins_the_next_round():
    sim = Sim(elastic_plan(17070, 1, 12), boards=BOARDS3, min_boards=2, hello_drop=lose_first_hello(),
              adopt_period_seconds=60.0, duty_cycle=True)
    c = sim.controller
    sim.start()
    member3 = lambda: c.membership.get(dev_eui_for(3))
    sim.run_until(lambda: member3() is not None and member3().board_id is not None, max_steps=600)
    assert sim.clock.now - 1000.0 < 60.0                          # recovered within one period ...
    assert sim.hellos_dropped == 1 and c.counters["adopt_for_unbound"] >= 1
    assert c.counters["ignored_device_without_hello"] >= 1
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    recs = sim.round_records(0)
    first = min(r["round"] for r in recs if 2 in r["participants"])
    assert first >= 1 and all(2 in r["participants"] for r in recs if r["round"] >= first)   # ... and joined
    assert member3().state is State.ACTIVE
    assert_matches_twin(sim, 0)


def test_without_hello_on_adopt_a_lost_hello_is_never_recovered():
    """Firmware that does not answer an ADOPT with a HELLO: the board stays unbound for the whole run."""
    sim = Sim(elastic_plan(17070, 1, 12), boards=BOARDS3, min_boards=2, hello_drop=lose_first_hello(),
              adopt_period_seconds=60.0, legacy_rev42=True, duty_cycle=True)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and sim.adopts_heard[3] >= 1 and sim.adopt_hellos[3] == 0
    member = c.membership.get(dev_eui_for(3))
    assert member is None or member.board_id is None
    assert all(3 - 1 not in r["participants"] for r in sim.round_records(0))


def test_a_lost_hello_is_recovered_in_the_middle_of_a_phase():
    sim = Sim(elastic_plan(17070, 1, 12), boards=BOARDS3, min_boards=2, adopt_period_seconds=60.0,
              duty_cycle=True)
    c = sim.controller
    sim.start(only={1, 2})
    sim.run_until(lambda: len(sim.out.records("round")) >= 3)
    sim.hello_drop = lose_first_hello()
    board = sim.boards[3]
    board.hello()                                                 # its HELLO is lost; READY follows
    assert c.membership.get(dev_eui_for(3)) is None or c.membership.get(dev_eui_for(3)).board_id is None
    sim.run_until(lambda: c.membership.get(dev_eui_for(3)) is not None
                  and c.membership.get(dev_eui_for(3)).board_id is not None, max_steps=600)
    assert sim.adopt_hellos[3] >= 1
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert any(2 in r["participants"] for r in sim.round_records(0))
    assert_matches_twin(sim, 0)


def test_unbound_adopts_are_rate_limited_to_one_per_period():
    sim = Sim(elastic_plan(17070, 1, 12), boards=BOARDS3, min_boards=2, adopt_period_seconds=60.0,
              hello_drop=lambda b, n: b == 3)                     # board 3 never gets bound
    c = sim.controller
    sim.start()
    for k in range(40):                                           # a READY every ~10 s for 400 s
        for _ in range(10):
            sim.step()
        sim.boards[3].ready(p4.NO_SLOT)
    times = adopt_times(sim)
    assert len(times) >= 3
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g >= 59.0 for g in gaps), gaps
    assert c.counters["adopt_for_unbound"] <= 400 // 60 + 1


def test_an_unrostered_device_never_triggers_an_adopt():
    allowed = {dev_eui_for(1), dev_eui_for(2)}
    sim = Sim(elastic_plan(17070, 1, 4), boards=BOARDS3, min_boards=2, is_rostered=lambda d: d in allowed,
              adopt_period_seconds=30.0)
    sim.start()
    for _ in range(200):
        sim.step()
        sim.boards[3].ready(p4.NO_SLOT)
    assert sim.controller.counters["adopt_for_unbound"] == 0
    assert sim.adopts_heard[3] == 0 or sim.controller.counters["adopt_commands"] == 0


# ---------------------------------------------------------------------------
# Duplicate HELLOs after an ADOPT are no-ops
# ---------------------------------------------------------------------------
def membership_events(sim):
    return [x for e, _, x in sim.events if e == "membership"]


def test_hellos_answering_periodic_adopts_in_waiting_start_change_nothing():
    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,), 3: (2,), 4: (3,)}, min_boards=4,
              adopt_period_seconds=60.0)
    c = sim.controller
    sim.start(only={1, 2, 3})
    for _ in range(400):
        sim.step()
    assert c.stage is Stage.WAITING_START
    assert len(adopt_times(sim)) >= 5
    assert sum(sim.adopt_hellos[b] for b in (1, 2, 3)) >= 15
    for b in (1, 2, 3):
        member = c.membership.get(dev_eui_for(b))
        assert member.state is State.BOUND and member.hello_count == 1   # never re-bound
        assert member.flash_run_id == RUN_ID                             # ... but it knows the run now
    assert sim.names().count("board_hello") == 3
    assert [x["reason"] for x in membership_events(sim)] == ["first_hello"] * 3
    assert c.counters["hello_repeats"] >= 12 and c.counters["hello_adopt_acks"] == 3
    noops = [x for e, _, x in sim.events if e == "hello_noop"]
    assert noops and all(x["level"] == "debug" for x in noops)
    assert "board_rebooted_mid_run" not in sim.names()


def test_hellos_after_an_adopt_in_every_phase_state_reset_nothing():
    sim = Sim(elastic_plan(17070, 1, 10), boards={b: (b - 1,) for b in range(1, 5)}, min_boards=4, duty_cycle=True)
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    stage_seen = set()
    hello_counts = {b: c.membership.get(dev_eui_for(b)).hello_count for b in sim.boards}
    n_membership = len(membership_events(sim))
    n_board_hello = sim.names().count("board_hello")
    for k in range(4):                                            # four ADOPTs, at different points
        c._queue_adopt("test")
        sim.run_until(lambda: len(adopt_times(sim)) > k)
        stage_seen.add(c.stage)
        for _ in range(60):
            sim.step()
    assert len(stage_seen) >= 1
    assert all(sim.adopt_hellos[b] >= 1 for b in sim.boards)      # boards answer (one in its RX window misses it)
    assert sum(sim.adopt_hellos.values()) >= 8
    assert {b: c.membership.get(dev_eui_for(b)).hello_count for b in sim.boards} == hello_counts
    assert sim.names().count("board_hello") == n_board_hello
    assert len(membership_events(sim)) == n_membership or all(
        x["reason"] != "first_hello" for x in membership_events(sim)[n_membership:])
    assert "board_rebooted_mid_run" not in sim.names()
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert c.stage is Stage.DONE
    assert c.membership.counts()["ACTIVE"] == 4
    assert_matches_twin(sim, 0)


def test_a_board_holding_the_run_from_a_stored_run_answers_an_adopt_with_an_identical_hello():
    sim = Sim(elastic_plan(17070, 1, 4), boards=BOARDS3, min_boards=3,
              stored_runs={1: RUN_ID, 2: RUN_ID, 3: RUN_ID}, duty_cycle=True)
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 1)
    for _ in range(40):                                           # boards mid-upload are deaf: keep asking
        c._queue_adopt("test")
        for _ in range(8):
            sim.step()
        if all(sim.adopt_hellos[b] >= 1 for b in sim.boards):
            break
    assert all(sim.adopt_hellos[b] >= 1 for b in sim.boards)
    sim.run_until(lambda: c.counters["hello_repeats"] >= 3)       # (queued behind the 1 % duty cycle)
    assert [c.membership.get(dev_eui_for(b)).hello_count for b in sim.boards] == [1, 1, 1]
    assert c.counters["hello_repeats"] >= 3 and c.counters["hello_adopt_acks"] == 0
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert c.stage is Stage.DONE


def test_a_reboot_hello_long_after_an_adopt_is_still_a_rebind():
    sim = Sim(elastic_plan(17070, 1, 8), boards={1: (0,), 2: (1,)}, min_boards=2, adopt_period_seconds=30.0)
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.out.records("round")) >= 2)
    board = sim.boards[2]
    member = c.membership.get(dev_eui_for(2))
    member.flash_run_id = 0                                       # the HELLO the server holds predates the adoption
    c._adopt_last_t = sim.clock.now - 500.0                       # no ADOPT for a long time
    board.uplink(p4.encode_board_hello(2, 1, board.flags, board.stored_run_id, board.positions))
    assert member.hello_count == 2 and "board_rebooted_mid_run" in sim.names()
