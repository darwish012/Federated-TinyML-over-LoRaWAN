"""HardwareExperimentController: a full level with fake boards, losses, repair and replay masks."""
from __future__ import annotations

import json

import numpy as np
import pytest

import protocol_v3 as p3
from controller_v3 import Phase
from protocol_v3 import (
    ALL_GLOBAL_FRAGMENTS,
    LAYERS,
    MSG_UPDATE_FRAGMENT,
    NO_ROUND,
    QuantisedModel,
    board_of_position,
    encode_alarm,
    encode_board_hello,
    encode_class_c_ready,
    encode_global_status,
    encode_self_check,
    encode_update,
    fragment_update,
    global_round_id,
    update_round_id,
    quantize_model,
    unflatten,
)
from run_outputs import RunOutputs
from study_plan import active_positions, load_fl_modules
from v3_sim import GROUP, RUN_ID, Sim, device_id_for


# ---------------------------------------------------------------------------
# Reference: what the twin would compute from the same deltas
# ---------------------------------------------------------------------------
def on_the_wire(model: QuantisedModel) -> QuantisedModel:
    """What the server actually decodes: the same codes with float32 scales."""
    return QuantisedModel(codes=model.codes, scales=model.wire_scales())


def f32_quantised(params) -> QuantisedModel:
    return quantize_model(tuple(np.asarray(p, dtype=np.float32) for p in params))


def reference_run(sim: Sim, level: int, uplink_mask=None):
    """Recompute every GLOBAL of a level with twin_server.Server (the research module)."""
    twin_server, fl = load_fl_modules()
    li = p3.level_index_of(level)
    positions = active_positions(level)
    server = twin_server.Server(fl.init(list(LAYERS), sim.controller.seed))
    globals_ = [f32_quantised(server.p)]
    for r in range(12):
        pairs = [
            (on_the_wire(sim.sent[(li, r, p)]).dequantize_tuple(), 1.0)
            for i, p in enumerate(positions)
            if uplink_mask is None or uplink_mask[r][i]
        ]
        server.round(pairs)
        globals_.append(f32_quantised(server.p))
    return server, globals_


def fl_server_arithmetic(fl, params, deltas_by_round):
    """fl.Server.round's update expression (fl.py) applied to ready-made deltas."""
    m = tuple(np.zeros_like(v) for v in params)
    v = tuple(np.zeros_like(x) for x in params)
    t = 0
    for deltas in deltas_by_round:
        if not deltas:
            continue
        old = params
        d = tuple(np.mean(np.stack([q[i] for q in deltas]), 0) for i in range(len(old)))
        t += 1
        m = tuple(fl.B1 * m[i] + (1 - fl.B1) * d[i] for i in range(len(old)))
        v = tuple(fl.B2 * v[i] + (1 - fl.B2) * d[i] * d[i] for i in range(len(old)))
        params = tuple(old[i] + fl.ETA * (m[i] / (1 - fl.B1 ** t)) /
                       (np.sqrt(v[i] / (1 - fl.B2 ** t)) + fl.TAU) for i in range(len(old)))
    return params


def logged_global_codes(sim: Sim, level: int) -> list:
    li = p3.level_index_of(level)
    rows = sorted((g for g in sim.out.global_models if g["level_index"] == li),
                  key=lambda g: g["round"])
    return [np.array(g["codes"], dtype=np.int8) for g in rows]


def round_records(sim: Sim, level: int) -> list:
    return [r for r in sim.out.records("round") if r["level"] == level]


# ---------------------------------------------------------------------------
# A full level, 3 positions per board, with lost UPDATEs and repair
# ---------------------------------------------------------------------------
def lossy_uplink(board, position, fragment, attempt, level_index, round_index):
    if round_index == 2 and position == 10 and fragment == 0 and attempt == 1:
        return True                                     # fragment 0 lost: 1 and 2 still arrive, each naming its position
    if round_index == 4 and position == 20 and fragment == 2 and attempt == 1:
        return True                                     # only the last fragment lost
    if round_index == 6 and position == 13 and fragment == 1 and attempt <= 2:
        return True                                     # the first repair loses it again
    if round_index == 8 and position == 5:
        return attempt == 1                             # the whole UPDATE lost
    return False


@pytest.fixture(scope="module")
def lossy_level28():
    sim = Sim(levels=(28,), uplink_drop=lossy_uplink)
    sim.run_sweep()
    return sim


def test_level_28_completes_with_repairs_and_three_positions_per_board(lossy_level28):
    sim = lossy_level28
    c = sim.controller
    assert c.phase is Phase.COLLECTING and not c.failed
    assert [len(p3_board) for p3_board in (
        [p for p in range(28) if board_of_position(p) == b] for b in range(1, 9))] == [3] * 8
    records = round_records(sim, 28)
    assert [r["round"] for r in records] == list(range(12))
    for record in records:
        assert record["update_arrived_positions"] == list(range(28))
        assert record["update_missing_positions"] == []
        assert record["close_reason"] == "all_delivered"
        assert record["participating_boards"] == list(range(1, 11))
        assert record["boards_holding_result_global"] == list(range(1, 11))


def test_repair_requests_name_the_board_position_and_missing_fragments(lossy_level28):
    sim = lossy_level28
    assert sim.repairs == [
        (1, 10, 0b001),      # fragment 0 lost; 1 and 2 were attributed by their own position byte
        (1, 20, 0b100),      # only fragment 2
        (4, 13, 0b010),      # fragment 1 ...
        (4, 13, 0b010),      # ... lost again on the resend
        (6, 5, 0b111),       # the whole UPDATE
    ]
    by_round = {r["round"]: r for r in round_records(sim, 28)}
    assert by_round[2]["update_repairs_by_position"] == {"10": 1}
    assert by_round[4]["update_repairs_by_position"] == {"20": 1}
    assert by_round[6]["update_repairs_by_position"] == {"13": 2}
    assert by_round[8]["update_repairs_by_position"] == {"5": 1}
    assert by_round[0]["update_repair_requests"] == 0
    # 28 machines x 3 fragments, plus exactly what was resent
    assert by_round[0]["update_fragments_received"] == 84
    assert by_round[2]["update_fragments_received"] == 84 - 1 + 1
    assert by_round[4]["update_fragments_received"] == 84 - 1 + 1
    assert by_round[6]["update_fragments_received"] == 84 - 1 + 1
    assert by_round[8]["update_fragments_received"] == 84 - 3 + 3


def test_repair_downlinks_carry_the_position_byte_on_the_wire(lossy_level28):
    frames = [f for d, f, _ in lossy_level28.published if d != GROUP]
    assert len(frames) == 5
    for frame in frames:
        decoded = p3.decode_frame(frame)
        assert decoded.message_type == p3.MSG_UPDATE_REPAIR and len(frame) == 10
    mask, position = p3.decode_repair_request(p3.decode_frame(frames[0]))
    assert (mask, position) == (0b001, 10)
    assert p3.decode_frame(frames[0]).round_id == update_round_id(6, 2) == 6 * 13 + 3


def test_aggregated_global_is_bit_identical_to_twin_server(lossy_level28):
    """Every UPDATE was eventually delivered, so the controller must equal the twin exactly."""
    sim = lossy_level28
    server, expected = reference_run(sim, 28)
    logged = logged_global_codes(sim, 28)
    assert len(logged) == 13
    for round_index, model in enumerate(expected):
        assert np.array_equal(logged[round_index], model.codes), f"GLOBAL round {round_index}"
    level = sim.controller.level_history[0]
    for ours, theirs in zip(level.server.p, server.p):
        assert np.array_equal(ours, theirs)                   # float64 master weights, bit for bit
    assert level.server.t == 12


def test_aggregated_global_is_bit_identical_to_fl_server_arithmetic(lossy_level28):
    sim = lossy_level28
    _, fl = load_fl_modules()
    li = p3.level_index_of(28)
    deltas = [
        [on_the_wire(sim.sent[(li, r, p)]).dequantize_tuple() for p in range(28)]
        for r in range(12)
    ]
    expected = fl_server_arithmetic(fl, fl.init(list(LAYERS), sim.controller.seed), deltas)
    for ours, theirs in zip(sim.controller.level_history[0].server.p, expected):
        assert np.array_equal(ours, theirs)


def test_initial_global_is_the_quantised_fl_init(lossy_level28):
    _, fl = load_fl_modules()
    first = logged_global_codes(lossy_level28, 28)[0]
    expected = f32_quantised(fl.init(list(LAYERS), 17070))
    assert np.array_equal(first, expected.codes)


def test_frame_header_round_ids_follow_level_index_times_twelve_plus_round(lossy_level28):
    headers = {p3.decode_frame(f).round_id for d, f, _ in lossy_level28.published if d == GROUP}
    assert headers == {global_round_id(6, r) for r in range(13)} == set(range(78, 91))


def test_replay_masks_document(lossy_level28):
    masks = lossy_level28.out.replay_masks
    assert masks["schema"] == "tinyfl-replay-masks-v2" and masks["run_id"] == RUN_ID
    level = masks["levels"]["28"]
    assert level["level_index"] == 6 and level["n_machines"] == 28
    assert level["rounds_completed"] == 12 and level["complete"] is True
    assert level["positions"] == list(range(28))
    assert len(level["uplink"]) == 12 and len(level["downlink"]) == 12
    assert all(len(row) == 28 and all(isinstance(v, bool) for v in row)
               for row in level["uplink"] + level["downlink"])
    assert all(all(row) for row in level["uplink"]) and all(all(row) for row in level["downlink"])
    assert len(level["machines"]) == 28 and len(set(level["machines"])) == 28
    json.dumps(masks)                                              # serialisable as-is


def test_delivery_log_records(lossy_level28):
    sim = lossy_level28
    starts, rounds, ends = (sim.out.records(k) for k in ("level_start", "round", "level_end"))
    assert len(starts) == 1 and len(rounds) == 12 and len(ends) == 1
    assert starts[0]["boards_holding_initial_global"] == list(range(1, 11))
    r = rounds[2]
    for key in ("level", "level_index", "round", "active_positions", "update_arrived_positions",
                "boards_holding_result_global", "boards_holding_seed_global", "timings",
                "update_fragments_received", "update_repair_requests", "global_fragments_sent",
                "global_bursts", "close_reason", "train_loss"):
        assert key in r, key
    assert r["timings"]["upload_duration_s"] >= 0 and r["timings"]["downlink_duration_s"] >= 0
    assert r["global_fragments_sent"] == 3                          # one clean burst of 3
    assert ends[0]["status"] == "complete" and ends[0]["rounds_completed"] == 12
    assert ends[0]["updates_delivered_per_round"] == [28] * 12
    assert ends[0]["final_global_crc32"] == f"{p3.decode_global(sim.controller._g.payload).crc32:08X}"


def test_final_model_is_the_round_12_global(lossy_level28):
    sim = lossy_level28
    final = logged_global_codes(sim, 28)[12]
    for board in sim.boards.values():
        model = board.held_models[(6, 12)]
        assert np.array_equal(model.model.codes, final)
        assert model.round == 12


# ---------------------------------------------------------------------------
# Losses that the repair flow cannot fix
# ---------------------------------------------------------------------------
def test_abandoned_position_is_logged_missing_and_the_round_still_closes():
    def drop(board, position, fragment, attempt, level_index, round_index):
        return round_index == 3 and position == 1            # never gets through

    sim = Sim(levels=(4,), boards=range(1, 5), uplink_drop=drop)
    sim.run_sweep()
    records = round_records(sim, 4)
    assert records[3]["update_arrived_positions"] == [0, 2, 3]
    assert records[3]["update_missing_positions"] == [1]
    assert records[3]["update_abandoned_positions"] == [1]
    assert records[3]["close_reason"] == "all_resolved"
    assert len(sim.repairs) == sim.controller.max_repair_rounds
    masks = sim.out.replay_masks["levels"]["4"]
    assert masks["uplink"][3] == [True, False, True, True]
    assert all(all(row) for i, row in enumerate(masks["uplink"]) if i != 3)
    # The global is exactly what the twin computes from the deltas that did arrive.
    _, expected = reference_run(sim, 4, masks["uplink"])
    for round_index, model in enumerate(expected):
        assert np.array_equal(logged_global_codes(sim, 4)[round_index], model.codes)


def test_a_silent_board_closes_the_round_by_timeout():
    def crash(board, level_index, round_index):
        return board == 3 and round_index == 5              # holds the global, then says nothing

    sim = Sim(levels=(4,), boards=range(1, 5), crash=crash, round_timeout_seconds=100.0,
              ready_wait_seconds=30.0)
    sim.run_sweep()
    records = round_records(sim, 4)
    assert records[5]["close_reason"] == "timeout"
    assert records[5]["update_missing_positions"] == [2]
    assert records[5]["timings"]["upload_duration_s"] >= 100
    assert "boards_not_ready" in sim.names()
    assert sim.controller.phase is Phase.COLLECTING
    assert records[6]["update_arrived_positions"] == [0, 1, 2, 3]      # it recovers next round
    masks = sim.out.replay_masks["levels"]["4"]["uplink"]
    assert masks[5] == [True, True, False, True]


def test_an_empty_round_leaves_the_server_unchanged_and_fedadam_t_does_not_advance():
    def crash(board, level_index, round_index):
        return round_index == 3                              # nobody uploads round 3

    sim = Sim(levels=(4,), boards=range(1, 5), crash=crash, round_timeout_seconds=50.0,
              ready_wait_seconds=20.0)
    sim.run_sweep()
    records = round_records(sim, 4)
    assert records[3]["update_arrived_positions"] == []
    codes = logged_global_codes(sim, 4)
    assert np.array_equal(codes[3], codes[4])                # GLOBAL after round 3 == before it
    assert not np.array_equal(codes[4], codes[5])
    assert sim.controller.level_history[0].server.t == 11
    _, expected = reference_run(sim, 4, sim.out.replay_masks["levels"]["4"]["uplink"])
    for round_index, model in enumerate(expected):
        assert np.array_equal(codes[round_index], model.codes)


# ---------------------------------------------------------------------------
# Downlink (GLOBAL multicast) loss and repair
# ---------------------------------------------------------------------------
def test_global_repair_resends_only_the_union_of_what_boards_reported_missing():
    header = global_round_id(0, 2)

    def drop(board, header_id, fragment, count):
        return board == 2 and header_id == header and fragment == 1 and count == 1

    sim = Sim(levels=(4,), boards=range(1, 5), downlink_drop=drop,
              multicast_report_window_seconds=300.0, idle_timeout=60.0)
    sim.run_sweep()
    assert sim.global_transmissions[(header, 1)] == 2         # fragment 1 sent twice
    assert sim.global_transmissions[(header, 0)] == 1 and sim.global_transmissions[(header, 2)] == 1
    record = round_records(sim, 4)[1]                        # the round that produced GLOBAL 2
    assert record["global_bursts"] == 2 and record["global_fragments_sent"] == 4
    assert record["boards_holding_result_global"] == [1, 2, 3, 4]
    assert "multicast_repair" in sim.names()


def test_a_silent_board_gets_the_whole_global_again_after_the_report_window():
    header = global_round_id(0, 4)

    def drop(board, header_id, fragment, count):
        return board == 3 and header_id == header and count == 1

    sim = Sim(levels=(4,), boards=range(1, 5), downlink_drop=drop,
              multicast_report_window_seconds=20.0, idle_timeout=500.0)
    sim.run_sweep()
    assert [sim.global_transmissions[(header, i)] for i in range(3)] == [2, 2, 2]
    assert round_records(sim, 4)[3]["boards_holding_result_global"] == [1, 2, 3, 4]


def test_a_board_that_never_gets_a_global_is_a_downlink_hole_in_the_mask():
    header = global_round_id(0, 4)

    def drop(board, header_id, fragment, count):
        return board == 2 and header_id == header            # never receives GLOBAL round 4

    sim = Sim(levels=(4,), boards=range(1, 5), downlink_drop=drop,
              multicast_report_window_seconds=20.0, idle_timeout=500.0,
              round_timeout_seconds=100.0, ready_wait_seconds=20.0)
    sim.run_sweep()
    assert sim.controller.phase is Phase.COLLECTING
    masks = sim.out.replay_masks["levels"]["4"]
    # GLOBAL 4 is the result of round 3: board 2 (position 1) did not get it.
    assert masks["downlink"][3] == [True, False, True, True]
    assert all(all(row) for i, row in enumerate(masks["downlink"]) if i != 3)
    # It could not train round 4 (no seed global), so its uplink is missing that round.
    assert masks["uplink"][4] == [True, False, True, True]
    records = round_records(sim, 4)
    assert records[3]["boards_holding_result_global"] == [1, 3, 4]
    assert records[4]["boards_holding_seed_global"] == [1, 3, 4]
    assert records[4]["close_reason"] == "timeout"
    assert sim.controller.level_history[0].status == "complete"
    _, expected = reference_run(sim, 4, masks["uplink"])
    for round_index, model in enumerate(expected):
        assert np.array_equal(logged_global_codes(sim, 4)[round_index], model.codes)


def test_a_board_that_never_gets_the_initial_global_fails_the_level():
    def drop(board, header_id, fragment, count):
        return board == 2 and header_id == global_round_id(0, 0)

    sim = Sim(levels=(4,), boards=range(1, 5), downlink_drop=drop,
              multicast_report_window_seconds=10.0, idle_timeout=500.0)
    sim.run_sweep()
    c = sim.controller
    assert c.failed and "never received GLOBAL" in c.failure_reason
    assert "[2]" in c.failure_reason
    assert sim.out.level_summaries[0]["status"] == "failed"


def test_deterministic_downlink_rejection_stops_the_run_with_the_reason():
    sim = Sim(levels=(4,), boards=range(1, 5))
    sim.start()
    while not sim.published:                    # a fragment is now in flight, unconfirmed
        sim.controller.tick()
        sim.clock.now += 1
    reason = json.dumps({"error": {"name": "application_downlink_too_long",
                                   "attributes": {"length": 221, "max": "51"}}})
    sim.controller.handle_downlink_failed(GROUP, reason)
    assert sim.controller.failed
    assert "application_downlink_too_long" in sim.controller.failure_reason
    assert "rx2_data_rate_index" in sim.controller.failure_reason


def test_an_unconfirmed_downlink_is_retried_after_the_inflight_timeout():
    sim = Sim(levels=(4,), boards=range(1, 5), inflight_downlink_timeout_seconds=20.0)
    sim.start()
    c = sim.controller
    while not sim.published:
        c.tick()
        sim.clock.now += 1
    assert len(sim.published) == 1
    first = p3.decode_frame(sim.published[0][1])
    sim.clock.now += 5
    c.tick()
    assert len(sim.published) == 1                              # slot still busy
    sim.clock.now += 20
    c.tick()
    c.tick()
    assert "downlink_status_timeout" in sim.names()
    sim.clock.now += 5
    c.tick()
    assert len(sim.published) == 2
    assert p3.decode_frame(sim.published[1][1]).fragment_index == first.fragment_index


# ---------------------------------------------------------------------------
# Several levels
# ---------------------------------------------------------------------------
def test_two_levels_use_canonical_level_indices_and_a_fresh_model():
    sim = Sim(levels=(4, 12), boards=range(1, 11))
    sim.run_sweep()
    assert sim.controller.phase is Phase.COLLECTING
    assert [lv.level for lv in sim.controller.level_history] == [4, 12]
    headers = sorted({p3.decode_frame(f).round_id for d, f, _ in sim.published if d == GROUP})
    # level 4 is canonical level_index 0 (slots 0..12), level 12 is index 2 (26..38),
    # NOT index 1: the index is the position in HARDWARE_LEVELS, not in the sweep.
    assert headers == list(range(0, 13)) + list(range(26, 39))
    # level 12 restarts from the same fl.init: its GLOBAL 0 equals level 4's
    first_4, first_12 = logged_global_codes(sim, 4)[0], logged_global_codes(sim, 12)[0]
    assert np.array_equal(first_4, first_12)
    for level in (4, 12):
        _, expected = reference_run(sim, level)
        for round_index, model in enumerate(expected):
            assert np.array_equal(logged_global_codes(sim, level)[round_index], model.codes)
    summaries = sim.out.level_summaries
    assert [s["level"] for s in summaries] == [4, 12]
    assert all(s["status"] == "complete" for s in summaries)


# ---------------------------------------------------------------------------
# Starting the run: BOARD_HELLO
# ---------------------------------------------------------------------------
def test_the_run_starts_when_the_expected_number_of_boards_have_said_hello():
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=4)
    boards = list(sim.boards.values())
    for board in boards[:3]:
        board.hello()
        board.ready()
    sim.controller.tick()
    assert sim.controller.phase is Phase.WAITING_BOARDS
    assert "experiment_started" not in sim.names()
    boards[3].hello()
    assert sim.controller.phase is Phase.DOWNLINK
    assert "experiment_started" in sim.names()
    assert sim.controller.counted_boards() == [1, 2, 3, 4]


def test_a_board_without_an_sd_card_is_not_counted():
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=4)
    boards = list(sim.boards.values())
    boards[0].hello(flags=0)                                 # SD not mounted, no IMU
    for board in boards[1:]:
        board.hello()
    assert sim.controller.phase is Phase.WAITING_BOARDS
    assert "board_not_counted" in sim.names()
    boards[0].hello()                                        # reboots with the card seated
    assert sim.controller.phase is Phase.DOWNLINK


def test_two_cards_claiming_the_same_board_number_are_flagged():
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=4)
    sim.boards[1].hello()
    impostor = p3.encode_board_hello(1, 3, 1, 0b011)
    sim.controller.handle_uplink("eui-70b3d57ed0099999", 10, impostor)
    assert "board_number_conflict" in sim.names()
    assert sim.controller.counted_boards() == [1]


def test_a_level_that_needs_a_board_nobody_announced_fails_at_start():
    sim = Sim(levels=(12,), boards=range(1, 10), expected_boards=9)      # board 10 absent
    for board in sim.boards.values():
        board.hello()
    assert sim.controller.failed and "missing [10]" in sim.controller.failure_reason


def test_uplink_from_a_device_that_never_said_hello_is_ignored():
    sim = Sim(levels=(4,), boards=range(1, 5))
    sim.controller.handle_uplink("eui-0000000000000001", 10, encode_class_c_ready(0, 5, 5))
    assert "ignored_device_without_hello" in sim.names()
    assert not sim.controller.failed


def test_malformed_frames_are_rejected_without_failing_the_run():
    sim = Sim(levels=(4,), boards=range(1, 5))
    sim.start()
    sim.controller.handle_uplink(device_id_for(1), 10, b"\xf1\x02\x02\x00\x00\x03\x02\x05\x05")   # v2
    sim.controller.handle_uplink(device_id_for(1), 10, b"garbage")
    sim.controller.handle_uplink(device_id_for(1), 11, encode_class_c_ready(0, 5, 5))
    assert sim.names().count("frame_rejected") == 2
    assert "ignored_wrong_fport" in sim.names()
    assert not sim.controller.failed
    assert sim.controller.summary()["counters"]["frames_rejected"] == 2


# ---------------------------------------------------------------------------
# UPDATE validation
# ---------------------------------------------------------------------------
def test_position_on_another_board_or_inactive_is_rejected():
    sim = Sim(levels=(4,), boards=range(1, 5), crash=lambda b, l, r: r == 0)
    sim.start()
    sim.run_until(lambda: sim.controller.phase is Phase.UPLOAD)
    c = sim.controller
    delta = quantize_model(np.zeros(562) + 0.01)
    header = update_round_id(0, 0)
    # board 1 claims position 1 (which belongs to board 2)
    frames = fragment_update(encode_update(1, 0, 0, 10, delta, 0.0, run_id=RUN_ID), header)
    c.handle_uplink(device_id_for(1), 10, frames[0])
    assert "update_position_rejected" in sim.names()
    # board 2 claims position 5, which is not active at level 4 (and belongs to board 6)
    frames = fragment_update(encode_update(5, 0, 0, 10, delta, 0.0, run_id=RUN_ID), header)
    c.handle_uplink(device_id_for(2), 10, frames[0])
    assert sim.names().count("update_position_rejected") == 2
    assert c._u.updates == {} and c._u.partials == {}


def test_wrong_round_id_and_late_updates_are_dropped():
    sim = Sim(levels=(4,), boards=range(1, 5), crash=lambda b, l, r: r == 0)
    sim.start()
    sim.run_until(lambda: sim.controller.phase is Phase.UPLOAD)
    c = sim.controller
    delta = quantize_model(np.zeros(562) + 0.01)
    stale = fragment_update(encode_update(0, 0, 0, 10, delta, 0.0, run_id=RUN_ID), update_round_id(0, 3))
    c.handle_uplink(device_id_for(1), 10, stale[0])
    assert "stale_update_fragment" in sim.names()
    assert c._u.updates == {}


def test_a_corrupted_update_fails_its_crc_and_is_repaired():
    sim = Sim(levels=(4,), boards=range(1, 5), crash=lambda b, l, r: r == 0 and b == 1)
    sim.start()
    sim.run_until(lambda: sim.controller.phase is Phase.UPLOAD)
    c = sim.controller
    header = update_round_id(0, 0)
    delta = quantize_model(np.linspace(-0.1, 0.1, 562))
    frames = fragment_update(encode_update(0, 0, 0, 10, delta, 0.0, run_id=RUN_ID), header)
    corrupted = bytearray(frames[1])
    corrupted[20] ^= 0xFF
    dev = device_id_for(1)
    c.handle_uplink(dev, 10, frames[0])
    c.handle_uplink(dev, 10, bytes(corrupted))
    c.handle_uplink(dev, 10, frames[2])
    assert "update_rejected" in sim.names()
    assert 0 not in c._u.updates and 0 not in c._u.partials
    c.handle_uplink(dev, 10, encode_class_c_ready(header, 5, 5))
    repair = [e for e in sim.events if e[0] == "update_repair_requested"][-1][2]
    assert repair["position"] == 0 and repair["missing_mask"] == "0007"


def _board1_at_level12_upload(**kwargs):
    """Level 12: board 1 hosts positions 0 and 10. It is silenced so the test can be the board."""
    sim = Sim(levels=(12,), boards=range(1, 11), crash=lambda b, l, r: r == 0 and b == 1, **kwargs)
    sim.start()
    sim.run_until(lambda: sim.controller.phase is Phase.UPLOAD)
    return sim, sim.controller


def _update_for(position, level_index, round_index, seed):
    delta = quantize_model(np.random.default_rng(seed).normal(0, 0.05, 562))
    payload = encode_update(position, level_index, round_index, 100, delta, 0.5, run_id=RUN_ID)
    return delta, payload


def test_fragments_of_two_positions_interleaved_on_one_board_reassemble_correctly():
    """Rev 3.1: every fragment names its position, so order and interleaving do not matter."""
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    delta_a, payload_a = _update_for(0, 2, 0, 1)
    delta_b, payload_b = _update_for(10, 2, 0, 2)
    a, b = fragment_update(payload_a, header), fragment_update(payload_b, header)
    # every fragment body starts with its own position byte
    assert [f[7] for f in a] == [0, 0, 0] and [f[7] for f in b] == [10, 10, 10]
    dev = device_id_for(1)
    for frame in (a[0], b[0], a[1], b[1], a[2], b[2]):        # strictly interleaved
        c.handle_uplink(dev, 10, frame)
    assert {0, 10} <= set(c._u.updates)
    assert c._u.partials == {}
    for position, delta in ((0, delta_a), (10, delta_b)):
        expected = QuantisedModel(delta.codes, delta.wire_scales()).dequantize_tuple()
        assert all(np.array_equal(x, y) for x, y in zip(c._u.updates[position].deq, expected))
    assert "update_rejected" not in sim.names()


def test_out_of_order_interleaved_and_duplicated_fragments_reassemble():
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    _, payload_a = _update_for(0, 2, 0, 3)
    _, payload_b = _update_for(10, 2, 0, 4)
    a, b = fragment_update(payload_a, header), fragment_update(payload_b, header)
    dev = device_id_for(1)
    for frame in (b[2], a[1], a[1], b[0], a[2], b[1], a[0], b[1]):
        c.handle_uplink(dev, 10, frame)
    assert {0, 10} <= set(c._u.updates)
    assert "duplicate_update_fragment" in sim.names()


def test_a_lost_fragment_zero_no_longer_orphans_the_others():
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    _, payload_a = _update_for(0, 2, 0, 5)
    a = fragment_update(payload_a, header)
    dev = device_id_for(1)
    c.handle_uplink(dev, 10, a[1])
    c.handle_uplink(dev, 10, a[2])
    assert sorted(c._u.partials[0].parts) == [1, 2]           # attributed without fragment 0
    c.handle_uplink(dev, 10, encode_class_c_ready(header, 5, 5))
    repair = [e for e in sim.events if e[0] == "update_repair_requested"]
    by_position = {e[2]["position"]: e[2]["missing_mask"] for e in repair}
    assert by_position[0] == "0001"                           # only fragment 0
    c.handle_uplink(dev, 10, a[0])
    assert 0 in c._u.updates


def test_the_fragment_position_byte_must_match_the_position_inside_the_update():
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    _, payload_b = _update_for(10, 2, 0, 6)                   # the UPDATE says position 10 ...
    frames = [p3.encode_frame(MSG_UPDATE_FRAGMENT, header, i, 3,
                              bytes((0,)) + payload_b[i * 213:(i + 1) * 213])   # ... fragments say 0
              for i in range(3)]
    dev = device_id_for(1)
    for frame in frames:
        c.handle_uplink(dev, 10, frame)
    assert not {0, 10} & set(c._u.updates) and 0 not in c._u.partials
    assert c.counters["update_position_mismatches"] == 1
    rejected = [e for e in sim.events if e[0] == "update_rejected"]
    assert "fragments carry position 0" in rejected[-1][2]["error"]


def test_a_fragment_whose_position_belongs_to_another_board_is_rejected():
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    _, payload = _update_for(1, 2, 0, 7)                      # position 1 is board 2's
    for frame in fragment_update(payload, header):
        c.handle_uplink(device_id_for(1), 10, frame)
    assert sim.names().count("update_position_rejected") == 3
    assert c._u.partials == {}


def test_an_update_fragment_body_is_the_position_byte_plus_213_payload_bytes():
    _, payload = _update_for(5, 0, 0, 8)
    frames = fragment_update(payload, update_round_id(0, 0))
    assert [len(f) for f in frames] == [221, 221, 176]
    assert b"".join(f[8:] for f in frames) == payload
    assert all(f[7] == 5 for f in frames)
    with pytest.raises(p3.ProtocolError):
        p3.split_update_fragment(p3.Frame(1, 0, 0, 3, b""))


# ---------------------------------------------------------------------------
# ALARM and SELF_CHECK
# ---------------------------------------------------------------------------
def test_alarm_and_self_check_are_recorded_and_misplaced_positions_are_rejected():
    sim = Sim(levels=(4,), boards=range(1, 5))
    sim.run_sweep()
    c = sim.controller
    dev1 = device_id_for(1)
    c.handle_uplink(dev1, 10, encode_alarm(0, 0, 4242, 3.5, 2.0))
    c.handle_uplink(dev1, 10, encode_alarm(0, 0, 4242, 3.5, 2.0))       # duplicate
    c.handle_uplink(dev1, 10, encode_alarm(0, 0, 9999, 8.0, 2.0))       # same (position, level): ignored
    c.handle_uplink(dev1, 10, encode_alarm(1, 0, 7, 9.0, 2.0))           # position 1 is board 2's: rejected
    c.handle_uplink(dev1, 10, encode_alarm(0, 1, 5, 4.0, 2.0))           # another level: kept
    c.handle_uplink(dev1, 10, encode_self_check(0, 800, 0.5, 0.125))
    c.handle_uplink(dev1, 10, encode_self_check(1, 800, 0.5, 0.125))     # rejected
    assert [(a["position"], a["level_index"]) for a in sim.out.alarms] == [(0, 0), (0, 1)]
    assert sim.out.alarms[0] == {**sim.out.alarms[0], "board": 1, "level": 4,
                                 "acquisition_index": 4242, "score": 3.5, "threshold": 2.0}
    assert sim.names().count("duplicate_alarm") == 2
    assert sim.names().count("alarm_rejected") == 1 and sim.names().count("self_check_rejected") == 1
    assert [s["position"] for s in sim.out.self_checks] == [0]
    c.finalize()
    assert sim.out.level_summaries[0]["alarms"] == 1        # level_index 0; the other is level_index 1


def test_replay_done_separates_no_alarm_from_never_finished():
    sim = Sim(levels=(4, 8), boards=range(1, 9))
    sim.run_sweep()
    c = sim.controller
    dev = device_id_for
    # position 0 (board 1): both levels scored, one alarm latched and reported
    c.handle_uplink(dev(1), 10, encode_alarm(0, 1, 11, 3.0, 2.0))
    c.handle_uplink(dev(1), 10, p3.encode_replay_done(0, 0b11, 5000, 1))
    # position 1 (board 2): both levels, nothing latched: a genuine "no alarm"
    c.handle_uplink(dev(2), 10, p3.encode_replay_done(1, 0b11, 5000, 0))
    # position 2 (board 3): REPLAY_DONE claims only level index 1 was scored
    c.handle_uplink(dev(3), 10, p3.encode_replay_done(2, 0b10, 2500, 0))
    # position 3 (board 4): latched 2 alarms but only one ALARM message arrived
    c.handle_uplink(dev(4), 10, encode_alarm(3, 0, 1, 3.0, 2.0))
    c.handle_uplink(dev(4), 10, p3.encode_replay_done(3, 0b11, 5000, 2))
    # position 4 (board 5): only in level 8
    c.handle_uplink(dev(5), 10, p3.encode_replay_done(4, 0b10, 2500, 0))
    # positions 5, 6, 7: nothing at all
    c.handle_uplink(dev(1), 10, p3.encode_replay_done(11, 0b11, 1, 0))      # position 11 is board 2's: rejected
    c.finalize()

    summary = sim.out.replay_summary
    assert summary["positions_expected"] == 8 and summary["levels_completed"] == [4, 8]
    status = {int(k): v["status"] for k, v in summary["positions"].items()}
    assert status == {0: "alarm", 1: "no_alarm", 2: "incomplete_levels", 3: "alarm", 4: "no_alarm",
                      5: "never_finished", 6: "never_finished", 7: "never_finished"}
    assert summary["positions"]["2"]["levels_missing"] == [0]
    assert summary["positions"]["4"]["levels_expected"] == [1]      # position 4 only exists at level 8
    assert summary["positions"]["3"]["alarm_messages_lost"] is True
    assert summary["alarm_messages_lost_positions"] == [3]
    assert summary["counts"] == {"alarm": 2, "incomplete_levels": 1, "never_finished": 3, "no_alarm": 2}
    assert [r["position"] for r in sim.out.replay_dones] == [0, 1, 2, 3, 4]
    assert "replay_done_rejected" in sim.names()
    assert summary["positions"]["0"]["machine"]                       # labelled with the machine id


def test_replay_summary_files_are_written(tmp_path):
    outputs = RunOutputs(tmp_path)
    sim = Sim(levels=(4,), boards=range(1, 5), outputs=outputs)
    sim.run_sweep()
    sim.controller.handle_uplink(device_id_for(1), 10, p3.encode_replay_done(0, 0b1, 100, 0))
    sim.controller.finalize()
    outputs.close()
    summary = json.loads((tmp_path / "replay_summary.json").read_text())
    assert summary["positions"]["0"]["status"] == "no_alarm"
    assert summary["positions"]["1"]["status"] == "never_finished"
    rows = (tmp_path / "replay_done.csv").read_text().splitlines()
    assert rows[0].startswith("timestamp_utc,device_id,board,position,level_mask") and len(rows) == 2


def test_run_outputs_write_the_files(tmp_path):
    outputs = RunOutputs(tmp_path)
    sim = Sim(levels=(4,), boards=range(1, 5), outputs=outputs)
    sim.run_sweep()
    c = sim.controller
    c.handle_uplink(device_id_for(2), 10, encode_alarm(1, 0, 99, 5.0, 2.0))
    c.handle_uplink(device_id_for(2), 10, encode_self_check(1, 10, 0.25, 0.0625))
    c.finalize()
    outputs.close()

    lines = [json.loads(line) for line in (tmp_path / "delivery_log.jsonl").read_text().splitlines()]
    assert [line["record"] for line in lines] == ["level_start"] + ["round"] * 12 + ["level_end"]
    masks = json.loads((tmp_path / "replay_masks.json").read_text())
    assert len(masks["levels"]["4"]["uplink"]) == 12
    models = [json.loads(line) for line in (tmp_path / "global_models.jsonl").read_text().splitlines()]
    assert len(models) == 13 and len(models[0]["codes"]) == 562
    alarms = (tmp_path / "alarms.csv").read_text().splitlines()
    assert alarms[0].startswith("timestamp_utc,device_id,board,position")
    assert len(alarms) == 2 and ",99," in alarms[1]
    assert len((tmp_path / "self_checks.csv").read_text().splitlines()) == 2
    summary = json.loads((tmp_path / "level_summary.json").read_text())
    assert summary[0]["level"] == 4 and summary[0]["alarms"] == 1
    boards = json.loads((tmp_path / "boards.json").read_text())
    assert sorted(boards) == ["1", "2", "3", "4"]


def test_collecting_ends_after_the_configured_window():
    sim = Sim(levels=(4,), boards=range(1, 5), post_run_collect_seconds=30.0)
    sim.run_sweep()
    assert sim.controller.phase is Phase.COLLECTING
    sim.run_until(lambda: sim.controller.finished)
    assert sim.controller.phase is Phase.DONE
    summary = sim.controller.summary()
    assert summary["finished"] and not summary["failed"]
    assert summary["levels_summary"][0]["rounds_completed"] == 12


def test_global_status_from_a_board_without_machines_is_harmless():
    """Level 4 uses boards 1..4; board 7 hears the multicast too and acknowledges it."""
    sim = Sim(levels=(4,), boards=range(1, 8), expected_boards=4)
    sim.start()
    sim.run_until(lambda: sim.controller.phase is Phase.COLLECTING)
    assert "global_status_ignored" in sim.names()
    assert round_records(sim, 4)[0]["participating_boards"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# round_id slots (13 per level) and READY sequencing (rev 3.1)
# ---------------------------------------------------------------------------
def test_round_id_slots_never_collide_across_a_level_boundary():
    sim = Sim(levels=(4, 8), boards=range(1, 9))
    sim.run_sweep()
    assert sim.controller.phase is Phase.COLLECTING
    records = sim.out.global_models
    assert len(records) == 26 and len({g["round_id"] for g in records}) == 26     # all slots distinct
    assert all(g["round_id"] == g["level_index"] * 13 + g["round"] for g in records)
    on_air = sorted({p3.decode_frame(f).round_id for d, f, _ in sim.published if d == GROUP})
    global_slots = {slot: {slot // 13} for slot in on_air}
    # GLOBAL slots: level 4 (index 0) uses 0..12, level 8 (index 1) uses 13..25.
    assert sorted(global_slots) == list(range(0, 26))
    assert all(len(levels) == 1 for levels in global_slots.values())      # no slot shared by two levels
    assert global_round_id(0, 12) == 12 and global_round_id(1, 0) == 13
    # models on the boards carry the level in their payload, matching the slot's level
    for board in list(sim.boards.values())[:8]:
        for (level_index, round_index), model in board.held_models.items():
            assert model.level_index == level_index and model.round == round_index
    # UPDATE slots are the slot of the GLOBAL the round produces: g = r for round r = 1..12.
    rounds = round_records(sim, 4) + round_records(sim, 8)
    assert [r["round_id"] for r in rounds if r["level"] == 4] == list(range(1, 13))
    assert [r["round_id"] for r in rounds if r["level"] == 8] == list(range(14, 26))
    assert [r["seed_round_id"] for r in rounds if r["level"] == 8] == list(range(13, 25))
    update_slots_4 = {r["round_id"] for r in rounds if r["level"] == 4}
    update_slots_8 = {r["round_id"] for r in rounds if r["level"] == 8}
    assert not update_slots_4 & update_slots_8
    # the payload's own round field stays 0..11
    assert update_round_id(1, 0) == 14 and update_round_id(1, 11) == 25 and update_round_id(6, 11) == 90


def test_slot_arithmetic_and_limits():
    assert global_round_id(6, 12) == 90
    for bad in ((0, 13), (7, 0), (-1, 0)):
        with pytest.raises(p3.ProtocolError):
            global_round_id(*bad)
    with pytest.raises(p3.ProtocolError):
        update_round_id(0, 12)


def test_hello_alarm_and_self_check_carry_round_id_0xff_and_it_is_ignored():
    assert p3.decode_frame(encode_board_hello(1, 3, 1, 3)).round_id == NO_ROUND == 0xFF
    assert p3.decode_frame(encode_alarm(0, 0, 1, 1.0, 2.0)).round_id == 0xFF
    assert p3.decode_frame(encode_self_check(0, 1, 0.0, 0.0)).round_id == 0xFF
    sim = Sim(levels=(4,), boards=range(1, 5))
    sim.run_sweep()
    sim.controller.handle_uplink(device_id_for(1), 10, encode_alarm(0, 0, 5, 3.0, 2.0))
    assert len(sim.out.alarms) == 1


def test_ready_carries_the_slot_of_the_global_the_board_waits_for():
    sim = Sim(levels=(4, 12), boards=range(1, 11))
    sim.run_sweep()
    readies = [(d, x["waiting_for_slot"]) for e, d, x in sim.events if e == "class_c_ready"
               and d == device_id_for(1)]
    slots = [s for _, s in readies]
    # after HELLO: level start; after each upload of level 4: 1..12; after GLOBAL(4,12): the
    # next level start in levels.csv (index 1 -> slot 13); then level 12's rounds 27..38
    assert slots[0] == 0
    assert slots[1:13] == list(range(1, 13))
    assert slots[13] == 13
    assert slots[14:26] == list(range(27, 39))


def test_a_levels_first_global_accepts_any_level_start_slot():
    """levels.csv lists all seven levels, a sweep may run only some: (4, 12) skips index 1."""
    sim = Sim(levels=(4, 12), boards=range(1, 11))
    sim.run_sweep()
    starts = [x for e, d, x in sim.events if e == "global_multicast_starting"]
    level12_start = [x for x in starts if x["round_id"] == global_round_id(2, 0)]
    assert level12_start and level12_start[0]["why"] == "all_boards_ready"
    assert "boards_not_ready" not in sim.names()


def test_a_board_ready_for_the_wrong_slot_is_not_ready():
    sim = Sim(levels=(4,), boards=range(1, 5), ready_wait_seconds=30.0)
    for number, board in sim.boards.items():
        board.hello()
        board.ready(0 if number != 3 else 5)                  # board 3 waits for something else
    sim.run_until(lambda: "global_multicast_starting" in sim.names())
    assert [x for e, _, x in sim.events if e == "boards_not_ready"][0]["boards"] == [3]
    starting = [x for e, _, x in sim.events if e == "global_multicast_starting"][0]
    assert starting["why"] == "ready_wait_expired"


def test_a_repair_answer_ready_for_the_update_slot_triggers_the_next_repair_check():
    """READY for slot L*13+r means 'my upload is done'; anything else does not."""
    sim, c = _board1_at_level12_upload()
    dev = device_id_for(1)
    c.handle_uplink(dev, 10, encode_class_c_ready(update_round_id(2, 0) + 1, 5, 5))   # other slot
    assert "update_repair_requested" not in sim.names()
    c.handle_uplink(dev, 10, encode_class_c_ready(update_round_id(2, 0), 5, 5))
    repairs = [x for e, _, x in sim.events if e == "update_repair_requested"]
    assert sorted(x["position"] for x in repairs) == [0, 10]


# ---------------------------------------------------------------------------
# rev 3.2: run_id
# ---------------------------------------------------------------------------
def test_run_id_is_random_non_zero_and_in_every_global(lossy_level28):
    from controller_v3 import HardwareExperimentController
    from study_plan import load_fl_modules as fl_modules
    ids = set()
    for _ in range(30):
        c = HardwareExperimentController(
            levels=[4], seed=1, expected_boards=4, publish=lambda *a: None, log=lambda *a: None,
            multicast_device_id=GROUP, fl_modules=fl_modules())
        assert 1 <= c.run_id <= 0xFFFF
        ids.add(c.run_id)
    assert len(ids) > 20                                           # not a constant
    with pytest.raises(ValueError):
        HardwareExperimentController(levels=[4], seed=1, expected_boards=4, run_id=0,
                                     publish=lambda *a: None, log=lambda *a: None,
                                     multicast_device_id=GROUP, fl_modules=fl_modules())
    sim = lossy_level28
    assert all(g["run_id"] == RUN_ID for g in sim.out.global_models)
    for board in sim.boards.values():
        assert all(m.run_id == RUN_ID for m in board.held_models.values())


def test_an_update_from_another_run_is_rejected_and_logged():
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    _, payload = _update_for(0, 2, 0, 21)
    stale_run = encode_update(0, 2, 0, 100, quantize_model(np.zeros(562) + 0.01), 0.5,
                              run_id=RUN_ID + 1)
    for frame in fragment_update(stale_run, header):
        c.handle_uplink(device_id_for(1), 10, frame)
    assert 0 not in c._u.updates and 0 in c._u.abandoned
    assert c.counters["update_run_id_mismatches"] == 1
    rejected = [x for e, _, x in sim.events if e == "update_rejected"]
    assert "run_id" in rejected[-1]["error"] and str(RUN_ID) in rejected[-1]["error"]
    # the right run is accepted
    for frame in fragment_update(payload, header):
        c.handle_uplink(device_id_for(1), 10, frame)


def test_hello_reports_the_run_id_held_in_flash():
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=4)
    sim.boards[1].hello(run_id=999)                       # a board from an earlier run
    sim.boards[2].hello(run_id=RUN_ID)
    sim.boards[3].hello(run_id=0)
    assert [e for e in sim.names()].count("board_holds_other_run") == 1
    assert sim.controller.boards[1].flash_run_id == 999


# ---------------------------------------------------------------------------
# rev 3.2: base_slot and stale-base semantics
# ---------------------------------------------------------------------------
def stale_board2(board, level_index, round_index):
    """Board 2 trains from GLOBAL slot g-1 in rounds 3 and 7 (its downlink was not repaired in time)."""
    return round_index - 1 if board == 2 and round_index in (3, 7) else None


def reference_with_bases(sim: Sim, level: int, treat_stale_as_fresh: bool = False):
    twin_server, fl = load_fl_modules()
    li = p3.level_index_of(level)
    positions = active_positions(level)
    server = twin_server.Server(fl.init(list(LAYERS), sim.controller.seed))
    globals_ = [f32_quantised(server.p)]
    for r in range(12):
        seed = on_the_wire(globals_[r]).dequantize()
        pairs = []
        for p in positions:
            delta = on_the_wire(sim.sent[(li, r, p)]).dequantize()
            base = sim.sent_base[(li, r, p)]
            if base == r or treat_stale_as_fresh:
                pairs.append((p3.unflatten(delta), 1.0))
            else:
                client_model = on_the_wire(globals_[base]).dequantize() + delta
                pairs.append((p3.unflatten(client_model - seed), 1.0))
        server.round(pairs)
        globals_.append(f32_quantised(server.p))
    return server, globals_


def test_stale_base_update_is_aggregated_as_client_model_minus_current_global():
    """A board trains from GLOBAL g-1: aggregate (global[g-1] + delta) - global[g], twin/dropout_fl semantics."""
    sim = Sim(levels=(4,), boards=range(1, 5), stale_base=stale_board2)
    sim.run_sweep()
    c = sim.controller
    assert c.phase is Phase.COLLECTING and not c.failed
    server, expected = reference_with_bases(sim, 4)
    logged = logged_global_codes(sim, 4)
    for r, model in enumerate(expected):
        assert np.array_equal(logged[r], model.codes), f"GLOBAL round {r}"
    for ours, theirs in zip(c.level_history[0].server.p, server.p):
        assert np.array_equal(ours, theirs)                       # bit for bit
    # the test is sensitive: treating the stale UPDATE as relative to the current global differs
    _, naive = reference_with_bases(sim, 4, treat_stale_as_fresh=True)
    assert not np.array_equal(logged[4], naive[4].codes)
    assert np.array_equal(logged[3], naive[3].codes)              # identical before the first stale round

    # dropout_fl semantics: the effective delta equals (local model) - (current global) up to the
    # int8 quantisation of the delta, where local = the stale global it trained from + its update.
    li, r, position = 0, 3, 1
    seed = on_the_wire(expected[r]).dequantize()
    base = on_the_wire(expected[r - 1]).dequantize()
    local_model = base + sim.sent_true[(li, r, position)]
    effective = base + on_the_wire(sim.sent[(li, r, position)]).dequantize() - seed
    for i, sl in enumerate(p3.TENSOR_SLICES):
        limit = sim.sent[(li, r, position)].scales[i] / 2 + 1e-12
        assert np.max(np.abs(effective[sl] - (local_model - seed)[sl])) <= limit

    # delivery log and replay masks say which GLOBAL each UPDATE was trained from
    records = {rec["round"]: rec for rec in round_records(sim, 4)}
    assert records[3]["base_slot_by_position"] == {"0": 3, "1": 2, "2": 3, "3": 3}
    assert records[3]["stale_base_positions"] == [1] and records[7]["stale_base_positions"] == [1]
    assert records[2]["stale_base_positions"] == []
    masks = sim.out.replay_masks["levels"]["4"]
    assert masks["base_round"][3] == [3, 2, 3, 3] and masks["base_slot"][3] == [3, 2, 3, 3]
    assert masks["base_round"][7] == [7, 6, 7, 7]
    assert masks["base_round"][5] == [5, 5, 5, 5]
    assert masks["uplink"][3] == [True] * 4                       # a stale UPDATE still arrived
    assert c.counters["stale_base_updates"] == 2


def test_base_slots_in_the_second_level_are_level_relative():
    def stale(board, level_index, round_index):
        return round_index - 1 if level_index == 1 and board == 1 and round_index == 4 else None

    sim = Sim(levels=(4, 8), boards=range(1, 9), stale_base=stale)
    sim.run_sweep()
    rec = {r["round"]: r for r in round_records(sim, 8)}[4]
    assert rec["base_slot_by_position"]["0"] == 13 + 3            # level_index 1 slots start at 13
    assert sim.out.replay_masks["levels"]["8"]["base_round"][4][0] == 3
    server, expected = reference_with_bases(sim, 8)
    for r, model in enumerate(expected):
        assert np.array_equal(logged_global_codes(sim, 8)[r], model.codes)


@pytest.mark.parametrize("base_slot, why", [
    (200, "unknown"),            # no such slot
    (12, "previous level"),      # level 12's slots are 26..38; 12 belongs to level index 0
    (28, "newer than the seed"), # the seed of round 0 is slot 26
    (0, "level 4's slot"),
])
def test_an_unknown_or_out_of_level_base_slot_is_rejected_and_logged(base_slot, why):
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    delta = quantize_model(np.zeros(562) + 0.01)
    payload = encode_update(0, 2, 0, 100, delta, 0.5, run_id=RUN_ID, base_slot=base_slot)
    for frame in fragment_update(payload, header):
        c.handle_uplink(device_id_for(1), 10, frame)
    assert 0 not in c._u.updates and 0 in c._u.abandoned, why
    assert c.counters["update_base_slot_rejections"] == 1
    rejected = [x for e, _, x in sim.events if e == "update_rejected"]
    assert f"base_slot {base_slot}" in rejected[-1]["error"]


# ---------------------------------------------------------------------------
# rev 3.2: board identity and the start rule
# ---------------------------------------------------------------------------
def test_reboots_of_one_board_do_not_count_twice():
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=4)
    for _ in range(4):
        sim.boards[1].hello()                                    # the same DevEUI over and over
    sim.boards[2].hello()
    sim.boards[3].hello()
    assert sim.controller.counted_boards() == [1, 2, 3]
    assert sim.controller.phase is Phase.WAITING_BOARDS
    assert sim.controller.boards[1].hello_count == 4
    sim.boards[4].hello()
    assert sim.controller.phase is Phase.DOWNLINK


def test_a_second_deveui_cannot_take_over_a_bound_board_number():
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=4)
    sim.boards[1].hello()
    thief = "eui-70b3d57ed0099999"
    sim.controller.handle_uplink(thief, 10, encode_board_hello(1, 3, 1, 3))
    assert sim.controller.boards[1].device_id == device_id_for(1)
    assert thief not in sim.controller._board_by_device
    assert sim.controller.counters["board_binding_conflicts"] == 1
    # and the thief's UPDATEs/ALARMs are ignored: it has no bound board
    sim.controller.handle_uplink(thief, 10, encode_alarm(0, 0, 1, 1.0, 2.0))
    assert sim.out.alarms == [] and "ignored_device_without_hello" in sim.names()
    # a bound DevEUI cannot switch to another board number either
    sim.controller.handle_uplink(device_id_for(1), 10, encode_board_hello(2, 3, 1, 3))
    assert sim.controller.boards[1].device_id == device_id_for(1) and 2 not in sim.controller.boards
    assert sim.controller.counters["board_binding_conflicts"] == 2


@pytest.mark.parametrize("flags, counted", [
    (0b011, True),      # SD mounted and self-check passed
    (0b001, False),     # SD mounted, self-check NOT passed
    (0b010, False),     # self-check without SD
    (0b000, False),
    (0b100, True),      # IMU source: bit 1 is waived
    (0b101, True),
    (0b110, True),
])
def test_only_boards_with_sd_and_self_check_or_an_imu_count_towards_the_start(flags, counted):
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=1)
    sim.boards[1].hello(flags=flags)
    assert (sim.controller.counted_boards() == [1]) is counted
    # with expected_boards=1 a counted board triggers the start check (level 4 then fails: it needs boards 1..4)
    assert (sim.controller.phase is not Phase.WAITING_BOARDS) is counted


def test_start_needs_distinct_bound_boards_not_hellos():
    sim = Sim(levels=(4,), boards=range(1, 5), expected_boards=4)
    for number in (1, 2, 3):
        sim.boards[number].hello()
    sim.boards[3].hello(flags=0b001)          # board 3 reboots with a failing self-check
    sim.boards[4].hello()
    assert sim.controller.counted_boards() == [1, 2, 4]
    assert sim.controller.phase is Phase.WAITING_BOARDS
    sim.boards[3].hello()
    assert sim.controller.phase is not Phase.WAITING_BOARDS


def test_results_from_a_position_of_another_board_are_rejected():
    sim = Sim(levels=(4,), boards=range(1, 5))
    sim.run_sweep()
    c = sim.controller
    c.handle_uplink(device_id_for(1), 10, encode_self_check(11, 5, 0.0, 0.0))    # 11 -> board 2
    c.handle_uplink(device_id_for(2), 10, encode_self_check(11, 5, 0.0, 0.0))    # board 2 owns 11
    assert [s["position"] for s in sim.out.self_checks] == [11]
    assert c.counters["results_rejected"] == 1


# ---------------------------------------------------------------------------
# rev 3.2: exact control bodies and fragment validation in the controller
# ---------------------------------------------------------------------------
def test_old_control_bodies_are_rejected_by_the_controller():
    sim = Sim(levels=(4,), boards=range(1, 5))
    sim.start()
    c = sim.controller
    dev = device_id_for(1)
    c.handle_uplink(dev, 10, p3.encode_frame(p3.MSG_GLOBAL_STATUS, 0, 0, 3, bytes(6)))       # v2 status
    c.handle_uplink(dev, 10, p3.encode_frame(p3.MSG_BOARD_HELLO, 0xFF, 0, 1, bytes((1, 3, 1, 3))))
    c.handle_uplink(dev, 10, p3.encode_frame(p3.MSG_GLOBAL_STATUS, 0, 0, 3, b"\x08\x00"))   # bit 3
    assert c.counters["frames_rejected"] == 3
    assert not c.failed


def test_a_fragment_with_a_bad_header_is_not_stored():
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    _, payload = _update_for(0, 2, 0, 31)
    good = fragment_update(payload, header)
    dev = device_id_for(1)
    for offset, value in ((4, 3), (5, 2), (5, 4)):               # index >= count, count != 3
        frame = bytearray(good[1])
        frame[offset] = value
        c.handle_uplink(dev, 10, bytes(frame))
    truncated = good[2][:-1]
    c.handle_uplink(dev, 10, truncated)                          # declared body_len no longer matches
    assert 0 not in c._u.partials and 0 not in c._u.updates
    assert c.counters["frames_rejected"] == 4


def test_a_conflicting_duplicate_fragment_is_logged_and_not_stored():
    sim, c = _board1_at_level12_upload()
    header = update_round_id(2, 0)
    delta, payload = _update_for(0, 2, 0, 41)
    frames = fragment_update(payload, header)
    dev = device_id_for(1)
    c.handle_uplink(dev, 10, frames[1])
    altered = bytearray(frames[1])
    altered[20] ^= 0x55                                          # same slot, different bytes
    c.handle_uplink(dev, 10, bytes(altered))
    assert c.counters["update_fragment_conflicts"] == 1
    assert "update_fragment_conflict" in sim.names()
    assert 0 in c._u.partials                                    # the partial survived
    c.handle_uplink(dev, 10, frames[0])
    c.handle_uplink(dev, 10, frames[2])
    assert 0 in c._u.updates                                     # completed from the ORIGINAL bytes
    expected = QuantisedModel(delta.codes, delta.wire_scales()).dequantize_tuple()
    assert all(np.array_equal(x, y) for x, y in zip(c._u.updates[0].deq, expected))
