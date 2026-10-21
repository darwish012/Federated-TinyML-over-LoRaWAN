"""ControllerV4: one board, the planned thesis sweep with losses, stale bases, REPLAY / END,
start rules, idle phases and uplink validation, all against fake boards."""
from __future__ import annotations

import json

import numpy as np
import pytest

import protocol_v4 as p4
from controller_v4 import Stage
from plan_v4 import elastic_plan, planned_plan, thesis_plan
from run_outputs_v4 import RunOutputs
from study_plan import load_machine_order
from v4_ref import assert_matches_twin, f32_quantised, on_the_wire, twin_reference
from v4_sim import (GROUP, HELLO_OK, RUN_ID, Sim, device_id_for, dev_eui_for, pseudo_delta,
                    thesis_boards)


# ---------------------------------------------------------------------------
# One board
# ---------------------------------------------------------------------------
def test_one_board_elastic_run_completes_and_matches_the_twin():
    sim = Sim(elastic_plan(17070, 1, 5), boards={1: (0,)})
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and not c.failed
    records = sim.round_records(0)
    assert [r["round"] for r in records] == [1, 2, 3, 4, 5]
    for r in records:
        assert r["arrived_positions"] == [0] and r["close_reason"] == "all_delivered"
        assert r["participating_boards"] == [1]
    server = assert_matches_twin(sim, 0)
    for ours, theirs in zip(c.phases[0].server.p, server.p):
        assert np.array_equal(ours, theirs)                    # float64 master weights, bit for bit
    assert c.phases[0].server.t == 5
    states = {m.dev_eui: m.state.value for m in c.membership.members.values()}
    assert states == {dev_eui_for(1): "ACTIVE"}


def test_slots_are_run_wide_monotonic_and_headers_carry_them():
    sim = Sim(elastic_plan(17070, 2, 3), boards={1: (0,), 2: (1,)})
    sim.run_to_done()
    slots = [g["slot"] for g in sim.out.global_models]
    assert slots == list(range(1, 9))                          # 2 phases x (3 rounds + initial)
    assert [(g["phase_index"], g["round"]) for g in sim.out.global_models] == [
        (p, r) for p in range(2) for r in range(4)]
    assert [g["final"] for g in sim.out.global_models] == [False, False, False, True] * 2
    assert all(g["run_id"] == RUN_ID for g in sim.out.global_models)
    # every multicast fragment's header round_id is its GLOBAL's slot, never reused
    seen = {k[0] for k in sim.global_transmissions}
    assert seen == set(slots)
    # the second phase starts again from fl.init(seed): its round-0 GLOBAL equals phase 0's
    assert sim.out.global_models[4]["codes"] == sim.out.global_models[0]["codes"]
    assert_matches_twin(sim, 0)
    assert_matches_twin(sim, 1)


def test_initial_global_is_the_quantised_fl_init():
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,)})
    sim.run_to_done()
    import study_plan
    _, fl = study_plan.load_fl_modules()
    first = np.array(sim.out.global_models[0]["codes"], dtype=np.int8)
    assert np.array_equal(first, f32_quantised(fl.init(list(p4.LAYERS), 17070)).codes)
    assert sim.out.global_models[0]["crc32"] == f"{p4.decode_global(sim.controller.phases[0].globals_payload[1]).crc32:08X}"


# ---------------------------------------------------------------------------
# The planned thesis sweep: 7 phases, 10 boards, losses and repairs
# ---------------------------------------------------------------------------
def sweep_loss(board, position, fragment, attempt, phase, round_):
    if phase == 6 and round_ == 2 and position == 10 and fragment == 0 and attempt == 1:
        return True                                  # fragment 0 lost: 1 and 2 still arrive, each naming its position
    if phase == 6 and round_ == 4 and position == 20 and fragment == 2 and attempt == 1:
        return True                                  # only the last fragment
    if phase == 6 and round_ == 6 and position == 13 and fragment == 1 and attempt <= 2:
        return True                                  # the first repair loses it again
    if phase == 6 and round_ == 8 and position == 5:
        return attempt == 1                          # the whole UPDATE
    if phase == 2 and round_ == 3 and position == 9:
        return attempt == 1                          # an earlier phase too
    return False


@pytest.fixture(scope="module")
def sweep():
    order = load_machine_order()[0]
    sim = Sim(thesis_plan(17070, 12, order), boards=thesis_boards(10, 28), uplink_drop=sweep_loss)
    sim.run_to_done()
    return sim


def test_thesis_sweep_is_seven_phases_of_twelve_rounds(sweep):
    c = sweep.controller
    assert c.stage is Stage.DONE and not c.failed
    plan = c.plan
    assert [len(p.positions) for p in plan.phases] == [4, 8, 12, 16, 20, 24, 28]
    assert [p.rounds for p in plan.phases] == [12] * 7
    assert [p.positions for p in plan.phases] == [tuple(range(n)) for n in (4, 8, 12, 16, 20, 24, 28)]
    assert plan.machine_order == tuple(load_machine_order()[0])        # built from machine_order_v2
    assert len(sweep.out.records("round")) == 84
    assert len(sweep.out.global_models) == 91
    for phase in range(7):
        recs = sweep.round_records(phase)
        assert [r["round"] for r in recs] == list(range(1, 13))
        n = len(plan.phases[phase].positions)
        for r in recs:
            assert r["arrived_positions"] == list(range(n)), (phase, r["round"])
            assert r["missing_positions"] == [] and r["close_reason"] == "all_delivered"
            assert r["expected_positions"] == list(range(n))


def test_repairs_name_the_board_position_and_missing_fragments(sweep):
    by = {(p, r): rec for p in range(7) for rec in sweep.round_records(p) for r in [rec["round"]]}
    assert by[(6, 2)]["update_repairs_by_position"] == {"10": 1}
    assert by[(6, 4)]["update_repairs_by_position"] == {"20": 1}
    assert by[(6, 6)]["update_repairs_by_position"] == {"13": 2}
    assert by[(6, 8)]["update_repairs_by_position"] == {"5": 1}
    assert by[(2, 3)]["update_repairs_by_position"] == {"9": 1}
    masks = {(pos, m) for _, pos, m in sweep.repairs}
    assert (10, 0b001) in masks and (20, 0b100) in masks and (13, 0b010) in masks
    assert (5, 0b111) in masks and (9, 0b111) in masks
    assert by[(6, 1)]["update_repair_requests"] == 0
    assert by[(6, 2)]["update_fragments_received"] == 3 * 28 - 1 + 1


def test_repair_downlinks_are_7_byte_control_bodies_with_the_round_slot(sweep):
    frames = [f for d, f, _ in sweep.published if d != GROUP]
    assert frames
    for frame in frames:
        decoded = p4.decode_frame(frame)
        assert decoded.message_type == p4.MSG_UPDATE_REPAIR and len(frame) == 8 + 7
        rep = p4.decode_update_repair(decoded)
        assert rep.run_id == RUN_ID
        # header round_id = slot of the round's seed GLOBAL, a real slot
        assert 1 <= decoded.round_id <= 91


def test_every_phase_aggregate_is_bit_identical_to_twin_server(sweep):
    c = sweep.controller
    for phase in range(7):
        server = assert_matches_twin(sweep, phase)
        for ours, theirs in zip(c.phases[phase].server.p, server.p):
            assert np.array_equal(ours, theirs)
        assert c.phases[phase].server.t == 12


def test_planned_boards_upload_only_their_active_positions(sweep):
    """Rev 4.3: the GLOBAL's participation mask tells a board which hosted positions take part,
    so nothing out of phase is uploaded (rev 4.2 boards uploaded everything and the server dropped it)."""
    c = sweep.controller
    assert c.counters["updates_ignored_not_in_phase"] == 0 and c.counters["stale_update_fragments"] == 0
    for phase, n in ((0, 4), (3, 16)):
        assert all(r["update_fragments_received"] >= 3 * n for r in sweep.round_records(phase))
        assert all(r["update_fragments_received"] <= 3 * n + 3 * n for r in sweep.round_records(phase))
    assert sweep.out.phase_summaries[0]["updates_arrived_per_round"] == [4] * 12
    assert sweep.out.phase_summaries[6]["updates_arrived_per_round"] == [28] * 12


def test_replay_masks_document(sweep):
    masks = sweep.out.replay_masks
    assert masks["schema"] == "tinyfl-replay-masks-v4" and masks["run_id"] == RUN_ID
    assert [p["phase_index"] for p in masks["phases"]] == list(range(7))     # rev 4.1: a list
    assert masks["mode"] == "planned" and masks["seed"] == 17070
    top = sweep.masks(6)
    assert top["rounds_in_phase"] == 12 and top["planned_positions"] == list(range(28))
    assert top["positions"] == list(range(28)) and top["rounds_completed"] == 12 and top["complete"]
    assert len(top["uplink"]) == len(top["downlink"]) == len(top["base_slot"]) == 12
    assert all(len(row) == 28 for row in top["uplink"] + top["downlink"] + top["base_slot"])
    assert all(all(row) for row in top["uplink"]) and all(all(row) for row in top["downlink"])
    first = top["first_slot"]
    assert first == 6 * 13 + 1
    for r, row in enumerate(top["base_slot"]):
        assert row == [first + r] * 28                            # all trained from the round's seed
        assert top["base_round"][r] == [r] * 28
    assert top["machines"] == list(load_machine_order()[0][:28])
    small = sweep.masks(0)
    assert small["positions"] == [0, 1, 2, 3] and small["machines"] == list(load_machine_order()[0][:4])


def test_delivery_log_records(sweep):
    starts = sweep.out.records("phase_start")
    ends = sweep.out.records("phase_end")
    assert [s["phase_index"] for s in starts] == list(range(7))
    assert [e["phase_index"] for e in ends] == list(range(7))
    assert all(e["status"] == "complete" and e["rounds_completed"] == 12 for e in ends)
    rec = sweep.round_records(0)[0]
    # PROTOCOL_V4 4.1: participants, arrived/stale/joined/left positions and the counters
    for key in ("participants", "arrived_positions", "stale_positions", "joined_positions",
                "left_positions", "update_fragments_received", "update_repair_requests",
                "update_repair_downlinks_sent", "global_fragments_sent", "global_repair_rounds",
                "global_bursts", "run_id"):
        assert key in rec, key
    for key in ("participating_boards", "base_slot_by_position", "joined_boards", "left_boards",
                "membership_changes", "timings", "boards_holding_result_global",
                "result_global_crc32", "seed_slot", "result_slot"):
        assert key in rec, key
    assert rec["participants"] == [0, 1, 2, 3] and rec["joined_positions"] == []   # first row: none
    assert rec["run_id"] == RUN_ID
    assert rec["seed_slot"] == 1 and rec["result_slot"] == 2
    assert rec["participating_boards"] == [1, 2, 3, 4]
    assert starts[0]["boards_holding_initial_global"] == [1, 2, 3, 4]      # only they host positions of phase 0
    assert starts[0]["planned_positions"] == [0, 1, 2, 3]
    # the first round of a run records the boards that joined (BOUND -> ACTIVE on their acks)
    assert sorted(rec["joined_boards"]) == [1, 2, 3, 4]


def test_global_models_jsonl_has_codes_scales_and_crc(sweep):
    g = sweep.out.global_models[13]
    assert g["slot"] == 14 and g["phase_index"] == 1 and g["round"] == 0
    assert len(g["codes"]) == 562 and len(g["scales_f32"]) == 4 and len(g["crc32"]) == 8
    payload = sweep.controller.phases[1].globals_payload[14]
    decoded = p4.decode_global(payload)
    assert decoded.crc32 == int(g["crc32"], 16)
    assert decoded.model.codes.tolist() == g["codes"]
    assert list(decoded.model.scales) == g["scales_f32"]


# ---------------------------------------------------------------------------
# Stale bases
# ---------------------------------------------------------------------------
def stale_board2(board, phase, round_):
    """Board 2 trains from the GLOBAL one round older in rounds 4 and 8."""
    return 1 if board == 2 and round_ in (4, 8) else None


def test_stale_base_update_is_aggregated_as_client_model_minus_current_global():
    sim = Sim(planned_plan([4], 17070, 12), boards=thesis_boards(4, 4), stale_base=stale_board2)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and not c.failed
    server = assert_matches_twin(sim, 0)
    for ours, theirs in zip(c.phases[0].server.p, server.p):
        assert np.array_equal(ours, theirs)
    # sensitive: treating the stale UPDATE as relative to the current GLOBAL gives a different model
    _, naive = twin_reference(sim, 0, treat_stale_as_fresh=True)
    logged = sim.logged_global_codes(0)
    assert np.array_equal(logged[3], naive[3].codes)              # identical before the first stale round
    assert not np.array_equal(logged[4], naive[4].codes)
    # dropout_fl semantics: effective delta = (stale base + update) - current global
    expected = twin_reference(sim, 0)[1]
    seed_vec = on_the_wire(expected[3]).dequantize()
    base_vec = on_the_wire(expected[2]).dequantize()
    local_model = base_vec + pseudo_delta(1, 0, 4)
    effective = base_vec + on_the_wire(sim.sent[(0, 4, 1)]).dequantize() - seed_vec
    for i, sl in enumerate(p4.TENSOR_SLICES):
        limit = sim.sent[(0, 4, 1)].scales[i] / 2 + 1e-12
        assert np.max(np.abs(effective[sl] - (local_model - seed_vec)[sl])) <= limit
    # logs
    recs = {r["round"]: r for r in sim.round_records(0)}
    first = recs[1]["seed_slot"]
    assert recs[4]["base_slot_by_position"] == {"0": first + 3, "1": first + 2, "2": first + 3, "3": first + 3}
    assert recs[4]["stale_positions"] == [1] and recs[8]["stale_positions"] == [1]
    assert recs[5]["stale_positions"] == []
    top = sim.masks(0)
    assert top["base_round"][3] == [3, 2, 3, 3] and top["base_slot"][3] == [first + 3, first + 2, first + 3, first + 3]
    assert top["base_round"][7] == [7, 6, 7, 7] and top["base_round"][5] == [5] * 4
    assert top["uplink"][3] == [True] * 4                           # a stale UPDATE still arrived
    assert c.counters["stale_base_updates"] == 2


def test_base_slots_in_the_second_phase_are_run_wide_slots():
    def stale(board, phase, round_):
        return 1 if phase == 1 and board == 1 and round_ == 5 else None

    sim = Sim(elastic_plan(17070, 2, 6), boards={1: (0,), 2: (1,)}, stale_base=stale)
    sim.run_to_done()
    rec = sim.round_records(1)[4]
    assert rec["base_slot_by_position"]["0"] == 8 + 3             # phase 1's round-0 slot is 8; base = round 3's GLOBAL
    assert sim.masks(1)["base_round"][4][0] == 3
    assert_matches_twin(sim, 0)
    assert_matches_twin(sim, 1)


# ---------------------------------------------------------------------------
# Hand-made UPDATEs against a controller held in the UPLOAD stage
# ---------------------------------------------------------------------------
def upload_stage(plan=None, boards=None, **kwargs):
    """A sim whose single board takes every GLOBAL but never trains: tests inject UPDATEs."""
    sim = Sim(plan or elastic_plan(17070, 1, 4), boards=boards or {1: (0,)},
              crash=lambda *a: True, **kwargs)
    sim.start()
    sim.run_until(lambda: sim.controller.stage is Stage.UPLOAD)
    return sim, sim.controller


def good_frames(sim, position=0, phase=0, round_=1, run_id=RUN_ID, base_slot=None, seed_slot=1,
                header_slot=None):
    delta = p4.quantize_model(np.zeros(562) + 0.01)
    payload = p4.encode_update(position, run_id, phase, round_, seed_slot if base_slot is None else base_slot,
                               100, delta, 0.5)
    return p4.fragment_update(payload, seed_slot if header_slot is None else header_slot), delta


def send(c, board, frames):
    for frame in frames:
        c.handle_uplink(device_id_for(board), 10, frame)


def event_names(sim):
    return [e for e, _, _ in sim.events]


def test_a_valid_handmade_update_is_accepted():
    sim, c = upload_stage()
    frames, delta = good_frames(sim)
    send(c, 1, frames)
    assert 0 in c._u.updates and c.counters["frames_rejected"] == 0
    sim.run_until(lambda: c.stage is Stage.DOWNLINK and c._g.round == 1)


@pytest.mark.parametrize("kwargs, counter", [
    (dict(run_id=RUN_ID + 1), "update_run_id_mismatches"),
    (dict(base_slot=99), "update_base_slot_rejections"),         # unknown slot
    (dict(base_slot=2), "update_base_slot_rejections"),          # newer than the seed (seed is slot 1)
    (dict(base_slot=0), "update_base_slot_rejections"),          # before the phase's first slot
])
def test_unacceptable_updates_are_rejected_and_abandoned(kwargs, counter):
    sim, c = upload_stage()
    frames, _ = good_frames(sim, **kwargs)
    send(c, 1, frames)
    assert c.counters[counter] == 1
    assert 0 not in c._u.updates and 0 in c._u.abandoned
    assert "update_rejected" in event_names(sim)


def test_wrong_phase_or_round_is_dropped_without_abandoning():
    sim, c = upload_stage()
    for phase, round_ in ((1, 1), (0, 2), (0, 9)):
        frames, _ = good_frames(sim, phase=phase, round_=round_)
        send(c, 1, frames)
    assert 0 not in c._u.updates and 0 not in c._u.abandoned
    assert event_names(sim).count("update_rejected") == 3


def test_fragment_header_slot_must_be_the_rounds_seed_slot():
    sim, c = upload_stage()
    frames, _ = good_frames(sim, header_slot=2)                    # the produced slot, not the seed
    send(c, 1, frames)
    assert c.counters["stale_update_fragments"] == 3 and c._u.fragments_total == 0
    frames, _ = good_frames(sim)
    send(c, 1, frames)
    assert 0 in c._u.updates


def test_position_prefix_must_match_the_update_and_belong_to_the_board():
    sim, c = upload_stage(boards={1: (0,), 2: (7,)})
    frames, _ = good_frames(sim, position=0)
    swapped = [p4.encode_frame(p4.MSG_UPDATE_FRAGMENT, 1, i, 3, b"\x05\x00" + p4.decode_frame(f).data[2:])
               for i, f in enumerate(frames)]                      # prefix says 5, UPDATE says 0
    send(c, 1, swapped)
    assert c.counters["update_positions_rejected"] == 3            # position 5 is not hosted by board 1
    # board 2 claims board 1's position with a matching prefix
    frames, _ = good_frames(sim, position=0)
    send(c, 2, frames)
    assert c.counters["update_positions_rejected"] == 6 and not c._u.updates
    # a prefix of a hosted position but a different position inside the UPDATE
    frames, _ = good_frames(sim, position=7)
    mismatched = [p4.encode_frame(p4.MSG_UPDATE_FRAGMENT, 1, i, 3, b"\x00\x00" + p4.decode_frame(f).data[2:])
                  for i, f in enumerate(frames)]
    send(c, 1, mismatched)
    assert c.counters["update_position_mismatches"] == 1


def test_a_corrupted_update_fails_its_crc_and_is_repaired_by_the_board_flow():
    sim, c = upload_stage()
    frames, _ = good_frames(sim)
    bad = bytearray(frames[1])
    bad[40] ^= 0xFF
    send(c, 1, [frames[0], bytes(bad), frames[2]])
    assert c.counters["update_crc_failures"] == 1 and 0 not in c._u.updates
    assert 0 not in c._u.partials                                    # discarded; the repair flow asks again


def test_conflicting_duplicate_fragment_is_not_stored():
    sim, c = upload_stage()
    frames, _ = good_frames(sim)
    send(c, 1, [frames[0]])
    altered = p4.encode_frame(p4.MSG_UPDATE_FRAGMENT, 1, 0, 3, b"\x00\x00" + bytes(211))
    send(c, 1, [altered])
    assert c.counters["update_fragment_conflicts"] == 1
    assert c._u.partials[0].parts[0] == p4.decode_frame(frames[0]).data[2:]


def test_late_fragments_after_the_round_closed_are_dropped():
    sim, c = upload_stage()
    frames, _ = good_frames(sim)
    send(c, 1, frames)
    sim.run_until(lambda: c.stage is Stage.DOWNLINK and c._g.round == 1)
    before = c.counters["late_update_fragments"]
    send(c, 1, frames)
    assert c.counters["late_update_fragments"] + c.counters["stale_update_fragments"] > before


def test_frames_from_unknown_devices_and_malformed_frames_are_survivable():
    sim, c = upload_stage()
    c.handle_uplink("eui-70b3d57ed0ffff01", 10, p4.encode_class_c_ready(0, 1))
    assert c.counters["ignored_device_without_hello"] == 1
    c.handle_uplink(device_id_for(1), 11, b"\x00")                  # wrong FPort
    c.handle_uplink(device_id_for(1), 10, b"\xf1\x04")              # truncated
    c.handle_uplink(device_id_for(1), 10, p4.encode_global_status(RUN_ID, 1, 0)[:-1])
    c.handle_uplink(device_id_for(1), 10, p4.fragment_global(p4.encode_global(
        RUN_ID, 0, 0, 4, p4.quantize_model(np.zeros(562) + 0.1)), 3)[0])    # a downlink type uplinked
    c.handle_uplink("board-one", 10, p4.encode_class_c_ready(0, 1))
    assert c.counters["frames_rejected"] >= 4
    assert not c.failed


def test_old_v3_frames_are_rejected():
    import protocol_v3 as p3
    sim, c = upload_stage()
    c.handle_uplink(device_id_for(1), 10, p3.encode_board_hello(1, 1, 1, 3, 0))
    assert c.counters["frames_rejected"] == 1


# ---------------------------------------------------------------------------
# CLASS_C_READY / GLOBAL_STATUS semantics
# ---------------------------------------------------------------------------
def test_ready_accepts_the_any_phase_start_slot_and_gates_the_first_global():
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,)})
    c = sim.controller
    sim.boards[1].hello()                      # HELLO + READY(0xFFFF)
    assert c.stage is Stage.DOWNLINK
    assert c.membership.get(dev_eui_for(1)).ready_slot == p4.NO_SLOT
    assert c._ready_for(c.membership.get(dev_eui_for(1)), c._g)
    sim.run_until(lambda: c._g.ever_armed)
    assert "all_boards_ready" in [x["why"] for e, _, x in sim.events if e == "global_multicast_starting"]


def test_a_board_that_is_not_ready_delays_the_multicast_until_ready_wait_expires():
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,)}, ready_wait_seconds=40.0)
    c = sim.controller
    board = sim.boards[1]
    board.uplink(p4.encode_board_hello(1, 1, HELLO_OK, 0, (0,)))      # HELLO but never READY
    sim.run_until(lambda: c._g.ever_armed, max_steps=200)
    why = [x["why"] for e, _, x in sim.events if e == "global_multicast_starting"]
    assert why == ["ready_wait_expired"]
    assert "boards_not_ready" in event_names(sim)


def test_global_repair_resends_only_the_union_of_what_boards_reported_missing():
    seen = []

    def downlink_drop(board, slot, fragment, count):
        if slot == 1 and count == 1:
            seen.append((board, fragment))
            return (board == 1 and fragment == 0) or (board == 2 and fragment == 2)
        return False

    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,), 2: (1,), 3: (2,)}, downlink_drop=downlink_drop,
              multicast_report_window_seconds=300.0, idle_timeout=60.0)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE
    repairs = [x for e, _, x in sim.events if e == "multicast_prompt_repair"]      # rev 4.3: prompt, not timer-driven
    assert repairs and repairs[0]["fragments"] == [0, 2] and repairs[0]["union_mask"] == "05"
    assert sim.global_transmissions[(1, 1)] == 1                    # fragment 1 was never resent
    assert sim.global_transmissions[(1, 0)] == 2 and sim.global_transmissions[(1, 2)] == 2
    start = sim.out.records("phase_start")[0]
    assert start["global_bursts"] == 2 and start["boards_holding_initial_global"] == [1, 2, 3]
    assert_matches_twin(sim, 0)


def test_a_silent_board_gets_the_whole_global_again_after_the_report_window():
    def drop(board, slot, fragment, count):
        return board == 3 and slot == 4 and count == 1               # board 3 hears nothing of slot 4 once

    sim = Sim(elastic_plan(17070, 1, 5), boards={1: (0,), 2: (1,), 3: (2,)}, downlink_drop=drop,
              multicast_report_window_seconds=20.0, idle_timeout=500.0)
    sim.run_to_done()
    assert [sim.global_transmissions[(4, i)] for i in range(3)] == [2, 2, 2]
    rec = sim.round_records(0)[2]                                    # the round that produced slot 4
    assert rec["result_slot"] == 4 and rec["boards_holding_result_global"] == [1, 2, 3]


def test_a_board_that_never_gets_a_global_is_a_downlink_hole_in_the_mask():
    def drop(board, slot, fragment, count):
        return board == 2 and slot == 4                              # never receives slot 4 (GLOBAL round 3)

    sim = Sim(elastic_plan(17070, 1, 6), boards={1: (0,), 2: (1,), 3: (2,)}, downlink_drop=drop,
              multicast_report_window_seconds=20.0, idle_timeout=500.0,
              round_timeout_seconds=100.0, ready_wait_seconds=20.0)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and c.phases[0].status == "complete"
    masks = sim.masks(0)
    # GLOBAL round 3 is the result of round 3: board 2 (position 1) did not get it.
    assert masks["downlink"][2] == [True, False, True]
    assert all(all(row) for i, row in enumerate(masks["downlink"]) if i != 2)
    # it could not train round 4 (no seed GLOBAL), so its uplink is missing that round
    assert masks["uplink"][3] == [True, False, True]
    records = sim.round_records(0)
    assert records[2]["boards_holding_result_global"] == [1, 3]
    assert records[3]["boards_holding_seed_global"] == [1, 3]
    assert records[3]["close_reason"] == "timeout"
    server = assert_matches_twin(sim, 0)
    for ours, theirs in zip(c.phases[0].server.p, server.p):
        assert np.array_equal(ours, theirs)


def test_a_board_that_never_gets_the_initial_global_is_logged_not_fatal():
    def downlink_drop(board, slot, fragment, count):
        return board == 3 and slot == 1            # board 3 never hears phase 0's initial GLOBAL

    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,), 3: (2,)}, downlink_drop=downlink_drop,
              max_global_repair_rounds=1, round_timeout_seconds=100.0)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and not c.failed
    start = sim.out.records("phase_start")[0]
    assert start["boards_missing_initial_global"] == [3]
    # board 3 was never ACTIVE in round 1 (it did not hold the seed), so round 1 only waits for 1 and 2
    assert sim.round_records(0)[0]["participating_boards"] == [1, 2]
    assert 2 not in sim.round_records(0)[0]["arrived_positions"]
    assert_matches_twin(sim, 0)


def test_unconfirmed_downlink_is_retried_after_the_inflight_timeout():
    # the TTS "down/sent" event of the very first downlink never arrives
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,)}, inflight_downlink_timeout_seconds=5.0,
              sent_drop=lambda device_id, frame, n: n == 1)
    sim.start()
    sim.run_until(lambda: len(sim.published) >= 2, max_steps=100)
    assert sim.published[1][1] == sim.published[0][1]                # the same fragment again
    assert p4.decode_frame(sim.published[0][1]).fragment_index == 0
    assert "downlink_status_timeout" in event_names(sim)


def test_a_deterministically_rejected_downlink_stops_the_run_with_the_reason():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,)}, sent_drop=lambda *a: True)
    c = sim.controller
    sim.start()
    sim.run_until(lambda: len(sim.published) >= 1)
    reason = json.dumps({"error": {"name": "no_downlink_path", "attributes": {}}})
    c.handle_downlink_failed(GROUP, reason)
    assert c.failed and "no_downlink_path" in c.failure_reason


# ---------------------------------------------------------------------------
# Round closure
# ---------------------------------------------------------------------------
def test_abandoned_position_is_logged_missing_and_the_round_still_closes():
    def always(board, position, fragment, attempt, phase, round_):
        return position == 1 and round_ == 2

    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,)}, uplink_drop=always,
              max_repair_rounds=2)
    sim.run_to_done()
    rec = sim.round_records(0)[1]
    assert rec["arrived_positions"] == [0] and rec["missing_positions"] == [1]
    assert rec["abandoned_positions"] == [1] and rec["close_reason"] == "all_resolved"
    assert rec["update_repairs_by_position"] == {"1": 2}
    assert "update_abandoned" in event_names(sim)
    assert_matches_twin(sim, 0)


def test_a_silent_board_closes_the_round_by_timeout_and_an_empty_round_leaves_fedadam_t():
    sim = Sim(elastic_plan(17070, 1, 4), boards={1: (0,)}, crash=lambda b, p, r: r == 2,
              round_timeout_seconds=50.0, suspect_after_rounds=5, depart_after_rounds=6)
    sim.run_to_done()
    recs = sim.round_records(0)
    assert recs[1]["close_reason"] == "timeout" and recs[1]["arrived_positions"] == []
    assert recs[1]["missing_positions"] == [0]
    assert sim.controller.phases[0].server.t == 3                    # the empty round did not advance t
    server = assert_matches_twin(sim, 0)
    assert np.array_equal(sim.controller.phases[0].server.p[0], server.p[0])
    g = sim.logged_global_codes(0)
    assert np.array_equal(g[2], g[1])                                 # the empty round left the model unchanged


# ---------------------------------------------------------------------------
# Start rules
# ---------------------------------------------------------------------------
def test_min_boards_waits_for_enough_counted_bound_boards():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,), 2: (1,), 3: (2,)}, min_boards=3)
    c = sim.controller
    sim.boards[1].hello()
    sim.boards[2].hello()
    for _ in range(30):
        sim.step()
    assert c.stage is Stage.WAITING_START
    sim.boards[3].hello()
    assert c.stage is Stage.DOWNLINK


def test_boards_without_sd_or_selfcheck_do_not_count_towards_min_boards():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,), 2: (1,)}, min_boards=2)
    sim.boards[2].flags = p4.HELLO_SD_MOUNTED                       # self-check failed
    sim.boards[1].hello()
    sim.boards[2].hello()
    assert sim.controller.stage is Stage.WAITING_START
    assert "board_not_counted" in event_names(sim)
    sim.boards[2].flags = p4.HELLO_IMU_SOURCE                       # bit 2 waives bit 1
    sim.boards[2].hello()
    assert sim.controller.stage is Stage.DOWNLINK


def test_start_after_seconds_starts_with_whoever_is_bound():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,)}, min_boards=5, start_after_seconds=50.0)
    c = sim.controller
    sim.boards[1].hello()
    sim.run_until(lambda: c.stage is not Stage.WAITING_START, max_steps=200)
    assert sim.clock.now - 1000.0 >= 50
    started = [x for e, _, x in sim.events if e == "experiment_started"][0]
    assert started["started_by"] == "start_after_seconds"


def test_start_after_seconds_with_no_board_keeps_waiting():
    sim = Sim(elastic_plan(17070, 1, 1), boards={1: (0,)}, start_after_seconds=10.0)
    for _ in range(60):
        sim.step()
    assert sim.controller.stage is Stage.WAITING_START


def test_planned_absent_positions_are_logged_not_fatal():
    sim = Sim(planned_plan([8], 17070, 3), boards={1: (0,), 2: (1,), 3: (2,)})
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE and not c.failed
    missing = [x for e, _, x in sim.events if e == "planned_positions_without_board"][0]
    assert missing["positions"] == [3, 4, 5, 6, 7] and missing["phase_index"] == 0
    rec = sim.round_records(0)[0]
    assert rec["expected_positions"] == [0, 1, 2] and rec["arrived_positions"] == [0, 1, 2]
    masks = sim.masks(0)
    assert masks["positions"] == list(range(8))                       # planned positions stay columns
    assert [row[3:] for row in masks["uplink"]] == [[False] * 5] * 3
    assert_matches_twin(sim, 0)


def test_require_all_positions_waits_then_fails_at_start_after_seconds():
    sim = Sim(planned_plan([4], 17070, 2), boards={1: (0,), 2: (1,), 3: (2,)},
              require_all_positions=True, start_after_seconds=30.0)
    sim.start()
    for _ in range(25):
        sim.step()
    assert sim.controller.stage is Stage.WAITING_START
    sim.run_until(lambda: sim.controller.stage is Stage.FAILED, max_steps=100)
    assert "planned positions have no board" in sim.controller.failure_reason


def test_require_all_positions_starts_once_every_position_is_hosted():
    sim = Sim(planned_plan([4], 17070, 2), boards={1: (0,), 2: (1,), 3: (2,), 4: (3,)},
              require_all_positions=True)
    sim.boards[1].hello(); sim.boards[2].hello(); sim.boards[3].hello()
    sim.step()
    assert sim.controller.stage is Stage.WAITING_START
    sim.boards[4].hello()
    assert sim.controller.stage is Stage.DOWNLINK


# ---------------------------------------------------------------------------
# REPLAY / END
# ---------------------------------------------------------------------------
def replay_sim(**kwargs):
    return Sim(elastic_plan(17070, 2, 2), boards={1: (0,), 2: (1,), 3: (2,)}, **kwargs)


def test_replay_is_repeated_until_every_replay_done_then_end_is_sent():
    sim = replay_sim(replay_ignore=lambda board, n: board == 2 and n <= 2,
                     alarm_for=lambda position, phase: position == 1 and phase == 0)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE
    commands = [(rc.command, rc.phases) for rc in sim.run_controls]
    assert all(x[0] != p4.CMD_ADOPT for x in commands)             # every board held no other run: no ADOPT storm at the first GLOBAL
    replays = [x for x in commands if x[0] == p4.CMD_REPLAY]
    assert len(replays) >= 3 and set(replays) == {(p4.CMD_REPLAY, (0, 1))}        # board 2 ignored the first two
    assert commands[-1] == (p4.CMD_END, ()) and commands.count((p4.CMD_END, ())) == 1  # one END reaches every board
    assert len(sim.end_heard) == 3                                                 # ... and silence ends the run
    assert all(rc.run_id == RUN_ID for rc in sim.run_controls)
    summary = sim.out.replay_summary
    assert summary["counts"] == {"alarm": 1, "no_alarm": 2}
    assert summary["positions"]["1"]["status"] == "alarm"
    assert summary["positions"]["1"]["alarm_phase_indices"] == [0]
    assert summary["positions"]["0"]["phases_scored"] == [0, 1]
    assert {(d["position"], d["phase_index"]) for d in sim.out.replay_dones} == {
        (p, ph) for p in range(3) for ph in range(2)}               # one REPLAY_DONE per (position, phase)
    assert [(d["alarm_latched"]) for d in sim.out.replay_dones if (d["position"], d["phase_index"]) == (1, 0)] == [1]
    assert [a["position"] for a in sim.out.alarms] == [1] and sim.out.alarms[0]["phase_index"] == 0
    assert all(b.ended for b in sim.boards.values())


def test_replay_gives_up_at_the_collect_window_and_reports_never_finished():
    sim = replay_sim(replay_ignore=lambda board, n: board == 3, post_run_collect_seconds=200.0,
                     replay_repeat_seconds=50.0)
    sim.run_to_done()
    commands = [rc.command for rc in sim.run_controls if rc.command != p4.CMD_ADOPT]
    assert commands.count(p4.CMD_REPLAY) >= 3 and commands[-1] == p4.CMD_END and commands.count(p4.CMD_END) == 1
    summary = sim.out.replay_summary
    assert summary["positions"]["2"]["status"] == "never_finished"
    assert summary["counts"]["never_finished"] == 1 and summary["counts"]["no_alarm"] == 2
    why = [x["why"] for e, _, x in sim.events if e == "run_control_ending"]
    assert why == ["collect_window_elapsed"]


def test_replay_done_for_another_run_or_a_position_not_owned_is_rejected():
    sim = replay_sim()
    sim.run_to_collecting()
    c = sim.controller
    c.handle_uplink(device_id_for(1), 10, p4.encode_replay_done(RUN_ID + 1, 0, 1, 10, 0))
    c.handle_uplink(device_id_for(1), 10, p4.encode_replay_done(RUN_ID, 1, 1, 10, 0))
    c.handle_uplink(device_id_for(1), 10, p4.encode_alarm(RUN_ID, 2, 0, 1, 1.0, 1.0))
    c.handle_uplink(device_id_for(1), 10, p4.encode_alarm(RUN_ID, 0, 9, 1, 1.0, 1.0))
    assert c.counters["results_other_run"] == 1 and c.counters["results_rejected"] == 2
    assert c.counters["frames_rejected"] >= 1                                 # phase 9 is not in the plan
    c.handle_uplink(device_id_for(1), 10, p4.encode_replay_done(RUN_ID, 0, 9, 10, 0))
    assert c.counters["frames_rejected"] >= 2


def test_alarms_are_deduplicated_per_position_and_phase_and_self_checks_recorded():
    sim = replay_sim()
    sim.run_to_collecting()
    c = sim.controller
    frame = p4.encode_alarm(RUN_ID, 0, 1, 77, 9.0, 3.0)
    c.handle_uplink(device_id_for(1), 10, frame)
    c.handle_uplink(device_id_for(1), 10, frame)
    assert len(sim.out.alarms) == 1 and "duplicate_alarm" in event_names(sim)
    c.handle_uplink(device_id_for(1), 10, p4.encode_self_check(0, 48, 0.5, 0.1))
    assert sim.out.self_checks[0]["position"] == 0


def test_run_with_no_completed_phase_just_sends_end():
    # the only phase plans position 9; nobody hosts it, so the phase idles out and is aborted
    sim = Sim(planned_plan([{"positions": [9]}], 17070, 2), boards={1: (0,)},
              idle_phase_timeout_seconds=100.0, end_linger_seconds=700.0, end_quiet_seconds=420.0)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE
    assert [p.status for p in c.phases] == ["aborted_idle"]
    commands = [rc.command for rc in sim.run_controls]
    assert commands and set(commands) == {p4.CMD_END}                       # never armed: no ADOPT either
    assert [x["why"] for e, _, x in sim.events if e == "run_control_ending"] == ["nothing_to_replay"]
    # the board never held this run (no GLOBAL was ever multicast), so it ignores END and keeps
    # sending its idle HELLO/READY: END is repeated after each uplink, and the linger limit ends the run
    assert len(commands) >= 3 and sim.end_heard == [] and c.end_unconfirmed == [1]
    assert sim.out.records("phase_end")[0]["status"] == "aborted_idle"


# ---------------------------------------------------------------------------
# Idle phase pause
# ---------------------------------------------------------------------------
def idle_plan():
    return planned_plan([2, {"positions": [5], "name": "late"}], 17070, 3)


def test_a_phase_with_no_participants_pauses_until_a_board_appears():
    sim = Sim(idle_plan(), boards={1: (0,), 2: (1,), 3: (5,)}, idle_phase_timeout_seconds=2000.0,
              min_boards=2)
    sim.start(only={1, 2})
    c = sim.controller
    sim.run_until(lambda: c.stage is Stage.IDLE, max_steps=3000)
    assert c.phase.spec.index == 1 and "phase_paused" in event_names(sim)
    for _ in range(50):
        sim.step()
    assert c.stage is Stage.IDLE                                       # still paused: nobody hosts position 5
    sim.boards[3].hello()
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=20000)
    assert c.stage is Stage.DONE
    assert "phase_resuming" in event_names(sim)
    recs = sim.round_records(1)
    assert [r["arrived_positions"] for r in recs] == [[5]] * 3
    assert_matches_twin(sim, 1)


def test_a_phase_that_stays_idle_is_aborted_at_the_idle_timeout_and_the_run_moves_on():
    sim = Sim(idle_plan(), boards={1: (0,), 2: (1,)}, idle_phase_timeout_seconds=300.0)
    sim.run_to_done()
    c = sim.controller
    assert c.stage is Stage.DONE
    assert [p.status for p in c.phases] == ["complete", "aborted_idle"]
    assert "phase_idle_timeout" in event_names(sim)
    ends = sim.out.records("phase_end")
    assert [e["status"] for e in ends] == ["complete", "aborted_idle"]
    replays = [rc.phases for rc in sim.run_controls if rc.command == p4.CMD_REPLAY]
    assert replays and all(phases == (0,) for phases in replays)       # only the completed phase is replayed


# ---------------------------------------------------------------------------
# Outputs on disk
# ---------------------------------------------------------------------------
def test_run_outputs_write_every_file(tmp_path):
    out = RunOutputs(tmp_path)
    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,)}, outputs=out,
              alarm_for=lambda position, phase: position == 0)
    sim.run_to_done()
    sim.controller.finalize()
    out.close()
    names = {p.name for p in tmp_path.iterdir()}
    assert {"delivery_log.jsonl", "global_models.jsonl", "membership_events.jsonl",
            "replay_masks.json", "replay_summary.json", "phase_summary.json", "boards.json",
            "alarms.csv", "self_checks.csv", "replay_done.csv", "checkpoint.json"} <= names
    models = [json.loads(line) for line in (tmp_path / "global_models.jsonl").read_text().splitlines()]
    assert [m["slot"] for m in models] == [1, 2, 3, 4] and len(models[0]["codes"]) == 562
    log = [json.loads(line) for line in (tmp_path / "delivery_log.jsonl").read_text().splitlines()]
    assert [r["record"] for r in log] == ["phase_start", "round", "round", "round", "phase_end"]
    masks = json.loads((tmp_path / "replay_masks.json").read_text())
    assert masks["phases"][0]["rounds_completed"] == 3 and masks["phases"][0]["phase_index"] == 0
    boards = json.loads((tmp_path / "boards.json").read_text())
    assert {b["state"] for b in boards.values()} == {"ACTIVE"}
    members = [json.loads(line) for line in (tmp_path / "membership_events.jsonl").read_text().splitlines()]
    assert {m["to"] for m in members} == {"BOUND", "ACTIVE"}
    alarms = (tmp_path / "alarms.csv").read_text().splitlines()
    assert alarms[0].startswith("run_id,position,phase_index,acquisition_index,score,threshold") and len(alarms) == 2
    done = (tmp_path / "replay_done.csv").read_text().splitlines()
    assert done[0].startswith("run_id,position,phase_index,acquisitions_scored,alarm_latched") and len(done) == 1 + 2 * 1


def test_status_and_summary_are_json_serialisable():
    sim = Sim(elastic_plan(17070, 1, 2), boards={1: (0,), 2: (1,)})
    sim.run_to_done()
    c = sim.controller
    json.dumps(c.status())
    summary = c.summary()
    json.dumps(summary)
    assert summary["protocol"] == "v4" and summary["membership"]["ACTIVE"] == 2
    line = c.status_line()
    assert "[done]" in line and "active=2" in line
