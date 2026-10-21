"""Crash recovery: an atomic checkpoint after every round, and --resume continuing bit-identically."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

import protocol_v4 as p4
from controller_v4 import CHECKPOINT_VERSION, ControllerV4, Stage
from plan_v4 import elastic_plan, planned_plan
from run_outputs_v4 import RunOutputs, load_checkpoint
from v4_ref import f32_quantised, on_the_wire
from v4_sim import GROUP, RUN_ID, Sim, dev_eui_for, thesis_boards

PLAN = planned_plan([4, 8], 17070, 6)         # two phases, 6 rounds each
BOARDS = thesis_boards(8, 8)                  # board b hosts position b - 1


def lossy(board, position, fragment, attempt, phase, round_):
    """Repairable losses, deterministic in (position, fragment, attempt, phase, round)."""
    if (position + round_ + phase) % 5 == 0 and fragment == 2 and attempt == 1:
        return True
    return position == 3 and round_ == 4 and phase == 1 and attempt <= 2     # a double loss, then repaired


def make_sim(directory, **kwargs):
    out = RunOutputs(directory)
    sim = Sim(PLAN, boards=BOARDS, outputs=out, uplink_drop=lossy, post_run_collect_seconds=300.0,
              replay_repeat_seconds=60.0, alarm_for=lambda position, phase: position == 1 and phase == 1,
              **kwargs)
    return sim, out


def closed(sim):
    return len(sim.out.records("round"))


def read_models(directory):
    rows = [json.loads(s) for s in (directory / "global_models.jsonl").read_text().splitlines()]
    return sorted(rows, key=lambda r: r["slot"])


def read_rounds(directory):
    rows = [json.loads(s) for s in (directory / "delivery_log.jsonl").read_text().splitlines()]
    return [r for r in rows if r["record"] == "round"]


def csv_lines(directory, name):
    return (directory / name).read_text().splitlines()


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("uninterrupted")
    sim, out = make_sim(directory)
    sim.run_to_done()
    out.close()
    assert sim.controller.stage is Stage.DONE
    return directory, sim


# ---------------------------------------------------------------------------
# kill mid-phase, resume, identical globals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rounds_closed, extra_steps", [
    (3, 0),          # right after a round closed and its checkpoint, before the GLOBAL went out
    (3, 4),          # while the GLOBAL is being multicast / boards are training
    (3, 40),         # mid-upload with UPDATEs already in flight
    (6, 0),          # the very end of phase 0: the phase boundary
    (6, 12),         # early in phase 1
    (8, 9),          # mid phase 1
    (11, 3),         # the last round of phase 1
])
def test_resume_is_bit_identical_to_an_uninterrupted_run(uninterrupted, tmp_path, rounds_closed, extra_steps):
    ref_dir, ref = uninterrupted
    sim, out = make_sim(tmp_path)
    sim.start()
    sim.run_until(lambda: closed(sim) >= rounds_closed)
    for _ in range(extra_steps):
        sim.step()
    assert sim.controller.stage not in (Stage.DONE, Stage.FAILED)
    out.close()                                              # the process dies here
    checkpoint = load_checkpoint(tmp_path)
    assert checkpoint is not None and checkpoint["version"] == CHECKPOINT_VERSION
    out2 = RunOutputs(tmp_path)                              # --resume reopens the same run directory
    sim.out = out2
    resumed = sim.resume_controller(checkpoint, outputs=out2)
    assert resumed.run_id == RUN_ID and resumed.slot_counter >= 1
    sim.run_until(lambda: resumed.stage in (Stage.DONE, Stage.FAILED), max_steps=80000)
    out2.close()
    assert resumed.stage is Stage.DONE and not resumed.failed

    # the final (and every) global equals the uninterrupted run's, code for code
    a, b = read_models(ref_dir), read_models(tmp_path)
    assert [(m["slot"], m["phase_index"], m["round"]) for m in a] == [(m["slot"], m["phase_index"], m["round"]) for m in b]
    for x, y in zip(a, b):
        assert x["codes"] == y["codes"] and x["crc32"] == y["crc32"] and x["scales_f32"] == y["scales_f32"]
    # FedAdam's float64 master state of the last phase, bit for bit
    for name in ("p", "m", "v"):
        for ours, theirs in zip(getattr(resumed.phases[-1].server, name), getattr(ref.controller.phases[-1].server, name)):
            assert np.array_equal(ours, theirs), name
    assert resumed.phases[-1].server.t == ref.controller.phases[-1].server.t == 6

    # the logs are continuous: no duplicate slot / round, the same rounds as the reference
    rounds_ref, rounds_b = read_rounds(ref_dir), read_rounds(tmp_path)
    assert [(r["phase_index"], r["round"]) for r in rounds_b] == [(r["phase_index"], r["round"]) for r in rounds_ref]
    assert [r["arrived_positions"] for r in rounds_b] == [r["arrived_positions"] for r in rounds_ref]
    assert [r["base_slot_by_position"] for r in rounds_b] == [r["base_slot_by_position"] for r in rounds_ref]
    masks_ref = json.loads((ref_dir / "replay_masks.json").read_text())
    masks_b = json.loads((tmp_path / "replay_masks.json").read_text())
    for pa, pb in zip(masks_ref["phases"], masks_b["phases"]):
        assert pa["uplink"] == pb["uplink"] and pa["base_slot"] == pb["base_slot"]
        assert pa["positions"] == pb["positions"] and pb["complete"]
    kinds = [json.loads(s)["record"] for s in csv_lines(tmp_path, "delivery_log.jsonl")]
    assert kinds.count("phase_start") == 2 and kinds.count("phase_end") == 2 and kinds.count("round") == 12
    # the same boards are in the same state, and the replay finished identically
    assert {m.dev_eui: m.state for m in resumed.membership.members.values()} == {
        m.dev_eui: m.state for m in ref.controller.membership.members.values()}
    assert out2.replay_summary["counts"] == ref.out.replay_summary["counts"]
    alarms = csv_lines(tmp_path, "alarms.csv")
    assert len(alarms) == 2 and alarms[1].startswith(f"{RUN_ID},1,1,")             # no duplicate alarm rows
    assert len(csv_lines(tmp_path, "replay_done.csv")) == len(csv_lines(ref_dir, "replay_done.csv"))


def test_the_resumed_controller_continues_the_slot_counter_and_re_requests_lost_updates(tmp_path):
    sim, out = make_sim(tmp_path)
    sim.start()
    sim.run_until(lambda: closed(sim) >= 2)
    # kill the server mid-round, after some UPDATEs of the round already arrived (and will be lost)
    sim.run_until(lambda: sim.controller.stage is Stage.UPLOAD and sim.controller._u.updates, max_steps=5000)
    slots_before = sim.controller.slot_counter
    out.close()
    out2 = RunOutputs(tmp_path)
    sim.out = out2
    c = sim.resume_controller(load_checkpoint(tmp_path), outputs=out2)
    assert c.slot_counter == slots_before
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=80000)
    names = sim.names()
    assert "resume_loaded" in names and "resume_repair_requested" in names
    slots = [m["slot"] for m in read_models(tmp_path)]
    assert slots == list(range(1, 15))                                              # 2 phases x (6 rounds + initial)
    out2.close()


def assert_matches_twin_from_disk(directory, sim):
    """Re-derive the globals from the delivery log and the deltas the boards sent."""
    import study_plan
    rounds = read_rounds(directory)
    models = {(m["phase_index"], m["round"]): m for m in read_models(directory)}
    twin_server, fl = study_plan.load_fl_modules()
    for phase in (0, 1):
        server = twin_server.Server(fl.init(list(p4.LAYERS), 17070))
        assert models[(phase, 0)]["codes"] == f32_quantised(server.p).codes.tolist()
        first = models[(phase, 0)]["slot"]
        for rec in [r for r in rounds if r["phase_index"] == phase]:
            r = rec["round"]
            pairs = [(p4.unflatten(on_the_wire(sim.sent[(phase, r, p)]).dequantize()), 1.0)
                     for p in sorted(rec["arrived_positions"])]
            assert all(rec["base_slot_by_position"][str(p)] == first + r - 1 for p in rec["arrived_positions"])
            server.round(pairs)
            assert models[(phase, r)]["codes"] == f32_quantised(server.p).codes.tolist(), (phase, r)


def test_checkpoint_now_mid_round_is_resumable(tmp_path):
    """Ctrl-C: checkpoint_now() at an arbitrary moment, then resume."""
    sim, out = make_sim(tmp_path)
    sim.start()
    sim.run_until(lambda: closed(sim) >= 4)
    for _ in range(11):
        sim.step()
    sim.controller.checkpoint_now()
    out.close()
    ck = load_checkpoint(tmp_path)
    out2 = RunOutputs(tmp_path)
    sim.out = out2
    c = sim.resume_controller(ck, outputs=out2)
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=80000)
    assert c.stage is Stage.DONE
    assert_matches_twin_from_disk(tmp_path, sim)
    out2.close()


def test_resume_during_collecting_finishes_replay_and_end_without_duplicates(uninterrupted, tmp_path):
    ref_dir, _ = uninterrupted
    sim, out = make_sim(tmp_path, replay_ignore=lambda board, n: board == 2 and n <= 1)
    sim.start()
    sim.run_until(lambda: sim.controller.stage is Stage.COLLECTING)
    sim.run_until(lambda: sim.controller.replay_done, max_steps=3000)      # some REPLAY_DONE / ALARM arrived
    for _ in range(3):
        sim.step()
    assert sim.controller.stage is Stage.COLLECTING and sim.controller.replay_done
    sim.controller.checkpoint_now()
    out.close()
    out2 = RunOutputs(tmp_path)
    sim.out = out2
    c = sim.resume_controller(load_checkpoint(tmp_path), outputs=out2)
    assert c.stage is Stage.COLLECTING and c.replay_done                  # what arrived before the crash is kept
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=80000)
    out2.close()
    assert c.stage is Stage.DONE
    commands = [rc.command for rc in sim.run_controls]
    assert commands.count(p4.CMD_END) >= 1 and commands[-1] == p4.CMD_END
    assert len(csv_lines(tmp_path, "alarms.csv")) == 2
    assert len(csv_lines(tmp_path, "replay_done.csv")) == len(csv_lines(ref_dir, "replay_done.csv"))
    assert [m["codes"] for m in read_models(tmp_path)] == [m["codes"] for m in read_models(ref_dir)]


def test_resume_a_finished_run_stays_done(uninterrupted, tmp_path):
    ref_dir, ref = uninterrupted
    (tmp_path / "checkpoint.json").write_text((ref_dir / "checkpoint.json").read_text())
    ck = load_checkpoint(tmp_path)
    assert ck["stage"] == "done"
    c = ControllerV4.resume(ck, publish=lambda *a: None, log=lambda *a: None, multicast_device_id=GROUP,
                            params=ref.params, fl_modules=ref._fl)
    assert c.stage is Stage.DONE and c.run_id == RUN_ID


# ---------------------------------------------------------------------------
# What the checkpoint holds, and how it is written
# ---------------------------------------------------------------------------
def test_checkpoint_holds_everything_the_contract_lists(tmp_path):
    sim, out = make_sim(tmp_path)
    sim.start()
    sim.run_until(lambda: closed(sim) >= 3)
    ck = load_checkpoint(tmp_path)
    out.close()
    assert ck["version"] == CHECKPOINT_VERSION and ck["run_id"] == RUN_ID
    assert ck["plan"]["mode"] == "planned" and len(ck["plan"]["phases"]) == 2 and ck["plan"]["seed"] == 17070
    assert ck["phase_index"] == 0 and 3 <= ck["round"] <= 4 and ck["slot_counter"] == ck["current_slot"]
    fed = ck["fedadam"]
    assert fed["t"] >= 3 and fed["dtype"] == "float64"
    for key in ("p", "m", "v"):
        assert [len(base64.b64decode(x)) // 8 for x in fed[key]] == [264, 12, 264, 22]
    slots = sorted(int(s) for s in ck["phase"]["globals"])
    assert slots == list(range(1, ck["slot_counter"] + 1))                       # every GLOBAL of the phase
    for slot, b64 in ck["phase"]["globals"].items():
        decoded = p4.decode_global(base64.b64decode(b64))
        assert decoded.run_id == RUN_ID and decoded.round == int(slot) - 1
    members = {m["dev_eui"]: m for m in ck["membership"]["members"]}
    assert len(members) == 8 and {m["state"] for m in members.values()} <= {"ACTIVE", "BOUND"}
    assert members[dev_eui_for(1)]["positions"] == [0] and members[dev_eui_for(1)]["board_id"] == 1
    assert set(ck["repair_state"]) >= {"phase_index", "round", "repair_rounds", "abandoned"}
    assert ck["stage"] in ("downlink", "upload")
    assert ck["pending_round"] is None or "record" in ck["pending_round"]


def test_checkpoint_is_written_atomically_with_a_previous_generation(tmp_path):
    sim, out = make_sim(tmp_path)
    sim.start()
    sim.run_until(lambda: closed(sim) >= 3)
    out.close()
    names = {p.name for p in tmp_path.iterdir()}
    assert "checkpoint.json" in names and "checkpoint.prev.json" in names
    assert not any(n.endswith(".tmp") for n in names)                              # os.replace left no temp file
    good = json.loads((tmp_path / "checkpoint.json").read_text())
    (tmp_path / "checkpoint.json").write_text('{"truncated": ')                    # a torn write
    fallback = load_checkpoint(tmp_path)
    assert fallback is not None and fallback["version"] == CHECKPOINT_VERSION and fallback != good
    assert load_checkpoint(tmp_path / "does_not_exist") is None


def test_a_checkpoint_is_taken_for_every_global_of_every_phase():
    sim = Sim(PLAN, boards=BOARDS)
    seen = []
    original = sim.out.write_checkpoint

    def spy(ck):
        seen.append((ck["phase_index"], ck["round"], ck["slot_counter"]))
        original(ck)

    sim.out.write_checkpoint = spy
    sim.run_to_done()
    taken = {(p, r) for p, r, _ in seen if r is not None}
    for phase in (0, 1):
        for r in range(0, 7):
            assert (phase, r) in taken, (phase, r)       # GLOBAL(0) .. GLOBAL(6) of both phases
    assert [s for _, _, s in seen] == sorted(s for _, _, s in seen)               # monotone slot counter


def test_resume_refuses_a_foreign_checkpoint_version(tmp_path):
    sim, out = make_sim(tmp_path)
    sim.start()
    sim.run_until(lambda: closed(sim) >= 1)
    ck = load_checkpoint(tmp_path)
    out.close()
    ck["version"] = 3
    with pytest.raises(ValueError, match="checkpoint version"):
        ControllerV4.resume(ck, publish=lambda *a: None, log=lambda *a: None, multicast_device_id=GROUP,
                            params=sim.params, fl_modules=sim._fl)


def test_resume_before_the_run_started_waits_for_boards_again():
    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,)}, min_boards=2)
    sim.boards[1].hello()
    sim.controller.checkpoint_now()
    ck = sim.out.read_checkpoint()
    assert ck["phase"] is None and ck["stage"] == "waiting_start"
    c = sim.resume_controller(ck)
    assert c.stage is Stage.WAITING_START
    sim.boards[1].hello()
    sim.boards[2].hello()
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED))
    assert c.stage is Stage.DONE


def test_resume_after_an_elastic_join_keeps_the_membership(tmp_path):
    out = RunOutputs(tmp_path)
    sim = Sim(elastic_plan(17070, 1, 8), boards={1: (0,), 2: (1,), 3: (2,)}, outputs=out, min_boards=2)
    sim.start(only={1, 2})
    sim.run_until(lambda: closed(sim) >= 2)
    sim.boards[3].hello()
    sim.run_until(lambda: closed(sim) >= 4)
    out.close()
    ck = load_checkpoint(tmp_path)
    assert {m["dev_eui"]: m["state"] for m in ck["membership"]["members"]}[dev_eui_for(3)] == "ACTIVE"
    out2 = RunOutputs(tmp_path)
    sim.out = out2
    c = sim.resume_controller(ck, outputs=out2)
    sim.run_until(lambda: c.stage in (Stage.DONE, Stage.FAILED), max_steps=60000)
    out2.close()
    assert c.stage is Stage.DONE
    rounds = read_rounds(tmp_path)
    assert [r["round"] for r in rounds] == list(range(1, 9))
    assert all(r["participants"] == [0, 1, 2] for r in rounds if r["round"] >= 5)
