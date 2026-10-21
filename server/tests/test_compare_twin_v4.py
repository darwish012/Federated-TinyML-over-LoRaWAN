"""The v4 server's run directories through analysis/compare_hw_twin_v4.py (rev 4.1, section 4.1).

The simulator's boards do real learning (the int16 twin's local training, firmware float32
quantisation) and a real REPLAY pass; the server aggregates with twin_server, writes its normal
run directory (RunOutputs), and the unchanged analysis tool replays the run through the twin.
It must PASS on a 10-board planned sweep with losses and on an elastic run with mid-phase joins,
and FAIL when a logged global is corrupted.
"""
from __future__ import annotations

import csv
import functools
import json
import shutil

import pytest

try:
    import v4_learning_sim as ls
except Exception as exc:  # noqa: BLE001 - the analysis tool or its research dependencies are not available
    pytest.skip(f"analysis tool unavailable: {exc}", allow_module_level=True)

from controller_v4 import Stage
from plan_v4 import elastic_plan, thesis_plan
from run_outputs_v4 import RunOutputs
from study_plan import load_machine_order
from v4_sim import thesis_boards


@functools.lru_cache(maxsize=1)
def _machine_data():
    """(data, evaluator) of the analysis tool, loaded once (about 4 s); skips if the caches are absent."""
    try:
        return ls.load_machine_data()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"research data unavailable: {exc}")

cmp = ls.cmp


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def planned_losses():
    def uplink_drop(board, position, fragment, attempt, phase, round_):
        if (position + round_ + phase) % 11 == 0 and fragment == 1 and attempt == 1:
            return True                                    # one fragment lost, repaired
        if (position * 7 + round_ * 3 + phase) % 29 == 0 and attempt <= 4:
            return True                                    # the whole UPDATE never gets through
        return False

    def downlink_drop(board, slot, fragment, count):
        if (board + slot) % 17 == 0 and fragment == 0 and count == 1:
            return True                                    # repaired by the union repair
        if board == 3 and slot in (20, 21):
            return True                                    # a persistent hole
        if board == 4 and slot == 52:
            return True                                    # never receives phase 3's FINAL model
        return False

    def stale_base(board, phase, round_):
        return 1 if board == 2 and phase >= 1 and round_ in (5, 9) else None

    return uplink_drop, downlink_drop, stale_base


@pytest.fixture(scope="module")
def planned_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("planned")
    uplink_drop, downlink_drop, stale_base = planned_losses()
    out = RunOutputs(directory)
    sim = ls.LearningSim(thesis_plan(17070, 12, load_machine_order()[0]), boards=thesis_boards(10, 28),
                         data=_machine_data()[0], outputs=out, uplink_drop=uplink_drop, downlink_drop=downlink_drop,
                         stale_base=stale_base, round_timeout_seconds=900.0, ready_wait_seconds=60.0)
    sim.run_to_done(max_steps=200000)
    out.close()
    assert sim.controller.stage is Stage.DONE
    return directory, sim


@pytest.fixture(scope="module")
def elastic_run(tmp_path_factory):
    """Two elastic phases of 6 rounds: boards 9 and 10 (positions 8, 9) join mid-phase, board 3
    (position 2) leaves and comes back, board 7 (position 6) joins at the start of phase 1."""
    directory = tmp_path_factory.mktemp("elastic")
    out = RunOutputs(directory)
    boards = {b: (b - 1,) for b in range(1, 11)}
    sim = ls.LearningSim(elastic_plan(17070, 2, 6), boards=boards, data=_machine_data()[0], outputs=out,
                         min_boards=6, round_timeout_seconds=100.0, suspect_after_rounds=1,
                         depart_after_rounds=2, ready_wait_seconds=30.0,
                         multicast_report_window_seconds=30.0, post_run_collect_seconds=300.0)
    closed = lambda: len(sim.out.records("round"))                   # noqa: E731
    sim.start(only={1, 2, 3, 4, 5, 6})
    sim.run_until(lambda: closed() >= 2)
    sim.boards[9].hello()                                            # mid-phase joins
    sim.run_until(lambda: closed() >= 3)
    sim.boards[10].hello()
    sim.boards[3].power_off()                                        # a board leaves ...
    sim.run_until(lambda: closed() >= 7)
    sim.boards[3].power_on()                                         # ... and returns in phase 1
    sim.boards[7].hello()
    sim.run_until(lambda: sim.controller.stage in (Stage.DONE, Stage.FAILED), max_steps=100000)
    out.close()
    assert sim.controller.stage is Stage.DONE
    return directory, sim


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
def compare(directory, **kwargs):
    data, evaluator = _machine_data()
    return cmp.compare(directory, data, evaluator, **kwargs)


def criteria_of(report):
    return {k: v["criteria"] for k, v in report["phases"].items()}


def test_planned_ten_board_sweep_with_losses_passes_the_twin_comparison(planned_run):
    directory, sim = planned_run
    report = compare(directory)
    assert report["pass"], json.dumps(criteria_of(report))
    assert report["mode"] == "planned" and sorted(report["phases"]) == [str(i) for i in range(7)]
    stats = [ph["loss_statistics"] for ph in report["phases"].values()]
    assert sum(s["missing_uplinks"] for s in stats) > 0              # the losses really happened
    assert sum(s["stale_bases"] for s in stats) > 0
    assert sum(s["update_repair_requests"] for s in stats) > 0
    for i, ph in enumerate(report["phases"].values()):
        assert ph["scope"] == list(range(4 * (i + 1))) and ph["rounds_completed"] == 12
        assert len(ph["rounds"]) == 13 and ph["absent_planned_positions"] == []
        assert all(r["identical_fraction"] >= 0.99 for r in ph["rounds"])
    assert report["phases"]["3"]["no_final_model"]                    # board 4 never got phase 3's final model
    assert sum(ph["first_alarm"]["both_alarming"] for ph in report["phases"].values()) > 20   # a real cohort
    assert all(not ph["strict_mismatches"] and not ph["unfinished"] and not ph["inconsistent_replay_done"]
               for ph in report["phases"].values())


def test_the_planned_run_directory_has_the_normative_fields(planned_run):
    directory, sim = planned_run
    masks = json.loads((directory / "replay_masks.json").read_text())
    assert masks["schema"] == "tinyfl-replay-masks-v4" and masks["mode"] == "planned"
    assert isinstance(masks["run_id"], int) and masks["seed"] == 17070
    assert isinstance(masks["phases"], list) and [p["phase_index"] for p in masks["phases"]] == list(range(7))
    for ph in masks["phases"]:
        for key in ("rounds_in_phase", "rounds_completed", "complete", "positions", "planned_positions",
                    "uplink", "downlink", "base_slot"):
            assert key in ph
        for r, row in enumerate(ph["uplink"]):
            assert [s is not None for s in ph["base_slot"][r]] == row           # null exactly where uplink is false
    models = [json.loads(s) for s in (directory / "global_models.jsonl").read_text().splitlines()]
    assert [m["slot"] for m in models] == sorted(m["slot"] for m in models) and len({m["slot"] for m in models}) == 91
    for key in ("run_id", "phase_index", "round", "slot", "codes", "scales_f32", "crc32"):
        assert key in models[0]
    rounds = [json.loads(s) for s in (directory / "delivery_log.jsonl").read_text().splitlines()]
    rounds = [r for r in rounds if r["record"] == "round"]
    assert len(rounds) == 84
    for key in ("participants", "arrived_positions", "stale_positions", "joined_positions", "left_positions",
                "update_fragments_received", "update_repair_requests", "update_repair_downlinks_sent",
                "global_fragments_sent", "global_repair_rounds", "global_bursts"):
        assert key in rounds[0], key
    with (directory / "alarms.csv").open() as handle:
        assert csv.DictReader(handle).fieldnames[:6] == ["run_id", "position", "phase_index",
                                                         "acquisition_index", "score", "threshold"]
    with (directory / "replay_done.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[:5] == ["run_id", "position", "phase_index", "acquisitions_scored", "alarm_latched"]
        rows = list(reader)
    assert {(int(r["position"]), int(r["phase_index"])) for r in rows} >= {(0, 0), (27, 6)}


def test_elastic_run_with_mid_phase_joins_passes_the_twin_comparison(elastic_run):
    directory, sim = elastic_run
    report = compare(directory)
    assert report["pass"], json.dumps(criteria_of(report))
    assert report["mode"] == "elastic"
    p0, p1 = report["phases"]["0"], report["phases"]["1"]
    m0 = p0["membership"]
    assert m0["8"]["joined_after_start"] and m0["9"]["joined_after_start"]       # mid-phase joins
    assert m0["8"]["first_uplink_row"] < m0["9"]["first_uplink_row"]
    assert p0["scope"] == [0, 1, 2, 3, 4, 5, 8, 9]                  # board 8 (position 7) never came
    assert 7 not in p1["participated"] and 6 in p1["participated"]  # position 6 joined in phase 1
    assert p0["joins_leaves_logged"]["joined"] and p0["loss_statistics"]["participation_basis"] == "delivery_log"
    assert m0["2"]["left_before_end"]                               # board 3 left in phase 0 ...
    assert p1["membership"]["2"]["uplink_rounds"] > 0               # ... and was re-admitted
    assert p0["membership_problems"] == [] and p1["membership_problems"] == []


def test_elastic_joiners_trained_from_their_first_rounds_seed(elastic_run):
    directory, sim = elastic_run
    for phase in (0, 1):
        masks = sim.masks(phase)
        rows = [r for r in sim.round_records(phase)]
        for r, rec in enumerate(rows):
            for p in rec["joined_positions"]:
                col = masks["positions"].index(p)
                if masks["uplink"][r][col]:
                    assert masks["base_round"][r][col] == r        # trained from that round's seed GLOBAL


@pytest.mark.parametrize("which", ["planned", "elastic"])
def test_a_corrupted_global_fails_the_comparison(which, planned_run, elastic_run, tmp_path):
    source, _ = planned_run if which == "planned" else elastic_run
    bad = tmp_path / "bad"
    shutil.copytree(source, bad)
    path = bad / "global_models.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    target = next(r for r in rows if r["phase_index"] == 1 and r["round"] == 4)
    target["codes"][0] = target["codes"][0] - 3 if target["codes"][0] > -120 else target["codes"][0] + 3
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = compare(bad)
    assert not report["pass"]
    assert report["phases"]["1"]["criteria"]["global_codes"] == "FAIL"
    assert report["phases"]["0"]["criteria"]["global_codes"] == "PASS"


def test_a_corrupted_downlink_mask_fails_the_membership_check(elastic_run, tmp_path):
    source, _ = elastic_run
    bad = tmp_path / "badmask"
    shutil.copytree(source, bad)
    path = bad / "replay_masks.json"
    masks = json.loads(path.read_text())
    ph = masks["phases"][0]
    col = ph["positions"].index(8)                                    # joined mid-phase, trained from GLOBAL b > 0
    row = next(r for r, up in enumerate(ph["uplink"]) if up[col])
    base_round = ph["base_slot"][row][col] - ph["first_slot"]
    assert base_round > 0
    ph["downlink"][base_round - 1][col] = False                       # "it never held the GLOBAL it trained from"
    path.write_text(json.dumps(masks))
    report = compare(bad)
    assert report["phases"]["0"]["criteria"]["membership"] == "FAIL" and not report["pass"]


def test_cli_exit_codes_on_server_output(planned_run, elastic_run, tmp_path, monkeypatch, capsys):
    import sys
    from unittest import mock

    monkeypatch.setattr(cmp, "load_data", _machine_data)

    def run(args):
        with mock.patch.object(sys, "argv", ["compare_hw_twin_v4.py"] + [str(a) for a in args]):
            try:
                return cmp.main()
            except SystemExit as exit_:
                return exit_.code

    assert run([elastic_run[0], "--output", tmp_path / "ok"]) == 0
    assert capsys.readouterr().out.strip() == "PASS"
    bad = tmp_path / "bad"
    shutil.copytree(elastic_run[0], bad)
    rows = [json.loads(s) for s in (bad / "global_models.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["phase_index"] == 0 and r["round"] == 6)
    row["codes"][5] = row["codes"][5] - 4 if row["codes"][5] > -120 else row["codes"][5] + 4
    (bad / "global_models.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert run([bad, "--output", tmp_path / "fail"]) == 1
    assert capsys.readouterr().out.strip() == "FAIL"
