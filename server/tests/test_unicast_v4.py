"""Rev 4.3 testbed rerun fixes: prompt GLOBAL repair limits (PROTOCOL_V4 section 4) and the unicast
queue lifecycle (UPDATE_REPAIR by down/replace, cleared at round close, down/sent matched by
correlation id), plus the testbed fake network server's down/replace."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import config_v4 as cv
import protocol_v4 as p4
from controller_v4 import ControllerParams, Stage
from plan_v4 import elastic_plan
from v4_sim import RUN_ID, Sim, dev_eui_for, device_id_for

DEAF_FRAGMENTS = (1, 2)


def deaf_board_2(board, slot, fragment, count):
    """Board 2 hears fragment 0 of every GLOBAL after the first and nothing else: it keeps sending
    GLOBAL_STATUS (missing 0b110) and never hears a repair (the firmware bug of the testbed rerun)."""
    return board == 2 and slot >= 2 and fragment in DEAF_FRAGMENTS


def deaf_sim(**kw):
    kw.setdefault("idle_timeout", 10.0)
    kw.setdefault("multicast_report_window_seconds", 180.0)
    return Sim(elastic_plan(17070, 1, 4), boards={1: (0,), 2: (1,), 3: (2,)}, downlink_drop=deaf_board_2, **kw)


# ---------------------------------------------------------------------------
# 1. prompt repair: one per board per min gap, counted, periodic timer not postponed
# ---------------------------------------------------------------------------
def test_a_status_storm_from_a_deaf_board_cannot_starve_the_round_and_the_run_progresses():
    sim = deaf_sim()
    sim.run_to_done(max_steps=6000)
    c = sim.controller
    assert c.stage is Stage.DONE and not c.failed
    ends = [x for e, _, x in sim.events if e == "global_delivery_end"]
    assert [x["gave_up"] for x in ends if x["slot"] >= 2][0] is True       # the deaf board is given up on
    assert len(sim.round_records(0)) == 4                                   # every round closed
    assert c.counters["global_prompt_repairs_throttled"] > 0                # the STATUS storm was throttled


def test_at_most_one_prompt_repair_per_board_per_min_gap():
    sim = deaf_sim(prompt_repair_min_gap_seconds=30.0)
    sim.run_to_done(max_steps=6000)
    by_slot: dict = {}
    for (e, _, x), t in zip(sim.events, sim.event_t):
        if e == "multicast_prompt_repair":
            by_slot.setdefault(x["slot"], []).append(t)
    assert by_slot
    for slot, times in by_slot.items():
        assert all(b - a >= 29.0 for a, b in zip(times, times[1:])), (slot, times)


def test_prompt_repairs_count_toward_max_global_repair_rounds():
    sim = deaf_sim(max_global_repair_rounds=2, prompt_repair_min_gap_seconds=15.0)
    sim.run_to_done(max_steps=6000)
    assert sim.controller.stage is Stage.DONE
    per_slot: dict = {}
    for e, _, x in sim.events:
        if e == "multicast_prompt_repair":
            per_slot[x["slot"]] = per_slot.get(x["slot"], 0) + 1
        if e == "multicast_repair":
            per_slot[x["round"] + 1] = per_slot.get(x["round"] + 1, 0) + 1   # round r is slot r + 1 here
    assert per_slot and all(n <= 2 for n in per_slot.values()), per_slot
    assert sim.controller.counters["global_prompt_repairs_exhausted"] > 0


def test_prompt_repairs_never_postpone_the_periodic_repair_timer():
    """The periodic timer runs from the end of the first (non-prompt) burst: the round gives up on the
    deaf board no later than one report window (plus the guard) after that, however many prompts ran."""
    window = 120.0
    sim = deaf_sim(multicast_report_window_seconds=window, max_global_repair_rounds=4)
    sim.run_to_done(max_steps=6000)
    first_sent = {}
    gave_up = {}
    for (e, _, x), t in zip(sim.events, sim.event_t):
        if e == "global_multicast_starting":
            first_sent[x["slot"]] = t
        if e == "global_delivery_end" and x["gave_up"]:
            gave_up[x["slot"]] = t
    assert gave_up
    for slot, t in gave_up.items():
        assert t - first_sent[slot] <= window + 60.0, (slot, t - first_sent[slot])


def test_a_board_that_keeps_reporting_and_never_hears_is_abandoned_per_the_existing_rules():
    """Deaf in every round: it is given up on each round, misses are counted, and it is eventually
    SUSPECT / DEPARTED; the other boards' rounds close and the run finishes."""
    def deaf(board, slot, fragment, count):
        return board == 2 and slot >= 2 and fragment in DEAF_FRAGMENTS

    sim = Sim(elastic_plan(17070, 1, 8), boards={1: (0,), 2: (1,), 3: (2,)}, downlink_drop=deaf,
              idle_timeout=10.0, multicast_report_window_seconds=120.0, suspect_after_rounds=1,
              depart_after_rounds=2)
    sim.run_to_done(max_steps=20000)
    c = sim.controller
    assert c.stage is Stage.DONE
    states = [x["to"] for ev, d, x in sim.events if ev == "membership" and d == dev_eui_for(2)]
    assert "SUSPECT" in states or "DEPARTED" in states
    assert all(0 in r["arrived_positions"] and 2 in r["arrived_positions"] for r in sim.round_records(0))


def test_the_old_unthrottled_behaviour_is_what_the_min_gap_of_zero_gives():
    sim = deaf_sim(prompt_repair_min_gap_seconds=0.0, max_global_repair_rounds=1000)
    sim.start()
    sim.run_until(lambda: sim.controller.counters["global_prompt_repairs"] >= 10, max_steps=3000)
    assert sim.controller.counters["global_prompt_repairs_throttled"] == 0


def test_min_gap_config_key_is_validated_and_defaults_to_30():
    assert ControllerParams().prompt_repair_min_gap_seconds == 30.0
    assert "prompt_repair_min_gap_seconds" in cv.V4_KEYS


# ---------------------------------------------------------------------------
# 2. unicast queue lifecycle
# ---------------------------------------------------------------------------
def lossy_updates(board, position, fragment, attempt, phase, round_):
    return fragment == 2 and attempt <= 2 and position in (0, 1)           # repairs are needed (and re-lost)


def unicast_not_sent(device_id, frame, n):
    return device_id != "tinyfl-v4-global"                                  # TTS never reports repairs as sent


def repair_sim(**kw):
    return Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,), 3: (2,)}, uplink_drop=lossy_updates,
               sent_drop=unicast_not_sent, **kw)


def repair_frames(sim):
    return [(d, c) for d, f, c in sim.published
            if d != "tinyfl-v4-global" and p4.decode_frame(f).message_type == p4.MSG_UPDATE_REPAIR]


def test_update_repairs_go_out_by_replace_never_by_push():
    sim = repair_sim()
    sim.run_to_done()
    repairs = repair_frames(sim)
    assert repairs and sorted(c for _, c in repairs) == sorted(c for _, d, c in sim.replaced)
    assert sim.controller.stage is Stage.DONE


def test_a_round_close_clears_every_device_that_still_has_a_pending_repair():
    sim = repair_sim()
    sim.run_to_done()
    assert sim.cleared                                                       # nobody reported them sent
    replaced_devices = {d for _, d, _ in sim.replaced}
    assert {d for _, d in sim.cleared} <= replaced_devices
    # one clear per (round, device) that had a pending repair: not for devices that never had one
    never = {device_id_for(3)}
    assert not (never & {d for _, d in sim.cleared})
    closes = sim.event_times("round_closing")
    for t, device in sim.cleared:
        assert any(abs(t - tc) < 1e-6 for tc in closes), (t, closes)
    assert sim.controller.counters["update_repair_queue_cleared"] == len(sim.cleared)


def test_a_repair_confirmed_sent_is_not_cleared_later():
    def lossy(board, position, fragment, attempt, phase, round_):
        return fragment == 2 and attempt == 1 and position == 0

    sim = Sim(elastic_plan(17070, 1, 3), boards={1: (0,), 2: (1,), 3: (2,)}, uplink_drop=lossy)
    sim.run_to_done()
    assert sim.replaced and sim.cleared == []                                # down/sent arrived: nothing pending


def test_an_in_flight_repair_is_cancelled_at_round_close_and_the_next_downlink_is_not_blocked():
    sim = repair_sim()
    sim.start()
    sim.run_until(lambda: sim.replaced, max_steps=4000)
    c = sim.controller
    assert c._inflight is not None and c._inflight.kind == "repair"
    c._close_round(sim.clock.now, "test")
    assert c._inflight is None and not [o for o in c._outbox if o.kind == "repair"]
    assert sim.cleared


def test_down_sent_is_matched_by_correlation_id_not_by_device():
    sim = repair_sim()
    sim.start()
    sim.run_until(lambda: sim.replaced, max_steps=4000)
    c = sim.controller
    out = c._inflight
    assert out.kind == "repair"
    device, position = out.device_id, out.position
    before = c.counters["downlink_sent_unmatched"]
    c.handle_downlink_sent(device, ["v4-sOLD-p9-repair1"])                  # a delayed event of an older repair
    assert c._inflight is out and c.counters["downlink_sent_unmatched"] == before + 1
    assert not [x for e, _, x in sim.events if e == "update_repair_sent"]
    c.handle_downlink_sent(device, ["as:downlink:xyz", out.correlation])
    sent = [x for e, _, x in sim.events if e == "update_repair_sent"]
    assert sent == [{"position": position, "application_bytes": len(out.frame)}]
    assert c._inflight is None and device not in c._repair_pending


def test_down_failed_is_matched_by_correlation_id_too():
    sim = repair_sim()
    sim.start()
    sim.run_until(lambda: sim.replaced, max_steps=4000)
    c = sim.controller
    out = c._inflight
    c.handle_downlink_failed(out.device_id, "{}", ["something-else"])
    assert c._inflight is out and c.counters["downlink_failed_unmatched"] == 1
    c.handle_downlink_failed(out.device_id, "{}", [out.correlation])
    assert c._inflight is None or c._inflight is not out


def test_a_requeued_repair_of_a_closed_round_is_dropped_not_resent():
    sim = repair_sim(inflight_downlink_timeout_seconds=20.0)
    sim.start()
    sim.run_until(lambda: sim.replaced, max_steps=4000)
    c = sim.controller
    out = c._inflight
    c._u.closed = True                                                       # the round ended meanwhile
    c._requeue(out, sim.clock.now)
    assert out not in c._outbox
    assert "update_repair_obsolete" in sim.names()


# ---------------------------------------------------------------------------
# TTS topic and payload for replace / clear
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.messages.append((topic, json.loads(payload)))
        return types.SimpleNamespace(rc=0)


def bridge(tmp_path):
    import test_ops_v4 as ops
    cfg = cv.parse_config(ops.config_dict(tmp_path), tmp_path)
    server = ops.make_server(cfg, tmp_path / "run")
    server.client = Recorder()
    return server


def test_tts_replace_and_clear_topic_and_payload(tmp_path):
    server = bridge(tmp_path)
    frame = p4.encode_update_repair(5, RUN_ID, 0, 0b101)
    server._publish_replace("dev-1", frame, "corr-1")
    server._publish_replace("dev-1", None, "")
    server._publish_downlink("dev-1", frame, "corr-2")
    (t1, b1), (t2, b2), (t3, b3) = server.client.messages
    root = server.topic_root
    assert t1 == f"{root}/devices/dev-1/down/replace" == t2
    assert t3 == f"{root}/devices/dev-1/down/push"
    assert list(b1) == ["downlinks"] and len(b1["downlinks"]) == 1
    d = b1["downlinks"][0]
    assert d["f_port"] == 10 and d["confirmed"] is False and d["priority"] == "NORMAL"
    assert d["correlation_ids"] == ["corr-1"] and "class_b_c" not in d
    import base64
    assert base64.b64decode(d["frm_payload"]) == frame
    assert b2 == {"downlinks": []}                                           # the clear
    assert b3["downlinks"][0]["correlation_ids"] == ["corr-2"]               # push unchanged


def test_tts_events_hand_the_correlation_ids_to_the_controller(tmp_path):
    server = bridge(tmp_path)
    seen = []
    server.controller.handle_downlink_sent = lambda dev, ids=None: seen.append(("sent", dev, ids))
    server.controller.handle_downlink_failed = lambda dev, reason, ids=None: seen.append(("failed", dev, ids))
    root = server.topic_root

    def deliver(suffix, event):
        server._on_message(None, None, types.SimpleNamespace(
            topic=f"{root}/devices/dev-9/{suffix}", payload=json.dumps(event).encode()))
    deliver("down/sent", {"correlation_ids": ["as:1", "c1"], "downlink_sent": {"correlation_ids": ["c1"]}})
    deliver("down/failed", {"downlink_failed": {"downlink": {"correlation_ids": ["c2"]}, "error": {}}})
    deliver("down/sent", {})
    assert seen[0] == ("sent", "dev-9", ("as:1", "c1", "c1"))
    assert seen[1] == ("failed", "dev-9", ("c2",))
    assert seen[2] == ("sent", "dev-9", ())                                  # names nothing: matches nothing


# ---------------------------------------------------------------------------
# 3. the testbed fake network server implements down/replace
# ---------------------------------------------------------------------------
TESTBED = Path(__file__).resolve().parents[2] / "testbed"


def load_fake_client():
    sys.path.insert(0, str(TESTBED))
    try:
        import simulator
    except Exception as exc:                                                 # pragma: no cover
        pytest.skip(f"testbed simulator not importable here: {exc}")
    finally:
        sys.path.remove(str(TESTBED))
    return simulator.FakeClient


def test_testbed_fake_network_server_replaces_and_clears_the_device_fifo():
    FakeClient = load_fake_client()
    from collections import deque
    queued = []
    sim = types.SimpleNamespace(
        app="app", group="grp", server_epoch=0,
        boards={1: {"device": "dev-a"}, 2: {"device": "dev-b"}},
        radio=types.SimpleNamespace(unicast={1: deque(), 2: deque()},
                                    publish=lambda dev, frame, corr, epoch: (
                                        queued.append((dev, corr)),
                                        sim.radio.unicast[1 if dev == "dev-a" else 2].append((dev, frame, corr, epoch)))))
    client = FakeClient(sim)

    def msg(device, mode, ids):
        import base64
        downs = [{"f_port": 10, "frm_payload": base64.b64encode(b"x").decode(), "confirmed": False,
                  "priority": "NORMAL", "correlation_ids": [i]} for i in ids]
        client.publish(f"v3/app/devices/{device}/down/{mode}", json.dumps({"downlinks": downs}))

    msg("dev-a", "push", ["p1"])
    msg("dev-a", "push", ["p2"])                                             # push appends: the stale pile-up
    msg("dev-b", "push", ["b1"])
    assert [i[2] for i in sim.radio.unicast[1]] == ["p1", "p2"]
    msg("dev-a", "replace", ["r1"])                                          # replace swaps the device FIFO
    assert [i[2] for i in sim.radio.unicast[1]] == ["r1"]
    assert [i[2] for i in sim.radio.unicast[2]] == ["b1"]                    # other devices untouched
    msg("dev-a", "replace", [])                                              # an empty replace clears it
    assert list(sim.radio.unicast[1]) == [] and len(sim.radio.unicast[2]) == 1
