from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path
import unittest


SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))

from controller import ExperimentController  # noqa: E402
from protocol import (  # noqa: E402
    ALL_GLOBAL_FRAGMENTS as GLOBAL_FRAGMENTS_MASK,
    ProtocolError,
    GLOBAL_FRAGMENTS,
    MSG_CLASS_C_READY,
    MSG_GLOBAL_STATUS,
    MSG_UPDATE_FRAGMENT,
    UPDATE_FRAGMENTS,
    encode_frame,
    fragment_payload,
    expected_update_for_device,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ControllerTests(unittest.TestCase):
    def test_two_nodes_class_a_then_class_c(self) -> None:
        clock = FakeClock()
        published: list[tuple[str, bytes, str]] = []
        events = defaultdict(int)

        def publish(device_id: str, frame: bytes, correlation: str) -> None:
            published.append((device_id, frame, correlation))

        def log(event: str, device_id: str, details: dict) -> None:
            del device_id, details
            events[event] += 1

        controller = ExperimentController(
            ["eui-0000000000000b02", "eui-0000000000000b01"],
            1,
            publish,
            log,
            class_c_guard_seconds=0,
            retry_guard_seconds=0,
            clock=clock,
        )

        for device_id in controller.device_ids:
            update = expected_update_for_device(device_id, 1)
            for frame in fragment_payload(update, MSG_UPDATE_FRAGMENT, 1):
                controller.handle_uplink(device_id, 10, frame)
            ready = encode_frame(MSG_CLASS_C_READY, 1, 0, GLOBAL_FRAGMENTS, bytes((5, 5)))
            controller.handle_uplink(device_id, 10, ready)

        self.assertEqual(events["update_complete"], 2)
        self.assertEqual(events["aggregation_complete"], 1)

        sent_per_device = defaultdict(int)
        while sum(sent_per_device.values()) < GLOBAL_FRAGMENTS * 2:
            before = len(published)
            controller.tick()
            self.assertGreater(len(published), before)
            for device_id, _, _ in published[before:]:
                sent_per_device[device_id] += 1
                controller.handle_downlink_sent(device_id)

        for device_id in controller.device_ids:
            status = encode_frame(
                MSG_GLOBAL_STATUS,
                1,
                0,
                GLOBAL_FRAGMENTS,
                (0).to_bytes(2, "little") + controller.global_crc.to_bytes(4, "little"),
            )
            controller.handle_uplink(device_id, 10, status)

        self.assertTrue(controller.finished)
        self.assertFalse(controller.failed)
        summary = controller.summary()
        self.assertEqual(summary["update_fragments_per_node"], UPDATE_FRAGMENTS)
        self.assertEqual(summary["global_fragments_per_node"], GLOBAL_FRAGMENTS)


class UpdateRepairTests(unittest.TestCase):
    """Reproduces the 2026-08-26 run: one fragment lost per node in flight."""

    DEVICES = ["eui-0000000000000b02", "eui-0000000000000b01"]
    LOST = {"eui-0000000000000b02": 8, "eui-0000000000000b01": 7}

    def setUp(self) -> None:
        self.published: list[tuple[str, bytes, str]] = []
        self.events: list[tuple[str, str, dict]] = []
        self.controller = ExperimentController(
            device_ids=list(self.DEVICES),
            round_id=1,
            publish=lambda d, f, c: self.published.append((d, f, c)),
            log=lambda e, d, x: self.events.append((e, d, x)),
            clock=FakeClock(),
        )

    def _send_update(self, device_id: str, skip: set[int]) -> None:
        payload = expected_update_for_device(device_id, 1)
        for index, frame in enumerate(
            fragment_payload(payload, MSG_UPDATE_FRAGMENT, 1)
        ):
            if index in skip:
                continue
            self.controller.handle_uplink(device_id, 10, frame)

    def _send_ready(self, device_id: str) -> None:
        self.controller.handle_uplink(
            device_id, 10,
            encode_frame(MSG_CLASS_C_READY, 1, 0, GLOBAL_FRAGMENTS, bytes([5, 5])),
        )

    def _names(self) -> list[str]:
        return [event for event, _, _ in self.events]

    def test_repair_request_names_exactly_the_lost_fragment(self) -> None:
        device = self.DEVICES[0]
        self._send_update(device, {self.LOST[device]})
        self._send_ready(device)

        repairs = [e for e in self.events if e[0] == "update_repair_requested"]
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0][2]["missing_fragments"], [self.LOST[device]])
        self.assertEqual(repairs[0][2]["round"], 1)
        # The repair downlink must actually go out, and stay small.
        self.assertEqual(len(self.published), 1)
        self.assertEqual(len(self.published[0][1]), 9)
        # Class C must not start on an incomplete update.
        self.assertNotIn("update_complete", self._names())
        self.assertEqual(self.controller.updates, {})

    def test_resending_the_lost_fragment_completes_the_update(self) -> None:
        device = self.DEVICES[0]
        lost = self.LOST[device]
        self._send_update(device, {lost})
        self._send_ready(device)

        resent = fragment_payload(
            expected_update_for_device(device, 1), MSG_UPDATE_FRAGMENT, 1
        )[lost]
        self.controller.handle_uplink(device, 10, resent)

        self.assertIn("update_complete", self._names())
        self.assertIn(device, self.controller.updates)
        self.assertEqual(len(self.controller.updates[device]), 2202)

    def test_both_nodes_repair_then_class_c_starts(self) -> None:
        for device in self.DEVICES:
            self._send_update(device, {self.LOST[device]})
            self._send_ready(device)
        # Two repair requests, no Class C yet.
        self.assertEqual(self._names().count("update_repair_requested"), 2)
        self.assertEqual(self.controller.deliveries, {})

        for device in self.DEVICES:
            lost = self.LOST[device]
            frame = fragment_payload(
                expected_update_for_device(device, 1), MSG_UPDATE_FRAGMENT, 1
            )[lost]
            self.controller.handle_uplink(device, 10, frame)

        self.assertEqual(self._names().count("update_complete"), 2)
        # Both updates are now complete, but a repaired node has dropped out of
        # Class C to resend and has not yet re-announced. Starting here would
        # transmit into devices that are not listening -- observed on hardware
        # 2026-09-01, where a node received only 2 of 11 global fragments.
        self.assertEqual(
            self.controller.deliveries, {},
            "Class C must not start before every repaired node re-announces",
        )

        for device in self.DEVICES:
            self._send_ready(device)

        self.assertEqual(set(self.controller.deliveries), set(self.DEVICES))
        for delivery in self.controller.deliveries.values():
            self.assertEqual(len(delivery.frames), GLOBAL_FRAGMENTS)

    def test_repair_clears_stale_readiness(self) -> None:
        """A node told to repair must not still count as Class C ready."""
        device = self.DEVICES[0]
        self._send_update(device, {self.LOST[device]})
        self._send_ready(device)
        self.assertEqual(self._names().count("update_repair_requested"), 1)
        self.assertNotIn(
            device, self.controller.ready,
            "the node left Class C to repair; its readiness is stale",
        )

    def test_repair_downlink_bytes_are_counted_as_sent(self) -> None:
        """A repair frame is real airtime; queued and sent must agree."""
        device = self.DEVICES[0]
        self._send_update(device, {self.LOST[device]})
        self._send_ready(device)
        self.assertEqual(len(self.published), 1)
        # TTS confirms the repair frame went out.
        self.controller.handle_downlink_sent(device)
        counters = self.controller.counters[device]
        self.assertEqual(
            counters["downlink_application_bytes_sent"],
            counters["downlink_application_bytes_queued"],
            "the UPDATE_REPAIR frame was transmitted but not counted as sent",
        )
        self.assertEqual(counters["update_repair_downlinks_sent"], 1)

    def test_unknown_device_is_ignored_not_fatal(self) -> None:
        """A board not listed in device_ids must not abort everyone's run."""
        known = self.DEVICES[0]
        self._send_update(known, set())
        stranger = "eui-0000000000000c01"
        frame = fragment_payload(
            expected_update_for_device(stranger, 1), MSG_UPDATE_FRAGMENT, 1
        )[0]
        self.controller.handle_uplink(stranger, 10, frame)

        names = self._names()
        self.assertIn("ignored_unknown_device", names)
        self.assertFalse(self.controller.failed)
        # The known node's progress must be untouched.
        self.assertIn(known, self.controller.updates)
        self.assertNotIn(stranger, self.controller.counters)

    def test_repair_rounds_are_capped(self) -> None:
        device = self.DEVICES[0]
        self._send_update(device, {self.LOST[device]})
        for _ in range(self.controller.max_repair_rounds + 1):
            self._send_ready(device)
        self.assertTrue(self.controller.failed)
        self.assertIn("repair rounds", self.controller.failure_reason)


class ScalingAndSerializationTests(unittest.TestCase):
    """Any node count, and one downlink in flight across the whole gateway."""

    THREE = [
        "eui-0000000000000b02",
        "eui-0000000000000b01",
        "eui-0000000000000b03",
    ]

    def _controller(self, device_ids):
        self.published = []
        self.events = []
        self.clock = FakeClock()
        return ExperimentController(
            device_ids=list(device_ids),
            round_id=1,
            publish=lambda d, f, c: self.published.append((d, f, c)),
            log=lambda e, d, x: self.events.append((e, d, x)),
            class_c_guard_seconds=0,
            retry_guard_seconds=0,
            clock=self.clock,
        )

    def _complete(self, controller, device_id):
        for frame in fragment_payload(
            expected_update_for_device(device_id, 1), MSG_UPDATE_FRAGMENT, 1
        ):
            controller.handle_uplink(device_id, 10, frame)
        controller.handle_uplink(
            device_id, 10,
            encode_frame(MSG_CLASS_C_READY, 1, 0, GLOBAL_FRAGMENTS, bytes((5, 5))),
        )

    def test_single_node_runs(self) -> None:
        c = self._controller(self.THREE[:1])
        self._complete(c, self.THREE[0])
        self.assertEqual(set(c.deliveries), set(self.THREE[:1]))
        self.assertEqual(c.summary()["class_c_unicast_copies"], 1)

    def test_three_nodes_all_get_the_same_aggregate(self) -> None:
        c = self._controller(self.THREE)
        for device_id in self.THREE:
            self._complete(c, device_id)
        self.assertEqual(set(c.deliveries), set(self.THREE))
        self.assertEqual(c.summary()["class_c_unicast_copies"], 3)
        frames = [tuple(d.frames) for d in c.deliveries.values()]
        self.assertEqual(len(set(frames)), 1, "every node must receive identical bytes")

    def test_class_c_waits_for_the_last_node(self) -> None:
        c = self._controller(self.THREE)
        self._complete(c, self.THREE[0])
        self._complete(c, self.THREE[1])
        self.assertEqual(c.deliveries, {}, "must not start with a node outstanding")
        self._complete(c, self.THREE[2])
        self.assertEqual(set(c.deliveries), set(self.THREE))

    def test_only_one_downlink_in_flight_across_all_devices(self) -> None:
        c = self._controller(self.THREE)
        for device_id in self.THREE:
            self._complete(c, device_id)
        before = len(self.published)
        for _ in range(20):
            c.tick()
        self.assertEqual(
            len(self.published) - before, 1,
            "the half-duplex gateway can only carry one downlink at a time",
        )
        first_device = self.published[-1][0]
        c.handle_downlink_sent(first_device)
        for _ in range(20):
            c.tick()
        self.assertEqual(len(self.published) - before, 2)

    def test_devices_are_served_round_robin(self) -> None:
        c = self._controller(self.THREE)
        for device_id in self.THREE:
            self._complete(c, device_id)
        order = []
        for _ in range(6):
            c.tick()
            device_id = self.published[-1][0]
            order.append(device_id)
            c.handle_downlink_sent(device_id)
        self.assertEqual(
            len(set(order[:3])), 3,
            f"first three downlinks went to {order[:3]}; no node may starve the others",
        )
        self.assertEqual(order[:3], order[3:6], "rotation must repeat cleanly")


class MulticastTests(unittest.TestCase):
    """The aggregate is transmitted once for everyone, not once per node."""

    THREE = [
        "eui-0000000000000b02",
        "eui-0000000000000b01",
        "eui-0000000000000b03",
    ]
    GROUP = "tinyfl-v91-global"

    def _controller(self, multicast):
        self.published = []
        self.events = []
        self.clock = FakeClock()
        return ExperimentController(
            device_ids=list(self.THREE),
            round_id=1,
            publish=lambda d, f, c: self.published.append((d, f, c)),
            log=lambda e, d, x: self.events.append((e, d, x)),
            class_c_guard_seconds=0,
            retry_guard_seconds=0,
            multicast_device_id=self.GROUP if multicast else None,
            clock=self.clock,
        )

    def _complete(self, c, device_id):
        for frame in fragment_payload(
            expected_update_for_device(device_id, 1), MSG_UPDATE_FRAGMENT, 1
        ):
            c.handle_uplink(device_id, 10, frame)
        c.handle_uplink(
            device_id, 10,
            encode_frame(MSG_CLASS_C_READY, 1, 0, GLOBAL_FRAGMENTS, bytes((5, 5))),
        )

    def _drain(self, c, limit=200):
        for _ in range(limit):
            before = len(self.published)
            c.tick()
            if len(self.published) == before:
                break
            c.handle_downlink_sent(self.published[-1][0])

    def _status(self, c, device_id, missing):
        crc = c.global_crc if missing == 0 else 0
        body = missing.to_bytes(2, "little") + crc.to_bytes(4, "little")
        c.handle_uplink(
            device_id, 10,
            encode_frame(MSG_GLOBAL_STATUS, 1, 0, GLOBAL_FRAGMENTS, body),
        )

    def test_one_transmission_serves_every_node(self) -> None:
        c = self._controller(multicast=True)
        for d in self.THREE:
            self._complete(c, d)
        self._drain(c)
        self.assertEqual(
            len(self.published), GLOBAL_FRAGMENTS,
            "multicast must send 11 frames total, not 11 per node",
        )
        self.assertEqual({d for d, _, _ in self.published}, {self.GROUP})

    def test_unicast_still_sends_once_per_node(self) -> None:
        c = self._controller(multicast=False)
        for d in self.THREE:
            self._complete(c, d)
        self._drain(c)
        self.assertEqual(len(self.published), GLOBAL_FRAGMENTS * len(self.THREE))

    def test_repair_sends_the_union_once(self) -> None:
        c = self._controller(multicast=True)
        for d in self.THREE:
            self._complete(c, d)
        self._drain(c)
        sent = len(self.published)
        # Different nodes miss different fragments.
        self._status(c, self.THREE[0], 1 << 3)
        self._status(c, self.THREE[1], 1 << 7)
        self._status(c, self.THREE[2], 0)
        self._drain(c)
        self.assertEqual(
            len(self.published) - sent, 2,
            "the union {3,7} must cost two transmissions, not two per node",
        )
        self.assertFalse(c.finished, "two nodes have not acknowledged yet")
        self._status(c, self.THREE[0], 0)
        self._status(c, self.THREE[1], 0)
        self.assertTrue(c.finished)

    def test_repair_downlink_to_a_node_is_accepted_in_multicast_mode(self) -> None:
        """UPDATE_REPAIR goes to one node even when GLOBAL is multicast.

        Reproduces the 2026-09-01 abort: TTS confirmed a repair downlink on the
        node's own topic and the delivery-target guard rejected it, failing the
        whole run with "unexpected downlink target".
        """
        c = self._controller(multicast=True)
        lossy = self.THREE[0]
        # This node loses a fragment, so it gets an UPDATE_REPAIR downlink.
        for index, frame in enumerate(
            fragment_payload(expected_update_for_device(lossy, 1),
                             MSG_UPDATE_FRAGMENT, 1)
        ):
            if index != 4:
                c.handle_uplink(lossy, 10, frame)
        c.handle_uplink(
            lossy, 10,
            encode_frame(MSG_CLASS_C_READY, 1, 0, GLOBAL_FRAGMENTS, bytes((5, 5))),
        )
        self.assertEqual(self.published[-1][0], lossy,
                         "the repair must be addressed to the node, not the group")
        # TTS confirms it on the NODE's topic; this must not raise.
        c.handle_downlink_sent(lossy)
        self.assertEqual(c.counters[lossy]["update_repair_downlinks_sent"], 1)
        # A genuinely unknown target must still be rejected.
        with self.assertRaises(ProtocolError):
            c.handle_downlink_sent("eui-deadbeefdeadbeef")

    def test_waiting_node_reporting_early_is_not_fatal(self) -> None:
        """A finished node idling in Class C while another repairs is normal.

        Reproduces the 2026-09-01 abort: one node completed, entered Class C,
        hit its 120 s idle timeout while the other was still repairing, and
        reported "I have heard nothing". The server raised
        "GLOBAL_STATUS arrived before global delivery started" and failed
        the whole run.
        """
        c = self._controller(multicast=True)
        done, slow = self.THREE[0], self.THREE[1]
        self._complete(c, done)
        # The other node is still mid-upload, so Class C has not started.
        self.assertEqual(c.deliveries, {})

        body = (GLOBAL_FRAGMENTS_MASK).to_bytes(2, "little") + (0).to_bytes(4, "little")
        c.handle_uplink(
            done, 10,
            encode_frame(MSG_GLOBAL_STATUS, 1, 0, GLOBAL_FRAGMENTS, body),
        )
        names = [e for e, _, _ in self.events]
        self.assertIn("global_status_while_waiting", names)
        self.assertFalse(c.failed, "a waiting node must not fail the run")
        # And it must not be mistaken for a broken upload.
        self.assertNotIn("update_repair_requested", names)

    def test_lost_class_c_ready_recovers_via_global_status(self) -> None:
        """A CLASS_C_READY lost in the air must not stall the run forever.

        READY is unconfirmed, so a node whose READY is lost still enters Class
        C and waits. The server never marks it ready, so the phase never
        starts. Its eventual GLOBAL_STATUS is the only other evidence it is
        listening, and must therefore stand in for the lost READY.
        """
        c = self._controller(multicast=True)
        lost, others = self.THREE[0], self.THREE[1:]
        # Its update arrives, but its READY never does.
        for frame in fragment_payload(
            expected_update_for_device(lost, 1), MSG_UPDATE_FRAGMENT, 1
        ):
            c.handle_uplink(lost, 10, frame)
        for device_id in others:
            self._complete(c, device_id)

        self.assertEqual(c.deliveries, {}, "one node never announced readiness")

        # Its Class C timeout fires: nothing heard, everything missing.
        body = GLOBAL_FRAGMENTS_MASK.to_bytes(2, "little") + (0).to_bytes(4, "little")
        c.handle_uplink(
            lost, 10, encode_frame(MSG_GLOBAL_STATUS, 1, 0, GLOBAL_FRAGMENTS, body)
        )

        implied = [x for e, _, x in self.events if e == "global_status_while_waiting"]
        self.assertTrue(implied and implied[-1]["implied_class_c_ready"])
        self.assertEqual(
            set(c.deliveries), {self.GROUP},
            "the implied readiness should have started the Class C phase",
        )

    def test_repair_waits_for_reports_before_re_bursting(self) -> None:
        """One node acking must not trigger a full re-burst.

        Reproduces the 2026-09-01 15:20 run: 884 acked 2 s after the burst,
        and because node_missing starts at ALL the server read the other two
        nodes' silence as total loss and resent all 11 frames. 14b3 then acked
        mid-burst, proving it had been fine. 27 frames sent where ~13 sufficed.
        """
        c = self._controller(multicast=True)
        for device_id in self.THREE:
            self._complete(c, device_id)
        self._drain(c)
        after_first_burst = len(self.published)
        self.assertEqual(after_first_burst, GLOBAL_FRAGMENTS)

        # The fast node acks; the other two have not spoken yet.
        self._status(c, self.THREE[0], 0)
        self._drain(c)
        self.assertEqual(
            len(self.published), after_first_burst,
            "must not re-burst while nodes have yet to report",
        )
        self.assertIn("multicast_awaiting_reports",
                      [e for e, _, _ in self.events])

        # Now they speak: one was fine, one missed a single fragment.
        self._status(c, self.THREE[1], 0)
        self._status(c, self.THREE[2], 1 << 3)
        self._drain(c)
        self.assertEqual(
            len(self.published) - after_first_burst, 1,
            "the union is just fragment 3, so exactly one frame should go out",
        )

    def test_summary_reports_multicast(self) -> None:
        c = self._controller(multicast=True)
        for d in self.THREE:
            self._complete(c, d)
        summary = c.summary()
        self.assertEqual(summary["class_c_mode"], "multicast")
        self.assertEqual(summary["class_c_unicast_copies"], 1)
        self.assertEqual(summary["class_c_group_id"], self.GROUP)


if __name__ == "__main__":
    unittest.main()
