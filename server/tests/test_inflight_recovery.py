"""Downlink inflight-slot recovery.

Both cases here come from the same defect class: the single inflight slot
being claimed and never released. Before the fix, tick() published the frame
and only then set delivery.inflight, and nothing ever timed the slot out. A
down/sent that arrived early was discarded because inflight was still None,
and a down/sent lost to a broker disconnect pinned the slot for good -- every
later tick() returned at the "one downlink in flight" guard, the configured
retry budget went unspent, and the run failed at the overall timeout having
sent nothing further.

That is the mechanism behind runs that stall mid-burst, so these tests assert
recovery rather than the failure being merely logged.
"""
from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path
import unittest


SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))

from controller import ExperimentController  # noqa: E402
from protocol import (  # noqa: E402
    GLOBAL_FRAGMENTS,
    MSG_CLASS_C_READY,
    MSG_UPDATE_FRAGMENT,
    encode_frame,
    fragment_payload,
    expected_update_for_device,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


DEVICES = ["eui-0000000000000b02", "eui-0000000000000b01"]


def build(clock, publish, *, inflight_timeout=45.0):
    events = defaultdict(int)

    def log(event: str, device_id: str, details: dict) -> None:
        del device_id, details
        events[event] += 1

    controller = ExperimentController(
        list(DEVICES),
        1,
        publish,
        log,
        class_c_guard_seconds=0,
        retry_guard_seconds=0,
        inflight_downlink_timeout_seconds=inflight_timeout,
        clock=clock,
    )
    for device_id in controller.device_ids:
        update = expected_update_for_device(device_id, 1)
        for frame in fragment_payload(update, MSG_UPDATE_FRAGMENT, 1):
            controller.handle_uplink(device_id, 10, frame)
        controller.handle_uplink(
            device_id,
            10,
            encode_frame(MSG_CLASS_C_READY, 1, 0, GLOBAL_FRAGMENTS, bytes((5, 5))),
        )
    return controller, events


class InflightRecoveryTests(unittest.TestCase):
    def test_lost_downlink_status_does_not_wedge_the_run(self) -> None:
        """No down/sent, no down/failed: the slot must still come back."""
        clock = FakeClock()
        published: list[tuple[str, bytes, str]] = []
        controller, events = build(
            clock, lambda d, f, c: published.append((d, f, c)), inflight_timeout=30.0
        )

        controller.tick()
        self.assertEqual(len(published), 1)
        stalled_device = published[0][0]
        # The status event is simply dropped, as QoS 0 permits.

        # While the deadline holds, the slot is legitimately busy.
        clock.now += 29.0
        controller.tick()
        self.assertEqual(len(published), 1)

        # Past the deadline the slot is released and sending resumes. The
        # round-robin cursor has already moved on, so the next frame goes to
        # the other node; the stalled fragment is back in its own queue.
        clock.now += 2.0
        controller.tick()
        self.assertEqual(len(published), 2)
        self.assertEqual(events["downlink_status_timeout"], 1)
        stalled = controller.deliveries[stalled_device]
        self.assertIsNone(stalled.inflight)
        self.assertEqual(stalled.pending[0], 0)

        # And the run goes on to complete rather than dying at the timeout.
        delivered = 0
        guard = 0
        while delivered < GLOBAL_FRAGMENTS * len(DEVICES) and guard < 500:
            guard += 1
            before = len(published)
            controller.tick()
            for device_id, _, _ in published[before:]:
                delivered += 1
                controller.handle_downlink_sent(device_id)
            clock.now += 1.0
        self.assertEqual(delivered, GLOBAL_FRAGMENTS * len(DEVICES))

    def test_timeout_requeue_respects_the_attempt_limit(self) -> None:
        """A frame nobody ever confirms must fail the run, not loop forever."""
        clock = FakeClock()
        published: list[tuple[str, bytes, str]] = []
        controller, _ = build(
            clock, lambda d, f, c: published.append((d, f, c)), inflight_timeout=10.0
        )

        for _ in range(200):
            controller.tick()
            clock.now += 11.0
            if controller.failed:
                break
        self.assertTrue(controller.failed)
        self.assertIn("retry limit", controller.failure_reason)

    def test_status_arriving_during_publish_is_not_lost(self) -> None:
        """The callback runs while publish() is still on the stack.

        This is the real ordering on the wire: paho delivers down/sent on its
        network thread the moment the broker answers. With the registration
        after publish() the callback saw inflight None, dropped itself, and
        tick() then pinned a slot nothing would ever clear.
        """
        clock = FakeClock()
        published: list[tuple[str, bytes, str]] = []
        controller: ExperimentController | None = None

        def publish(device_id: str, frame: bytes, correlation: str) -> None:
            published.append((device_id, frame, correlation))
            # Reentrant: the lock is already held by tick() on this thread,
            # which is exactly what serialises the real network thread.
            controller.handle_downlink_sent(device_id)

        controller, _ = build(clock, publish)

        controller.tick()
        self.assertEqual(len(published), 1)
        device_id = published[0][0]
        self.assertIsNone(controller.deliveries[device_id].inflight)
        self.assertEqual(
            controller.counters[device_id]["class_c_downlinks_sent"], 1
        )

        # The next tick proceeds instead of finding the slot occupied.
        controller.tick()
        self.assertEqual(len(published), 2)

    def test_publish_failure_releases_the_slot_and_the_attempt(self) -> None:
        """An immediate MQTT error must not consume the retry budget."""
        clock = FakeClock()
        calls: list[str] = []

        def publish(device_id: str, frame: bytes, correlation: str) -> None:
            calls.append(device_id)
            raise RuntimeError("MQTT publish failed immediately with code 4")

        controller, _ = build(clock, publish)

        with self.assertRaises(RuntimeError):
            controller.tick()
        self.assertEqual(len(calls), 1)

        delivery = controller.deliveries[calls[0]]
        self.assertIsNone(delivery.inflight)
        self.assertIsNone(delivery.inflight_deadline)
        # Fragment back at the head of the queue, attempt count rolled back.
        self.assertEqual(delivery.pending[0], 0)
        self.assertEqual(delivery.attempts[0], 0)


if __name__ == "__main__":
    unittest.main()
