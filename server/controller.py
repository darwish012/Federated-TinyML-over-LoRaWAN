"""Pure experiment state machine, independent of MQTT."""

from __future__ import annotations

from collections import Counter, deque
import functools
import json
from dataclasses import dataclass, field
import threading
import time
from typing import Callable

from protocol import (
    ALL_GLOBAL_FRAGMENTS,
    F_PORT,
    GLOBAL_BYTES,
    GLOBAL_FRAGMENTS,
    MSG_CLASS_C_READY,
    MSG_GLOBAL_FRAGMENT,
    MSG_GLOBAL_STATUS,
    MSG_UPDATE_FRAGMENT,
    MSG_UPDATE_REPAIR,
    UPDATE_BYTES,
    UPDATE_FRAGMENTS,
    Frame,
    ProtocolError,
    Reassembler,
    decode_frame,
    encode_repair_request,
    expected_update_for_device,
    fragment_payload,
    make_global_payload,
    validate_crc,
)


EventLogger = Callable[[str, str, dict], None]
Publisher = Callable[[str, bytes, str], None]


def _synchronized(method):
    """Serialise an entry point against the MQTT network thread.

    paho runs on_message on its own thread (loop_start), while tick() runs on
    the main thread. Both mutate Delivery state, so every entry point that can
    be reached from either side takes the same reentrant lock.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Delivery:
    frames: list[bytes]
    pending: deque[int] = field(default_factory=deque)
    inflight: int | None = None
    # Wall-clock deadline for the inflight fragment. Without it, a down/sent or
    # down/failed event lost to a broker disconnect leaves inflight set for
    # good: every later tick() returns early, the configured retry budget is
    # never spent, and the run dies at the overall timeout having sent nothing.
    inflight_deadline: float | None = None
    attempts: Counter = field(default_factory=Counter)
    next_send_at: float = 0.0
    acknowledged: bool = False
    failed: bool = False


class ExperimentController:
    def __init__(
        self,
        device_ids: list[str],
        round_id: int,
        publish: Publisher,
        log: EventLogger,
        class_c_guard_seconds: float = 8.0,
        retry_guard_seconds: float = 8.0,
        max_downlink_attempts: int = 3,
        max_repair_rounds: int = 4,
        multicast_report_window_seconds: float = 180.0,
        multicast_device_id: str | None = None,
        inflight_downlink_timeout_seconds: float = 45.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not device_ids:
            raise ValueError("at least one device ID is required")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("device IDs must be unique")
        if not 0 <= round_id <= 255:
            raise ValueError("round_id must fit in one byte")
        self.device_ids = tuple(device_ids)
        self.round_id = round_id
        self.publish = publish
        self.log = log
        self.class_c_guard_seconds = class_c_guard_seconds
        self.retry_guard_seconds = retry_guard_seconds
        self.max_downlink_attempts = max_downlink_attempts
        self.max_repair_rounds = max_repair_rounds
        # How long to let nodes report after a multicast burst before
        # deciding what to repair. Must exceed the node's Class C idle
        # timeout (120 s), because a node missing fragments only speaks
        # once that fires -- a complete node acks within seconds.
        self.multicast_report_window_seconds = multicast_report_window_seconds
        # When set, the aggregate is transmitted ONCE to this multicast
        # group instead of once per node. Nodes still acknowledge
        # individually over their own unicast uplinks.
        self.multicast_device_id = multicast_device_id or None
        # down/sent and down/failed arrive at QoS 0, so either can simply be
        # lost. This bounds how long one fragment may hold the single inflight
        # slot before it is requeued and retried on the normal budget.
        self.inflight_downlink_timeout_seconds = inflight_downlink_timeout_seconds
        self.clock = clock
        # Guards every mutation below against the paho network thread.
        self._lock = threading.RLock()

        self.reassemblers = {
            device_id: Reassembler(
                MSG_UPDATE_FRAGMENT,
                round_id,
                UPDATE_FRAGMENTS,
                UPDATE_BYTES,
            )
            for device_id in self.device_ids
        }
        self.expected_updates = {
            device_id: expected_update_for_device(device_id, round_id)
            for device_id in self.device_ids
        }
        self.updates: dict[str, bytes] = {}
        self.ready: set[str] = set()
        self.repair_rounds: Counter = Counter()
        # Per-node GLOBAL fragments still outstanding. In multicast mode the
        # server retransmits the UNION of these, since one transmission has
        # to satisfy every node that is still missing something.
        self.node_missing: dict[str, int] = {}
        # UPDATE_REPAIR frames published but not yet confirmed sent. They are
        # real airtime and must land in downlink_application_bytes_sent, but
        # they are not part of any Delivery, so handle_downlink_sent() would
        # otherwise drop them and undercount the measured transport cost.
        self.pending_repair_bytes: dict[str, deque[int]] = {}
        self._multicast_rounds = 0
        # Who has reported since the last burst, and when that burst ended.
        # node_missing starts at ALL, so without tracking this a node that
        # simply has not spoken yet is treated as missing everything and
        # inflates the repair union to the full set.
        self._reported_since_burst: set[str] = set()
        self._burst_ended_at: float | None = None
        self.deliveries: dict[str, Delivery] = {}
        self._downlink_cursor = 0
        self.global_payload: bytes | None = None
        self.global_crc: int | None = None
        # The multicast group gets its own counter row: its downlink totals are
        # the measurement that justifies multicast in the first place.
        self.counters: dict[str, Counter] = {
            device_id: Counter() for device_id in self.device_ids
        }
        if self.multicast_device_id is not None:
            self.counters[self.multicast_device_id] = Counter()
        self.failure_reason = ""

    @property
    @_synchronized
    def finished(self) -> bool:
        if not self.deliveries:
            return False
        if self.multicast:
            # One shared transmission, but every node must still confirm.
            return bool(self.node_missing) and all(
                mask == 0 for mask in self.node_missing.values()
            )
        return all(delivery.acknowledged for delivery in self.deliveries.values())

    @property
    @_synchronized
    def failed(self) -> bool:
        return bool(self.failure_reason) or any(
            delivery.failed for delivery in self.deliveries.values()
        )

    def _require_device(self, device_id: str) -> None:
        if device_id not in self.device_ids:
            raise ProtocolError(f"unexpected device ID {device_id!r}")

    @property
    def multicast(self) -> bool:
        return self.multicast_device_id is not None

    def _require_delivery_target(self, device_id: str) -> None:
        # Two different kinds of downlink are confirmed through here. GLOBAL
        # fragments go to _delivery_targets() -- the multicast group, or every
        # node when unicast. But UPDATE_REPAIR is always addressed to ONE node
        # even in multicast mode, because it repairs that node's own Class A
        # upload rather than the shared global model. Accepting only the group
        # made TTS's down/sent for a repair abort the whole run.
        if device_id in self._delivery_targets() or device_id in self.device_ids:
            return
        raise ProtocolError(f"unexpected downlink target {device_id!r}")

    @staticmethod
    def _clear_inflight(delivery: Delivery) -> None:
        """Release the single inflight slot and its deadline together."""
        delivery.inflight = None
        delivery.inflight_deadline = None

    def _expire_inflight_downlinks(self, now: float) -> None:
        """Requeue any fragment whose status event never arrived.

        A lost down/sent or down/failed would otherwise pin the one inflight
        slot for the rest of the run. The fragment goes back to the head of
        the queue and is retried under the existing per-fragment attempt
        limit, so a genuinely undeliverable frame still fails the run rather
        than looping.
        """
        for device_id, delivery in self.deliveries.items():
            if delivery.inflight is None or delivery.inflight_deadline is None:
                continue
            if now < delivery.inflight_deadline:
                continue
            index = delivery.inflight
            self._clear_inflight(delivery)
            delivery.pending.appendleft(index)
            delivery.next_send_at = now + self.retry_guard_seconds
            self.log(
                "downlink_status_timeout",
                device_id,
                {
                    "fragment": index,
                    "attempt": delivery.attempts[index],
                    "timeout_seconds": self.inflight_downlink_timeout_seconds,
                },
            )

    def _delivery_targets(self) -> tuple[str, ...]:
        """Who downlinks are addressed to: one group, or every node."""
        if self.multicast:
            return (self.multicast_device_id,)
        return self.device_ids

    @_synchronized
    def handle_uplink(self, device_id: str, f_port: int, payload: bytes) -> None:
        if device_id not in self.device_ids:
            # A board flashed with the same firmware but not listed in
            # device_ids. Almost always a forgotten config entry, and killing
            # the run over it discards the Class A phase already in flight for
            # every other node. Say so loudly and carry on.
            self.log(
                "ignored_unknown_device",
                device_id,
                {"hint": "add this id to experiment.device_ids if it belongs "
                         "to the run; it is being ignored"},
            )
            return
        if f_port != F_PORT:
            self.log("ignored_wrong_fport", device_id, {"f_port": f_port})
            return
        frame = decode_frame(payload)
        if frame.round_id != self.round_id:
            raise ProtocolError(
                f"device {device_id} sent round {frame.round_id}; expected {self.round_id}"
            )
        self.counters[device_id]["class_a_uplinks"] += 1
        self.counters[device_id]["uplink_application_bytes"] += len(payload)

        if frame.message_type == MSG_UPDATE_FRAGMENT:
            self._handle_update(device_id, frame)
        elif frame.message_type == MSG_CLASS_C_READY:
            self._handle_ready(device_id, frame)
        elif frame.message_type == MSG_GLOBAL_STATUS:
            self._handle_status(device_id, frame)
        elif frame.message_type == MSG_UPDATE_REPAIR:
            raise ProtocolError("UPDATE_REPAIR is a downlink and cannot arrive as an uplink")
        else:
            raise ProtocolError("a GLOBAL fragment cannot arrive as an uplink")

    def _handle_update(self, device_id: str, frame: Frame) -> None:
        self.counters[device_id]["update_fragments_received"] += 1
        result = self.reassemblers[device_id].add(frame)
        self.log(
            "update_fragment",
            device_id,
            {
                "fragment": frame.fragment_index,
                "count": frame.fragment_count,
                "body_bytes": len(frame.data),
            },
        )
        if result is None:
            return
        crc = validate_crc(result, UPDATE_BYTES)
        expected = self.expected_updates[device_id]
        if result != expected:
            # Self-consistent CRC but the wrong bytes: the firmware and the
            # server disagree about how the payload is seeded from the DevEUI.
            first = next(
                i for i in range(len(expected)) if result[i:i + 1] != expected[i:i + 1]
            )
            self.failure_reason = (
                f"{device_id} reassembled a valid-CRC update that does not match "
                f"its DevEUI-derived payload (first difference at byte {first})"
            )
            self.log(
                "update_payload_mismatch",
                device_id,
                {"first_differing_byte": first, "crc32": f"{crc:08X}"},
            )
            return
        self.updates[device_id] = result
        self.log("update_complete", device_id, {"bytes": len(result), "crc32": f"{crc:08X}"})
        self._start_if_ready()

    def _handle_ready(self, device_id: str, frame: Frame) -> None:
        if frame.fragment_index != 0 or frame.fragment_count != GLOBAL_FRAGMENTS:
            raise ProtocolError("CLASS_C_READY declared the wrong GLOBAL fragment count")
        if len(frame.data) != 2:
            raise ProtocolError("CLASS_C_READY must report uplink DR and RX2 DR")
        uplink_dr, rx2_dr = frame.data
        if rx2_dr not in (4, 5):
            raise ProtocolError(
                f"{device_id} reported RX2 DR{rx2_dr}; 221-byte Class-C frames need DR4/DR5"
            )
        self.ready.add(device_id)
        self.log("class_c_ready", device_id, {"uplink_dr": uplink_dr, "rx2_dr": rx2_dr})
        # The node believes it sent all 11 fragments, but unconfirmed uplinks
        # have no retransmission, so some may never have reached us. Ask for
        # exactly those before the Class C phase can start.
        missing = self.reassemblers[device_id].missing_mask()
        if missing:
            self._request_update_repair(device_id, missing)
            return
        self._start_if_ready()

    def _request_update_repair(self, device_id: str, missing: int) -> None:
        # Receiving a repair request makes the node call LMIC_enableClassC(0)
        # and drop back to Class A to resend, only re-entering Class C after a
        # second CLASS_C_READY. Its earlier readiness is therefore stale: if we
        # keep it, _start_if_ready() can open the Class C phase while this node
        # is still repairing and not listening on RX2, and every fragment sent
        # before its second READY is transmitted into a device that cannot hear
        # it. Observed on hardware 2026-09-01: a node received only fragments
        # 9 and 10 of 11 for exactly this reason.
        self.ready.discard(device_id)
        self.repair_rounds[device_id] += 1
        rounds = self.repair_rounds[device_id]
        if rounds > self.max_repair_rounds:
            self.failure_reason = (
                f"{device_id} still missing UPDATE fragments after "
                f"{self.max_repair_rounds} repair rounds"
            )
            self.log("experiment_failed", device_id, {"reason": self.failure_reason})
            return
        frame = encode_repair_request(self.round_id, missing)
        correlation = f"v91-hw-r{self.round_id}-{device_id}-repair{rounds}"
        self.publish(device_id, frame, correlation)
        self.pending_repair_bytes.setdefault(device_id, deque()).append(len(frame))
        self.counters[device_id]["update_repairs_requested"] += 1
        self.counters[device_id]["downlink_application_bytes_queued"] += len(frame)
        self.log(
            "update_repair_requested",
            device_id,
            {
                "round": rounds,
                "missing_mask": f"{missing:04X}",
                "missing_fragments": [
                    index
                    for index in range(UPDATE_FRAGMENTS)
                    if missing >> index & 1
                ],
                "application_bytes": len(frame),
                "correlation_id": correlation,
            },
        )

    def _refresh_multicast_pending(self) -> None:
        """Queue the union of what any node still needs; one send serves all."""
        delivery = self.deliveries[self.multicast_device_id]
        union = 0
        for mask in self.node_missing.values():
            union |= mask
        if union == 0:
            delivery.pending.clear()
            self._clear_inflight(delivery)
            delivery.acknowledged = True
            return
        outstanding = set(self.device_ids) - self._reported_since_burst
        window_open = (
            self._burst_ended_at is not None
            and self.clock() - self._burst_ended_at
            < self.multicast_report_window_seconds
        )
        if outstanding and window_open:
            # Repairing now would resend fragments those nodes may already
            # hold, because their silence is being read as total loss.
            self.log(
                "multicast_awaiting_reports",
                "server",
                {"outstanding": sorted(outstanding)},
            )
            return

        self._multicast_rounds += 1
        if self._multicast_rounds > self.max_downlink_attempts:
            delivery.failed = True
            self.failure_reason = (
                f"multicast group still incomplete after "
                f"{self.max_downlink_attempts} repair rounds"
            )
            self.log("experiment_failed", "server", {"reason": self.failure_reason})
            return
        wanted = [
            index for index in range(GLOBAL_FRAGMENTS) if union & (1 << index)
        ]
        # A fragment may legitimately be resent several times when different
        # nodes miss different ones, so the per-fragment attempt limit is not
        # the right bound here; the round counter above is.
        delivery.attempts.clear()
        self._reported_since_burst.clear()
        self._burst_ended_at = None
        delivery.pending = deque(wanted)
        self._clear_inflight(delivery)
        delivery.next_send_at = self.clock() + self.retry_guard_seconds
        self.log(
            "multicast_repair",
            "server",
            {
                "round": self._multicast_rounds,
                "union_mask": f"{union:04X}",
                "fragments": wanted,
                "still_missing": {
                    device_id: f"{mask:04X}"
                    for device_id, mask in self.node_missing.items()
                    if mask
                },
            },
        )

    def _handle_status_before_delivery(self, device_id: str) -> None:
        """A node reported GLOBAL_STATUS before the Class C phase began.

        Two very different situations reach here, and only one is a problem:

        - the node's own UPDATE is still incomplete, so its repair downlink was
          probably lost. Its Class C timeout is the retry opportunity.
        - the node's UPDATE is complete and it is simply waiting in Class C
          while some OTHER node finishes repairing. It has heard nothing
          because nothing has been sent yet. That is entirely normal, and used
          to abort the whole run.
        """
        still_missing = self.reassemblers[device_id].missing_mask()
        if still_missing:
            self.log(
                "global_status_before_delivery",
                device_id,
                {"update_missing_mask": f"{still_missing:04X}"},
            )
            self._request_update_repair(device_id, still_missing)
            return
        # GLOBAL_STATUS is only ever sent from requestStatus(), reachable only
        # from the Class C timeout or a finished global -- every one of those
        # means the node IS in Class C. That is exactly what CLASS_C_READY
        # asserts, so treat this as an implicit one. Without it a single lost
        # READY (unconfirmed, so the node never knows) would stall the whole
        # run until the 90-minute timeout, because nothing else re-announces.
        implied_ready = device_id not in self.ready
        self.ready.add(device_id)
        self.log(
            "global_status_while_waiting",
            device_id,
            {
                "waiting_for": sorted(set(self.device_ids) - set(self.updates)),
                "implied_class_c_ready": implied_ready,
            },
        )
        if implied_ready:
            self._start_if_ready()

    def _handle_status_multicast(
        self, device_id: str, missing: int, crc: int
    ) -> None:
        if not self.deliveries:
            self._handle_status_before_delivery(device_id)
            return
        if missing == 0:
            if crc != self.global_crc:
                raise ProtocolError(
                    f"{device_id} acknowledged CRC {crc:08X}; expected {self.global_crc:08X}"
                )
            self.node_missing[device_id] = 0
            self.counters[device_id]["class_c_acknowledged"] += 1
            self.log("global_acknowledged", device_id, {"crc32": f"{crc:08X}"})
        else:
            self.node_missing[device_id] = missing
            self.log(
                "global_missing",
                device_id,
                {
                    "missing_bitmap": f"0x{missing:03X}",
                    "fragments": [
                        i for i in range(GLOBAL_FRAGMENTS) if missing & (1 << i)
                    ],
                },
            )
        self._reported_since_burst.add(device_id)
        self._refresh_multicast_pending()

    def _handle_status(self, device_id: str, frame: Frame) -> None:
        if len(frame.data) != 6 or frame.fragment_count != GLOBAL_FRAGMENTS:
            raise ProtocolError("GLOBAL_STATUS must contain missing bitmap and CRC32")
        missing = int.from_bytes(frame.data[:2], "little")
        crc = int.from_bytes(frame.data[2:], "little")
        if missing & ~ALL_GLOBAL_FRAGMENTS:
            raise ProtocolError("GLOBAL_STATUS sets bits outside the 11-fragment bitmap")
        if self.multicast:
            self._handle_status_multicast(device_id, missing, crc)
            return
        delivery = self.deliveries.get(device_id)
        if delivery is None:
            # The Class C phase never started for this node, so its UPDATE is
            # still incomplete, or it is simply waiting on another node.
            self._handle_status_before_delivery(device_id)
            return
        if missing == 0:
            if crc != self.global_crc:
                raise ProtocolError(
                    f"{device_id} acknowledged CRC {crc:08X}; expected {self.global_crc:08X}"
                )
            delivery.acknowledged = True
            delivery.pending.clear()
            self._clear_inflight(delivery)
            # Counted here as well as on the multicast path so that both arms
            # of the comparison expose the SAME per-device field in
            # summary.json. Without this, unicast runs silently omit
            # class_c_acknowledged and the two modes cannot be tabulated
            # against each other without special-casing the mode.
            self.counters[device_id]["class_c_acknowledged"] += 1
            self.log("global_acknowledged", device_id, {"crc32": f"{crc:08X}"})
            return

        delivery.pending = deque(
            index for index in range(GLOBAL_FRAGMENTS) if missing & (1 << index)
        )
        self._clear_inflight(delivery)
        delivery.next_send_at = self.clock() + self.retry_guard_seconds
        self.log(
            "global_missing",
            device_id,
            {"missing_bitmap": f"0x{missing:03X}", "fragments": list(delivery.pending)},
        )

    def _start_if_ready(self) -> None:
        if self.deliveries:
            return
        if set(self.updates) != set(self.device_ids) or self.ready != set(self.device_ids):
            return
        self.global_payload = make_global_payload(self.updates, self.round_id)
        self.global_crc = validate_crc(self.global_payload, GLOBAL_BYTES)
        frames = fragment_payload(self.global_payload, MSG_GLOBAL_FRAGMENT, self.round_id)
        if len(frames) != GLOBAL_FRAGMENTS:
            raise AssertionError("global payload did not produce 11 fragments")
        start_at = self.clock() + self.class_c_guard_seconds
        self.node_missing = {
            device_id: ALL_GLOBAL_FRAGMENTS for device_id in self.device_ids
        }
        self.deliveries = {
            target: Delivery(
                frames=frames,
                pending=deque(range(GLOBAL_FRAGMENTS)),
                next_send_at=start_at,
            )
            for target in self._delivery_targets()
        }
        self.log(
            "aggregation_complete",
            "server",
            {
                "input_devices": list(self.device_ids),
                "mode": "multicast" if self.multicast else "unicast",
                "transmissions": len(frames) * len(self._delivery_targets()),
                "global_bytes": len(self.global_payload),
                "global_fragments": len(frames),
                "crc32": f"{self.global_crc:08X}",
            },
        )

    @_synchronized
    def tick(self) -> None:
        now = self.clock()
        # Before anything else: reclaim a slot whose status event never came.
        self._expire_inflight_downlinks(now)
        # A node that never reports must not stall the repair forever.
        if (
            self.multicast
            and self.deliveries
            and self._burst_ended_at is not None
            and now - self._burst_ended_at >= self.multicast_report_window_seconds
            and not self.finished
        ):
            delivery = self.deliveries[self.multicast_device_id]
            if not delivery.pending and delivery.inflight is None:
                self._refresh_multicast_pending()
        # One downlink in flight across ALL devices. The gateway is half duplex,
        # so letting each device hold its own inflight slot made concurrent
        # Class C sends collide (TTS reported COLLISION_PACKET / TOO_LATE) and
        # burned retry attempts that have nothing to do with radio conditions.
        if any(
            delivery.inflight is not None for delivery in self.deliveries.values()
        ):
            return
        # Round-robin from where the last send left off. Iterating the dict
        # from the top every time would let the first device drain all 11 of
        # its fragments before any other node received one, which matters more
        # the more nodes there are: the last one waits out everybody else while
        # its own 15-minute Class C timer is already running.
        devices = list(self.deliveries.items())
        for offset in range(len(devices)):
            position = (self._downlink_cursor + offset) % len(devices)
            device_id, delivery = devices[position]
            if (
                delivery.failed
                or delivery.acknowledged
                or delivery.inflight is not None
                or not delivery.pending
                or now < delivery.next_send_at
            ):
                continue
            index = delivery.pending.popleft()
            delivery.attempts[index] += 1
            if delivery.attempts[index] > self.max_downlink_attempts:
                delivery.failed = True
                self.failure_reason = (
                    f"{device_id} fragment {index} exceeded the downlink retry limit"
                )
                self.log("experiment_failed", device_id, {"reason": self.failure_reason})
                return
            correlation = (
                f"v91-hw-r{self.round_id}-{device_id}-f{index}-a{delivery.attempts[index]}"
            )
            # Claim the inflight slot BEFORE publishing. The broker can deliver
            # down/sent on the network thread while publish() is still
            # returning here; with the assignment after the call that callback
            # found inflight unset, discarded itself, and this line then pinned
            # the slot forever.
            delivery.inflight = index
            delivery.inflight_deadline = now + self.inflight_downlink_timeout_seconds
            try:
                self.publish(device_id, delivery.frames[index], correlation)
            except Exception:
                # Nothing left the process, so put the fragment back rather
                # than letting the slot and the attempt both be consumed.
                self._clear_inflight(delivery)
                delivery.pending.appendleft(index)
                delivery.attempts[index] -= 1
                delivery.next_send_at = now + self.retry_guard_seconds
                raise
            self.counters[device_id]["downlinks_queued"] += 1
            self.counters[device_id]["downlink_application_bytes_queued"] += len(
                delivery.frames[index]
            )
            self.log(
                "global_fragment_queued",
                device_id,
                {
                    "fragment": index,
                    "count": GLOBAL_FRAGMENTS,
                    "application_bytes": len(delivery.frames[index]),
                    "attempt": delivery.attempts[index],
                    "correlation_id": correlation,
                },
            )
            # Exactly one downlink per tick; the next starts only once this one
            # is confirmed sent or failed.
            self._downlink_cursor = (position + 1) % len(devices)
            return

    @_synchronized
    def handle_downlink_sent(self, device_id: str) -> None:
        self._require_delivery_target(device_id)
        # A repair frame is confirmed before any Delivery exists, so account
        # for it first rather than returning and losing its bytes.
        pending_repairs = self.pending_repair_bytes.get(device_id)
        if pending_repairs:
            repair_bytes = pending_repairs.popleft()
            self.counters[device_id]["update_repair_downlinks_sent"] += 1
            self.counters[device_id]["downlink_application_bytes_sent"] += repair_bytes
            self.log("update_repair_sent", device_id, {"application_bytes": repair_bytes})
            return
        delivery = self.deliveries.get(device_id)
        if delivery is None or delivery.inflight is None:
            return
        index = delivery.inflight
        self._clear_inflight(delivery)
        delivery.next_send_at = self.clock()
        self.counters[device_id]["class_c_downlinks_sent"] += 1
        self.counters[device_id]["downlink_application_bytes_sent"] += len(
            delivery.frames[index]
        )
        self.log("global_fragment_sent", device_id, {"fragment": index})
        if self.multicast and not delivery.pending:
            # Burst drained; start the window in which nodes may report.
            self._burst_ended_at = self.clock()

    @staticmethod
    def _explain_downlink_failure(reason: str):
        """Turn a raw TTS rejection into (error name, plain explanation).

        The explanation is only set for failures that can never succeed on a
        retry, so the caller can stop instead of spending the retry budget on a
        deterministic rejection.
        """
        try:
            error = json.loads(reason).get("error", {})
        except (ValueError, AttributeError, TypeError):
            return "", None
        name = str(error.get("name", ""))
        attributes = error.get("attributes") or {}

        if name == "application_downlink_too_long":
            length = attributes.get("length", "?")
            maximum = str(attributes.get("max", "?"))
            # EU868 application payload ceilings by data rate.
            implied = {"51": "DR0-DR2 (SF12-SF10)", "115": "DR3 (SF9)",
                       "222": "DR4/DR5 (SF8/SF7)"}.get(maximum, f"max {maximum}")
            return name, (
                "the target's RX2 data rate only allows " + maximum +
                " bytes " + implied + " but this frame is " + str(length) +
                "; set the device's rx2_data_rate_index to 5 (SF7, 222 bytes)."
                " For a multicast group that is the ACTUAL rx2_data_rate_index,"
                " not the 'desired' field, and an existing session may need to"
                " be recreated before a changed value takes effect"
            )
        if name == "no_downlink_path":
            return name, (
                "the network server has no gateway to transmit through; a "
                "multicast group never uplinks, so multicast_gateway_ids must "
                "name the gateway"
            )
        return name, None

    @_synchronized
    def handle_downlink_failed(self, device_id: str, reason: str) -> None:
        self._require_delivery_target(device_id)
        delivery = self.deliveries.get(device_id)
        if delivery is None or delivery.inflight is None:
            return
        index = delivery.inflight
        self._clear_inflight(delivery)

        name, explanation = self._explain_downlink_failure(reason)
        self.log(
            "global_fragment_failed",
            device_id,
            {"fragment": index, "error": name or None,
             "explanation": explanation, "reason": reason},
        )
        if explanation is not None:
            # Deterministic: the identical frame is rejected the same way every
            # time, so retrying only spends the budget and buries the cause.
            delivery.failed = True
            self.failure_reason = name + ": " + explanation
            self.log("experiment_failed", device_id,
                     {"reason": self.failure_reason})
            return

        delivery.pending.appendleft(index)
        delivery.next_send_at = self.clock() + self.retry_guard_seconds

    @_synchronized
    def summary(self) -> dict:
        return {
            "round_id": self.round_id,
            "device_ids": list(self.device_ids),
            "update_logical_bytes_per_node": UPDATE_BYTES,
            "update_fragments_per_node": UPDATE_FRAGMENTS,
            "global_logical_bytes_per_node": GLOBAL_BYTES,
            "global_fragments_per_node": GLOBAL_FRAGMENTS,
            "class_c_mode": "multicast" if self.multicast else "unicast",
            "class_c_unicast_copies": 1 if self.multicast else len(self.device_ids),
            "class_c_group_id": self.multicast_device_id,
            "finished": self.finished,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "global_crc32": None if self.global_crc is None else f"{self.global_crc:08X}",
            "per_device": {
                device_id: dict(sorted(counter.items()))
                for device_id, counter in self.counters.items()
            },
        }
