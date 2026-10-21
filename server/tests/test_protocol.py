from __future__ import annotations

import random
import sys
from pathlib import Path
import unittest


SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))

from protocol import (  # noqa: E402
    GLOBAL_BYTES,
    GLOBAL_FRAGMENTS,
    HEADER_BYTES,
    MAX_APPLICATION_BYTES,
    MSG_GLOBAL_FRAGMENT,
    MSG_UPDATE_FRAGMENT,
    UPDATE_BYTES,
    UPDATE_FRAGMENTS,
    Reassembler,
    decode_frame,
    fragment_payload,
    make_global_payload,
    make_synthetic_update,
    validate_crc,
)


class ProtocolTests(unittest.TestCase):
    def test_exact_v91_packet_counts_and_lengths(self) -> None:
        update = make_synthetic_update(1, 1)
        update_frames = fragment_payload(update, MSG_UPDATE_FRAGMENT, 1)
        self.assertEqual(len(update), UPDATE_BYTES)
        self.assertEqual(len(update_frames), UPDATE_FRAGMENTS)
        self.assertEqual([len(frame) for frame in update_frames[:-1]], [221] * 10)
        self.assertEqual(len(update_frames[-1]), HEADER_BYTES + 62)

        global_payload = make_global_payload(
            {
                "lilygo-node-1": update,
                "lilygo-node-2": make_synthetic_update(2, 1),
            },
            1,
        )
        global_frames = fragment_payload(global_payload, MSG_GLOBAL_FRAGMENT, 1)
        self.assertEqual(len(global_payload), GLOBAL_BYTES)
        self.assertEqual(len(global_frames), GLOBAL_FRAGMENTS)
        self.assertEqual([len(frame) for frame in global_frames[:-1]], [221] * 10)
        self.assertEqual(len(global_frames[-1]), HEADER_BYTES + 101)
        self.assertTrue(all(len(frame) <= MAX_APPLICATION_BYTES for frame in global_frames))

    def test_out_of_order_reassembly_and_crc(self) -> None:
        payload = make_synthetic_update(1, 1)
        frames = fragment_payload(payload, MSG_UPDATE_FRAGMENT, 1)
        random.Random(91).shuffle(frames)
        reassembler = Reassembler(
            MSG_UPDATE_FRAGMENT, 1, UPDATE_FRAGMENTS, UPDATE_BYTES
        )
        result = None
        for raw in frames:
            result = reassembler.add(decode_frame(raw)) or result
        self.assertEqual(result, payload)
        validate_crc(result, UPDATE_BYTES)


if __name__ == "__main__":
    unittest.main()
