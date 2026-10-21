"""Unit tests for dropout_fl.py (see dropout.json)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fl  # noqa: E402
import dropout_fl as dfl  # noqa: E402

LAYERS = [22, 12, 22]


def _synthetic_clients(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return {f"c{i}": rng.normal(size=(6 + i, 22)) for i in range(n)}


class BitExactAllTrueTests(unittest.TestCase):
    def check(self, algorithm):
        clients = _synthetic_clients()
        n = len(clients)
        rounds = 6
        uplink = np.ones((rounds, n), dtype=bool)
        downlink = np.ones((rounds, n), dtype=bool)
        server, finals = dfl.train_with_loss(clients, LAYERS, 17070, algorithm, rounds, uplink, downlink)

        ref = fl.Server(algorithm, fl.init(LAYERS, 17070))
        for _ in range(rounds):
            ref.round(clients)

        for a, b in zip(server, ref.p):
            np.testing.assert_array_equal(a, b)
        # every client received every global model (downlink all True), so
        # each client's final copy must equal the server global exactly.
        for cid in clients:
            for a, b in zip(finals[cid], server):
                np.testing.assert_array_equal(a, b)

    def test_fedadam_bit_exact(self):
        self.check("fedadam")

    def test_fedavg_bit_exact(self):
        self.check("fedavg")


class EmptyRoundTests(unittest.TestCase):
    def test_empty_round_leaves_server_unchanged(self):
        clients = _synthetic_clients()
        n = len(clients)
        uplink = np.zeros((1, n), dtype=bool)
        downlink = np.ones((1, n), dtype=bool)
        server, _ = dfl.train_with_loss(clients, LAYERS, 5, "fedadam", 1, uplink, downlink)
        init = fl.init(LAYERS, 5)
        for a, b in zip(server, init):
            np.testing.assert_array_equal(a, b)

    def check_counter_not_advanced(self, algorithm):
        clients = _synthetic_clients()
        n = len(clients)
        # 2 rounds: round 1 empty (no uplink), round 2 full uplink+downlink.
        uplink2 = np.array([[False] * n, [True] * n])
        downlink2 = np.ones((2, n), dtype=bool)
        server2, finals2 = dfl.train_with_loss(clients, LAYERS, 9, algorithm, 2, uplink2, downlink2)

        # A single fresh round should be bit-identical IF the empty round
        # truly left the server (and the FedAdam step counter) untouched:
        # a wrongly-advanced counter would change FedAdam's bias correction
        # on the second round.
        uplink1 = np.ones((1, n), dtype=bool)
        downlink1 = np.ones((1, n), dtype=bool)
        server1, finals1 = dfl.train_with_loss(clients, LAYERS, 9, algorithm, 1, uplink1, downlink1)

        for a, b in zip(server2, server1):
            np.testing.assert_array_equal(a, b)
        for cid in clients:
            for a, b in zip(finals2[cid], finals1[cid]):
                np.testing.assert_array_equal(a, b)

    def test_fedadam_counter_not_advanced(self):
        self.check_counter_not_advanced("fedadam")

    def test_fedavg_counter_not_advanced(self):
        self.check_counter_not_advanced("fedavg")


class StaleDownlinkTests(unittest.TestCase):
    def test_client_with_all_downlinks_false_scored_with_initial_model(self):
        clients = _synthetic_clients()
        client_ids = list(clients.keys())
        n = len(client_ids)
        rounds = 5
        rng = np.random.default_rng(1)
        uplink = rng.random((rounds, n)) >= 0.2
        downlink = np.ones((rounds, n), dtype=bool)
        stale_idx = 1
        downlink[:, stale_idx] = False  # this client never receives a global model

        _, finals = dfl.train_with_loss(clients, LAYERS, 3, "fedadam", rounds, uplink, downlink)
        init = fl.init(LAYERS, 3)
        stale_cid = client_ids[stale_idx]
        for a, b in zip(finals[stale_cid], init):
            np.testing.assert_array_equal(a, b)
        # sanity: at least one other client's copy actually moved.
        moved_cid = client_ids[0]
        moved = any(not np.array_equal(a, b) for a, b in zip(finals[moved_cid], init))
        self.assertTrue(moved)


class S3StratificationTests(unittest.TestCase):
    def _devices(self):
        # 3 datasets of sizes 10, 6, 4 = 20 devices total.
        devices = []
        for i in range(10):
            devices.append({"machine_id": f"a::{i}", "dataset": "a"})
        for i in range(6):
            devices.append({"machine_id": f"b::{i}", "dataset": "b"})
        for i in range(4):
            devices.append({"machine_id": f"c::{i}", "dataset": "c"})
        return devices

    def test_requested_count_and_no_duplicates(self):
        devices = self._devices()
        for f in (0.1, 0.2, 0.3):
            rng = np.random.default_rng(42)
            chosen = dfl.select_never_participating(devices, f, rng)
            expected_total = int(np.floor(f * len(devices) + 0.5))
            self.assertEqual(len(chosen), expected_total)
            self.assertEqual(len(chosen), len(set(chosen)))
            self.assertTrue(chosen <= set(d["machine_id"] for d in devices))

    def test_stratified_floor_per_dataset_is_respected(self):
        devices = self._devices()
        f = 0.3
        rng = np.random.default_rng(7)
        chosen = dfl.select_never_participating(devices, f, rng)
        for ds, size in (("a", 10), ("b", 6), ("c", 4)):
            n_chosen_ds = sum(1 for mid in chosen if mid.startswith(ds + "::"))
            floor_ds = int(np.floor(f * size))
            self.assertGreaterEqual(n_chosen_ds, floor_ds)

    def test_deterministic(self):
        devices = self._devices()
        c1 = dfl.select_never_participating(devices, 0.2, np.random.default_rng(123))
        c2 = dfl.select_never_participating(devices, 0.2, np.random.default_rng(123))
        self.assertEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
