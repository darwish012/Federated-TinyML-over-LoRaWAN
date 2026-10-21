"""Unit tests for run_amendment.py: capped stop bound, 70-device loader
guard rails, and fragment arithmetic. No cache files are loaded."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fl  # noqa: E402
import run_amendment as amend  # noqa: E402


class CappedStopTests(unittest.TestCase):
    def test_capped_stop_never_exceeds_15(self):
        # A fast, steadily-decaying curve: relative improvement per patience window stays
        # far above every amendment1 epsilon, so the rule never fires within 15 rounds.
        losses = [1.0 * 0.5 ** i for i in range(15)]
        for eps in (0.01, 0.02, 0.05):
            t, why = fl.stop_round(losses, eps, 3, 15)
            self.assertLessEqual(t, 15)
            self.assertGreaterEqual(t, 1)

    def test_capped_stop_hits_cap_when_never_triggered(self):
        losses = [1.0 * 0.5 ** i for i in range(15)]
        t, why = fl.stop_round(losses, 0.01, 3, 15)
        self.assertEqual((t, why), (15, "cap"))

    def test_capped_stop_can_fire_before_15(self):
        losses = [1.0, 0.5, 0.3, 0.29, 0.289, 0.2889, 0.2888, 0.2888, 0.2888,
                  0.2888, 0.2888, 0.2888, 0.2888, 0.2888, 0.2888]
        t, why = fl.stop_round(losses, 0.05, 3, 15)
        self.assertLessEqual(t, 15)
        self.assertEqual(why, "rule")


class SeventyDeviceLoaderGuardTests(unittest.TestCase):
    """Reproduces the count/consistency assertions run_amendment.part_b makes,
    against small synthetic manifest dicts, with no file IO."""

    def _make_devices(self, n_fault=59, n_control=6, n_unknown=5, n_ldm=16):
        devices = []
        i = 0
        for _ in range(n_fault):
            devices.append({"machine_id": f"src::f{i:03d}", "dataset": "src", "truth": 100, "truth_kind": "onset"})
            i += 1
        for _ in range(n_control):
            devices.append({"machine_id": f"src::c{i:03d}", "dataset": "src", "truth": None, "truth_kind": "healthy_control"})
            i += 1
        for _ in range(n_unknown):
            devices.append({"machine_id": f"src::u{i:03d}", "dataset": "src", "truth": None, "truth_kind": None})
            i += 1
        # fold some of the "fault" devices into the ldm_upb dataset so the ldm count is controllable
        for k in range(min(n_ldm, len(devices))):
            devices[k] = {**devices[k], "machine_id": f"ldm_upb::d{k:03d}", "dataset": "ldm_upb"}
        return devices

    def _check(self, devices, expect_total=70, expect_counts=(59, 6, 5)):
        ids = [d["machine_id"] for d in devices]
        if len(devices) != expect_total or len(set(ids)) != expect_total:
            raise RuntimeError("70-device count/uniqueness mismatch")
        faults = sum(d.get("truth") is not None for d in devices)
        controls = sum(d.get("truth_kind") == "healthy_control" for d in devices)
        unknown = sum(d.get("truth") is None and d.get("truth_kind") != "healthy_control" for d in devices)
        if (faults, controls, unknown) != expect_counts:
            raise RuntimeError("70-cohort truth-count mismatch")
        ldm = [d for d in devices if d["dataset"] == "ldm_upb"]
        nonldm = [d for d in devices if d["dataset"] != "ldm_upb"]
        if len(ldm) != 16 or len(nonldm) != 54:
            raise RuntimeError("ldm_upb/non-LDM split mismatch")
        return True

    def test_valid_manifest_passes(self):
        devices = self._make_devices()
        self.assertTrue(self._check(devices))

    def test_wrong_ldm_count_rejected(self):
        devices = self._make_devices(n_ldm=10)  # 70 total but only 10 tagged ldm_upb, not 16
        with self.assertRaises(RuntimeError):
            self._check(devices)

    def test_wrong_total_rejected(self):
        devices = self._make_devices(n_fault=58, n_control=6, n_unknown=5, n_ldm=16)  # 69 total, not 70
        with self.assertRaises(RuntimeError):
            self._check(devices)

    def test_wrong_truth_counts_rejected(self):
        devices = self._make_devices(n_fault=59, n_control=7, n_unknown=4, n_ldm=16)  # counts don't match (59,6,5)
        with self.assertRaises(RuntimeError):
            self._check(devices)


class FragmentArithmeticTests(unittest.TestCase):
    def test_h12_fragments_per_client_at_12_rounds(self):
        frag = fl.fragments(amend.ARCH["h12"])
        self.assertEqual(2 * frag * 12, 288)

    def test_deep_fragments_per_client_at_12_rounds(self):
        frag = fl.fragments(amend.ARCH["deep"])
        self.assertEqual(2 * frag * 12, 504)


if __name__ == "__main__":
    unittest.main()
