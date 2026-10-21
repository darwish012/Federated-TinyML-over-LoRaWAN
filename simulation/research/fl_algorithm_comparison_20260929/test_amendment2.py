"""Unit tests for run_amendment2.py: Spearman helper, stop-rule logic, the
reference() replication of cadence.dense_flags, and LODO tie-breaking.
No cache files are loaded."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_causal_20260908"))
import cadence  # noqa: E402
import run_amendment2 as amend2  # noqa: E402


class SpearmanTests(unittest.TestCase):
    def test_hand_example_with_ties(self):
        # ranks(a) = [1, 2.5, 2.5, 4]; ranks(b) = [4, 2.5, 2.5, 1]
        # perfectly (tie-consistent) anti-monotonic -> rho = -1.0 exactly.
        a = [1, 2, 2, 3]
        b = [4, 3, 3, 1]
        self.assertAlmostEqual(amend2.spearman(a, b), -1.0, places=12)

    def test_perfect_positive_with_ties(self):
        a = [1, 2, 2, 3]
        b = [1, 2, 2, 3]
        self.assertAlmostEqual(amend2.spearman(a, b), 1.0, places=12)

    def test_matches_scipy_when_available(self):
        try:
            from scipy.stats import spearmanr
        except ImportError:
            self.skipTest("scipy not importable")
        rng = np.random.default_rng(0)
        a = rng.normal(size=30)
        b = rng.normal(size=30)
        # introduce ties
        a[5] = a[6] = a[7]
        expected = spearmanr(a, b).statistic
        self.assertAlmostEqual(amend2.spearman(a, b), float(expected), places=10)

    def test_constant_array_denominator_zero(self):
        a = [5.0, 5.0, 5.0]
        b = [1.0, 2.0, 3.0]
        # ra is all-equal -> variance 0 -> denom 0, and a != b -> defined as 0.0
        self.assertEqual(amend2.spearman(a, b), 0.0)
        self.assertEqual(amend2.spearman(a, a), 1.0)


class RankdataTests(unittest.TestCase):
    def test_average_ties(self):
        r = amend2.rankdata_avg([1, 2, 2, 3])
        np.testing.assert_allclose(r, [1.0, 2.5, 2.5, 4.0])

    def test_no_ties(self):
        r = amend2.rankdata_avg([30, 10, 20])
        np.testing.assert_allclose(r, [3.0, 1.0, 2.0])


class StopLogicTests(unittest.TestCase):
    def test_requires_two_consecutive_rounds(self):
        # D1-style: criterion is rho_t >= constant. True at rounds 3 and 5 only
        # (not consecutive) -> must NOT stop there; only a genuine two-in-a-row
        # triggers a stop.
        signals = {t: {"rho": 0.0, "delta": 1.0, "param_change": 1.0} for t in range(2, 16)}
        signals[3] = {"rho": 0.999, "delta": 1.0, "param_change": 1.0}
        signals[5] = {"rho": 0.999, "delta": 1.0, "param_change": 1.0}
        t_stop, why = amend2.rule_stop("D1", 0.995, signals)
        self.assertEqual((t_stop, why), (15, "cap"))

    def test_stops_at_first_consecutive_pair(self):
        signals = {t: {"rho": 0.0, "delta": 1.0, "param_change": 1.0} for t in range(2, 16)}
        signals[4] = {"rho": 0.999, "delta": 1.0, "param_change": 1.0}
        signals[5] = {"rho": 0.999, "delta": 1.0, "param_change": 1.0}
        signals[6] = {"rho": 0.999, "delta": 1.0, "param_change": 1.0}
        t_stop, why = amend2.rule_stop("D1", 0.995, signals)
        self.assertEqual((t_stop, why), (5, "rule"))

    def test_earliest_possible_stop_is_3(self):
        signals = {t: {"rho": 0.999, "delta": 0.0, "param_change": 0.0} for t in range(2, 16)}
        t_stop, why = amend2.rule_stop("D1", 0.995, signals)
        self.assertEqual((t_stop, why), (3, "rule"))

    def test_caps_at_15_when_never_two_in_a_row(self):
        signals = {t: {"rho": 0.0, "delta": 1.0, "param_change": 1.0} for t in range(2, 16)}
        # alternate true/false -> never two consecutive
        for t in range(2, 16, 2):
            signals[t]["rho"] = 0.999
        t_stop, why = amend2.rule_stop("D1", 0.995, signals)
        self.assertEqual((t_stop, why), (15, "cap"))

    def test_d2_and_d3_use_strict_less_than(self):
        signals = {t: {"rho": 0.0, "delta": 0.5, "param_change": 0.5} for t in range(2, 16)}
        signals[3] = {"rho": 0.0, "delta": 0.02, "param_change": 0.5}
        signals[4] = {"rho": 0.0, "delta": 0.02, "param_change": 0.5}
        t_stop, why = amend2.rule_stop("D2", 0.02, signals)
        self.assertEqual((t_stop, why), (15, "cap"))  # 0.02 < 0.02 is False
        signals[3]["delta"] = 0.019
        signals[4]["delta"] = 0.019
        t_stop, why = amend2.rule_stop("D2", 0.02, signals)
        self.assertEqual((t_stop, why), (4, "rule"))


class ReferenceHelperTests(unittest.TestCase):
    def test_reproduces_dense_flags_on_synthetic_series(self):
        rng = np.random.default_rng(42)
        n = 200
        score = np.abs(rng.normal(0.1, 0.05, size=n))
        # inject an elevated fault tail so flags are non-trivial
        score[150:] += 0.5
        ti = np.arange(0, 40)
        ci = np.arange(40, 60)
        flags = cadence.dense_flags(score, ti, ci, 21, 13, 2.0)
        ref = amend2.reference(score, ti, ci)
        amend2.verify_reference(score, ti, ci, ref, flags)  # must not raise

    def test_mismatch_raises(self):
        rng = np.random.default_rng(7)
        score = np.abs(rng.normal(0.1, 0.05, size=100))
        ti = np.arange(0, 30)
        ci = np.arange(30, 50)
        flags = cadence.dense_flags(score, ti, ci, 21, 13, 2.0)
        wrong_ref = amend2.reference(score, ti, ci) * 0.5  # deliberately wrong
        with self.assertRaises(RuntimeError):
            amend2.verify_reference(score, ti, ci, wrong_ref, flags)

    def test_reference_matches_fallback_branch(self):
        # short history (< window) forces the quantile(ci, .995) fallback
        rng = np.random.default_rng(3)
        n = 30
        score = np.abs(rng.normal(0.1, 0.02, size=n))
        ti = np.arange(0, 5)
        ci = np.arange(5, 10)
        ref = amend2.reference(score, ti, ci)
        expected = float(np.quantile(score[ci], 0.995))
        self.assertAlmostEqual(ref, expected, places=12)
        flags = cadence.dense_flags(score, ti, ci, 21, 13, 2.0)
        amend2.verify_reference(score, ti, ci, ref, flags)


class LodoTieBreakTests(unittest.TestCase):
    def _fake_table(self, overrides):
        """Build a minimal fake `table` dict for select_constant: keys are
        (arch, alg, family, constant, seed) -> {'stop_round', 'overall',
        'per_dataset': {ds: {'strict_successes': n}}}."""
        table = {}
        arch, alg, family = "h12", "fedadam", "D2"
        for c in amend2.FAMILIES[family]:
            for seed in amend2.SEEDS:
                o = overrides[c]
                table[(arch, alg, family, c, seed)] = {
                    "stop_round": o["stop_round"],
                    "overall": {"strict_successes": o["overall"]},
                    "per_dataset": {"dlr": {"strict_successes": o["dlr"]}},
                }
        return table

    def test_picks_maximum_total_not_d(self):
        # constant 0.02 (primary) gives the best held-out total -> chosen
        overrides = {
            0.02: {"stop_round": 5, "overall": 40, "dlr": 4},   # not-d = 36
            0.01: {"stop_round": 5, "overall": 38, "dlr": 4},   # not-d = 34
            0.05: {"stop_round": 5, "overall": 39, "dlr": 4},   # not-d = 35
        }
        table = self._fake_table(overrides)
        chosen = amend2.select_constant("h12", "fedadam", "D2", amend2.FAMILIES["D2"], "dlr", table)
        self.assertEqual(chosen, 0.02)

    def test_tie_on_total_broken_by_fewer_mean_rounds(self):
        # 0.02 and 0.05 tie on not-d total; 0.05 stops earlier -> chosen
        overrides = {
            0.02: {"stop_round": 8, "overall": 40, "dlr": 4},   # not-d = 36
            0.01: {"stop_round": 5, "overall": 30, "dlr": 4},   # not-d = 26
            0.05: {"stop_round": 4, "overall": 40, "dlr": 4},   # not-d = 36, fewer rounds
        }
        table = self._fake_table(overrides)
        chosen = amend2.select_constant("h12", "fedadam", "D2", amend2.FAMILIES["D2"], "dlr", table)
        self.assertEqual(chosen, 0.05)

    def test_tie_on_total_and_rounds_broken_by_primary(self):
        # 0.01 and 0.05 tie with 0.02 (primary) on both total and rounds -> primary wins
        overrides = {
            0.02: {"stop_round": 6, "overall": 40, "dlr": 4},   # not-d = 36
            0.01: {"stop_round": 6, "overall": 40, "dlr": 4},   # not-d = 36, same rounds
            0.05: {"stop_round": 6, "overall": 30, "dlr": 4},   # not-d = 26, loses
        }
        table = self._fake_table(overrides)
        chosen = amend2.select_constant("h12", "fedadam", "D2", amend2.FAMILIES["D2"], "dlr", table)
        self.assertEqual(chosen, 0.02)  # primary preferred over 0.01 on full tie


class FragmentAndConstantSanityTests(unittest.TestCase):
    def test_primary_constants_are_in_sensitivity_lists(self):
        for family, primary in amend2.PRIMARY.items():
            self.assertIn(primary, amend2.FAMILIES[family])

    def test_report_rounds_within_bounds(self):
        for r in amend2.REPORT_ROUNDS:
            self.assertGreaterEqual(r, 1)
            self.assertLessEqual(r, amend2.MAX_ROUNDS)


if __name__ == "__main__":
    unittest.main()
