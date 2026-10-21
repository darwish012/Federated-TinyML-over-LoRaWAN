"""Unit tests: gradients, h12 equivalence to wider_models, stopping rule."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_causal_20260908"))
import fl  # noqa: E402
import wider_models as wider  # noqa: E402


def _loss(x, p):
    y, _ = fl.forward(x, p)
    ws = p[0::2]
    return float(np.mean((y - x) ** 2) + 0.5 * fl.L2 * sum(np.sum(w * w) for w in ws))


class GradientTests(unittest.TestCase):
    def check(self, layers):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(9, layers[0]))
        p = fl.init(layers, 11)
        g = fl.grads(x, p)
        eps = 1e-6
        for k in range(len(p)):
            flat = p[k].ravel()
            for j in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                plus = [v.copy() for v in p]; minus = [v.copy() for v in p]
                plus[k].ravel()[j] += eps; minus[k].ravel()[j] -= eps
                num = (_loss(x, tuple(plus)) - _loss(x, tuple(minus))) / (2 * eps)
                self.assertAlmostEqual(num, g[k].ravel()[j], places=6)

    def test_h12_gradients(self):
        self.check([22, 12, 22])

    def test_deep_gradients(self):
        self.check([22, 16, 8, 16, 22])

    def test_prox_gradient(self):
        layers = [22, 12, 22]
        rng = np.random.default_rng(4)
        x = rng.normal(size=(7, 22))
        p = fl.init(layers, 1); q = fl.init(layers, 2)
        g0 = fl.grads(x, p); g1 = fl.grads(x, p, prox=q, mu=0.5)
        for a, b, pi, qi in zip(g0, g1, p, q):
            np.testing.assert_allclose(b - a, 0.5 * (pi - qi), atol=1e-12)


class EquivalenceTests(unittest.TestCase):
    def test_h12_fedadam_matches_wider_models(self):
        rng = np.random.default_rng(5)
        clients = {f"c{i}": np.abs(rng.normal(size=(20 + i, 22))) for i in range(4)}
        ref = wider.fit(clients, "global_h12", seed=17070)["global"]
        s = fl.Server("fedadam", fl.init([22, 12, 22], 17070))
        for _ in range(wider.ROUNDS):
            s.round(clients)
        for a, b in zip(s.p, ref):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_param_counts(self):
        self.assertEqual(fl.n_params([22, 12, 22]), 562)
        self.assertEqual(fl.n_params([22, 16, 8, 16, 22]), 1022)
        self.assertEqual(fl.fragments([22, 12, 22]), 12)
        self.assertEqual(fl.fragments([22, 16, 8, 16, 22]), 21)

    def test_fedprox_mu_zero_is_fedavg(self):
        rng = np.random.default_rng(6)
        clients = {f"c{i}": rng.normal(size=(15, 22)) for i in range(3)}
        a = fl.Server("fedavg", fl.init([22, 12, 22], 1))
        b = fl.Server("fedprox", fl.init([22, 12, 22], 1), mu=0.0)
        for _ in range(3):
            a.round(clients); b.round(clients)
        for u, v in zip(a.p, b.p):
            np.testing.assert_array_equal(u, v)


class StoppingTests(unittest.TestCase):
    def test_plateau_stops(self):
        losses = [1.0, 0.5, 0.3, 0.29, 0.289, 0.2889, 0.2888]
        t, why = fl.stop_round(losses, 0.005, 3, 60)
        self.assertEqual(why, "rule")
        # t=6: (B3=0.3 - B6=0.2889)/0.3 = 3.7%, keep going;
        # t=7: (B4=0.29 - B7=0.2888)/0.29 = 0.41% < 0.5%, stop.
        self.assertEqual(t, 7)

    def test_never_stops_is_cap(self):
        losses = [1.0 * 0.9 ** i for i in range(60)]
        self.assertEqual(fl.stop_round(losses, 0.005, 3, 60), (60, "cap"))

    def test_causal(self):
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.1, 0.05]
        t, _ = fl.stop_round(losses, 0.005, 3, 60)
        t2, _ = fl.stop_round(losses[:t], 0.005, 3, 60)
        self.assertEqual(t, t2)


if __name__ == "__main__":
    unittest.main()
