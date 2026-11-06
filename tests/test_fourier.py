"""Fourier transforms S(Q) <-> G(r), g(r), RDF, T(r), N(r) and windows."""

from __future__ import annotations

import unittest

import helpers  # noqa: F401
from helpers import synthetic_liquid

import numpy as np
from scipy import integrate, special

from toscana.math.fourier import backFT, getSineFT, getSineIntegral, sineFT
from toscana.math.line_shapes import step
from toscana.math.windows import Lorch, LorchN

RHO = 0.1
SQ, R_TRUE, G_TRUE = synthetic_liquid(RHO)


def _true_coordination(r_lo: float, r_hi: float) -> float:
    m = (R_TRUE >= r_lo) & (R_TRUE <= r_hi)
    return float(integrate.simpson(4 * np.pi * RHO * R_TRUE[m] ** 2 * G_TRUE[m], x=R_TRUE[m]))


class TestSineFT(unittest.TestCase):
    nr = np.arange(0, 6, 0.02)

    @classmethod
    def setUpClass(cls) -> None:
        cls.fou = sineFT(SQ, cls.nr, qmax=25.0, density=RHO, window=0)

    def test_column_relations(self) -> None:
        r, pcf, pdf, rdf, tor = (self.fou[:, k] for k in range(5))
        pos = r > 0
        np.testing.assert_allclose(pdf[pos], pcf[pos] / (4 * np.pi * RHO * r[pos]) + 1)
        np.testing.assert_allclose(rdf, 4 * np.pi * RHO * r**2 * np.where(pos, pdf, 0), atol=1e-12)
        np.testing.assert_allclose(tor[pos], 4 * np.pi * RHO * r[pos] * pdf[pos])

    def test_low_r_limit(self) -> None:
        low = (self.nr > 0.5) & (self.nr < 1.5)
        self.assertLess(abs(float(self.fou[low, 2].mean())), 0.02)

    def test_coordination_number(self) -> None:
        m = (self.nr >= 2.3) & (self.nr <= 3.3)
        n = float(integrate.simpson(self.fou[m, 3], x=self.nr[m]))
        self.assertAlmostEqual(n / _true_coordination(2.3, 3.3), 1.0, delta=0.05)

    def test_running_integral_from_zero(self) -> None:
        rdf, run = self.fou[:, 3], self.fou[:, 5]
        for i in (1, 2, 7, 100, len(self.nr) - 1):
            self.assertAlmostEqual(
                run[i], float(integrate.simpson(rdf[: i + 1], x=self.nr[: i + 1])), places=10
            )

    def test_running_integral_grid_not_starting_at_zero(self) -> None:
        nr = np.arange(1.0, 3.0, 0.1)
        fou = sineFT(SQ, nr, qmax=25.0, density=RHO)
        self.assertEqual(fou[0, 5], 0.0)
        for i in (1, 5, len(nr) - 1):
            self.assertAlmostEqual(
                fou[i, 5], float(integrate.simpson(fou[: i + 1, 3], x=nr[: i + 1])), places=10
            )

    @unittest.expectedFailure
    def test_lorch_low_r_limit_pending_C1(self) -> None:
        """C1 (pending decision): with the Lorch window g(r) should also -> 0 at low r."""
        fou = sineFT(SQ, self.nr, qmax=25.0, density=RHO, window=1)
        low = (self.nr > 0.5) & (self.nr < 1.5)
        self.assertLess(abs(float(fou[low, 2].mean())), 0.05)


class TestBackFT(unittest.TestCase):
    ar = np.arange(0, 10, 0.02)
    qu = np.arange(0, 20, 0.02)

    @classmethod
    def setUpClass(cls) -> None:
        cls.g = sineFT(SQ, cls.ar, qmax=25.0, density=RHO)[:, 2]
        cls.out = backFT(cls.qu, cls.ar, cls.g, RHO, [2.0])

    def test_all_outputs_finite(self) -> None:
        for arr in self.out:
            self.assertTrue(np.all(np.isfinite(arr)))

    def test_q_zero_limit(self) -> None:
        g_cut = np.where(self.ar < 2.0, 0.0, self.g)
        expected = 1 + 4 * np.pi * RHO * integrate.simpson(self.ar**2 * (g_cut - 1), x=self.ar)
        self.assertAlmostEqual(self.out[0][0], expected, places=6)
        # S(Q) continuous at Q = 0
        self.assertAlmostEqual(self.out[0][0], self.out[0][1], delta=0.01)

    def test_r_relations(self) -> None:
        _soq, _pcf, pdf, rdf, tor, run = self.out
        self.assertEqual(pdf[0], 0.0)
        np.testing.assert_allclose(tor, 4 * np.pi * RHO * self.ar * pdf)
        np.testing.assert_allclose(rdf, 4 * np.pi * RHO * self.ar**2 * pdf)
        for i in (1, 2, 50, len(self.ar) - 1):
            self.assertAlmostEqual(
                run[i], float(integrate.simpson(rdf[: i + 1], x=self.ar[: i + 1])), places=10
            )

    def test_auto_cut_without_negative_values_cuts_nothing(self) -> None:
        pdf = 1 + 0.1 * np.sin(self.ar)  # never negative
        auto = backFT(self.qu, self.ar, pdf, RHO, [-1])
        none = backFT(self.qu, self.ar, pdf, RHO, [-0.5])
        np.testing.assert_allclose(auto[0], none[0])

    def test_cut_intervals(self) -> None:
        many = [1.0] + [x for k in range(8) for x in (2.0 + k, 2.5 + k)]  # 17 values
        backFT(self.qu, self.ar, self.g, RHO, many)
        with self.assertRaises(ValueError):
            backFT(self.qu, self.ar, self.g, RHO, [1.0, 2.0])


class TestSineHelpers(unittest.TestCase):
    def test_sine_integral(self) -> None:
        for x in (0.5, 1.0, np.pi, 10.0, 40.0):
            self.assertAlmostEqual(getSineIntegral(x), float(special.sici(x)[0]), places=10)
        self.assertEqual(getSineIntegral(0.0), 0.0)

    def test_get_sine_ft_vmax_truncates(self) -> None:
        v = np.arange(0, 20, 0.01)
        y = 1 + np.exp(-v) * np.sin(3 * v)
        u = np.linspace(0.5, 5, 7)
        keep = v <= 10.0
        np.testing.assert_allclose(
            getSineFT(v, y, u, vmax=10.0), getSineFT(v[keep], y[keep], u), rtol=1e-12
        )
        # default vmax: full range, unchanged
        np.testing.assert_allclose(getSineFT(v, y, u), getSineFT(v, y, u, vmax=float(v[-1])))

    def test_windows_reject_bad_qmax(self) -> None:
        with self.assertRaises(ValueError):
            Lorch(1.0, 0.0)
        with self.assertRaises(ValueError):
            step(1.0, -1.0)

    def test_lorch_shape(self) -> None:
        self.assertEqual(Lorch(0.0, 10.0), 1.0)
        self.assertAlmostEqual(Lorch(5.0, 10.0), 2 / np.pi)
        self.assertEqual(Lorch(10.0, 10.0), 0.0)

    @unittest.expectedFailure
    def test_lorch_n_unit_at_zero_pending_C1(self) -> None:
        """C1 (pending decision): standard Lorch has M(0) = 1."""
        self.assertAlmostEqual(float(LorchN(np.arange(0, 23.5, 0.02), 23.5)[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
