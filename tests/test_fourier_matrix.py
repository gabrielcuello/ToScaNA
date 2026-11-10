"""Matrix form of the Fourier transforms with exact errors (fourier_with_errors,
back_fourier_iterations), fit masks and the fit-prediction error."""

from __future__ import annotations

import unittest

import helpers
from helpers import synthetic_liquid

import numpy as np

from toscana.numerics import (
    FourierSet,
    back_fourier_iterations,
    backFT,
    fit_prediction_error,
    fourier_with_errors,
    ft_matrix,
    qmask,
    sineFT,
    ymask,
)

_ = helpers
RHO = 0.1
_SQ, _, _ = synthetic_liquid(RHO)
Q = _SQ[::3, 0]
S = _SQ[::3, 1]
SIG = 0.01 * (1 + Q / 10)
R = np.round(np.arange(0, 8, 0.05), 10)
R[0] = 1e-6


class TestValues(unittest.TestCase):
    def test_same_as_sineFT(self) -> None:
        for window in (0, 1):
            ref = sineFT(np.column_stack([Q, S, SIG]), R, qmax=Q[-1], density=RHO,
                         window=window, with_errors=True)
            fs = fourier_with_errors(Q, S, SIG, R, RHO, window=window)
            self.assertIsInstance(fs, FourierSet)
            self.assertEqual(fs.table().shape, (len(R), 11))
            np.testing.assert_allclose(fs.table(), ref, rtol=0, atol=1e-9)

    def test_qmax_truncates(self) -> None:
        W = ft_matrix(Q, R, 0, qmax=10.0)
        self.assertTrue(np.all(W[:, Q >= 10.0] == 0))
        ref = sineFT(np.column_stack([Q, S, SIG]), R, qmax=10.0, density=RHO, window=1)
        fs = fourier_with_errors(Q, S, SIG, R, RHO, window=1, qmax=10.0)
        np.testing.assert_allclose(fs.G, ref[:, 1], atol=1e-10)

    def test_invalid_window(self) -> None:
        with self.assertRaises(ValueError):
            ft_matrix(Q, R, window=2)

    def test_back_iterations_same_as_backFT(self) -> None:
        for cut in ([0.8], [0.8, 1.5, 1.7], [-1]):
            start = fourier_with_errors(Q, S, SIG, R, RHO, window=1)
            its = back_fourier_iterations(Q, SIG, start, RHO, cut, 3)
            self.assertEqual(len(its), 3)
            g = start.g.copy()
            for it in its:
                soq, pcf, pdf, rdf, tor, run = backFT(Q, R, g, RHO, cut=cut)
                np.testing.assert_allclose(it.S, soq, atol=1e-10)
                np.testing.assert_allclose(it.G, pcf, atol=1e-10)
                np.testing.assert_allclose(it.g, pdf, atol=1e-10)
                np.testing.assert_allclose(it.N, run, atol=1e-9)
                g = pdf


class TestErrors(unittest.TestCase):
    def test_monte_carlo(self) -> None:
        """Propagated errors of the Lorch functions and of 2 back-FT iterations
        agree with the spread of transforms of noisy S(Q)."""
        rng = np.random.default_rng(5)
        cut = [0.8]
        start = fourier_with_errors(Q, S, SIG, R, RHO, window=1)
        last = back_fourier_iterations(Q, SIG, start, RHO, cut, 2)[-1]
        g_mc, n_mc, gb_mc, sb_mc = [], [], [], []
        for _ in range(300):
            sp = S + rng.normal(0.0, SIG)
            f = fourier_with_errors(Q, sp, SIG, R, RHO, window=1)
            b = back_fourier_iterations(Q, SIG, f, RHO, cut, 2)[-1]
            g_mc.append(f.G)
            n_mc.append(f.N)
            gb_mc.append(b.G)
            sb_mc.append(b.S)
        k = R > 1.0
        for mc, an in ((g_mc, start.dG), (n_mc, start.dN), (gb_mc, last.dG)):
            ratio = np.std(mc, axis=0)[k] / an[k]
            self.assertAlmostEqual(float(np.median(ratio)), 1.0, delta=0.1)
        ratio = np.std(sb_mc, axis=0)[Q > 1] / last.dS[Q > 1]
        self.assertAlmostEqual(float(np.median(ratio)), 1.0, delta=0.1)

    def test_error_relations(self) -> None:
        fs = fourier_with_errors(Q, S, SIG, R, RHO, window=0)
        np.testing.assert_allclose(fs.dT, fs.dG)
        np.testing.assert_allclose(fs.dRDF, R * fs.dG)
        np.testing.assert_allclose(fs.dg[1:], fs.dG[1:] / (4 * np.pi * RHO * R[1:]))
        self.assertEqual(fs.dN[0], 0.0)


class TestMasks(unittest.TestCase):
    def test_qmask_several_ranges(self) -> None:
        x = np.arange(0.0, 10.0, 0.5)
        m = qmask(x, [(1.0, 2.0), (5.0, 6.0)])
        np.testing.assert_array_equal(x[m], [1.0, 1.5, 2.0, 5.0, 5.5, 6.0])
        self.assertFalse(qmask(x, []).any())

    def test_ymask(self) -> None:
        y = np.array([0.0, 1.0, np.nan, 3.0])
        np.testing.assert_array_equal(ymask(y), [True, True, False, True])
        np.testing.assert_array_equal(ymask(y, (0.5, 3.0)), [False, True, False, True])

    def test_fit_prediction_error_linear_model(self) -> None:
        def line(x, a, b):
            return a + b * x

        x = np.array([0.0, 1.0, 2.0])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        expected = np.sqrt(cov[0, 0] + 2 * x * cov[0, 1] + x**2 * cov[1, 1])
        np.testing.assert_allclose(fit_prediction_error(line, x, [1.0, 2.0], cov), expected,
                                   rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
