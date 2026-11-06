"""Line shapes with error propagation, weighted sums, fitting-range helpers."""

from __future__ import annotations

import unittest
from typing import Callable

import helpers  # noqa: F401
from helpers import quiet

import numpy as np

from toscana.math.fitting import fit_range, fittingRange, fittingRange0
from toscana.math.line_shapes import (
    Gaussian,
    Gaussian_error,
    LorGau,
    LorGau_error,
    Lorentzian,
    Lorentzian_error,
)
from toscana.math.operations import wsum2

X = np.linspace(-4, 5, 37)


def numeric_error(f: Callable[..., np.ndarray], p: list[float], e: list[float]) -> np.ndarray:
    """Linear error propagation with central finite differences."""
    var = np.zeros_like(X)
    for k, (pk, ek) in enumerate(zip(p, e)):
        h = 1e-6 * max(1.0, abs(pk))
        up, dn = list(p), list(p)
        up[k] += h
        dn[k] -= h
        var += ((f(X, *up) - f(X, *dn)) / (2 * h) * ek) ** 2
    return np.sqrt(var)


class TestLineShapeErrors(unittest.TestCase):
    def test_lorentzian_error(self) -> None:
        p, e = [2.0, 0.3, 1.5, 0.1], [0.05, 0.02, 0.07, 0.01]
        val, err = Lorentzian_error(X, *p, *e)
        np.testing.assert_allclose(val, Lorentzian(X, *p))
        np.testing.assert_allclose(err, numeric_error(Lorentzian, p, e), rtol=1e-5)

    def test_gaussian_error(self) -> None:
        p, e = [2.0, 0.3, 1.5, 0.1], [0.05, 0.02, 0.07, 0.01]
        val, err = Gaussian_error(X, *p, *e)
        np.testing.assert_allclose(val, Gaussian(X, *p))
        np.testing.assert_allclose(err, numeric_error(Gaussian, p, e), rtol=1e-5)

    def test_lorgau_error(self) -> None:
        p, e = [1.7, 0.4, 1.2, 0.9, 0.05], [0.03, 0.02, 0.04, 0.05, 0.01]
        val, err = LorGau_error(X, *zip(p, e))
        np.testing.assert_allclose(val, LorGau(X, *p))
        np.testing.assert_allclose(err, numeric_error(LorGau, p, e), rtol=1e-5)


class TestWsum2(unittest.TestCase):
    d1 = np.array([[1.0, 10.0, 3.0], [2.0, 12.0, 4.0]])
    d2 = np.array([[1.0, 5.0, 4.0], [2.0, 6.0, 3.0]])

    def test_general_weights(self) -> None:
        out = wsum2(1.0, self.d1, -0.5, self.d2)
        np.testing.assert_allclose(out[:, 1], [7.5, 9.0])
        np.testing.assert_allclose(out[:, 2], np.sqrt(self.d1[:, 2] ** 2 + 0.25 * self.d2[:, 2] ** 2))

    def test_negative_single_weight_gives_positive_error(self) -> None:
        out = wsum2(0.0, self.d1, -1.0, self.d2)
        np.testing.assert_allclose(out[:, 1], -self.d2[:, 1])
        np.testing.assert_allclose(out[:, 2], self.d2[:, 2])

    def test_inverse_variance_weighting(self) -> None:
        out = wsum2(0.0, self.d1, 0.0, self.d2)
        w = self.d2[:, 2] ** 2 / (self.d1[:, 2] ** 2 + self.d2[:, 2] ** 2)
        np.testing.assert_allclose(out[:, 1], w * self.d1[:, 1] + (1 - w) * self.d2[:, 1])
        np.testing.assert_allclose(
            out[:, 2], self.d1[:, 2] * self.d2[:, 2] / np.hypot(self.d1[:, 2], self.d2[:, 2])
        )

    def test_invalid_mixing_weight_raises(self) -> None:
        with self.assertRaises(ValueError):
            quiet(wsum2, 2.0, self.d1, 0.0, self.d2)


class TestFittingRanges(unittest.TestCase):
    def test_three_variants_agree(self) -> None:
        rng = np.random.default_rng(3)
        x, y, e = rng.uniform(0, 10, 200), rng.uniform(0, 5, 200), rng.uniform(0.1, 1, 200)
        a = quiet(fittingRange0, 2, 8, 1, 4, x, y, e)
        b = np.hstack(quiet(fittingRange, 2, 8, 1, 4, x, y, e))
        c = quiet(fit_range, 2, 8, 1, 4, np.column_stack([x, y, e]))
        m = (x >= 2) & (x <= 8) & (y >= 1) & (y <= 4)
        np.testing.assert_array_equal(a, np.column_stack([x[m], y[m], e[m]]))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)


if __name__ == "__main__":
    unittest.main()
