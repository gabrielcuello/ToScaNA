"""Error bars of G(r), g(r), RDF(r), T(r), N(r) from sineFT(with_errors=True)."""

from __future__ import annotations

import unittest

import helpers
from helpers import in_tempdir, quiet, synthetic_liquid

import numpy as np

from toscana.dataio.saving import saveRSCF
from toscana.numerics.fourier import sineFT

_ = helpers
RHO = 0.1
_SQ, _, _ = synthetic_liquid(RHO)
Q = _SQ[::3, 0]  # 0.03 1/Å steps keep the Monte Carlo test fast
S = _SQ[::3, 1]
SIG = 0.01 * (1 + Q / 10)  # errors growing with Q, as in real data
SQE = np.column_stack([Q, S, SIG])
NR = np.round(np.arange(0, 6, 0.05), 10)


class TestErrorColumns(unittest.TestCase):
    def test_values_identical_to_plain_transform(self) -> None:
        for window in (0, 1):
            plain = sineFT(SQE[:, :2], NR, qmax=25.0, density=RHO, window=window)
            full = sineFT(SQE, NR, qmax=25.0, density=RHO, window=window, with_errors=True)
            self.assertEqual(full.shape, (len(NR), 11))
            np.testing.assert_array_equal(full[:, [0, 1, 3, 5, 7, 9]], plain)

    def test_error_relations(self) -> None:
        f = sineFT(SQE, NR, qmax=25.0, density=RHO, window=1, with_errors=True)
        r, d_g_func = f[:, 0], f[:, 2]
        pos = r > 0
        np.testing.assert_allclose(f[pos, 4], d_g_func[pos] / (4 * np.pi * RHO * r[pos]))
        self.assertEqual(f[0, 4], 0.0)
        np.testing.assert_allclose(f[:, 6], r * d_g_func)
        np.testing.assert_allclose(f[:, 8], d_g_func)
        self.assertEqual(f[0, 10], 0.0)
        self.assertTrue(np.all(np.diff(f[1:, 10]) > 0))  # σN grows with r

    def test_missing_error_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            sineFT(SQE[:, :2], NR, qmax=25.0, density=RHO, with_errors=True)

    def test_errors_scale_linearly(self) -> None:
        a = sineFT(SQE, NR, qmax=25.0, density=RHO, with_errors=True)
        doubled = SQE.copy()
        doubled[:, 2] *= 2
        b = sineFT(doubled, NR, qmax=25.0, density=RHO, with_errors=True)
        np.testing.assert_allclose(b[:, 2::2], 2 * a[:, 2::2])


class TestRunningSimpsonMatrix(unittest.TestCase):
    def test_matches_scipy_on_any_grid(self) -> None:
        from scipy import integrate

        from toscana.numerics.fourier import _running_simpson_matrix

        rng = np.random.default_rng(7)
        grids = [NR, np.arange(1.0, 3.0, 0.1), np.sort(rng.uniform(0, 5, 41)), np.array([0.0, 0.3])]
        for x in grids:
            c = _running_simpson_matrix(x)
            f = rng.standard_normal(len(x))
            expected = [0.0] + [integrate.simpson(f[: i + 1], x=x[: i + 1]) for i in range(1, len(x))]
            np.testing.assert_allclose(c @ f, expected, rtol=1e-12, atol=1e-13)


class TestAgainstMonteCarlo(unittest.TestCase):
    def test_analytic_errors_match_monte_carlo(self) -> None:
        n_samples = 1500
        rng = np.random.default_rng(12345)
        for window in (0, 1):
            analytic = sineFT(SQE, NR, qmax=25.0, density=RHO, window=window, with_errors=True)
            samples = np.empty((n_samples, len(NR), 6))
            for k in range(n_samples):
                perturbed = np.column_stack([Q, S + rng.standard_normal(len(Q)) * SIG])
                samples[k] = sineFT(perturbed, NR, qmax=25.0, density=RHO, window=window)
            mc_sigma = samples.std(axis=0, ddof=1)
            # statistical precision of a std from 1500 samples: ~1.8% (1σ)
            for col, name in ((1, "G"), (2, "g"), (3, "RDF"), (4, "T"), (5, "N")):
                ok = NR > (0.2 if name == "g" else 0.0)
                if name == "N":
                    ok &= NR > 0.5
                ratio = analytic[ok, 2 * col] / mc_sigma[ok, col]
                with self.subTest(window=window, function=name):
                    self.assertLess(float(np.max(np.abs(ratio - 1))), 0.08, name)
                    self.assertAlmostEqual(float(np.mean(ratio)), 1.0, delta=0.02)


class TestSaveRSCF(unittest.TestCase):
    def test_file_with_errors(self) -> None:
        f = sineFT(SQE, NR, qmax=25.0, density=RHO, window=1, with_errors=True)
        with in_tempdir() as tmp:
            quiet(saveRSCF, "out.rscf", f, ["Sample: test", "Qmax = 25"])
            text = (tmp / "out.rscf").read_text(encoding="utf-8")
            back = np.loadtxt(tmp / "out.rscf")
        self.assertEqual(back.shape, (len(NR), 11))
        np.testing.assert_allclose(back, f, atol=5e-7)
        header = [line for line in text.splitlines() if line.startswith("#")]
        self.assertIn("# Sample: test", header)
        for i, name in enumerate(["r", "G", "dG", "g", "dg", "RDF", "dRDF", "T", "dT", "N", "dN"]):
            self.assertTrue(
                any(line.split()[1:3] == [str(i + 1), name] for line in header if len(line.split()) > 2),
                name,
            )
        self.assertTrue(any("1σ statistical" in line for line in header))

    def test_file_without_errors_keeps_six_columns(self) -> None:
        f = sineFT(SQE[:, :2], NR, qmax=25.0, density=RHO)
        with in_tempdir() as tmp:
            quiet(saveRSCF, "out.rscf", f, [])
            back = np.loadtxt(tmp / "out.rscf")
            text = (tmp / "out.rscf").read_text(encoding="utf-8")
        self.assertEqual(back.shape, (len(NR), 6))
        self.assertNotIn("dG", text)

    def test_wrong_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            saveRSCF("never_written.txt", np.zeros((3, 5)), [])


if __name__ == "__main__":
    unittest.main()
