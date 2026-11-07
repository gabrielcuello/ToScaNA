"""Correction workflow: Measurement -> pycorrect (-> MCAMS) -> S(Q)."""

from __future__ import annotations

import contextlib
import io
import math
import unittest

import helpers
from helpers import in_tempdir, quiet

import numpy as np

import synthetic_d4 as syn

_ = helpers


def _measurement(folder, **kw):
    from toscana.dataio.running_params import getRunningParams
    from toscana.experiment.measurement import Measurement

    truth = syn.make_experiment(folder, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        m = Measurement(getRunningParams("sample.par"))
    return m, truth


class TestEndToEnd(unittest.TestCase):
    """Synthetic data built with the physics CORRECT inverts, known S(Q)."""

    @classmethod
    def setUpClass(cls) -> None:
        from toscana.workflows.correction import run_correction, structure_factor

        cls._tmp = in_tempdir()
        cls.folder = cls._tmp.__enter__()
        cls.m, cls.truth = _measurement(cls.folder)
        cls.result = run_correction(cls.m, placzek="none")
        cls.sq = structure_factor(cls.result, cls.m)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.__exit__(None, None, None)

    def test_recovers_true_structure_factor(self) -> None:
        diff = self.sq[:, 1] - syn.s_true(self.sq[:, 0])
        self.assertLess(float(np.abs(diff).max()), 1e-3)
        self.assertAlmostEqual(float(self.sq[self.sq[:, 0] > 15, 1].mean()), 1.0, delta=1e-3)

    def test_correct_factors_match_the_forward_model(self) -> None:
        self.assertAlmostEqual(self.result.factors["delta_sa"], self.truth["delta_sa"], places=6)
        self.assertAlmostEqual(self.result.factors["delta_va"], self.truth["delta_va"], places=6)
        self.assertEqual(self.result.source, "pycorrect")

    def test_errors_propagated(self) -> None:
        # sigma_S = sigma(dsigma/dOmega) / <b>^2, point by point (sorted by Q)
        order = np.argsort(self.result.q)
        np.testing.assert_allclose(self.sq[:, 2], self.result.err[order] / self.truth["b2"])

    def test_regular_q_grid(self) -> None:
        from toscana.workflows.correction import structure_factor

        sq = structure_factor(self.result, self.m, q_grid=(0.5, 23.0, 0.05))
        np.testing.assert_allclose(np.diff(sq[:, 0]), 0.05)
        inside = sq[:, 1] != 0
        self.assertLess(float(np.abs(sq[inside, 1] - syn.s_true(sq[inside, 0])).max()), 0.02)

    def test_feeds_sinefT_with_errors(self) -> None:
        from toscana.numerics.fourier import sineFT

        r = np.round(np.arange(0, 10, 0.02), 10)
        fou = sineFT(self.sq, r, qmax=20.0, density=self.m.AtomicDensity, window=1, with_errors=True)
        self.assertEqual(fou.shape, (len(r), 11))
        self.assertTrue(np.all(np.isfinite(fou)))

    def test_save_structure_factor(self) -> None:
        from toscana.workflows.correction import save_structure_factor

        with in_tempdir() as tmp:
            save_structure_factor("ni.soq", self.sq, ["synthetic Ni"])
            back = np.loadtxt(tmp / "ni.soq")
            text = (tmp / "ni.soq").read_text()
        self.assertEqual(back.shape, self.sq.shape)
        self.assertIn("S(Q) (Faber-Ziman", text)


class TestMcams(unittest.TestCase):
    def test_mcams_delta_replaces_multiple_scattering(self) -> None:
        from toscana.workflows.correction import run_correction

        with in_tempdir() as folder:
            m, _truth = _measurement(folder)
            opts = dict(n_particles=20000, n_batches=3, ms_particles=8000)
            ref = run_correction(m, placzek="none")
            r = run_correction(m, placzek="none", mcams="delta", mcams_options=opts)
            r_all = run_correction(m, placzek="none", mcams="all", mcams_options=opts)
        self.assertEqual(r.source, "mcams-delta")
        self.assertEqual(np.shape(r.factors["delta_sa"]), r.theta.shape)
        # Known result (docs/validation): CORRECT's series for higher orders is too low
        self.assertGreater(float(np.mean(r.factors["delta_sa"])), ref.factors["delta_sa"])
        # attenuation unchanged with "delta", replaced with "all" (within MC noise ~1 %)
        np.testing.assert_array_equal(r.factors["assch"], ref.factors["assch"])
        self.assertFalse(np.array_equal(r_all.factors["assch"], ref.factors["assch"]))
        np.testing.assert_allclose(r_all.factors["assch"], ref.factors["assch"], rtol=0.03)
        self.assertIsNotNone(r.mcams)
        self.assertIn("delta_sa_err", r.mcams)


class TestBuildState(unittest.TestCase):
    def test_mapping_from_measurement(self) -> None:
        from toscana.isotopes import Element
        from toscana.workflows.correction import build_state, material_properties

        with in_tempdir() as folder:
            m, _ = _measurement(folder)
            st = build_state(m)
        self.assertAlmostEqual(st.ca_r1, 0.30)        # 6 mm diameter -> 0.3 cm radius
        self.assertAlmostEqual(st.ca_r2, 0.32)
        self.assertAlmostEqual(st.va_r, 0.30)
        self.assertAlmostEqual(st.beam_width, 1.5)
        self.assertAlmostEqual(st.beam_height, 5.0)
        self.assertAlmostEqual(st.sa_rho, m.EffAtomicDensity)
        ni = Element("Ni")
        self.assertEqual((st.species[0].scatt_cs, st.species[0].abs_cs), (ni.sig_sca, ni.sig_abs))
        self.assertEqual((st.va_rho, st.va_scat_cs, st.va_abs_cs), material_properties("V"))
        self.assertEqual(st.files["bkg_sample"], "background.adat")
        self.assertEqual(st.placzek_correction, "SERIES_EXPANSION")

    def test_material_properties(self) -> None:
        from toscana.experiment.info import VANADIUM_ATOMIC_DENSITY
        from toscana.workflows.correction import material_properties

        rho, s, a = material_properties("V")
        self.assertAlmostEqual(rho, VANADIUM_ATOMIC_DENSITY, places=12)
        self.assertAlmostEqual(s, 5.10466)
        rho, s, a = material_properties("TiZr")   # CORRECT preset: 0.052, 5.01, 4.14
        self.assertAlmostEqual(rho, 0.052, delta=0.002)
        self.assertAlmostEqual(s, 5.01, delta=0.1)
        self.assertAlmostEqual(a, 4.14, delta=0.1)
        rho, s, a = material_properties(({"Al": 1}, 2.70))
        self.assertAlmostEqual(rho, 2.70 * 0.602214076 / 26.9815, places=4)
        with self.assertRaises(KeyError):
            material_properties("Unobtainium")

    def test_invalid_options(self) -> None:
        from toscana.workflows.correction import build_state, run_correction

        with in_tempdir() as folder:
            m, _ = _measurement(folder)
            with self.assertRaises(ValueError):
                build_state(m, environment="oven")
            with self.assertRaises(ValueError):
                build_state(m, placzek="maybe")
            with self.assertRaises(ValueError):
                run_correction(m, mcams="sometimes")
            m.InnerDiam = 2.0
            with self.assertRaises(ValueError):
                build_state(m)


class TestPycorrectAdditions(unittest.TestCase):
    def test_factor_override_validation(self) -> None:
        from toscana.corrections.pycorrect.correct_2axis import correct_2axis
        from toscana.workflows.correction import build_state

        with in_tempdir() as folder:
            m, _ = _measurement(folder)
            st = build_state(m, placzek="none")
            with self.assertRaises(ValueError):
                quiet(correct_2axis, st, False, {"unknown": 1.0})
            with self.assertRaises(ValueError):
                quiet(correct_2axis, st, False, {"assch": np.ones(5)})
            r0 = correct_2axis(st, verbose=False)[0]
            r1 = correct_2axis(st, verbose=False, factors={"delta_sa": r0["delta_sa"]})[0]
        np.testing.assert_array_equal(r0["dcs"], r1["dcs"])   # same value -> same result

    def test_reads_toscana_adat(self) -> None:
        from toscana.corrections.pycorrect.fileio import read_file

        with in_tempdir() as folder:
            syn.make_experiment(folder)
            b = read_file(str(folder / "sample.adat"), 1)
            a = np.loadtxt(folder / "sample.adat")
        np.testing.assert_array_equal(b.x, a[:, 0])
        np.testing.assert_array_equal(b.y, a[:, 1])
        self.assertEqual(b.nrun, 1)


if __name__ == "__main__":
    unittest.main()
