"""mcams_correct (intensities before normalisation, multiple-scattering modes),
without_container, atoms_in_beam, regroup argument handling, save_columns."""

from __future__ import annotations

import contextlib
import io
import math
import unittest

import helpers
from helpers import in_tempdir

import numpy as np

import synthetic_d4 as syn

_ = helpers
OPTS = dict(n_points=4000, ms_particles=20000)


def _measurement(folder):
    from toscana.dataio.running_params import getRunningParams
    from toscana.experiment.measurement import Measurement

    syn.make_experiment(folder)
    with contextlib.redirect_stdout(io.StringIO()):
        return Measurement(getRunningParams("sample.par"))


class TestMcamsCorrect(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from toscana.workflows.correction import mcams_correct, without_container

        with in_tempdir() as folder:
            m = _measurement(folder)
            cls.m = m
            cls.full = {mode: mcams_correct(m, ms_mode=mode, factor_options=OPTS)
                        for mode in ("additive", "proportional", "none")}
            cls.bare = mcams_correct(without_container(m), ms_mode="proportional",
                                     factor_options=OPTS)

    def test_sorted_and_consistent(self) -> None:
        c = self.full["additive"]
        self.assertTrue(np.all(np.diff(c["q"]) > 0))
        np.testing.assert_allclose(c["I_s"], c["I_s_additive"])
        res = c["result"]
        np.testing.assert_allclose(c["I_s"], np.asarray(res.intensities["I_s"])[c["order"]])

    def test_modes(self) -> None:
        none, add, prop = (self.full[k]["I_s"] for k in ("none", "additive", "proportional"))
        np.testing.assert_allclose(none, self.full["additive"]["I_s_noMS"])
        k = self.full["none"]["q"] > 1
        self.assertTrue(np.all(prop[k] < none[k]))      # MS removed in both modes
        self.assertTrue(np.all(add[k] < none[k]))
        # the vanadium does not depend on the mode
        np.testing.assert_allclose(self.full["proportional"]["I_v"], self.full["none"]["I_v"])

    def test_proportional_single_configuration(self) -> None:
        """Without container (one configuration) the proportional mode is
        exactly I(no MS)/(1 + delta)."""
        c = self.bare
        fs = c["result"].factors
        self.assertEqual(list(fs.configurations), ["s"])
        d = np.asarray(fs.delta["delta_s"], float)[c["order"]]
        np.testing.assert_allclose(c["I_s"], c["I_s_noMS"] / (1 + d), rtol=1e-10)

    def test_invalid_mode(self) -> None:
        from toscana.workflows.correction import mcams_correct

        with self.assertRaises(ValueError):
            mcams_correct(self.m, ms_mode="sometimes")

    def test_without_container_does_not_modify(self) -> None:
        from toscana.workflows.correction import without_container

        m2 = without_container(self.m)
        self.assertEqual(m2.container[0], "None")
        self.assertNotEqual(self.m.container[0], "None")

    def test_atoms_in_beam(self) -> None:
        from toscana.workflows.correction import atoms_in_beam

        m = self.m
        n_s, n_v, info = atoms_in_beam(m)
        h = min(m.Height, m.beamHeight) if m.Fullness >= 1 else m.Fullness * m.beamHeight
        vol = math.pi * (m.OuterRadius**2 - m.InnerRadius**2) * h / 1000
        self.assertAlmostEqual(n_s, vol * 1e24 * m.EffAtomicDensity, delta=1e-9 * n_s)
        self.assertAlmostEqual(info["V_sample_cm3"], vol)
        self.assertGreater(n_v, 0)
        self.assertAlmostEqual(info["rho_vanadium"], 0.0722, delta=0.0005)


class TestDataio(unittest.TestCase):
    def test_save_columns(self) -> None:
        from toscana.dataio import save_columns

        with in_tempdir():
            with contextlib.redirect_stdout(io.StringIO()):
                save_columns("t.dat", [[1, 2], [3, 4]], ["a", "b"], ["head"])
            data = np.loadtxt("t.dat")
            np.testing.assert_allclose(data, [[1, 3], [2, 4]])
            with open("t.dat", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            self.assertEqual(lines[1], "# head")
            with self.assertRaises(ValueError):
                save_columns("u.dat", [[1, 2]], ["a", "b"])

    def test_regroup_skips_when_present(self) -> None:
        from toscana.dataio import regroup

        with in_tempdir():
            open("x.qdat", "w").close()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(regroup("missing.par", ["x.qdat"], force=False))


if __name__ == "__main__":
    unittest.main()
