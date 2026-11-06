"""Group B physical choices: annulus volume, free cross sections, sigmoid,
shared Q0 default, weighted Ni fit."""

from __future__ import annotations

import inspect
import sys
import types
import unittest
from typing import Any
from unittest import mock

import helpers
from helpers import in_tempdir, quiet

import numpy as np

from toscana.isotopes import Element
from toscana.models.scattering import inelastic, vanaQdep
from toscana.numerics.line_shapes import SIGMOID_Q0, siglin, sigmoidal
from toscana.physics.properties import getMassNumber

_ = helpers


def _free(symbols_conc: dict[str, float], attr: str = "sig_sca") -> float:
    total = 0.0
    for sym, c in symbols_conc.items():
        e = Element(sym)
        a = getMassNumber(e.weight)  # m_i / m_n
        total += c * getattr(e, attr) * (a / (a + 1)) ** 2
    return total


def _water_measurement(inner_diam: float, outer_diam: float, height: float) -> Any:
    from toscana.experiment.measurement import Measurement

    with in_tempdir() as tmp:
        data = tmp / "d.dat"
        data.write_text("1 1 0.1\n2 1 0.1\n")
        par: dict[str, tuple[Any, ...]] = {}
        for key in ["<par>", "<ins>", "<pro>", "<mpr>", "<lco>", "<sta>", "<end>", "<cyc>",
                    "<nbk>", "<pag>", "<env>", "<Cmater>", "<Amater>", "<Emater>",
                    "<Sdescr>", "<Stitle>"]:
            par[key] = (key, "x")
        for key in ["<zac>", "<wle>", "<lft>", "<rgt>", "<top>", "<bot>", "<apr>", "<StempK>",
                    "<Smassc>", "<Sdegcc>", "<Sfulln>"]:
            par[key] = (key, "1.0")
        for key in ["<asc>", "<qsc>", "<rsc>"]:
            par[key] = (key, "0", "10", "0.1")
        for key in ["<Cshape>", "<Nshape>", "<Ashape>", "<Eshape>"]:
            par[key] = (key, "cylinder", str(outer_diam), "7", "60")
        par["<Sshape>"] = ("<Sshape>", "cylinder", str(inner_diam), str(outer_diam), str(height))
        for key in ["<CoFile>", "<NoFile>", "<EnFile>", "<SaFile>", "<AbFile>"]:
            par[key] = (key, str(data))
        par["<SNelem>"] = ("n", 2)
        par["<Sato01>"] = ("Atom 01", "H", "2")
        par["<Sato02>"] = ("Atom 02", "O", "1")
        return quiet(Measurement, par)


class TestMeasurement(unittest.TestCase):
    def test_annulus_volume(self) -> None:
        m = _water_measurement(4.0, 6.0, 50.0)
        self.assertAlmostEqual(m.Volume, np.pi * 50.0 * (3.0**2 - 2.0**2) / 1000.0, places=12)

    def test_solid_cylinder_volume_unchanged(self) -> None:
        m = _water_measurement(0.0, 6.0, 50.0)
        self.assertAlmostEqual(m.Volume, np.pi * 50.0 * 3.0**2 / 1000.0, places=12)

    def test_free_cross_section_per_element(self) -> None:
        m = _water_measurement(0.0, 6.0, 50.0)
        expected = _free({"H": 2 / 3, "O": 1 / 3})
        self.assertAlmostEqual(m.free, expected, places=10)
        # Legacy (mean-mass) value is far off for water: σ_free(H) is ~1/4 of bound
        legacy = m.ssca * (m.molarMass / (m.molarMass + Element("n1").weight)) ** 2
        self.assertLess(m.free / legacy, 0.5)


class TestSetSampleInfo(unittest.TestCase):
    def test_free_cross_sections_per_element(self) -> None:
        from toscana.experiment.info import setSampleInfo

        conc = {"H": 2 / 3, "O": 1 / 3}
        elements = {k: Element(k) for k in conc}
        s = quiet(setSampleInfo, {"H2O": 1}, {"H": 2, "O": 1}, 3, elements, conc)
        for key, attr in (("FreeScaXS", "sig_sca"), ("FreeIncXS", "sig_inc"), ("FreeCohXS", "sig_coh")):
            self.assertAlmostEqual(s[key], _free(conc, attr), places=10, msg=key)


class TestSigmoid(unittest.TestCase):
    q = np.linspace(-50, 1e4, 20001)

    def test_limits_midpoint_monotonic_no_overflow(self) -> None:
        with np.errstate(all="raise"):
            f = sigmoidal(self.q, lowQ=0.4, highQ=0.2, Q0=7.0, dQ=2.4)
        self.assertAlmostEqual(float(f[0]), 0.4, places=8)
        self.assertAlmostEqual(float(f[-1]), 0.2, places=12)
        self.assertAlmostEqual(float(sigmoidal(7.0, 0.4, 0.2, 7.0, 2.4)), 0.3, places=15)
        self.assertTrue(np.all(np.diff(f) <= 0))
        self.assertTrue(np.all((f >= 0.2) & (f <= 0.4)))

    def test_siglin_adds_linear_term(self) -> None:
        q = np.linspace(0, 25, 51)
        np.testing.assert_allclose(siglin(q, a1=0.003), sigmoidal(q) + 0.003 * q)
        np.testing.assert_allclose(siglin(q, a1=0.0), sigmoidal(q))

    def test_inelastic_is_the_same_logistic(self) -> None:
        q = np.linspace(0, 25, 51)
        ratio = (51.0 / 52.0) ** 2
        np.testing.assert_allclose(
            inelastic(q, A=51.0, lowQ=0.4), sigmoidal(q, lowQ=0.4, highQ=0.4 * ratio), rtol=1e-12
        )

    def test_single_q0_default(self) -> None:
        self.assertEqual(SIGMOID_Q0, 7.0)
        for func in (sigmoidal, siglin, inelastic, vanaQdep):
            self.assertEqual(inspect.signature(func).parameters["Q0"].default, 7.0, func.__name__)


class TestWeightedNiFit(unittest.TestCase):
    def _capture(self, e: np.ndarray) -> dict[str, Any]:
        from toscana.calibration.d4nifit import fitting_model_d4nifit

        captured: dict[str, Any] = {}

        class FakeModel:
            def __init__(self, func: Any) -> None:
                pass

            def set_param_hint(self, *a: Any, **k: Any) -> None:
                pass

            def fit(self, y: Any, **kwargs: Any) -> str:
                captured.update(kwargs)
                return "result"

        fake = types.ModuleType("lmfit")
        fake.Model = FakeModel  # type: ignore[attr-defined]
        x = np.arange(10, 60, 0.5)
        nickel = types.SimpleNamespace(x=x, y=np.ones_like(x), e=e)
        with mock.patch.dict(sys.modules, {"lmfit": fake}):
            quiet(fitting_model_d4nifit, nickel, [0, 100, 0, 10], [0.0] * 35)
        return captured

    def test_weights_are_inverse_errors(self) -> None:
        e = np.linspace(0.01, 0.05, 100)
        np.testing.assert_allclose(self._capture(e)["weights"], 1 / e)

    def test_no_valid_errors_means_unweighted(self) -> None:
        self.assertIsNone(self._capture(np.full(100, -1.0))["weights"])


if __name__ == "__main__":
    unittest.main()
