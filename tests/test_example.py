"""
The correction example (``example/``), full Paalman & Pings formalism: the
furnace is a body (configurations h, ch, sch, vh) and the absorber
measurement is a configuration (ach) of the system solved for B, I_h, I_c, K.

* noise free (reduced statistics, forward model of the example): every
  unknown recovered to rounding error, also with an extra empty-instrument
  measurement (over-determined system, weighted least squares);
* the example data (noise): truth recovered within the error bars;
* command line (`toscana.workflows.correct_cli`) outputs;
* the CORRECT command file of the example equals run_correction(method="pycorrect").
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import unittest

import helpers
from helpers import in_tempdir

import numpy as np

_ = helpers
EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "example"
NAMES = {"sample": "I_SCE", "container": "I_CE", "environment": "I_E",
         "vanadium": "I_VE", "absorber": "I_ACE"}
FAST = dict(n_points=20000, ms_particles=200000)
FULL = dict(environment="layer", vanadium_in_environment=True)


@contextlib.contextmanager
def _in_example():
    old = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        yield
    finally:
        os.chdir(old)


def _measurement():
    from toscana.dataio.running_params import getRunningParams
    from toscana.experiment.measurement import Measurement

    with contextlib.redirect_stdout(io.StringIO()):
        return Measurement(getRunningParams("Ni_powder.par"))


def _generator():
    if str(EXAMPLE) not in sys.path:
        sys.path.insert(0, str(EXAMPLE))
    import make_synthetic_data

    return make_synthetic_data


class TestExactNoiseFree(unittest.TestCase):
    def test_system_recovers_everything(self) -> None:
        from toscana.corrections.factors import compute_factors
        from toscana.corrections.setup import (absorber_setup_from_measurement,
                                               setups_from_measurement)
        from toscana.workflows.correction import run_correction

        gen = _generator()
        with _in_example():
            m = _measurement()
        theta = gen.THETA[::8]
        ss, vs = setups_from_measurement(m, **FULL)
        setups = {"sample": ss, "vanadium": vs,
                  "absorber": absorber_setup_from_measurement(m, environment="layer")}
        factors = {"sample": compute_factors(ss, theta, **FAST),
                   "vanadium": compute_factors(vs, theta, configurations=["vh"], **FAST),
                   "absorber": compute_factors(setups["absorber"], theta, configurations=["ach"],
                                               **FAST)}
        _q, meas, truth = gen.forward(m, theta, factors, setups)
        meas["I_room"] = truth["B"]
        with in_tempdir() as tmp:
            for name, y in meas.items():
                gen.write_regrouped(tmp / f"{name}.adat", theta, y, np.sqrt(y), "angle")
            files = {k: str(tmp / f"{n}.adat") for k, n in NAMES.items()}
            pre = dict(factors=factors["sample"], vanadium_factors=factors["vanadium"],
                       absorber_factors=factors["absorber"])
            r = run_correction(m, files=files, **FULL, **pre)
            r2 = run_correction(m, files=dict(files, empty=str(tmp / "I_room.adat")), **FULL,
                                **pre)
        for res in (r, r2):
            np.testing.assert_allclose(res.dcs, truth["dcs"], rtol=1e-6)
            np.testing.assert_allclose(res.intensities["I_h"], truth["I_h"], rtol=1e-6)
            np.testing.assert_allclose(res.intensities["I_c"], truth["I_c"], rtol=1e-6)
            np.testing.assert_allclose(res.intensities["K"], truth["K"], rtol=1e-6)
            np.testing.assert_allclose(res.backgrounds["B"], truth["B"], rtol=1e-6)
        self.assertTrue(np.all(r2.err <= r.err * (1 + 1e-9)))   # one more measurement: smaller errors
        self.assertEqual(sorted(r.absorber_factors.A), ["A_a,ach", "A_c,ach", "A_h,ach"])

    def test_configuration_needs_environment_body(self) -> None:
        from toscana.workflows.correction import run_correction

        with _in_example():
            m = _measurement()
            with self.assertRaises(ValueError):
                run_correction(m, environment="background", absorber_model="configuration")


class TestExampleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from toscana.workflows.correction import run_correction

        with _in_example():
            cls.m = _measurement()
            files = {k: f"data/{n}.adat" for k, n in NAMES.items()}
            cls.res = run_correction(cls.m, files=files, **FULL)
            cls.truth = np.loadtxt("data/truth_angle.dat")

    def test_recovers_truth_within_errors(self) -> None:
        z = (self.res.dcs - self.truth[:, 2]) / self.res.err
        self.assertLess(abs(z.mean()), 0.15)
        self.assertGreater(z.std(), 0.85)          # errors not overestimated
        self.assertLess(z.std(), 1.15)             # nor underestimated
        # room background: a small difference of large numbers, check it in error bars
        zb = (self.res.backgrounds["B"] - self.truth[:, 8]) / self.res.backgrounds["B_err"]
        self.assertLess(abs(zb.mean()), 0.15)
        self.assertLess(abs(zb.std() - 1), 0.15)

    def test_qdat_input(self) -> None:
        from toscana.workflows.correction import run_correction

        with _in_example():
            files = {k: f"data/{n}.qdat" for k, n in NAMES.items()}
            r = run_correction(self.m, files=files, factor_options=FAST, **FULL)
            truth = np.loadtxt("data/truth_q.dat")
        # reduced Monte Carlo statistics: within ~1 % of the truth on average
        self.assertLess(abs(float(np.median(r.dcs / truth[:, 2])) - 1), 0.01)

    def test_command_line(self) -> None:
        from toscana.workflows.correct_cli import main

        with in_tempdir() as tmp, _in_example():
            res = main(["Ni_powder.par", "--environment", "layer", "--vanadium-in-environment",
                        "--outdir", str(tmp / "out"), "--quiet", "--n-points", "20000",
                        "--ms-particles", "200000"])
            out = sorted(p.name for p in (tmp / "out").iterdir())
            dcs = np.loadtxt(tmp / "out" / "Ni_powder_mcams_dcs.dat")
        self.assertEqual(res.method, "mcams")
        for name in ("Ni_powder_mcams.soq", "Ni_powder_mcams_dcs.dat",
                     "Ni_powder_mcams_sample_factors_q.dat", "Ni_powder_mcams_intensities_angle.dat",
                     "Ni_powder_mcams_absorber_factors_angle.dat",
                     "Ni_powder_mcams_vanadium_factors_q.dat"):
            self.assertIn(name, out)
        np.testing.assert_allclose(dcs[:, 2], res.dcs, rtol=1e-6)

    def test_com_file_equals_workflow(self) -> None:
        from toscana.corrections.pycorrect import Correct
        from toscana.workflows.correction import run_correction

        with _in_example():
            p = run_correction(self.m, method="pycorrect", environment="furnace",
                               vanadium_in_environment=True,
                               files={"background": "data/I_ACE.adat"})
        with in_tempdir() as tmp:
            for f in NAMES.values():
                (tmp / f"{f}.adat").write_bytes((EXAMPLE / "data" / f"{f}.adat").read_bytes())
            (tmp / "Ni_powder.com").write_bytes((EXAMPLE / "data" / "Ni_powder.com").read_bytes())
            c = Correct(verbose=False, ask=False)
            with contextlib.redirect_stdout(io.StringIO()):
                c.run_file("Ni_powder.com")
        self.assertEqual(p.vanadium_factors.configurations, ["h", "vh"])
        # the .com file rounds radii and densities to 4-6 digits
        np.testing.assert_allclose(c.results[0]["dcs"], p.dcs, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
