"""
The correction example (``example/``): the MCAMS chain recovers the synthetic
truth within the error bars, from .adat and .qdat; the command line
(`toscana.workflows.correct_cli`) writes all its outputs; the CORRECT command
file of the example gives the same as run_correction(method="pycorrect").
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import unittest

import helpers
from helpers import in_tempdir

import numpy as np

_ = helpers
EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "example"
NAMES = {"sample": "I_SCE", "container": "I_CE", "environment": "I_E",
         "vanadium": "I_VE", "absorber": "I_ACE"}


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


class TestExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from toscana.workflows.correction import run_correction

        cls.res = {}
        with _in_example():
            cls.m = _measurement()
            for ext in ("adat", "qdat"):
                files = {k: f"data/{n}.{ext}" for k, n in NAMES.items()}
                files["empty"] = files["environment"]
                cls.res[ext] = run_correction(cls.m, files=files)
            cls.truth = {e: np.loadtxt(f"data/truth_{s}.dat")
                         for e, s in (("adat", "angle"), ("qdat", "q"))}

    def test_recovers_truth_within_errors(self) -> None:
        for ext, r in self.res.items():
            z = (r.dcs - self.truth[ext][:, 2]) / r.err
            self.assertLess(abs(z.mean()), 0.15, ext)
            self.assertGreater(z.std(), 0.85, ext)       # errors not overestimated
            self.assertLess(z.std(), 1.15, ext)          # nor underestimated
            np.testing.assert_allclose(r.intensities["I_v"], self.truth[ext][:, 5], rtol=0.05)

    def test_command_line(self) -> None:
        from toscana.workflows.correct_cli import main

        with in_tempdir() as tmp, _in_example():
            res = main(["Ni_powder.par", "--input", "qdat", "--outdir", str(tmp / "out"),
                        "--quiet", "--n-points", "20000", "--ms-particles", "200000"])
            out = sorted(p.name for p in (tmp / "out").iterdir())
            pyc = main(["Ni_powder.par", "--method", "pycorrect", "--outdir", str(tmp / "pc"),
                        "--quiet"])
            dcs = np.loadtxt(tmp / "out" / "Ni_powder_mcams_dcs.dat")
        self.assertEqual(res.method, "mcams")
        for name in ("Ni_powder_mcams.soq", "Ni_powder_mcams_dcs.dat",
                     "Ni_powder_mcams_sample_factors_q.dat", "Ni_powder_mcams_intensities_angle.dat",
                     "Ni_powder_mcams_vanadium_factors_angle.dat"):
            self.assertIn(name, out)
        np.testing.assert_allclose(dcs[:, 2], res.dcs, rtol=1e-6)
        self.assertEqual(pyc.method, "pycorrect")

    def test_com_file_equals_workflow(self) -> None:
        from toscana.corrections.pycorrect import Correct
        from toscana.workflows.correction import run_correction

        with _in_example():
            p = run_correction(self.m, method="pycorrect")
        with in_tempdir() as tmp:
            for f in list(NAMES.values()):
                (tmp / f"{f}.adat").write_bytes((EXAMPLE / "data" / f"{f}.adat").read_bytes())
            (tmp / "Ni_powder.com").write_bytes((EXAMPLE / "data" / "Ni_powder.com").read_bytes())
            c = Correct(verbose=False, ask=False)
            with contextlib.redirect_stdout(io.StringIO()):
                c.run_file("Ni_powder.com")
        # the .com file rounds radii and densities to 4-6 digits
        np.testing.assert_allclose(c.results[0]["dcs"], p.dcs, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
