"""Isotopes, data files, Ni calibration set-up, experiment helpers, source hygiene."""

from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from typing import Any
from unittest import mock

import helpers
from helpers import in_tempdir, quiet

import numpy as np

from toscana.isotopes.core import Elemento

_ = helpers


class TestIsotopes(unittest.TestCase):
    def test_bound_cross_sections_in_barns(self) -> None:
        ni = Elemento("Ni")
        self.assertAlmostEqual(ni.scoh_bound.real, 4 * np.pi * 10.3**2 / 100, places=3)
        self.assertAlmostEqual(ni.scoh_bound.real / ni.sig_coh, 1.0, delta=0.01)

    def test_unknown_isotope_raises(self) -> None:
        with self.assertRaises(ValueError):
            Elemento("Xx99")

    @unittest.expectedFailure
    def test_vanadium_b_consistent_with_sigma_pending_C16(self) -> None:
        """C16 (pending decision): b_coh and σ_coh of V should agree."""
        v = Elemento("V")
        self.assertAlmostEqual(4 * np.pi * v.re_bcoh**2 / 100 / v.sig_coh, 1.0, delta=0.02)


class TestDataXYE(unittest.TestCase):
    def test_file_without_points_in_peak_range(self) -> None:
        from toscana.data.xy import DataXYE

        with in_tempdir() as tmp:
            f = tmp / "low.dat"
            f.write_text("# r-space\n" + "".join(f"{x:.2f} {np.sin(x):.5f}\n" for x in np.arange(0, 4, 0.1)))
            d = DataXYE(str(f))
        self.assertEqual(len(d.peaks_x), 0)
        self.assertTrue(np.isnan(d.xminr))

    def test_plot_saved_before_show(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from toscana.data.xy import DataXYE

        saved_lines: list[int] = []
        real_savefig = plt.savefig

        def spy_savefig(*args: Any, **kwargs: Any) -> None:
            saved_lines.append(len(plt.gca().get_lines()))
            real_savefig(*args, **kwargs)

        with in_tempdir() as tmp:
            (tmp.parent / "regdata").mkdir()
            f = tmp / "ni.adat"  # relative name: plot() saves to ../regdata/<basename>
            x = np.arange(5, 40, 0.1)
            f.write_text("".join(f"{a:.2f} {1 + 10 * np.exp(-(a - 14) ** 2):.5f} 0.1\n" for a in x))
            d = DataXYE("ni.adat")
            # an interactive show() closes the figure when the window is closed
            with mock.patch.object(plt, "show", lambda: plt.close("all")), \
                 mock.patch.object(plt, "savefig", spy_savefig):
                quiet(d.plot, "png")
        self.assertEqual(saved_lines, [1])


class TestNiCalibration(unittest.TestCase):
    def test_initial_amplitudes_mapping(self) -> None:
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
        initial = [float(i) for i in range(35)]
        nickel = types.SimpleNamespace(
            x=np.arange(10, 60, 0.5), y=np.ones(100), e=np.ones(100)
        )
        with mock.patch.dict(sys.modules, {"lmfit": fake}):
            quiet(fitting_model_d4nifit, nickel, [0, 100, 0, 10], initial)
        for i in range(10):
            self.assertEqual(captured[f"A{i}"], initial[5 + i], f"A{i}")
            self.assertEqual(captured[f"G{i}"], initial[15 + i], f"G{i}")
            self.assertEqual(captured[f"S{i}"], initial[25 + i], f"S{i}")


class TestMeasurementShowTable(unittest.TestCase):
    def test_stdout_restored_and_file_closed(self) -> None:
        from toscana.experiment.measurement import Measurement

        m = Measurement.__new__(Measurement)
        keys = ["<par>", "<ins>", "<cyc>", "<env>", "<pro>", "<mpr>", "<lco>", "<sta>",
                "<nbk>", "<pag>", "<wle>", "<zac>", "<apr>", "<top>", "<bot>", "<lft>", "<rgt>"]
        m.inputPar = {k: (k, "1") for k in keys}
        m.beamHeight, m.beamWidth = 50.0, 5.0
        buf = io.StringIO()
        with in_tempdir() as tmp, contextlib.redirect_stdout(buf):
            m.showTable(str(tmp / "table.txt"))
            self.assertIs(sys.stdout, buf)
            text = (tmp / "table.txt").read_text()
        self.assertIn("Input parameters", text)


class TestSourceHygiene(unittest.TestCase):
    def test_no_mojibake(self) -> None:
        root = helpers._pkg.ROOT
        # UTF-8 text decoded as cp1252: "Ã…" (Å), "Ëš" (˚), "Â±" (±), "Â³" (³)
        bad = ["\u00c3\u2026", "\u00cb\u0161", "\u00c2\u00b1", "\u00c2\u00b3"]
        offenders = [
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if "tests" not in p.parts
            and any(b in p.read_text(encoding="utf-8") for b in bad)
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
