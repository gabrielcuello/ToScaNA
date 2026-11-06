"""D4 reduction chain: corrections, regrouping, rebinning, parameter parsing."""

from __future__ import annotations

import pathlib
import unittest
from typing import Any
from unittest import mock

import helpers
from helpers import in_tempdir, quiet

import numpy as np

import toscana.io.acquisition as acquisition
import toscana.io.corrections as corrections
from toscana.io.d4creg_outputs import getDiffA
from toscana.io.numors import getNumorFiles
from toscana.math.binning import get_xlim, rebin
from toscana.math.operations import binary_sum

_ = helpers


def _fake_log(tmp: pathlib.Path) -> Any:
    log = tmp / "test.log"
    return mock.patch.multiple(
        corrections, d4creg_log_path=lambda: log
    ), mock.patch.object(acquisition, "d4creg_log_path", lambda: log)


class TestCorrections(unittest.TestCase):
    def test_binary_sum_variances(self) -> None:
        d1 = np.array([[1.0, 10.0, 2.0]])
        d2 = np.array([[1.0, 5.0, 2.0]])
        out = binary_sum(1.0, d1, -1.0, d2)
        self.assertAlmostEqual(out[0, 1], 5.0)
        self.assertAlmostEqual(out[0, 2], np.sqrt(8.0))

    def test_normalise_relative_error_independent_of_monitor(self) -> None:
        for mon in (1e5, 1e6, 1e7):
            c, e = corrections.normalise(
                np.array([[1e4]]), np.array([[100.0]]),
                corrections.getNormalFactors(mon, 60.0), "monitor", 60.0, 1e6,
            )
            self.assertAlmostEqual(c[0, 0], 1e4 * 1e6 / mon)
            self.assertAlmostEqual(e[0, 0] / c[0, 0], np.sqrt(1e-4 + 1 / mon), places=12)

    def test_normalise_zero_counts_flagged(self) -> None:
        c, e = corrections.normalise(
            np.array([[0.0]]), np.array([[0.0]]),
            corrections.getNormalFactors(1e6, 60.0), "monitor", 60.0, 1e6,
        )
        self.assertTrue(np.isnan(c[0, 0]) and np.isnan(e[0, 0]))

    def test_dead_time(self) -> None:
        with in_tempdir() as tmp:
            p1, p2 = _fake_log(tmp)
            with p1, p2:
                with self.assertRaises(ValueError):  # rate * tau = 1.4
                    corrections.getDTC_dcounts(np.array([[2e5, 10.0]]), 1.0, 1.0, 7e-6)
                raw = np.array([[1e4, 4e4]])
                out, total = corrections.getDTC_dcounts(raw, 5e4, 1.0, 7e-6)
                np.testing.assert_allclose(out, raw / (1 - raw * 7e-6))
                self.assertAlmostEqual(total, float(out.sum()))

    def test_efficiency_file_without_final_newline(self) -> None:
        with in_tempdir() as tmp:
            f = tmp / "eff.eff"
            f.write_text("# header\n1 1 0.9\n9 64 1.0234")
            eff = quiet(corrections.getEff, str(f))
            self.assertAlmostEqual(eff[0, 0], 0.9)
            self.assertAlmostEqual(eff[8, 63], 1.0234)

    def test_numor_list_expansion(self) -> None:
        self.assertEqual(
            getNumorFiles(["000010", "000020-000022"]),
            ["000010", "000020", "000021", "000022"],
        )
        with self.assertRaises(ValueError):
            getNumorFiles(["000022-000020"])


class TestOneNumor(unittest.TestCase):
    def test_poisson_errors_from_raw_counts(self) -> None:
        raw = np.full((9, 64), 400.0)
        head = {"ndet": 9, "ncell": 64, "Theta_lue(deg)": 10.0, "Theta_des(deg)": 10.0,
                "MonitorCnts": 1e6, "CntTime (sec)": 60.0, "TotalCnts": raw.sum(),
                "RtrPower (MW)": 52}
        with in_tempdir() as tmp:
            eff = tmp / "eff.eff"
            eff.write_text("".join(f"{d} {c} 0.8\n" for d in range(1, 10) for c in range(1, 65)))
            run_info = {"writeNumor": 0, "efffile": str(eff), "decfile": "zeros",
                        "normalisation": ("monitor", 60, 1e6), "dead_time": (0.0, 0.0),
                        "cellTolerance": (0.1, 1.0), "dataFormat": "Nexus",
                        "ext": ("", "", "", "", ".nxs", ""), "angular_range_bank": 8.0,
                        "wavelength": 0.5}
            p1, p2 = _fake_log(tmp)
            with p1, p2, mock.patch.object(
                acquisition, "load_nxs", lambda p, ncell=64: (dict(head), raw.copy())
            ):
                _x, y, e, _q, _c, _h = quiet(acquisition.getOneNumor, "f", run_info)
        # counts: 400/0.8 = 500; error: sqrt(400)/0.8 (+ monitor term 500*1e-3)
        self.assertAlmostEqual(y[0], 500.0)
        self.assertAlmostEqual(e[0], np.sqrt((20 / 0.8) ** 2 + (500 * 1e-3) ** 2))


def _numors(n: int = 2, shift: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(1)
    numor = np.zeros((n, 5, 576))
    for k in range(n):
        numor[k, 0, :] = 10 + 0.125 * np.arange(576) + shift * k
        numor[k, 1, :] = rng.uniform(50, 150, 576)
        numor[k, 2, :] = 3.0
        numor[k, 4, :] = 1001 + 1000 * (np.arange(576) // 64) + np.arange(576) % 64
    return numor


class TestGetDiffA(unittest.TestCase):
    run_info = {"angular_scale": (0.0, 140.0, 0.125), "angular_range_bank": 8.0,
                "wavelength": 0.5}
    head = [{"ndet": 9, "ncell": 64, "MonitorCnts": 1e6}] * 2

    def test_constant_input_gives_constant_output(self) -> None:
        numor = _numors()
        numor[:, 1, :] = 7.0
        diff = quiet(getDiffA, numor, self.head, self.run_info)
        vals = diff[0, 1][~np.isnan(diff[0, 1])]
        np.testing.assert_allclose(vals, 7.0)

    def test_errors_are_combined_not_averaged(self) -> None:
        diff = quiet(getDiffA, _numors(), self.head, self.run_info)
        err = diff[0, 2][~np.isnan(diff[0, 2])]
        self.assertTrue(np.all(err <= 3.0 + 1e-12))
        self.assertLess(float(np.median(err)), 2.5)

    def test_non_commensurate_step(self) -> None:
        info = dict(self.run_info, angular_scale=(1.0, 140.0, 0.3))
        diff = quiet(getDiffA, _numors(), self.head, info)
        self.assertEqual(diff.shape[2], len(np.arange(1.0, 140.0, 0.3)))

    def test_out_of_range_cells_are_dropped(self) -> None:
        numor = _numors()
        numor[:, 0, 0] = -5.0
        numor[:, 0, -1] = 200.0
        diff = quiet(getDiffA, numor, self.head, self.run_info)
        self.assertTrue(np.all(np.isnan(diff[0, 1, -45:])))


class TestRebin(unittest.TestCase):
    def test_bin_count(self) -> None:
        self.assertEqual(get_xlim(0.0, 0.3, 0.1)[0], 4)
        self.assertEqual(get_xlim(0.0, 0.25, 0.1)[0], 3)

    def test_error_of_mean(self) -> None:
        x = np.arange(0, 10, 0.01)
        data = np.column_stack([x, np.full_like(x, 100.0), np.full_like(x, 10.0)])
        out = quiet(rebin, 0.01, 0.0, data, 1.0, 8.0, 0.1)
        np.testing.assert_allclose(out[5:-5, 1], 100.0)
        np.testing.assert_allclose(out[5:-5, 2], 10 / np.sqrt(10), rtol=0.05)


class TestWorkflow(unittest.TestCase):
    def test_read_param_and_d4creg_end_to_end(self) -> None:
        rng = np.random.default_rng(0)

        def fake_nxs(path: str, ncell: int = 64) -> tuple[dict[str, Any], np.ndarray]:
            n = int(path.split("/")[-1][:6]) - 100000
            meta = {"ndet": 9, "ncell": 64, "Theta_lue(deg)": 1.5 + 4 * n,
                    "Theta_des(deg)": 1.5 + 4 * n, "MonitorCnts": 2e5, "CntTime (sec)": 60.0,
                    "TotalCnts": 0.0, "RtrPower (MW)": 52, "label": "d4 x", "user": "u",
                    "LC": "lc", "proposal": "p", "subtitle": "s", "numor": str(100000 + n),
                    "startTime": "", "endTime": ""}
            return meta, rng.poisson(1000, (9, 64)).astype(float)

        from toscana.io.parameters import readParam

        with in_tempdir() as tmp, mock.patch.object(acquisition, "load_nxs", fake_nxs):
            (tmp / "par.txt").write_text(
                "# test\n<fmt> Nexus\n<eff> ones\n<dec> zeros   # comment\n"
                "<num> 100000-100003\n<asc> 0.0 140.0 0.25\n<qsc> 0.3 23.0 0.02\n"
                "<wle> 0.5\n<zac> 0.0\n<wri> False\n<out> e2e\n<run>\n"
            )
            info = quiet(readParam, "par.txt")
            qdat = np.loadtxt(tmp / "e2e.qdat")
            adat = np.loadtxt(tmp / "e2e.adat")
        self.assertEqual(info["dataFormat"], "Nexus")
        self.assertEqual(info["decfile"], "zeros")
        self.assertEqual(info["numorLst"], ["100000-100003"])
        self.assertGreater(len(adat), 100)
        self.assertAlmostEqual(qdat[0, 0], 0.3)
        self.assertTrue(np.all(np.isfinite(qdat)))


if __name__ == "__main__":
    unittest.main()
