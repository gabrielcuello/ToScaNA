"""D4 reduction chain: corrections, regrouping, rebinning, parameter parsing."""

from __future__ import annotations

import pathlib
import unittest
from typing import Any
from unittest import mock

import helpers
from helpers import in_tempdir, quiet

import numpy as np

import toscana.dataio.acquisition as acquisition
import toscana.dataio.corrections as corrections
from toscana.dataio.d4creg_outputs import getDiffA
from toscana.dataio.numors import getNumorFiles
from toscana.numerics.binning import get_xlim, rebin
from toscana.numerics.operations import binary_sum

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


def _one_cell(center: float, step: float) -> np.ndarray:
    """Regroup a single cell (width 8/64 = 0.125 deg) centred at `center`."""
    numor = np.zeros((1, 5, 1))
    numor[0, 0, 0] = center
    numor[0, 1, 0] = 1.0
    numor[0, 2, 0] = 1.0
    numor[0, 4, 0] = 1001
    info = {"angular_scale": (0.0, 20.0, step), "angular_range_bank": 8.0, "wavelength": 0.5}
    return quiet(getDiffA, numor, [{"ndet": 9, "ncell": 64, "MonitorCnts": 1e6}], info)


def _overlap_weights(centers: np.ndarray, width: float, edges: np.ndarray) -> np.ndarray:
    """Independent reference: overlap of each cell with each bin / cell width."""
    lo, hi = centers - width / 2, centers + width / 2
    weights = np.zeros(len(edges) - 1)
    for j in range(len(edges) - 1):
        overlap = np.clip(np.minimum(hi, edges[j + 1]) - np.maximum(lo, edges[j]), 0, None)
        weights[j] = overlap.sum() / width
    return weights


class TestGetDiffASplit(unittest.TestCase):
    """Cells split between bins by their overlap (bug found while documenting)."""

    def test_cell_straddling_an_edge_with_larger_step(self) -> None:
        d = _one_cell(10.25, 0.25)  # cell [10.1875, 10.3125], edge at 10.25
        i = 40  # bin [10.00, 10.25)
        np.testing.assert_allclose(d[0, 3, i : i + 2], [0.5, 0.5])

    def test_cell_inside_one_bin(self) -> None:
        d = _one_cell(10.125, 0.25)  # [10.0625, 10.1875] inside [10.0, 10.25)
        self.assertAlmostEqual(d[0, 3, 40], 1.0)
        self.assertTrue(np.isnan(d[0, 1, 41]))  # neighbouring bin stays empty

    def test_weights_equal_overlaps_for_several_steps(self) -> None:
        numor = _numors(n=1)
        cells = numor[0, 0, :]
        for step in (0.125, 0.2, 0.25, 0.3, 0.5):
            info = dict(TestGetDiffA.run_info, angular_scale=(0.0, 140.0, step))
            d = quiet(getDiffA, numor, TestGetDiffA.head[:1], info)
            edges = np.append(d[0, 0, :], d[0, 0, -1] + step)
            expected = _overlap_weights(cells, 0.125, edges)
            with self.subTest(step=step):
                np.testing.assert_allclose(d[0, 3, :], expected, atol=1e-12)
                # every cell is fully inside the grid: total weight = number of cells
                self.assertAlmostEqual(float(d[0, 3, :].sum()), len(cells), places=9)

    def test_edge_on_bin_edge_leaves_next_bin_empty(self) -> None:
        d = _one_cell(10.0625, 0.125)  # cell [10.0, 10.125] = exactly one bin
        self.assertAlmostEqual(d[0, 3, 80], 1.0)
        self.assertTrue(np.isnan(d[0, 1, 81]) and np.isnan(d[0, 1, 79]))


class TestRunningParams(unittest.TestCase):
    """getRunningParams: values never keep the newline (same fix as C25)."""

    def test_values_at_end_of_line_and_extra_spaces(self) -> None:
        import contextlib
        import io

        from toscana.dataio.running_params import getRunningParams

        text = (
            "# no inline comments, several spaces and a tab\n"
            "<ins> D4\n"
            "<Cmater>   V\n"
            "<Cshape> cylinder\t5.0  5.2 60.0\n"
            "<Sdescr> Silica glass at room temperature\n"
            "<Sato01>   Si   1\n"
            "<Sato02> O 2   # with a comment\n"
            "<Stitle> SiO2_RT\n"
            "<SaFile> sample.qdat\n"
            "<wle> 0.4986\n"
        )
        buf = io.StringIO()
        with in_tempdir() as tmp, contextlib.redirect_stdout(buf):
            (tmp / "s.par").write_text(text)
            p = getRunningParams("s.par")
        self.assertEqual(buf.getvalue(), "")  # no false "not available" warning
        self.assertEqual(p["<ins>"][1], "D4")
        self.assertEqual(p["<Cmater>"][1], "V")
        self.assertEqual(p["<Cshape>"][1:], ("cylinder", "5.0", "5.2", "60.0"))
        self.assertEqual(p["<Sdescr>"][1], "Silica glass at room temperature")
        self.assertEqual(p["<Sato01>"][1:], ("Si", "1"))
        self.assertEqual(p["<Sato02>"][1:], ("O", "2"))
        self.assertEqual(p["<SNelem>"][1], 2)
        self.assertEqual(p["<Stitle>"][1], "SiO2_RT")
        self.assertEqual(p["<SaFile>"][1], "sample.qdat")
        self.assertEqual(p["<wle>"][1], 0.4986)

    def test_unknown_material_still_warns(self) -> None:
        import contextlib
        import io

        from toscana.dataio.running_params import getRunningParams

        buf = io.StringIO()
        with in_tempdir() as tmp, contextlib.redirect_stdout(buf):
            (tmp / "s.par").write_text("<Cmater> Unobtainium\n")
            getRunningParams("s.par")
        self.assertIn("Unobtainium is not available", buf.getvalue())

    def test_bad_first_character_raises(self) -> None:
        from toscana.dataio.running_params import getRunningParams

        with in_tempdir() as tmp:
            (tmp / "s.par").write_text("ins D4\n")
            with self.assertRaises(ValueError):
                quiet(getRunningParams, "s.par")


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

        from toscana.dataio.parameters import readParam

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
