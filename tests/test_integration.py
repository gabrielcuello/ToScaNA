"""Integration of the NeXus reader, d4numor, pycorrect, MCAMS and the bridge."""

from __future__ import annotations

import glob
import os
import pathlib
import shutil
import subprocess
import sys
import unittest

import helpers
from helpers import in_tempdir, quiet

import numpy as np

ROOT = helpers._pkg.ROOT
RAWDATA = os.environ.get("TOSCANA_RAWDATA", "/Users/cuello/AI-lab/rawdataD4")


def _env() -> dict[str, str]:
    """Environment in which `python -m toscana...` works (package via a link)."""
    return dict(os.environ, PYTHONPATH=str(_link_dir()), PYTHONDONTWRITEBYTECODE="1",
                MPLBACKEND="Agg")


_LINK: list[pathlib.Path] = []


def _link_dir() -> pathlib.Path:
    if not _LINK:
        import tempfile

        d = pathlib.Path(tempfile.mkdtemp())
        (d / "toscana").symlink_to(ROOT)
        _LINK.append(d)
    return _LINK[0]


def _write_nexus(path: pathlib.Path, old_format: bool) -> np.ndarray:
    """Minimal D4 NeXus file in the old or the current layout; returns counts."""
    import h5py

    counts = np.arange(576, dtype=np.int32) % 97 + 10
    with h5py.File(path, "w") as h:
        e = h.create_group("entry0")
        for name, value in (("user/name", b"cuello"), ("user/namelocalcontact", b"CUELLO"),
                            ("user/proposal", b"5-26-228"), ("start_time", b"13-Jun-25 08:44:13"),
                            ("end_time", b"13-Jun-25 08:49:09"), ("title", b"Test"),
                            ("experiment_identifier", b"Vanadium rod")):
            e.create_dataset(name, data=np.array([value]))
        e.create_dataset("run_number", data=np.array([454615], dtype=np.int32))
        e.create_dataset("monitor/monsum", data=np.array([2185001.0], dtype=np.float32))
        e.create_dataset("time", data=np.array([296.3576], dtype=np.float32))
        e.create_dataset("data_scan/total_steps", data=np.array([1], dtype=np.int32))
        e.create_dataset("instrument/omega/value", data=np.array([98.19], dtype=np.float32))
        motor = "instrument/2theta" if old_format else "instrument/masked2theta_s"
        e.create_dataset(motor + "/value", data=np.array([3.258], dtype=np.float32))
        e.create_dataset(motor + "/target_value", data=np.array([3.25], dtype=np.float32))
        total = "TotalAcquisitionSpy" if old_format else "TotalCount"
        names = [total, "Time", "Monitor", "Position"]
        values = [[counts.sum()], [296.3576], [2185001.0], [3.258]]
        e.create_dataset("data_scan/scanned_variables/variables_names/property",
                         data=np.array([n.encode() for n in names]))
        e.create_dataset("data_scan/scanned_variables/data", data=np.array(values, dtype=np.float64))
        shape = (1, 576, 1) if old_format else (1, 1, 576)
        e.create_dataset("data_scan/detector_data/data", data=counts.reshape(shape))
    return counts.reshape(9, 64).astype(float)


class TestLoadNexus(unittest.TestCase):
    def test_both_formats(self) -> None:
        from toscana.dataio.acquisition import load_nxs

        for old_format in (True, False):
            with self.subTest(old_format=old_format), in_tempdir() as tmp:
                path = tmp / "454615.nxs"
                counts = _write_nexus(path, old_format)
                meta, c = load_nxs(str(path))
            np.testing.assert_array_equal(c, counts)
            self.assertAlmostEqual(float(meta["Theta_lue(deg)"]), 3.258, places=5)
            self.assertAlmostEqual(float(meta["Theta_des(deg)"]), 3.25, places=5)
            self.assertEqual(float(meta["MonitorCnts"]), 2185001.0)
            self.assertEqual(float(meta["TotalCnts"]), counts.sum())
            self.assertEqual(meta["numor"], "454615")
            self.assertEqual(meta["subtitle"], "Vanadium rod")

    @unittest.skipUnless(os.path.isdir(os.path.join(RAWDATA, "rawdata_MW")),
                         "real D4 data not available (set TOSCANA_RAWDATA)")
    def test_real_pairs_ascii_vs_nexus(self) -> None:
        from toscana.dataio import acquisition

        files = sorted(glob.glob(os.path.join(RAWDATA, "rawdata_*", "*.nxs")))
        pairs = [f for f in files if os.path.isfile(f[:-4])][::200]
        self.assertGreater(len(pairs), 5)
        with helpers.in_tempdir():
            for f in pairs:
                ma, ca = quiet(acquisition.readD4numorASCII, f[:-4])
                mn, cn = acquisition.load_nxs(f)
                np.testing.assert_array_equal(ca, cn, err_msg=f)
                for key, tol in (("Theta_lue(deg)", 5e-4), ("Theta_des(deg)", 5e-4),
                                 ("MonitorCnts", 0), ("TotalCnts", 0), ("CntTime (sec)", 1e-4)):
                    self.assertLessEqual(abs(float(ma[key]) - float(mn[key])), tol, (f, key))


class TestD4numor(unittest.TestCase):
    def test_uses_bundled_reader(self) -> None:
        from toscana import d4numor
        from toscana.dataio import nexus

        self.assertIs(d4numor.get_reader(), nexus)
        self.assertEqual(d4numor.parse_numor_list("454600-454602,454610"),
                         [454600, 454601, 454602, 454610])

    def test_command_line_starts(self) -> None:
        out = subprocess.run([sys.executable, "-m", "toscana.d4numor", "--help"],
                             env=_env(), capture_output=True, text=True, cwd=_link_dir())
        self.assertEqual(out.returncode, 0, out.stderr[-1000:])
        self.assertIn("usage", out.stdout.lower())


class TestPycorrectIntegration(unittest.TestCase):
    def test_d4_example_reproduces_recorded_output(self) -> None:
        src = ROOT / "docs" / "pycorrect" / "examples" / "d4_data"
        with in_tempdir() as tmp:
            for f in src.iterdir():
                shutil.copy(f, tmp)
            recorded = (tmp / "corrected.dat").read_bytes()
            (tmp / "corrected.dat").unlink()
            out = subprocess.run([sys.executable, "-m", "toscana.corrections.pycorrect",
                                  "d4_example.com"], env=_env(), cwd=tmp,
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-1000:])
            self.assertEqual((tmp / "corrected.dat").read_bytes(), recorded)


class TestMcamsNuclearData(unittest.TestCase):
    def test_toscana_table_backend(self) -> None:
        from toscana.corrections.mcams.formula import parse_species
        from toscana.corrections.mcams.nuclear_data import TableError, ToscanaData
        from toscana.isotopes import Element

        data = ToscanaData()
        fe = Element("Fe")
        self.assertEqual(data.neutron_xs(parse_species("Fe")), (fe.sig_sca, fe.sig_abs))
        self.assertEqual(data.mass(parse_species("Fe")), fe.weight)
        d = Element("H2")
        self.assertEqual(data.neutron_xs(parse_species("D")), (d.sig_sca, d.sig_abs))
        self.assertEqual(data.mass(parse_species("[56Fe]")), Element("Fe56").weight)
        with self.assertRaises(TableError):
            data.neutron_xs(parse_species("Cm"))  # no cross sections in the table
        with self.assertRaises(TableError):
            data.mass(parse_species("[999Fe]"))

    def test_layer_builder_default_is_toscana_table(self) -> None:
        from toscana.corrections.mcams.layer_builder import compute_layer_properties
        from toscana.isotopes import Element

        fe = Element("Fe")
        layer = compute_layer_properties("Fe plate", 1.0, "Fe", 7.87)
        n_per_cm3 = 7.87 * 6.02214076e23 / fe.weight
        self.assertAlmostEqual(layer.mu_scat, n_per_cm3 * fe.sig_sca * 1e-24 / 10, places=6)


class TestBridge(unittest.TestCase):
    def test_units_and_imports(self) -> None:
        from toscana.corrections import bridge

        self.assertAlmostEqual(bridge.cm_to_mm(0.3), 3.0)
        self.assertAlmostEqual(bridge.per_cm_to_per_mm(1.0), 0.1)


if __name__ == "__main__":
    unittest.main()
