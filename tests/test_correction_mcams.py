"""
Correction workflow, method="mcams" (default): ToScaNA's Paalman & Pings
chain with MCAMS factors.

* Set-ups that pycorrect cannot describe (hollow sample, container with a
  gap, furnace + extra shield, absorber): synthetic data are built forward
  with the MCAMS factors of the set-up, and the chain must recover the true
  dσ/dΩ to rounding error, also from ``.qdat`` files.
* With CORRECT's own factors the chain reproduces CORRECT's dσ/dΩ exactly.
* Set-up description: ``<LayNN>`` lines, transmissions, factor files on both
  scales, pycorrect's limits.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import unittest

import helpers
from helpers import in_tempdir

import numpy as np

import synthetic_d4 as syn

_ = helpers

FAST = dict(n_points=20000, ms_particles=200000)


def _load(par="sample.par"):
    from toscana.dataio.running_params import getRunningParams
    from toscana.experiment.measurement import Measurement

    with contextlib.redirect_stdout(io.StringIO()):
        return Measurement(getRunningParams(par))


def _edit_par(folder, replacements, extra=""):
    path = folder / "sample.par"
    lines = []
    for line in path.read_text().splitlines():
        tag = line.split()[0] if line.split() else ""
        lines.append(replacements.get(tag, line))
    path.write_text("\n".join(lines) + "\n" + extra)


def _write(path, theta, y, e, scale="angle", lam=0.5):
    x = theta if scale == "angle" else 4 * math.pi / lam * np.sin(np.radians(theta) / 2)
    caption = "2Theta (degrees)" if scale == "angle" else "Q (1/A)"
    with open(path, "w") as fh:
        fh.write(f"# {path.name}\n#X caption : {caption}\n#Y caption : Counts/monitor\n")
        for a, b, c in zip(x, y, e):
            fh.write(f"{a:.12f} {b:.12e} {c:.12e}\n")


def make_complex_experiment(folder):
    """Hollow Ni sample in a V can with a gap, Nb furnace + Nb shield, absorber.

    Returns (Measurement, truth dict)."""
    from toscana.corrections.factors import compute_factors
    from toscana.corrections.setup import setups_from_measurement

    syn.make_experiment(folder)
    _edit_par(folder, {
        "<Sshape>": "<Sshape> cylinder 2.0 6.0 50.0 # hollow sample",
        "<Cshape>": "<Cshape> cylinder 6.6 7.0 60.0 # can with a gap",
        "<Amater>": "<Amater> Cd # absorber",
        "<Emater>": "<Emater> Nb # furnace",
        "<Eshape>": "<Eshape> cylinder 20.0 20.1 100.0 # furnace element",
    }, extra="<Lay01> shield environment Nb 30.0 30.1 100.0 # heat shield\n")
    m = _load(str(folder / "sample.par"))
    ss, vs = setups_from_measurement(m, environment="layer")
    theta = syn.THETA
    lam = float(m.wavelength)
    fs = compute_factors(ss, theta, **FAST)
    fv = compute_factors(vs, theta, **FAST)

    q = 4 * math.pi / lam * np.sin(np.radians(theta) / 2)
    from toscana.isotopes import Element

    ni = Element("Ni")
    dcs = ni.sig_sca / (4 * math.pi) + abs(ni.bcoh) ** 2 / 100 * (syn.s_true(q) - 1)
    k = 1e5 * (1 - 0.15 * (theta / 135) ** 2)            # flux x efficiency
    single = {"s": k * fs.atoms_per_area["s"] * dcs}
    for b in ("c", "h"):
        single[b] = k * ss.scattering_power(b) / (4 * math.pi)
    empty = 5 + 200 * np.exp(-theta / 1.5)
    absorber = 3 + 50 * np.exp(-theta / 1.5)
    data = {"empty": empty, "absorber": absorber}
    names = {"h": "furnace", "ch": "can", "sch": "sample"}
    for cfg in fs.configurations:
        b = fs.T[f"T_{cfg}"] * empty + (1 - fs.T[f"T_{cfg}"]) * absorber
        iso = sum(fs.A[f"A_{x},{cfg}"] * ss.scattering_power(x) for x in cfg) / (4 * math.pi)
        data[names[cfg]] = (b + sum(fs.A[f"A_{x},{cfg}"] * single[x] for x in cfg)
                            + fs.delta[f"delta_{cfg}"] * k * iso)
    bv = fv.T["T_v"] * empty + (1 - fv.T["T_v"]) * absorber
    data["vanadium"] = bv + (1 + fv.delta["delta_v"]) * fv.A["A_v,v"] * k * \
        vs.scattering_power("v") / (4 * math.pi)
    for name, y in data.items():
        scale = "q" if name == "sample" else "angle"
        ext = "qdat" if scale == "q" else "adat"
        _write(folder / f"{name}.{ext}", theta, y, np.sqrt(y) * 1e-3, scale, lam)
    files = {"sample": str(folder / "sample.qdat"), "container": str(folder / "can.adat"),
             "environment": str(folder / "furnace.adat"), "vanadium": str(folder / "vanadium.adat"),
             "empty": str(folder / "empty.adat"), "absorber": str(folder / "absorber.adat")}
    return m, {"dcs": dcs, "single": single, "fs": fs, "fv": fv, "files": files, "k": k}


class TestComplexSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from toscana.workflows.correction import run_correction

        cls._tmp = in_tempdir()
        cls.folder = cls._tmp.__enter__()
        cls.m, cls.truth = make_complex_experiment(cls.folder)
        cls.result = run_correction(cls.m, environment="layer", placzek="none", absorber_model="transmission",
                                    files=cls.truth["files"], factor_options=FAST)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.__exit__(None, None, None)

    def test_setup(self) -> None:
        fs = self.result.factors
        self.assertEqual(fs.configurations, ["h", "ch", "sch"])
        self.assertEqual(self.result.vanadium_factors.configurations, ["v"])
        self.assertEqual(self.result.method, "mcams")
        for key in ("A_h,h", "A_c,ch", "A_h,ch", "A_s,sch", "A_c,sch", "A_h,sch"):
            self.assertIn(key, fs.A)
            self.assertTrue(np.all((fs.A[key] > 0) & (fs.A[key] <= 1)))
        self.assertTrue(0 < fs.T["T_sch"] < fs.T["T_ch"] < fs.T["T_h"] < 1)

    def test_recovers_true_cross_section(self) -> None:
        # computed (same seed) factors: the chain inverts the forward model exactly
        np.testing.assert_allclose(self.result.dcs, self.truth["dcs"], rtol=1e-7)

    def test_intermediate_intensities(self) -> None:
        I = self.result.intensities
        for body in ("s", "c", "h"):
            np.testing.assert_allclose(I[f"I_{body}"], self.truth["single"][body], rtol=1e-7)
        self.assertIn("B_sch", self.result.backgrounds)
        self.assertIn("MS_sch", I)
        self.assertTrue(np.all(I["I_s_err"] > 0))

    def test_precomputed_factors(self) -> None:
        from toscana.workflows.correction import run_correction

        r = run_correction(self.m, environment="layer", placzek="none", absorber_model="transmission", files=self.truth["files"],
                           factors=self.truth["fs"], vanadium_factors=self.truth["fv"])
        np.testing.assert_allclose(r.dcs, self.truth["dcs"], rtol=1e-7)

    def test_absorber_matters(self) -> None:
        from toscana.workflows.correction import run_correction

        files = dict(self.truth["files"], absorber="")
        m = copy.copy(self.m)
        m.absorber = ["None"] + list(m.absorber[1:])
        r = run_correction(m, environment="layer", placzek="none", absorber_model="transmission", files=files,
                           factors=self.truth["fs"], vanadium_factors=self.truth["fv"])
        # ignoring the absorber biases the result far beyond rounding (1e-7)
        self.assertGreater(float(np.abs(r.dcs / self.truth["dcs"] - 1).max()), 1e-4)

    def test_save_factors_and_intensities(self) -> None:
        from toscana.corrections.factors import load_factors

        with in_tempdir() as tmp:
            names = self.result.save_factors(str(tmp / "run"), scale="q")
            x, cols, scalars = load_factors(names[0])
            header = open(names[0]).read()
            xa, cols_a, _ = load_factors(self.result.save_factors(str(tmp / "a"), "angle")[0])
            self.result.save_intensities(str(tmp / "I.dat"), scale="q")
            table = np.loadtxt(tmp / "I.dat")
        self.assertEqual(len(names), 2)
        np.testing.assert_allclose(np.sort(x), np.sort(self.result.q), rtol=1e-6)
        np.testing.assert_allclose(np.sort(xa), np.sort(self.result.theta), rtol=1e-6)
        self.assertIn("A_s,sch", cols)
        self.assertIn("delta_sch", cols)
        self.assertIn("# P_s = ", header)
        self.assertIn("# T_sch = ", header)
        self.assertEqual(table.shape[0], len(self.result.q))

    def test_pycorrect_refuses(self) -> None:
        from toscana.corrections.factors import pycorrect_supports
        from toscana.corrections.setup import setups_from_measurement
        from toscana.workflows.correction import run_correction

        ok, reason = pycorrect_supports(setups_from_measurement(self.m, environment="layer")[0])
        self.assertFalse(ok)
        self.assertTrue(reason)
        with self.assertRaises(ValueError):
            run_correction(self.m, method="pycorrect", files=self.truth["files"])


class TestAgainstCorrect(unittest.TestCase):
    def test_same_factors_same_result(self) -> None:
        """With CORRECT's factors (δ converted to the MCAMS definition, total
        multiple / total single) the chain gives CORRECT's dσ/dΩ."""
        from toscana.corrections.setup import setups_from_measurement
        from toscana.workflows.correction import run_correction

        with in_tempdir() as folder:
            syn.make_experiment(folder)
            m = _load()
            p = run_correction(m, method="pycorrect", placzek="none")
            r = run_correction(m, placzek="none", factor_options=FAST)
            fs, fv = copy.deepcopy(r.factors), copy.deepcopy(r.vanadium_factors)
            ss, _ = setups_from_measurement(m)
            fs.A.update(p.factors.A)
            fv.A.update(p.vanadium_factors.A)
            a_s = fs.A["A_s,sc"] * ss.scattering_power("s")
            a_c = fs.A["A_c,sc"] * ss.scattering_power("c")
            fs.delta["delta_sc"] = p.factors.delta["delta_sc"] * a_s / (a_s + a_c)
            fs.delta["delta_c"] = 0 * fs.delta["delta_c"]
            fv.delta["delta_v"] = p.vanadium_factors.delta["delta_v"]
            h = run_correction(m, placzek="none", factors=fs, vanadium_factors=fv)
        np.testing.assert_allclose(h.dcs, p.dcs, rtol=1e-10)
        # MCAMS factors themselves: same attenuation within MC noise
        np.testing.assert_allclose(r.factors.A["A_s,sc"], p.factors.A["A_s,sc"], rtol=0.01)


class TestReading(unittest.TestCase):
    def test_qdat_and_adat(self) -> None:
        from toscana.workflows.correction import read_regrouped

        theta = np.linspace(2, 130, 50)
        y = 1 + theta / 100
        with in_tempdir() as tmp:
            _write(tmp / "x.adat", theta, y, 0.1 * y)
            _write(tmp / "x.qdat", theta, y, 0.1 * y, "q", 0.5)
            np.savetxt(tmp / "plain.qdat", np.column_stack([4 * math.pi / 0.5 * np.sin(
                np.radians(theta) / 2), y]))
            a = read_regrouped(str(tmp / "x.adat"), 0.5)
            q = read_regrouped(str(tmp / "x.qdat"), 0.5)
            p = read_regrouped(str(tmp / "plain.qdat"), 0.5)
            with self.assertRaises(ValueError):
                np.savetxt(tmp / "plain.txt", np.column_stack([theta, y]))
                read_regrouped(str(tmp / "plain.txt"), 0.5)
        self.assertEqual((a[3], q[3], p[3]), ("angle", "q", "q"))
        np.testing.assert_allclose(q[0], theta, rtol=1e-8)
        np.testing.assert_allclose(p[0], theta, rtol=1e-6)
        np.testing.assert_allclose(q[2], 0.1 * y)
        np.testing.assert_array_equal(p[2], 0)


class TestSetup(unittest.TestCase):
    def test_layer_tags(self) -> None:
        with in_tempdir() as folder:
            syn.make_experiment(folder)
            _edit_par(folder, {}, extra="<Lay02> wall2 container V 7.0 7.2 60.0 # 2nd wall\n"
                                        "<Lay01> shield environment Nb 30.0 30.1 100.0 # shield\n")
            m = _load()
        self.assertEqual([x["name"] for x in m.extra_layers], ["shield", "wall2"])
        self.assertEqual(m.extra_layers[1]["outer_diam"], 7.2)

    def test_layer_tag_too_short(self) -> None:
        with in_tempdir() as folder:
            syn.make_experiment(folder)
            _edit_par(folder, {}, extra="<Lay01> shield environment Nb 30.0 # incomplete\n")
            with self.assertRaises(ValueError):
                _load()

    def test_transmission_thin_limit(self) -> None:
        from toscana.corrections.setup import CylinderSetup, make_layer

        # very weak absorber: 1 - T = mu <L>, <L> = π R² / w for a beam wider than the rod
        layer = make_layer("rod", "s", ({"V": 1}, 6.11e-4), 0.0, 6.0, 100.0)
        setup = CylinderSetup([layer], 10.0, 20.0, 0.5)
        mu = layer.mu(0.5)
        self.assertAlmostEqual(setup.mean_path("s"), math.pi * 9 / 10, places=4)
        self.assertAlmostEqual((1 - setup.transmission("s")) / (mu * setup.mean_path("s")), 1,
                               places=3)
        # annulus: mean path = π (R² - r²) / w
        ring = make_layer("ring", "c", "V", 4.0, 6.0, 100.0)
        self.assertAlmostEqual(CylinderSetup([ring], 10.0, 20.0, 0.5).mean_path("c"),
                               math.pi * (9 - 4) / 10, places=4)


if __name__ == "__main__":
    unittest.main()
