"""Physics helpers: conversions, crystallography, densities, models."""

from __future__ import annotations

import unittest

import helpers  # noqa: F401

import numpy as np

from toscana.models.cross_sections import XS_model
from toscana.models.scattering import inelastic
from toscana.physics.conversions import ang2q, q2ang
from toscana.physics.crystallography import reflections_fcc
from toscana.physics.geometry import getCylVolume
from toscana.physics.properties import getAbsXS, getAtomicDensity, getFreeXS


class TestConversions(unittest.TestCase):
    def test_round_trip(self) -> None:
        angles = np.linspace(1, 140, 50)
        np.testing.assert_allclose(q2ang(ang2q(angles, 0.5), 0.5), angles, rtol=1e-12)

    def test_known_value(self) -> None:
        # 2θ = 60°, λ = 0.5 Å  ->  Q = 4π/0.5 * sin(30°) = 4π
        self.assertAlmostEqual(float(ang2q(60.0, 0.5)), 4 * np.pi, places=12)

    def test_bad_wavelength(self) -> None:
        with self.assertRaises(ValueError):
            ang2q(10.0, 0.0)


class TestCrystallography(unittest.TestCase):
    def test_ni_reflections_match_bragg(self) -> None:
        a, lam = 3.52024, 0.5
        hkl = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2),
               (4, 0, 0), (3, 3, 1), (4, 2, 0), (4, 2, 2), (3, 3, 3)]
        expected = [2 * np.degrees(np.arcsin(lam * np.sqrt(h * h + k * k + l * l) / (2 * a)))
                    for h, k, l in hkl]
        np.testing.assert_allclose(reflections_fcc(lam, 0.0, a)[:10], expected, atol=1e-9)

    def test_default_lattice_is_ni_3_52(self) -> None:
        from toscana.physics.crystallography import NI_LATTICE_PARAMETER

        self.assertEqual(NI_LATTICE_PARAMETER, 3.52)
        np.testing.assert_array_equal(reflections_fcc(0.5, 0.0), reflections_fcc(0.5, 0.0, 3.52))

    def test_zero_angle_shift(self) -> None:
        np.testing.assert_allclose(
            reflections_fcc(0.5, 0.2)[:10], reflections_fcc(0.5, 0.0)[:10] + 0.2
        )


class TestProperties(unittest.TestCase):
    def test_water_atomic_density(self) -> None:
        # 1 g/cm3, 6 g/mol/atom (H2O: 18/3)  ->  ~0.1004 atoms/Å3
        self.assertAlmostEqual(getAtomicDensity(1.0, 6.0), 0.602214076 / 6.0, places=12)

    def test_cylinder_volume(self) -> None:
        self.assertAlmostEqual(getCylVolume(2.0, 10.0), np.pi * 10.0, places=12)

    def test_free_cross_section(self) -> None:
        self.assertAlmostEqual(getFreeXS(5.08, 50.0), 5.08 * (50 / 51) ** 2, places=12)

    def test_absorption_reference_wavelength(self) -> None:
        """σ_abs is tabulated at 2200 m/s, λ = 1.798 Å (C30)."""
        self.assertEqual(getAbsXS(5.08, 1.798), 5.08)
        self.assertAlmostEqual(getAbsXS(5.08, 0.5), 5.08 * 0.5 / 1.798, places=12)

    def test_thermal_references_consistent(self) -> None:
        from toscana.models.cross_sections import scattering_probability
        from toscana.physics.properties import THERMAL_ENERGY_MEV, THERMAL_WAVELENGTH

        self.assertEqual(THERMAL_WAVELENGTH, 1.798)
        # E[meV] = 81.8042 / λ[Å]^2, and v = 2200 m/s
        self.assertAlmostEqual(THERMAL_ENERGY_MEV, 25.30, places=2)
        v = 3956.034 / THERMAL_WAVELENGTH  # h/m_n in m·Å/s
        self.assertAlmostEqual(v, 2200.0, delta=1.0)
        # energy route and wavelength route give the same absorption
        lam = 0.5
        e0 = 81.8042 / lam**2
        p = scattering_probability(e0, 5.0, [1.0], [5.08])
        sabs = getAbsXS(5.08, lam)
        self.assertAlmostEqual(p, 1 - sabs / (sabs + 5.0), places=12)


class TestVanadiumStandard(unittest.TestCase):
    def test_default_density_and_consistency(self) -> None:
        import inspect

        from helpers import quiet
        from toscana.experiment import info

        vana = quiet(info.setVanaInfo)
        self.assertEqual(vana["den_gcc"], 6.11)
        self.assertAlmostEqual(vana["den_aac"], 6.11 * 0.602214076 / 50.9415, places=12)
        vanadens = inspect.signature(info.setSampleInfo).parameters["vanadens"].default
        self.assertAlmostEqual(vanadens, vana["den_aac"], places=15)


class TestModels(unittest.TestCase):
    def test_inelastic_limits(self) -> None:
        a = 51.0
        self.assertAlmostEqual(float(inelastic(-1e3, A=a, lowQ=0.4)), 0.4, places=10)
        self.assertAlmostEqual(
            float(inelastic(1e3, A=a, lowQ=0.4)), 0.4 * (a / (a + 1)) ** 2, places=10
        )

    def test_xs_model_high_energy_limit(self) -> None:
        # E >> E_eff: σ -> Σ c σ_bound (A/(A+1))^2
        val = XS_model(1e12, 1.0, [1.0], [5.0], [50.0])
        self.assertAlmostEqual(val, 5.0 * (50 / 51) ** 2, places=9)


if __name__ == "__main__":
    unittest.main()
