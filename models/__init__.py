"""
Model functions for D4 data analysis.

These functions are the physical or empirical models fitted to (or evaluated
against) the data during reduction and calibration. They are plain NumPy
functions whose signatures are suitable for ``lmfit.Model`` and
``scipy.optimize.curve_fit`` (abscissa first, then the parameters).

Submodules and public names
---------------------------
scattering
    ``inelastic``  -- inelasticity (Placzek-like) decrease of the self
    scattering from the bound value at low Q to the free value at high Q.
    ``self024``    -- empirical self-scattering curve s0 + q2 Q² + q4 Q⁴.
    ``vanaQdep``   -- Q dependence of the vanadium signal: polyQ2 × inelastic.
cross_sections
    ``XS_model``   -- epithermal total scattering cross section
    Σ c_i σ_i [A_i/(A_i+1)]² [1 + E_eff/(2 A_i E)].
    ``scattering_probability`` -- σ_s / (σ_a + σ_s) with 1/v absorption.
ni
    ``niPeaks10``  -- Ni powder model (quadratic background + 10 Gaussian
    peaks at the fcc reflections) used for the λ and 2θ0 calibration.

Conventions and units
---------------------
Q in 1/Å; 2θ in degrees; λ and lattice parameters in Å; cross sections in
barn; energies in meV (or any common unit where only ratios enter); mass
ratios A = m/m_n dimensionless.
"""

from .cross_sections import XS_model, scattering_probability
from .ni import niPeaks10
from .scattering import inelastic, self024, vanaQdep

__all__ = [
    "inelastic",
    "niPeaks10",
    "scattering_probability",
    "self024",
    "vanaQdep",
    "XS_model",
]
