"""
Physical constants, unit conversions and simple sample physics.

``toscana.physics`` is the lowest-level physics layer of ToScaNA. It does not
import any higher-level ToScaNA package and is used by ``dataio`` (regrouping,
parameter files), ``models`` (Ni model, cross sections), ``calibration`` (Ni
calibration of λ and 2θ0) and ``experiment`` (sample, container and vanadium
bookkeeping).

Submodules and public names re-exported here
--------------------------------------------
conversions
    ``ang2q``  -- scattering angle 2θ (degrees) -> momentum transfer Q (1/Å).
    ``q2ang``  -- momentum transfer Q (1/Å) -> scattering angle 2θ (degrees).
crystallography
    ``sf_fcc``, ``sf_bcc``, ``sf_sc`` -- geometric structure factors of the
    cubic Bravais lattices for a plane (hkl).
    ``reflections_fcc`` -- 2θ positions (degrees) of the allowed fcc reflections
    for a wavelength, zero-angle offset and lattice parameter.
    ``NI_LATTICE_PARAMETER`` (3.52 Å, not re-exported here; import it from
    ``toscana.physics.crystallography``) -- default lattice parameter of the
    Ni calibration standard.
geometry
    ``getCylVolume`` -- volume of a solid cylinder, π (d/2)² h.
properties
    ``getAtomicDensity`` -- g/cm³ -> atoms/Å³.
    ``getDensity``       -- atoms/Å³ -> g/cm³.
    ``getMassNumber``    -- mass ratio A = M / m_n.
    ``getFreeXS``        -- free-atom cross section σ_bound [A/(A+1)]².
    ``getAbsXS``         -- absorption cross section at λ (1/v law).
    ``getNofAtoms``      -- number of atoms in a formula unit.
    ``getConcentrations`` -- atomic fractions c_i = n_i / Σ n.
    ``AtomicAvg``        -- concentration-weighted average Σ c_i m_i.
    ``THERMAL_WAVELENGTH`` (1.798 Å) and ``THERMAL_ENERGY_MEV`` (25.30 meV)
    (not re-exported here; import them from ``toscana.physics.properties``)
    -- reference of tabulated (2200 m/s) absorption cross sections.

Conventions and units
---------------------
Angles 2θ in degrees; wavelengths and lattice parameters in Å; Q in 1/Å;
macroscopic densities in g/cm³; atomic densities in atoms/Å³; molar masses in
g/mol (numerically equal to atomic mass units); cross sections in barn
(1 b = 100 fm² = 1e-24 cm²); neutron energies in meV.
"""

from .conversions import ang2q, q2ang
from .crystallography import reflections_fcc, sf_bcc, sf_fcc, sf_sc
from .geometry import getCylVolume
from .properties import (
    AtomicAvg,
    getAbsXS,
    getAtomicDensity,
    getConcentrations,
    getDensity,
    getFreeXS,
    getMassNumber,
    getNofAtoms,
)

__all__ = [
    "AtomicAvg",
    "ang2q",
    "getAbsXS",
    "getAtomicDensity",
    "getConcentrations",
    "getCylVolume",
    "getDensity",
    "getFreeXS",
    "getMassNumber",
    "getNofAtoms",
    "q2ang",
    "reflections_fcc",
    "sf_bcc",
    "sf_fcc",
    "sf_sc",
]
