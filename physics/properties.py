"""
Sample properties: densities, mass numbers, cross sections, concentrations.

These helpers are used when building the sample, container and vanadium
descriptions (``toscana.experiment``: ``setSampleInfo``, ``setVanaInfo``,
``Measurement``) and the cross-section models (``toscana.models``).

Public constants
----------------
THERMAL_WAVELENGTH : float
    1.798 Å, wavelength of the 2200 m/s thermal neutrons to which tabulated
    absorption cross sections refer: λ = h / (m_n v).
THERMAL_ENERGY_MEV : float
    81.8042 / THERMAL_WAVELENGTH² = 25.30 meV (25.3044...), the same reference
    as an energy, from E[meV] = 81.8042 / λ[Å]².

Public functions
----------------
getAtomicDensity(density=1.0, molarMass=6.0)
    Atomic density (atoms/Å³) from density (g/cm³) and mean molar mass per atom.
getDensity(atomic_density=0.1, molarM=6.0)
    Inverse of `getAtomicDensity`: density (g/cm³).
getMassNumber(molarMass=6.0)
    Mass ratio A = M / m_n with m_n = 1.0086649 u.
getFreeXS(BoundXS=5.08, A=50.9415)
    Free-atom cross section σ_bound [A/(A+1)]² of one nucleus (barn).
getAbsXS(AbsXS=5.08, wavelength=THERMAL_WAVELENGTH)
    Absorption cross section at λ by the 1/v law: σ_abs λ / 1.798 (barn).
getNofAtoms(atoms)
    Number of atoms in a formula unit given as {"Si": 1, "O": 2}.
getConcentrations(atoms)
    Atomic fractions c_i = n_i / Σ n.
AtomicAvg(concentration, magnitude)
    Concentration-weighted average Σ c_i m_i of a per-element property.

Conventions and units
---------------------
Densities in g/cm³; atomic densities in atoms/Å³; molar masses in g/mol
(numerically equal to atomic mass units); cross sections in barn; wavelengths
in Å; energies in meV. Formula units are dictionaries {element symbol: number
of atoms}; per-element properties are dictionaries with the same keys.

Notes
-----
Changes from the legacy code (audit 2026-11, C30): three different thermal
references were in use: 1.8089 Å in `getAbsXS` (σ_abs 0.6 % too small),
1.8 Å in ``setSampleInfo`` and 25.3 meV in ``scattering_probability``. All
now derive from ``THERMAL_WAVELENGTH`` = 1.798 Å and ``THERMAL_ENERGY_MEV``,
so the wavelength route (`getAbsXS`) and the energy route
(``toscana.models.cross_sections.scattering_probability``) give the same
absorption.

Changes from the legacy code (audit 2026-11, P9): free cross sections of
mixtures are now computed element by element, σ_free = Σ c_i σ_i [A_i/(A_i+1)]²
(by ``Measurement.free`` and ``setSampleInfo``, using `getMassNumber` and
`getFreeXS`), not once with the mean mass.
"""

from __future__ import annotations

from typing import Mapping

# Thermal reference for tabulated cross sections: neutrons at 2200 m/s.
# λ = h / (m_n v) = 1.798 Å, E = 81.8042 meV·Å² / λ² = 25.30 meV.
# Audit 2026-11, C30: single source for the thermal reference (legacy used
# 1.8089 Å, 1.8 Å and 25.3 meV in different places).
THERMAL_WAVELENGTH = 1.798  # Å
THERMAL_ENERGY_MEV = 81.8042 / THERMAL_WAVELENGTH**2  # meV


def getAtomicDensity(density: float = 1.0, molarMass: float = 6.0) -> float:
    """
    Calculate the atomic density (atoms/Å³).

    Given the macroscopic density (g/cm³) and the average atomic molar mass
    (g/mole), this function returns the atomic density (atoms/Å³).

    Atomic density [atom/Å³] =
           density [g/cm³] * NA [atom/mole] / molarM [g/mole] * 10⁻²⁴
    where molarM is the average molar mass per atom and NA = 6.02214076 × 10²³
    atoms/mole is the Avogadro's number.

    If density is less or equal to 0, the function returns water atomic density,
    i.e, 0.1 at/Å³.

    Parameters
    ----------
    density : float, optional
        Macroscopic density in g/cm³. The default is 1.0.
    molarMass : float, optional
        Average molar mass per atom, in g/mole/atom (molar mass of the
        formula unit divided by its number of atoms). The default is 6.0
        (water: 18.015/3 ≈ 6).

    Returns
    -------
    float
        Atomic density, in atoms/Å³.

    Raises
    ------
    ZeroDivisionError
        If ``molarMass == 0`` (Python float) and ``density > 0``.

    Notes
    -----
    Water fallback: if ``density <= 0`` both arguments are replaced by the
    water values density = 1.0 g/cm³ and molarMass = 6.0 g/mol (the given
    `molarMass` is ignored), the result is 0.602214076/6 = 0.10037 atoms/Å³,
    and two lines are printed::

        Attention!
            Using the water density in getAtomicDensity function.

    See Also
    --------
    getDensity : inverse conversion.

    Examples
    --------
    >>> round(getAtomicDensity(8.908, 58.6934), 6)  # Ni
    0.091399
    """
    # NA * 10^-24 = 6.02214076 * 10^23 * 10^-24 = 0.602214076
    avogadro_scaled = 0.602214076

    if density <= 0:
        density = 1.0
        molarMass = 6.0
        # Consider logging this instead of printing in a library context
        print("Attention!")
        print("    Using the water density in getAtomicDensity function.")

    return float(density * avogadro_scaled / molarMass)


def getDensity(atomic_density: float = 0.1, molarM: float = 6.0) -> float:
    """
    Calculate the macroscopic density (g/cm³) from atomic density (atoms/Å³).

    density [g/cm³] = atomic_density [atoms/Å³] / 0.602214076 · molarM [g/mol]

    where 0.602214076 = N_A · 10⁻²⁴ (Avogadro's number times the Å³ -> cm³
    factor). Inverse of `getAtomicDensity`.

    Parameters
    ----------
    atomic_density : float, optional
        Atomic density in atoms/Å³. Default 0.1.
    molarM : float, optional
        Average molar mass per atom in g/mol. Default 6.0 (water, per atom).

    Returns
    -------
    float
        Macroscopic density in g/cm³; 1.0 (water) if ``atomic_density <= 0``.

    Notes
    -----
    This function preserves the legacy behavior in `legacy/lib/toscana.py`,
    including printing a warning and returning `1.0` when `atomic_density <= 0`.
    The printed line is "Attention! Using the water density in getDensity
    function.". The fallback ignores `molarM`. No exception is raised for any
    input (there is no division by `molarM`).

    See Also
    --------
    getAtomicDensity : direct conversion.

    Examples
    --------
    >>> round(getDensity(0.0913990838664654, 58.6934), 6)
    8.908
    """
    avogadro_scaled = 0.602214076

    result = float(atomic_density / avogadro_scaled * molarM)
    if atomic_density <= 0:
        result = 1.0
        print("Attention! Using the water density in getDensity function.")

    return result


def getMassNumber(molarMass: float = 6.0) -> float:
    """
    Calculate the mass number A = molarMass / neutronMass (legacy convention).

    A here is the ratio of the nuclear (atomic) mass to the neutron mass,
    m / m_n, as needed by the free-atom factor [A/(A+1)]² and by the
    inelasticity models; it is not the integer nucleon number.

    Parameters
    ----------
    molarMass : float, optional
        Molar mass in amu (equivalently g/mol). Default 6.0.

    Returns
    -------
    float
        Mass ratio A = molarMass / 1.0086649 (dimensionless). No validation
        of `molarMass` and no exception (division by a constant).

    Notes
    -----
    The neutron mass is m_n = 1.0086649 u. Since audit 2026-11 (P9) this is
    the only neutron mass used for free cross sections (the isotope table
    value 1.00866 u was also used before).

    Examples
    --------
    >>> round(getMassNumber(50.9415), 4)  # vanadium
    50.5039
    """
    neutronMass = 1.0086649  # amu (legacy constant)
    return float(molarMass / neutronMass)


def getFreeXS(BoundXS: float = 5.08, A: float = 50.9415) -> float:
    """
    Calculate the free cross section from the bound cross section (barns).

    Legacy formula:
        sigma_free = sigma_bound * A^2 / (1 + A)^2

    A nucleus of mass ratio A = m/m_n bound in a solid scatters with the bound
    cross section σ_bound = 4π b²; a free (recoiling) nucleus scatters with
    the reduced-mass scattering length b·A/(A+1), hence the factor [A/(A+1)]².
    This is the high-energy (or high-Q) limit approached by the self
    scattering (see ``toscana.models.scattering.inelastic``).

    Parameters
    ----------
    BoundXS : float, optional
        Bound cross section in barn. Default 5.08 b (vanadium incoherent).
    A : float, optional
        Mass ratio m/m_n of the nucleus (use `getMassNumber`). Default
        50.9415, which is the molar mass of vanadium in g/mol rather than its
        mass ratio (50.504); the difference in the factor is ~0.03 %.

    Returns
    -------
    float
        Free cross section in barn (same unit as `BoundXS`).

    Raises
    ------
    ZeroDivisionError
        If ``A == -1``.

    Notes
    -----
    For one nucleus only. For a mixture the factor must be applied element by
    element, σ_free = Σ c_i σ_i [A_i/(A_i+1)]² (audit 2026-11, P9); this is
    done by the callers (``Measurement.free``, ``setSampleInfo``).

    Examples
    --------
    >>> round(getFreeXS(5.08, getMassNumber(50.9415)), 4)
    4.8846
    """
    return float(BoundXS * A**2 / (1.0 + A) ** 2)


def getAbsXS(AbsXS: float = 5.08, wavelength: float = THERMAL_WAVELENGTH) -> float:
    """
    Scale absorption cross section (barns) by wavelength (Å), 1/v law:

        sigma_abs(lambda) = sigma_thermal * lambda / 1.798

    Tabulated absorption cross sections refer to 2200 m/s neutrons
    (λ = 1.798 Å). Legacy divided by 1.8089 Å (-0.6% bias).

    Parameters
    ----------
    AbsXS : float, optional
        Tabulated (thermal, 2200 m/s) absorption cross section in barn.
        Default 5.08 b (vanadium).
    wavelength : float, optional
        Neutron wavelength λ in Å. Default ``THERMAL_WAVELENGTH`` (1.798 Å),
        which returns `AbsXS` unchanged.

    Returns
    -------
    float
        Absorption cross section at λ, in barn. No validation of the inputs
        (λ = 0 gives 0, negative λ a negative value).

    Notes
    -----
    Changes from the legacy code (audit 2026-11, C30): the reference
    wavelength was 1.8089 Å, which made σ_abs 0.6 % too small; it is now
    ``THERMAL_WAVELENGTH`` = 1.798 Å, consistent with the energy route used
    by ``toscana.models.cross_sections.scattering_probability``.

    Examples
    --------
    >>> round(getAbsXS(4.49, 0.5), 6)  # Ni at 0.5 Å
    1.24861
    """
    return float(AbsXS * wavelength / THERMAL_WAVELENGTH)


def getNofAtoms(atoms: Mapping[str, float]) -> float:
    """
    Calculate the number of atoms in the basic unit (legacy behavior).

    Returns Σ n_i, the sum of the values of `atoms`.

    Parameters
    ----------
    atoms : Mapping[str, float]
        Formula unit as {element symbol: number of atoms}, e.g.
        ``{"Si": 1, "O": 2}`` for SiO₂. Non-integer values are allowed.

    Returns
    -------
    float
        Total number of atoms in the formula unit (0.0 for an empty mapping).

    Examples
    --------
    >>> getNofAtoms({"Si": 1, "O": 2})
    3.0
    """
    natoms: float = 0.0
    for _key, value in atoms.items():
        natoms += value
    return natoms


def getConcentrations(atoms: Mapping[str, float]) -> dict[str, float]:
    """
    Calculate atomic concentrations for a sample (legacy behavior).

    c_i = n_i / Σ_j n_j for every element of the formula unit.

    Parameters
    ----------
    atoms : Mapping[str, float]
        Formula unit as {element symbol: number of atoms}, e.g.
        ``{"Si": 1, "O": 2}``.

    Returns
    -------
    dict[str, float]
        Atomic fractions with the same keys (and order) as `atoms`; they sum
        to 1.

    Raises
    ------
    ZeroDivisionError
        If the atom numbers sum to exactly 0 and `atoms` is not empty
        (after the error message below is printed).

    Notes
    -----
    Preserves legacy behavior including printing an error when `natoms <= 0`.
    When `natoms == 0`, this function will raise `ZeroDivisionError` as legacy does.
    The printed message is "---> ERROR! The number of atoms must be positive.".
    For a negative sum the message is printed and the (meaningless) ratios
    are still returned; for an empty mapping it is printed and ``{}`` is
    returned.

    Examples
    --------
    >>> getConcentrations({"Si": 1, "O": 2})
    {'Si': 0.3333333333333333, 'O': 0.6666666666666666}
    """
    concentration: dict[str, float] = {}
    natoms = float(sum(atoms.values()))
    if natoms <= 0:
        print("---> ERROR! The number of atoms must be positive.")
    for key, value in atoms.items():
        concentration[key] = value / natoms
    return concentration


def AtomicAvg(
    concentration: Mapping[str, float], magnitude: Mapping[str, float]
) -> float:
    """
    Compute an atomic average of a magnitude (legacy behavior).

    ⟨m⟩ = Σ_i c_i m_i, summed over the keys of `concentration`.

    Parameters
    ----------
    concentration : Mapping[str, float]
        Atomic fractions c_i, e.g. from `getConcentrations`.
    magnitude : Mapping[str, float]
        Per-element property m_i (molar mass, scattering length, cross
        section, ...), keyed by the same element symbols. Extra keys are
        ignored.

    Returns
    -------
    float
        The concentration-weighted average, in the unit of `magnitude`.

    Raises
    ------
    KeyError
        If a key of `concentration` is missing from `magnitude`.

    Examples
    --------
    >>> round(AtomicAvg({"Si": 1/3, "O": 2/3}, {"Si": 28.0855, "O": 15.999}), 4)
    20.0278
    """
    average: float = 0.0
    for key, value in concentration.items():
        average += float(value) * float(magnitude[key])
    return average
