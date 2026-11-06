"""
Energy-dependent neutron cross-section models.

Used to estimate the total scattering cross section of a sample at epithermal
energies and the probability that a neutron interacting with the sample is
scattered rather than absorbed (e.g. for multiple-scattering and attenuation
estimates).

Public functions
----------------
XS_model(Eval, Eeff, composition_vec, bound_xs_vec, A_vec)
    Epithermal total scattering cross section
    σ(E) = Σ c_i σ_i [A_i/(A_i+1)]² [1 + E_eff/(2 A_i E)]   (barn).
scattering_probability(E0, scat_xs, composition_vec, abs_xs_vec)
    σ_s / (σ_a + σ_s), with σ_a = Σ c_i σ_a,i / sqrt(E0/E_th) (1/v law).

Conventions and units
---------------------
Cross sections in barn per atom (averaged over the formula unit with the atomic
fractions c_i); energies in meV for `scattering_probability`; any common unit
for `XS_model` (only the ratio E_eff/E enters). The thermal reference energy is
``toscana.physics.properties.THERMAL_ENERGY_MEV`` = 25.30 meV (2200 m/s,
1.798 Å).

Notes
-----
Changes from the legacy code (audit 2026-11, C30): `scattering_probability`
used a hard-coded 25.3 meV; it now uses ``THERMAL_ENERGY_MEV`` =
81.8042/1.798² = 25.3044 meV, the same reference as
``toscana.physics.properties.getAbsXS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toscana.physics.properties import THERMAL_ENERGY_MEV

if TYPE_CHECKING:
    from numpy.typing import NDArray


def XS_model(  # noqa: N802 (legacy API)
    Eval: float,
    Eeff: float,
    composition_vec: list[float] | NDArray[np.float64],
    bound_xs_vec: list[float] | NDArray[np.float64],
    A_vec: list[float] | NDArray[np.float64],
) -> float:
    """
    Calculate the neutron total cross section in the epithermal limit (legacy behavior).

    Implemented formula (sum over the elements i of the sample):

        σ(E) = Σ_i c_i · σ_i · [A_i/(A_i+1)]² · [1 + E_eff / (2 A_i E)]

    [A_i/(A_i+1)]² converts the bound cross section σ_i to the free-atom one;
    the last factor is the first-order correction of the free-gas model for
    the thermal motion of the target atoms, valid for E ≫ E_eff. For
    E -> ∞ the result tends to the free cross section Σ c_i σ_i [A_i/(A_i+1)]².

    Parameters
    ----------
    Eval : float
        Neutron energy E at which the cross section is evaluated. Same unit
        as `Eeff` (e.g. meV).
    Eeff : float
        Effective energy E_eff of the thermal motion of the atoms (k_B T_eff),
        same unit as `Eval`.
    composition_vec : sequence of float
        Atomic fractions c_i (normally summing to 1).
    bound_xs_vec : sequence of float
        Bound scattering cross sections σ_i in barn, same order and length as
        `composition_vec`.
    A_vec : sequence of float
        Mass ratios A_i = m_i/m_n (see
        ``toscana.physics.properties.getMassNumber``), same order and length.

    Returns
    -------
    float
        Total scattering cross section per atom, in barn.

    Raises
    ------
    ValueError
        If the three vectors cannot be broadcast together (different lengths).

    Notes
    -----
    No input validation: ``Eval == 0`` or ``A_i == 0`` give inf or NaN with a
    NumPy ``RuntimeWarning`` rather than an exception.

    See Also
    --------
    toscana.physics.properties.getFreeXS : free cross section of one nucleus.

    Examples
    --------
    >>> round(XS_model(1e12, 1.0, [1.0], [5.0], [50.0]), 6)  # ≈ free value
    4.805844
    """
    composition_arr = np.array(composition_vec, dtype=np.float64)
    bound_xs_arr = np.array(bound_xs_vec, dtype=np.float64)
    a_arr = np.array(A_vec, dtype=np.float64)

    # Free-atom factor [A/(A+1)]² per element
    a_vals = (a_arr / (a_arr + 1.0)) ** 2
    result = np.sum(
        composition_arr
        * bound_xs_arr
        * a_vals
        * (1.0 + float(Eeff) / (2.0 * a_arr * float(Eval)))
    )
    return float(result)


def scattering_probability(  # noqa: N802 (legacy API)
    E0: float,
    scat_xs: float,
    composition_vec: list[float] | NDArray[np.float64],
    abs_xs_vec: list[float] | NDArray[np.float64],
) -> float:
    """
    Calculate the scattering probability for a given neutron incident energy (legacy behavior).

    Implemented formula:

        σ_a(E0) = [Σ_i c_i σ_a,i] / sqrt(E0 / E_th)
        P = 1 - σ_a / (σ_a + σ_s) = σ_s / (σ_a + σ_s)

    with E_th = ``THERMAL_ENERGY_MEV`` = 25.30 meV. The absorption follows the
    1/v law (σ_a ∝ 1/v ∝ 1/sqrt(E)), scaled from the tabulated thermal values;
    the scattering cross section σ_s is taken as given (energy independent
    here; it can come from `XS_model`).

    Parameters
    ----------
    E0 : float
        Incident neutron energy in meV.
    scat_xs : float
        Scattering cross section σ_s per atom at E0, in barn.
    composition_vec : sequence of float
        Atomic fractions c_i.
    abs_xs_vec : sequence of float
        Tabulated (2200 m/s) absorption cross sections σ_a,i in barn, same
        order and length as `composition_vec`.

    Returns
    -------
    float
        Probability (0 to 1) that an interaction is a scattering event.

    Raises
    ------
    ValueError
        If `composition_vec` and `abs_xs_vec` have different lengths
        (``np.dot``).

    Notes
    -----
    No input validation: E0 <= 0 or σ_a = σ_s = 0 give inf/NaN with a NumPy
    ``RuntimeWarning`` rather than an exception.

    Changes from the legacy code (audit 2026-11, C30): the reference energy
    was a hard-coded 25.3 meV; it is now ``THERMAL_ENERGY_MEV`` (25.3044 meV),
    consistent with the 1.798 Å used by
    ``toscana.physics.properties.getAbsXS``.

    Examples
    --------
    >>> from toscana.physics.properties import THERMAL_ENERGY_MEV
    >>> round(scattering_probability(THERMAL_ENERGY_MEV, 5.0, [1.0], [5.08]), 6)
    0.496032
    """
    composition_arr = np.array(composition_vec, dtype=np.float64)
    abs_xs_arr = np.array(abs_xs_vec, dtype=np.float64)

    # 1/v law relative to the 2200 m/s reference (E0 in meV)
    abs_xs = float(np.dot(composition_arr, abs_xs_arr)) / np.sqrt(
        float(E0) / THERMAL_ENERGY_MEV
    )
    total_xs = abs_xs + float(scat_xs)
    return float(1.0 - abs_xs / total_xs)
