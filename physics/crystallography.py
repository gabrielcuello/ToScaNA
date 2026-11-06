"""
Crystallography of cubic lattices, used for the Ni powder calibration.

The wavelength λ and the zero-angle offset 2θ0 of D4 are calibrated by fitting
the diffractogram of a Ni powder (fcc) standard (see
``toscana.calibration.d4nifit`` and ``toscana.models.ni.niPeaks10``). This
module provides the positions of the fcc reflections used by that model, and
the geometric structure factors of the cubic Bravais lattices.

Public names
------------
NI_LATTICE_PARAMETER : float
    Default cubic lattice parameter of the Ni calibration standard, 3.52 Å.
    Every function using it accepts a ``lattice`` argument to override it
    (audit 2026-11, P7). In the Ni fit it is held fixed, because the peak
    positions only determine λ/a (λ and a are fully correlated): the fitted λ
    scales linearly with the value chosen here.
sf_fcc(h, k, l)
    Geometric structure factor of the fcc lattice (4 or 0).
sf_bcc(h, k, l)
    Geometric structure factor of the bcc lattice (2 or 0).
sf_sc(h, k, l)
    Geometric structure factor of the simple cubic lattice (always 1).
reflections_fcc(wavelength=0.5, twotheta0=0.0, lattice=NI_LATTICE_PARAMETER)
    2θ positions (degrees) of the allowed fcc reflections, plus their mirrored
    set 180° - 2θ - 2·2θ0.

Conventions and units
---------------------
Miller indices are integers; lattice parameters and wavelengths in Å; angles in
degrees. Cubic d-spacing d_hkl = a / sqrt(h² + k² + l²); Bragg law
λ = 2 d sin θ.

Notes
-----
Changes from the legacy code (audit 2026-11, P7): the Ni lattice parameter was
hard-coded as 3.52024 Å in five places; it is now this single default (3.52 Å)
and user-configurable. Going from 3.52024 to 3.52 Å lowers the fitted λ by
0.007 %.
"""

from __future__ import annotations

import numpy as np

# Default lattice parameter of the Ni powder calibration standard (fcc), in Å.
# Every function using it accepts a `lattice` argument to override it.
NI_LATTICE_PARAMETER = 3.52


def sf_fcc(miller_h: int, miller_k: int, miller_l: int) -> np.complex128:
    """
    Calculate the structure factor for a fcc lattice for a given plane hkl.

    Geometric structure factor (unit scattering length) of the face-centred
    cubic lattice with atoms at (0,0,0), (0,½,½), (½,0,½), (½,½,0):

        F(hkl) = 1 + exp(-iπ(k+l)) + exp(-iπ(h+l)) + exp(-iπ(h+k))

    For integer indices F = 4 when h, k, l are all even or all odd (allowed
    reflection) and F = 0 when they have mixed parity (extinct reflection).

    Preserves the legacy implementation from `legacy/lib/toscana.py`.

    Parameters
    ----------
    miller_h, miller_k, miller_l : int
        Miller indices h, k, l of the plane. Non-integer values are accepted
        by the arithmetic but have no crystallographic meaning.

    Returns
    -------
    numpy.complex128
        F(hkl). Because it is evaluated with complex exponentials, the
        imaginary part is not exactly 0 but a rounding residue (~1e-16);
        callers use the real part (as `reflections_fcc` does).

    See Also
    --------
    sf_bcc, sf_sc, reflections_fcc

    Examples
    --------
    >>> float(sf_fcc(1, 1, 1).real), float(sf_fcc(1, 0, 0).real)
    (4.0, 0.0)
    """
    result = (
        1.0
        + np.exp(-1j * np.pi * (miller_k + miller_l))
        + np.exp(-1j * np.pi * (miller_h + miller_l))
        + np.exp(-1j * np.pi * (miller_h + miller_k))
    )
    return np.complex128(result)


def sf_bcc(miller_h: int, miller_k: int, miller_l: int) -> np.complex128:
    """
    Calculate the structure factor for a bcc lattice for a given plane hkl.

    Geometric structure factor of the body-centred cubic lattice with atoms at
    (0,0,0) and (½,½,½):

        F(hkl) = 1 + exp(-iπ(h+k+l))

    For integer indices F = 2 when h+k+l is even (allowed) and F = 0 when it
    is odd (extinct).

    Preserves the legacy implementation from `legacy/lib/toscana.py`.

    Parameters
    ----------
    miller_h, miller_k, miller_l : int
        Miller indices h, k, l of the plane.

    Returns
    -------
    numpy.complex128
        F(hkl), with a rounding residue (~1e-16) in the imaginary part.

    See Also
    --------
    sf_fcc, sf_sc

    Examples
    --------
    >>> float(sf_bcc(1, 1, 0).real)
    2.0
    """
    result = 1.0 + np.exp(-1j * np.pi * (miller_h + miller_k + miller_l))
    return np.complex128(result)


def sf_sc(miller_h: int, miller_k: int, miller_l: int) -> float:
    """
    Calculate the structure factor for a sc lattice for a given plane hkl.

    The simple cubic lattice has one atom per cell at (0,0,0), so
    F(hkl) = 1 for every plane (no systematic extinctions).

    Legacy behavior: always returns 1.0.

    Parameters
    ----------
    miller_h, miller_k, miller_l : int
        Miller indices h, k, l (ignored; kept for a uniform signature with
        `sf_fcc` and `sf_bcc`).

    Returns
    -------
    float
        Always 1.0 (a Python float, not complex).

    See Also
    --------
    sf_fcc, sf_bcc
    """
    _ = (miller_h, miller_k, miller_l)
    return 1.0


def reflections_fcc(
    wavelength: float = 0.5,
    twotheta0: float = 0.0,
    lattice: float = NI_LATTICE_PARAMETER,
) -> np.ndarray:
    """
    Produce a list of the angular positions for a fcc lattice (legacy behavior).

    Returns angles in degrees for allowed reflections in the range [0, 180].
    `lattice` is the cubic lattice parameter in Å (default: Ni, 3.52 Å).

    Algorithm, as implemented:

    1. ``moduleMax = int(2·lattice/wavelength)``. Loop over all non-negative
       Miller indices 0 <= h, k, l < moduleMax.
    2. For each (hkl) compute m = sqrt(h² + k² + l²) and the real part of
       ``sf_fcc(h, k, l)``; keep the plane if ``sf·m > 0`` (allowed fcc
       reflection, excluding (000)).
    3. Bragg law: sin θ = m·λ / (2a) (i.e. λ = 2 d sin θ with d = a/m); keep it
       if sin θ < 1.
    4. Remove duplicate sin θ values (``set``), so planes with the same
       h²+k²+l² (e.g. (333) and (511), or permutations of the indices) give a
       single reflection, and sort in increasing order.
    5. First set: 2θ_i + 2θ0, with 2θ_i = 2 arcsin(sin θ_i) in degrees.
    6. Mirrored set: ``sorted(180 - first set) - 2θ0``, i.e.
       180° - 2θ_i - 2·2θ0, in increasing order (so the reflection with the
       largest 2θ_i comes first).
    7. Return the concatenation [first set, mirrored set].

    For Ni the first allowed reflections are, in order, h²+k²+l² = 3, 4, 8,
    11, 12, 16, 19, 20, 24, 27, i.e. (111), (200), (220), (311), (222),
    (400), (331), (420), (422), (333)/(511).

    Parameters
    ----------
    wavelength : float, optional
        Neutron wavelength λ in Å. Default 0.5 Å.
    twotheta0 : float, optional
        Zero-angle offset 2θ0 in degrees, added to every computed 2θ of the
        first set (and subtracted twice in the mirrored set). Default 0.0.
    lattice : float, optional
        Cubic lattice parameter a in Å. Default ``NI_LATTICE_PARAMETER``
        (3.52 Å, Ni).

    Returns
    -------
    numpy.ndarray
        1-D float64 array of length 2N, N being the number of distinct allowed
        reflections with sin θ < 1. Elements [0, N) are 2θ_i + 2θ0 in increasing
        order; elements [N, 2N) are 180° - 2θ_i - 2·2θ0 in increasing order.
        Callers (``niPeaks10``, ``setting_model_d4nifit``,
        ``showing_results_d4nifit``) take the first 10 elements.

    Raises
    ------
    ZeroDivisionError
        If ``wavelength == 0``.

    Notes
    -----
    - The first set lies in [0, 180] only for 2θ0 = 0; with an offset it is
      shifted by 2θ0.
    - A negative ``wavelength`` gives ``moduleMax <= 0``, no loop iteration,
      and an empty array (no error).
    - Only N >= 10 reflections exist when sqrt(27)·λ/(2a) < 1, i.e.
      λ < 1.355 Å for Ni. For longer wavelengths the first 10 elements taken
      by the callers would include mirrored angles (or fewer than 10
      elements would exist). D4 works at λ ≈ 0.35-0.7 Å, so this does not
      happen in practice.
    - The loop is O(moduleMax³): about 2.4e3 iterations for λ = 0.5 Å.
    - Changes from the legacy code (audit 2026-11, P7): the lattice
      parameter is an argument (default 3.52 Å) instead of the hard-coded
      3.52024 Å.

    See Also
    --------
    sf_fcc : selection rule used here.
    toscana.models.ni.niPeaks10 : Ni powder model built on the first 10 angles.

    Examples
    --------
    >>> r = reflections_fcc(0.5)
    >>> len(r)
    134
    >>> [round(float(v), 3) for v in r[:4]]
    [14.132, 16.332, 23.177, 27.249]
    >>> [round(float(v), 3) for v in reflections_fcc(0.5, 0.1)[-2:]]
    [163.468, 165.668]
    """
    moduleMax = int(2.0 * lattice / wavelength)
    sinus: list[float] = []
    for miller_h in range(moduleMax):
        for miller_k in range(moduleMax):
            for miller_l in range(moduleMax):
                # m = sqrt(h² + k² + l²) = a / d_hkl
                module = float(
                    np.sqrt(
                        miller_h * miller_h + miller_k * miller_k + miller_l * miller_l
                    )
                )
                # fcc selection rule: sf = 4 (unmixed parity) or 0 (mixed parity)
                sf = float(sf_fcc(miller_h, miller_k, miller_l).real)
                if sf * module > 0:
                    # Bragg law: sin θ = λ / (2 d) = m λ / (2 a)
                    sintheta = module * wavelength / 2.0 / lattice
                    if sintheta < 1.0:
                        sinus.append(float(sintheta))
    # Distinct sin θ values in increasing order -> 2θ in radians
    rad_fcc = 2.0 * np.arcsin(sorted(set(sinus)))
    # First set: 2θ + 2θ0 (degrees)
    deg_fcc1 = twotheta0 + np.array(180.0 / np.pi * rad_fcc, dtype=np.float64)
    # Mirrored set: 180 - (2θ + 2θ0) - 2θ0 = 180 - 2θ - 2·2θ0, increasing order
    deg_fcc2 = -twotheta0 + np.array(sorted(180.0 - deg_fcc1), dtype=np.float64)
    deg_fcc = np.concatenate([deg_fcc1, deg_fcc2])
    return np.asarray(deg_fcc, dtype=np.float64)
