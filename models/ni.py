"""
Ni powder diffraction model for the calibration of λ and 2θ0.

The D4 wavelength λ and zero-angle offset 2θ0 are obtained by fitting the
diffractogram of a Ni powder standard (fcc, a = 3.52 Å by default) with
`niPeaks10` (see ``toscana.calibration.d4nifit``, which builds the initial
parameters, runs the lmfit fit and shows the results).

Public functions
----------------
niPeaks10(x, I0, slope, quad, wavelength, twotheta0, A0..A9, G0..G9, S0..S9,
          lattice=NI_LATTICE_PARAMETER)
    Quadratic background plus ten asymmetric Gaussian peaks centred at the
    first ten fcc reflections computed from (wavelength, twotheta0, lattice).

Conventions and units
---------------------
x and peak positions in degrees (2θ); wavelength and lattice in Å; intensities
in the (arbitrary, normalised) units of the data; areas in intensity × degrees;
widths (Gaussian standard deviations) in degrees; asymmetries in 1/degree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toscana.numerics.line_shapes import GaussianA
from toscana.physics.crystallography import NI_LATTICE_PARAMETER, reflections_fcc

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def niPeaks10(
    x: ArrayLike,
    I0: float,
    slope: float,
    quad: float,
    wavelength: float,
    twotheta0: float,
    A0: float,
    A1: float,
    A2: float,
    A3: float,
    A4: float,
    A5: float,
    A6: float,
    A7: float,
    A8: float,
    A9: float,
    G0: float,
    G1: float,
    G2: float,
    G3: float,
    G4: float,
    G5: float,
    G6: float,
    G7: float,
    G8: float,
    G9: float,
    S0: float,
    S1: float,
    S2: float,
    S3: float,
    S4: float,
    S5: float,
    S6: float,
    S7: float,
    S8: float,
    S9: float,
    lattice: float = NI_LATTICE_PARAMETER,
) -> NDArray[np.float64]:
    """
    Evaluate a model with 10 asymmetric Gaussian peaks for Ni fcc reflections.

    `lattice` is the Ni lattice parameter in Å (default 3.52 Å). In an lmfit
    fit it is a parameter and must be kept fixed: λ and `lattice` are
    fully correlated (only λ/a is determined by the peak positions).

    Implemented model:

        y(x) = I0 + slope·x + quad·x² + Σ_{i=0..9} GaussianA(x, A_i, C_i, S_i, G_i)

        GaussianA(x, A, C, S, G) = A / (sqrt(2π) S) · exp(-(x-C)²/(2 S²))
                                   · exp(-G (x-C))

    where C_0..C_9 = ``reflections_fcc(wavelength, twotheta0, lattice)[:10]``
    are the 2θ positions (including the 2θ0 offset) of the first ten allowed
    fcc reflections: (111), (200), (220), (311), (222), (400), (331), (420),
    (422), (333)/(511).

    Parameters
    ----------
    x : array_like
        Scattering angles 2θ in degrees.
    I0 : float
        Constant background, in intensity units.
    slope : float
        Linear background coefficient, intensity/degree.
    quad : float
        Quadratic background coefficient, intensity/degree².
    wavelength : float
        Neutron wavelength λ in Å; with `lattice` it fixes the peak positions.
    twotheta0 : float
        Zero-angle offset 2θ0 in degrees, added to every peak position.
    A0, ..., A9 : float
        Peak areas (intensity × degrees) of reflections 0..9. The area equals
        A_i exactly only when G_i = 0.
    G0, ..., G9 : float
        Asymmetry parameters in 1/degree (exponential factor exp(-G (x-C))).
        G = 0 gives a symmetric Gaussian; the calibration fit keeps them
        fixed at their initial values (0).
    S0, ..., S9 : float
        Gaussian standard deviations σ in degrees (FWHM = 2.3548 σ). Must be
        non-zero.
    lattice : float, optional
        Cubic lattice parameter a of Ni in Å. Default ``NI_LATTICE_PARAMETER``
        (3.52 Å).

    Returns
    -------
    numpy.ndarray
        Model intensities, float64, same shape as `x`.

    Raises
    ------
    ValueError
        If ``reflections_fcc`` returns fewer than 10 angles (very long
        wavelength or very small lattice, so the 10-tuple unpacking fails).
    ZeroDivisionError
        If ``wavelength == 0`` (raised by ``reflections_fcc``).

    Notes
    -----
    - Only λ/a is determined by the peak positions (sin θ = λ·m/(2a)), so in a
      fit `lattice` must be fixed; the fitted λ is then proportional to the
      chosen lattice parameter (audit 2026-11, P7).
    - For λ >= 1.355 Å (with a = 3.52 Å) fewer than ten direct reflections
      exist and some C_i would be taken from the mirrored set of
      ``reflections_fcc``; this never happens at D4 wavelengths.
    - Each call recomputes the reflections (an O(N³) loop in
      ``reflections_fcc``), which dominates the evaluation time in a fit.
    - Changes from the legacy code (audit 2026-11, P7): the lattice parameter
      is an argument instead of the hard-coded 3.52024 Å.

    See Also
    --------
    toscana.physics.crystallography.reflections_fcc : peak positions.
    toscana.numerics.line_shapes.GaussianA : peak shape.
    toscana.calibration.d4nifit.fitting_model_d4nifit : the fit using this model.
    """
    q = np.asarray(x, dtype=np.float64)
    # Peak centres: first 10 allowed fcc reflections, 2θ + 2θ0 (degrees)
    C0, C1, C2, C3, C4, C5, C6, C7, C8, C9 = reflections_fcc(
        wavelength, twotheta0, lattice=lattice
    )[:10]

    diff = (
        I0
        + slope * q
        + quad * q * q
        + (
            GaussianA(q, A0, C0, S0, G0)
            + GaussianA(q, A1, C1, S1, G1)
            + GaussianA(q, A2, C2, S2, G2)
            + GaussianA(q, A3, C3, S3, G3)
            + GaussianA(q, A4, C4, S4, G4)
            + GaussianA(q, A5, C5, S5, G5)
            + GaussianA(q, A6, C6, S6, G6)
            + GaussianA(q, A7, C7, S7, G7)
            + GaussianA(q, A8, C8, S8, G8)
            + GaussianA(q, A9, C9, S9, G9)
        )
    )
    return np.asarray(diff, dtype=np.float64)
