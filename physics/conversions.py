"""
Conversions between scattering angle 2θ and momentum transfer Q.

D4 measures intensities as a function of the scattering angle 2θ; the
structure-factor analysis works in momentum transfer Q. These two functions are
the elastic (Bragg) conversion used everywhere in ToScaNA (regrouping, Q-space
output, plots of reflections).

Public functions
----------------
ang2q(x, wlength=0.5)
    2θ (degrees) -> Q (1/Å):  Q = (4π/λ) sin(θ).
q2ang(x, wlength=0.5)
    Q (1/Å) -> 2θ (degrees):  2θ = (360/π) arcsin(Qλ/4π).

Conventions and units
---------------------
2θ in degrees (θ is half the scattering angle), λ in Å, Q in 1/Å. The
conversion is elastic: it assumes that the scattered neutron has the incident
wavelength. Both functions accept scalars or array-likes and always return a
NumPy float64 value or array of the same shape (a 0-d input gives a NumPy
scalar).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def ang2q(x: ArrayLike, wlength: float = 0.5) -> NDArray[np.float64]:
    """
    Convert scattering angle (2θ, degrees) to momentum transfer Q (1/Å).

    Uses Bragg law (elastic scattering):
        Q = 4π/λ * sin((2θ)/2 * π/180)

    i.e. Q = (4π/λ) sin θ with θ = 2θ/2 converted to radians.

    Parameters
    ----------
    x : array_like
        Scattering angle(s) 2θ in degrees. Converted to a float64 array.
    wlength : float, optional
        Wavelength λ in Å. Must be positive. Default 0.5 Å (a typical D4
        wavelength; the actual value comes from the Ni calibration).

    Returns
    -------
    numpy.ndarray
        Q value(s) in 1/Å, same shape as `x` (NumPy scalar for scalar input).
        The function is not restricted to 0 <= 2θ <= 180°: negative angles give
        negative Q, angles beyond 180° give decreasing Q.

    Raises
    ------
    ValueError
        If ``wlength <= 0`` (message "Wrong wavelength in ang2q function").

    See Also
    --------
    q2ang : inverse conversion.

    Examples
    --------
    >>> float(ang2q(90.0, 0.5))
    17.771531752633464
    """
    if wlength <= 0:
        raise ValueError("Wrong wavelength in ang2q function")
    x_arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    return (4.0 * np.pi / wlength) * np.sin(x_arr / 2.0 * np.pi / 180.0)


def q2ang(x: ArrayLike, wlength: float = 0.5) -> NDArray[np.float64]:
    """
    Convert momentum transfer Q (1/Å) to scattering angle (2θ, degrees).

    Uses Bragg law (elastic scattering):
        2θ = 360/π * arcsin(Q * λ / 4π)

    This is the inverse of `ang2q` on 0 <= 2θ <= 180°.

    Parameters
    ----------
    x : array_like
        Q value(s) in 1/Å. Converted to a float64 array.
    wlength : float, optional
        Wavelength λ in Å. Must be positive. Default 0.5 Å.

    Returns
    -------
    numpy.ndarray
        Scattering angle(s) 2θ in degrees, same shape as `x`, in [-180, 180].

    Raises
    ------
    ValueError
        If ``wlength <= 0`` (message "Wrong wavelength in q2ang function").

    Notes
    -----
    Values with |Q| > 4π/λ (not reachable at this wavelength) are not
    rejected: ``np.arcsin`` returns NaN for them and NumPy emits a
    ``RuntimeWarning`` (invalid value).

    See Also
    --------
    ang2q : direct conversion.

    Examples
    --------
    >>> round(float(q2ang(17.771531752633464, 0.5)), 10)
    90.0
    """
    if wlength <= 0:
        raise ValueError("Wrong wavelength in q2ang function")
    x_arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    return (360.0 / np.pi) * np.arcsin(x_arr * wlength / 4.0 / np.pi)
