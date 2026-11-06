"""
Low-degree polynomials in the form needed by `scipy.optimize.curve_fit`.

Place in the processing chain: models for smooth backgrounds and Q
dependences (e.g. polynomial fits of corrections versus Q) during the
reduction and calibration.

Public functions
----------------
polyQ1(x, a0, a1)
    a0 + a1·x.
polyQ2(x, a0, a1, a2)
    a0 + a1·x + a2·x².
polyQ4(x, a0, a1, a2, a3, a4)
    a0 + a1·x + ... + a4·x⁴.

Conventions
-----------
Coefficients are given in increasing order of the power (the opposite of
`numpy.polyval`). Evaluation uses the Horner scheme. `x` is converted to a
float64 array; the result has its shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def polyQ1(x: ArrayLike, a0: float, a1: float) -> NDArray[np.float64]:
    """
    Evaluate a 1st-degree polynomial: poly(x) = a0 + a1 * x.

    Parameters
    ----------
    x : array_like
        The range of abscissas where the function is evaluated.
    a0, a1 : float
        Polynomial coefficients.

    Returns
    -------
    numpy.ndarray
        The evaluated polynomial (float64, same shape as `x`).
    """
    q = np.asarray(x, dtype=np.float64)
    return a1 * q + a0


def polyQ2(x: ArrayLike, a0: float, a1: float, a2: float) -> NDArray[np.float64]:
    """
    Evaluate a 2nd-degree polynomial: poly(x) = a0 + a1 * x + a2 * x**2.

    Parameters
    ----------
    x : array_like
        The range of abscissas where the function is evaluated.
    a0, a1, a2 : float
        Polynomial coefficients.

    Returns
    -------
    numpy.ndarray
        The evaluated polynomial (float64, same shape as `x`).

    Notes
    -----
    Horner scheme: (a2·x + a1)·x + a0.
    """
    q = np.asarray(x, dtype=np.float64)
    return (a2 * q + a1) * q + a0


def polyQ4(
    x: ArrayLike, a0: float, a1: float, a2: float, a3: float, a4: float
) -> NDArray[np.float64]:
    """
    Evaluate a 4th-degree polynomial: poly(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4.

    Parameters
    ----------
    x : array_like
        The range of abscissas where the function is evaluated.
    a0, a1, a2, a3, a4 : float
        Polynomial coefficients.

    Returns
    -------
    numpy.ndarray
        The evaluated polynomial (float64, same shape as `x`).

    Notes
    -----
    Horner scheme: a0 + (a1 + (a2 + (a3 + a4·x)·x)·x)·x.
    """
    q = np.asarray(x, dtype=np.float64)
    return a0 + (a1 + (a2 + (a3 + a4 * q) * q) * q) * q
