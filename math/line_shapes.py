from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def sigmoidal(
    Q: ArrayLike,
    lowQ: float = 0.4,
    highQ: float = 0.2,
    Q0: float = 7.0,
    dQ: float = 2.4,
) -> NDArray[np.float64]:
    """
    Calculate a sigmoidal-like function, preserving legacy behavior.

    Notes
    -----
    The legacy implementation uses:
        delta = exp((Q - Q0) / dQ)
        return delta*lowQ + (1-delta)*highQ
    which is not a standard logistic form, but we preserve it for parity.
    """
    q = np.asarray(Q, dtype=np.float64)
    delta = np.exp((q - Q0) / dQ)
    return delta * lowQ + (1.0 - delta) * highQ


def siglin(
    Q: ArrayLike,
    lowQ: float = 0.4,
    highQ: float = 0.2,
    Q0: float = 7.0,
    dQ: float = 2.4,
    a1: float = 0.001,
) -> NDArray[np.float64]:
    """
    Calculate a sigmoidal-like function with an (unused) linear term (legacy behavior).

    Notes
    -----
    The legacy implementation ignores `a1` and returns the same expression as `sigmoidal`.
    We preserve that behavior for parity.
    """
    _ = a1
    return sigmoidal(Q, lowQ=lowQ, highQ=highQ, Q0=Q0, dQ=dQ)


def step(q: float, qmax: float = 23.5) -> float:
    """
    Evaluate the Step function (legacy behavior).

    Returns 1.0 for 0 <= q < qmax, 0.0 otherwise.
    Raises ValueError if qmax <= 0 (legacy printed an error and returned -999,
    which was then used silently as a window value).
    """
    if qmax <= 0:
        raise ValueError("Non positive value for qmax in step function.")

    if (q < 0.0) or (q >= qmax):
        return 0.0
    return 1.0


def Lorentzian_error(
    x: ArrayLike,
    A: float = 1,
    x0: float = 0,
    gamma: float = 1,
    bckg: float = 0,
    e_A: float = 0,
    e_x0: float = 0,
    e_gamma: float = 0,
    e_bckg: float = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Evaluate a Lorentzian (height A, FWHM gamma) and its propagated error.

    Notes
    -----
    Bug fix vs legacy: the x0 term was `(...)**2 * peak**2 * e_x0 * 22`
    (typo for e_x0**2, and missing one factor of `peak`). With
    p = (γ/2)^2 / ((x-x0)^2 + (γ/2)^2), ∂L/∂x0 = 2A(x-x0)/(γ/2)^2 · p^2.
    """
    q = np.asarray(x, dtype=np.float64)
    gam2 = gamma / 2.0
    lor = A * gam2**2 / ((q - x0) ** 2 + gam2**2) + bckg
    peak = gam2**2 / ((q - x0) ** 2 + gam2**2)
    error = e_bckg**2 + peak**2 * e_A**2
    error += (A / gam2 * peak) ** 2 * (1 - peak) ** 2 * e_gamma**2
    error += (2 * A * (q - x0) / gam2**2 * peak**2) ** 2 * e_x0**2
    error = np.sqrt(error)
    return np.asarray(lor, dtype=np.float64), np.asarray(error, dtype=np.float64)


def Gaussian_error(
    x: ArrayLike,
    A: float = 1,
    x0: float = 0,
    sigma: float = 1,
    bckg: float = 0,
    e_A: float = 0,
    e_x0: float = 0,
    e_sigma: float = 0,
    e_bckg: float = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Evaluate a Gaussian and its propagated error (legacy behavior).
    """
    q = np.asarray(x, dtype=np.float64)
    gau = (
        A / np.sqrt(2 * np.pi) / sigma * np.exp(-((q - x0) ** 2) / 2.0 / sigma**2)
        + bckg
    )
    peak = (gau - bckg) / A
    error = e_bckg**2 + peak**2 * e_A**2
    error += (
        peak**2 * (A / sigma) ** 2 * (((q - x0) ** 2) / sigma**2 - 1) ** 2 * e_sigma**2
    )
    error += peak**2 * (A / sigma**2 * (q - x0)) ** 2 * e_x0**2
    error = np.sqrt(error)
    return np.asarray(gau, dtype=np.float64), np.asarray(error, dtype=np.float64)


def LorGau(
    x: ArrayLike,
    f0: float = 1.0,
    eta: float = 0.5,
    sigma: float = 2.0,
    gamma: float = 2.0,
    bckg: float = 0.0,
) -> NDArray[np.float64]:
    """
    Evaluate a Gaussian/Lorentzian mixture peak centered at 0 (legacy behavior).
    """
    q = np.asarray(x, dtype=np.float64)
    lor = gamma * gamma / 4.0 / (q * q + gamma * gamma / 4.0)
    gau = np.exp(-1.0 * q * q / 2.0 / sigma / sigma)
    return np.asarray(f0 * (eta * gau + (1.0 - eta) * lor) + bckg, dtype=np.float64)


def LorGau_error(
    x: ArrayLike,
    f0: tuple[float, float] = (1.0, 0.0),
    eta: tuple[float, float] = (0.5, 0.0),
    sigma: tuple[float, float] = (2.0, 0.0),
    gamma: tuple[float, float] = (2.0, 0.0),
    bckg: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Evaluate the `LorGau` mixture peak and its propagated error.

    Each argument is a (value, error) pair; the value is identical to `LorGau`.

    Notes
    -----
    Bug fixes vs legacy: legacy used the area-normalised `Gaussian_error`
    (peak height 1/(σ√2π)) while `LorGau` uses a unit-height Gaussian, so the
    returned value differed from `LorGau`; the η, σ and γ error terms also
    lacked the factor f0.
    """
    q = np.asarray(x, dtype=np.float64)
    g2 = gamma[0] / 2.0
    lor = g2**2 / (q * q + g2**2)
    gau = np.exp(-q * q / 2.0 / sigma[0] ** 2)
    lorgau = bckg[0] + f0[0] * (eta[0] * gau + (1.0 - eta[0]) * lor)

    d_f0 = eta[0] * gau + (1.0 - eta[0]) * lor
    d_eta = f0[0] * (gau - lor)
    d_sigma = f0[0] * eta[0] * gau * q * q / sigma[0] ** 3
    d_gamma = f0[0] * (1.0 - eta[0]) * lor * (1.0 - lor) / g2
    error = np.sqrt(
        bckg[1] ** 2
        + (d_f0 * f0[1]) ** 2
        + (d_eta * eta[1]) ** 2
        + (d_sigma * sigma[1]) ** 2
        + (d_gamma * gamma[1]) ** 2
    )
    return np.asarray(lorgau, dtype=np.float64), np.asarray(error, dtype=np.float64)


def Lorentzian(
    x: ArrayLike, A: float, x0: float, gamma: float, bckg: float
) -> NDArray[np.float64]:
    """
    Evaluate a Lorentzian line-shape, preserving legacy behavior.
    """
    q = np.asarray(x, dtype=np.float64)
    lor = A * gamma * gamma / 4.0 / ((q - x0) * (q - x0) + gamma * gamma / 4.0) + bckg
    return lor


def Gaussian(
    x: ArrayLike, A: float, x0: float, sigma: float, bckg: float
) -> NDArray[np.float64]:
    """
    Evaluate a normalized Gaussian line-shape, preserving legacy behavior.
    """
    q = np.asarray(x, dtype=np.float64)
    gau = (
        A / np.sqrt(2 * np.pi) / sigma * np.exp(-((q - x0) ** 2) / 2.0 / sigma**2)
        + bckg
    )
    return np.asarray(gau, dtype=np.float64)


def GaussianA(
    x: ArrayLike, A: float, x0: float, sigma: float, asym: float = 0
) -> NDArray[np.float64]:
    """
    Evaluate an asymmetric Gaussian (exponential asymmetry), preserving legacy behavior.
    """
    q = np.asarray(x, dtype=np.float64)
    result = (
        A
        / np.sqrt(2 * np.pi)
        / sigma
        * np.exp(-((q - x0) ** 2) / 2.0 / sigma**2)
        * np.exp(-asym * (q - x0))
    )
    return np.asarray(result, dtype=np.float64)
