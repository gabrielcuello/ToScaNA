"""
Peak and step functions (line shapes) with optional error propagation.

Place in the processing chain: used to model the Q dependence of corrections
(`sigmoidal`, `siglin`: inelasticity / Placzek-like transitions used by
`models.scattering.inelastic` and `vanaQdep`), as the truncation window of
the Fourier transform (`step`), and to fit Bragg peaks (Gaussian, Lorentzian,
mixtures; `GaussianA` for the asymmetric Ni reflections of the calibration).

Public functions and constants
------------------------------
SIGMOID_Q0 = 7.0, SIGMOID_DQ = 2.4
    Default centre and width (Å⁻¹) of the low-Q -> high-Q transition.
sigmoidal(Q, lowQ, highQ, Q0, dQ)
    Logistic step from lowQ (Q -> -∞) to highQ (Q -> +∞).
siglin(Q, lowQ, highQ, Q0, dQ, a1)
    sigmoidal(Q) + a1·Q.
step(q, qmax)
    1 for 0 <= q < qmax, else 0 (the "no window" case of `sineFT`).
Lorentzian(x, A, x0, gamma, bckg) / Lorentzian_error(...)
    Lorentzian of height A and FWHM γ plus background [value and error].
Gaussian(x, A, x0, sigma, bckg) / Gaussian_error(...)
    Area-normalised Gaussian (area A, standard deviation σ) plus background
    [value and error].
GaussianA(x, A, x0, sigma, asym)
    Area-normalised Gaussian times exp(-asym·(x - x0)).
LorGau(x, f0, eta, sigma, gamma, bckg) / LorGau_error(...)
    Mixture of unit-height Gaussian and Lorentzian centred at 0 [value and
    error].

Conventions
-----------
x, x0, sigma, gamma are in the units of the abscissa (degrees of 2θ or Å⁻¹);
heights/areas and backgrounds in the units of the ordinate. Error functions
return (value, error) with 1σ errors propagated to first order, assuming
uncorrelated parameters (no covariance terms):
σ_f² = Σ_p (∂f/∂p)² σ_p².
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


# Default centre (1/Å) and width (1/Å) of the low-Q -> high-Q transition
# used by `sigmoidal`, `siglin`, `models.scattering.inelastic` and `vanaQdep`.
SIGMOID_Q0 = 7.0
SIGMOID_DQ = 2.4


def sigmoidal(
    Q: ArrayLike,
    lowQ: float = 0.4,
    highQ: float = 0.2,
    Q0: float = SIGMOID_Q0,
    dQ: float = SIGMOID_DQ,
) -> NDArray[np.float64]:
    """
    Logistic step from `lowQ` (Q << Q0) to `highQ` (Q >> Q0).

        f(Q) = (lowQ + highQ·δ) / (1 + δ),   δ = exp((Q - Q0)/dQ)

    f(Q0) = (lowQ + highQ)/2 (inflexion point); dQ is the transition width
    (f goes from 27 % to 73 % of the step between Q0 - dQ and Q0 + dQ).

    Parameters
    ----------
    Q : array_like
        Momentum transfer (Å⁻¹), any shape.
    lowQ : float, optional
        Limit for Q -> -∞ (default 0.4). Note that f(0) is not exactly lowQ
        unless Q0/dQ >> 1 (with the defaults f(0) ≈ 0.4 - 0.2·0.051 = 0.390).
    highQ : float, optional
        Limit for Q -> +∞ (default 0.2).
    Q0 : float, optional
        Centre of the transition (Å⁻¹), default ``SIGMOID_Q0`` = 7.0.
    dQ : float, optional
        Width of the transition (Å⁻¹), default ``SIGMOID_DQ`` = 2.4. A
        negative value reverses the direction; 0 divides by zero.

    Returns
    -------
    ndarray of float64, same shape as `Q`
        f(Q), in the units of lowQ/highQ.

    Notes
    -----
    Implemented as lowQ + (highQ - lowQ)·expit((Q - Q0)/dQ), which equals the
    formula above.

    Bug fix vs legacy (audit 2026-11, C26): legacy returned
    δ·lowQ + (1-δ)·highQ, which is unbounded, has negative weights for Q > Q0,
    and goes from highQ to lowQ (the opposite of what the parameter names
    state). Computed with `scipy.special.expit` so large |Q - Q0|/dQ does not
    overflow.
    """
    q = np.asarray(Q, dtype=np.float64)
    weight_high = expit((q - Q0) / dQ)  # δ / (1 + δ)
    return np.asarray(lowQ + (highQ - lowQ) * weight_high, dtype=np.float64)


def siglin(
    Q: ArrayLike,
    lowQ: float = 0.4,
    highQ: float = 0.2,
    Q0: float = SIGMOID_Q0,
    dQ: float = SIGMOID_DQ,
    a1: float = 0.001,
) -> NDArray[np.float64]:
    """
    Logistic step (`sigmoidal`) plus a linear term: f(Q) = sigmoidal(Q) + a1·Q.

    Parameters
    ----------
    Q : array_like
        Momentum transfer (Å⁻¹), any shape.
    lowQ, highQ, Q0, dQ : float, optional
        As in `sigmoidal` (defaults 0.4, 0.2, ``SIGMOID_Q0``, ``SIGMOID_DQ``).
    a1 : float, optional
        Slope of the linear term (units of f per Å⁻¹), default 0.001. The
        term a1·Q is added over the whole Q range (it is 0 at Q = 0).

    Returns
    -------
    ndarray of float64, same shape as `Q`
        sigmoidal(Q) + a1·Q.

    Notes
    -----
    Divergence from legacy: legacy ignored `a1` and returned `sigmoidal`.
    The linear term is now applied as an additive slope over the whole Q
    range; use a1 = 0 to obtain the pure sigmoid.
    """
    q = np.asarray(Q, dtype=np.float64)
    return np.asarray(sigmoidal(q, lowQ=lowQ, highQ=highQ, Q0=Q0, dQ=dQ) + a1 * q)


def step(q: float, qmax: float = 23.5) -> float:
    """
    Evaluate the step (box) function: 1 for 0 <= q < qmax, else 0.

    Used by `sineFT` with ``window=0`` as the "no window" modification
    function (plain truncation at qmax).

    Parameters
    ----------
    q : float
        Momentum transfer Q (Å⁻¹).
    qmax : float, optional
        Cut-off (Å⁻¹), default 23.5. Must be > 0.

    Returns
    -------
    float
        1.0 for 0 <= q < qmax, 0.0 otherwise (q < 0 or q >= qmax). A NaN q
        returns 1.0 (both comparisons are False).

    Raises
    ------
    ValueError
        If ``qmax <= 0``.

    Notes
    -----
    Changes from the legacy code (audit 2026-11, P5): legacy printed an
    error and returned -999 for qmax <= 0, which was then used silently as a
    window value; it now raises.
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

        L(x) = A·p + bckg,   p = (γ/2)² / ((x - x0)² + (γ/2)²)

    (same value as `Lorentzian`), with the first-order error

        σ_L² = σ_b² + p²σ_A² + [(A/(γ/2))·p(1 - p)]²σ_γ²
               + [2A(x - x0)/(γ/2)² · p²]²σ_x0²,

    using ∂L/∂A = p, ∂L/∂γ = (A/(γ/2))·p(1 - p),
    ∂L/∂x0 = 2A(x - x0)/(γ/2)² · p², ∂L/∂bckg = 1.

    Parameters
    ----------
    x : array_like
        Abscissas (units of x), any shape.
    A : float, optional
        Peak height above background (default 1).
    x0 : float, optional
        Peak position (default 0).
    gamma : float, optional
        Full width at half maximum γ (default 1); must be != 0.
    bckg : float, optional
        Constant background (default 0).
    e_A, e_x0, e_gamma, e_bckg : float, optional
        1σ errors of the parameters (default 0).

    Returns
    -------
    value : ndarray of float64
        L(x), same shape as `x`.
    error : ndarray of float64
        σ_L(x), same shape as `x`.

    Notes
    -----
    Parameter correlations (covariances from a fit) are not included.

    Bug fix vs legacy (audit 2026-11, C11): the x0 term was
    `(...)**2 * peak**2 * e_x0 * 22` (typo for e_x0**2, and missing one
    factor of `peak`). With p = (γ/2)^2 / ((x-x0)^2 + (γ/2)^2), ∂L/∂x0 = 2A(x-x0)/(γ/2)^2 · p^2.
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
    Evaluate an area-normalised Gaussian and its propagated error.

        G(x) = A·φ(x) + bckg,   φ(x) = exp(-(x - x0)²/(2σ²)) / (σ√(2π))

    (same value as `Gaussian`; A is the area), with the first-order error

        σ_G² = σ_b² + φ²σ_A² + φ²(A/σ)²[(x - x0)²/σ² - 1]²σ_σ²
               + φ²[A(x - x0)/σ²]²σ_x0²,

    using ∂G/∂A = φ, ∂G/∂σ = (A/σ)φ[(x - x0)²/σ² - 1],
    ∂G/∂x0 = A(x - x0)/σ² · φ, ∂G/∂bckg = 1.

    Parameters
    ----------
    x : array_like
        Abscissas (units of x), any shape.
    A : float, optional
        Area of the peak (default 1). Must be != 0 (see Notes).
    x0 : float, optional
        Peak position (default 0).
    sigma : float, optional
        Standard deviation σ (default 1); FWHM = 2√(2 ln 2)·σ ≈ 2.3548σ.
    bckg : float, optional
        Constant background (default 0).
    e_A, e_x0, e_sigma, e_bckg : float, optional
        1σ errors of the parameters (default 0).

    Returns
    -------
    value : ndarray of float64
        G(x), same shape as `x`.
    error : ndarray of float64
        σ_G(x), same shape as `x`.

    Notes
    -----
    φ is recovered as (G - bckg)/A, so A = 0 gives NaN errors (0/0), and a
    background much larger than the peak loses some precision in φ.
    Parameter correlations are not included.
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
    Evaluate a Gaussian/Lorentzian mixture peak centred at 0.

        f(x) = f0·[η·exp(-x²/(2σ²)) + (1 - η)·(γ/2)²/(x² + (γ/2)²)] + bckg

    Both components have unit height at x = 0, so f(0) = f0 + bckg for any η.

    Parameters
    ----------
    x : array_like
        Abscissas (units of x), measured from the peak centre.
    f0 : float, optional
        Peak height above background (default 1.0).
    eta : float, optional
        Gaussian fraction η (default 0.5); normally 0 <= η <= 1 (not
        checked).
    sigma : float, optional
        Standard deviation of the Gaussian component (default 2.0).
    gamma : float, optional
        FWHM of the Lorentzian component (default 2.0).
    bckg : float, optional
        Constant background (default 0.0).

    Returns
    -------
    ndarray of float64, same shape as `x`
        f(x).

    Notes
    -----
    Legacy behaviour, unchanged. The Gaussian and Lorentzian widths are
    independent parameters (this is not a pseudo-Voigt with a common FWHM).
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

    Each argument is a (value, error) pair; the value is identical to
    `LorGau`. With g = exp(-x²/(2σ²)) and l = (γ/2)²/(x² + (γ/2)²):

        ∂f/∂f0 = η g + (1 - η) l,      ∂f/∂η = f0 (g - l),
        ∂f/∂σ  = f0 η g x²/σ³,       ∂f/∂γ = f0 (1 - η) l (1 - l)/(γ/2),
        ∂f/∂bckg = 1,
        σ_f² = Σ_p (∂f/∂p)² σ_p²      (uncorrelated parameters).

    Parameters
    ----------
    x : array_like
        Abscissas (units of x), measured from the peak centre.
    f0, eta, sigma, gamma, bckg : tuple of (float, float), optional
        (value, 1σ error) of each `LorGau` parameter; defaults (1.0, 0.0),
        (0.5, 0.0), (2.0, 0.0), (2.0, 0.0), (0.0, 0.0).

    Returns
    -------
    value : ndarray of float64
        f(x), same shape as `x`.
    error : ndarray of float64
        σ_f(x), same shape as `x`.

    Notes
    -----
    Bug fixes vs legacy (audit 2026-11, C12): legacy used the area-normalised
    `Gaussian_error` (peak height 1/(σ√2π)) while `LorGau` uses a unit-height
    Gaussian, so the returned value differed from `LorGau`; the η, σ and γ error terms also
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
    Evaluate a Lorentzian line shape of height A and FWHM gamma.

        L(x) = A·(γ/2)² / ((x - x0)² + (γ/2)²) + bckg

    L(x0) = A + bckg; L(x0 ± γ/2) = A/2 + bckg. Area of the peak: A·π·γ/2.

    Parameters
    ----------
    x : array_like
        Abscissas (units of x), any shape.
    A : float
        Peak height above background.
    x0 : float
        Peak position.
    gamma : float
        Full width at half maximum γ.
    bckg : float
        Constant background.

    Returns
    -------
    ndarray of float64, same shape as `x`
        L(x).

    Notes
    -----
    Legacy behaviour, unchanged. See `Lorentzian_error` for the propagated
    error.
    """
    q = np.asarray(x, dtype=np.float64)
    lor = A * gamma * gamma / 4.0 / ((q - x0) * (q - x0) + gamma * gamma / 4.0) + bckg
    return lor


def Gaussian(
    x: ArrayLike, A: float, x0: float, sigma: float, bckg: float
) -> NDArray[np.float64]:
    """
    Evaluate an area-normalised Gaussian line shape plus background.

        G(x) = A/(σ√(2π)) · exp(-(x - x0)²/(2σ²)) + bckg

    A is the area of the peak (not its height; the height is A/(σ√(2π))).

    Parameters
    ----------
    x : array_like
        Abscissas (units of x), any shape.
    A : float
        Area of the peak (units of y × units of x).
    x0 : float
        Peak position.
    sigma : float
        Standard deviation σ; FWHM = 2√(2 ln 2)·σ ≈ 2.3548σ.
    bckg : float
        Constant background.

    Returns
    -------
    ndarray of float64, same shape as `x`
        G(x).

    Notes
    -----
    Legacy behaviour, unchanged. See `Gaussian_error` for the propagated
    error.
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
    Evaluate an asymmetric Gaussian (exponential asymmetry), without background.

        f(x) = A/(σ√(2π)) · exp(-(x - x0)²/(2σ²)) · exp(-asym·(x - x0))

    Used for the asymmetric Ni reflections of the calibration.

    Parameters
    ----------
    x : array_like
        Abscissas (units of x), any shape.
    A : float
        Area of the peak when ``asym = 0``.
    x0 : float
        Centre of the Gaussian factor.
    sigma : float
        Standard deviation σ of the Gaussian factor.
    asym : float, optional
        Asymmetry parameter (units of 1/x), default 0 (symmetric Gaussian).
        asym > 0 enhances the low-x side.

    Returns
    -------
    ndarray of float64, same shape as `x`
        f(x).

    Notes
    -----
    Legacy behaviour, unchanged. Completing the square, f is a Gaussian of
    the same σ centred at x0 - asym·σ², multiplied by exp(asym²σ²/2): for
    asym != 0 the maximum is at x0 - asym·σ² (not at x0) and the area is
    A·exp(asym²σ²/2) (not A). There is no background term.
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
