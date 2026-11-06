"""
Scattering models for total scattering data analysis.

Models of the Q dependence of the self (single-atom) scattering and of the
vanadium normalisation signal, used when correcting and normalising D4 data.

Public functions
----------------
inelastic(x, A=51.0, lowQ=0.4, Q0=SIGMOID_Q0, dQ=SIGMOID_DQ)
    Inelasticity (Placzek-like) decrease of the self scattering, a logistic
    step from lowQ (bound value, Q -> 0) to lowQ·[A/(A+1)]² (free value).
self024(x, self0, q2, q4)
    Empirical self-scattering curve self0 + q2·Q² + q4·Q⁴.
vanaQdep(x, a0, a1, a2, A=51.0, lowQ=0.4, Q0=SIGMOID_Q0, dQ=SIGMOID_DQ)
    Vanadium Q dependence: polyQ2(Q, a0, a1, a2) × inelastic(Q, ...).

Conventions and units
---------------------
Q (the abscissa `x`) in 1/Å; Q0 and dQ in 1/Å; A = m/m_n dimensionless;
lowQ in the unit of the modelled quantity (cross section in barn, or
normalised intensity). The default transition (Q0, dQ) = (SIGMOID_Q0,
SIGMOID_DQ) = (7.0, 2.4) 1/Å comes from ``toscana.numerics.line_shapes`` and
is shared with ``sigmoidal`` and ``siglin`` (audit 2026-11, P21). The defaults
A = 51.0 and lowQ = 0.4 refer to vanadium.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toscana.numerics.line_shapes import SIGMOID_DQ, SIGMOID_Q0
from toscana.numerics.polynomials import polyQ2


def inelastic(
    x: ArrayLike,
    A: float = 51.0,
    lowQ: float = 0.4,
    Q0: float = SIGMOID_Q0,
    dQ: float = SIGMOID_DQ,
) -> NDArray[np.float64]:
    """
    Calculate the inelastic behaviour using a sigmoidal function.

    The inelasticity is modeled as:
        inelastic(x) = lowQ * [1 + (A/(1+A))² * delta] / (1 + delta)
    where delta = exp[(x - Q0) / dQ]

    Note that:
        - inelastic(0) ≈ lowQ
        - inelastic(∞) = lowQ * (A / (1+A))²
        i.e. sigma_free = sigma_bound * (A / (1+A))²

    This is the logistic step of ``toscana.numerics.line_shapes.sigmoidal``
    with highQ = lowQ·(A/(1+A))²; at x = Q0 it is (lowQ + highQ)/2. It
    describes the empirical decrease of the self scattering with Q caused by
    the recoil of the scattering nucleus (Placzek-like inelasticity effect).

    Parameters
    ----------
    x : array_like
        The range of abscissas, usually the Q-scale (momentum transfer), 1/Å.
    A : float, optional
        The mass number (mass/neutron mass). Default is 51.0 (Vanadium).
    lowQ : float, optional
        The limiting value of the function at Q -> 0 (bound value). Default
        is 0.4.
    Q0 : float, optional
        Position of the inflexion point of the sigmoidal function, 1/Å.
        Default is ``SIGMOID_Q0`` = 7.0.
    dQ : float, optional
        Width of the transition of the sigmoidal function, 1/Å. Default is
        ``SIGMOID_DQ`` = 2.4. Must be non-zero.

    Returns
    -------
    numpy.ndarray
        The values of the inelastic function corresponding to the given
        abscissas, float64, same shape as `x`.

    Notes
    -----
    Evaluated with ``np.exp`` directly (not ``expit`` as ``sigmoidal``):
    for (x - Q0)/dQ > ~709 delta overflows to inf and the result is NaN (with
    a ``RuntimeWarning``). This needs x > ~1700 1/Å with the defaults, far
    outside any measured range.

    Changes from the legacy code (audit 2026-11, P21): the defaults Q0 and dQ
    are now the shared constants ``SIGMOID_Q0`` and ``SIGMOID_DQ`` (7.0 and
    2.4 1/Å); Q0 = 7.0 was already the legacy default of `inelastic`.

    See Also
    --------
    vanaQdep : vanadium model using this function.
    toscana.physics.properties.getFreeXS : the free/bound factor [A/(A+1)]².

    Examples
    --------
    >>> [round(float(v), 6) for v in inelastic([0.0, 7.0, 50.0])]
    [0.399218, 0.392382, 0.384763]
    """
    q = np.asarray(x, dtype=np.float64)
    delta = np.exp((q - Q0) / dQ)
    ratio = (A / (1.0 + A)) ** 2
    return lowQ * (1.0 + ratio * delta) / (1.0 + delta)


def self024(x: ArrayLike, self0: float, q2: float, q4: float) -> NDArray[np.float64]:
    """
    Curve fitting function for self-scattering: self0 + q2*x² + q4*x⁴.

    Evaluated as self0 + (q2 + q4·x²)·x². An even polynomial in Q, used as an
    empirical description of the self scattering to be fitted to the data.

    Parameters
    ----------
    x : array_like
        The range of abscissas (normally Q in 1/Å).
    self0 : float
        Constant term (Q=0).
    q2 : float
        Coefficient for the x² term (unit of the data × Å²).
    q4 : float
        Coefficient for the x⁴ term (unit of the data × Å⁴).

    Returns
    -------
    numpy.ndarray
        The evaluated function, float64, same shape as `x`.

    Examples
    --------
    >>> [float(v) for v in self024([0.0, 1.0, 2.0], 1.0, 0.5, 0.25)]
    [1.0, 1.75, 7.0]
    """
    q = np.asarray(x, dtype=np.float64)
    q2_val = q**2
    return self0 + (q2 + q4 * q2_val) * q2_val


def vanaQdep(
    x: ArrayLike,
    a0: float,
    a1: float,
    a2: float,
    A: float = 51.0,
    lowQ: float = 0.4,
    Q0: float = SIGMOID_Q0,
    dQ: float = SIGMOID_DQ,
) -> NDArray[np.float64]:
    """
    Function describing the general behaviour of incoherent Vanadium signal.

    It combines a 2nd-degree polynomial (instrument resolution effects)
    with a sigmoidal function (inelasticity effect).

    vanadium(Q) = polyQ2(Q, a0, a1, a2) * inelastic(Q, A, lowQ, Q0, dQ)

    with polyQ2(Q, a0, a1, a2) = a0 + a1·Q + a2·Q² (resolution, attenuation
    and other slow trends) and `inelastic` the bound-to-free logistic step.

    Parameters
    ----------
    x : array_like
        The range of abscissas (Q-scale), 1/Å.
    a0, a1, a2 : float
        Polynomial coefficients for polyQ2 (constant, linear and quadratic
        coefficients in Q). Note that a0 and lowQ are degenerate as overall
        scale factors: in a fit only their product is determined.
    A : float, optional
        Mass number (mass ratio m/m_n) for inelasticity. Default is 51.0
        (vanadium).
    lowQ : float, optional
        Limiting value of `inelastic` at Q -> 0. Default is 0.4.
    Q0 : float, optional
        Inflexion point, 1/Å. Default is ``SIGMOID_Q0`` = 7.0 (legacy default
        here was 7.4; unified with `inelastic`).
    dQ : float, optional
        Transition width, 1/Å. Default is ``SIGMOID_DQ`` = 2.4.

    Returns
    -------
    numpy.ndarray
        The evaluated Vanadium Q-dependence, float64, same shape as `x`.

    Notes
    -----
    Changes from the legacy code (audit 2026-11, P21): the default Q0 was
    7.4 1/Å here and 7.0 1/Å in `inelastic`; both now use ``SIGMOID_Q0`` =
    7.0 (effect on vanaQdep <= 0.16 %).

    See Also
    --------
    inelastic : the inelasticity factor.
    toscana.numerics.polynomials.polyQ2 : the polynomial factor.
    """
    q = np.asarray(x, dtype=np.float64)
    polynomial = polyQ2(q, a0, a1, a2)
    sigmoidal = inelastic(q, A=A, lowQ=lowQ, Q0=Q0, dQ=dQ)
    return sigmoidal * polynomial
