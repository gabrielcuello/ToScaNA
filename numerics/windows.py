"""
Window (modification) functions for the sine Fourier transform.

Place in the processing chain: used by `toscana.numerics.fourier.sineFT`
(``window=1``) and `getSineFT` (``w=1``) to multiply F(Q) = Q[S(Q) - 1]
before the transform S(Q) -> G(r).

Public functions
----------------
Lorch(q, qmax=23.5)
    Standard Lorch window at one point (float).
LorchArray(q, qmax)
    The same, vectorised over an array.

Only the standard Lorch function is provided:

    M(Q) = sin(πQ/Qmax) / (πQ/Qmax)   for 0 <= Q < Qmax,  0 otherwise,

with M(0) = 1, M(Qmax/2) = 2/π, M(Qmax) = 0. Q and Qmax in Å⁻¹ (or any
consistent units: only the ratio Q/Qmax matters). The plain truncation
(no window) is `toscana.numerics.line_shapes.step`.

Why M(0) = 1 conserves coordination numbers: M(Q) is the Fourier transform
of a box of half-width Δ = π/Qmax, so multiplying F(Q) by M(Q) is exactly
equivalent to replacing the (truncated) G(r) by its running average over
[r - Δ, r + Δ] (Lorch's original construction, 1969). A running average with
a kernel of unit integral (M(0) = 1) conserves the area of G(r); integrals
between finite limits [r1, r2] change only by edge terms of order
(Δ²/6)[G'(r2) - G'(r1)] (and (Δ²/3)[G(r2) - G(r1)] for N(r), which
integrates r·G), which vanish when the limits lie in flat regions (e.g. from
the g(r) = 0 region to a minimum). The box kernel has no negative lobes,
unlike the sin(Qmax r)/r kernel of plain truncation.

Changes from the legacy code (audit 2026-11, C1, P5): legacy also had a
"normalised" Lorch (`LorchN`), rescaled so that ∫_0^Qmax M dQ = Qmax, i.e.
M(0) = π/Si(π) ≈ 1.70. It inflated all real-space integrals by that factor
(g(r) ≈ -0.7 at low r) and has been removed; `LorchArray` replaces it.
`Lorch` with Qmax <= 0 returned -999 (used silently as a window value); it now
raises ValueError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def Lorch(q: float, qmax: float = 23.5) -> float:  # noqa: N802 (legacy API)
    """
    Evaluate the standard Lorch window at one point (M(0) = 1).

    M(q) = sin(πq/qmax)/(πq/qmax) for 0 < q < qmax; M(0) = 1; M(q) = 0 for
    q < 0 or q >= qmax.

    Parameters
    ----------
    q : float
        Momentum transfer Q (Å⁻¹).
    qmax : float, optional
        Cut-off Qmax (Å⁻¹), default 23.5. Must be > 0.

    Returns
    -------
    float
        The window value, in [0, 1] for 0 <= q < qmax (0 elsewhere).

    Raises
    ------
    ValueError
        If ``qmax <= 0``.

    Notes
    -----
    q = 0 is handled explicitly (the 0/0 limit). Used point by point by
    `sineFT` with ``window=1``. Same values as `LorchArray` (up to rounding:
    `LorchArray` uses `numpy.sinc`), except that a NaN q returns NaN here and
    0 in `LorchArray`.

    Changes from the legacy code (audit 2026-11, P5): for qmax <= 0 legacy
    printed an error and returned -999, which was then used silently as a
    window value; it now raises.
    """
    if qmax <= 0:
        # Bug fix vs legacy: legacy printed an error and returned -999, which
        # was then used silently as a window value.
        raise ValueError("Non positive value for qmax in Lorch function.")

    if (q < 0.0) or (q >= qmax):
        return 0.0
    if q == 0.0:
        return 1.0

    a = q * np.pi / qmax
    return float(np.sin(a) / a)


def LorchArray(q: ArrayLike, qmax: float) -> NDArray[np.float64]:  # noqa: N802
    """
    Evaluate the standard Lorch window over an array (M(0) = 1).

    Same values as `Lorch`, vectorised: M(Q) = sin(πQ/qmax)/(πQ/qmax) for
    0 <= Q < qmax (M(0) = 1), and 0 outside 0 <= Q < qmax.

    Parameters
    ----------
    q : array_like
        Momentum transfer values Q (Å⁻¹), any shape (a scalar gives a 0-d
        array).
    qmax : float
        Cut-off Qmax (Å⁻¹), no default. Must be > 0.

    Returns
    -------
    ndarray of float64, same shape as `q`
        The window values.

    Raises
    ------
    ValueError
        If ``qmax <= 0``.

    Notes
    -----
    Computed as ``numpy.sinc(Q/qmax)`` (numpy's sinc(x) = sin(πx)/(πx),
    sinc(0) = 1). NaN in `q` gives 0 (the ``inside`` test is False). Used by
    `getSineFT` with ``w=1`` (there with qmax = vmax).

    New in the audit 2026-11 (C1): replaces the removed, rescaled `LorchN`
    (M(0) ≈ 1.70).
    """
    if qmax <= 0:
        raise ValueError("Non positive value for qmax in Lorch function.")
    qa = np.asarray(q, dtype=np.float64)
    inside = (qa >= 0.0) & (qa < qmax)
    # np.sinc(x) = sin(πx)/(πx), with sinc(0) = 1
    return np.where(inside, np.sinc(qa / qmax), 0.0).astype(np.float64)
