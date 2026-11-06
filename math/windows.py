"""
Window (modification) functions for the sine Fourier transform.

Only the standard Lorch function is provided:

    M(Q) = sin(πQ/Qmax) / (πQ/Qmax)   for 0 <= Q < Qmax,  0 otherwise,

with M(0) = 1. In r-space it replaces G(r) by its running average over
[r - Δ, r + Δ], Δ = π/Qmax, which conserves the area of G(r) and hence
coordination numbers integrated between flat regions (e.g. from the g(r) = 0
region to a minimum).

Legacy also had a "normalised" Lorch (`LorchN`), rescaled so that
∫_0^Qmax M dQ = Qmax, i.e. M(0) = π/Si(π) ≈ 1.70. It inflated all real-space
integrals by that factor (g(r) ≈ -0.7 at low r) and has been removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def Lorch(q: float, qmax: float = 23.5) -> float:  # noqa: N802 (legacy API)
    """
    Evaluate the standard Lorch window at one point (M(0) = 1).
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

    Same values as `Lorch`, vectorised: 0 outside 0 <= Q < qmax.
    """
    if qmax <= 0:
        raise ValueError("Non positive value for qmax in Lorch function.")
    qa = np.asarray(q, dtype=np.float64)
    inside = (qa >= 0.0) & (qa < qmax)
    # np.sinc(x) = sin(πx)/(πx), with sinc(0) = 1
    return np.where(inside, np.sinc(qa / qmax), 0.0).astype(np.float64)
