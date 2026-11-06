from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate, special  # type: ignore[import-untyped]

from .line_shapes import step
from .windows import Lorch, LorchN

if TYPE_CHECKING:
    from numpy.typing import NDArray


def getSineIntegral(x: float) -> float:  # noqa: N802 (legacy API)
    """
    Evaluate the sine integral Si(x) = ∫_0^x sin(t)/t dt.

    Notes
    -----
    Bug fix vs legacy: legacy integrated numerically on a grid ending at
    x(1 - 1e-4) (relative error ~1e-4) and returned NaN for x = 0. The exact
    `scipy.special.sici` is used instead.
    """
    return float(special.sici(x)[0])


def getSineFT(  # noqa: N802 (legacy API)
    vvec: NDArray[np.float64],
    yvec: NDArray[np.float64],
    uvec: NDArray[np.float64],
    vmax: float = 0,
    c: float = 1.0,
    s: float = 1.0,
    w: int = 0,
) -> NDArray[np.float64] | None:
    """
    Calculate a sine Fourier transform:

        stf(u) = 2/π · c · ∫ (y(v) - s) · v · sin(v·u) · W(v) dv

    Notes
    -----
    Bug fix vs legacy: with `w=0` (no window) legacy ignored `vmax` and always
    integrated over the full range. The integral now stops at `vmax`; with the
    default `vmax` (full range) results are unchanged.
    """
    nbr_v = len(vvec)
    nbr_y = len(yvec)
    if nbr_v != nbr_y:
        print("ERROR: abcissa and ordinate do not have the same dimension")
        print("       in the Fourier Transform function (getSineFT).")
        return None

    if (vmax <= 0) or (vmax > vvec[-1]):
        vmax = float(vvec[-1])

    vv = np.asarray(vvec, dtype=np.float64)
    yy = np.asarray(yvec, dtype=np.float64)
    win: NDArray[np.float64] = np.ones(nbr_v, dtype=np.float64)
    if w == 1:
        win = LorchN(vv, vmax)
    else:
        keep = vv <= vmax
        vv, yy, win = vv[keep], yy[keep], win[keep]

    stf: NDArray[np.float64] = np.zeros(len(uvec), dtype=np.float64)
    for i in range(len(uvec)):
        integ = (yy - s) * vv * np.sin(vv * uvec[i]) * win
        stf[i] = 2.0 / np.pi * c * integrate.simpson(integ, x=vv)
    return stf


def sineFT(  # noqa: N802 (legacy API)
    StrFactor: NDArray[np.float64],
    nr: NDArray[np.float64],
    qmax: float = 23.5,
    density: float = 0.1,
    constant: float = 1.0,
    selfsca: float = 1.0,
    window: int = 0,
) -> NDArray[np.float64]:
    """
    Perform the sine Fourier transform of the structure factor (legacy behavior).
    """
    soq = StrFactor[:, 1]
    q = StrFactor[:, 0]

    pcf: list[float] = []
    pdf: list[float] = []
    rdf: list[float] = []
    tor: list[float] = []
    run: list[float] = []

    win: list[float] = []
    if window == 0:
        for qu in q:
            win.append(float(step(float(qu), qmax)))
    else:
        for qu in q:
            win.append(float(Lorch(float(qu), qmax)))
        integral_lorch = integrate.simpson(win, q)
        for i in range(len(win)):
            win[i] = win[i] * qmax / float(integral_lorch)

    for r in nr:
        integ = (soq - selfsca) * q * np.sin(q * r) * win
        result = 2.0 / np.pi * constant * integrate.simpson(integ, q)
        pcf.append(float(result))
        if r <= 0:
            pdf.append(0.0)
        else:
            pdf.append(float(result / 4.0 / np.pi / float(r) / density + 1.0))
        rdf.append(float(result * float(r) + 4.0 * np.pi * density * float(r) ** 2))
        tor.append(float(result + 4.0 * np.pi * density * float(r)))

    # Running coordination number N(r) = ∫_{r0}^{r} RDF dr'.
    # Bug fix vs legacy: the slice end was computed as 1 + int(r/Δr), which
    # assumes nr[0] = 0 and exact float division; even on a grid from 0 in
    # steps of 0.02 Å, rounding (e.g. 0.58/0.02 = 28.999...) stopped the
    # integral one point short at 16 of 400 points. The loop index is used now.
    rdf_arr0 = np.asarray(rdf, dtype=np.float64)
    nr_arr0 = np.asarray(nr, dtype=np.float64)
    run.append(0.0)
    for i in range(1, len(nr_arr0)):
        run.append(float(integrate.simpson(rdf_arr0[: i + 1], x=nr_arr0[: i + 1])))

    rrr = np.array(nr, dtype=np.float64)
    pcf_arr = np.array(pcf, dtype=np.float64)
    pdf_arr = np.array(pdf, dtype=np.float64)
    rdf_arr = np.array(rdf, dtype=np.float64)
    tor_arr = np.array(tor, dtype=np.float64)
    run_arr = np.array(run, dtype=np.float64)

    rrr = rrr.reshape(rrr.shape[0], 1)
    pcf_arr = pcf_arr.reshape(pcf_arr.shape[0], 1)
    pdf_arr = pdf_arr.reshape(pdf_arr.shape[0], 1)
    rdf_arr = rdf_arr.reshape(rdf_arr.shape[0], 1)
    tor_arr = tor_arr.reshape(tor_arr.shape[0], 1)
    run_arr = run_arr.reshape(run_arr.shape[0], 1)

    fou = np.concatenate((rrr, pcf_arr), axis=1)
    fou = np.concatenate((fou, pdf_arr), axis=1)
    fou = np.concatenate((fou, rdf_arr), axis=1)
    fou = np.concatenate((fou, tor_arr), axis=1)
    fou = np.concatenate((fou, run_arr), axis=1)
    return fou


def backFT(  # noqa: N802 (legacy API)
    qu: NDArray[np.float64],
    ar: NDArray[np.float64],
    pdf: NDArray[np.float64],
    density: float,
    cut: list[float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Back Fourier transform of a pair distribution function g(r).

    g(r) is first cut (set to 0) in the regions given by `cut`, transformed to
    S(Q) = 1 + (4πρ/Q) ∫ r [g(r) - 1] sin(Qr) dr, and transformed back to real
    space. Returns (S(Q), G(r), g(r), RDF(r), T(r), N(r)).

    `cut` conventions:
    - ``[-1]``: zero g(r) up to the last point where it is negative
      (nothing is cut if g(r) is never negative);
    - ``[r0, a1, b1, a2, b2, ...]`` (odd length): zero r < r0 and every open
      interval (a_k, b_k).

    Notes
    -----
    Bug fixes vs legacy: Q = 0 and r = 0 are handled with their analytic
    limits (legacy produced NaN/inf); the running integral includes the
    current point (legacy integrated up to r_{i-1}), matching `sineFT`;
    `[-1]` with no negative value no longer zeroes the first two points;
    any odd-length `cut` is accepted and an even length raises ValueError
    (legacy silently ignored unsupported lengths).
    """
    ar = np.asarray(ar, dtype=np.float64)
    qu = np.asarray(qu, dtype=np.float64)
    pdf_cut = _apply_cut(ar, np.asarray(pdf, dtype=np.float64), cut)

    stf = getSineFT(ar, pdf_cut, qu, w=0)
    if stf is None:
        raise ValueError("backFT: getSineFT returned None (dimension mismatch).")

    soq = np.empty_like(qu)
    pos_q = qu > 0
    soq[pos_q] = 1 + 2.0 * np.pi**2 * density / qu[pos_q] * stf[pos_q]
    # Q -> 0 limit: sin(Qr)/Q -> r
    soq[~pos_q] = 1 + 4.0 * np.pi * density * float(
        integrate.simpson(ar**2 * (pdf_cut - 1.0), x=ar)
    )

    pcf = getSineFT(qu, soq, ar, w=0)
    if pcf is None:
        raise ValueError("backFT: getSineFT returned None (dimension mismatch).")

    # Same convention as sineFT: g(r) = 0 for r <= 0
    pdf_new = np.zeros_like(ar)
    pos_r = ar > 0
    pdf_new[pos_r] = 1 + pcf[pos_r] / 4.0 / np.pi / density / ar[pos_r]
    rdf = 4.0 * np.pi * density * ar * ar * pdf_new
    tor = 4.0 * np.pi * density * ar * pdf_new

    run = np.zeros(len(ar), dtype=np.float64)
    for i in range(1, len(ar)):
        run[i] = integrate.simpson(rdf[: i + 1], x=ar[: i + 1])

    return (
        np.asarray(soq, dtype=np.float64),
        np.asarray(pcf, dtype=np.float64),
        np.asarray(pdf_new, dtype=np.float64),
        np.asarray(rdf, dtype=np.float64),
        np.asarray(tor, dtype=np.float64),
        np.asarray(run, dtype=np.float64),
    )


def _apply_cut(
    ar: NDArray[np.float64], pdf: NDArray[np.float64], cut: list[float]
) -> NDArray[np.float64]:
    """Zero g(r) in the regions defined by `cut` (see `backFT`)."""
    out = pdf.copy()
    cut = [float(c) for c in cut]
    if len(cut) == 1 and cut[0] == -1:
        negative = np.nonzero(pdf < 0)[0]
        if negative.size:
            out[: negative[-1] + 1] = 0.0
        return out
    if len(cut) % 2 == 0:
        raise ValueError(
            "backFT: `cut` must be [-1] or [r0, a1, b1, a2, b2, ...] (odd length)."
        )
    mask = ar < cut[0]
    for lo, hi in zip(cut[1::2], cut[2::2]):
        mask |= (lo < ar) & (ar < hi)
    out[mask] = 0.0
    return out
