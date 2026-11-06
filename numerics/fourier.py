from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate, special  # type: ignore[import-untyped]

from .line_shapes import step
from .windows import Lorch, LorchArray

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

        stf(u) = 2/π · c · ∫_0^vmax (y(v) - s) · v · sin(v·u) · W(v) dv

    `w` selects the window W: 0 = none (truncation at vmax),
    1 = standard Lorch with W(0) = 1 (see `toscana.numerics.windows`).

    Notes
    -----
    Divergences from legacy: with `w=0` legacy ignored `vmax` and always
    integrated over the full range; with `w=1` it used the normalised Lorch
    (W(0) ≈ 1.70), which has been removed. Other values of `w` raise.
    """
    if w not in (0, 1):
        raise ValueError("getSineFT: w must be 0 (no window) or 1 (Lorch).")
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
    keep = vv <= vmax
    vv, yy = vv[keep], yy[keep]
    if w == 1:
        win = LorchArray(vv, vmax)
    else:
        win = np.ones(len(vv), dtype=np.float64)

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
    with_errors: bool = False,
) -> NDArray[np.float64]:
    """
    Sine Fourier transform of the structure factor S(Q) (columns Q, S[, σS]).

    Returns an (N_r, 6) array with columns r, G(r), g(r), RDF(r), T(r), N(r):

        G(r)   = 2/π · constant · ∫_0^qmax Q [S(Q) - selfsca] M(Q) sin(Qr) dQ
        g(r)   = 1 + G(r) / (4πρr)          (0 at r <= 0)
        RDF(r) = 4πρ r² g(r),   T(r) = 4πρ r g(r),   N(r) = ∫_{r0}^{r} RDF dr'

    `window` selects M(Q):
      0 = no window (truncation at qmax);
      1 = standard Lorch, M(0) = 1. Conserves coordination numbers integrated
          between flat regions of g(r) (e.g. from g = 0 to a minimum).

    Notes
    -----
    Divergence from legacy: `window=1` was a normalised Lorch rescaled to
    ∫ M dQ = qmax (M(0) ≈ 1.70), which inflated G(r) and coordination
    numbers and drove g(r) to ≈ -0.7 at low r. It has been removed. Values
    other than 0 and 1 now raise (legacy treated any non-zero as Lorch).

    Error bars (`with_errors=True`)
    -------------------------------
    Requires a third column σS in `StrFactor`. Returns an (N_r, 11) array:

        r, G, σG, g, σg, RDF, σRDF, T, σT, N, σN

    The transform is linear in S(Q), so the 1σ statistical errors are
    propagated exactly, assuming uncorrelated S(Q) points:
        σG(r_j)² = Σ_k W_jk² σS_k²,  W_jk = 2/π·constant·Q_k M(Q_k) sin(Q_k r_j) w_k
    (w_k: the Simpson weights used for G). Then σT = σG, σRDF = r·σG and
    σg = σG/(4πρr) (0 at r <= 0). σN is propagated from S(Q) directly with
    the same running Simpson rule as N(r), so the strong correlations between
    neighbouring G(r) points are accounted for. Systematic errors (normalisation,
    density, self-scattering, truncation) are not included.
    """
    if window not in (0, 1):
        raise ValueError("sineFT: window must be 0 (no window) or 1 (Lorch).")
    if with_errors and StrFactor.shape[1] < 3:
        raise ValueError("sineFT: with_errors=True needs a third column with σS(Q).")
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
    if not with_errors:
        return fou

    sig_g_func, sig_run = _sine_ft_errors(
        np.asarray(q, dtype=np.float64),
        np.asarray(StrFactor[:, 2], dtype=np.float64),
        np.asarray(win, dtype=np.float64),
        np.asarray(nr, dtype=np.float64),
        constant,
    )
    r_col = fou[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_pdf = np.where(r_col > 0, sig_g_func / (4.0 * np.pi * density * r_col), 0.0)
    return np.column_stack(
        [
            fou[:, 0],
            fou[:, 1], sig_g_func,
            fou[:, 2], sig_pdf,
            fou[:, 3], r_col * sig_g_func,
            fou[:, 4], sig_g_func,
            fou[:, 5], sig_run,
        ]
    )


def _simpson_weights(x: NDArray[np.float64], chunk: int = 512) -> NDArray[np.float64]:
    """Weights w such that integrate.simpson(f, x=x) == Σ w·f (same rule)."""
    n = len(x)
    weights = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        unit = np.zeros((stop - start, n), dtype=np.float64)
        unit[np.arange(stop - start), np.arange(start, stop)] = 1.0
        weights[start:stop] = integrate.simpson(unit, x=x, axis=1)
    return weights


def _sine_ft_errors(
    q: NDArray[np.float64],
    sig_s: NDArray[np.float64],
    win: NDArray[np.float64],
    nr: NDArray[np.float64],
    constant: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1σ errors of G(r) and of the running integral N(r) (see `sineFT`)."""
    base = 2.0 / np.pi * constant * q * win * _simpson_weights(q)
    var_s = sig_s**2
    # G = W @ (S - selfsca),  W[j, k] = base_k sin(q_k r_j)
    w_g = np.sin(np.outer(nr, q)) * base
    sig_g = np.sqrt((w_g**2) @ var_s)
    # N(r_i) = Σ_j C_ij r_j G_j + const  ->  linear in S through C·diag(r)·W,
    # with C the exact running-Simpson matrix used for N(r) itself.
    w_run = _running_simpson_matrix(nr) @ (nr[:, None] * w_g)
    sig_run = np.sqrt((w_run**2) @ var_s)
    return sig_g, sig_run


def _running_simpson_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Matrix C with (C @ f)[i] == integrate.simpson(f[:i+1], x=x[:i+1]) (0 for i=0).

    Built incrementally in O(N²): odd-length prefixes are composite Simpson
    (one 3-point panel added at a time); an even-length prefix adds SciPy's
    correction for the last interval, which depends only on the last 4 points.
    The result is checked against SciPy and rebuilt the slow, direct way if a
    SciPy version used a different rule.
    """
    n = len(x)
    c = np.zeros((n, n), dtype=np.float64)
    if n > 1:
        c[1, :2] = integrate.simpson(np.eye(2), x=x[:2], axis=1)
    odd = np.zeros(n, dtype=np.float64)
    for m in range(3, n + 1):  # prefix length m -> row m - 1
        if m % 2 == 1:
            odd[m - 3 : m] += integrate.simpson(np.eye(3), x=x[m - 3 : m], axis=1)
            c[m - 1, :m] = odd[:m]
        else:
            x4 = x[m - 4 : m]
            delta = integrate.simpson(np.eye(4), x=x4, axis=1)
            delta[:3] -= integrate.simpson(np.eye(3), x=x4[:3], axis=1)
            c[m - 1, :m] = odd[:m]
            c[m - 1, m - 4 : m] += delta

    rows = sorted({i for i in (1, 2, 3, 4, n // 2, n - 2, n - 1) if 1 <= i < n})
    if not all(
        np.allclose(c[i, : i + 1], integrate.simpson(np.eye(i + 1), x=x[: i + 1], axis=1),
                    rtol=1e-12, atol=1e-14)
        for i in rows
    ):
        for i in range(1, n):
            c[i, : i + 1] = integrate.simpson(np.eye(i + 1), x=x[: i + 1], axis=1)
    return c


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
