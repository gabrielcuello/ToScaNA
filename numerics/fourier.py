"""
Sine Fourier transforms between reciprocal space S(Q) and real space G(r), g(r).

Place in the processing chain
-----------------------------
raw numors -> corrections -> regrouping on 2θ / Q grids -> CORRECT -> S(Q)
-> **this module** -> G(r), g(r), RDF(r), T(r), N(r) (with optional exact
statistical error bars) -> output files (`saveRSCF`) and plots.
`backFT` goes the other way (g(r) -> S(Q) -> g(r)) to study the effect of
cutting unphysical features of g(r).

Public functions
----------------
sineFT(StrFactor, nr, qmax, density, constant, selfsca, window, with_errors)
    Main transform S(Q) -> r, G, g, RDF, T, N (and their 1σ errors).
getSineFT(vvec, yvec, uvec, vmax, c, s, w)
    Generic sine transform 2/π·c·∫ [y(v) - s] v sin(vu) W(v) dv; used by
    `backFT` in both directions.
backFT(qu, ar, pdf, density, cut)
    Cut g(r), transform to S(Q) and back; returns (S, G, g, RDF, T, N).
getSineIntegral(x)
    Sine integral Si(x) = ∫_0^x sin(t)/t dt (exact, `scipy.special.sici`).
fourier_with_errors(q, s, ds, r, rho, window, qmax)
    Same transform as `sineFT`, written as a matrix product, returning a
    `FourierSet` (values, exact errors and the Jacobians dG/dS, dg/dS).
back_fourier_iterations(q, ds, start, rho, cut, niter)
    `backFT` applied `niter` times from a `FourierSet`, with exact errors
    propagated through the iterations (the cut is a projection).
ft_matrix(q, r, window, qmax)
    The matrix W of G(r) = W (S - 1).
FourierSet
    Container of r, G, g, RDF, T, N, their errors and Jacobians.

Private helpers: `_simpson_weights` (quadrature weights of SciPy's Simpson
rule), `_sine_ft_errors` (exact error propagation), `_running_simpson_matrix`
(matrix of the running Simpson integral), `_apply_cut` (the `cut` argument of
`backFT`).

Conventions and units
---------------------
- Q in Å⁻¹, r in Å, atomic number density ρ in atoms/Å³.
- With F(Q) = Q[S(Q) - 1] and a window (modification function) M(Q) that
  vanishes for Q >= Qmax:

      G(r)   = (2/π) ∫_0^Qmax Q [S(Q) - 1] M(Q) sin(Qr) dQ = 4πρr [g(r) - 1]
      g(r)   = 1 + G(r)/(4πρr)          (g(0) ≡ 0 by convention)
      RDF(r) = 4πρr² g(r) = r G(r) + 4πρr²
      T(r)   = 4πρr g(r)  = G(r) + 4πρr  (no ⟨b⟩² factor)
      N(r)   = ∫_{r0}^{r} RDF(r') dr'    (r0 = first point of the r grid)

  and for the back-transform

      S(Q) = 1 + (4πρ/Q) ∫ r [g(r) - 1] sin(Qr) dr,
      S(0) = 1 + 4πρ ∫ r² [g(r) - 1] dr.

  Units: G in Å⁻², g dimensionless, RDF in atoms/Å, T in Å⁻², N in atoms.
- All integrals use Simpson's rule (`scipy.integrate.simpson`) directly on
  the given, possibly non-uniform, grids; no FFT is used.
- Windows: 0 = none (truncation at Qmax), 1 = standard Lorch
  M(Q) = sin(πQ/Qmax)/(πQ/Qmax), M(0) = 1 (see `toscana.numerics.windows`).
  The Lorch function is the Fourier transform of a box of half-width
  Δ = π/Qmax, so it replaces the truncated G(r) by its running average over
  [r - Δ, r + Δ]; with M(0) = 1 this conserves areas and hence coordination
  numbers integrated between flat regions of g(r).

Audit 2026-11: C1 (window), C17 (`backFT`), C18 (running N(r)), C20 (sine
integral), P4 (`vmax` in `getSineFT`); see Appendix A of
docs/ToScaNA_appendices.tex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import integrate, special  # type: ignore[import-untyped]

from .line_shapes import step
from .windows import Lorch, LorchArray

if TYPE_CHECKING:
    from numpy.typing import NDArray


def getSineIntegral(x: float) -> float:  # noqa: N802 (legacy API)
    """
    Evaluate the sine integral Si(x) = ∫_0^x sin(t)/t dt.

    Parameters
    ----------
    x : float
        Upper limit of the integral (dimensionless). Any real value; Si is odd,
        Si(0) = 0, Si(∞) = π/2.

    Returns
    -------
    float
        Si(x), e.g. Si(1) = 0.946083..., Si(π) = 1.851937...

    Notes
    -----
    Computed as ``scipy.special.sici(x)[0]`` (exact to machine precision).
    Si(π) appears in the discussion of the removed "normalised" Lorch window,
    whose value at Q = 0 was π/Si(π) ≈ 1.70.

    Changes from the legacy code (audit 2026-11, C20): legacy integrated
    numerically on a grid ending at x(1 - 1e-4) (relative error ~1e-4: Si(1)
    gave 0.945999 instead of 0.946083) and returned NaN for x = 0. The exact
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
    Calculate a generic sine Fourier transform on arbitrary grids.

    Computes, for every u in `uvec`,

        stf(u) = 2/π · c · ∫_{v_0}^{vmax} (y(v) - s) · v · sin(v·u) · W(v) dv

    where v_0 = vvec[0] (the integral starts at the first data point, which
    is normally 0 or close to it). `w` selects the window W: 0 = none
    (truncation at vmax), 1 = standard Lorch with W(0) = 1,
    W(v) = sin(πv/vmax)/(πv/vmax) (see `toscana.numerics.windows`).

    The same function serves both directions:

    - Q -> r: ``getSineFT(Q, S, r)`` gives G(r) = 2/π ∫ Q[S(Q)-1] sin(Qr) dQ;
    - r -> Q: ``getSineFT(r, g, Q)`` gives 2/π ∫ r[g(r)-1] sin(Qr) dr, i.e.
      [S(Q) - 1]·Q/(2π²ρ) (this is how `backFT` uses it).

    Parameters
    ----------
    vvec : ndarray, shape (N,)
        Integration variable (abscissa), increasing; e.g. Q in Å⁻¹ or r in Å.
        Need not be uniform.
    yvec : ndarray, shape (N,)
        Function values y(v) on `vvec` (e.g. S(Q) or g(r)).
    uvec : ndarray, shape (M,)
        Conjugate variable at which the transform is evaluated (e.g. r in Å or
        Q in Å⁻¹).
    vmax : float, optional
        Upper limit of the integral, in the units of `vvec`. Default 0: any
        value ``<= 0`` or ``> vvec[-1]`` is replaced by ``vvec[-1]`` (the end
        of the data). Also sets the width of the Lorch window when ``w=1``.
    c : float, optional
        Multiplicative constant (default 1.0).
    s : float, optional
        Value subtracted from y before the transform (default 1.0, i.e. the
        asymptotic value of a normalised S(Q) or g(r)).
    w : int, optional
        Window: 0 (default) = none, 1 = standard Lorch.

    Returns
    -------
    ndarray, shape (M,), or None
        stf(u) for every u in `uvec` (units: [y]·[v]² ·c). ``None`` if `vvec`
        and `yvec` have different lengths (after printing an error message).

    Raises
    ------
    ValueError
        If `w` is not 0 or 1 (checked before the length check).

    Notes
    -----
    - Only the points with ``vvec <= vmax`` are kept; the integral is the
      Simpson rule (`scipy.integrate.simpson`) over these points. With the
      Lorch window the point v = vmax (if present) has weight W = 0.
    - The integrand is evaluated in a Python loop over `uvec`
      (O(N·M) operations).
    - Side effect: prints two error lines when the lengths differ.

    Changes from the legacy code (audit 2026-11, P4 and C1): with ``w=0``
    legacy ignored `vmax` and always integrated over the full range (results
    unchanged when `vmax` is the default, the end of the data); with ``w=1``
    it used the normalised Lorch (W(0) ≈ 1.70), which has been removed. Any
    value of `w` other than 0 and 1 raises (legacy treated any non-zero value
    as Lorch).

    See Also
    --------
    sineFT : the main S(Q) -> real-space transform.
    backFT : uses this function in both directions.
    toscana.numerics.windows.LorchArray : the window for ``w=1``.
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
    Sine Fourier transform of the structure factor S(Q) to real space.

    Computes, on the r grid `nr`, the real-space functions

        G(r)   = 2/π · constant · ∫ Q [S(Q) - selfsca] M(Q) sin(Qr) dQ
        g(r)   = 1 + G(r) / (4πρr)               (set to 0 for r <= 0)
        RDF(r) = r·G(r) + 4πρr²                  (= 4πρr² g(r) for r > 0)
        T(r)   = G(r) + 4πρr                     (= 4πρr g(r) for r > 0)
        N(r)   = ∫_{r0}^{r} RDF(r') dr'          (r0 = nr[0], N(r0) = 0)

    with ρ = `density`. The Q integral runs over all rows of `StrFactor`
    (Simpson's rule on the given Q grid); the window M(Q) is zero for
    Q < 0 and Q >= qmax, so effectively only 0 <= Q < qmax contributes.

    Parameters
    ----------
    StrFactor : ndarray, shape (N_Q, 2) or (N_Q, >=3)
        Structure factor: column 0 = Q (Å⁻¹, increasing, any spacing),
        column 1 = S(Q), column 2 (needed only with ``with_errors=True``) =
        1σ error of S(Q). Further columns are ignored.
    nr : ndarray, shape (N_r,)
        r grid in Å (any spacing, normally increasing from 0 or a small
        value). N(r) is integrated from ``nr[0]``.
    qmax : float, optional
        Maximum Q (Å⁻¹) of the window, default 23.5. Must be > 0 (otherwise
        `step`/`Lorch` raise ValueError).
    density : float, optional
        Atomic number density ρ in atoms/Å³, default 0.1. Used for g, RDF,
        T, N (not for G).
    constant : float, optional
        Multiplicative factor applied to G(r) (default 1.0), e.g. a
        normalisation correction.
    selfsca : float, optional
        Value subtracted from S(Q) before the transform (default 1.0, the
        self-scattering level of a normalised S(Q)).
    window : int, optional
        Modification function M(Q): 0 (default) = none (step: 1 for
        0 <= Q < qmax, else 0); 1 = standard Lorch
        M(Q) = sin(πQ/qmax)/(πQ/qmax), M(0) = 1, 0 for Q >= qmax.
    with_errors : bool, optional
        If True, also return the propagated 1σ statistical errors
        (default False).

    Returns
    -------
    ndarray, shape (N_r, 6)
        If ``with_errors=False``: columns r (Å), G(r) (Å⁻²), g(r)
        (dimensionless), RDF(r) (atoms/Å), T(r) (Å⁻²), N(r) (atoms).
    ndarray, shape (N_r, 11)
        If ``with_errors=True``: columns
        r, G, σG, g, σg, RDF, σRDF, T, σT, N, σN (each error beside its
        function, same units).

    Raises
    ------
    ValueError
        If `window` is not 0 or 1; if ``with_errors=True`` and `StrFactor`
        has fewer than 3 columns; if ``qmax <= 0`` (raised by `step` or
        `Lorch`).

    Notes
    -----
    Numerical method. The window is evaluated point by point with the
    scalar functions `step` (window=0) or `Lorch` (window=1). For each r, the
    integrand (S - selfsca)·Q·sin(Qr)·M(Q) is integrated with
    ``scipy.integrate.simpson`` over the full Q grid (points with M = 0
    contribute zero but still take part in the Simpson panels). No FFT is
    used. NaN values in S(Q) propagate to every G(r) (no masking).

    Edge cases. At r = 0: G(0) = 0 (sin 0 = 0), g(0) is set to 0 (the
    analytic limit 1 + G'(0)/(4πρ) is not used), RDF(0) = T(0) = 0. Points
    with r < 0 also get g = 0. N(nr[0]) = 0 whatever nr[0] is; N(r_i) for
    i >= 1 is ``simpson(RDF[:i+1], x=nr[:i+1])`` (on an even number of
    points SciPy's end correction is used).

    Window and coordination numbers (audit C1). The Lorch function is the
    Fourier transform of a box of half-width Δ = π/qmax, so multiplying F(Q)
    by M(Q) is exactly equivalent to replacing the (truncated) G(r) by its
    running average over [r - Δ, r + Δ] (Lorch 1969). Because M(0) = 1 the
    kernel has unit integral: the total area of G(r) is conserved; integrals
    between finite limits change only by edge terms ~ (Δ²/6)[G'(r2) - G'(r1)]
    (plus (Δ²/3)[G(r2) - G(r1)] for N, which integrates r·G), negligible when
    the limits lie in flat regions; and the kernel has no negative lobes,
    unlike the sin(qmax·r)/r kernel of plain truncation. Practical rule:
    integrate N from a point where g(r) = 0 to a minimum of g(r).

    Error bars (``with_errors=True``). The transform is linear in S(Q), so
    the 1σ statistical errors are propagated exactly (no Monte Carlo),
    assuming uncorrelated S(Q) points:

        σG(r_j)² = Σ_k W_jk² σS_k²,
        W_jk = 2/π · constant · Q_k M(Q_k) sin(Q_k r_j) w_k

    (w_k: the Simpson weights used for G, obtained from SciPy itself by
    `_simpson_weights`). Then σT = σG, σRDF = r·σG (as computed, i.e.
    negative for r < 0) and σg = σG/(4πρr) (0 for r <= 0). σN is propagated
    from S(Q) directly through the matrix C·diag(r)·W, where C is the exact
    running Simpson matrix used for N(r) itself (`_running_simpson_matrix`),
    so the strong correlations between neighbouring G(r) points are
    accounted for. `selfsca` and `density` do not enter σG or σN. Validated
    against 1500 Monte Carlo perturbations (ratio analytic/MC ≈ 1.00 ± 0.05).
    Limits: statistical errors only (normalisation, density, self-scattering
    and truncation errors are not included); the fractional rebinning
    correlates neighbouring S(Q) points slightly, so the errors may be
    somewhat underestimated; errors at different r are strongly correlated.

    Changes from the legacy code (audit 2026-11):

    - C1: ``window=1`` was a "normalised" Lorch rescaled to ∫ M dQ = qmax
      (M(0) = π/Si(π) ≈ 1.70), which multiplied G(r) and coordination
      numbers by ≈ 1.70 and drove g(r) to ≈ -0.7 at low r. It has been
      removed. Values other than 0 and 1 now raise (legacy treated any
      non-zero value as Lorch).
    - C18: the end of the running integral for N(r) was computed as
      ``1 + int(r/Δr)``, which assumes nr[0] = 0 and exact float division
      (it stopped one point short at 16 of 400 points on a 0.02 Å grid, and
      was wrong everywhere on a grid not starting at 0). The loop index is
      used now; N(nr[0]) = 0.
    - New: the ``with_errors`` option (11-column output).

    Usage::

        fou = sineFT(SQ, r, qmax=23.5, density=rho, window=1, with_errors=True)
        saveRSCF("sample.rscf", fou, ["Sample: ...", "Qmax = 23.5, Lorch"])

    See Also
    --------
    getSineFT : generic sine transform.
    backFT : back-transform g(r) -> S(Q) -> g(r).
    toscana.numerics.windows.Lorch : the Lorch window.
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
    """
    Quadrature weights of SciPy's Simpson rule on the grid `x`.

    Returns w such that ``integrate.simpson(f, x=x) == Σ_k w_k f_k`` for any
    f (up to rounding), i.e. exactly the rule used for the values.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Abscissas (any spacing), e.g. Q in Å⁻¹.
    chunk : int, optional
        Number of unit vectors integrated per SciPy call (default 512);
        only limits memory (a (chunk, N) array), not the result.

    Returns
    -------
    ndarray, shape (N,)
        The weights w_k, in the units of `x`.

    Notes
    -----
    Obtained by integrating the unit vectors e_k with
    ``integrate.simpson(..., x=x, axis=1)``; since the rule is linear, the
    integral of e_k is w_k. This makes the error propagation independent of
    SciPy's internal formula (including its end correction for an even
    number of points).
    """
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
    """
    Exact 1σ errors of G(r) and of the running integral N(r) (see `sineFT`).

    Builds the linear map W from S(Q) to G(r),

        W_jk = 2/π · constant · Q_k M(Q_k) sin(Q_k r_j) w_k,

    with w_k the Simpson weights of the Q grid, and returns

        σG_j² = Σ_k W_jk² σS_k²,
        σN_i² = Σ_k [(C · diag(r) · W)_ik]² σS_k²,

    where C is the running Simpson matrix (`_running_simpson_matrix`), since
    N(r_i) = Σ_j C_ij r_j G(r_j) + Σ_j C_ij 4πρ r_j² (the second term has no
    error).

    Parameters
    ----------
    q : ndarray, shape (N_Q,)
        Q grid (Å⁻¹), the full grid used for G (including points with M = 0).
    sig_s : ndarray, shape (N_Q,)
        1σ errors of S(Q), assumed uncorrelated.
    win : ndarray, shape (N_Q,)
        Window values M(Q_k) (step or Lorch), as used for G.
    nr : ndarray, shape (N_r,)
        r grid (Å).
    constant : float
        The multiplicative constant of `sineFT`.

    Returns
    -------
    sig_g : ndarray, shape (N_r,)
        1σ error of G(r) (Å⁻²).
    sig_run : ndarray, shape (N_r,)
        1σ error of N(r) (atoms); 0 at r = nr[0].

    Notes
    -----
    Memory and time: W and C·diag(r)·W are dense (N_r, N_Q) arrays and C is
    (N_r, N_r). NaN in `sig_s` propagates to every output. Neither
    `selfsca` nor `density` enters these errors. A first version used
    trapezoidal weights for C and gave σN up to 11 % wrong (Monte Carlo
    test); the exact running Simpson matrix is used now.
    """
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
    Matrix C of the running Simpson integral on the grid `x`.

    (C @ f)[i] == integrate.simpson(f[:i+1], x=x[:i+1]) for i >= 1, and
    row 0 is zero (the running integral starts at 0 at x[0]). This is exactly
    the rule used for N(r) in `sineFT`.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Abscissas (any spacing), e.g. r in Å.

    Returns
    -------
    ndarray, shape (N, N)
        Lower-triangular matrix C (units of `x`).

    Notes
    -----
    Built incrementally in O(N²) (≈ 0.1 s for 2000 points):

    - row 1 (2 points): SciPy's rule on 2 points (the trapezoid);
    - odd prefix length m: composite Simpson, extended by one 3-point panel
      on x[m-3:m] at a time (accumulated in ``odd``);
    - even prefix length m: the composite Simpson of the first m - 1 points
      plus SciPy's correction for the last interval, obtained as
      simpson(4 points) - simpson(first 3 of those points) on x[m-4:m]; it
      depends only on the last 4 points.

    Self-check: rows 1, 2, 3, 4, N//2, N-2, N-1 (those that exist) are
    compared with a direct SciPy evaluation (rtol 1e-12, atol 1e-14). If any
    differs (e.g. a future SciPy uses a different rule), the whole matrix is
    rebuilt the slow, direct way (one SciPy call per row, O(N³)). For N <= 1
    the result is the zero matrix.
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

    Typical use: remove the unphysical oscillations of g(r) below the first
    peak and see their effect on S(Q). Steps:

    1. g(r) is cut (set to 0) in the regions given by `cut` (`_apply_cut`).
    2. Transform to Q space with ``getSineFT(ar, g_cut, qu, w=0)``
       (no window, s = 1, full r range):

           S(Q) = 1 + (2π²ρ/Q) · stf(Q) = 1 + (4πρ/Q) ∫ r [g(r) - 1] sin(Qr) dr
           S(Q) = 1 + 4πρ ∫ r² [g(r) - 1] dr        for Q <= 0 (Q -> 0 limit)

    3. Transform back with ``getSineFT(qu, S, ar, w=0)``:
       G(r) = 2/π ∫ Q [S(Q) - 1] sin(Qr) dQ over the whole `qu` grid.
    4. g(r) = 1 + G/(4πρr) for r > 0 and 0 for r <= 0;
       RDF = 4πρr² g; T = 4πρr g; N(r_i) = simpson(RDF[:i+1], x=ar[:i+1]),
       N(ar[0]) = 0 (same convention as `sineFT`).

    Parameters
    ----------
    qu : array_like, shape (N_Q,)
        Q grid (Å⁻¹) on which S(Q) is computed; also the integration grid of
        the transform back to r. Points with Q <= 0 get the S(0) limit.
    ar : array_like, shape (N_r,)
        r grid (Å) of `pdf`, also the output r grid.
    pdf : array_like, shape (N_r,)
        g(r) (dimensionless) on `ar`.
    density : float
        Atomic number density ρ (atoms/Å³).
    cut : list of float
        Regions where g(r) is set to 0 before the transform:

        - ``[-1]``: zero g(r) from the first point up to and including the
          last point where g(r) < 0 (nothing is cut if g(r) is never
          negative);
        - ``[r0, a1, b1, a2, b2, ...]`` (odd length): zero every point with
          r < r0 and every point strictly inside an open interval
          (a_k, b_k), in Å. ``[r0]`` alone cuts only r < r0.

    Returns
    -------
    tuple of 6 ndarrays
        (S(Q) on `qu`, dimensionless; G(r) (Å⁻²), g(r), RDF(r) (atoms/Å),
        T(r) (Å⁻²), N(r) (atoms), all on `ar`).

    Raises
    ------
    ValueError
        If `cut` has an even length (including empty) and is not ``[-1]``;
        if `getSineFT` returns None, i.e. `ar` and `pdf` (or `qu` and S)
        have different lengths (a message is also printed by `getSineFT`).

    Notes
    -----
    - Both transforms use no window; the r -> Q transform integrates over the
      whole `ar` grid and the Q -> r transform over the whole `qu` grid, so
      truncation effects of both grids are present in the result.
    - The type hint allows None for G(r), but None is never returned (the
      function raises instead).

    Changes from the legacy code (audit 2026-11, C17): Q = 0 and r = 0 are
    handled with their analytic limits (legacy divided by Q and r and
    produced NaN/inf); T = 4πρr g is computed directly; the running integral
    includes the current point (legacy integrated up to r_{i-1}), matching
    `sineFT`; ``[-1]`` with no negative value no longer zeroes the first two
    points; any odd-length `cut` is accepted and an even length raises
    ValueError (legacy silently ignored lengths above 13).

    See Also
    --------
    getSineFT, sineFT, _apply_cut
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
    """
    Zero g(r) in the regions defined by `cut` (see `backFT`).

    Parameters
    ----------
    ar : ndarray, shape (N,)
        r grid (Å).
    pdf : ndarray, shape (N,)
        g(r) on `ar`; not modified.
    cut : list of float
        ``[-1]`` (exactly one element equal to -1): zero ``pdf[:i+1]`` where
        i is the index of the last negative value (no change if none);
        otherwise ``[r0, a1, b1, ...]`` of odd length: zero where r < r0 or
        a_k < r < b_k (strict inequalities). Values are converted to float.

    Returns
    -------
    ndarray, shape (N,)
        A copy of `pdf` with the selected points set to 0.

    Raises
    ------
    ValueError
        If `cut` is not ``[-1]`` and has an even length (including 0).

    Notes
    -----
    The ``[-1]`` rule uses the index order, not r: everything before the
    last negative value is zeroed even if g(r) is positive in between.
    Intervals with a_k >= b_k select nothing.
    """
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


# ============================================================================
# Matrix form of the transforms, with exact errors (values identical to
# sineFT / backFT; added 2026-11 for the D4 notebooks of D4examples_v3)
# ============================================================================
@dataclass
class FourierSet:
    """
    Real-space functions on an r grid with their 1σ errors.

    Attributes
    ----------
    r : ndarray, shape (N_r,)
        r grid (Å).
    G, dG : ndarray
        Pair correlation function G(r) (Å⁻²) and its error.
    g, dg : ndarray
        Pair distribution function g(r) and its error (g = 0 for r <= 0).
    RDF, dRDF : ndarray
        Radial distribution function 4πρr²g (atoms/Å) and its error.
    T, dT : ndarray
        T(r) = 4πρr g (Å⁻²) and its error.
    N, dN : ndarray
        Running integral of the RDF from r[0] (atoms) and its error.
    jac_G, jac_g : ndarray, shape (N_r, N_Q)
        Jacobians dG/dS and dg/dS with respect to the original S(Q); any
        linear function y = J (S - 1) has σ_y² = (J²) σ_S².
    S, dS : ndarray or None
        S(Q) and its error (only for the sets of `back_fourier_iterations`).
    """

    r: NDArray[np.float64]
    G: NDArray[np.float64]
    dG: NDArray[np.float64]
    g: NDArray[np.float64]
    dg: NDArray[np.float64]
    RDF: NDArray[np.float64]
    dRDF: NDArray[np.float64]
    T: NDArray[np.float64]
    dT: NDArray[np.float64]
    N: NDArray[np.float64]
    dN: NDArray[np.float64]
    jac_G: NDArray[np.float64]
    jac_g: NDArray[np.float64]
    S: Optional[NDArray[np.float64]] = None
    dS: Optional[NDArray[np.float64]] = None

    def table(self) -> NDArray[np.float64]:
        """
        Return the (N_r, 11) array r, G, dG, g, dg, RDF, dRDF, T, dT, N, dN.

        Same layout as ``sineFT(..., with_errors=True)``, so it can be
        written with ``toscana.dataio.saveRSCF``.
        """
        return np.column_stack([self.r, self.G, self.dG, self.g, self.dg, self.RDF,
                                self.dRDF, self.T, self.dT, self.N, self.dN])


def ft_matrix(
    q: NDArray[np.float64], r: NDArray[np.float64], window: int = 0,
    qmax: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    Matrix W of the sine transform, G(r) = W @ (S(Q) - 1).

        W_jk = 2/π · Q_k M(Q_k) sin(Q_k r_j) w_k

    with w_k the Simpson weights of the Q grid (`_simpson_weights`, the rule
    of `sineFT`) and M the window evaluated as in `sineFT`.

    Parameters
    ----------
    q : array_like, shape (N_Q,)
        Q grid (Å⁻¹), increasing.
    r : array_like, shape (N_r,)
        r grid (Å).
    window : int, optional
        0 (default): step, M = 1 for 0 <= Q < qmax; 1: standard Lorch,
        M(0) = 1 (not normalised).
    qmax : float, optional
        Window limit (Å⁻¹); default the last Q. Points with Q >= qmax have
        M = 0.

    Returns
    -------
    ndarray, shape (N_r, N_Q)

    Raises
    ------
    ValueError
        If `window` is not 0 or 1.
    """
    if window not in (0, 1):
        raise ValueError("ft_matrix: window must be 0 (no window) or 1 (Lorch).")
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    qmax = float(q[-1]) if qmax is None else float(qmax)
    func = step if window == 0 else Lorch
    win = np.array([float(func(float(x), qmax)) for x in q])
    base = 2.0 / np.pi * q * win * _simpson_weights(q)
    return np.sin(np.outer(r, q)) * base


def _functions_from_G(r, G, jac_G, var_s, rho, crun):
    """g, RDF, T, N and the errors of all the functions, from G and dG/dS."""
    pos = r > 0
    inv = np.zeros_like(r)
    inv[pos] = 1.0 / (4.0 * np.pi * rho * r[pos])
    g = np.where(pos, 1.0 + G * inv, 0.0)
    jac_g = jac_G * inv[:, None]
    rdf = r * G + 4.0 * np.pi * rho * r**2
    tor = G + 4.0 * np.pi * rho * r
    run = crun @ rdf

    def err(jac):
        return np.sqrt((jac**2) @ var_s)

    dG = err(jac_G)
    dn = err(crun @ (r[:, None] * jac_G))
    return FourierSet(r, G, dG, g, err(jac_g), rdf, np.abs(r) * dG, tor, dG, run, dn,
                      jac_G, jac_g)


def fourier_with_errors(
    q: NDArray[np.float64], s: NDArray[np.float64], ds: NDArray[np.float64],
    r: NDArray[np.float64], rho: float, window: int = 0, qmax: Optional[float] = None,
    crun: Optional[NDArray[np.float64]] = None,
) -> FourierSet:
    """
    S(Q) -> G(r), g(r), RDF(r), T(r), N(r), each with its exact 1σ error.

    The same transform and quadrature as `sineFT` (values agree to rounding,
    ~1e-11), written as matrix products: G = W (S - 1) with W from
    `ft_matrix`, g = 1 + G/(4πρr) (0 for r <= 0), RDF = 4πρr²g,
    T = 4πρr g, N = C·RDF with C the running Simpson matrix
    (`_running_simpson_matrix`). Every function is linear in S(Q), so for
    uncorrelated errors σ_y,j² = Σ_k J_jk² σ_S,k² with its Jacobian J; σN
    includes the correlations of neighbouring G(r) points through C. The
    Jacobians are kept in the result for `back_fourier_iterations`.

    Parameters
    ----------
    q, s, ds : array_like, shape (N_Q,)
        Q (Å⁻¹, increasing), S(Q) and its 1σ error.
    r : array_like, shape (N_r,)
        r grid (Å).
    rho : float
        Atomic density (atoms/Å³).
    window : int, optional
        0 = none (default), 1 = Lorch (M(0) = 1).
    qmax : float, optional
        Truncation / Lorch width (Å⁻¹); default the last Q.
    crun : ndarray, optional
        Running Simpson matrix of `r` (to reuse it between calls).

    Returns
    -------
    FourierSet
    """
    q, s, ds, r = (np.asarray(a, dtype=np.float64) for a in (q, s, ds, r))
    W = ft_matrix(q, r, window, qmax)
    crun = _running_simpson_matrix(r) if crun is None else crun
    return _functions_from_G(r, W @ (s - 1.0), W, ds**2, rho, crun)


def back_fourier_iterations(
    q: NDArray[np.float64], ds: NDArray[np.float64], start: FourierSet, rho: float,
    cut: list[float], niter: int, crun: Optional[NDArray[np.float64]] = None,
) -> list[FourierSet]:
    """
    Back-Fourier iterations (`backFT` applied `niter` times) with exact errors.

    Starting from g(r) of `start`, each iteration (as `backFT`):

    1. g(r) = 0 in the regions given by `cut` (`_apply_cut`);
    2. S(Q) = 1 + (4πρ/Q) ∫ r [g(r) - 1] sin(Qr) dr over the whole r grid
       (S(0) = 1 + 4πρ ∫ r²[g - 1] dr for Q <= 0);
    3. G(r) = 2/π ∫ Q [S(Q) - 1] sin(Qr) dQ over the whole Q grid, no window;
       then g, RDF, T, N.

    Every step is linear (affine) in the original S(Q); the cut is a
    projection (the derivative is 0 at the points set to 0). The Jacobian
    dg/dS is carried through the iterations, so the errors of the functions
    and of the S(Q) of every iteration are exact for uncorrelated errors `ds`
    of the original S(Q).

    Parameters
    ----------
    q : array_like, shape (N_Q,)
        Q grid of the original S(Q) (Å⁻¹).
    ds : array_like, shape (N_Q,)
        1σ errors of the original S(Q).
    start : FourierSet
        Result of `fourier_with_errors` (no window or Lorch) on the same q.
    rho : float
        Atomic density (atoms/Å³).
    cut : list of float
        As in `backFT`: ``[r0, a1, b1, ...]`` (odd length) or ``[-1]``.
    niter : int
        Number of iterations.
    crun : ndarray, optional
        Running Simpson matrix of ``start.r``.

    Returns
    -------
    list of FourierSet
        Iterations 1..niter, each with S and dS.
    """
    q = np.asarray(q, dtype=np.float64)
    var_s = np.asarray(ds, dtype=np.float64) ** 2
    r = start.r
    crun = _running_simpson_matrix(r) if crun is None else crun
    wr = _simpson_weights(r)
    # S - 1 = A (g_cut - 1)
    A = np.empty((len(q), len(r)))
    pos_q = q > 0
    A[pos_q] = (4.0 * np.pi * rho / q[pos_q])[:, None] * np.sin(np.outer(q[pos_q], r)) * (r * wr)
    A[~pos_q] = 4.0 * np.pi * rho * r**2 * wr
    # G = W0 (S - 1): whole Q grid, no window
    W0 = np.sin(np.outer(r, q)) * (2.0 / np.pi * q * _simpson_weights(q))
    out = []
    g, jac_g = start.g, start.jac_g
    for _ in range(int(niter)):
        g_cut = _apply_cut(r, g, cut)
        keep = (g_cut != 0.0) | (g == 0.0)
        S = 1.0 + A @ (g_cut - 1.0)
        jac_S = A @ (jac_g * keep[:, None])
        fset = _functions_from_G(r, W0 @ (S - 1.0), W0 @ jac_S, var_s, rho, crun)
        fset.S, fset.dS = S, np.sqrt((jac_S**2) @ var_s)
        out.append(fset)
        g, jac_g = fset.g, fset.jac_g
    return out
