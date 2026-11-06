"""
Fractional rebinning of (x, y, σ) data onto a regular grid.

Place in the processing chain: after the corrections, the data of all
detector cells (one point per cell and numor, with its angular width) are
regrouped onto a common 2θ grid (degrees) or Q grid (Å⁻¹) before being written
for CORRECT. Each input point is treated as a rectangle of finite width that
is shared among the bins it overlaps, in proportion to the overlap.

Public functions
----------------
get_xlim(xmin, xmax, dbin)
    Number of bins, bin edges and bin centres of a regular grid.
get_bins(x, xdel, x_lim)
    Bins covered by a rectangle of width `xdel` centred at `x` and the
    fraction of the rectangle in each.
rebin(xdel, wlength, data, xmin, xmax, dbin)
    Rebin a 3-column (x, y, σ) array: weighted mean per bin with propagated
    errors.

Conventions and units
---------------------
x is 2θ in degrees or Q in Å⁻¹; bin centres are xmin + i·dbin, edges at
centres ± dbin/2. Errors are 1σ, assumed uncorrelated. Changes from the
legacy code (audit 2026-11, C19, C29) are documented in the functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_xlim(
    xmin: float, xmax: float, dbin: float
) -> tuple[int, list[float], list[float]]:
    """
    Calculate the number, edges and centres of the bins of a regular grid.

    Bin i has centre xmin + i·dbin and edges xmin + (i ∓ 1/2)·dbin. The
    number of bins is

        nb = floor((xmax - xmin + dbin)/dbin + 1e-9),

    i.e. floor((xmax - xmin)/dbin) + 1: the grid starts at xmin and includes
    xmax when (xmax - xmin)/dbin is an integer (up to a 1e-9 tolerance).

    Parameters
    ----------
    xmin : float
        Centre of the first bin (units of x: degrees for 2θ, Å⁻¹ for Q).
    xmax : float
        Centre of the last bin, if commensurate with `dbin`; otherwise the
        last centre is the largest xmin + i·dbin <= xmax.
    dbin : float
        Bin width (> 0, same units as x).

    Returns
    -------
    nb : int
        Number of bins.
    x_lim : list of float
        Bin edges, length nb + 1: xmin - dbin/2 + i·dbin.
    x_bin : list of float
        Bin centres, length nb: xmin + i·dbin.

    Notes
    -----
    Edges and centres are computed as xmin + i·dbin (not by accumulation),
    so rounding errors do not grow along the grid. ``dbin = 0`` raises
    ZeroDivisionError (from the float division); a negative `dbin` is not
    meaningful.

    Changes from the legacy code (audit 2026-11, C19): ``int()`` of a float
    ratio such as 3.9999999 dropped the last bin (``get_xlim(0, 0.3, 0.1)``
    gave 3 bins instead of 4). A small tolerance (1e-9) keeps floor semantics
    for genuinely non-commensurate ranges.
    """
    xini = xmin - dbin / 2.0
    xfin = xmax + dbin / 2.0
    # Bug fix vs legacy: int() of a float ratio such as 3.9999999 dropped the
    # last bin (get_xlim(0, 0.3, 0.1) gave 3 bins instead of 4). A small
    # tolerance keeps floor semantics for genuinely non-commensurate ranges.
    nb = int(np.floor((xfin - xini) / dbin + 1e-9))

    x_lim = [xmin - dbin / 2.0 + i * dbin for i in range(nb + 1)]
    x_bin = [xmin + i * dbin for i in range(nb)]
    return nb, x_lim, x_bin


def get_bins(
    x: float, xdel: float, x_lim: list[float]
) -> tuple[list[int], list[float]]:
    """
    Determine the bins covered by a rectangle and the fraction in each.

    The rectangle is [x - xdel/2, x + xdel/2]. With dbin = x_lim[1] - x_lim[0]
    the first and last bins touched are

        b1 = floor((x1 - x_lim[0])/dbin),  b2 = floor((x2 - x_lim[0])/dbin),

    and the fraction of the rectangle in each bin is

    - b1 == b2: [1.0] (whole rectangle in one bin, whatever its width);
    - b2 == b1 + 1: (x_lim[b1+1] - x1)/xdel and 1 - that;
    - b2 > b1 + 1: (x_lim[b1+1] - x1)/xdel, then dbin/xdel for every
      intermediate bin, then (x2 - x_lim[b2])/xdel.

    The fractions sum to 1.

    Parameters
    ----------
    x : float
        Centre of the rectangle (units of x).
    xdel : float
        Full width of the rectangle (> 0, units of x).
    x_lim : list of float
        Bin edges from `get_xlim` (regular grid, at least 2 edges).

    Returns
    -------
    bins : list of int
        Indices of the bins covered (consecutive, increasing).
    frac : list of float
        Fraction of the rectangle in each of these bins.
        Both lists are empty if the rectangle is rejected (see Notes).

    Notes
    -----
    Rejection rule (as implemented): the rectangle is rejected only if
    ``b1 < 0`` (it starts below the first edge x_lim[0]) or
    ``b2 >= len(x_lim)`` (it ends at or beyond x_lim[-1] + dbin). A rectangle
    that ends beyond the last edge by less than one bin width is NOT
    rejected: the part beyond the last edge is assigned to the index
    ``nb = len(x_lim) - 1``, which is not a real bin. `rebin` allocates an
    extra accumulator for that index and discards it, so such a point
    contributes to the last real bin with only its overlapping fraction (and
    a point lying entirely in [x_lim[-1], x_lim[-1] + dbin) is silently
    dropped). The low and high ends are therefore treated asymmetrically.

    A rectangle ending exactly on an edge gets a zero fraction in the next
    bin. ``xdel = 0`` gives one bin with fraction 1 (x1 = x2, so no division
    by xdel occurs). A NaN x or xdel raises ValueError (``int(nan)``). With a
    negative `xdel` that crosses an edge (b2 < b1), prints
    "ERROR in get_bins" and returns empty lists.

    Changes from the legacy code (audit 2026-11, C19): floor instead of
    truncation toward zero, so a point starting just below the first bin
    edge gives b1 = -1 (and is rejected) instead of being counted as fully
    inside bin 0.
    """
    bins: list[int] = []
    frac: list[float] = []

    dbin = x_lim[1] - x_lim[0]

    x1 = x - xdel / 2.0
    x2 = x + xdel / 2.0

    # Bug fix vs legacy: floor instead of truncation toward zero, so a point
    # starting just below the first bin edge gives b1 = -1 (and is rejected
    # below) instead of being counted as fully inside bin 0.
    b1 = int(np.floor((x1 - x_lim[0]) / dbin))
    b2 = int(np.floor((x2 - x_lim[0]) / dbin))
    if b1 < 0 or b2 >= len(x_lim):
        return bins, frac

    deltab = b2 - b1
    if deltab == 0:
        bins.append(b1)
        frac.append(1.0)
    elif deltab == 1:
        f1 = (x_lim[b1 + 1] - x1) / xdel
        bins.append(b1)
        frac.append(f1)
        bins.append(b1 + 1)
        frac.append(1.0 - f1)
    elif deltab > 1:
        f1 = (x_lim[b1 + 1] - x1) / xdel
        bins.append(b1)
        frac.append(f1)
        for i in range(1, deltab):
            bins.append(b1 + i)
            frac.append(dbin / xdel)
        f2 = (x2 - x_lim[b2]) / xdel
        bins.append(b2)
        frac.append(f2)
    else:
        print("ERROR in get_bins")

    return bins, frac


def rebin(  # noqa: N802 (legacy API)
    xdel: float,
    wlength: float,
    data: NDArray[np.float64],
    xmin: float,
    xmax: float,
    dbin: float,
) -> NDArray[np.float64]:
    """
    Rebin experimental (x, y, σ) data onto a regular 2θ or Q grid.

    Each input point i is a rectangle of width Δx_i centred at x_i, shared
    among the bins it overlaps with fractions f_ib (`get_bins`). For each bin
    b the result is the weighted mean and its error

        y_b = Σ_i f_ib y_i / Σ_i f_ib,
        σ_b = sqrt(Σ_i f_ib² σ_i²) / Σ_i f_ib.

    Width of the rectangles:

    - ``wlength <= 0``: x is in the units of the grid (normally 2θ in
      degrees) and Δx_i = `xdel` for all points;
    - ``wlength > 0``: x is Q (Å⁻¹), `wlength` the wavelength λ (Å) and
      `xdel` an angular width Δ(2θ) in degrees, converted point by point to

          ΔQ_i = (2π/λ) cos θ_i · Δ(2θ) · π/180,
          cos θ_i = sqrt(1 - (Q_i λ/4π)²)

      (from Q = (4π/λ) sin θ, dQ/d(2θ) = (2π/λ) cos θ).

    Parameters
    ----------
    xdel : float
        Width of each input point: in units of x if ``wlength <= 0``, in
        degrees of 2θ if ``wlength > 0`` (typically the angular width of a
        detector cell).
    wlength : float
        Incident wavelength λ (Å) when x is Q; any value ``<= 0`` means "x is
        already in the units of the grid, no conversion".
    data : ndarray, shape (N, >=3)
        Columns x, y, σ_y (further columns ignored). Need not be sorted.
    xmin, xmax, dbin : float
        Output grid as in `get_xlim`: centres xmin + i·dbin up to xmax, bin
        width dbin (units of x).

    Returns
    -------
    ndarray, shape (nb, 3)
        Columns: bin centre x_b, mean y_b, error σ_b. Empty bins (no
        contribution, Σ f = 0) have y = 0 and σ = 0.

    Raises
    ------
    ValueError
        From `get_bins` (``int(nan)``) if a point with finite y and σ has a
        NaN x, or if ``wlength > 0`` and Q_i λ/4π > 1 (cos θ is then NaN; a
        NumPy RuntimeWarning is also emitted).

    Notes
    -----
    - Points whose y or σ is NaN are skipped entirely.
    - Points that start below the first edge are dropped; points that stick
      out beyond the last edge contribute only their overlapping fraction to
      the last bin (see `get_bins`, Notes); this extra "overflow" accumulator
      (index nb) is computed and discarded.
    - Because the rectangles of neighbouring input points overlap several
      bins, neighbouring output points are slightly correlated; the errors
      above ignore this.
    - Side effect: prints a debug line ``"0 ", len(x), len(y), len(σ)``.

    Changes from the legacy code (audit 2026-11, C29; derived from
    `legacy/lib/toscana.py::rebin`): the errors are combined as
    sqrt(Σ f²σ²)/Σ f (error of a weighted mean); legacy used Σ fσ / Σ f, i.e.
    averaged the errors, which does not decrease when points are combined.
    The bin count (`get_xlim`, C19) and bin indexing (`get_bins`, floor) were
    also fixed.
    """
    x_dat = data[:, 0]
    y_dat = data[:, 1]
    e_dat = data[:, 2]
    print("0 ", len(x_dat), len(y_dat), len(e_dat))

    nbins, x_lim, x_bin = get_xlim(xmin, xmax, dbin)

    y_bin: list[float] = []
    e_bin: list[float] = []
    f_bin: list[float] = []

    for _ in range(nbins + 1):
        y_bin.append(0.0)
        e_bin.append(0.0)
        f_bin.append(0.0)

    if wlength <= 0:
        for i in range(len(x_dat)):
            if (np.isnan(y_dat[i]) == False) and (np.isnan(e_dat[i]) == False):  # noqa: E712
                bins, frac = get_bins(float(x_dat[i]), xdel, x_lim)
                for j in range(len(bins)):
                    y_bin[bins[j]] += frac[j] * float(y_dat[i])
                    e_bin[bins[j]] += (frac[j] * float(e_dat[i])) ** 2
                    f_bin[bins[j]] += frac[j]
    else:
        for i in range(len(x_dat)):
            if (np.isnan(y_dat[i]) == False) and (np.isnan(e_dat[i]) == False):  # noqa: E712
                qdel = (
                    2.0
                    * np.pi
                    / wlength
                    * np.sqrt(1.0 - (float(x_dat[i]) * wlength / 4.0 / np.pi) ** 2)
                )
                qdel *= xdel * np.pi / 180.0
                bins, frac = get_bins(float(x_dat[i]), float(qdel), x_lim)
                for j in range(len(bins)):
                    y_bin[bins[j]] += frac[j] * float(y_dat[i])
                    e_bin[bins[j]] += (frac[j] * float(e_dat[i])) ** 2
                    f_bin[bins[j]] += frac[j]

    # Bug fix vs legacy: errors are combined as sqrt(sum(f^2 e^2)) / sum(f)
    # (error of a weighted mean). Legacy used sum(f e) / sum(f), i.e. averaged
    # the errors, which does not decrease when points are combined.
    for i in range(nbins + 1):
        if f_bin[i] != 0:
            y_bin[i] /= f_bin[i]
            e_bin[i] = float(np.sqrt(e_bin[i])) / f_bin[i]

    x_bin_arr: NDArray[np.float64] = np.array(x_bin, dtype=np.float64)
    y_bin_arr: NDArray[np.float64] = np.array(y_bin[0:-1], dtype=np.float64)
    e_bin_arr: NDArray[np.float64] = np.array(e_bin[0:-1], dtype=np.float64)

    x_bin_arr = x_bin_arr.reshape(x_bin_arr.shape[0], 1)
    y_bin_arr = y_bin_arr.reshape(y_bin_arr.shape[0], 1)
    e_bin_arr = e_bin_arr.reshape(e_bin_arr.shape[0], 1)

    binned = np.concatenate((x_bin_arr, y_bin_arr), axis=1)
    binned = np.concatenate((binned, e_bin_arr), axis=1)
    return binned
