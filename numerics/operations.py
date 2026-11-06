"""
Combination of two data sets with propagated errors.

Place in the processing chain: used during the corrections (e.g. subtracting
an empty-container or background measurement, mixing two measurements,
averaging two runs, normalising to vanadium) on data sets that share the same
x grid.

Public functions
----------------
binary_sum(w1, data1, w2, data2)
    w1·y1 + w2·y2 with σ = sqrt(w1²σ1² + w2²σ2²) (3-column arrays).
wsum2(w1, data1, w2, data2)
    Four modes: inverse-variance mean, scaling of set 2, mixture
    w1·y1 + (1 - w1)·y2, or general weighted sum.
ratio(y1, e1, y2, e2)
    y1/y2 with propagated error (lists).

Conventions
-----------
Data sets are arrays with columns (x, y, σ); the x column of the result is
copied from the first set without checking that the two x grids agree.
Errors are 1σ and the two sets are assumed uncorrelated. On a size mismatch,
`binary_sum` and `wsum2` print a message and return None (legacy behaviour).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def binary_sum(
    w1: float, data1: NDArray[np.float64], w2: float, data2: NDArray[np.float64]
) -> Optional[NDArray[np.float64]]:
    """
    Weighted sum (or subtraction) of two (x, y, error) data sets.

        y = w1·y1 + w2·y2,   σ = sqrt(w1²σ1² + w2²σ2²)

    Use a negative weight to subtract (e.g. w1 = 1, w2 = -t for a container
    subtraction with transmission factor t).

    Parameters
    ----------
    w1 : float
        Weight for the first set of data (dimensionless).
    data1 : numpy.ndarray, shape (N, >=3)
        First set of data: columns x, y, error (1σ). Columns beyond the
        third are copied unchanged to the result.
    w2 : float
        Weight for the second set of data.
    data2 : numpy.ndarray
        Second set of data (x, y, error), same shape as `data1`.

    Returns
    -------
    numpy.ndarray or None
        A copy of `data1` with column 1 replaced by the weighted sum and
        column 2 by its error; x (column 0) comes from `data1`. None if the
        shapes differ (after printing two error lines).

    Notes
    -----
    The x columns of the two sets are not compared. Uncorrelated errors are
    assumed. NaN in either set propagates to the result.

    Changes from the legacy code (audit 2026-11, C3): the legacy error
    propagation used e instead of e² for the variances (dimensionally wrong;
    it gave 2 instead of sqrt(8) for e1 = e2 = 2 and unit weights); this is
    now corrected.
    """
    if data1.shape != data2.shape:
        print("--- Error in the binary sum (binary_sum).")
        print("--- The input data have not the same shape.")
        return None

    data3 = np.copy(data1)
    data3[:, 1] = w1 * data1[:, 1] + w2 * data2[:, 1]
    # Bug fix vs legacy: the variances are e^2 (legacy used e, which is
    # dimensionally wrong and gave sqrt(8)->2 for e1 = e2 = 2).
    data3[:, 2] = np.sqrt(w1**2 * data1[:, 2] ** 2 + w2**2 * data2[:, 2] ** 2)
    return data3


def ratio(
    y1: Sequence[float], e1: Sequence[float], y2: Sequence[float], e2: Sequence[float]
) -> tuple[list[float], list[float]]:
    """
    Calculate the ratio of two sets of values with propagated errors.

        r = y1/y2,   σ_r = sqrt((y2·e1)² + (y1·e2)²) / y2²

    (equivalent to |r|·sqrt((e1/y1)² + (e2/y2)²) for y1 != 0).

    Parameters
    ----------
    y1, e1 : sequence of float
        Numerator values and their 1σ errors.
    y2, e2 : sequence of float
        Denominator values and their 1σ errors.

    Returns
    -------
    y_rat : list of float
        The ratios; 0.0 where y2 == 0.
    e_rat : list of float
        Their errors; 0.0 where y2 == 0.

    Notes
    -----
    The legacy implementation uses `zip(...)` and therefore truncates to the
    shortest input length, without warning. This function preserves that
    behavior, as well as returning 0 ± 0 (rather than inf or NaN) for a zero
    denominator; such points are therefore indistinguishable from a genuine
    zero ratio with zero error. Uncorrelated numerator and denominator are
    assumed.
    """
    y_rat: list[float] = []
    e_rat: list[float] = []

    for y1_val, e1_val, y2_val, e2_val in zip(y1, e1, y2, e2):
        if y2_val != 0:
            r = y1_val / y2_val
            err = float(
                np.sqrt((y2_val * e1_val) ** 2 + (y1_val * e2_val) ** 2) / y2_val**2
            )
        else:
            r = 0.0
            err = 0.0

        y_rat.append(float(r))
        e_rat.append(float(err))

    return y_rat, e_rat


def wsum2(
    w1: float,
    data1: NDArray[np.float64],
    w2: float,
    data2: NDArray[np.float64],
) -> Optional[NDArray[np.float64]]:
    """
    Weighted sum, mixture or inverse-variance mean of two 3-column data sets.

    The mode is selected by the weights:

    - ``w1 == 0 and w2 == 0``: inverse-variance weighted mean, point by
      point, with w = σ2²/(σ1² + σ2²) (the weight of set 1, ∝ 1/σ1²):

          y = w·y1 + (1 - w)·y2,  σ = sqrt(w²σ1² + (1 - w)²σ2²)
                                    (= σ1σ2/sqrt(σ1² + σ2²)).

      If σ1 = σ2 = 0 at a point, w = 0 and set 2 is taken (σ = 0).
    - ``w1 == 0`` (w2 != 0): scaling of set 2 only (set 1's y and σ are
      ignored): y = w2·y2, σ = |w2|·σ2.
    - ``w2 == 0`` (w1 != 0): mixture y = w1·y1 + (1 - w1)·y2,
      σ = sqrt(w1²σ1² + (1 - w1)²σ2²); requires 0 < w1 <= 1.
    - otherwise: general sum y = w1·y1 + w2·y2,
      σ = sqrt(w1²σ1² + w2²σ2²) (as `binary_sum`).

    Parameters
    ----------
    w1 : float
        Weight of set 1 (dimensionless); 0 has the special meanings above.
    data1 : numpy.ndarray, shape (N, >=3)
        Set 1: columns x, y, σ (1σ). Its x column is used for the result.
    w2 : float
        Weight of set 2; 0 has the special meanings above.
    data2 : numpy.ndarray, shape (N, >=3)
        Set 2: columns x (not used), y, σ.

    Returns
    -------
    numpy.ndarray of shape (N, 3), or None
        Columns x (from `data1`), y, σ. None if the two sets have different
        numbers of rows (after printing two error lines).

    Raises
    ------
    ValueError
        If ``w2 == 0`` and `w1` is not in (0, 1].

    Notes
    -----
    The x columns are not compared. Uncorrelated errors are assumed. In the
    inverse-variance mode a NaN error makes w NaN and propagates.

    Changes from the legacy code (audit 2026-11, C13): with ``w1 == 0`` the
    error is |w2|·σ2 (a negative weight gave negative errors); with
    ``w2 == 0`` and w1 outside (0, 1] legacy printed an error and then
    crashed in ``np.concatenate`` with a confusing message, it now raises
    ValueError.
    """
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    e1 = data1[:, 2]
    y2 = data2[:, 1]
    e2 = data2[:, 2]

    if len(y1) != len(y2):
        print("--- Error in the binary sum (wsum2).")
        print("--- The input vectors have not the same length.")
        return None

    length = len(y1)
    ysum: list[float] = []
    esum: list[float] = []

    if (w1 == 0) and (w2 == 0):
        for i in range(length):
            w = 0.0
            sqerr = e1[i] ** 2 + e2[i] ** 2
            if sqerr != 0:
                w = float(e2[i] ** 2 / sqerr)
            ysum.append(float(w * y1[i] + (1 - w) * y2[i]))
            esum.append(float(np.sqrt(w**2 * e1[i] ** 2 + (1.0 - w) ** 2 * e2[i] ** 2)))
    elif w1 == 0:
        for i in range(length):
            ysum.append(float(w2 * y2[i]))
            # Bug fix vs legacy: |w2| (a negative weight gave negative errors)
            esum.append(float(abs(w2) * e2[i]))
    elif w2 == 0:
        if (w1 > 0) and (w1 <= 1):
            for i in range(length):
                ysum.append(float(w1 * y1[i] + (1 - w1) * y2[i]))
                esum.append(
                    float(np.sqrt(w1**2 * e1[i] ** 2 + (1.0 - w1) ** 2 * e2[i] ** 2))
                )
        else:
            # Bug fix vs legacy: legacy printed an error and then crashed in
            # np.concatenate with a confusing message.
            raise ValueError(
                "wsum2: with w2 = 0 the weight of the first set must be in (0, 1]."
            )
    else:
        for i in range(length):
            ysum.append(float(w1 * y1[i] + w2 * y2[i]))
            esum.append(float(np.sqrt(w1**2 * e1[i] ** 2 + w2**2 * e2[i] ** 2)))

    ysum_arr: NDArray[np.float64] = np.array(ysum, dtype=np.float64)
    esum_arr: NDArray[np.float64] = np.array(esum, dtype=np.float64)

    x1_col = x1.reshape(x1.shape[0], 1)
    ysum_col = ysum_arr.reshape(ysum_arr.shape[0], 1)
    esum_col = esum_arr.reshape(esum_arr.shape[0], 1)

    summed = np.concatenate((x1_col, ysum_col), axis=1)
    summed = np.concatenate((summed, esum_col), axis=1)
    return summed
