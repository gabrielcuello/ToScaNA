"""
Helpers for fitting: goodness of fit, parabola extremum, data selection.

Place in the processing chain: used in the calibration (e.g. the Ni
reflections, zero-angle and wavelength determination) and in interactive
analysis to select the points of a data set inside a rectangle before a fit
and to compare a model with data.

Public functions
----------------
get_chi(y1, y2)
    Root-mean-square difference sqrt(⟨(y1 - y2)²⟩).
fit_and_find_extremum(x, y)
    Least-squares parabola and its extremum x = -b/(2a).
fittingRange0(xmin, xmax, ymin, ymax, abcissas, ordinates, errors)
    Points inside a rectangle, as one (n, 3) array.
fittingRange(xmin, xmax, ymin, ymax, abcissas, ordinates, errors)
    Points inside a rectangle, as three (n, 1) arrays.
fit_range(xmin, xmax, ymin, ymax, data)
    Points of a 3-column array inside a rectangle, as one (n, 3) array.

Conventions
-----------
The rectangle is closed: xmin <= x <= xmax and ymin <= y <= ymax. Problems
(wrong limits, fewer than 10 points, no points) are reported by printed
messages, not by exceptions (legacy behaviour).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def get_chi(y1: ArrayLike, y2: ArrayLike) -> float:
    """
    Calculates the chi (RMS difference) between 2 sets of y-values.

        chi = sqrt( (1/n) Σ_i (y1_i - y2_i)² )

    No weighting by errors and no division by the number of degrees of
    freedom (this is not the reduced χ²).

    Parameters
    ----------
    y1 : array_like
        Array of values of the first set of y-values.
    y2 : array_like
        Array of values of the second set of y-values.

    Returns
    -------
    float
        Value of chi between the 2 sets of y-values (units of y). NaN if
        either input contains NaN or if both are empty (mean of an empty
        array, with a RuntimeWarning).

    Raises
    ------
    ValueError
        If the shapes of `y1` and `y2` cannot be broadcast together.

    Notes
    -----
    Arrays of different but broadcastable shapes (e.g. a scalar) are
    broadcast by NumPy without warning.
    """
    return float(np.sqrt(np.mean((np.asarray(y1) - np.asarray(y2)) ** 2)))


def fit_and_find_extremum(
    x: ArrayLike, y: ArrayLike
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Fits a second-degree polynomial to the given points, calculates the derivative,
    and finds the x value where the derivative is 0.

    The unweighted least-squares fit (``numpy.polyfit(x, y, 2)``) gives
    y ≈ a x² + b x + c; the extremum is at x_e = -b/(2a) (a minimum if a > 0,
    a maximum if a < 0) with value a x_e² + b x_e + c.

    Parameters
    ----------
    x : array_like, shape (n,)
        x values (n >= 3 distinct values for a well-defined parabola).
    y : array_like, shape (n,)
        y values.

    Returns
    -------
    extremum_x : ndarray of float64, shape (1,)
        Position of the extremum, -b/(2a). It may lie outside the range of
        `x`.
    extremum_y : ndarray of float64, shape (1,)
        Value of the fitted parabola at `extremum_x`.
    fit_values : ndarray of float64, shape (n,)
        Fitted parabola evaluated at `x`.

    Raises
    ------
    ZeroDivisionError
        If the fitted coefficient a is exactly 0.0 (a nearly collinear data
        set gives a tiny but non-zero a and a huge, meaningless extremum
        instead).
    TypeError, numpy.linalg.LinAlgError
        From `numpy.polyfit` for invalid input (e.g. empty arrays,
        different lengths).

    Notes
    -----
    No error weighting. With fewer than 3 points `numpy.polyfit` emits a
    RankWarning and the result is not meaningful.
    """
    x_arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    y_arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

    # Fit a second-degree polynomial: a*x^2 + b*x + c
    coeffs: NDArray[np.float64] = np.asarray(
        np.polyfit(x_arr, y_arr, 2), dtype=np.float64
    )
    a, b, _c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # Extremum at derivative zero: 2*a*x + b = 0 -> x = -b/(2*a)
    # (matches `np.poly1d(coeffs).deriv().r` for well-behaved fits, but is easier to type-check)
    extremum_x: NDArray[np.float64] = np.asarray([-b / (2.0 * a)], dtype=np.float64)
    extremum_y: NDArray[np.float64] = np.asarray(
        np.polyval(coeffs, extremum_x), dtype=np.float64
    )
    fit_values: NDArray[np.float64] = np.asarray(
        np.polyval(coeffs, x_arr), dtype=np.float64
    )
    return extremum_x, extremum_y, fit_values


def _select_in_rectangle(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x_in: NDArray[np.float64],
    y_in: NDArray[np.float64],
    e_in: NDArray[np.float64],
    where: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Select the points inside a closed rectangle, as (n, 1) columns.

    Keeps the points with xmin <= x <= xmax and ymin <= y <= ymax (NaN
    values are never inside).

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Limits of the rectangle (units of x and y).
    x_in, y_in, e_in : ndarray, shape (N,)
        Abscissas, ordinates and errors.
    where : str
        Name of the calling function, used in the printed messages.

    Returns
    -------
    tuple of 3 ndarrays, each of shape (n, 1)
        Selected x, y and errors, in their original order.

    Raises
    ------
    ValueError or IndexError
        From NumPy if `x_in`, `y_in` and `e_in` have incompatible lengths
        (the mismatch of x and y is first reported by a printed message).

    Notes
    -----
    Side effects (printed, nothing raised): "ERROR" if len(x_in) !=
    len(y_in) or if xmax < xmin or ymax < ymin (the selection is then
    empty); "WARNING" if fewer than 10 points are selected; "ERROR" if none
    is selected (empty arrays are still returned).
    """
    if len(x_in) != len(y_in):
        print(f"ERROR: Ordinate and abcissa must have the same length in {where}.")
    if (xmax < xmin) or (ymax < ymin):
        print(f"ERROR: The limits of the rectangle are wrong in {where}.")

    mask = (x_in >= xmin) & (x_in <= xmax) & (y_in >= ymin) & (y_in <= ymax)
    n = int(np.count_nonzero(mask))
    if n < 10:
        print(f"WARNING: Less than 10 points for the fitting in {where}.")
    if n == 0:
        print(f"ERROR: No points for the fitting in {where}.")

    return (
        x_in[mask].reshape(-1, 1),
        y_in[mask].reshape(-1, 1),
        e_in[mask].reshape(-1, 1),
    )


def fittingRange0(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    abcissas: ArrayLike,
    ordinates: ArrayLike,
    errors: ArrayLike,
) -> NDArray[np.float64]:
    """
    Define a subset of points inside a rectangle.

    Returns a 3-column matrix (x, y, e).

    Parameters
    ----------
    xmin, xmax : float
        Abscissa limits (inclusive).
    ymin, ymax : float
        Ordinate limits (inclusive).
    abcissas, ordinates, errors : array_like, shape (N,)
        x, y and error values.

    Returns
    -------
    ndarray of float64, shape (n, 3)
        Columns x, y, e of the n points with xmin <= x <= xmax and
        ymin <= y <= ymax, in their original order.

    Notes
    -----
    Messages, warnings (fewer than 10 points) and errors are printed by
    `_select_in_rectangle`, never raised; the messages name the function
    "fittingRange" (not "fittingRange0").

    Changes from the legacy code (audit 2026-11, P20): the three range
    functions share one selection helper, `_select_in_rectangle`.
    """
    x, y, e = _select_in_rectangle(
        xmin, xmax, ymin, ymax,
        np.asarray(abcissas, dtype=np.float64),
        np.asarray(ordinates, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        "function fittingRange",
    )
    return np.concatenate((x, y, e), axis=1)


def fittingRange(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    abcissas: ArrayLike,
    ordinates: ArrayLike,
    errors: ArrayLike,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Define a subset of points inside a rectangle.

    Returns 3 separate (n, 1) arrays: x, y, e.

    Parameters
    ----------
    xmin, xmax : float
        Abscissa limits (inclusive).
    ymin, ymax : float
        Ordinate limits (inclusive).
    abcissas, ordinates, errors : array_like, shape (N,)
        x, y and error values.

    Returns
    -------
    x, y, e : ndarray of float64, each of shape (n, 1)
        The n points with xmin <= x <= xmax and ymin <= y <= ymax, in their
        original order.

    Notes
    -----
    Same selection and printed messages as `fittingRange0`; only the
    output format differs.
    """
    return _select_in_rectangle(
        xmin, xmax, ymin, ymax,
        np.asarray(abcissas, dtype=np.float64),
        np.asarray(ordinates, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        "function fittingRange",
    )


def fit_range(
    xmin: float, xmax: float, ymin: float, ymax: float, data: ArrayLike
) -> NDArray[np.float64]:
    """
    Define a subset of points inside a rectangle.

    Parameters
    ----------
    xmin, xmax : float
        Abscissa limits (inclusive).
    ymin, ymax : float
        Ordinate limits (inclusive).
    data : array_like, shape (N, >=3)
        A 3-column matrix containing x, y, e (further columns ignored).

    Returns
    -------
    ndarray of float64, shape (n, 3)
        Columns x, y, e of the n rows with xmin <= x <= xmax and
        ymin <= y <= ymax, in their original order.

    Notes
    -----
    Same selection and printed messages as `fittingRange0` (they name "the
    function fit_range").
    """
    dat = np.asarray(data, dtype=np.float64)
    x, y, e = _select_in_rectangle(
        xmin, xmax, ymin, ymax, dat[:, 0], dat[:, 1], dat[:, 2], "the function fit_range"
    )
    return np.concatenate((x, y, e), axis=1)
