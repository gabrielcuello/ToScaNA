from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def get_chi(y1: ArrayLike, y2: ArrayLike) -> float:
    """
    Calculates the chi (RMS difference) between 2 sets of y-values.

    Parameters
    ----------
    y1 : array_like
        Array of values of the first set of y-values.
    y2 : array_like
        Array of values of the second set of y-values.

    Returns
    -------
    float
        Value of chi between the 2 sets of y-values.
    """
    return float(np.sqrt(np.mean((np.asarray(y1) - np.asarray(y2)) ** 2)))


def fit_and_find_extremum(
    x: ArrayLike, y: ArrayLike
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Fits a second-degree polynomial to the given points, calculates the derivative,
    and finds the x value where the derivative is 0.

    Parameters
    ----------
    x : array_like
        x values.
    y : array_like
        y values.

    Returns
    -------
    tuple
        (extremum_x, extremum_y, fit_values)
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
    """Points with xmin <= x <= xmax and ymin <= y <= ymax, as (n, 1) columns."""
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
    data : array_like
        A 3-column matrix containing x, y, e.

    Returns a 3-column matrix (x, y, e).
    """
    dat = np.asarray(data, dtype=np.float64)
    x, y, e = _select_in_rectangle(
        xmin, xmax, ymin, ymax, dat[:, 0], dat[:, 1], dat[:, 2], "the function fit_range"
    )
    return np.concatenate((x, y, e), axis=1)
