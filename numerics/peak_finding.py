"""
Simple search for minima and peaks of a curve inside an x range.

Place in the processing chain: used in the analysis of real-space functions
and calibration data, e.g. to locate the minimum after the first peak of
g(r) (a natural upper limit for coordination numbers) or the Bragg peaks of
a diffraction pattern.

Public functions
----------------
find_minimum_within_range(x, y, x_min, x_max)
    (x, y) of the minimum of y for x_min <= x <= x_max.
find_peaks_in_range(x, y, x_min, x_max)
    Local maxima in the range whose height divided by the range minimum
    exceeds 3.5.

Conventions
-----------
Ranges are closed (x_min <= x <= x_max). Values are taken from the data
points; there is no interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import find_peaks  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def find_minimum_within_range(
    x: ArrayLike, y: ArrayLike, x_min: float, x_max: float
) -> tuple[np.float64, np.float64]:
    """
    Find the (x, y) pair where y is minimal within the specified x range (legacy behavior).

    Parameters
    ----------
    x : array_like, shape (N,)
        Abscissas (need not be sorted).
    y : array_like, shape (N,)
        Ordinates.
    x_min, x_max : float
        Closed range of x searched (x_min <= x <= x_max).

    Returns
    -------
    x_at_min : numpy.float64
        x of the smallest y in the range (the first one if there are ties).
    y_min : numpy.float64
        That smallest y value.

    Raises
    ------
    ValueError
        If no point lies in the range ("attempt to get argmin of an empty
        sequence", from `numpy.argmin`).

    Notes
    -----
    The minimum is a data point, not interpolated. If y contains NaN in the
    range, `numpy.argmin` returns the first NaN.
    """
    x_arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    y_arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

    mask = (x_arr >= x_min) & (x_arr <= x_max)
    x_range = x_arr[mask]
    y_range = y_arr[mask]

    min_index = int(np.argmin(y_range))
    min_x_position = np.float64(x_range[min_index])
    min_y_value = np.float64(y_range[min_index])

    return min_x_position, min_y_value


def find_peaks_in_range(
    x: ArrayLike, y: ArrayLike, x_min: float, x_max: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Find peaks within an x range and return (peak_x, peak_y/ymin) above a threshold.

    The points with x_min <= x <= x_max are kept (in their original order),
    local maxima are found with ``scipy.signal.find_peaks`` (default
    arguments: any point strictly higher than its neighbours, end points
    excluded), each peak height is divided by the minimum y of the range
    (`find_minimum_within_range`), and the peaks whose ratio exceeds 3.5 are
    returned.

    Parameters
    ----------
    x : array_like, shape (N,)
        Abscissas (should be sorted: neighbours are taken by index).
    y : array_like, shape (N,)
        Ordinates.
    x_min, x_max : float
        Closed range of x searched.

    Returns
    -------
    peak_x : ndarray of float64
        Positions of the selected peaks (data points, not interpolated).
    peak_ratio : ndarray of float64
        Peak height divided by the minimum of y in the range (> 3.5), not
        the peak height itself.

    Raises
    ------
    ValueError
        If no point lies in the range (from `find_minimum_within_range`).

    Notes
    -----
    Preserves the legacy behavior in `legacy/lib/toscana.py`. The threshold
    3.5 is hard-coded and the criterion is meaningful only if the minimum of
    y in the range is positive: a negative minimum flips the sign of the
    ratios (no peak is selected) and a zero minimum gives ±inf with a NumPy
    divide warning.
    """
    x_arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    y_arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

    mask = (x_arr >= x_min) & (x_arr <= x_max)
    x_range = x_arr[mask]
    y_range = y_arr[mask]

    peaks, _props = find_peaks(y_range)
    _xminr, ymin = find_minimum_within_range(x_arr, y_arr, x_min, x_max)

    peak_x_positions = x_range[peaks]
    peak_maximum_values = y_range[peaks] / ymin

    sel = peak_maximum_values > 3.5
    peak_x = peak_x_positions[sel]
    peak_y = peak_maximum_values[sel]

    return peak_x, peak_y
