"""
Simple smoothing of a curve.

Place in the processing chain: auxiliary tool (e.g. to smooth a noisy curve
before searching for minima or for display); not used in the main
correction chain.

Public functions
----------------
smooth_curve(x, y, smoothing_factor)
    Centred moving average over a number of points proportional to the
    length of the curve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def smooth_curve(
    x: ArrayLike, y: ArrayLike, smoothing_factor: float
) -> NDArray[np.float64]:
    """
    Smooths a function using a moving average-window.

    Parameters
    ----------
    x : array_like
        Array of x values.
    y : array_like
        Array of y values.
    smoothing_factor : float
        Smoothing factor between 0 and 1. A smaller smoothing factor results
        in less smoothing, while a larger factor results in more aggressive
        smoothing.

    Returns
    -------
    numpy.ndarray
        Array of smoothed y values.

    Notes
    -----
    With n = len(y), w = int(smoothing_factor·n) (truncation toward zero) and
    h = w // 2, the output is

        ys_i = mean(y[max(0, i - h) : min(n, i + h + 1)]),

    i.e. a centred unweighted average over 2h + 1 points (w + 1 points for
    even w), using fewer points within h of the ends. If w == 0 the input is
    returned unchanged (as ``numpy.asarray(y)``, possibly the same object,
    keeping its dtype); w == 1 gives h = 0, i.e. a float copy of y.
    NaN values contaminate every window containing them.

    The window is a centred number of POINTS, 2*(window_size//2)+1, shrinking
    at the edges; `x` is not used, so on a non-uniform grid the window width in
    x varies along the curve.
    """
    y_arr = np.asarray(y)
    window_size = int(smoothing_factor * len(y_arr))
    if window_size == 0:
        return y_arr

    smoothed_y = []
    for i in range(len(y_arr)):
        start = max(0, i - window_size // 2)
        end = min(len(y_arr), i + window_size // 2 + 1)
        smoothed_value = np.mean(y_arr[start:end])
        smoothed_y.append(smoothed_value)
    return np.array(smoothed_y)
