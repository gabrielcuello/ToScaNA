"""
Diagnostic plots of the d4creg regrouping workflow.

Called by `toscana.workflows.d4creg` when plotting is requested in the
d4creg parameter file. Each plot opens a new figure and calls
``plt.show()``; nothing is saved.

Public names
------------
defColors
    Fixed list of 20 CSS4 colours (hex strings).
makePlotsA
    Three figures in angle: individual numors, individual detector banks,
    total diffractogram.
makePlotsQ
    One figure: total diffractogram in Q.

Inputs (from `toscana.dataio`): ``runInfo`` is the d4creg run dictionary
(keys used: ``"file"``, ``"nbrNumors"``, ``"angular_scale"`` (min, max, step
in degrees), ``"q_scale"`` (min, max, step in 1/Å), ``"normalisation"``
(mode, time in s, monitor counts)); ``head`` the list of numor headers;
``numor`` the array of numors, shape (n_numors, ≥2, n_points), rows 0 and 1
being angle (degrees) and counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def defColors() -> list[str]:  # noqa: N802 (legacy API)
    """
    Define a deterministic list of CSS4 colors (legacy behavior).

    Returns
    -------
    list of str
        20 hex colour strings from ``matplotlib.colors.CSS4_COLORS``, in the
        order red, blue, green, black, cyan, magenta, gold, orange, brown,
        gray, silver, pink, purple, navy, teal, lime, olive, salmon, maroon,
        chartreuse (indices 0-19).
    """
    import matplotlib.colors as mcolors

    colors: list[str] = []
    colors.append(str(mcolors.CSS4_COLORS["red"]))  # 0
    colors.append(str(mcolors.CSS4_COLORS["blue"]))  # 1
    colors.append(str(mcolors.CSS4_COLORS["green"]))  # 2
    colors.append(str(mcolors.CSS4_COLORS["black"]))  # 3
    colors.append(str(mcolors.CSS4_COLORS["cyan"]))  # 4
    colors.append(str(mcolors.CSS4_COLORS["magenta"]))  # 5
    colors.append(str(mcolors.CSS4_COLORS["gold"]))  # 6
    colors.append(str(mcolors.CSS4_COLORS["orange"]))  # 7
    colors.append(str(mcolors.CSS4_COLORS["brown"]))  # 8
    colors.append(str(mcolors.CSS4_COLORS["gray"]))  # 9
    colors.append(str(mcolors.CSS4_COLORS["silver"]))  # 10
    colors.append(str(mcolors.CSS4_COLORS["pink"]))  # 11
    colors.append(str(mcolors.CSS4_COLORS["purple"]))  # 12
    colors.append(str(mcolors.CSS4_COLORS["navy"]))  # 13
    colors.append(str(mcolors.CSS4_COLORS["teal"]))  # 14
    colors.append(str(mcolors.CSS4_COLORS["lime"]))  # 15
    colors.append(str(mcolors.CSS4_COLORS["olive"]))  # 16
    colors.append(str(mcolors.CSS4_COLORS["salmon"]))  # 17
    colors.append(str(mcolors.CSS4_COLORS["maroon"]))  # 18
    colors.append(str(mcolors.CSS4_COLORS["chartreuse"]))  # 19
    return colors


def makePlotsA(  # noqa: N802 (legacy API)
    runInfo: Mapping[str, Any],
    diffA: np.ndarray,
    head: Sequence[Mapping[str, Any]],
    numor: np.ndarray,
) -> None:
    """
    Plot angular-scale diffractograms (legacy behavior).

    Three figures, each followed by ``plt.show()``:

    1. "Individual numors (<nbrNumors> in total) for sample <file>", 9 x 5
       inches: ``numor[i, 1, :]`` versus ``numor[i, 0, :]`` for every numor,
       colour ``colors[i % 20]``, label ``head[i]["numor"]`` (no legend is
       drawn).
    2. "Individual detectors for <file>", 9 x 6: ``diffA[i, 1, :]`` versus
       ``diffA[i, 0, :]`` for the detector banks i = 1..9, labels
       "Detector i", with legend.
    3. "Diffractogram for <file>", 9 x 6: the total ``diffA[0]``, solid
       line, label `file`, with legend.

    All three use the abscissa range ``runInfo["angular_scale"][0:2]``
    (degrees) and the ordinate range 0.9 × min to 1.1 × max of the TOTAL
    diffractogram ``diffA[0, 1, :]`` (NaN ignored), so individual numors or
    banks may fall outside. The ordinate label is "Counts/(<N> monitor counts)" with N =
    ``runInfo["normalisation"][2]`` when ``runInfo["normalisation"][0] ==
    "monitor"``, otherwise "Counts/(<t> seconds)" with t =
    ``runInfo["normalisation"][1]``.

    Parameters
    ----------
    runInfo : Mapping[str, Any]
        d4creg run dictionary; keys ``"nbrNumors"``, ``"file"``,
        ``"angular_scale"``, ``"normalisation"``.
    diffA : ndarray, shape (10, ≥2, n_angles)
        Regrouped diffractograms from `toscana.dataio.d4creg_outputs.getDiffA`:
        index 0 the total, 1..9 the detector banks; row 0 the angle
        (degrees), row 1 the counts.
    head : sequence of mappings
        Numor headers; ``head[i]["numor"]`` is used as label.
    numor : ndarray, shape (n_numors, ≥2, n_points)
        Individual numors; rows 0 (angle) and 1 (counts) are plotted.

    Raises
    ------
    KeyError, IndexError
        If a key is missing or `diffA` has fewer than 10 entries.
    """
    import matplotlib.pyplot as plt

    colors = defColors()
    plt.figure(figsize=(9, 5))
    plt.title(
        "Individual numors ({} in total) for sample {}".format(
            runInfo["nbrNumors"], runInfo["file"]
        )
    )

    amin, amax = runInfo["angular_scale"][0], runInfo["angular_scale"][1]
    ymin, ymax = 0.9 * np.nanmin(diffA[0, 1, :]), 1.1 * np.nanmax(diffA[0, 1, :])
    plt.axis((amin, amax, ymin, ymax))
    plt.xlabel("Scattering angle (degrees)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    for i in range(numor.shape[0]):
        plt.plot(numor[i, 0, :], numor[i, 1, :], colors[i % 20], label=head[i]["numor"])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.title("Individual detectors for " + runInfo["file"])
    amin, amax = runInfo["angular_scale"][0], runInfo["angular_scale"][1]
    ymin, ymax = 0.9 * np.nanmin(diffA[0, 1, :]), 1.1 * np.nanmax(diffA[0, 1, :])
    plt.axis((amin, amax, ymin, ymax))
    plt.xlabel("Scattering angle (degrees)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    for i in range(1, 10):
        plt.plot(diffA[i, 0, :], diffA[i, 1, :], colors[i], label="Detector " + str(i))
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.title("Diffractogram for " + runInfo["file"])
    amin, amax = runInfo["angular_scale"][0], runInfo["angular_scale"][1]
    ymin, ymax = 0.9 * np.nanmin(diffA[0, 1, :]), 1.1 * np.nanmax(diffA[0, 1, :])
    plt.axis((amin, amax, ymin, ymax))
    plt.xlabel("Scattering angle (degrees)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    plt.plot(
        diffA[0, 0, :],
        diffA[0, 1, :],
        color=colors[0],
        linestyle="solid",
        label=runInfo["file"],
    )
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    return


def makePlotsQ(  # noqa: N802 (legacy API)
    runInfo: Mapping[str, Any],
    diffQbin: np.ndarray,
    head: Sequence[Mapping[str, Any]],
    numor: np.ndarray,
) -> None:
    """
    Plot Q-scale diffractograms (legacy behavior).

    One 9 x 6 figure "Diffractogram for <file>": ``diffQbin[:, 1]`` versus
    ``diffQbin[:, 0]``, red solid line, label `file`, abscissa "Momentum
    transfer (1/Å)" over ``runInfo["q_scale"][0:2]``, ordinate range 0.9 ×
    min to 1.1 × max of the counts (NaN ignored), legend, grid, then
    ``plt.show()``. The ordinate label is "Counts/(<N> monitor counts)" with N =
    ``runInfo["normalisation"][2]`` when ``runInfo["normalisation"][0] ==
    "monitor"``, otherwise "Counts/(<t> seconds)" with t =
    ``runInfo["normalisation"][1]``.

    Parameters
    ----------
    runInfo : Mapping[str, Any]
        d4creg run dictionary; keys ``"file"``, ``"q_scale"``,
        ``"normalisation"``.
    diffQbin : ndarray, shape (n_q, ≥2)
        Diffractogram rebinned in Q; column 0 Q (1/Å), column 1 counts.
    head, numor
        Unused; kept for symmetry with `makePlotsA`.
    """
    import matplotlib.pyplot as plt

    colors = defColors()
    plt.figure(figsize=(9, 6))
    plt.title("Diffractogram for " + runInfo["file"])
    qmin, qmax = runInfo["q_scale"][0], runInfo["q_scale"][1]
    ymin, ymax = 0.9 * np.nanmin(diffQbin[:, 1]), 1.1 * np.nanmax(diffQbin[:, 1])
    plt.axis((qmin, qmax, ymin, ymax))
    plt.xlabel("Momentum transfer (1/Å)")
    if runInfo["normalisation"][0] == "monitor":
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][2]) + " monitor counts)")
    else:
        plt.ylabel("Counts/(" + str(runInfo["normalisation"][1]) + " seconds)")
    plt.plot(
        diffQbin[:, 0],
        diffQbin[:, 1],
        color=colors[0],
        linestyle="solid",
        label=runInfo["file"],
    )
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    return
