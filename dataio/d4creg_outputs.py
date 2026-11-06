"""
Regrouping of D4 numors on a 2θ grid and the d4creg output files.

Position in the processing chain
--------------------------------
Second stage of ``workflows.d4creg.d4creg``::

    acquisition.getNumors  -> numor (n_numors, 5, ndet*ncell), head
    getDiffA               -> diffA (10, 5, n_angles): total + 9 banks on a 2θ grid
    saveDiffAngle          -> <file>.adat (CORRECT) and <file>.reg (D4 format)
    (d4creg: 2θ -> Q, numerics.binning.rebin onto the Q grid)
    saveDiffQ              -> <file>.qdat (CORRECT, Q scale)

Functions
---------
getDiffA
    Regroup all cells of all numors on ``np.arange(amin, amax, step)``, total and
    per bank, with monitor-weighted means and propagated errors.
saveDiffAngle
    Write the regrouped diffractograms in angle: ``.adat`` and ``.reg`` files.
saveDiffQ
    Write a 3-column (Q, y, e) diffractogram in CORRECT format: ``.qdat`` file.

Conventions
-----------
- Angles in degrees (2θ, zero angle already subtracted), Q in 1/Å, wavelength in Å.
- Intensities are counts normalised to the monitor (or time) of
  ``runInfo["normalisation"]``.
- Output files are ``runInfo["file"] + runInfo["ext"][k]`` (k = 0 ``.reg``,
  1 ``.adat``, 2 ``.qdat`` by default), written in the current working directory
  (or wherever ``runInfo["file"]`` points), overwriting existing files.
- Only points with a positive count and non-NaN count·error are written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from toscana.physics.conversions import ang2q


def getDiffA(  # noqa: N802 (legacy API)
    numor: np.ndarray,
    head: Sequence[Mapping[str, Any]],
    runInfo: Mapping[str, Any],
) -> np.ndarray:
    """
    Create diffractograms in angular scale.

    Derived from `legacy/lib/toscana.py::getDiffA`, with bug fixes for the grid
    size, out-of-range bin indices and error combination (see inline comments).

    Algorithm:

    1. Grid. ``w_c = runInfo["angular_range_bank"] / ncell`` is the angular
       width of a cell (0.125° by default). The step ``Δ`` is
       ``runInfo["angular_scale"][2]``, but if it is smaller than ``w_c`` a
       warning is printed and ``Δ = w_c``. The grid is
       ``np.arange(amin, amax, Δ)`` (``amax`` excluded), with ``n_angles``
       points g_k = amin + k·Δ.
    2. Weights. ``M_i`` is the (dead-time corrected) monitor of numor i,
       ``head[i]["MonitorCnts"]``; its share is ``m_i = M_i / Σ_j M_j``.
    3. Distribution. Every cell with non-NaN count and error, centred at θ, is
       represented by the interval [θ − w_c/2, θ + w_c/2]. With
       ``u1 = (θ − w_c/2 − amin)/Δ``, ``u2 = (θ + w_c/2 − amin)/Δ``,
       ``a1 = floor(u1)``, ``a2 = floor(u2)`` and ``r = u1 − a1``, the cell
       contributes the fraction ``f1 = 1 − r`` to bin a1 and ``f2 = r`` to bin a2.
       Contributions with f <= 0 or with a bin index outside [0, n_angles) are
       discarded. The weight is ``w = f·m_i``; for the total (index 0) and for
       the bank of the cell (index ``cell_id // 1000``) the bin accumulates
       Σ w·y, Σ (w·σ)² and Σ w.
    4. Normalisation of every bin of every diffractogram::

           y = Σ w y_i / Σ w,      σ = √(Σ w² σ_i²) / Σ w

       (weighted mean and its error). Bins with Σ w = 0 (or whose total is
       already NaN) get y = σ = NaN.

    The fractions f1, f2 are the exact overlaps of the cell with the bins only
    when Δ = w_c (the default). When Δ > w_c they are computed as if the cell
    were one bin wide: if the cell lies inside one bin (a1 = a2) it gives weight
    f1 + f2 = 1 to that bin (exact); if it straddles a bin edge, the split
    (1 − r, r) is not the true overlap ratio (example: Δ = 0.25°, w_c = 0.125°,
    amin = 0, cell centred at 0.25° gives 0.25/0.75 instead of 0.5/0.5). In all
    cases each cell has total weight m_i (unless part of it is outside the grid).

    Parameters
    ----------
    numor : numpy.ndarray, shape (n_numors, 5, n_cells)
        Stack from ``acquisition.getNumors``: rows 2θ − twotheta0 (degrees),
        counts, errors, Q, cell identifier. Row 3 (Q) is not used.
    head : sequence of dict
        Metadata of every numor. Keys read: ``head[0]["ndet"]``,
        ``head[0]["ncell"]``, ``head[i]["MonitorCnts"]`` for all i.
    runInfo : mapping
        Keys read: ``angular_scale`` (amin, amax, step in degrees),
        ``angular_range_bank`` (degrees covered by one bank; 8.0 by default),
        ``wavelength`` (Å). Not modified.

    Returns
    -------
    numpy.ndarray of float64, shape (10, 5, n_angles)
        First index: 0 = sum of all banks, 1..9 = individual banks (always 10
        entries; for ndet < 9 the unused entries keep 0 in rows 0 and 4 and NaN
        in rows 1 and 2). Second index (rows):

        - 0: angle g_k in degrees = LOWER EDGE of the bin (open item P1: a cell
          centred at g_k + Δ/2 goes entirely to bin k, so the label is shifted by
          −Δ/2 with respect to the bin centre);
        - 1: weighted mean intensity (NaN for empty bins);
        - 2: its error (NaN for empty bins);
        - 3: sum of weights Σ w (not normalised);
        - 4: Q = 4π/λ · sin(g_k/2) in 1/Å, computed from the lower edge.

    Notes
    -----
    Prints a 4-line warning when the requested step is smaller than the cell
    width. Does not log and does not write files.

    Changes from the legacy code (audit 2026-11):

    - C6: the number of angles is taken from the grid itself; legacy used
      ``int(range/step)`` while ``np.arange`` yields ⌈range/step⌉ points, which
      crashed for any step not dividing the range.
    - C7: bin indices use ``floor`` (not truncation toward zero) and are bounds
      checked. Legacy put cells just below ``amin`` in bin 0, cells further below
      (negative index) at the END of the grid, and raised ``IndexError`` for
      cells beyond ``amax``. Out-of-range parts are now discarded.
    - C29: variances (w² σ²) are accumulated and σ = √(Σ w² σ²)/Σ w; legacy
      averaged the errors (Σ w σ / Σ w), so they did not decrease when several
      cells/numors were combined.
    """
    ndet = head[0]["ndet"]  # Number of detectors
    ncell = head[0]["ncell"]  # Number of cells in one detector
    angmin = runInfo["angular_scale"][0]
    angmax = runInfo["angular_scale"][1]
    angstep = runInfo["angular_scale"][2]
    angular_range = runInfo["angular_range_bank"]
    angular_step = angular_range / ncell  # Angular step for 1 cell (in deg)

    if angstep < angular_step:
        print("WARNING: In getDiffA function")
        print("         The angular step should not be smaller than")
        print("         {:6.3f} degrees".format(angular_step))
        print(
            "         Then, it has been changed to {:6.3f} degrees".format(angular_step)
        )
        angstep = angular_step

    # Bug fix vs legacy: the number of angles is taken from the grid itself.
    # Legacy used int(range/step) while np.arange yields ceil(range/step)
    # points, which crashed for any step not dividing the range.
    angle_grid = np.arange(angmin, angmax, angstep)
    nangles = len(angle_grid)

    # Defining the matrix that will contain the 10 diffractograms
    diff = np.zeros((10, 5, nangles))

    # The angular scale is the same for all diffractograms
    for i in range(ndet + 1):
        diff[i, 0, :] = angle_grid

    # The Q-scale is the same for all diffractograms
    for i in range(ndet + 1):
        diff[i, 4, :] = ang2q(diff[i, 0, :], wlength=runInfo["wavelength"])

    # Number of numors to regroup and number of angles in one numor
    nbrNumors = len(numor[:, 0, 0])
    nbrData = len(numor[0, 0, :])

    # This variable will be used for normalisation
    # It is divided by 1000000 just to avoid large numbers
    totalmon = 0
    for num in range(nbrNumors):
        totalmon += head[num]["MonitorCnts"] / 1000000

    for num in range(nbrNumors):
        angle = numor[num, 0, :]
        count = numor[num, 1, :]
        error = numor[num, 2, :]
        cell = numor[num, 4, :]
        mon = head[num]["MonitorCnts"] / 1000000 / totalmon
        for ang in range(nbrData):
            if np.isnan(count[ang]) or np.isnan(error[ang]):
                continue
            angle1 = angle[ang] - angular_step / 2.0
            ang1 = (angle1 - angmin) / angstep
            angle2 = angle[ang] + angular_step / 2.0
            ang2 = (angle2 - angmin) / angstep
            # Bug fix vs legacy: floor (not truncation toward zero) and bounds
            # checks. Legacy used int(), so cells below angmin landed in bin 0
            # or, via negative indices, at the END of the grid; cells beyond
            # angmax raised IndexError. Out-of-range parts are now discarded.
            a1 = int(np.floor(ang1))
            a2 = int(np.floor(ang2))
            frac1 = 1.0 - (ang1 - a1)
            frac2 = ang1 - a1
            det = int(cell[ang] / 1000)
            for a, frac in ((a1, frac1), (a2, frac2)):
                if frac <= 0.0 or a < 0 or a >= nangles:
                    continue
                w = frac * mon
                for d in (0, det):
                    diff[d, 1, a] += w * count[ang]
                    # Bug fix vs legacy: accumulate variances (w^2 e^2), not
                    # w*e; legacy averaged the errors, so they did not
                    # decrease when several cells/numors were combined.
                    diff[d, 2, a] += (w * error[ang]) ** 2
                    diff[d, 3, a] += w

    # Normalisation by the weighting factor of each bin:
    #   y = sum(w y_i) / sum(w),   e = sqrt(sum(w^2 e_i^2)) / sum(w)
    for det in range(10):
        for i in range(len(diff[0, 1, :])):
            if (not np.isnan(diff[0, 1, i])) and (diff[det, 3, i] > 0):
                diff[det, 1, i] = diff[det, 1, i] / diff[det, 3, i]
                diff[det, 2, i] = np.sqrt(diff[det, 2, i]) / diff[det, 3, i]
            else:
                diff[det, 1, i] = np.nan
                diff[det, 2, i] = np.nan

    return diff


def saveDiffAngle(  # noqa: N802 (legacy API)
    diffA: np.ndarray,
    head: Sequence[Mapping[str, Any]],
    runInfo: Mapping[str, Any],
) -> None:
    """
    Write diffractograms in angle scale to CORRECT + D4 formats (legacy behavior).

    Two files are written (overwritten if they exist):

    1. ``runInfo["file"] + runInfo["ext"][1]`` (default ``<file>.adat``), CORRECT
       format, total diffractogram only (``diffA[0]``). Header lines starting with
       ``#``: file name, ``Block 1``, instrument (first two characters of
       ``head[0]["label"]``), user and ``exp_<proposal>/processed``, run numbers
       (the Python repr of ``runInfo["numorLst"]``), spectrum 1, title (= file
       name), run date, X caption with the REQUESTED angular scale and
       ``int((max − min)/step) + 1`` bins, Y caption ``Counts/monitor``,
       ``Histogram : F`` and the number of points written. Then one line per
       point ``angle count error`` (formats ``8.4f``, ``12.8f``, ``12.8f``).
    2. ``runInfo["file"] + runInfo["ext"][0]`` (default ``<file>.reg``), D4
       format. Header: file name, equivalent d4creg command line
       (``runInfo["d4creg_cl"]``), efficiency and shift files, sample
       (``head[0]["subtitle"]``), user, local contact, proposal, run date,
       requested numors and number of numors included, normalisation type, zero
       angle (deg), wavelength (Å), monitor and detector dead times (s), the
       relation ``2theta = 2theta.raw - Zero-angle + shifts``, the angular grid
       and the Q formula. Then, for banks 1 to 9, a block
       ``# Detector d`` with lines ``angle count error Q`` (``8.4f``, ``12.8f``,
       ``12.8f``, ``8.4f``). The total diffractogram (index 0) is NOT in this
       file.

    In both files only points with ``count > 0`` and ``count·error`` not NaN are
    written (±inf values are not filtered).

    Parameters
    ----------
    diffA : numpy.ndarray, shape (10, 5, n_angles)
        Output of `getDiffA` (rows angle, counts, error, weight, Q).
    head : sequence of dict
        Metadata of the numors; keys read from ``head[0]``: ``label``, ``user``,
        ``proposal``, ``subtitle``, ``LC``.
    runInfo : mapping
        Keys read: ``file``, ``ext`` (indices 0 and 1), ``numorLst``,
        ``runDate``, ``angular_scale``, ``d4creg_cl``, ``efffile``, ``decfile``,
        ``nbrNumors``, ``normalisation`` (index 0), ``twotheta0``,
        ``wavelength``, ``dead_time``. ``runDate``, ``nbrNumors`` and
        ``d4creg_cl`` are added by ``workflows.d4creg.d4creg``. Not modified.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If a key above is missing.
    OSError
        If a file cannot be written.

    Notes
    -----
    Prints one line per file written (``"    File <name> (CORRECT format, in
    angle scale)"`` and ``"    File <name> (D4 format, in angle and Q-scale)"``).
    The files are written with the platform default encoding; the ``.reg`` header
    contains the character ``Å``. The bin count in the ``.adat`` header uses the
    requested step even if `getDiffA` enlarged it to the cell width.
    """
    ndata = 0
    for i in range(len(diffA[0, 0, :])):
        count = diffA[0, 1, i]
        error = diffA[0, 2, i]
        if (not np.isnan(count * error)) and (count > 0):
            ndata += 1

    file = runInfo["file"] + runInfo["ext"][1]
    with open(file, "w") as adatfile:
        adatfile.write("# " + file + "\n")
        adatfile.write("#Block  1\n")
        adatfile.write("#========\n")
        adatfile.write("#\n")
        adatfile.write("#Instrument:" + head[0]["label"][0:2] + "\n")
        adatfile.write(
            "#User      :"
            + head[0]["user"].strip(" ")
            + "    exp_"
            + head[0]["proposal"].strip(" ")
            + "/processed"
            + "\n"
        )
        adatfile.write(f"#Run number: {runInfo['numorLst']}\n")
        adatfile.write("#Spectrum  :            1\n")
        adatfile.write("#Title     :" + file + "\n")

        adatfile.write("#Run date  :" + runInfo["runDate"] + "\n")

        mini = runInfo["angular_scale"][0]
        maxi = runInfo["angular_scale"][1]
        step = runInfo["angular_scale"][2]
        bins = int((maxi - mini) / step) + 1
        adatfile.write(
            "#X caption : 2Theta (degrees) binned from {} to {} by {} ({} bins)".format(
                mini, maxi, step, bins
            )
            + "\n"
        )
        adatfile.write("#Y caption : Counts/monitor \n")
        adatfile.write("#Histogram :            F \n")
        adatfile.write("#Points    :         {}   ".format(ndata) + "\n")

        for i in range(len(diffA[0, 0, :])):
            angle = diffA[0, 0, i]
            count = diffA[0, 1, i]
            error = diffA[0, 2, i]
            if (not np.isnan(count * error)) and (count > 0):
                adatfile.write(
                    "{:8.4f}     {:12.8f}     {:12.8f} \n".format(angle, count, error)
                )
    print(
        4 * " "
        + "File "
        + runInfo["file"]
        + runInfo["ext"][1]
        + " (CORRECT format, in angle scale)"
    )

    listNum = " "
    for i in range(len(runInfo["numorLst"])):
        listNum += runInfo["numorLst"][i] + " "

    file = runInfo["file"] + runInfo["ext"][0]
    with open(file, "w") as regfile:
        regfile.write("# " + file + "\n")
        regfile.write("# Equivalent command line: \n")
        regfile.write("#     " + runInfo["d4creg_cl"] + "\n")
        regfile.write("# Efficiency file: " + runInfo["efffile"] + "\n")
        regfile.write("# Shift file: " + runInfo["decfile"] + "\n")
        regfile.write("# Sample: " + head[0]["subtitle"] + "\n")
        regfile.write("# User: " + head[0]["user"].strip(" ") + "\n")
        regfile.write("# Local contact: " + head[0]["LC"].strip(" ") + "\n")
        regfile.write("# Proposal: " + head[0]["proposal"].strip(" ") + "\n")
        regfile.write("# Run date  :" + runInfo["runDate"] + "\n")
        regfile.write(
            "# Requested numors: {}, {} numors included \n".format(
                listNum, runInfo["nbrNumors"]
            )
        )
        regfile.write("# Normalisation type: " + runInfo["normalisation"][0] + "\n")
        regfile.write("# Zero-angle = " + str(runInfo["twotheta0"]) + " deg\n")
        regfile.write("# Wavelength = " + str(runInfo["wavelength"]) + " Å\n")
        regfile.write("# Monitor dead-time = " + str(runInfo["dead_time"][0]) + " s\n")
        regfile.write("# Detector dead-time = " + str(runInfo["dead_time"][1]) + " s\n")
        regfile.write("# 2theta = 2theta.raw - Zero-angle + shifts \n")
        regfile.write(
            "# Binned on angle (from {} to {} in steps of {}, deg), but not on Q \n".format(
                *runInfo["angular_scale"]
            )
        )
        regfile.write(
            "#     Q = 4pi/{}Å * sin(angle/2) \n".format(str(runInfo["wavelength"]))
        )

        for det in range(1, 10):
            regfile.write("# ----------- \n")
            regfile.write("# Detector " + str(det) + "\n")
            regfile.write("# ----------- \n")
            regfile.write("# Angle(deg)     Counts           Error            Q(1/Å)\n")
            for i in range(len(diffA[0, 0, :])):
                angle = diffA[det, 0, i]
                count = diffA[det, 1, i]
                error = diffA[det, 2, i]
                qval = diffA[det, 4, i]
                if (not np.isnan(count * error)) and (count > 0):
                    regfile.write(
                        "{:8.4f}     {:12.8f}     {:12.8f}     {:8.4f} \n".format(
                            angle, count, error, qval
                        )
                    )

    print(
        "    File "
        + runInfo["file"]
        + runInfo["ext"][0]
        + " (D4 format, in angle and Q-scale)"
    )
    return


def saveDiffQ(  # noqa: N802 (legacy API)
    diffQ: np.ndarray,
    head: Sequence[Mapping[str, Any]],
    runInfo: Mapping[str, Any],
) -> None:
    """
    Write diffractograms in Q scale to CORRECT format (legacy behavior).

    Writes ``runInfo["file"] + runInfo["ext"][2]`` (default ``<file>.qdat``,
    overwritten if it exists). Header lines starting with ``#``: file name,
    ``Block 1``, instrument (first two characters of ``head[0]["label"]``), user
    and ``exp_<proposal>/processed``, run numbers (repr of
    ``runInfo["numorLst"]``), spectrum 1, title (= file name), run date, X
    caption with the Q grid ``runInfo["q_scale"]`` (formats ``4.3f``) and
    ``int((max − min)/step) + 1`` bins, Y caption ``Counts/monitor``,
    ``Histogram : F`` and the number of points written. Then one line
    ``Q count error`` per point (formats ``8.4f``, ``12.8f``, ``12.8f``), only for
    points with ``count > 0`` and ``count·error`` not NaN.

    Parameters
    ----------
    diffQ : numpy.ndarray, shape (n, 3)
        Columns Q (1/Å), intensity, error (the rebinned, masked array of
        ``workflows.d4creg.d4creg``).
    head : sequence of dict
        Keys read from ``head[0]``: ``label``, ``user``, ``proposal``.
    runInfo : mapping
        Keys read: ``file``, ``ext`` (index 2), ``numorLst``, ``runDate``,
        ``q_scale`` (min, max, step in 1/Å). Not modified.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If a key above is missing.
    OSError
        If the file cannot be written.

    Notes
    -----
    Prints ``"    File <name> (CORRECT format, in Q scale)"``. Platform default
    encoding; the header contains ``Å``.
    """
    ndata = 0
    for i in range(len(diffQ)):
        count = diffQ[i, 1]
        error = diffQ[i, 2]
        if (not np.isnan(count * error)) and (count > 0):
            ndata += 1

    file = runInfo["file"] + runInfo["ext"][2]
    with open(file, "w") as qdatfile:
        qdatfile.write("# " + file + "\n")
        qdatfile.write("#Block  1\n")
        qdatfile.write("#========\n")
        qdatfile.write("#\n")
        qdatfile.write("#Instrument:" + head[0]["label"][0:2] + "\n")
        qdatfile.write(
            "#User      :"
            + head[0]["user"].strip(" ")
            + "    exp_"
            + head[0]["proposal"].strip(" ")
            + "/processed"
            + "\n"
        )
        qdatfile.write(f"#Run number: {runInfo['numorLst']}\n")
        qdatfile.write("#Spectrum  :            1\n")
        qdatfile.write("#Title     :" + file + "\n")
        qdatfile.write("#Run date  :" + runInfo["runDate"] + "\n")
        mini = runInfo["q_scale"][0]
        maxi = runInfo["q_scale"][1]
        step = runInfo["q_scale"][2]
        bins = int((maxi - mini) / step) + 1
        qdatfile.write(
            "#X caption : Q (1/Å) binned from {:4.3f} to {:4.3f} by {:4.3f} ({} bins)".format(
                mini, maxi, step, bins
            )
            + "\n"
        )
        qdatfile.write("#Y caption : Counts/monitor\n")
        qdatfile.write("#Histogram :            F\n")
        qdatfile.write("#Points    :         {}   ".format(ndata) + "\n")

        for i in range(len(diffQ)):
            q = diffQ[i, 0]
            count = diffQ[i, 1]
            error = diffQ[i, 2]
            if (not np.isnan(count * error)) and (count > 0):
                qdatfile.write(
                    "{:8.4f}     {:12.8f}     {:12.8f} \n".format(q, count, error)
                )
    print(
        "    File "
        + runInfo["file"]
        + runInfo["ext"][2]
        + " (CORRECT format, in Q scale)"
    )
    return
