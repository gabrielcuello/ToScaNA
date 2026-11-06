"""
Per-cell detector corrections of D4 numors.

Position in the processing chain
--------------------------------
Called by ``acquisition.getOneNumor`` for every numor, in this order::

    raw counts N (ndet, ncell)
      -> getErrorsNumor      σ = √N on the RAW counts
      -> getDTC_mcounts      dead-time correction of the monitor
      -> getDTC_dcounts      dead-time correction of every cell (σ scaled alike)
      -> getEff              division of counts and errors by the efficiencies
      -> getNormalFactors + normalise   scaling to a fixed monitor or time
      -> getDec + getAngles  2θ angle and identifier of every cell

The result is then regrouped by ``d4creg_outputs.getDiffA``.

Functions
---------
getDTC_mcounts
    Dead-time corrected monitor counts (non-paralysable model).
getDTC_dcounts
    Dead-time corrected detector counts, cell by cell, and corrected total.
getEff
    Relative efficiency of every cell, from a file (or all ones).
getDec
    Angular shift of every detector bank, from a file (or zeros / built-in list).
getErrorsNumor
    Poisson errors √N of raw counts.
getNormalFactors
    Monitor and counting time with their absolute errors.
normalise
    Scale counts and errors to a reference monitor or time, with error propagation.
getAngles
    2θ angle and identifier of every cell.

Conventions
-----------
- Count arrays have shape (ndet, ncell): row = detector bank (0-based here,
  1-based in files and cell identifiers), column = cell within the bank.
- Times and dead times in seconds; angles and shifts in degrees.
- Efficiency and shift files are read from the path given (relative to the current
  working directory if not absolute); lines containing ``#`` are comments.
- The dead-time functions append a line to ``../logfiles/d4creg.log`` (relative to
  the current working directory, see ``_logfiles.d4creg_log_path``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._logfiles import d4creg_log_path

if TYPE_CHECKING:
    from numpy.typing import NDArray


def getDTC_mcounts(  # noqa: N802 (legacy API)
    monitor: float, time: float, dead_time: float = 2.4e-6
) -> float:
    """
    Calculate the dead-time corrected monitor counts (legacy behavior).

    Non-paralysable dead-time model applied to the mean monitor rate::

        ν_m = monitor / time                      (measured rate, c/s)
        ν   = ν_m / (1 - ν_m τ)                   (true rate)
        returned value = ν · time = monitor / (1 - ν_m τ)

    Parameters
    ----------
    monitor : float
        Measured (raw) monitor counts.
    time : float
        Counting time in s. A value ``<= 0`` is replaced by 1e-6 s (which, for
        any realistic monitor, makes ν_m τ >= 1 and therefore raises).
    dead_time : float, optional
        Monitor dead time τ in s. Default 2.4e-6 s (D4 monitor; the value used
        by ``getOneNumor`` is ``runInfo["dead_time"][0]``).

    Returns
    -------
    float
        Dead-time corrected monitor counts (same counting time).

    Raises
    ------
    ValueError
        If ``1 - ν_m τ <= 0``, i.e. ν_m τ >= 1 (message
        ``"getDTC_mcounts: correction factor < 0"``).

    Notes
    -----
    Appends to ``../logfiles/d4creg.log`` either the line
    ``"  Monitor DT = <τ> s. Correction factor = <1/(1-ν_m τ)>"`` or, before
    raising, ``"--- ERROR in Function getDTC_mcounts: correction factor < 0"``.
    Prints nothing.

    See Also
    --------
    getDTC_dcounts : same model for the detector cells.
    """
    if time <= 0:
        time = 1e-6
    nu_meas = monitor / time
    correction = 1.0 - nu_meas * dead_time
    if correction > 0:
        factor = 1.0 / correction
        nu = nu_meas * factor
        with d4creg_log_path().open("a", encoding="utf-8") as file:
            file.write(
                "  Monitor DT = {:4.2e} s. Correction factor = {:10.6f} \n".format(
                    dead_time, factor
                )
            )
        return float(nu * time)

    with d4creg_log_path().open("a", encoding="utf-8") as file:
        file.write("--- ERROR in Function getDTC_mcounts: correction factor < 0 \n")
    raise ValueError("getDTC_mcounts: correction factor < 0")


def getDTC_dcounts(  # noqa: N802 (legacy API)
    counts: NDArray[np.float64],
    total_counts: float,
    time: float,
    dead_time: float = 7e-6,
) -> tuple[NDArray[np.float64], float]:
    """
    Calculate the dead-time corrected detector counts (non-paralysable model, per cell).

    For every cell (d, c)::

        ν_m(d, c)   = counts(d, c) / time            (measured rate, c/s)
        k(d, c)     = 1 / (1 - ν_m(d, c) τ)          (correction factor)
        N_corr(d, c) = k(d, c) · counts(d, c)

    The total detector counts are scaled by the overall factor
    ``K = Σ N_corr / Σ counts`` (K = 1 if Σ counts <= 0).

    Parameters
    ----------
    counts : numpy.ndarray, shape (ndet, ncell)
        Raw counts of every cell.
    total_counts : float
        Total detector counts of the numor as recorded in the file
        (``head["TotalCnts"]``).
    time : float
        Counting time in s. A value ``<= 0`` is replaced by 1e-6 s.
    dead_time : float, optional
        Detector dead time τ in s. Default 7e-6 s (the value used by
        ``getOneNumor`` is ``runInfo["dead_time"][1]``).

    Returns
    -------
    corrected : numpy.ndarray of float64, shape (ndet, ncell)
        Dead-time corrected counts. The per-cell factor k can be recovered as
        ``corrected / counts`` (this is how ``getOneNumor`` scales the errors).
    total : float
        ``total_counts · K``.

    Raises
    ------
    ValueError
        If any factor ``1 - ν_m τ`` is not strictly positive, i.e. some cell has
        ν_m τ >= 1, or if ``counts`` contains NaN (message
        ``"getDTC_dcounts: correction factor < 0"``).

    Notes
    -----
    Appends to ``../logfiles/d4creg.log`` either
    ``"  Detector DT = <τ> s. Correction factor = <K> (overall), max = <max k>"``
    or, before raising, ``"--- ERROR in Function getDTC_dcounts: correction factor
    < 0"``. Prints nothing.

    Whether the dead time should apply to the rate of each cell or of the whole
    bank is an open item of the audit (P2).

    Changes from the legacy code (audit 2026-11, C4) -- divergences from legacy
    (bug fixes):

    - Legacy tested `(correction.all()) > 0`, which only checks that no factor is
      exactly zero; negative factors (rate * tau > 1) were silently accepted and
      produced negative counts. All factors are now required to be positive.
    - Legacy corrected `total_counts` with the factor of cell [0, 0] only. The
      total is now scaled by sum(corrected) / sum(raw), consistent with the
      per-cell correction.
    - The log message said "Monitor DT" for the detector; it now says
      "Detector DT".
    """
    if time <= 0:
        time = 1e-6
    nu_meas = counts / time
    correction = 1.0 - nu_meas * dead_time

    if np.all(correction > 0):
        factor = 1.0 / correction
        corrected = (nu_meas * factor * time).astype(np.float64, copy=False)
        raw_sum = float(np.sum(counts))
        total_factor = float(np.sum(corrected)) / raw_sum if raw_sum > 0 else 1.0
        with d4creg_log_path().open("a", encoding="utf-8") as file:
            file.write(
                "  Detector DT = {:4.2e} s. Correction factor = {:10.6f}".format(
                    dead_time, total_factor
                )
                + " (overall), max = {:10.6f}\n".format(float(np.max(factor)))
            )
        return corrected, float(total_counts * total_factor)

    with d4creg_log_path().open("a", encoding="utf-8") as file:
        file.write("--- ERROR in Function getDTC_dcounts: correction factor < 0 \n")
    raise ValueError("getDTC_dcounts: correction factor < 0")


def getEff(  # noqa: N802 (legacy API)
    filename: str = "effd4c.eff", ndet: int = 9, ncells: int = 64
) -> NDArray[np.float64]:
    """
    Generate the relative efficiencies (legacy behavior).

    Starts from an array of ones and overwrites the cells listed in the file.
    Each non-comment line of the file is ``det cell efficiency`` (white-space
    separated; ``det`` and ``cell`` are 1-based integers; extra columns are
    ignored). Lines containing ``#`` anywhere and empty lines are skipped. An
    efficiency ``<= 0`` is stored as NaN: after the division in ``getOneNumor``
    the counts of that cell are NaN and the cell is ignored by the regrouping
    (``getDiffA``). If a cell appears several times, the last line wins.

    Parameters
    ----------
    filename : str, optional
        Path of the efficiency file (``runInfo["efffile"]``, tag ``<eff>``).
        Default ``"effd4c.eff"`` in the current working directory. The special
        value ``"ones"`` skips the file and returns all ones.
    ndet : int, optional
        Number of detector banks (rows). Default 9 (D4).
    ncells : int, optional
        Number of cells per bank (columns). Default 64 (D4).

    Returns
    -------
    numpy.ndarray of float64, shape (ndet, ncells)
        Relative efficiency of every cell (dimensionless); 1 for cells not in the
        file, NaN for cells with efficiency <= 0.

    Raises
    ------
    FileNotFoundError
        If ``filename`` (other than ``"ones"``) does not exist.
    ValueError
        If a field cannot be converted to int/float.
    IndexError
        If a line has fewer than 3 columns, or ``det > ndet`` or
        ``cell > ncells``.

    Notes
    -----
    Prints ``"No relative efficiencies"`` when ``filename == "ones"``.

    A ``det`` or ``cell`` value of 0 (or negative) is NOT rejected: the 0-based
    index becomes negative and, by Python indexing, the value is silently stored
    in a cell counted from the end of the array.

    Changes from the legacy code (audit 2026-11, C27): legacy removed the last
    character of each line (``[0:-1]``), so a final line without newline lost a
    digit (1.0234 -> 1.023). Lines are now stripped.
    """
    efficiencies: NDArray[np.float64] = np.ones((ndet, ncells), dtype=np.float64)
    if filename == "ones":
        print("No relative efficiencies")
        return efficiencies

    with open(filename, "r") as shifts:
        lines = shifts.readlines()

    for line in lines:
        if "#" in line:
            continue
        # Bug fix vs legacy: `[0:-1]` dropped the last character of a final line
        # without a trailing newline (e.g. 1.0234 -> 1.023).
        row = line.strip()
        if len(row) == 0:
            continue
        col = row.split()
        if float(col[2]) <= 0:
            col[2] = "nan"
        efficiencies[int(col[0]) - 1, int(col[1]) - 1] = float(col[2])

    return efficiencies


def getDec(filename: str = "dec.dec", ndet: int = 9) -> NDArray[np.float64]:  # noqa: N802 (legacy API)
    """
    Read the angular shifts for each detection bank (legacy behavior).

    Each non-comment line of the file is ``det shift``: 1-based bank number and
    angular shift in degrees (white-space separated; extra columns ignored).
    Lines containing ``#`` anywhere and empty lines are skipped. Banks not
    listed get 0. The shift of bank d is added to the angle of all its cells by
    ``getAngles``.

    Parameters
    ----------
    filename : str, optional
        Path of the shift file (``runInfo["decfile"]``, tag ``<dec>``). Default
        ``"dec.dec"`` in the current working directory. Special values:

        - ``"zeros"``: no file read; returns ``ndet`` zeros;
        - ``"manual"``: no file read; returns the built-in D4 shifts
          ``[0.000, -0.027, -0.026, -0.087, -0.125, -0.160, -0.220, -0.250,
          -0.340]`` (always 9 values, whatever ``ndet``).
    ndet : int, optional
        Number of detector banks. Default 9 (D4).

    Returns
    -------
    numpy.ndarray of float64, shape (ndet,)
        Angular shift of every bank, in degrees (shape (9,) for ``"manual"``).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a field cannot be converted to int/float.
    IndexError
        If a line has a single column or ``det > ndet``.

    Notes
    -----
    Prints ``"No angular shifts"`` (``"zeros"``) or ``"Built-in shifts:"``
    (``"manual"``; the values themselves are not printed).

    As in `getEff`, a bank number <= 0 is not rejected and is stored, through
    negative indexing, in a bank counted from the end.

    Changes from the legacy code (audit 2026-11, C27): lines are stripped instead
    of losing their last character (see `getEff`).
    """
    if filename == "zeros":
        print("No angular shifts")
        return np.zeros(ndet, dtype=np.float64)
    if filename == "manual":
        print("Built-in shifts:")
        return np.array(
            [0.000, -0.027, -0.026, -0.087, -0.125, -0.160, -0.220, -0.250, -0.340],
            dtype=np.float64,
        )

    zero = np.zeros(ndet, dtype=np.float64)
    with open(filename, "r") as shifts:
        lines = shifts.readlines()
    for line in lines:
        if "#" in line:
            continue
        row = line.strip()  # bug fix vs legacy: see getEff
        if len(row) == 0:
            continue
        columns = row.split()
        zero[int(columns[0]) - 1] = float(columns[1])
    return zero


def getErrorsNumor(counts: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N802 (legacy API)
    """
    Calculate Poisson errors for detector counts.

    σ = √N, element by element.

    Must be called on RAW counts; the errors are then scaled by the same factors
    as the counts (see `getOneNumor`).

    Parameters
    ----------
    counts : numpy.ndarray
        Raw counts, any shape (normally (ndet, ncell)).

    Returns
    -------
    numpy.ndarray of float64, same shape as ``counts``
        √counts. Negative counts give NaN (with a NumPy RuntimeWarning).

    Notes
    -----
    Changes from the legacy code (audit 2026-11, C5): legacy took √N of the
    counts already corrected for dead time and efficiency, i.e.
    √(N_raw/ε) instead of √N_raw/ε. The function itself is unchanged; the fix is
    in the order of the calls in ``acquisition.getOneNumor``.
    """
    errors = np.sqrt(counts)
    return errors.astype(np.float64, copy=False)


def getNormalFactors(monitor: float, time: float) -> NDArray[np.float64]:  # noqa: N802 (legacy API)
    """
    Define normalisation factors (legacy behavior).

    Parameters
    ----------
    monitor : float
        Monitor counts M of the numor (dead-time corrected in ``getOneNumor``).
    time : float
        Counting time t in s.

    Returns
    -------
    numpy.ndarray of float64, shape (2, 2)
        ``[[M, √M], [t, 0.01]]``: row 0 = monitor and its absolute (Poisson)
        error; row 1 = counting time and its absolute error, fixed to 0.01 s.
        Row index 0/1 is the ``ntype`` used by `normalise`.
    """
    normal = np.zeros((2, 2), dtype=np.float64)
    normal[0][0] = monitor
    normal[0][1] = np.sqrt(monitor)
    normal[1][0] = time
    normal[1][1] = 0.01
    return normal


def normalise(  # noqa: N802 (legacy API)
    counts: NDArray[np.float64],
    errors: NDArray[np.float64],
    normal: NDArray[np.float64],
    norm_type: str = "monitor",
    norm_time: float = 120,
    norm_mon: float = 1_000_000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalise counts and errors by monitor counts or counting time.

    With N and σ_N the normaliser and its error taken from ``normal`` (row 0 for
    monitor, row 1 for time) and ``norm`` the reference value (``norm_mon`` or
    ``norm_time``)::

        f   = norm / N
        y_n = f · y
        σ_n = √[ (f σ)² + (y_n σ_N / N)² ]

    i.e. the statistical error of the cell and the relative error of the
    normaliser are added in quadrature. Cells whose INPUT counts are exactly 0
    get NaN for both y_n and σ_n. NaN inputs (e.g. cells with NaN efficiency)
    stay NaN.

    Parameters
    ----------
    counts : numpy.ndarray
        Counts y (already corrected for dead time and efficiency), any shape.
    errors : numpy.ndarray
        Absolute errors σ of ``counts``, same shape.
    normal : numpy.ndarray, shape (2, 2)
        Normalisers as returned by `getNormalFactors`:
        ``[[M, σ_M], [t, σ_t]]``.
    norm_type : str, optional
        ``"time"`` normalises to counting time; ANY other value (default
        ``"monitor"``, but also a misspelling) normalises to monitor.
    norm_time : float, optional
        Reference counting time in s used when ``norm_type == "time"``.
        Default 120 s. (``getOneNumor`` passes ``runInfo["normalisation"][1]``,
        default 80 s from ``readParam``.)
    norm_mon : float, optional
        Reference monitor counts used otherwise. Default 1 000 000
        (``runInfo["normalisation"][2]``).

    Returns
    -------
    counts_n : numpy.ndarray of float64
        Normalised counts y_n: counts per ``norm_mon`` monitor counts (or per
        ``norm_time`` s); NaN where the input counts are 0.
    errors_n : numpy.ndarray of float64
        Absolute errors σ_n of ``counts_n``; NaN where the input counts are 0.

    Notes
    -----
    No output, no files. A zero normaliser N gives inf/NaN (NumPy warnings),
    not an exception.

    Changes from the legacy code (audit 2026-11, C2) -- bug fix vs legacy:
    legacy formed `errors / counts` AFTER `counts` had been normalised while
    `errors` had not, so the relative error was multiplied by monitor/norm_mon.
    The variance is now propagated as
    var_n = (f * e)^2 + (y_n * sigma_N / N)^2,   f = norm / N,  y_n = f * y.
    Cells with zero counts are flagged NaN (excluded downstream), which is the
    effective legacy behaviour (legacy produced NaN errors via 0/0).
    """
    norm = [norm_mon, norm_time]
    ntype = 0
    if norm_type == "time":
        ntype = 1
    factor = norm[ntype] / normal[ntype][0]
    rel_norm = normal[ntype][1] / normal[ntype][0]
    counts_n = factor * counts
    errors_n = np.sqrt((factor * errors) ** 2 + (counts_n * rel_norm) ** 2)
    zero = counts == 0
    counts_n = np.where(zero, np.nan, counts_n).astype(np.float64, copy=False)
    errors_n = np.where(zero, np.nan, errors_n).astype(np.float64, copy=False)
    return counts_n, errors_n


def getAngles(  # noqa: N802 (legacy API)
    counts: NDArray[np.float64],
    ttheta_ref: float,
    zeros: NDArray[np.float64],
    cell_step: float = 0.125,
    det_step: float = 15.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Determine angles and cell identifiers for each detector/cell (legacy behavior).

    For bank ``d`` and cell ``c`` (both 0-based)::

        2θ(d, c)  = ttheta_ref + zeros[d] + d · det_step + c · cell_step
        id(d, c)  = 1000 · (d + 1) + (c + 1)

    so that ``id // 1000`` is the 1-based bank and ``id % 1000`` the 1-based cell
    (e.g. 1001 = bank 1, cell 1; 9064 = bank 9, cell 64).

    Parameters
    ----------
    counts : numpy.ndarray, shape (ndet, ncell)
        Only its shape is used.
    ttheta_ref : float
        2θ of the first cell of the first bank, in degrees (``getOneNumor``
        passes the read angle ``head["Theta_lue(deg)"]``).
    zeros : numpy.ndarray, shape (ndet,) or longer
        Angular shift of every bank in degrees (from `getDec`).
    cell_step : float, optional
        Angular width of one cell in degrees. Default 0.125° (D4: 8°/64 cells).
        ``getOneNumor`` uses the default, independently of
        ``runInfo["angular_range_bank"]``.
    det_step : float, optional
        Angular distance between consecutive banks in degrees. Default 15°.

    Returns
    -------
    angles : numpy.ndarray of float64, shape (ndet, ncell)
        2θ of every cell in degrees. The zero-angle correction ``twotheta0`` is
        NOT applied here (it is subtracted in ``getNumors``).
    cells : numpy.ndarray of float64, shape (ndet, ncell)
        Cell identifiers ``1000 · bank + cell`` (1-based), stored as floats.

    Raises
    ------
    IndexError
        If ``zeros`` has fewer than ``ndet`` elements.
    """
    angles = np.zeros((counts.shape[0], counts.shape[1]), dtype=np.float64)
    cells = np.zeros((counts.shape[0], counts.shape[1]), dtype=np.float64)
    ndet = counts.shape[0]
    ncel = counts.shape[1]
    for det in range(ndet):
        for cel in range(ncel):
            angles[det][cel] = (
                ttheta_ref + zeros[det] + (det * det_step + cel * cell_step)
            )
            cells[det][cel] = (det + 1) * 1000 + cel + 1
    return angles, cells
