"""
Reading of D4 numors and per-numor corrections.

Position in the processing chain
--------------------------------
First stage of the d4creg regrouping (``workflows.d4creg.d4creg``)::

    runInfo["numorLst"] --getNumorFiles--> numors
      for each numor: getOneNumor
          read (load_nxs for NeXus, readD4numorASCII for ILL ASCII)
          -> angle check -> σ = √N -> dead time -> efficiency -> normalisation
          -> angles and cell identifiers
      getNumors: stack -> array (n_numors, 5, ndet*ncell) -> d4creg_outputs.getDiffA

Functions
---------
printRates
    Log the counts and counting rates (per s and per MW) of the monitor or detector.
getRefLines
    Locate the R, A, F and S marker lines of an ILL ASCII numor.
readD4numorASCII
    Read one numor in ILL ASCII format: metadata dictionary and counts (ndet, ncell).
load_nxs
    Read one numor in NeXus format (requires h5py): metadata and counts (9, ncell).
getOneNumor
    Read one numor and apply all per-cell corrections; return flat lists.
getNumors
    Read all numors of ``runInfo`` and stack them in one 3-D array.

Private helper
--------------
_parse_float_block
    Parse one ``F`` (floating-point parameters) block of an ASCII numor.

Conventions
-----------
- Angles in degrees (2θ), times in s, dead times in s, Q in 1/Å, wavelength in Å.
- Metadata dictionaries (``head``) use the ILL parameter labels as keys, e.g.
  ``"CntTime (sec)"``, ``"MonitorCnts"``, ``"TotalCnts"``, ``"Theta_lue(deg)"``
  (read 2θ of the first cell), ``"Theta_des(deg)"`` (requested 2θ),
  ``"RtrPower (MW)"``, plus ``numor``, ``label``, ``user``, ``LC``, ``title``,
  ``subtitle``, ``proposal``, ``startTime``, ``endTime``, ``ndet``, ``ncell``.
- Raw data files are ``runInfo["path_raw"] + numor`` (+ ``runInfo["ext"][4]``,
  default ``".nxs"``, for NeXus), relative to the current working directory.
- Counting rates, dead-time factors and angle warnings are appended to
  ``../logfiles/d4creg.log`` (relative to the current working directory).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..physics.conversions import ang2q
from ._logfiles import d4creg_log_path
from .corrections import (
    getAngles,
    getDec,
    getDTC_dcounts,
    getDTC_mcounts,
    getEff,
    getErrorsNumor,
    getNormalFactors,
    normalise,
)
from .numors import getNumorFiles
from .saving import saveOneNumor

if TYPE_CHECKING:
    from numpy.typing import NDArray


def printRates(  # noqa: N802 (legacy API)
    detector: str, counts: float, time: float, reactor_power: float = 52
) -> None:
    """
    Print out the basic information about counting rates (legacy behavior).

    Writes two lines to the log file::

        "  Counts on <detector> = <counts>  ===> counting-rate = <counts/time> c/s"
        "  Reactor-power normalised <detector> counting-rate = <counts/time/P> c/s/MW"

    Parameters
    ----------
    detector : str
        Label inserted in the messages, e.g. ``"monitor"`` or ``"detector"``.
    counts : float
        Number of counts.
    time : float
        Counting time in s. A value ``<= 0`` is replaced by 1e-6 s.
    reactor_power : float, optional
        Reactor power P in MW. Default 52 MW (nominal ILL power). A value
        ``<= 0`` is replaced by 52. NeXus numors give 1 MW (see `load_nxs`).

    Returns
    -------
    None

    Notes
    -----
    Legacy writes to `../logfiles/d4creg.log` and does not print to stdout.
    The log file is opened twice in append mode (created if missing, see
    ``_logfiles.d4creg_log_path``).
    """
    if time <= 0:
        time = 1e-6

    out = "  Counts on {} = {:10.8g}  ===> counting-rate = {:10.8g} c/s".format(
        detector, counts, counts / time
    )
    with d4creg_log_path().open("a", encoding="utf-8") as file:
        file.write(out + "\n")

    if reactor_power <= 0:
        reactor_power = 52
    out = "  Reactor-power normalised {} counting-rate = {:10.6g} c/s/MW".format(
        detector, counts / time / reactor_power
    )
    with d4creg_log_path().open("a", encoding="utf-8") as file:
        file.write(out + "\n")
    return


def getRefLines(Lines: list[str]) -> tuple[int, int, int, int, int, int, int]:  # noqa: N802 (legacy API)
    """
    Determine the reference lines in the ASCII numor files (for D4).

    An ILL ASCII numor is divided in blocks, each starting with a line whose first
    10 characters are the same letter: ``RRRRRRRRRR`` (run), ``AAAAAAAAAA``
    (text), ``FFFFFFFFFF`` (floating-point parameters), ``SSSSSSSSSS`` (spectrum,
    i.e. counts). This function returns the 0-based indices of those marker lines.

    Parameters
    ----------
    Lines : list of str
        The lines of the file, as returned by ``readlines()``.

    Returns
    -------
    tuple of 7 int
        `(lineR, lineAL, lineAI, lineFD, lineFS, lineFM, lineSI)` matching legacy:

        - ``lineR``: the LAST ``R`` marker line (each match overwrites the
          previous one; a D4 numor has only one);
        - ``lineAL``, ``lineAI``: first and second ``A`` marker lines (label
          block, then the user/title/time block);
        - ``lineFD``, ``lineFS``, ``lineFM``: first, second and third ``F``
          marker lines (the three blocks of float parameters);
        - ``lineSI``: first ``S`` marker line (counts block).

    Raises
    ------
    ValueError
        If there is no ``R`` marker, fewer than 2 ``A``, fewer than 3 ``F`` or no
        ``S`` marker.
    """
    FS: list[int] = []
    AS: list[int] = []
    SS: list[int] = []
    lineR = -1

    for i, line in enumerate(Lines):
        if line[0:10] == "RRRRRRRRRR":
            lineR = i
        if line[0:10] == "FFFFFFFFFF":
            FS.append(i)
        if line[0:10] == "AAAAAAAAAA":
            AS.append(i)
        if line[0:10] == "SSSSSSSSSS":
            SS.append(i)

    if lineR < 0 or len(AS) < 2 or len(FS) < 3 or len(SS) < 1:
        raise ValueError(
            "Invalid ASCII numor file: could not locate reference markers."
        )

    return lineR, AS[0], AS[1], FS[0], FS[1], FS[2], SS[0]


def _parse_float_block(
    lines: list[str],
    data: dict[str, Any],
    start_line: int,
    *,
    cdel: int,
) -> None:
    """
    Parse one legacy 'F' float block into `data`.

    The block layout mirrors `legacy/lib/toscana.py::readD4numorASCII`:

    - line ``start_line`` is the ``FFFFFFFFFF`` marker;
    - on line ``start_line + 1`` (last character dropped, then split on white
      space) the SECOND integer gives ``ldel + 1``;
    - from line ``start_line + 3`` there are ``ldel`` lines of labels followed by
      ``ldel`` lines of values. Each line holds 5 fields of ``cdel`` characters;
      field j is taken as characters ``[1 + j·cdel : (j + 1)·cdel]`` (the first
      character of the line is skipped) and left-stripped.

    For every field, ``data[label] = float(value)``; a value that cannot be
    converted to float (including an empty field) is stored as 0.0. Trailing
    spaces of the labels are kept, so the keys are exactly the right-aligned ILL
    labels (e.g. ``"CntTime (sec)"``).

    Parameters
    ----------
    lines : list of str
        All lines of the numor file.
    data : dict
        Metadata dictionary, MODIFIED IN PLACE (one key added per field;
        existing keys with the same label are overwritten).
    start_line : int
        Index of the ``F`` marker line (from `getRefLines`).
    cdel : int, keyword-only
        Width of a field in characters (16 for D4).

    Returns
    -------
    None

    Raises
    ------
    IndexError, ValueError
        If the block is shorter than announced or the count line is malformed.
    """
    lref = start_line + 3
    col = (lines[start_line + 1].strip(" ")[:-1]).split()
    ldel = int(col[1]) - 1

    for i in range(ldel):
        for j in range(5):
            label = (lines[lref + i][1 + j * cdel : (j + 1) * cdel]).lstrip(" ")
            raw = (lines[lref + ldel + i][1 + j * cdel : (j + 1) * cdel]).lstrip(" ")
            try:
                parameter = float(raw)
            except Exception:
                parameter = float(0.0)
            data[label] = parameter


def readD4numorASCII(  # noqa: N802 (legacy API)
    filename: str,
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """
    Read one numor in ASCII format (legacy behavior).

    The marker lines are located with `getRefLines` (indices R, AL, AI, FD, FS,
    FM, SI below); the metadata are read at fixed positions:

    - ``numor``: first token of line R+1, zero-padded to 6 characters;
    - ``label``: line AL+2 without its last character (newline);
    - ``user``, ``LC`` (local contact): lines AI+2, AI+3, left-stripped, then
      characters [20:-1];
    - ``title``, ``subtitle``, ``proposal``: lines AI+4, AI+5, AI+6,
      characters [20:-1];
    - ``startTime``: line AI+7, characters [20:38]; ``endTime``: line AI+7,
      characters [60:79];
    - every parameter of the three float blocks (`_parse_float_block` with field
      width 16), e.g. ``"CntTime (sec)"``, ``"MonitorCnts"``, ``"TotalCnts"``,
      ``"Theta_lue(deg)"``, ``"Theta_des(deg)"``, ``"RtrPower (MW)"``;
    - ``ndet``: third token of line SI+1; ``ncell``: line SI+3.

    Counts: for bank ``det`` (0-based) the values are read 10 per line from the
    ``ncell // 10`` lines starting at SI+4+11·det, then the remaining values from
    the next line (SI+4+ncell//10+11·det). Consecutive banks are 11 lines apart.

    Parameters
    ----------
    filename : str
        Path of the numor file (no extension is added), UTF-8.

    Returns
    -------
    (data, counts)
        `data` is a mixed-type metadata dict (strings for the text fields,
        floats for the parameters, int for ``ndet`` and ``ncell``); `counts` is an
        (ndet, ncell) float64 array of raw counts (row = bank, column = cell).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a marker is missing (`getRefLines`) or a count is not numeric.
    KeyError
        If ``"CntTime (sec)"``, ``"Theta_lue(deg)"``, ``"MonitorCnts"`` or
        ``"TotalCnts"`` is not among the parameters.
    IndexError
        If the layout differs from the one described above.

    Notes
    -----
    Prints a summary (numor, subtitle, start/end times, counting time, angle of
    the first cell of the first bank) and an empty line. Logs the raw monitor and
    detector rates with `printRates` (reactor power ``"RtrPower (MW)"``, 52 MW if
    absent).

    The layout of 10 values per line and 11 lines per bank (open item P16) has
    only been checked for ncell = 64; when ``ncell`` is a multiple of 10 the
    "remaining values" line is still read.
    """
    with open(filename, "r", encoding="utf-8") as rawdata:
        Lines = rawdata.readlines()

    lineR, lineAL, lineAI, lineFD, lineFS, lineFM, lineSI = getRefLines(Lines)
    cdel = 16

    data: dict[str, Any] = {}
    col = (Lines[lineR + 1].strip(" ")[:-1]).split()
    data["numor"] = str(col[0]).zfill(6)
    data["label"] = Lines[lineAL + 2][:-1]
    data["user"] = Lines[lineAI + 2].strip(" ")[20:-1]
    data["LC"] = Lines[lineAI + 3].strip(" ")[20:-1]
    data["title"] = Lines[lineAI + 4][20:-1]
    data["subtitle"] = Lines[lineAI + 5][20:-1]
    data["proposal"] = Lines[lineAI + 6][20:-1]
    data["startTime"] = Lines[lineAI + 7][20:38]
    data["endTime"] = Lines[lineAI + 7][60:79]

    _parse_float_block(Lines, data, lineFD, cdel=cdel)
    _parse_float_block(Lines, data, lineFS, cdel=cdel)
    _parse_float_block(Lines, data, lineFM, cdel=cdel)

    lref = lineSI
    col = (Lines[lineSI + 1].strip(" ")[:-1]).split()
    ndet = int(col[2])
    data["ndet"] = ndet
    ncell = int(Lines[lref + 3].strip(" ")[:-1])
    data["ncell"] = ncell

    counts: NDArray[np.float64] = np.zeros((ndet, ncell), dtype=np.float64)

    for det in range(ndet):
        for j in range(int(ncell / 10)):
            col = (Lines[lref + 4 + j + 11 * det].strip(" ")[:-1]).split()
            for i in range(len(col)):
                counts[det, i + 10 * j] = float(col[i])

        col = (Lines[lref + 4 + int(ncell / 10) + 11 * det].strip(" ")[:-1]).split()
        for i in range(len(col)):
            counts[det, i + 10 * int(ncell / 10)] = float(col[i])

    print("Numor {} loaded without errors.".format(data["numor"]))
    print(2 * " ", data["subtitle"])
    print("  Starts: {}  Ends: {}".format(data["startTime"], data["endTime"]))
    print("  Counting time ={:10.6g} s".format(data["CntTime (sec)"]))
    print("  Angle (1st cell 1st det) ={:10.6g} degrees".format(data["Theta_lue(deg)"]))
    printRates(
        "monitor",
        float(data["MonitorCnts"]),
        float(data["CntTime (sec)"]),
        float(data.get("RtrPower (MW)", 52.0)),
    )
    printRates(
        "detector",
        float(data["TotalCnts"]),
        float(data["CntTime (sec)"]),
        float(data.get("RtrPower (MW)", 52.0)),
    )
    print()

    return data, counts


def load_nxs(  # noqa: N802 (legacy API)
    nxs_path: str, ncell: int = 64
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    """
    Read one numor in Nexus format (legacy behavior).

    Reads the HDF5 group ``entry0`` with h5py and builds a metadata dictionary
    with the same keys as `readD4numorASCII` where they exist:

    ========================  ==================================================
    key                       source in ``entry0`` (first element)
    ========================  ==================================================
    ``label``                 ``"d4 " + user/name + user/namelocalcontact +
                              start_time`` (concatenated without separators)
    ``user``, ``LC``          ``user/name``, ``user/namelocalcontact``
    ``proposal``              ``user/proposal``
    ``ndet``, ``ncell``       9 (fixed), argument ``ncell``
    ``RtrPower (MW)``         1 (fixed; open item P8)
    ``numor``                 ``str(run_number)`` (not zero-padded)
    ``subtitle``, ``title``   ``experiment_identifier``, ``title``
    ``startTime``/``endTime`` ``start_time``, ``end_time``
    ``MonitorCnts``           ``monitor/monsum``
    ``CntTime (sec)``         ``time``
    ``TotalSteps``            ``data_scan/total_steps``
    ``Theta_lue(deg)``        ``instrument/masked2theta_s/value``
    ``Theta_des(deg)``        ``instrument/masked2theta_s/target_value``
    ``Omega(deg)``            ``instrument/omega/value``
    ========================  ==================================================

    Then every scanned variable (names in
    ``data_scan/scanned_variables/variables_names/property``, values in
    ``data_scan/scanned_variables/data``) is added with its first value; the
    names ``TotalAcquisitionSpy`` and ``TotalCount`` are renamed ``TotalCnts``
    (if both exist, the last one wins). The counts are
    ``data_scan/detector_data/data[0]``, squeezed and reshaped to (9, ncell).

    Parameters
    ----------
    nxs_path : str
        Path of the ``.nxs`` file, including the extension.
    ncell : int, optional
        Number of cells per bank. Default 64 (D4).

    Returns
    -------
    metadata : dict
        Metadata as above (text decoded from UTF-8; numbers are NumPy scalars).
    counts : numpy.ndarray of float64, shape (9, ncell)
        Raw counts (row = bank, column = cell).

    Raises
    ------
    ImportError
        If h5py is not installed.
    FileNotFoundError, OSError
        If the file cannot be opened as HDF5.
    KeyError
        If one of the fields above is missing.
    ValueError
        If the counts cannot be reshaped to (9, ncell).

    Notes
    -----
    Prints and logs nothing (unlike `readD4numorASCII`). If the file has no
    ``TotalAcquisitionSpy``/``TotalCount`` scanned variable, ``TotalCnts`` is
    absent and `getOneNumor` fails with ``KeyError``.
    """
    try:
        import h5py  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover
        raise ImportError("load_nxs requires h5py to be installed.") from exc

    with h5py.File(nxs_path, "r") as nxs:
        entry0 = nxs["entry0"]

        metadata: dict[str, Any] = {
            "label": "d4 "
            + entry0["user/name"][0].decode("utf-8")
            + entry0["user/namelocalcontact"][0].decode("utf-8")
            + entry0["start_time"][0].decode("utf-8"),
            "user": entry0["user/name"][0].decode("utf-8"),
            "LC": entry0["user/namelocalcontact"][0].decode("utf-8"),
            "proposal": entry0["user/proposal"][0].decode("utf-8"),
            "ndet": 9,
            "ncell": ncell,
            "RtrPower (MW)": 1,
            "numor": str(entry0["run_number"][0]),
            "subtitle": entry0["experiment_identifier"][0].decode("utf-8"),
            "title": entry0["title"][0].decode("utf-8"),
            "startTime": entry0["start_time"][0].decode("utf-8"),
            "endTime": entry0["end_time"][0].decode("utf-8"),
            "MonitorCnts": entry0["monitor/monsum"][0],
            #"MonitorCnts": entry0["monitor/data"][0, 0, 0],
            "CntTime (sec)": entry0["time"][0],
            "TotalSteps": entry0["data_scan/total_steps"][0],
            "Theta_lue(deg)": entry0["instrument/masked2theta_s/value"][0],
            "Theta_des(deg)": entry0["instrument/masked2theta_s/target_value"][0],
            # "Theta_lue(deg)": entry0["instrument/2theta/value"][0],
            # "Theta_des(deg)": entry0["instrument/2theta/target_value"][0],
            "Omega(deg)": entry0["instrument/omega/value"][0],
            #"A1": entry0["instrument/A1/value"][0],
        }

        for name, datum in zip(
            entry0["data_scan/scanned_variables/variables_names/property"],
            entry0["data_scan/scanned_variables/data"],
        ):
            decoded = name.decode("utf-8")
            if decoded in ("TotalAcquisitionSpy", "TotalCount"):
                decoded = "TotalCnts"
            metadata[decoded] = datum[0]

        counts_raw = entry0["data_scan/detector_data/data"][0]
        counts = (
            np.asarray(counts_raw)
            .squeeze()
            .reshape(metadata["ndet"], metadata["ncell"])
        )

    return metadata, counts.astype(np.float64, copy=False)


def getOneNumor(  # noqa: N802 (legacy API)
    numorfile: str, runInfo: dict[str, Any]
) -> tuple[
    list[float],
    list[float],
    list[float],
    NDArray[np.float64],
    list[float],
    dict[str, Any],
]:
    """
    Read one numor and compute angle/count/error/Q/cell lists (legacy behavior).

    This is the core of the reduction of a single numor. Steps:

    1. Read the numor: `load_nxs` on ``numorfile + runInfo["ext"][4]`` if
       ``runInfo["dataFormat"] == "Nexus"`` (exact string), otherwise
       `readD4numorASCII` on ``numorfile``.
    2. Angle check. With the cell width ``w = runInfo["angular_range_bank"] /
       head["ncell"]`` (8/64 = 0.125° by default) and
       ``Δ = |Theta_lue - Theta_des|`` (read minus requested angle):
       if ``Δ > cellTolerance[1]·w`` an error is logged, printed and raised;
       else if ``Δ > cellTolerance[0]·w`` a warning is logged and printed.
    3. Poisson errors on the RAW counts: σ = √N (`getErrorsNumor`).
    4. Dead time (log line ``"Dead-time corrections"``): monitor with
       `getDTC_mcounts` (τ = ``dead_time[0]``), detector cells with
       `getDTC_dcounts` (τ = ``dead_time[1]``). The errors are multiplied by the
       same per-cell factor ``N_corr/N`` (factor 1 for cells with N = 0).
       The corrected monitor and total counts are logged with `printRates`.
    5. Efficiency: counts and errors divided by `getEff` (``runInfo["efffile"]``);
       cells with efficiency <= 0 become NaN.
    6. Normalisation with `getNormalFactors` (dead-time corrected monitor, time)
       and `normalise`: to ``normalisation[2]`` monitor counts, or to
       ``normalisation[1]`` s if ``normalisation[0] == "time"``. Cells with zero
       counts become NaN.
    7. Angles: shifts from `getDec` (``runInfo["decfile"]``) and `getAngles` with
       ``head["Theta_lue(deg)"]`` as angle of the first cell (default 0.125° per
       cell, 15° per bank). The zero angle is NOT subtracted here.
    8. A line of 80 dashes is logged; the (ndet, ncell) arrays are flattened bank
       by bank (bank 1 cells 1..ncell, then bank 2, ...).
    9. Q = 4π/λ · sin(2θ/2) from these angles (``runInfo["wavelength"]``).
    10. If ``int(runInfo["writeNumor"]) == 1``, the numor is written with
        ``saving.saveOneNumor``.

    Parameters
    ----------
    numorfile : str
        Path of the raw file WITHOUT the NeXus extension, normally
        ``runInfo["path_raw"] + numor``.
    runInfo : dict
        Run parameters (see ``parameters.readParam``). Keys read:
        ``writeNumor`` (0/1), ``efffile``, ``decfile``, ``normalisation``
        (type, time in s, monitor counts), ``dead_time`` (τ_monitor, τ_detector
        in s), ``cellTolerance`` (warning, error; in cells), ``dataFormat``,
        ``ext`` (index 4 = raw-data extension, index 3 used by
        ``saveOneNumor``), ``angular_range_bank`` (degrees), ``wavelength`` (Å).
        Not modified.

    Returns
    -------
    x : list of float
        2θ of every cell in degrees, without zero-angle correction; length
        ndet·ncell.
    y : list of float
        Corrected, normalised counts (counts per ``normalisation[2]`` monitor
        counts or per ``normalisation[1]`` s); NaN for excluded cells.
    e : list of float
        Absolute errors of ``y``; NaN for excluded cells.
    q : numpy.ndarray of float64
        Q in 1/Å computed from ``x`` (i.e. BEFORE the zero-angle correction).
    c : list of float
        Cell identifiers ``1000·bank + cell`` (1-based).
    head : dict
        Metadata of the numor (from the reader), with ``"MonitorCnts"`` and
        ``"TotalCnts"`` REPLACED by their dead-time corrected values.

    Raises
    ------
    ValueError
        If the read angle differs from the requested one by more than
        ``cellTolerance[1]`` cells; if a dead-time factor is not positive; or
        from the readers.
    KeyError, FileNotFoundError, ImportError
        From the readers, or if a ``runInfo`` key is missing.

    Notes
    -----
    Side effects: appends to ``../logfiles/d4creg.log`` (angle warning/error,
    ``"Dead-time corrections"``, dead-time factors, rates, a dashed line);
    prints the angle warning/error; the ASCII reader prints a summary; may write
    ``<numor><ext[3]>`` (default ``.cdat``) in the current working directory.

    Open item (found while writing Appendix C): ``q`` (and therefore the Q
    column of the per-numor ``.cdat`` file) is computed from angles before the
    zero angle is subtracted; the main ``.qdat`` output, computed by ``d4creg``
    from corrected angles, is not affected.

    Changes from the legacy code (audit 2026-11, C5): bug fix vs legacy --
    Poisson errors are taken on the RAW counts and then scaled by the same
    correction factors as the counts (legacy took sqrt of the already dead-time
    and efficiency corrected counts, i.e. √(N/ε) instead of √N/ε).

    See Also
    --------
    getNumors : calls this function for every numor and stacks the results.
    """
    write = int(runInfo["writeNumor"])
    efffile = str(runInfo["efffile"])
    decfile = str(runInfo["decfile"])

    normalisation = runInfo["normalisation"]
    norm_type = str(normalisation[0])
    norm_time = float(normalisation[1])
    norm_mon = float(normalisation[2])

    dtm = float(runInfo["dead_time"][0])
    dtd = float(runInfo["dead_time"][1])
    cell_tol = runInfo["cellTolerance"]

    if str(runInfo["dataFormat"]) == "Nexus":
        head, counts = load_nxs(numorfile + str(runInfo["ext"][4]))
    else:
        head, counts = readD4numorASCII(numorfile)

    angular_range_cell = float(runInfo["angular_range_bank"]) / float(head["ncell"])
    delta_theta = float(
        np.abs(float(head["Theta_lue(deg)"]) - float(head["Theta_des(deg)"]))
    )

    if delta_theta > float(cell_tol[1]) * angular_range_cell:
        msg = (
            "ERROR! Angle differs from requested angle by more than "
            f"{cell_tol[1]} cell (Angle={head['Theta_lue(deg)']}, requested={head['Theta_des(deg)']})"
        )
        with d4creg_log_path().open("a", encoding="utf-8") as file:
            file.write("   ERROR! The angle differs from the requested angle by\n")
            file.write("          more than {} cell\n".format(cell_tol[1]))
            file.write(
                "          Angle = {} deg, requested angle = {} deg\n".format(
                    head["Theta_lue(deg)"], head["Theta_des(deg)"]
                )
            )
        print("   ERROR! The angle differs from the requested angle by")
        print("          more than {} cell".format(cell_tol[1]))
        print(
            "          Angle = {} deg, requested angle = {} deg".format(
                head["Theta_lue(deg)"], head["Theta_des(deg)"]
            )
        )
        raise ValueError(msg)
    if delta_theta > float(cell_tol[0]) * angular_range_cell:
        with d4creg_log_path().open("a", encoding="utf-8") as file:
            file.write("   WARNING! The angle differs from the requested angle by\n")
            file.write("            more than {} cell\n".format(cell_tol[0]))
            file.write(
                "            Angle = {} deg, requested angle = {} deg\n".format(
                    head["Theta_lue(deg)"], head["Theta_des(deg)"]
                )
            )
        print("   WARNING! The angle differs from the requested angle by ")
        print("            more than {} cell".format(cell_tol[0]))
        print(
            "            Angle = {} deg, requested angle = {} deg".format(
                head["Theta_lue(deg)"], head["Theta_des(deg)"]
            )
        )

    with d4creg_log_path().open("a", encoding="utf-8") as file:
        file.write("Dead-time corrections\n")

    # Bug fix vs legacy: Poisson errors are taken on the RAW counts and then
    # scaled by the same correction factors as the counts (legacy took
    # sqrt of the already dead-time and efficiency corrected counts).
    counts = counts.astype(np.float64, copy=False)
    errors = getErrorsNumor(counts)

    head["MonitorCnts"] = getDTC_mcounts(
        float(head["MonitorCnts"]), float(head["CntTime (sec)"]), dead_time=dtm
    )
    counts_dt, head["TotalCnts"] = getDTC_dcounts(
        counts,
        float(head["TotalCnts"]),
        float(head["CntTime (sec)"]),
        dead_time=dtd,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_factor = np.where(counts > 0, counts_dt / counts, 1.0)
    errors = errors * dt_factor
    counts = counts_dt

    printRates(
        "monitor",
        float(head["MonitorCnts"]),
        float(head["CntTime (sec)"]),
        float(head.get("RtrPower (MW)", 52.0)),
    )
    printRates(
        "detector",
        float(head["TotalCnts"]),
        float(head["CntTime (sec)"]),
        float(head.get("RtrPower (MW)", 52.0)),
    )

    efficiency = getEff(
        filename=efffile, ndet=int(head["ndet"]), ncells=int(head["ncell"])
    )
    counts = counts / efficiency
    errors = errors / efficiency

    normal = getNormalFactors(float(head["MonitorCnts"]), float(head["CntTime (sec)"]))
    counts, errors = normalise(
        counts.astype(np.float64, copy=False),
        errors.astype(np.float64, copy=False),
        normal,
        norm_type=norm_type,
        norm_time=norm_time,
        norm_mon=norm_mon,
    )

    zeros = getDec(filename=decfile, ndet=int(head["ndet"]))
    angles, cells = getAngles(counts, float(head["Theta_lue(deg)"]), zeros)

    with d4creg_log_path().open("a", encoding="utf-8") as file:
        file.write(80 * "-" + "\n")

    x: list[float] = []
    y: list[float] = []
    e: list[float] = []
    c: list[float] = []
    for det in range(int(head["ndet"])):
        for cell in range(len(counts[0, :])):
            x.append(float(angles[det][cell]))
            y.append(float(counts[det][cell]))
            e.append(float(errors[det][cell]))
            c.append(float(cells[det][cell]))

    ang = np.array(x, dtype=np.float64)
    q = ang2q(ang, wlength=float(runInfo["wavelength"]))
    if write == 1:
        saveOneNumor(x, y, e, q, c, head, runInfo)
    return x, y, e, q, c, head


def getNumors(  # noqa: N802 (legacy API)
    runInfo: dict[str, Any],
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """
    Read and stack multiple numors into a single 3D matrix (legacy behavior).

    The list ``runInfo["numorLst"]`` is expanded with ``numors.getNumorFiles``;
    every numor is read and corrected with `getOneNumor` from the file
    ``runInfo["path_raw"] + numor``, and its flat lists are stored in one slice
    of the output array, with the zero angle subtracted from the angles.

    Parameters
    ----------
    runInfo : dict
        Run parameters (see ``parameters.readParam``). Keys read: ``numorLst``,
        ``path_raw``, ``file`` (only written to the log), ``twotheta0`` (zero
        angle 2θ₀ in degrees), plus all the keys read by `getOneNumor`.
        Key written: ``writeNumor`` is set to 1 if the list contains a single
        numor (the change stays in ``runInfo``).

    Returns
    -------
    numor : numpy.ndarray of float64, shape (n_numors, 5, ndet·ncell)
        For numor i, the rows are:

        - ``numor[i, 0]``: 2θ − twotheta0, in degrees;
        - ``numor[i, 1]``: normalised counts (NaN for excluded cells);
        - ``numor[i, 2]``: their absolute errors;
        - ``numor[i, 3]``: Q in 1/Å, computed from 2θ BEFORE subtracting
          twotheta0 (see `getOneNumor`);
        - ``numor[i, 4]``: cell identifier ``1000·bank + cell``.
    head : list of dict
        Metadata of every numor, in the same order (dead-time corrected
        ``MonitorCnts`` and ``TotalCnts``).

    Raises
    ------
    ValueError
        From `getNumorFiles` or `getOneNumor`; also if the numors do not all
        have the same number of cells (the array size is fixed by the first one).
    IndexError
        If the expanded list of numors is empty.

    Notes
    -----
    Appends to ``../logfiles/d4creg.log`` the base name ``runInfo["file"]`` and
    one line ``"Numor i/n <numor>"`` before each numor, plus everything logged by
    `getOneNumor`.
    """
    numor_files = getNumorFiles(list(runInfo["numorLst"]))
    if len(numor_files) == 1:
        runInfo["writeNumor"] = 1

    head: list[dict[str, Any]] = []

    numorfile = str(runInfo["path_raw"]) + numor_files[0]
    with d4creg_log_path().open("a", encoding="utf-8") as file:
        file.write(str(runInfo["file"]) + "\n")
        file.write("Numor 1/{}".format(len(numor_files)) + " " + numor_files[0] + "\n")

    angle, count, error, qval, cell, header = getOneNumor(numorfile, runInfo)
    angle_arr = np.array(angle, dtype=np.float64)
    head.append(header)

    numor: NDArray[np.float64] = np.zeros(
        (len(numor_files), 5, len(angle_arr)), dtype=np.float64
    )
    numor[0, 0, :] = angle_arr - float(runInfo["twotheta0"])
    numor[0, 1, :] = np.asarray(count, dtype=np.float64)
    numor[0, 2, :] = np.asarray(error, dtype=np.float64)
    numor[0, 3, :] = np.asarray(qval, dtype=np.float64)
    numor[0, 4, :] = np.asarray(cell, dtype=np.float64)

    for i in range(1, len(numor_files)):
        numorfile = str(runInfo["path_raw"]) + numor_files[i]
        with d4creg_log_path().open("a", encoding="utf-8") as file:
            file.write(
                "Numor {}/{}".format(i + 1, len(numor_files))
                + " "
                + numor_files[i]
                + "\n"
            )

        angle, count, error, qval, cell, header = getOneNumor(numorfile, runInfo)
        angle_arr = np.array(angle, dtype=np.float64)
        head.append(header)

        numor[i, 0, :] = angle_arr - float(runInfo["twotheta0"])
        numor[i, 1, :] = np.asarray(count, dtype=np.float64)
        numor[i, 2, :] = np.asarray(error, dtype=np.float64)
        numor[i, 3, :] = np.asarray(qval, dtype=np.float64)
        numor[i, 4, :] = np.asarray(cell, dtype=np.float64)

    return numor, head
