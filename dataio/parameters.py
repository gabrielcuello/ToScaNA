"""
Reader of the d4creg parameter file (the input of the regrouping workflow).

Position in the processing chain
--------------------------------
Entry point of the D4 regrouping. ``readParam`` reads a text file of tagged lines,
builds the dictionary ``runInfo`` used by every function of the chain
(``acquisition``, ``corrections``, ``d4creg_outputs``, ``workflows.d4creg``) and,
for every ``<run>`` line, calls ``workflows.d4creg.d4creg(runInfo)``.

Functions
---------
readParam
    Read a d4creg parameter file, return ``runInfo``; run d4creg at each ``<run>``.

Conventions
-----------
- Angles in degrees, wavelength in Å, Q in 1/Å, times and dead times in s.
- File names in the parameter file (efficiency, shifts, raw-data folder, output
  base name) are relative to the current working directory unless absolute.
- The log file ``../logfiles/d4creg.log`` (relative to the current working
  directory) is (re)created empty at each call.
- Import cycle: this module imports ``workflows.d4creg``, which imports ``dataio``;
  it works because ``dataio/__init__.py`` imports ``parameters`` last.
"""

from __future__ import annotations

from pathlib import Path
from toscana.workflows.d4creg import d4creg


def readParam(parametersFile: str) -> dict[str, object]:  # noqa: N802 (legacy API)
    """
    Read a d4creg parameter file and return the run dictionary ``runInfo``.

    The file is read sequentially, line by line; values set by a tag replace the
    previous ones, and each ``<run>`` line launches a reduction with the values
    read so far. Several ``<run>`` lines with changes in between therefore
    perform several reductions.

    Line syntax
    ~~~~~~~~~~~
    - A line of length <= 1 (empty line, i.e. only the newline) is skipped.
    - A line starting with ``#`` or ``!`` is a comment.
    - Any other line must start with a five-character tag ``<xxx>`` (``<`` at
      position 0 and ``>`` at position 4), followed by values separated by white
      space. Text after a ``#`` on a tag line is ignored (inline comment).
    - Lines with a valid tag shape but an unknown tag are SILENTLY ignored.
    - Values that the tag does not use (extra tokens) are ignored.

    Tags, ``runInfo`` keys and defaults
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    =======  ======================  =========================================
    Tag      ``runInfo`` key         Values (default)
    =======  ======================  =========================================
    (none)   ``angular_range_bank``  degrees covered by one bank (8.0); no tag
    <tol>    ``cellTolerance``       angle tolerance for warning and error, in
                                     cells (0.1, 1.0)
    <tau>    ``dead_time``           τ monitor, τ detector in s (2.4e-6, 7e-6)
    <ext>    ``ext``                 6 extensions: reg, adat, qdat, cdat, raw
                                     data, log (".reg", ".adat", ".qdat",
                                     ".cdat", ".nxs", ".log")
    <eff>    ``efffile``             efficiency file ("effd4c.eff"; "ones")
    <dec>    ``decfile``             bank shifts file ("dec.dec"; "zeros",
                                     "manual")
    <log>    ``logfile``             name used to rename the log at the end of
                                     d4creg ("d4creg" + ext[5] = "d4creg.log")
    <fmt>    ``dataFormat``          "Nexus" or anything else (= ILL ASCII)
                                     ("Nexus")
    <asc>    ``angular_scale``       min, max, step of the 2θ grid, degrees
                                     (0.0, 140.0, 0.125)
    <qsc>    ``q_scale``             min, max, step of the Q grid, 1/Å
                                     (0.0, 23.8, 0.02)
    <wri>    ``writeNumor``          1 if the value is exactly "True", else 0
                                     (0): write one file per numor
    <plo>    ``plotDiff``            1 if exactly "True", else 0 (0): plots
    <wle>    ``wavelength``          λ in Å (0.49857)
    <zac>    ``twotheta0``           zero angle 2θ₀ in degrees (-0.1013)
    <rdp>    ``path_raw``            raw-data folder, prefixed as a plain
                                     string, so keep the final "/" ("rawdata/")
    <nor>    ``normalisation``       type ("monitor" or "time"), time in s,
                                     monitor counts ("monitor", 80.0, 1e6)
    <out>    ``file``                base name of output files ("vanadium")
    <num>    ``numorLst``            REPLACES the list with [first value]
                                     (["387229-387288"])
    <add>    ``numorLst``            APPENDS the first value to the list
    <run>    --                      calls ``d4creg(runInfo)``
    =======  ======================  =========================================

    Numor entries are ``"123456"`` or ``"123456-123460"`` (see
    ``numors.getNumorFiles``); only the first value on a ``<num>``/``<add>`` line
    is used. ``<tol>``, ``<tau>``, ``<asc>``, ``<qsc>``, ``<nor>`` replace the
    whole tuple; ``<ext>`` needs at least six values.

    Parameters
    ----------
    parametersFile : str
        Path of the parameter file (UTF-8).

    Returns
    -------
    dict
        ``runInfo`` with all the keys of the table (defaults for the tags not
        present). If ``<run>`` lines were executed, it also contains the keys
        added by ``workflows.d4creg.d4creg`` (``runDate``, ``nbrNumors``,
        ``d4creg_cl``) and possibly ``writeNumor`` forced to 1 by
        ``acquisition.getNumors`` (single-numor run).

    Raises
    ------
    ValueError
        ``"Wrong input in line: <i> file: <name>"`` if a line longer than one
        character does not start with ``#``, ``!`` or ``<`` (checked for the
        whole file BEFORE anything is processed, so a bad line prevents all
        reductions); ``"Input error in line: ..."`` if a line starting with ``<``
        does not have ``>`` at position 4 (checked while processing, so earlier
        ``<run>`` lines have already been executed); also if a numeric value
        cannot be converted with ``float``.
    IndexError
        If a tag line has fewer values than the tag needs.
    FileNotFoundError
        If ``parametersFile`` does not exist (the log file has already been
        reset at that point).
    Exception
        Any exception raised by ``d4creg`` during a ``<run>`` propagates.

    Notes
    -----
    Side effects:

    - creates ``../logfiles`` if needed and (re)writes
      ``../logfiles/d4creg.log`` with the content ``"# d4creg.log\n\n"``,
      before reading the parameter file (relative to the current working
      directory);
    - prints one line per tag read (``<qsc>`` also prints a recommended Q step,
      ``asc_step / (asc_max − asc_min) · (q_max − q_min)``, i.e. the same number
      of points as the angular grid; ``<log>`` prints the SHIFTS file name, a
      legacy message bug kept on purpose; ``<run>`` prints
      ``"Calling d4creg..."``);
    - every ``<run>`` performs a full reduction (files written, log renamed,
      see ``workflows.d4creg.d4creg``) and mutates ``runInfo``. Because
      ``getNumors`` sets ``runInfo["writeNumor"] = 1`` for a single-numor run,
      that value stays for the following ``<run>`` lines unless ``<wri>`` is
      repeated.
    - ``runInfo["numorLst"]`` is the same list object that ``<add>`` appends to.

    Changes from the legacy code (audit 2026-11, C25): bug fix vs legacy --
    lines were split with ``split(" ")``, which kept the trailing newline on the
    last token ("Nexus\n", "True\n", "387229-387288\n") and produced empty
    tokens for repeated spaces. Inline ``#`` comments are now stripped and the
    line is split on any white space.

    Examples
    --------
    Usage (not run)::

        # par.txt
        # <eff> effd4c.eff
        # <out> vanadium
        # <num> 387229-387288
        # <run>
        runInfo = readParam("par.txt")
    """
    runInfo: dict[str, object] = {}

    angular_range_bank = 8.0
    dead_time: tuple[float, float] = (2.4e-6, 7.0e-6)
    cell_tolerance: tuple[float, float] = (0.1, 1.0)
    data_format = "Nexus"

    path_raw = "rawdata/"
    ext: tuple[str, str, str, str, str, str] = (
        ".reg",
        ".adat",
        ".qdat",
        ".cdat",
        ".nxs",
        ".log",
    )
    efffile = "effd4c.eff"
    decfile = "dec.dec"
    logfile = "d4creg" + ext[5]

    normalisation: tuple[str, float, float] = ("monitor", 80.0, 1000000.0)
    write_numor = 0
    plot_diff = 0
    twotheta0 = -0.1013
    wavelength = 0.49857
    angular_scale: tuple[float, float, float] = (0.0, 140.0, 0.125)
    q_scale: tuple[float, float, float] = (0.0, 23.8, 0.02)

    file_base = "vanadium"
    numor_lst: list[str] = ["387229-387288"]

    runInfo["angular_range_bank"] = angular_range_bank
    runInfo["dead_time"] = dead_time
    runInfo["cellTolerance"] = cell_tolerance
    runInfo["dataFormat"] = data_format

    runInfo["path_raw"] = path_raw
    runInfo["ext"] = ext
    runInfo["efffile"] = efffile
    runInfo["decfile"] = decfile
    runInfo["logfile"] = logfile

    runInfo["normalisation"] = normalisation
    runInfo["writeNumor"] = write_numor
    runInfo["plotDiff"] = plot_diff
    runInfo["twotheta0"] = twotheta0
    runInfo["wavelength"] = wavelength
    runInfo["angular_scale"] = angular_scale
    runInfo["q_scale"] = q_scale

    runInfo["file"] = file_base
    runInfo["numorLst"] = numor_lst

    # Legacy writes a logfile in a fixed relative location.
    log_path = Path("../logfiles/d4creg.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("# d4creg.log\n\n", encoding="utf-8")

    lines = Path(parametersFile).read_text(encoding="utf-8").splitlines(keepends=True)

    for i, line in enumerate(lines):
        if len(line) > 1:
            first = (line[0] != "#") and (line[0] != "!") and (line[0] != "<")
            if first:
                raise ValueError(f"Wrong input in line: {i + 1} file: {parametersFile}")

    for i, raw in enumerate(lines):
        if (raw[0] == "#") or (raw[0] == "!"):
            continue
        if len(raw) == 1:
            continue
        if (raw[0] == "<") and (len(raw) > 4) and (raw[4] == ">"):
            # Bug fix vs legacy: `raw.split(" ")` kept the trailing newline on
            # the last token ("Nexus\n", "True\n", "387229-387288\n") and
            # produced empty tokens for repeated spaces. Inline '#' comments are
            # stripped and the line is split on any whitespace.
            parts = raw.split("#")[0].split()
            tag = parts[0]

            if tag == "<tol>":
                cell_tolerance = (float(parts[1]), float(parts[2]))
                runInfo["cellTolerance"] = cell_tolerance
                print(
                    "Cell tolerance: {} cells for warning, {} cells for error".format(
                        *cell_tolerance
                    )
                )
            if tag == "<tau>":
                dead_time = (float(parts[1]), float(parts[2]))
                runInfo["dead_time"] = dead_time
                print(
                    "Dead times: {} s for monitor, {} s for detector".format(*dead_time)
                )
            if tag == "<ext>":
                ext = (
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                    parts[5],
                    parts[6],
                )
                runInfo["ext"] = ext
                print("Extensions: {}, {}, {}, {}, {}, {}".format(*ext))
            if tag == "<eff>":
                efffile = parts[1]
                runInfo["efffile"] = efffile
                print("Efficiency file: {}".format(efffile))
            if tag == "<dec>":
                decfile = parts[1]
                runInfo["decfile"] = decfile
                print("Shifts file: {}".format(decfile))
            if tag == "<log>":
                logfile = parts[1]
                runInfo["logfile"] = logfile
                # Legacy prints decfile here (bug); preserve message.
                print("Shifts file: {}".format(decfile))
            if tag == "<fmt>":
                data_format = parts[1]
                runInfo["dataFormat"] = data_format
                print("Data format: {}".format(data_format))
            if tag == "<asc>":
                angular_scale = (float(parts[1]), float(parts[2]), float(parts[3]))
                runInfo["angular_scale"] = angular_scale
                print(
                    "Angular scale: from {} deg to {} deg in steps of {} deg".format(
                        *angular_scale
                    )
                )
            if tag == "<qsc>":
                q_scale = (float(parts[1]), float(parts[2]), float(parts[3]))
                runInfo["q_scale"] = q_scale
                delta_a = angular_scale[1] - angular_scale[0]
                delta_q = q_scale[1] - q_scale[0]
                recommended_dq = angular_scale[2] / delta_a * delta_q
                print(f"Recommended dQ for the user is : {recommended_dq:.4f} 1/A")
                print(
                    "Q scale: from {} 1/A to {} 1/A in steps of {} 1/A".format(*q_scale)
                )
            if tag == "<wri>":
                write_numor = 1 if parts[1] == "True" else 0
                runInfo["writeNumor"] = write_numor
                print(
                    "Write individual files for each numor: {} {}".format(
                        parts[1], write_numor
                    )
                )
            if tag == "<plo>":
                plot_diff = 1 if parts[1] == "True" else 0
                runInfo["plotDiff"] = plot_diff
                print("Plot diffractograms: {} {}".format(parts[1], plot_diff))
            if tag == "<wle>":
                wavelength = float(parts[1])
                runInfo["wavelength"] = wavelength
                print("Incident wavelength: {} A".format(wavelength))
            if tag == "<zac>":
                twotheta0 = float(parts[1])
                runInfo["twotheta0"] = twotheta0
                print("Zero-angle: {} deg".format(twotheta0))
            if tag == "<rdp>":
                path_raw = parts[1]
                runInfo["path_raw"] = path_raw
                print("Relative path for rawdata: {}".format(path_raw))
            if tag == "<nor>":
                normalisation = (parts[1], float(parts[2]), float(parts[3]))
                runInfo["normalisation"] = normalisation
                print("Normalisation mode: {}".format(normalisation[0]))
                print(
                    "Normalisation constants: {1} s or {2} monitor counts".format(
                        *normalisation
                    )
                )
            if tag == "<out>":
                file_base = parts[1]
                runInfo["file"] = file_base
                print("Base name for output files: {}".format(file_base))
            if tag == "<num>":
                numor_lst = [parts[1]]
                runInfo["numorLst"] = numor_lst
                print("List of numors: {}".format(numor_lst))
            if tag == "<add>":
                numor_lst.append(parts[1])
                runInfo["numorLst"] = numor_lst
                print("List of numors: {}".format(numor_lst))
            elif tag == "<run>":
                print("Calling d4creg...")
                d4creg(runInfo)
        else:
            raise ValueError(f"Input error in line: {i + 1} file: {parametersFile}")

    return runInfo
