#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d4numor.py — D4 Numor Analysis without a graphical interface
=============================================================

Headless version of ``D4NumorGUI.py`` (v2.1-C).  The same analysis can be run

* from a terminal, as a command-line tool::

      python d4numor.py summary -d /data/rawdata -n 123456-123470
      python d4numor.py plot SampleTemp -d /data/rawdata -n 123456-123470 -o temp.png

* from Python / Jupyter, as a library::

      import d4numor as d4
      ns = d4.NumorSet.from_directory("/data/rawdata", numors="123456-123470")
      ns.summary()
      ns.plot(["Monitor", "TotalCount"], normalize=True)

Everything the GUI does is available:

=====================================  ==========================================
GUI feature                            Headless equivalent
=====================================  ==========================================
Browse + Open Numors                   ``NumorSet.from_directory`` / ``-d -n``
Variable tree (categories)             ``ns.list_variables()`` / ``list``
Weighted averages + table              ``ns.print_table(...)`` / ``table``
Plot (with *Normalize*)                ``ns.plot(..., normalize=True)`` / ``plot``
*Binary* mode (Y vs X)                 ``ns.plot_binary(y, x)`` / ``binary``
Detector movement information          ``ns.motion_summary()`` / ``summary``
Export Plot / Export Table             ``filename=...`` / ``ns.export_csv`` / ``export``
Scan angles + Preset + Calculate       ``ns.preset(value, angles)`` / ``preset``
(new) per-numor detector motion        ``ns.motion_table()`` / ``motion``
(new) details of a single numor        ``d4.numor_info(path)`` / ``info``
(new) pandas DataFrame                 ``ns.to_dataframe()``
=====================================  ==========================================

Requirements
------------
* Python >= 3.8, numpy, matplotlib (no Qt / PySide6 needed)
* ``nx_reader_D4.py`` (the same reader the GUI uses) importable: in the same
  folder as this file, in the current folder, on ``PYTHONPATH``, or given with
  ``--reader-dir`` (CLI) / ``reader_dir=`` (Python).
* pandas is optional (only for ``NumorSet.to_dataframe``).

Differences with the GUI (fixes and small improvements)
-------------------------------------------------------
* *Export Table* in the GUI wrote one CSV line per variable instead of one
  line per numor when several variables were selected.  Fixed here.
* Weighted averages ignore NaN values instead of returning NaN.
* Variables whose length is N-1 (e.g. quantities defined *between* two
  numors) are aligned with numors[1:] instead of crashing the plot.
* In binary mode, X and Y are matched numor by numor (not just truncated).
* If a numor file cannot be read, ``extract_multiple_variables`` drops its
  numor but keeps NaN data; numors are then rebuilt from the file names so
  that values stay attached to the right numor, and a warning is printed.
* For a counts input (>= 1000), the preset result is shown as a total scan
  time in s / min / h, and the moving fraction is computed too.

Original GUI author: Gabriel Cuello (cuello@ill.fr)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import io
import glob
import inspect
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# CENTRAL VARIABLE MAP (identical to the GUI so both stay in sync)
# ---------------------------------------------------------------------------
# internal_name: {"label": display name, "category": group, "tooltip": description}
# Add an entry here to give a nice label/category to a new variable.  Variables
# returned by the reader but absent from this map are still usable; they are
# listed in the category "Other".
# ---------------------------------------------------------------------------
VARIABLE_MAP: Dict[str, Dict[str, str]] = {
    # Axes
    "ttheta_value": {"label": "2θ [deg]", "category": "Axes", "tooltip": "First detector angle"},
    "omega_value": {"label": "ω (sample) [deg]", "category": "Axes", "tooltip": "Sample table rotation"},
    "A1_value": {"label": "A1 (mono) [deg]", "category": "Axes", "tooltip": "A1 goniometer"},
    "delta_theta": {"label": "Delta theta [deg]", "category": "Axes", "tooltip": "Detector angular displacement"},
    "angular_speed": {"label": "Angular speed [deg/s]", "category": "Axes", "tooltip": "Computed mean angular speed"},
    "moving_time": {"label": "2θ time movement [s]", "category": "Axes", "tooltip": "Moving time between 2 numors"},
    "BSC_value": {"label": "Beamstop Belljar [mm]", "category": "Axes", "tooltip": "BSC reading"},
    "BSD_value": {"label": "Beamstop Detector [mm]", "category": "Axes", "tooltip": "BSD reading"},

    # Counting
    "Monitor": {"label": "Monitor counts", "category": "Counting", "tooltip": "Monitor counter"},
    "TotalCount": {"label": "Total detector counts", "category": "Counting", "tooltip": "Sum of detector counts"},
    "Time": {"label": "Acquisition time [s]", "category": "Counting", "tooltip": "Counting time"},
    "monrate": {"label": "Monitor rate [cts/s]", "category": "Counting", "tooltip": "Monitor counts/time"},
    "detrate": {"label": "Detector rate [cts/s]", "category": "Counting", "tooltip": "Detector counts/time"},
    **{
        f"SumDet{i}": {"label": f"Detector {i} counts", "category": "Counting",
                       "tooltip": f"Sum of detector {i} counts"}
        for i in range(1, 10)
    },

    # Environment
    "SampleTemp": {"label": "Sample temperature [K]", "category": "Environment", "tooltip": "Measured sample temperature"},
    "RegTemp": {"label": "Regulation temperature [K]", "category": "Environment", "tooltip": "Regulation temperature"},
    "SetPTemp": {"label": "Set point temperature [K]", "category": "Environment", "tooltip": "Requested temperature"},
    "Pcloche": {"label": "Belljar pressure", "category": "Environment", "tooltip": "Pressure in belljar"},
    # Available with nx_reader_D4 v2 (current file format)
    "RegTemp2": {"label": "Regulation temperature 2 [K]", "category": "Environment", "tooltip": "Second regulation sensor"},
    "SetPTemp2": {"label": "Set point temperature 2 [K]", "category": "Environment", "tooltip": "Second set point"},
    "pressure": {"label": "Sample pressure", "category": "Environment", "tooltip": "entry0/sample/pressure"},
    "SetPPressure": {"label": "Set point pressure", "category": "Environment", "tooltip": "entry0/sample/setpoint_pressure"},
    "z_value": {"label": "z (sample height)", "category": "Axes", "tooltip": "Sample z axis"},
    "SCload_value": {"label": "Sample changer load axis", "category": "Axes", "tooltip": "SCloadaxis"},
    "SCrot_value": {"label": "Sample changer rotation", "category": "Axes", "tooltip": "SCrotaxis"},
    "ttheta_target": {"label": "2θ requested [deg]", "category": "Axes", "tooltip": "2theta target value"},
    "EI_value": {"label": "Incident energy [meV]", "category": "Beam", "tooltip": "Monochromator ei"},
    "wavelength": {"label": "Wavelength [Å]", "category": "Beam", "tooltip": "9.045/sqrt(Ei)"},

    # Metadata
    "username": {"label": "Username", "category": "Metadata", "tooltip": "Main proposer"},
    "LCname": {"label": "LC name", "category": "Metadata", "tooltip": "Local contact"},
    "proposal": {"label": "Proposal ID", "category": "Metadata", "tooltip": "Proposal number"},
    "subtitle": {"label": "Subtitle", "category": "Metadata", "tooltip": "Sample description"},
    "start_time": {"label": "Starting time", "category": "Metadata", "tooltip": "Starting time"},
    "end_time": {"label": "Ending time", "category": "Metadata", "tooltip": "Ending time"},
    "cycle": {"label": "Cycle", "category": "Metadata", "tooltip": "Cycle number"},
    "nbD4": {"label": "D4 Notebook", "category": "Metadata", "tooltip": "D4 notebook #"},
}

#: Variables requested from ``nx_reader_D4.extract_multiple_variables`` by default
#: (same list as the GUI).
DEFAULT_VARIABLES: List[str] = [
    "start_time", "end_time", "ttheta_value", "TotalCount", "Monitor", "Time",
    "SampleTemp", "omega_value", "A1_value", "Pcloche", "BSC_value", "BSD_value",
    "username", "LCname", "proposal", "subtitle", "cycle", "nbD4",
    "SumDet1", "SumDet2", "SumDet3", "SumDet4", "SumDet5", "SumDet6",
    "SumDet7", "SumDet8", "SumDet9",
]

#: Also read by default when nx_reader_D4 is version 2 or later (current file
#: format: regulation / set-point temperatures and sample pressure).
V2_VARIABLES: List[str] = ["RegTemp", "SetPTemp", "pressure"]

#: Default scan angles for the preset calculator (the 10 boxes of the GUI).
DEFAULT_SCAN_ANGLES: List[float] = [-0.5, -0.5, 0.0, 4.5, 5.0, 6.0, 7.5, 9.0, 10.0, 10.75]

#: Default file-name pattern used to turn a numor number into a file name.
DEFAULT_PATTERN = "{:06d}.nxs"

#: Display order of categories (others follow alphabetically).
CATEGORY_ORDER = ["Axes", "Counting", "Environment", "Beam", "Metadata", "Other"]


class D4Error(Exception):
    """Error raised for user-level problems (bad path, unknown variable, ...)."""


# ===========================================================================
# Reader import
# ===========================================================================
def get_reader(reader_dir: Optional[str] = None):
    """Import and return the ``nx_reader_D4`` module.

    Search order: ``reader_dir`` (if given), the folder containing this file,
    the current working directory, then the normal ``sys.path``.

    Raises
    ------
    D4Error
        If the module cannot be imported.
    """
    candidates = [reader_dir, os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for d in candidates:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        import nx_reader_D4  # noqa: WPS433 (runtime import on purpose)
    except ImportError as exc:
        raise D4Error(
            "Cannot import 'nx_reader_D4'. Put nx_reader_D4.py next to d4numor.py, "
            "in the current folder, or pass --reader-dir / reader_dir=. "
            f"(original error: {exc})"
        ) from exc
    return nx_reader_D4


# ===========================================================================
# Helpers: numor lists and file selection
# ===========================================================================
def parse_numor_list(spec: Union[str, int, Iterable[int]]) -> List[int]:
    """Convert a numor specification into a sorted list of unique integers.

    Accepted forms (combinable with commas)::

        "123456"                 -> [123456]
        "123456-123460"          -> [123456, ..., 123460]   (inclusive)
        "123456-123470:2"        -> every second numor in the range
        "123456-123460,123470"   -> range plus a single numor
        [123456, 123458]         -> list/range/iterable of ints

    Examples
    --------
    >>> parse_numor_list("10-12, 20")
    [10, 11, 12, 20]
    """
    if isinstance(spec, (int, np.integer)):
        return [int(spec)]
    if not isinstance(spec, str):
        return sorted({int(n) for n in spec})

    out = set()
    for chunk in spec.replace(" ", "").split(","):
        if not chunk:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?(?::(\d+))?", chunk)
        if not m:
            raise D4Error(f"Invalid numor specification: '{chunk}' "
                          "(use e.g. 123456, 123456-123470 or 123456-123470:2)")
        start = int(m.group(1))
        stop = int(m.group(2)) if m.group(2) else start
        step = int(m.group(3)) if m.group(3) else 1
        if stop < start:
            start, stop = stop, start
        out.update(range(start, stop + 1, max(step, 1)))
    return sorted(out)


def numor_from_path(path: str) -> Optional[int]:
    """Return the numor number contained in a file name (``.../123456.nxs`` -> 123456)."""
    m = re.search(r"(\d+)(?=\.[^.\\/]*$)", os.path.basename(path)) or \
        re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _sort_key(path: str):
    n = numor_from_path(path)
    return (0, n, path) if n is not None else (1, 0, path)


def build_file_list(directory: Optional[str] = None,
                    numors: Union[None, str, Iterable[int]] = None,
                    files: Optional[Sequence[str]] = None,
                    pattern: str = DEFAULT_PATTERN,
                    verbose: bool = True) -> List[str]:
    """Build the list of NeXus files to analyse.

    Three ways of selecting files (exactly as in the GUI, but without dialogs):

    1. ``directory`` + ``numors``: files are ``directory/pattern.format(numor)``.
       Missing files are reported and skipped.
    2. ``files``: explicit paths and/or glob patterns (``"/data/1234*.nxs"``).
    3. ``directory`` alone: every ``*.nxs`` file in the directory.

    The result is sorted by numor number.
    """
    paths: List[str] = []

    if files:
        for f in files:
            expanded = glob.glob(os.path.expanduser(f))
            if not expanded and os.path.isfile(f):
                expanded = [f]
            if not expanded:
                print(f"Warning: no file matches '{f}'", file=sys.stderr)
            paths.extend(expanded)

    if directory is not None:
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            raise D4Error(f"Directory not found: {directory}")
        if numors is not None:
            missing = []
            for n in parse_numor_list(numors):
                p = os.path.join(directory, pattern.format(n))
                if os.path.isfile(p):
                    paths.append(p)
                else:
                    missing.append(n)
            if missing and verbose:
                shown = ", ".join(str(n) for n in missing[:10])
                more = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
                print(f"Warning: {len(missing)} numor(s) not found in {directory}: {shown}{more}",
                      file=sys.stderr)
        elif not files:
            paths.extend(glob.glob(os.path.join(directory, "*.nxs")))
    elif numors is not None and not files:
        raise D4Error("A directory (-d/--dir) is required when numors are given.")

    paths = sorted(set(os.path.abspath(p) for p in paths), key=_sort_key)
    if not paths:
        raise D4Error("No numor file selected (check directory, numors and pattern).")
    return paths


# ===========================================================================
# Helpers: data conversion and formatting
# ===========================================================================
def _to_array(values) -> np.ndarray:
    """Convert reader output into a 1-D numpy array.

    Numeric data -> float array (NaN for missing).  Text data -> object array of
    Python strings (bytes are decoded as UTF-8).
    """
    if values is None:
        return np.array([], dtype=float)
    if isinstance(values, np.ndarray):
        if values.ndim > 1:
            # e.g. 'detdata': numpy stacked one array per numor into a 2-D
            # array.  Keep one element (an array) per numor.
            arr = np.empty(values.shape[0], dtype=object)
            arr[:] = list(values)
            return arr
        if values.dtype.kind in "biuf":
            return values.astype(float)
        values = values.tolist()
    elif isinstance(values, (str, bytes, int, float)):
        values = [values]
    else:
        values = list(values)

    clean = []
    for v in values:
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="ignore")
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                v = v.item()
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", errors="ignore")
            else:
                # One array per numor (e.g. 'detdata'): keep as an object array
                # of arrays.  Such variables are treated as non-numeric.
                arr = np.empty(len(values), dtype=object)
                arr[:] = [np.asarray(x) for x in values]
                return arr
        clean.append(v)
    try:
        return np.asarray([np.nan if v is None else v for v in clean], dtype=float)
    except (TypeError, ValueError):
        return np.asarray(["" if v is None else str(v) for v in clean], dtype=object)


def _fmt_value(v, width: int = 0) -> str:
    """Pretty-print a single table value (integers without decimals)."""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            s = "nan"
        elif float(v).is_integer() and abs(v) < 1e15:
            s = f"{int(v)}"
        elif abs(v) < 1e-3:          # e.g. belljar pressure in mbar
            s = f"{v:.4e}"
        else:
            s = f"{v:.4f}"
    else:
        s = str(v)
    return s.rjust(width) if width else s


def _fmt_column(values: Sequence) -> List[str]:
    """Format a column of a text table with one consistent style.

    Integer-valued columns -> no decimals; very small numbers (e.g. pressure)
    -> scientific notation; other numbers -> 4 decimals; text -> as is.
    """
    nums = [float(v) for v in values
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)]
    if not nums or len(nums) < sum(1 for v in values if v != ""):
        return [_fmt_value(v) for v in values]          # text or mixed column
    if all(x.is_integer() for x in nums) and max(abs(x) for x in nums) < 1e15:
        spec = "{:.0f}"
    elif 0 < max(abs(x) for x in nums) < 1e-2:
        spec = "{:.4e}"
    else:
        spec = "{:.4f}"
    out = []
    for v in values:
        if v == "" or v is None:
            out.append("")
        elif not np.isfinite(float(v)):
            out.append("nan")
        else:
            out.append(spec.format(float(v)))
    return out


def _csv_value(v) -> str:
    """Full-precision value for CSV files."""
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else (f"{int(v)}" if float(v).is_integer() and abs(v) < 1e15
                                                  else f"{v:.10g}")
    if isinstance(v, np.ndarray):
        return " ".join(f"{x:g}" for x in v.ravel())
    return str(v)


def label(var: str) -> str:
    """Friendly display name of a variable (falls back to the internal name)."""
    return VARIABLE_MAP.get(var, {}).get("label", var)


def category(var: str) -> str:
    """Category of a variable ("Other" if not in VARIABLE_MAP)."""
    return VARIABLE_MAP.get(var, {}).get("category", "Other")


def _ordered_categories(cats: Iterable[str]) -> List[str]:
    cats = set(cats)
    return [c for c in CATEGORY_ORDER if c in cats] + sorted(cats - set(CATEGORY_ORDER))


# ===========================================================================
# Main class
# ===========================================================================
class NumorSet:
    """A set of D4 numors and the variables extracted from them.

    Create it with :meth:`from_directory` or :meth:`from_files` (or directly
    with a list of file paths).

    Attributes
    ----------
    filepaths : list of str
        The NeXus files that were read, sorted by numor.
    numors : numpy.ndarray
        Numor numbers (float array, one per file).
    data : dict[str, numpy.ndarray]
        All variables: those returned by ``nx_reader_D4`` plus derived ones
        (``monrate``, ``detrate``, ``moving_time``, ``delta_theta``,
        ``angular_speed``).  Numeric variables are float arrays; text variables
        are object arrays of strings.
    motion : dict
        Raw result of ``nx_reader_D4.analyse_angular_motion`` (empty if not
        available).
    """

    # ------------------------------------------------------------------ init
    def __init__(self, filepaths: Sequence[str],
                 variables: Optional[Sequence[str]] = None,
                 extra_variables: Optional[Sequence[str]] = None,
                 reader=None, reader_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Parameters
        ----------
        filepaths : sequence of str
            NeXus files to read.
        variables : sequence of str, optional
            Variables to request from the reader (default: ``DEFAULT_VARIABLES``,
            plus ``V2_VARIABLES`` with nx_reader_D4 version 2).
        extra_variables : sequence of str, optional
            Additional variables appended to ``variables``.
        reader : module, optional
            An already imported ``nx_reader_D4`` module (or any object with the
            same functions).  Imported automatically if omitted.
        reader_dir : str, optional
            Folder containing ``nx_reader_D4.py`` (used only if ``reader`` is None).
        verbose : bool
            Print a short loading message.
        """
        self.filepaths = sorted((str(p) for p in filepaths), key=_sort_key)
        if not self.filepaths:
            raise D4Error("Empty file list.")
        self.reader = reader if reader is not None else get_reader(reader_dir)
        self.verbose = verbose
        self.numors = np.array([])
        self.data: Dict[str, np.ndarray] = {}
        self.motion: dict = {}

        if variables is not None:
            req = list(variables)
        else:
            req = list(DEFAULT_VARIABLES)
            if str(getattr(self.reader, "__version__", "1")).split(".")[0] >= "2":
                req += V2_VARIABLES
        for v in extra_variables or []:
            if v not in req:
                req.append(v)
        self.requested_variables = req
        self._load(req)

    @classmethod
    def from_directory(cls, directory: str,
                       numors: Union[None, str, Iterable[int]] = None,
                       pattern: str = DEFAULT_PATTERN, **kwargs) -> "NumorSet":
        """Load numors from a raw-data directory.

        Parameters
        ----------
        directory : str
            Raw-data folder.
        numors : str or iterable of int, optional
            Numors to load (see :func:`parse_numor_list`).  If omitted, all
            ``*.nxs`` files in the directory are loaded.
        pattern : str
            File-name pattern, default ``"{:06d}.nxs"``.
        **kwargs
            Passed to :class:`NumorSet` (``variables``, ``reader``, ``verbose``...).

        Example
        -------
        ::

            ns = NumorSet.from_directory("/data/rawdata", "123456-123470")
        """
        files = build_file_list(directory=directory, numors=numors, pattern=pattern,
                                verbose=kwargs.get("verbose", True))
        return cls(files, **kwargs)

    @classmethod
    def from_files(cls, files: Union[str, Sequence[str]], **kwargs) -> "NumorSet":
        """Load numors from explicit paths and/or glob patterns.

        Example::

            ns = NumorSet.from_files("/data/rawdata/12345*.nxs")
        """
        if isinstance(files, str):
            files = [files]
        return cls(build_file_list(files=files, verbose=kwargs.get("verbose", True)), **kwargs)

    # --------------------------------------------------------------- loading
    def _load(self, varnames: List[str]) -> None:
        """Call the reader and compute derived variables (mirrors the GUI)."""
        extract_fn = getattr(self.reader, "extract_multiple_variables", None)
        if extract_fn is None:
            raise D4Error("nx_reader_D4 has no function extract_multiple_variables().")

        # The GUI supports readers with signature f(files) or f(files, varnames)
        try:
            n_params = len(inspect.signature(extract_fn).parameters)
        except (TypeError, ValueError):
            n_params = 2
        raw_call = (lambda files: extract_fn(files, varnames)) if n_params >= 2 else extract_fn

        def call(files):
            # The reader prints [WARN]/[ERROR] lines on stdout.  Capture them:
            # errors are always shown, warnings only when verbose (on stderr,
            # so that the tables printed on stdout stay clean).
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    return raw_call(files)
            finally:
                for line in buf.getvalue().splitlines():
                    if line.strip() and (self.verbose or "ERROR" in line):
                        print(line, file=sys.stderr)

        try:
            res = call(self.filepaths)
        except Exception as exc:
            # The reader cannot build its arrays when an array variable (e.g.
            # 'detdata') is mixed with the NaN of an unreadable file.  Read the
            # files one by one and assemble the result here instead.
            if len(self.filepaths) < 2:
                raise D4Error(f"Failed extracting variables: {exc}") from exc
            numors_l, merged = [], {}
            for f in self.filepaths:
                try:
                    r = call([f])
                except Exception as exc2:
                    raise D4Error(f"Failed extracting variables from {f}: {exc2}") from exc2
                n_i, d_i = (r[0], r[1]) if isinstance(r, tuple) else (r.get("numor", []), r)
                numors_l.extend(list(np.atleast_1d(n_i)) or [np.nan])
                for k, v in d_i.items():
                    merged.setdefault(k, []).append(v[0] if len(v) else np.nan)
            res = (numors_l, {k: v for k, v in merged.items()})

        if isinstance(res, tuple) and len(res) >= 2:
            numors, raw = res[0], res[1]
        elif isinstance(res, dict):
            raw = res
            numors = raw.get("numor", raw.get("Numor", []))
        else:
            raise D4Error("extract_multiple_variables returned an unexpected result.")

        self.data = {str(k): _to_array(v) for k, v in (raw or {}).items()
                     if k not in ("numor", "Numor")}

        # Numor numbers.  The reader appends one value per file to every
        # variable (NaN if the file fails), so data[i] <-> filepaths[i].  The
        # numors are therefore taken from the file names.  The numor list
        # returned by the reader is used only as a fallback: it skips files
        # that fail (out of step with the data) and joins all digits of the
        # name ('D4_416440.nxs' -> 4416440).
        n_data = len(next(iter(self.data.values()))) if self.data else 0
        guess = [numor_from_path(p) for p in self.filepaths]
        if all(g is not None for g in guess) and (not n_data or len(guess) == n_data):
            self.numors = np.asarray(guess, dtype=float)
        else:
            try:
                self.numors = np.asarray(numors, dtype=float).ravel()
                if self.numors.size == 0 or (n_data and self.numors.size != n_data):
                    raise ValueError
            except (TypeError, ValueError):
                self.numors = np.arange(n_data, dtype=float)

        # Report numors that could not be read (all numeric values are NaN)
        num_vars = [v for v in self.data.values() if v.dtype.kind == "f" and v.size == len(self.numors)]
        if num_vars:
            bad = np.all(~np.isfinite(np.vstack(num_vars)), axis=0)
            self.unreadable = [int(n) for n in self.numors[bad] if np.isfinite(n)]
            if self.unreadable:
                print(f"Warning: {len(self.unreadable)} numor(s) could not be read and "
                      f"are ignored: {', '.join(map(str, self.unreadable[:10]))}", file=sys.stderr)
                # Drop them, otherwise one bad file breaks the motion analysis
                keep = ~bad
                n_all = len(self.numors)
                self.data = {k: (v[keep] if len(v) == n_all else v) for k, v in self.data.items()}
                self.numors = self.numors[keep]
                bad_set = set(self.unreadable)
                self.filepaths = [p for p in self.filepaths if numor_from_path(p) not in bad_set]
                if not len(self.numors):
                    raise D4Error("None of the selected numors could be read.")
        else:
            self.unreadable = []

        # Derived count rates (as in the GUI)
        mon, tot, tim = (self.data.get(k) for k in ("Monitor", "TotalCount", "Time"))
        with np.errstate(divide="ignore", invalid="ignore"):
            if "monrate" not in self.data and self._num_ok(mon, tim):
                self.data["monrate"] = np.where(tim > 0, mon / tim, np.nan)
            if "detrate" not in self.data and self._num_ok(tot, tim):
                self.data["detrate"] = np.where(tim > 0, tot / tim, np.nan)

        # Angular motion analysis (optional, as in the GUI)
        if hasattr(self.reader, "analyse_angular_motion") and \
                all(k in self.data for k in ("ttheta_value", "start_time", "end_time")):
            try:
                # analyse_angular_motion prints its totals even with
                # verbose=False; silence it (the same numbers are in summary()).
                with contextlib.redirect_stdout(io.StringIO()):
                    self.motion = self.reader.analyse_angular_motion(
                        self.numors.astype(int),
                        np.asarray(self.data["ttheta_value"], dtype=float),
                        np.asarray(self.data["start_time"]),
                        np.asarray(self.data["end_time"]),
                        verbose=False,
                    ) or {}
                for key in ("moving_time", "delta_theta", "angular_speed"):
                    if key in self.motion:
                        self.data[key] = _to_array(self.motion[key])
            except Exception as exc:
                print(f"Warning: analyse_angular_motion failed: {exc}", file=sys.stderr)

        if self.verbose:
            print(f"Loaded {len(self.numors)} numors "
                  f"({self._numor_str(self.numors[0])} -> {self._numor_str(self.numors[-1])}), "
                  f"{len(self.data)} variables.")

    @staticmethod
    def _num_ok(*arrays) -> bool:
        return all(a is not None and a.size and a.dtype.kind == "f" for a in arrays) and \
            len({a.size for a in arrays}) == 1

    @staticmethod
    def _numor_str(n) -> str:
        try:
            return f"{int(float(n)):06d}"
        except (TypeError, ValueError):
            return str(n)

    def __len__(self) -> int:
        return len(self.numors)

    def __repr__(self) -> str:
        if not len(self):
            return "<NumorSet: empty>"
        return (f"<NumorSet: {len(self)} numors {self._numor_str(self.numors[0])}-"
                f"{self._numor_str(self.numors[-1])}, {len(self.data)} variables>")

    # ----------------------------------------------------------- variables
    @property
    def variables(self) -> List[str]:
        """Names of all available variables (sorted by category, then name)."""
        def key(v):
            cat = category(v)
            return (CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99, v.lower())
        return sorted(self.data, key=key)

    def is_numeric(self, var: str) -> bool:
        """True if variable ``var`` exists and is numeric (can be plotted/averaged)."""
        try:
            return self.data[self.resolve(var)].dtype.kind == "f"
        except D4Error:
            return False

    def resolve(self, name: str) -> str:
        """Return the internal variable name matching ``name``.

        Accepts the exact internal name, a case-insensitive version of it, or
        the friendly label from ``VARIABLE_MAP``.  Raises :class:`D4Error` with
        suggestions if nothing matches.
        """
        if name in self.data:
            return name
        low = name.lower()
        for v in self.data:
            if v.lower() == low or label(v).lower() == low:
                return v
        close = difflib.get_close_matches(name, list(self.data), n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else \
            " Use 'list' / list_variables() to see available variables."
        if name in VARIABLE_MAP:
            raise D4Error(f"Variable '{name}' was not returned by the reader for these numors.{hint}")
        raise D4Error(f"Unknown variable '{name}'.{hint}")

    def get(self, var: str) -> np.ndarray:
        """Return the raw array of a variable (see :meth:`aligned` for its numors)."""
        return self.data[self.resolve(var)]

    def aligned(self, var: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(numors, values)`` for a variable, with matching lengths.

        * length N   -> all numors
        * length N-1 -> numors[1:] (a quantity between numor i-1 and i is
          attributed to numor i, e.g. detector moving time)
        * otherwise  -> the first min(N, len) numors
        """
        var = self.resolve(var)
        vals = self.data[var]
        n, m = len(self.numors), len(vals)
        if m == n:
            return self.numors, vals
        if m == n - 1:
            return self.numors[1:], vals
        k = min(m, n)
        return self.numors[:k], vals[:k]

    def _lookup(self, var: str) -> Dict[float, object]:
        x, v = self.aligned(var)
        return dict(zip(x.tolist(), v.tolist()))

    def list_variables(self, show_unavailable: bool = False, print_out: bool = True) -> str:
        """List variables grouped by category (equivalent of the GUI tree).

        Parameters
        ----------
        show_unavailable : bool
            Also list ``VARIABLE_MAP`` entries the reader did not return
            (shown greyed-out in the GUI, marked ``[ ]`` here).
        print_out : bool
            Print the listing (it is always returned as a string).
        """
        txt = format_variable_listing(set(self.data), self.data, show_unavailable)
        if print_out:
            print(txt)
        return txt

    # ------------------------------------------------------------ statistics
    def weighted_average(self, var: str, weight: str = "Monitor") -> float:
        """Average of ``var`` weighted by another variable (default: Monitor).

        Values are matched numor by numor; NaN values and negative weights are
        ignored.  Returns NaN if nothing is left.
        """
        var = self.resolve(var)
        if weight not in self.data or not self.is_numeric(var) or not self.is_numeric(weight):
            return float("nan")
        xa, va = self.aligned(var)
        xw, w = self.aligned(weight)
        _, ia, iw = np.intersect1d(xa, xw, return_indices=True)
        v, w = va[ia].astype(float), w[iw].astype(float)
        ok = np.isfinite(v) & np.isfinite(w) & (w >= 0)
        if not ok.any() or w[ok].sum() <= 0:
            return float("nan")
        return float(np.average(v[ok], weights=w[ok]))

    def weighted_averages(self, variables: Union[str, Sequence[str]]) -> Dict[str, Dict[str, float]]:
        """Monitor- and detector-weighted averages, as shown at the top of the GUI.

        Returns ``{var: {"monitor": ..., "detector": ..., "mean": ...}}``.
        """
        if isinstance(variables, str):
            variables = [variables]
        out = {}
        for var in variables:
            var = self.resolve(var)
            if not self.is_numeric(var):
                continue
            vals = self.data[var]
            out[var] = {
                "monitor": self.weighted_average(var, "Monitor"),
                "detector": self.weighted_average(var, "TotalCount"),
                "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
            }
        return out

    def stats(self, var: str) -> Dict[str, float]:
        """Basic statistics of a numeric variable: n, mean, std, min, max, weighted means."""
        var = self.resolve(var)
        if not self.is_numeric(var):
            raise D4Error(f"Variable '{var}' is not numeric.")
        v = self.data[var]
        f = v[np.isfinite(v)]
        nan = float("nan")
        return {
            "n": int(f.size),
            "mean": float(f.mean()) if f.size else nan,
            "std": float(f.std(ddof=1)) if f.size > 1 else nan,
            "min": float(f.min()) if f.size else nan,
            "max": float(f.max()) if f.size else nan,
            "monitor_weighted": self.weighted_average(var, "Monitor"),
            "detector_weighted": self.weighted_average(var, "TotalCount"),
        }

    # ---------------------------------------------------------------- tables
    def table(self, variables: Union[str, Sequence[str]], averages: bool = True) -> str:
        """Text table *Numor + variables* (plus weighted averages), as in the GUI.

        Works for numeric and text variables.  Returns the table as a string.
        """
        if isinstance(variables, str):
            variables = [variables]
        variables = [self.resolve(v) for v in variables]
        lines = []

        if averages:
            wa = self.weighted_averages(variables)
            if wa:
                lines.append("Weighted averages")
                w = max(len(v) for v in wa)
                def f(x):
                    return f"{x:14.4e}" if np.isfinite(x) and 0 < abs(x) < 1e-2 else f"{x:14.4f}"
                for var, d in wa.items():
                    lines.append(f"  {var.ljust(w)} : Monitor-weighted = {f(d['monitor'])}   "
                                 f"Detector-weighted = {f(d['detector'])}")
                lines.append("")

        lookups = [self._lookup(v) for v in variables]
        headers = ["Numor"] + [label(v) for v in variables]
        numors = self.numors.tolist()
        cols = [[self._numor_str(n) for n in numors]] + \
               [_fmt_column([lk.get(n, "") for n in numors]) for lk in lookups]
        rows = [list(r) for r in zip(*cols)]
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        lines.append(" | ".join(h.rjust(w) for h, w in zip(headers, widths)))
        lines.append("-+-".join("-" * w for w in widths))
        for r in rows:
            lines.append(" | ".join(c.rjust(w) for c, w in zip(r, widths)))
        return "\n".join(lines)

    def print_table(self, variables: Union[str, Sequence[str]], averages: bool = True) -> None:
        """Print :meth:`table`."""
        print(self.table(variables, averages=averages))

    def to_dataframe(self, variables: Union[None, str, Sequence[str]] = None,
                     labels: bool = False):
        """Return a pandas DataFrame indexed by numor (requires pandas).

        Parameters
        ----------
        variables : str or list, optional
            Columns to include (default: all variables).
        labels : bool
            Use friendly labels as column names instead of internal names.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise D4Error("pandas is required for to_dataframe()") from exc
        if variables is None:
            variables = self.variables
        elif isinstance(variables, str):
            variables = [variables]
        variables = [self.resolve(v) for v in variables]
        index = pd.Index(self.numors.astype(int), name="Numor")
        cols = {}
        for v in variables:
            x, vals = self.aligned(v)
            s = pd.Series(list(vals), index=x.astype(int))
            cols[label(v) if labels else v] = s.reindex(index)
        return pd.DataFrame(cols, index=index)

    # ----------------------------------------------------------------- plots
    def plot(self, variables: Union[str, Sequence[str]], normalize: bool = False,
             separate: bool = False, ax=None, filename: Optional[str] = None,
             show: bool = False, title: Optional[str] = None, style: str = "o-",
             figsize: Tuple[float, float] = (8, 5)):
        """Plot one or several numeric variables versus numor.

        Parameters
        ----------
        variables : str or list of str
            Variables to plot (text variables are skipped with a warning).
        normalize : bool
            Divide each curve by its mean (GUI *Normalize* check box).
        separate : bool
            One panel per variable (shared numor axis) instead of one panel.
        ax : matplotlib Axes, optional
            Draw into an existing axes (ignored if ``separate``).
        filename : str, optional
            Save the figure (png, pdf, svg... from the extension).
        show : bool
            Call ``plt.show()`` (not needed in Jupyter).
        title, style, figsize :
            Usual matplotlib options.

        Returns
        -------
        (fig, axes)
        """
        import matplotlib.pyplot as plt

        if isinstance(variables, str):
            variables = [variables]
        variables = [self.resolve(v) for v in variables]
        numeric = [v for v in variables if self.is_numeric(v)]
        for v in set(variables) - set(numeric):
            print(f"Warning: '{v}' is not numeric and is not plotted "
                  "(use table() to see it).", file=sys.stderr)
        if not numeric:
            raise D4Error("Nothing to plot: no numeric variable selected.")

        if separate and len(numeric) > 1:
            fig, axes = plt.subplots(len(numeric), 1, sharex=True,
                                     figsize=(figsize[0], max(figsize[1], 2.2 * len(numeric))),
                                     squeeze=False)
            axes = list(axes[:, 0])
        elif ax is not None:
            fig, axes = ax.figure, [ax] * len(numeric)
        else:
            fig, a = plt.subplots(figsize=figsize)
            axes = [a] * len(numeric)

        for var, a in zip(numeric, axes):
            x, y = self.aligned(var)
            y = y.astype(float).copy()
            if normalize:
                with np.errstate(invalid="ignore"):
                    m = np.nanmean(y)
                if m and np.isfinite(m):
                    y = y / m
            a.plot(x, y, style, label=label(var), markersize=4)
            a.grid(True, alpha=0.4)
            if separate and len(numeric) > 1:
                a.set_ylabel(label(var) + (" (norm.)" if normalize else ""))
        last = axes[-1]
        last.set_xlabel("Numor")
        last.ticklabel_format(axis="x", useOffset=False, style="plain")
        if not (separate and len(numeric) > 1):
            last.set_ylabel("Value (normalized)" if normalize else
                            (label(numeric[0]) if len(numeric) == 1 else "Value"))
            last.legend()
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        self._finish(fig, filename, show)
        return fig, (axes if separate else axes[0])

    def plot_binary(self, y: str, x: str, ax=None, filename: Optional[str] = None,
                    show: bool = False, style: str = "o-", annotate: bool = False,
                    figsize: Tuple[float, float] = (7, 5)):
        """Plot ``y`` versus ``x`` (GUI *Binary* mode: first selection = Y, second = X).

        Points are matched numor by numor.  ``annotate=True`` writes the numor
        next to each point.  Returns ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt

        common, xv, yv = self.binary_data(y, x)
        y, x = self.resolve(y), self.resolve(x)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.plot(xv, yv, style, markersize=4)
        if annotate:
            for n, xi, yi in zip(common, xv, yv):
                ax.annotate(self._numor_str(n), (xi, yi), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(label(x))
        ax.set_ylabel(label(y))
        ax.set_title(f"{label(y)} vs {label(x)}")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        self._finish(fig, filename, show)
        return fig, ax

    def binary_data(self, y: str, x: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(numors, x_values, y_values)`` matched numor by numor."""
        y, x = self.resolve(y), self.resolve(x)
        for v in (x, y):
            if not self.is_numeric(v):
                raise D4Error(f"Binary mode needs numeric variables; '{v}' is text.")
        xa, xv = self.aligned(x)
        ya, yv = self.aligned(y)
        common, ix, iy = np.intersect1d(xa, ya, return_indices=True)
        if common.size == 0:
            raise D4Error("Selected variables do not overlap or are empty.")
        return common, xv[ix], yv[iy]

    def binary_table(self, y: str, x: str) -> str:
        """Text table ``Numor | X | Y`` for binary mode."""
        common, xv, yv = self.binary_data(y, x)
        headers = ["Numor", label(self.resolve(x)), label(self.resolve(y))]
        rows = [list(r) for r in zip([self._numor_str(n) for n in common],
                                     _fmt_column(list(xv)), _fmt_column(list(yv)))]
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        out = [" | ".join(h.rjust(w) for h, w in zip(headers, widths)),
               "-+-".join("-" * w for w in widths)]
        out += [" | ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
        return "\n".join(out)

    @staticmethod
    def _finish(fig, filename: Optional[str], show: bool) -> None:
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {filename}")
        if show:
            import matplotlib.pyplot as plt
            plt.show()

    # ---------------------------------------------------------------- export
    def export_csv(self, variables: Union[None, str, Sequence[str]], path: str,
                   labels: bool = True) -> str:
        """Write ``Numor`` + variables to a CSV file (one line per numor).

        Parameters
        ----------
        variables : str, list or None
            Columns to write (None = all variables).
        path : str
            Output file.
        labels : bool
            Use friendly labels in the header (as the GUI) instead of internal names.
        """
        if variables is None:
            variables = self.variables
        elif isinstance(variables, str):
            variables = [variables]
        variables = [self.resolve(v) for v in variables]
        lookups = [self._lookup(v) for v in variables]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["Numor"] + [label(v) if labels else v for v in variables])
            for n in self.numors.tolist():
                w.writerow([self._numor_str(n)] +
                           [_csv_value(lk[n]) if n in lk else "" for lk in lookups])
        print(f"Table saved to {path}")
        return path

    def export_binary_csv(self, y: str, x: str, path: str) -> str:
        """Write ``Numor, X, Y`` (binary mode) to a CSV file."""
        common, xv, yv = self.binary_data(y, x)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["Numor", label(self.resolve(x)), label(self.resolve(y))])
            for n, a, b in zip(common, xv, yv):
                w.writerow([self._numor_str(n), _csv_value(a), _csv_value(b)])
        print(f"Binary table saved to {path}")
        return path

    # ---------------------------------------------------------------- motion
    def motion_summary(self) -> str:
        """Detector movement information (the line under the table in the GUI)."""
        rng = self.motion.get("angular_range")
        tmv = self.motion.get("time_movement")
        if rng is None or tmv in (None, 0):
            return "Detector movement information not available."
        return ("Detector Movement Information\n"
                f"  Angular range : {rng:10.3f} deg\n"
                f"  Movement time : {tmv:10.3f} s\n"
                f"  Mean speed    : {rng / tmv:10.5f} deg/s")

    # --------------------------------------------------------------- summary
    def _unique_text(self, var: str) -> str:
        if var not in self.data:
            return "-"
        vals = [_fmt_value(v) for v in self.data[var].tolist()]
        vals = [v for v in vals if v not in ("", "nan")]
        uniq = list(dict.fromkeys(vals))
        if not uniq:
            return "-"
        return uniq[0] if len(uniq) == 1 else f"{', '.join(uniq[:4])}" + \
            (f" (+{len(uniq) - 4} more)" if len(uniq) > 4 else "")

    def summary(self, print_out: bool = True) -> str:
        """Overview of the loaded numors: metadata, counting totals, temperature, motion."""
        L = []
        L.append("=" * 64)
        L.append(f"D4 numor summary  —  {len(self)} numors "
                 f"({self._numor_str(self.numors[0])} -> {self._numor_str(self.numors[-1])})")
        L.append("=" * 64)
        L.append(f"Directory   : {os.path.dirname(self.filepaths[0])}")
        for var in ("proposal", "username", "LCname", "subtitle", "cycle", "nbD4"):
            L.append(f"{label(var):<12}: {self._unique_text(var)}")
        if "start_time" in self.data:
            L.append(f"{'First start':<12}: {_fmt_value(self.data['start_time'][0])}")
        if "end_time" in self.data:
            L.append(f"{'Last end':<12}: {_fmt_value(self.data['end_time'][-1])}")
        L.append("-" * 64)

        def tot(var):
            return float(np.nansum(self.data[var])) if self.is_numeric(var) else float("nan")

        W = 26
        if self.is_numeric("Time"):
            t = tot("Time")
            L.append(f"{'Total counting time':<{W}}: {t:12.1f} s  ({t / 3600:.2f} h)")
        if self.is_numeric("Monitor"):
            L.append(f"{'Total monitor':<{W}}: {tot('Monitor'):12.0f}")
        if self.is_numeric("TotalCount"):
            L.append(f"{'Total detector':<{W}}: {tot('TotalCount'):12.0f}")
        for var in ("monrate", "detrate"):
            if self.is_numeric(var):
                s = self.stats(var)
                L.append(f"{label(var):<{W}}: mean {s['mean']:10.2f}  "
                         f"[{s['min']:.2f}, {s['max']:.2f}]")
        for var in ("SampleTemp", "RegTemp", "SetPTemp"):
            if self.is_numeric(var):
                s = self.stats(var)
                L.append(f"{label(var):<{W}}: mean {s['mean']:10.3f}  std {s['std']:.3f}  "
                         f"[{s['min']:.3f}, {s['max']:.3f}]")
        if self.is_numeric("ttheta_value"):
            s = self.stats("ttheta_value")
            L.append(f"{'2θ range [deg]':<{W}}: {s['min']:.3f} -> {s['max']:.3f}")
        L.append("-" * 64)
        L.append(self.motion_summary())
        L.append("=" * 64)
        txt = "\n".join(L)
        if print_out:
            print(txt)
        return txt

    # ---------------------------------------------------------------- preset
    def rates(self) -> Dict[str, float]:
        """Rates used by the preset calculator.

        Returns a dict with:

        * ``mon_rate_median``  : median of monitor/time over numors (GUI choice)
        * ``mon_rate_global``  : total monitor / total time
        * ``speed_gui``        : mean of the per-numor angular speeds, zeros
          included (what the GUI passes to get_preset)
        * ``speed_motion``     : total angular range / total moving time
          (the "Mean speed" of the detector-movement information)
        """
        nan = float("nan")
        out = {"mon_rate_median": nan, "mon_rate_global": nan,
               "speed_gui": nan, "speed_motion": nan}
        if self.is_numeric("monrate") and np.isfinite(self.data["monrate"]).any():
            out["mon_rate_median"] = float(np.nanmedian(self.data["monrate"]))
        if self.is_numeric("Monitor") and self.is_numeric("Time"):
            t = np.nansum(self.data["Time"])
            if t > 0:
                out["mon_rate_global"] = float(np.nansum(self.data["Monitor"]) / t)
        if self.is_numeric("angular_speed") and np.isfinite(self.data["angular_speed"]).any():
            out["speed_gui"] = float(np.nanmean(self.data["angular_speed"]))
        rng, tmv = self.motion.get("angular_range"), self.motion.get("time_movement")
        if rng is not None and tmv:
            out["speed_motion"] = float(rng / tmv)
        return out

    def preset(self, value: float, scan_angles: Optional[Sequence[float]] = None,
               mon_rate: Optional[float] = None, angular_speed: Union[None, float, str] = None,
               print_out: bool = True) -> dict:
        """Preset calculator (scan-angle boxes + *Calculate* button of the GUI).

        Calls ``nx_reader_D4.get_preset(value, scan_angles, mon_rate, angular_speed)``.

        Meaning of ``value`` (convention of ``get_preset``):

        * ``value < 1000``  -> desired total scan time in **minutes**; the result
          is the **monitor preset per point** (counts).
        * ``value >= 1000`` -> desired **monitor counts per point**; the result
          is the **total scan time** (get_preset returns seconds; minutes and
          hours are shown too).

        Parameters
        ----------
        value : float
            Preset input (minutes or monitor counts).
        scan_angles : list of float, optional
            Scan angles in degrees (default ``DEFAULT_SCAN_ANGLES``, the 10
            boxes of the GUI).
        mon_rate : float, optional
            Monitor rate [counts/s].  Default: median of the loaded numors
            (as the GUI).
        angular_speed : float or "gui" or "motion", optional
            Detector speed [deg/s].  ``None``/"gui" (default): mean of the
            per-numor speeds, exactly as the GUI.  "motion": total angular
            range / total moving time (usually larger, because the GUI mean
            includes the zero speeds of numors where the detector did not
            move).  A number: use it directly.
        print_out : bool
            Print the result.

        Returns
        -------
        dict with keys ``input``, ``input_kind``, ``scan_angles``, ``mon_rate``,
        ``angular_speed``, ``preset``, ``preset_units``, ``moving_time_min``,
        ``moving_ratio`` (fraction of the total time spent moving),
        ``total_scan_time_min`` (of the loaded numors) and ``raw``.
        """
        angles = [float(a) for a in (DEFAULT_SCAN_ANGLES if scan_angles is None else scan_angles)]
        if not angles:
            raise D4Error("Please provide at least one scan angle.")

        r = self.rates()
        if mon_rate is None:
            mon_rate = r["mon_rate_median"]
        if angular_speed is None or angular_speed == "gui":
            speed = r["speed_gui"]
        elif angular_speed == "motion":
            speed = r["speed_motion"]
        else:
            speed = float(angular_speed)
        if not np.isfinite(mon_rate) or mon_rate <= 0:
            raise D4Error("Monitor rate not available: pass mon_rate=... (counts/s).")
        if not np.isfinite(speed) or speed <= 0:
            raise D4Error("Angular speed not available (the detector did not move in the "
                          "loaded numors?): pass angular_speed=... (deg/s).")

        fn = getattr(self.reader, "get_preset", None)
        if fn is None:
            raise D4Error("nx_reader_D4.get_preset(...) not found.")
        try:
            res = fn(value, angles, mon_rate, speed)
        except Exception as exc:
            raise D4Error(f"get_preset() call failed: {exc}") from exc

        preset_val, moving = None, None
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            preset_val, moving = res[0], res[1]
        elif isinstance(res, dict):
            preset_val = res.get("preset", res.get("monitor", res.get("time")))
            moving = res.get("moving_time", res.get("moving_minutes", res.get("moving_time_min")))
        else:
            preset_val = res

        ratio = None
        try:
            if value < 1000 and moving is not None and value != 0:
                ratio = float(moving) / float(value)
            elif value >= 1000 and moving is not None and preset_val:
                ratio = float(moving) * 60.0 / float(preset_val)   # preset in s
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = None
        total_min = float(np.nansum(self.data["Time"]) / 60.0) if self.is_numeric("Time") else float("nan")

        out = {
            "input": value,
            "input_kind": "total time [min]" if value < 1000 else "monitor counts per point",
            "scan_angles": angles,
            "mon_rate": float(mon_rate),
            "angular_speed": float(speed),
            "preset": preset_val,
            "preset_units": "monitor counts per point" if value < 1000 else "total scan time [s]",
            "moving_time_min": moving,
            "moving_ratio": ratio,
            "total_scan_time_min": total_min,
            "raw": res,
        }
        if print_out:
            print(format_preset(out))
        return out

    def preset_table(self, values: Sequence[float], scan_angles: Optional[Sequence[float]] = None,
                     **kwargs) -> str:
        """Run :meth:`preset` for several inputs and return a compact table.

        Same idea as ``nx_reader_D4.analyze_scan``: e.g.
        ``ns.preset_table([30, 60, 120, 10000, 50000])``.
        """
        rows = []
        for v in values:
            r = self.preset(v, scan_angles, print_out=False, **kwargs)
            if v < 1000:
                res = f"Monitor = {float(r['preset']):10.0f} per point"
            else:
                res = f"Time    = {float(r['preset']) / 60:10.1f} min      "
            loss = "" if r["moving_ratio"] is None else f"{100 * r['moving_ratio']:5.1f} %"
            unit = "min" if v < 1000 else "counts"
            rows.append(f"{v:10g} {unit:<6} | {res} | move {float(r['moving_time_min']):6.1f} min | {loss}")
        first = self.preset(values[0], scan_angles, print_out=False, **kwargs)
        head = [f"Scan angles: {', '.join(f'{a:g}' for a in first['scan_angles'])}",
                f"Monitor rate: {first['mon_rate']:.2f} cts/s   Angular speed: "
                f"{first['angular_speed']:.5f} deg/s",
                f"{'Input':^17} | {'Preset result':^31} | {'Moving time':^15} | Loss",
                "-" * 78]
        return "\n".join(head + rows)

    def motion_table(self) -> str:
        """Per-numor detector motion: 2θ, Δ2θ, moving time and speed.

        Same content as ``analyse_angular_motion(..., verbose=True)``.  The
        values on a line refer to the movement *before* that numor.
        """
        need = ("ttheta_value", "delta_theta", "moving_time", "angular_speed")
        if not all(k in self.data for k in need):
            return "Detector movement information not available."
        hdr = f"{'N':>3} | {'Numor':>6} | {'2θ':>8} | {'Δ2θ':>8} | {'Time [s]':>8} | {'Speed':>9}"
        L = [hdr, "-" * len(hdr)]
        for i, n in enumerate(self.numors):
            L.append(f"{i:3d} | {self._numor_str(n):>6} | {self.data['ttheta_value'][i]:8.3f} | "
                     f"{self.data['delta_theta'][i]:8.3f} | {self.data['moving_time'][i]:8.1f} | "
                     f"{self.data['angular_speed'][i]:9.4f}")
        L.append("-" * len(hdr))
        L.append(self.motion_summary())
        return "\n".join(L)

    def info(self, numor: Union[int, str]) -> str:
        """Detailed information on one numor of the set (see :func:`numor_info`)."""
        n = int(numor)
        for p in self.filepaths:
            if numor_from_path(p) == n:
                return numor_info(p, reader=self.reader)
        raise D4Error(f"Numor {n} is not in this set.")


def numor_info(path: str, reader=None, reader_dir: Optional[str] = None,
               print_out: bool = True) -> str:
    """Detailed information on a single numor file.

    Uses ``nx_reader_D4.read_nexus`` and ``nx_reader_D4.show_single_numor``
    (proposal, user, times, reactor cycle, energy/wavelength, 2θ, counts,
    rates, temperature, beam stops, the nine detector sums...).
    """
    reader = reader if reader is not None else get_reader(reader_dir)
    if not os.path.isfile(path):
        raise D4Error(f"File not found: {path}")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            nx = reader.read_nexus(path)
            reader.show_single_numor(nx)
            if hasattr(nx, "close"):
                nx.close()
    except Exception as exc:
        raise D4Error(f"Cannot read {path}: {exc}") from exc
    txt = buf.getvalue().rstrip()
    if print_out:
        print(txt)
    return txt


# ===========================================================================
# Module-level formatting helpers (also used by the CLI)
# ===========================================================================
def format_variable_listing(available: set, data: Optional[dict] = None,
                            show_unavailable: bool = True) -> str:
    """Text listing of variables grouped by category.

    ``[x]`` = available, ``[ ]`` = defined in VARIABLE_MAP but not returned by the reader.
    """
    data = data or {}
    names = set(available) | (set(VARIABLE_MAP) if show_unavailable else set())
    groups: Dict[str, List[str]] = {}
    for v in names:
        groups.setdefault(category(v), []).append(v)
    w = max((len(v) for v in names), default=10)
    lines = []
    for cat in _ordered_categories(groups):
        lines.append(f"{cat}")
        for v in sorted(groups[cat], key=str.lower):
            mark = "[x]" if v in available else "[ ]"
            kind = ""
            if v in data:
                arr = data[v]
                if arr.dtype.kind == "f":
                    kind = "num"
                elif arr.size and isinstance(arr[0], np.ndarray):
                    kind = "arr"           # one array per numor (e.g. detdata)
                else:
                    kind = "text"
            tip = VARIABLE_MAP.get(v, {}).get("tooltip", "")
            lines.append(f"  {mark} {v.ljust(w)}  {kind:4}  {label(v)}"
                         + (f"  — {tip}" if tip else ""))
    return "\n".join(lines)


def format_preset(res: dict) -> str:
    """Format the result of :meth:`NumorSet.preset` like the GUI."""
    L = [f"Scan angles   : {', '.join(f'{a:g}' for a in res['scan_angles'])} "
         f"({len(res['scan_angles'])} points)",
         f"Monitor rate  : {res['mon_rate']:.2f} cts/s",
         f"Angular speed : {res['angular_speed']:.5f} deg/s",
         f"Input         : {res['input']:g} ({res['input_kind']})"]
    p = res["preset"]
    if p is not None:
        try:
            if res["input"] < 1000:
                L.append(f"Preset result : {float(p):.0f} monitor counts per point")
            else:
                s = float(p)
                L.append(f"Preset result : total scan time {s:.0f} s = {s / 60:.1f} min "
                         f"= {s / 3600:.2f} h")
        except (TypeError, ValueError):
            L.append(f"Preset result : {p}")
    if res["moving_time_min"] is not None:
        try:
            L.append(f"Moving time   : {float(res['moving_time_min']):.2f} min")
        except (TypeError, ValueError):
            L.append(f"Moving time   : {res['moving_time_min']}")
    r = res["moving_ratio"]
    if r is not None and not math.isnan(r):
        L.append(f"Moving/Total  : {100.0 * r:.2f} %")
    if not math.isnan(res["total_scan_time_min"]):
        L.append(f"(Counting time of the loaded numors: {res['total_scan_time_min']:.1f} min)")
    return "\n".join(L)


# ===========================================================================
# Command-line interface
# ===========================================================================
EPILOG = """\
numor specification (-n): 123456 | 123456-123470 | 123456-123470:2 | 123456-123460,123470
variable names: internal names (SampleTemp, Monitor, ttheta_value, ...), case-insensitive.
                Run the 'list' command to see them.

examples:
  python d4numor.py list
  python d4numor.py summary -d /data/rawdata -n 123456-123470
  python d4numor.py table Monitor TotalCount SampleTemp -d /data/rawdata -n 123456-123470
  python d4numor.py plot Monitor TotalCount --normalize -d /data/rawdata -n 123456-123470 -o counts.png
  python d4numor.py binary -y detrate -x SampleTemp -d /data/rawdata -n 123456-123470 -o rate_vs_T.png
  python d4numor.py export --all -d /data/rawdata -n 123456-123470 -o run.csv
  python d4numor.py preset 30 --angles -0.5 -0.5 0 4.5 5 6 7.5 9 10 10.75 -d /data/rawdata -n 123456-123470
  python d4numor.py preset 30 60 120 10000 50000 -d /data/rawdata -n 123456-123470
  python d4numor.py motion -d /data/rawdata -n 123456-123470
  python d4numor.py info 123456 -d /data/rawdata
"""


def _add_selection_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    g = p.add_argument_group("numor selection")
    g.add_argument("-d", "--dir", dest="directory", help="raw-data directory")
    g.add_argument("-n", "--numors", help="numors to read, e.g. 123456-123470,123480")
    g.add_argument("-f", "--files", nargs="+", metavar="FILE",
                   help="explicit files or glob patterns (quote globs: '/data/1234*.nxs')")
    g.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help=f"file-name pattern for numors (default: '{DEFAULT_PATTERN}')")
    g.add_argument("--reader-dir", help="folder containing nx_reader_D4.py")
    g.add_argument("--extra-vars", nargs="+", default=[], metavar="VAR",
                   help="additional variables to ask nx_reader_D4 for")
    g.add_argument("-q", "--quiet", action="store_true", help="less output")
    p.set_defaults(_selection_required=required)


def _load_from_args(args) -> NumorSet:
    if not (args.directory or args.files):
        raise D4Error("Select numors with -d DIR [-n NUMORS] or -f FILES.")
    files = build_file_list(directory=args.directory, numors=args.numors, files=args.files,
                            pattern=args.pattern, verbose=not args.quiet)
    return NumorSet(files, extra_variables=args.extra_vars, reader_dir=args.reader_dir,
                    verbose=not args.quiet)


def _cmd_list(args) -> int:
    if args.directory or args.files:
        ns = _load_from_args(args)
        ns.list_variables(show_unavailable=args.all)
    else:
        print("Variables known to d4numor (load numors with -d/-n to see which are available):\n")
        print(format_variable_listing(set(), {}, show_unavailable=True).replace("[ ] ", "    "))
    return 0


def _cmd_summary(args) -> int:
    _load_from_args(args).summary()
    return 0


def _cmd_table(args) -> int:
    ns = _load_from_args(args)
    print(ns.table(args.variables, averages=not args.no_averages))
    if args.output:
        ns.export_csv(args.variables, args.output)
    return 0


def _cmd_plot(args) -> int:
    ns = _load_from_args(args)
    out = args.output or (None if args.show else "d4_plot.png")
    ns.plot(args.variables, normalize=args.normalize, separate=args.separate,
            filename=out, show=args.show, title=args.title)
    if args.print_table:
        print(ns.table(args.variables))
    return 0


def _cmd_binary(args) -> int:
    ns = _load_from_args(args)
    out = args.output or (None if args.show else "d4_binary.png")
    ns.plot_binary(args.y, args.x, filename=out, show=args.show, annotate=args.annotate)
    print(ns.binary_table(args.y, args.x))
    if args.csv:
        ns.export_binary_csv(args.y, args.x, args.csv)
    return 0


def _cmd_export(args) -> int:
    ns = _load_from_args(args)
    if not args.all and not args.variables:
        raise D4Error("Give variables to export, or --all.")
    ns.export_csv(None if args.all else args.variables, args.output,
                  labels=not args.internal_names)
    return 0


def _cmd_preset(args) -> int:
    ns = _load_from_args(args)
    speed = args.speed
    if speed not in (None, "gui", "motion"):
        try:
            speed = float(speed)
        except ValueError:
            raise D4Error("--speed must be 'gui', 'motion' or a number in deg/s")
    if len(args.value) == 1:
        ns.preset(args.value[0], args.angles, mon_rate=args.mon_rate, angular_speed=speed)
    else:
        print(ns.preset_table(args.value, args.angles, mon_rate=args.mon_rate,
                              angular_speed=speed))
    return 0


def _cmd_motion(args) -> int:
    print(_load_from_args(args).motion_table())
    return 0


def _cmd_info(args) -> int:
    if args.directory:
        path = os.path.join(os.path.expanduser(args.directory), args.pattern.format(int(args.numor)))
    else:
        path = args.numor
    numor_info(path, reader_dir=args.reader_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="d4numor.py",
        description="D4 Numor Analysis — command-line version of D4NumorGUI.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(title="commands", metavar="COMMAND")

    def add(name, func, help_, **kw):
        p = sub.add_parser(name, help=help_, description=help_, epilog=EPILOG,
                           formatter_class=argparse.RawDescriptionHelpFormatter, **kw)
        p.set_defaults(func=func)
        return p

    p = add("list", _cmd_list, "list variables (grouped by category)")
    p.add_argument("--all", action="store_true",
                   help="also show variables defined but not returned by the reader")
    _add_selection_args(p, required=False)

    p = add("summary", _cmd_summary, "overview of the selected numors")
    _add_selection_args(p)

    p = add("table", _cmd_table, "print weighted averages and a table of variables")
    p.add_argument("variables", nargs="+", metavar="VAR")
    p.add_argument("-o", "--output", help="also write the table to this CSV file")
    p.add_argument("--no-averages", action="store_true", help="do not print weighted averages")
    _add_selection_args(p)

    p = add("plot", _cmd_plot, "plot variables versus numor")
    p.add_argument("variables", nargs="+", metavar="VAR")
    p.add_argument("-o", "--output", help="image file (png/pdf/svg). Default: d4_plot.png")
    p.add_argument("--normalize", action="store_true", help="divide each curve by its mean")
    p.add_argument("--separate", action="store_true", help="one panel per variable")
    p.add_argument("--title", help="figure title")
    p.add_argument("--show", action="store_true", help="open an interactive window (needs a display)")
    p.add_argument("--print-table", action="store_true", help="also print the table")
    _add_selection_args(p)

    p = add("binary", _cmd_binary, "plot Y versus X (GUI 'Binary' mode)")
    p.add_argument("-y", required=True, metavar="VAR", help="variable on the vertical axis")
    p.add_argument("-x", required=True, metavar="VAR", help="variable on the horizontal axis")
    p.add_argument("-o", "--output", help="image file. Default: d4_binary.png")
    p.add_argument("--csv", help="also write Numor, X, Y to this CSV file")
    p.add_argument("--annotate", action="store_true", help="write the numor next to each point")
    p.add_argument("--show", action="store_true", help="open an interactive window (needs a display)")
    _add_selection_args(p)

    p = add("export", _cmd_export, "write variables to a CSV file")
    p.add_argument("variables", nargs="*", metavar="VAR")
    p.add_argument("-o", "--output", required=True, help="CSV file")
    p.add_argument("--all", action="store_true", help="export every available variable")
    p.add_argument("--internal-names", action="store_true",
                   help="use internal names instead of labels in the header")
    _add_selection_args(p)

    p = add("preset", _cmd_preset, "preset calculator (monitor counts <-> time)")
    p.add_argument("value", type=float, nargs="+",
                   help="preset input(s): < 1000 = total time in minutes, "
                        ">= 1000 = monitor counts per point. Several values -> table")
    p.add_argument("--angles", nargs="+", type=float, default=None, metavar="ANGLE",
                   help=f"scan angles in degrees (default: {' '.join(f'{a:g}' for a in DEFAULT_SCAN_ANGLES)})")
    p.add_argument("--mon-rate", type=float, default=None,
                   help="monitor rate in counts/s (default: median of the loaded numors)")
    p.add_argument("--speed", default=None,
                   help="detector speed: 'gui' (default, as the GUI), 'motion' "
                        "(range/moving time) or a number in deg/s")
    _add_selection_args(p)

    p = add("motion", _cmd_motion, "per-numor detector motion (2θ, Δ2θ, time, speed)")
    _add_selection_args(p)

    p = add("info", _cmd_info, "detailed information on a single numor")
    p.add_argument("numor", help="numor number (with -d) or path to a .nxs file")
    p.add_argument("-d", "--dir", dest="directory", help="raw-data directory")
    p.add_argument("--pattern", default=DEFAULT_PATTERN, help="file-name pattern")
    p.add_argument("--reader-dir", help="folder containing nx_reader_D4.py")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the command-line interface. Returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    import matplotlib
    # Choose a non-interactive backend for terminal use, but never change the
    # backend if pyplot is already in use (e.g. d4.main([...]) called from Jupyter).
    if "matplotlib.pyplot" not in sys.modules:
        if getattr(args, "show", False):
            if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or
                                                          os.environ.get("WAYLAND_DISPLAY")):
                print("Warning: no display available, --show ignored (the figure is saved instead).",
                      file=sys.stderr)
                args.show = False
                matplotlib.use("Agg")
        else:
            matplotlib.use("Agg")  # no window: works on headless machines / over ssh

    try:
        return args.func(args) or 0
    except D4Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
