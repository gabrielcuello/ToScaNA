# nx_reader_D4.py
"""
nx_reader_D4.py — NeXus reader and small analysis utilities for D4 (ILL)
=========================================================================

Version 2.0 (October 2026).  Drop-in replacement of the previous
``nx_reader_D4.py``: same functions, same names, same return values, so
``D4NumorGUI.py`` and ``d4numor.py`` work without changes.

Why a new version
-----------------
The NeXus files written by D4 changed.  In the current format:

* ``entry0/data_scan/scanned_variables/data`` has only the scanned
  quantities of the scan (6 rows for a 2θ scan: omega, TotalCount, Time,
  Monitor, SampleTemp, 2theta) instead of the fixed 14 rows (TotalCount,
  SumDet1..9, Time, Monitor, SampleTemp, 2theta) used before.  Their order
  is given by ``scanned_variables/variables_names/{label,name,property}``.
  The previous reader used fixed row numbers: it now fails on every file
  (``index 10 is out of bounds``), and rows 1..5 would be read as SumDet1..5
  although they contain other quantities.
* ``entry0/monitor/data`` and ``entry0/sample/Pcloche`` no longer exist.
* ``entry0/sample`` has new entries: ``regulation_temperature(2)``,
  ``setpoint_temperature(2)``, ``pressure``, ``setpoint_pressure``, ``use_temp``.
* The detector data are stored as ``(1, 1, 576)`` instead of ``(1, 576, 1)``.
* New axes: ``SCloadaxis``, ``SCrotaxis``, ``masked2theta_s``.

What this version does
----------------------
* Scanned variables are found **by name** (label / name / property), so the
  reader does not depend on the order or number of rows.  For old files
  without names, the old fixed row numbers are used.
* Every quantity has a list of possible sources (e.g. Monitor: scanned
  'Monitor' -> monitor/monsum -> monitor/data).  The first one present is
  used; ``nx.sources[attr]`` tells which one.
* A missing dataset never stops the reading: the attribute is NaN (numbers)
  or '' (text) and is listed in ``nx.missing``.
* The nine detector sums ``SumDet1..9`` are taken from the scanned variables
  when present (old format) or computed from the detector cells, 576 cells
  = 9 detectors x 64 cells (``nx.sources['SumDet1'] == 'computed'``).
* New variables: RegTemp, RegTemp2, SetPTemp, SetPTemp2, pressure,
  SetPPressure, use_temp, SCload_*, SCrot_*, masked2theta_*, ...
* Diagnostics: ``list_datasets(path)`` prints the content of a file and
  ``check_file(path)`` shows where each variable comes from.  Use them when
  the format changes again.

Bugs of the previous version fixed here
---------------------------------------
* ``getReactorCycle``: the D4 notebook was searched with a fixed date
  (20-Sep-23) instead of the numor date.  Cycles of 2026 added.
* ``extract_multiple_variables``: the numor of a file that could not be read
  was dropped while NaN was added to the variables (numors and values out of
  step); arrays such as 'detdata' mixed with NaN raised an error.
* ``analyse_angular_motion`` printed its totals even with ``verbose=False``.
* ``read_nexus``: 'controller_list' read 'actual_step'; 'RCU331H_offset'
  was missing (RCU331V_offset written twice); wavelength was infinite when
  Ei = 0; the HDF5 file stayed open.
* The module no longer reconfigures the ``logging`` of the program that
  imports it.

Requires: h5py, numpy

Author of the original module: Gabriel Cuello (ILL).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

__version__ = "2.0.0"

logger = logging.getLogger("nx_reader_D4")

NAN = float("nan")

# ---------------------------------------------------------------------------
# Format description
# ---------------------------------------------------------------------------
#: Old format: fixed rows of entry0/data_scan/scanned_variables/data
LEGACY_SCAN_ROWS: Dict[str, int] = {
    "TotalCount": 0,
    **{f"SumDet{i}": i for i in range(1, 10)},
    "Time": 10, "Monitor": 11, "SampleTemp": 12, "2theta": 13,
}

#: Motors / axes stored as entry0/instrument/<group>/{offset_value,target_value,value}.
#: group name in the file -> prefix of the attributes (prefix_offset, _target, _value)
MOTORS: Dict[str, str] = {
    "2theta": "ttheta", "omega": "omega", "z": "z",
    "A1": "A1", "A2": "A2", "GM": "GM", "R1": "R1", "R2": "R2", "T1": "T1", "T2": "T2",
    "RCU220H": "RCU220H", "RCU220V": "RCU220V", "RCU331H": "RCU331H", "RCU331V": "RCU331V",
    "RSH": "RSH", "RSV": "RSV",
    "BSCloche": "BSC", "BSDetector": "BSD",
    "VolGams": "VolGams", "VolLohen": "VolLohen",
    "SCloadaxis": "SCload", "SCrotaxis": "SCrot", "masked2theta_s": "masked2theta",
}

#: Simple quantities: attribute -> list of candidate dataset paths (first found wins)
#: and default value.  A value of the form "scan:<key>" means a scanned variable.
FIELDS: Dict[str, Tuple[List[str], Any]] = {
    # General information
    "run_number":             (["entry0/run_number"], NAN),
    "instrument":             (["entry0/instrument/name"], ""),
    "instrument_version":     (["entry0/instrument/version"], ""),
    "nexus_compression_type": (["entry0/instrument/nexus_compression_type"], ""),
    "nexus_type":             (["entry0/instrument/nexus_type"], ""),
    "title":                  (["entry0/title"], ""),
    "subtitle":               (["entry0/experiment_identifier"], ""),
    "localcontactlist":       (["entry0/user/localcontactlist"], ""),
    "username":               (["entry0/user/name"], ""),
    "LCname":                 (["entry0/user/namelocalcontact"], ""),
    "proposal":               (["entry0/user/proposal"], ""),
    "userlist":               (["entry0/user/usernamelist"], ""),
    # Times
    "start_time": (["entry0/start_time"], ""),
    "end_time":   (["entry0/end_time"], ""),
    "time":       (["entry0/time", "entry0/duration"], NAN),
    "duration":   (["entry0/duration", "entry0/time"], NAN),
    "Time":       (["scan:Time", "entry0/time", "entry0/duration"], NAN),
    # 2theta read in the scan table (instrument/2theta/value is ttheta_value)
    "ttheta":     (["scan:2theta", "entry0/instrument/2theta/value"], NAN),
    # Detector
    "detsum":     (["entry0/instrument/Detector/detsum"], NAN),
    "TotalCount": (["scan:TotalCount", "entry0/instrument/Detector/detsum"], NAN),
    "detrate_file": (["entry0/instrument/Detector/detrate"], NAN),
    "detsize":    (["entry0/instrument/Detector/detsize"], NAN),
    "detheight":  (["entry0/instrument/Detector/height"], NAN),
    "detname":    (["entry0/instrument/Detector/name"], ""),
    "detradius":  (["entry0/instrument/Detector/rayon_det"], NAN),
    "detsample":  (["entry0/instrument/Detector/sample_distance"], NAN),
    "dettype":    (["entry0/instrument/Detector/type"], ""),
    "det_angular_width": (["entry0/instrument/Detector/angular_width"], NAN),
    "det_data_ordering": (["entry0/instrument/Detector/data_ordering"], ""),
    # Sample environment
    "SampleTemp":   (["scan:SampleTemp", "entry0/sample/temperature"], NAN),
    "RegTemp":      (["entry0/sample/regulation_temperature"], NAN),
    "RegTemp2":     (["entry0/sample/regulation_temperature2"], NAN),
    "SetPTemp":     (["entry0/sample/setpoint_temperature"], NAN),
    "SetPTemp2":    (["entry0/sample/setpoint_temperature2"], NAN),
    "use_temp":     (["entry0/sample/use_temp"], NAN),
    "Pcloche":      (["entry0/sample/Pcloche"], NAN),
    "pressure":     (["entry0/sample/pressure"], NAN),
    "SetPPressure": (["entry0/sample/setpoint_pressure"], NAN),
    # Monitor
    "mode":          (["entry0/mode"], NAN),
    "modestring":    (["entry0/modestring"], ""),
    "preset":        (["entry0/preset"], NAN),
    "Monitor":       (["scan:Monitor", "entry0/monitor/monsum", "entry0/monitor/data"], NAN),
    "monitor":       (["entry0/monitor/data", "entry0/monitor/monsum", "scan:Monitor"], NAN),
    "monmode":       (["entry0/monitor/mode"], NAN),
    "monmodestring": (["entry0/monitor/modestring"], ""),
    "monsum":        (["entry0/monitor/monsum", "entry0/monitor/data"], NAN),
    "monrate_file":  (["entry0/monitor/monrate"], NAN),
    "monpreset":     (["entry0/monitor/preset"], NAN),
    # Monochromator
    "DM_value":  (["entry0/instrument/Monochromator/dm"], NAN),
    "EI_value":  (["entry0/instrument/Monochromator/ei"], NAN),
    "EI_target": (["entry0/instrument/Monochromator/target_ei"], NAN),
    # Scan information
    "actual_step":     (["entry0/instrument/ScanInfo/actual_step"], NAN),
    "controller_list": (["entry0/instrument/ScanInfo/controller_list"], ""),
    "nb_controller":   (["entry0/instrument/ScanInfo/nb_controller"], NAN),
    "first_index":     (["entry0/instrument/ScanInfo/first_index"], NAN),
    "first_central":   (["entry0/instrument/ScanInfo/first_central_value"], NAN),
    "first_start":     (["entry0/instrument/ScanInfo/first_start_value"], NAN),
    "first_step":      (["entry0/instrument/ScanInfo/first_step_value"], NAN),
    "first_width":     (["entry0/instrument/ScanInfo/first_width_value"], NAN),
    "nb_step":         (["entry0/instrument/ScanInfo/nb_step"], NAN),
    "second_index":    (["entry0/instrument/ScanInfo/second_index"], NAN),
    "scan_total_steps": (["entry0/data_scan/total_steps"], NAN),
    "scan_actual_step": (["entry0/data_scan/actual_step"], NAN),
}

#: Number of D4 detectors (microstrip blocks) and cells per detector
N_DETECTORS = 9
CELLS_PER_DETECTOR = 64


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _decode(v: Any) -> Any:
    """bytes -> str (UTF-8), numpy scalar -> Python scalar, other unchanged."""
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        return bytes(v).decode("utf-8", errors="ignore").strip("\x00").strip()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _first(value: Any) -> Any:
    """First element of a dataset value, decoded (scalars returned as they are)."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    return _decode(value)


def _to_number(v: Any) -> Any:
    """Integers and floats as float (as the previous reader), others unchanged."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v


# -------------------------
# NeXusData class
# -------------------------
class NeXusData:
    """Content of a NeXus (.nxs / HDF5) file.

    All datasets are read when the object is created and the file is closed
    immediately.  ``data[path]`` holds the value of each dataset (byte
    strings decoded to str) and ``attrs[path]`` its HDF5 attributes.

    The physical quantities (Monitor, SampleTemp, ...) are added as
    attributes by :func:`read_nexus`.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None                  # kept for compatibility (always closed)
        self.data: Dict[str, Any] = {}
        self.attrs: Dict[str, dict] = {}
        self.read_error: Optional[str] = None
        try:
            self._load_file()
        except Exception as e:            # noqa: BLE001 - reported, not raised
            self.read_error = str(e)
            logger.warning("Error reading %s: %s", filepath, e)

    def _load_file(self) -> None:
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"NeXus file not found: {self.filepath}")
        with h5py.File(self.filepath, "r") as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = obj[()]
                        if isinstance(arr, (bytes, bytearray, np.bytes_)):
                            arr = _decode(arr)
                        elif isinstance(arr, np.ndarray) and arr.dtype.kind in {"S", "O"}:
                            arr = np.array([_decode(a) for a in arr.flat],
                                           dtype=object).reshape(arr.shape)
                        self.data[name] = arr
                        try:
                            self.attrs[name] = {k: _decode(v) for k, v in obj.attrs.items()}
                        except Exception:    # noqa: BLE001
                            self.attrs[name] = {}
                    except Exception as e:   # noqa: BLE001
                        logger.warning("Failed reading dataset %s: %s", name, e)
            f.visititems(visitor)

    def get(self, path: str, default: Any = KeyError) -> Any:
        """Value of a dataset.  Raises KeyError if missing, unless ``default`` is given."""
        if path not in self.data:
            if default is KeyError:
                raise KeyError(f"Dataset not found: {path}")
            return default
        return self.data[path]

    def has(self, path: str) -> bool:
        """True if the dataset exists in the file."""
        return path in self.data

    def keys(self) -> List[str]:
        """List of dataset paths."""
        return list(self.data.keys())

    def __repr__(self) -> str:
        return f"<NeXusData {os.path.basename(self.filepath)}: {len(self.data)} datasets>"

    def pretty_print(self, max_lines: int = 20) -> None:
        """Print the first ``max_lines`` datasets (path, shape, dtype)."""
        print(f"NeXus file: {self.filepath}")
        print(f"Total datasets: {len(self.data)}")
        for k in list(self.data.keys())[:max_lines]:
            val = self.data[k]
            print(f"{k:60s} | shape={getattr(val, 'shape', None)} | "
                  f"dtype={getattr(val, 'dtype', type(val).__name__)}")

    def close(self):
        """Nothing to do (the file is closed after reading); kept for compatibility."""
        self.file = None


# -------------------------
# Scanned variables
# -------------------------
def scanned_variables(nx: NeXusData) -> Dict[str, float]:
    """Values of the scanned variables of a numor (first scan step).

    Returns a dict whose keys are the lower-case ``label``, ``name`` and
    ``property`` of each row (all three point to the same value), e.g.
    'totalcount', 'detector1', 'time', 'count', 'monitor', 'monitor1',
    'sampletemp', 'position', '2theta', 'omega'...

    Old files without ``variables_names`` use :data:`LEGACY_SCAN_ROWS`.
    """
    table = nx.get("entry0/data_scan/scanned_variables/data", None)
    if table is None:
        return {}
    table = np.asarray(table, dtype=float)
    if table.ndim == 1:
        table = table[:, None]
    values = table[:, 0]
    out: Dict[str, float] = {}

    base = "entry0/data_scan/scanned_variables/variables_names/"
    columns = [nx.get(base + k, None) for k in ("label", "name", "property")]
    columns = [c for c in columns if c is not None]
    if columns:
        # Rows described by names: a name used by a 'Position' row (e.g.
        # '2theta', 'omega') is more specific than the label 'Position'.
        for col in columns:
            for i, key in enumerate(np.ravel(col)):
                key = str(_decode(key)).strip().lower()
                if key and i < len(values) and key not in out:
                    out[key] = float(values[i])
        # 'Position' is ambiguous when several axes are scanned: drop it
        props = [str(_decode(p)).lower() for p in np.ravel(nx.get(base + "property", []))]
        if props.count("position") > 1:
            out.pop("position", None)
    elif len(values) >= len(LEGACY_SCAN_ROWS):
        for key, i in LEGACY_SCAN_ROWS.items():
            out[key.lower()] = float(values[i])
    return out


# -------------------------
# Reactor cycles
# -------------------------
#: D4 notebook number -> (first day, last day).  Add new notebooks here.
D4_NOTEBOOKS: Dict[str, Tuple[str, str]] = {
    "44": ("18-Jun-25", "29-Sep-25"),
    "43": ("01-Apr-24", "17-Jun-25"),
    "42": ("23-Jun-23", "31-Mar-24"),
    "40": ("12-Mar-21", "17-Jul-21"),
    "39": ("24-Aug-20", "11-Mar-21"),
}

#: Reactor cycle -> data cycle and periods (start, end, power [MW], days).
#: 2026 cycles from https://www.ill.eu (reactor cycles page, Oct. 2026);
#: their data-cycle numbers (261, 262, 263) follow the usual convention
#: (year + cycle in the year) and should be checked.
REACTOR_CYCLES: Dict[str, dict] = {
    "201":  {"Cycle": "263", "SEPD": [("01-Sep-26", "29-Oct-26", 46.2, 58)]},
    "200":  {"Cycle": "262", "SEPD": [("08-Jun-26", "03-Aug-26", 47.9, 56)]},
    "199":  {"Cycle": "261", "SEPD": [("24-Mar-26", "11-May-26", 56.0, 48)]},
    "198":  {"Cycle": "252", "SEPD": [("26-Aug-25", "07-Oct-25", 41.2, 42), ("07-Oct-25", "14-Oct-25", 55.0, 7),
                                      ("14-Oct-25", "28-Oct-25", 41.2, 14)]},
    "197":  {"Cycle": "251", "SEPD": [("05-May-25", "06-Jul-25", 42.5, 63)]},
    "196":  {"Cycle": "242", "SEPD": [("14-May-24", "09-Jul-24", 47.9, 56)]},
    "195":  {"Cycle": "241", "SEPD": [("27-Feb-24", "16-Apr-24", 54.8, 49)]},
    "194":  {"Cycle": "234", "SEPD": [("24-Oct-23", "12-Dec-23", 57.0, 47)]},
    "192t": {"Cycle": "233", "SEPD": [("22-Aug-23", "27-Sep-23", 44.0, 36)]},
    "193":  {"Cycle": "232", "SEPD": [("11-May-23", "05-Jul-23", 48.6, 55)]},
    "192":  {"Cycle": "231", "SEPD": [("01-Mar-23", "08-Mar-23", 28.0, 7), ("29-Mar-23", "18-Apr-23", 44.0, 20)]},
    "191":  {"Cycle": "213", "SEPD": [("24-Aug-21", "13-Oct-21", 54.0, 50)]},
    "190":  {"Cycle": "212", "SEPD": [("11-May-21", "14-Jul-21", 42.0, 64)]},
    "189":  {"Cycle": "211", "SEPD": [("27-Jan-21", "30-Mar-21", 43.0, 62)]},
    "188":  {"Cycle": "202", "SEPD": [("11-Aug-20", "28-Sep-20", 56.0, 48)]},
    "187":  {"Cycle": "201", "SEPD": [("09-Jan-20", "26-Feb-20", 56.0, 48)]},
    "186":  {"Cycle": "192", "SEPD": [("28-Aug-19", "15-Oct-19", 56.0, 48)]},
    "185":  {"Cycle": "191", "SEPD": [("13-Jun-19", "04-Aug-19", 56.0, 48)]},
    "184":  {"Cycle": "183", "SEPD": [("03-Sep-18", "25-Oct-18", 52.5, 52)]},
    "183":  {"Cycle": "182", "SEPD": [("22-May-18", "12-Jul-18", 52.5, 51)]},
    "182":  {"Cycle": "181", "SEPD": [("28-Feb-18", "25-Apr-18", 55.4, 56)]},
    "181":  {"Cycle": "171", "SEPD": [("19-Jan-17", "08-Mar-17", 52.5, 48)]},
}


def _parse_date(date_str: str):
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%y", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    return None


def getReactorCycle(date_str):
    """Reactor period, data cycle, power, days and D4 notebook for a date.

    Parameters
    ----------
    date_str : str
        'dd-Mmm-yy' or 'dd-Mmm-yy hh:mm:ss' (e.g. '25-Oct-26 13:11:38').

    Returns
    -------
    tuple
        (reactor_period, cycle_number, reactor_power, days, notebookD4).
        ('999', '999', 999, 999, '999') when the date is not inside a known
        cycle; notebookD4 is '00' when the date is not in a known notebook.
        Extend REACTOR_CYCLES and D4_NOTEBOOKS for new cycles.
    """
    dt = _parse_date(date_str)
    if dt is None:
        return "999", "999", 999, 999, "999"
    date = dt.date()

    notebookD4 = "00"
    for nb, (ini_str, fin_str) in D4_NOTEBOOKS.items():
        ini = datetime.strptime(ini_str, "%d-%b-%y").date()
        fin = datetime.strptime(fin_str, "%d-%b-%y").date()
        if ini <= date <= fin:                 # (previous version: fixed date 20-Sep-23)
            notebookD4 = nb
            break

    for reactor_period, info in REACTOR_CYCLES.items():
        for start_str, end_str, power, days in info["SEPD"]:
            start = datetime.strptime(start_str, "%d-%b-%y").date()
            end = datetime.strptime(end_str, "%d-%b-%y").date()
            if start <= date <= end:
                return reactor_period, info["Cycle"], power, days, notebookD4

    return "999", "999", 999, 999, notebookD4


# -------------------------
# I/O convenience function
# -------------------------
def read_nexus(filepath: str) -> NeXusData:
    """Read a D4 NeXus file and return a NeXusData with all quantities as attributes.

    Works with the current format (2025-2026) and the previous one.

    Attributes added (main ones)
    ----------------------------
    run_number, title, subtitle, username, LCname, proposal, start_time,
    end_time, time, Time, ttheta, ttheta_value/_target/_offset, TotalCount,
    SumDet1..SumDet9, detsum, detdata, Monitor, monitor, monsum, SampleTemp,
    RegTemp, SetPTemp, pressure, Pcloche, omega_*, A1_*, BSC_*, BSD_*,
    EI_value, wavelength, monrate, detrate, reactor, cycle, power, days,
    nbD4 and all motors of MOTORS as <prefix>_offset/_target/_value.

    Extra attributes
    ----------------
    scanned : dict
        Scanned variables found in the file (see :func:`scanned_variables`).
    sources : dict
        attribute -> dataset path (or 'scan:<name>', 'computed') it came from.
    missing : list
        Attributes not found in the file (set to NaN or '').
    format : str
        'scan-table' (current, rows described by names) or 'legacy' (fixed rows).

    Raises
    ------
    OSError
        Only if the file itself cannot be opened/read.
    """
    nx = NeXusData(filepath)
    if nx.read_error is not None or not nx.data:
        raise OSError(f"Cannot read {filepath}: {nx.read_error or 'no dataset'}")

    nx.scanned = scanned_variables(nx)
    nx.sources = {}
    nx.missing = []
    has_names = nx.has("entry0/data_scan/scanned_variables/variables_names/label")
    nx.format = "scan-table" if has_names else "legacy"

    def lookup(candidates):
        for c in candidates:
            if c.startswith("scan:"):
                key = c[5:].lower()
                if key in nx.scanned and np.isfinite(nx.scanned[key]):
                    return nx.scanned[key], c
            elif nx.has(c):
                v = _first(nx.data[c])
                if v is not None:
                    return v, c
        return None, None

    for attr, (candidates, default) in FIELDS.items():
        v, src = lookup(candidates)
        if src is None:
            v = default
            nx.missing.append(attr)
        nx.sources[attr] = src
        setattr(nx, attr, _to_number(v) if src else v)

    # Motors: <prefix>_offset, <prefix>_target, <prefix>_value
    for group, prefix in MOTORS.items():
        for suffix, ds in (("offset", "offset_value"), ("target", "target_value"), ("value", "value")):
            attr, path = f"{prefix}_{suffix}", f"entry0/instrument/{group}/{ds}"
            v, src = lookup([path])
            if src is None:
                v = NAN
                nx.missing.append(attr)
            nx.sources[attr] = src
            setattr(nx, attr, _to_number(v) if src else v)

    # Detector cells (all formats: (1, ncells, 1) or (1, 1, ncells))
    raw = nx.get("entry0/data_scan/detector_data/data", None)
    if raw is not None:
        # first scan step, all cells (works for (1, ncells, 1) and (1, 1, ncells))
        det = np.asarray(raw, dtype=float)
        nx.detdata = det.reshape(det.shape[0], -1)[0] if det.ndim > 1 else det.ravel()
        nx.sources["detdata"] = "entry0/data_scan/detector_data/data"
    else:
        nx.detdata = np.array([])
        nx.sources["detdata"] = None
        nx.missing.append("detdata")

    # Sums of the 9 detectors: scanned (old format) or computed from the cells
    can_compute = nx.detdata.size == N_DETECTORS * CELLS_PER_DETECTOR
    for i in range(1, N_DETECTORS + 1):
        key = f"sumdet{i}"
        if key in nx.scanned:
            val, src = nx.scanned[key], f"scan:SumDet{i}"
        elif can_compute:
            block = nx.detdata[(i - 1) * CELLS_PER_DETECTOR: i * CELLS_PER_DETECTOR]
            val, src = float(block.sum()), "computed"
        else:
            val, src = NAN, None
            nx.missing.append(f"SumDet{i}")
        setattr(nx, f"SumDet{i}", val)
        nx.sources[f"SumDet{i}"] = src

    # Defaults kept from the previous version
    if len(str(nx.LCname)) <= 1:
        nx.LCname = "Unknown"
    if len(str(nx.proposal)) <= 1:
        nx.proposal = "C-KK-PPPP"

    # Derived quantities
    nx.reactor, nx.cycle, nx.power, nx.days, nx.nbD4 = getReactorCycle(nx.start_time)
    ei = nx.EI_value
    nx.wavelength = 9.045 / np.sqrt(ei) if isinstance(ei, float) and ei > 0 else NAN   # Å
    t = nx.time
    ok_t = isinstance(t, float) and np.isfinite(t) and t > 0
    nx.monrate = nx.monitor / t if ok_t else NAN
    nx.detrate = nx.detsum / t if ok_t else NAN
    for a in ("wavelength", "monrate", "detrate", "reactor", "cycle", "power", "days", "nbD4"):
        nx.sources[a] = "computed"
    return nx


def list_datasets(filepath: str, print_out: bool = True, max_values: int = 4) -> List[Tuple[str, str, str, str]]:
    """List every dataset of a file: path, dtype, shape and first values.

    Useful to see what a file contains when the format changes::

        list_datasets("../rawdata/499980.nxs")
    """
    nx = NeXusData(filepath)
    rows = []
    for k in sorted(nx.data):
        v = nx.data[k]
        if isinstance(v, np.ndarray):
            dt, sh = str(v.dtype), str(v.shape)
            vals = v.ravel()[:max_values].tolist()
        else:
            dt, sh, vals = type(v).__name__, "()", v
        txt = str(vals)
        rows.append((k, dt, sh, txt if len(txt) < 70 else txt[:67] + "..."))
    if print_out:
        print(f"{filepath}: {len(rows)} datasets")
        for r in rows:
            print(f"{r[0]:62s} {r[1]:8s} {r[2]:12s} {r[3]}")
    return rows


def check_file(filepath: str, print_out: bool = True) -> Dict[str, Any]:
    """Show where the main quantities come from and which ones are missing.

    Returns ``{"format", "scanned", "sources", "missing"}``.
    """
    nx = read_nexus(filepath)
    main = ["run_number", "start_time", "end_time", "subtitle", "Time", "Monitor", "TotalCount",
            "ttheta", "ttheta_value", "omega_value", "SampleTemp", "RegTemp", "SetPTemp",
            "Pcloche", "pressure", "SumDet1", "detdata", "EI_value", "wavelength",
            "BSC_value", "BSD_value"]
    if print_out:
        print(f"File   : {filepath}")
        print(f"Format : {nx.format}")
        print("Scanned variables: " + ", ".join(f"{k}={v:g}" for k, v in nx.scanned.items()))
        print(f"\n{'Variable':<14} {'Value':>22}  Source")
        print("-" * 78)
        for a in main:
            v = getattr(nx, a, None)
            if isinstance(v, np.ndarray):
                vs = f"array{v.shape}"
            elif isinstance(v, float):
                vs = f"{v:.6g}"
            else:
                vs = str(v)
            print(f"{a:<14} {vs[:22]:>22}  {nx.sources.get(a) or 'MISSING'}")
        print(f"\nMissing ({len(nx.missing)}): {', '.join(nx.missing) or 'none'}")
    return {"format": nx.format, "scanned": nx.scanned, "sources": nx.sources, "missing": nx.missing}


# -------------------------
# Utility / analysis functions
# -------------------------
def _numor_from_filename(fpath: str) -> float:
    """Numor from the file name: digits just before the extension (416440.nxs -> 416440)."""
    base = os.path.basename(fpath)
    m = re.search(r"(\d+)(?=\.[^.]*$)", base) or re.search(r"(\d+)", base)
    return int(m.group(1)) if m else NAN


def _as_array(values: list) -> np.ndarray:
    """List of per-numor values -> numpy array (object array if shapes differ)."""
    if any(isinstance(v, np.ndarray) and v.ndim > 0 for v in values):
        shapes = {np.shape(v) for v in values if isinstance(v, np.ndarray) and v.ndim > 0}
        if len(shapes) == 1 and all(isinstance(v, np.ndarray) and v.ndim > 0 for v in values):
            return np.array(values)
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        return arr
    try:
        return np.array(values)
    except ValueError:
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        return arr


def extract_variable(filepaths: List[str], varname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one variable from several NeXus files.

    Parameters
    ----------
    filepaths : list of str
        Paths of the .nxs files (numors).
    varname : str
        Attribute created by :func:`read_nexus` (e.g. 'Monitor', 'SampleTemp')
        or a dataset path (e.g. 'entry0/sample/temperature').

    Returns
    -------
    (numors, values) : tuple of numpy arrays, one element per file
        (NaN for files or variables that could not be read).

    Example
    -------
    >>> numors, temps = extract_variable(files, "RegTemp")      # doctest: +SKIP
    """
    numors, data = extract_multiple_variables(filepaths, [varname])
    return numors, data[varname]


def extract_multiple_variables(filepaths: List[str], varnames: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract several variables from a list of NeXus files (each opened once).

    Parameters
    ----------
    filepaths : list of str
        Paths of the .nxs files (numors).
    varnames : list of str
        Attributes created by :func:`read_nexus` ('Monitor', 'SampleTemp',
        'detdata', ...) or dataset paths.

    Returns
    -------
    (numors, data_dict)
        numors : numpy array, one numor per file (from the file name), even
                 for files that could not be read.
        data_dict : variable -> numpy array, one value per file (NaN when the
                 file or the variable is missing).  Arrays such as 'detdata'
                 give a 2-D array (or an object array if lengths differ).

    Example
    -------
    >>> numors, data = extract_multiple_variables(files, ["SampleTemp", "Monitor"])  # doctest: +SKIP
    """
    numors = []
    data_dict: Dict[str, list] = {var: [] for var in varnames}
    warned = set()

    for fpath in filepaths:
        numors.append(_numor_from_filename(fpath))
        base = os.path.basename(fpath)
        try:
            nx = read_nexus(fpath)
        except Exception as e:          # noqa: BLE001
            print(f"[ERROR] Failed reading {fpath}: {e}")
            for var in varnames:
                data_dict[var].append(NAN)
            continue

        for var in varnames:
            val = getattr(nx, var, None)
            if val is None:
                val = nx.data.get(var)
            if val is None or var in getattr(nx, "missing", ()):
                if (var, "missing") not in warned:
                    print(f"[WARN] Variable '{var}' not found in {base}"
                          + (" (and possibly other files)" if len(filepaths) > 1 else ""))
                    warned.add((var, "missing"))
                data_dict[var].append(NAN if val is None else val)
                continue
            if isinstance(val, np.ndarray):
                val = np.squeeze(val)
                if val.size == 1:
                    val = val.item()
            val = _to_number(_decode(val))
            data_dict[var].append(val)

    return np.array(numors), {var: _as_array(vals) for var, vals in data_dict.items()}


def mean_nonzero(values: Sequence[float]) -> float:
    """Mean of the non-zero elements of ``values`` (0.0 if none)."""
    arr = np.asarray(values, dtype=float)
    nonzero = arr[arr != 0]
    return float(nonzero.mean()) if nonzero.size > 0 else 0.0


def elapsed_time(t1: str, t2: str, fmt: str = "%d-%b-%y %H:%M:%S") -> Dict[str, float]:
    """Absolute number of seconds between two time strings ('dd-Mmm-yy HH:MM:SS').

    Returns ``{"total": seconds}`` (NaN if a string cannot be parsed).
    """
    try:
        dt1 = datetime.strptime(str(t1), fmt)
        dt2 = datetime.strptime(str(t2), fmt)
    except ValueError:
        return {"total": NAN}
    return {"total": abs((dt2 - dt1).total_seconds())}


def get_preset(input_var, scan_angles, mon_rate, angular_speed):
    """Monitor preset or total scan time for a planned 2θ scan.

    Parameters
    ----------
    input_var : float
        < 1000: desired total scan time in minutes;
        >= 1000: desired monitor counts per point.
    scan_angles : array-like
        Scan positions (degrees).
    mon_rate : float
        Monitor count rate (counts/s).
    angular_speed : float
        Detector speed (degrees/s).

    Returns
    -------
    (result, moving_time_minutes)
        input_var < 1000  -> result = monitor counts per point;
        input_var >= 1000 -> result = total scan time in seconds.
    """
    scan_angles = np.asarray(scan_angles, dtype=float)
    angular_range = np.sum(np.diff(scan_angles))          # = last - first
    moving_time = angular_range / angular_speed if angular_speed else NAN   # s

    if input_var < 1000:
        total_time = input_var * 60.0
        result = (total_time - moving_time) / len(scan_angles) * mon_rate
    else:
        result = len(scan_angles) * input_var / mon_rate + moving_time
    return result, moving_time / 60.0


def analyse_angular_motion(numors, ttheta, starttime, endtime, verbose=True):
    """Detector motion between consecutive numors.

    For each numor i > 0: moving time = start(i) - end(i-1),
    Δ2θ = 2θ(i) - 2θ(i-1), speed = |Δ2θ| / moving time (0 if no time).

    Parameters
    ----------
    numors, ttheta, starttime, endtime : sequences of equal length
        Times as 'dd-Mmm-yy HH:MM:SS'.
    verbose : bool
        Print the table and the totals.

    Returns
    -------
    dict with 'moving_time', 'delta_theta', 'angular_speed' (arrays) and
    'angular_range', 'time_movement', 'mean_speed' (floats).  Steps whose
    times cannot be parsed have NaN moving time and are left out of the totals.
    """
    n = len(numors)
    if not (len(ttheta) == len(starttime) == len(endtime) == n):
        raise ValueError("All input lists must have the same length.")
    ttheta = np.asarray(ttheta, dtype=float)

    moving_time = np.zeros(n)
    delta_theta = np.zeros(n)
    for i in range(1, n):
        moving_time[i] = elapsed_time(endtime[i - 1], starttime[i])["total"]
        delta_theta[i] = ttheta[i] - ttheta[i - 1]

    epsilon = 1e-6     # below this moving time (s) the speed is taken as 0
    with np.errstate(divide="ignore", invalid="ignore"):
        angular_speed = np.where(moving_time > epsilon, np.abs(delta_theta / moving_time), 0.0)
    angular_speed[~np.isfinite(moving_time)] = NAN

    ok = np.isfinite(moving_time) & np.isfinite(delta_theta)
    angular_range = float(np.sum(np.abs(delta_theta[ok])))
    time_movement = float(np.sum(moving_time[ok]))
    mean_speed = angular_range / time_movement if time_movement > 0 else 0.0

    if verbose:
        print("-" * 56)
        print(f'{"N":^3} | {"Numor":^6} | {"2θ":^8} | {"Δ2θ":^8} | {"Time":^7} | {"Speed":^9}')
        print("-" * 56)
        for i in range(n):
            print(f"{i:03d} | {int(numors[i]):06d} | {ttheta[i]:^8.3f} | "
                  f"{delta_theta[i]:^8.3f} | {moving_time[i]:^7.1f} | {angular_speed[i]:^9.4f}")
        print("-" * 56)
        print(f" Angular range: {angular_range:8.3f} deg")
        print(f" Movement time: {time_movement:8.1f} s")
        print(f" Mean speed   : {mean_speed:8.6f} deg/s")

    return {
        "moving_time": moving_time,
        "delta_theta": delta_theta,
        "angular_speed": angular_speed,
        "angular_range": angular_range,
        "time_movement": time_movement,
        "mean_speed": mean_speed,
    }


analyze_angular_motion = analyse_angular_motion      # American spelling


def print_scan_summary(numors, ttheta, time, monitor, total,
                       omega, A1, BSC, BSD, tempS, vacuum,
                       csv_file=None, md_file=None):
    """Formatted table of a scan, weighted means, optional text/Markdown export.

    Parameters are arrays with one value per numor.  ``vacuum`` is the belljar
    pressure (Pcloche) or, with the current files, any other pressure.

    Returns
    -------
    (means, mon_rate, det_rate)
        means = {"mean_omega", "mean_A1_det", "mean_A1_mon"}.
    """
    numors, ttheta, time, monitor, total = map(np.asarray, (numors, ttheta, time, monitor, total))
    omega, A1, BSC, BSD, tempS, vacuum = (np.asarray(a, dtype=float) for a in (omega, A1, BSC, BSD, tempS, vacuum))
    time, monitor, total = (np.asarray(a, dtype=float) for a in (time, monitor, total))

    with np.errstate(divide="ignore", invalid="ignore"):
        mon_rate = np.where(time != 0, monitor / time, np.nan)
        det_rate = np.where(time != 0, total / time, np.nan)

    header = (f' {"Numor":^6} | {"2θ":^7} | {"Time":^7} | {"Monitor":^9} | {"Total":^10} | '
              f'{"Mon rate":^9} | {"Det rate":^9} | {"Omega":^7} | {"A1":^8} | {"BS Bell":^7} | '
              f'{"BS Det":^7} | {"Temp S":^7} | {"Vacuum":^9}')
    print(header)
    print("-" * len(header))
    rows = []
    for i in range(len(numors)):
        row = [int(numors[i]), ttheta[i], time[i], monitor[i], total[i], mon_rate[i], det_rate[i],
               omega[i], A1[i], BSC[i], BSD[i], tempS[i], vacuum[i]]
        rows.append(row)
        print(f" {int(numors[i]):06d} | {ttheta[i]:7.3f} | {time[i]:7.3f} | "
              f"{monitor[i]:9.0f} | {total[i]:10.0f} | {mon_rate[i]:9.3f} | {det_rate[i]:9.3f} | "
              f"{omega[i]:7.3f} | {A1[i]:8.3f} | {BSC[i]:7.3f} | {BSD[i]:7.3f} | "
              f"{tempS[i]:7.2f} | {vacuum[i]:9.3E}")

    det_weight_sum = np.nansum(det_rate)
    mon_weight_sum = np.nansum(mon_rate)
    mean_omega = np.nansum(omega * det_rate) / det_weight_sum if det_weight_sum else np.nan
    mean_A1_det = np.nansum(A1 * det_rate) / det_weight_sum if det_weight_sum else np.nan
    mean_A1_mon = np.nansum(A1 * mon_rate) / mon_weight_sum if mon_weight_sum else np.nan

    print("\nThese values are only meaningful for rocking curve scans:")
    print(f"  - Omega mean value = {mean_omega:8.3f} deg")
    print(f"  - A1 mean value over detector rate = {mean_A1_det:8.3f} deg")
    print(f"  - A1 mean value over monitor rate  = {mean_A1_mon:8.3f} deg")

    if csv_file:
        csv_header = ["Numor", "2theta", "Time", "Monitor", "Total", "Mon_rate", "Det_rate",
                      "Omega", "A1", "BS_Bell", "BS_Det", "TempS", "Vacuum"]
        with open(csv_file, "w") as f:
            f.write("# " + " ".join(csv_header) + "\n")
            for r in rows:
                f.write("\t".join(f"{x:10.6g}" if isinstance(x, (float, np.floating)) else str(x)
                                  for x in r) + "\n")
        print(f"\nSpace-separated table saved to: {csv_file}")

    if md_file:
        md_header = ["Numor", "2θ", "Time", "Monitor", "Total", "Mon rate", "Det rate",
                     "Omega", "A1", "BS Bell", "BS Det", "Temp S", "Vacuum"]
        with open(md_file, mode="w") as f:
            f.write("| " + " | ".join(md_header) + " |\n")
            f.write("|" + " | ".join(["---"] * len(md_header)) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(f"{x:10.7g}" if isinstance(x, (float, np.floating)) else str(x)
                                          for x in r) + " |\n")
            f.write("\n**Mean values (rocking curve only):**\n\n")
            f.write(f"- Omega: {mean_omega:.3f} deg\n")
            f.write(f"- A1 (detector-weighted): {mean_A1_det:.3f} deg\n")
            f.write(f"- A1 (monitor-weighted): {mean_A1_mon:.3f} deg\n")
        print(f"Markdown table saved to: {md_file}")

    return {"mean_omega": mean_omega, "mean_A1_det": mean_A1_det,
            "mean_A1_mon": mean_A1_mon}, mon_rate, det_rate


def compute_angular_speed(delta_theta, moving_time, verbose=True):
    """Total angular range, total movement time and mean angular speed.

    Returns ``{"angular_range", "time_movement", "angular_speed"}``.
    """
    delta_theta = np.asarray(delta_theta, dtype=float)
    moving_time = np.asarray(moving_time, dtype=float)
    if len(delta_theta) != len(moving_time):
        raise ValueError("delta_theta and moving_time must have the same length.")
    angular_range = float(np.nansum(np.abs(delta_theta)))
    time_movement = float(np.nansum(moving_time))
    angular_speed = angular_range / time_movement if time_movement > 0 else 0.0
    if verbose:
        print(f" Angular range : {angular_range:10.4f} deg")
        print(f" Movement time : {time_movement:10.2f} s")
        print(f" Angular speed : {angular_speed:10.6f} deg/s")
    return {"angular_range": angular_range, "time_movement": time_movement,
            "angular_speed": angular_speed}


def analyze_scan(numors, monitor, time, scan_angles, angular_speed, input_values):
    """Run get_preset() for several inputs and print a compact table.

    Monitor rate = total monitor / total time of the given numors.
    ``input_values``: times in minutes (< 1000) or monitor counts (>= 1000).
    """
    monitor = np.asarray(monitor, dtype=float)
    time = np.asarray(time, dtype=float)
    mon_rate = np.nansum(monitor) / np.nansum(time)

    print(f"\nComputed monitor rate: {mon_rate:.3f} counts/s\n")
    print("Table showing the monitor counts required to complete a scan in a given time, "
          "for the following angles:")
    print("2θ =", list(scan_angles), "in deg.")
    print("Loss: Fraction of time that the detector is moving with respect to the total scanning time.\n")
    print("-" * 62)
    print(f"{'Input':^17} | {'Preset Result':^31} | {'Loss':^6}")
    print("-" * 62)
    for input_var in input_values:
        result_preset, moving_preset = get_preset(input_var, scan_angles, mon_rate, angular_speed)
        if input_var < 1000:
            print(f"{input_var:10.0f} {'min':^6} | Monitor={result_preset:8.0f} (move={moving_preset:3.0f} min) | "
                  f"{moving_preset / input_var * 100:4.1f}% ")
        else:
            print(f"{input_var:10.0f} {'counts':^6} | Time={result_preset / 60:4.0f} min (move={moving_preset:3.0f} min )   | "
                  f"{moving_preset / result_preset * 60 * 100:4.1f}% ")
    print("-" * 62)


analyse_scan = analyze_scan                                 # British spelling


def _f(v, fmt):
    """Format a number, or 'n/a' if it is NaN / not a number."""
    try:
        return format(float(v), fmt) if np.isfinite(float(v)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def show_single_numor(numor):
    """Print the main information of one numor (a NeXusData from read_nexus)."""
    n = numor
    run = int(n.run_number) if isinstance(n.run_number, float) and np.isfinite(n.run_number) else n.run_number
    print(f"Numor:               {run}  \t\t\t Proposal: \t\t {n.proposal}")
    print(f"Title:               {n.title}")
    print(f"User:                {n.username}    \t\t LC: \t\t\t {n.LCname}")
    print(f"Sample:              {n.subtitle}")
    print(f"Starting date/time:  {n.start_time}  \t Ending date/time: \t {n.end_time}")
    print(f"D4 Notebook:         {n.nbD4}")
    print(f"Reactor cycle:       {n.reactor}     \t\t\t Data cycle: \t\t {n.cycle}")
    print(f"Reactor power:       {n.power} MW     \t\t Duration: \t\t {n.days} days")
    print(f"Incident energy:     {_f(n.EI_value, '.2f')} meV \t\t Incident wavelength: \t {_f(n.wavelength, '.4f')} Å")
    print(f"2θ requested:        {_f(n.ttheta_target, '.3f')} deg  \t\t 2θ read: \t\t {_f(n.ttheta_value, '.3f')} deg")
    print(f"Counting time:       {_f(n.time, '.3f')} s \t\t Mode: \t\t\t {n.modestring}")
    print(f"Detector counts:     {_f(n.TotalCount, '.0f')} counts \t\t Detector rate: \t {_f(n.detrate, '.3f')} c/s")
    print(f"Monitor counts:      {_f(n.Monitor, '.0f')} counts \t\t Monitor rate: \t\t {_f(n.monrate, '.3f')} c/s")
    print(f"Sample T:            {_f(n.SampleTemp, '.3f')} K \t\t Regulation T: \t\t {_f(n.RegTemp, '.3f')} K")
    print(f"Set-point T:         {_f(n.SetPTemp, '.3f')} K \t\t P Belljar: \t\t {_f(n.Pcloche, '.3E')} mbar")
    print(f"Pressure:            {_f(n.pressure, '.3f')}     \t\t Set-point P: \t\t {_f(n.SetPPressure, '.3f')}")
    print(f"Omega:               {_f(n.omega_value, '.3f')} deg \t\t z: \t\t\t {_f(n.z_value, '.3f')}")
    print(f"BSC:                 {_f(n.BSC_value, '.3f')} mm \t\t BSD: \t\t\t {_f(n.BSD_value, '.3f')} mm")
    src = " (computed from cells)" if n.sources.get("SumDet1") == "computed" else ""
    print(f"Detector sums{src}:")
    print(f"Detector1: \t {_f(n.SumDet1, '.0f')} \t Detector2: \t {_f(n.SumDet2, '.0f')} \t Detector3: \t {_f(n.SumDet3, '.0f')}")
    print(f"Detector4: \t {_f(n.SumDet4, '.0f')} \t Detector5: \t {_f(n.SumDet5, '.0f')} \t Detector6: \t {_f(n.SumDet6, '.0f')}")
    print(f"Detector7: \t {_f(n.SumDet7, '.0f')} \t Detector8: \t {_f(n.SumDet8, '.0f')} \t Detector9: \t {_f(n.SumDet9, '.0f')}")


def print_numor_summary(numors, cycle, startime, endtime, notebookD4, user, LC, proposal, subtitle,
                        output_file=None, md_file=None):
    """Table of numor metadata (cycle, times, notebook, user, LC, proposal, subtitle).

    Optionally saved as fixed-width text (``output_file``) and/or Markdown (``md_file``).
    """
    def as_int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    columns = [("Numor", 6), ("Cycle", 5), ("Starting Time", 18), ("Ending Time", 18), ("NBD4", 4),
               ("User", 12), ("LC", 8), ("Proposal", 10), ("Subtitle", 30)]
    rows = []
    for i in range(len(numors)):
        sub = str(subtitle[i])
        if len(sub) > 30:
            sub = sub[:27] + "..."
        rows.append([f"{as_int(numors[i]):06d}", f"{as_int(cycle[i]):^5d}", f"{str(startime[i]):<18}",
                     f"{str(endtime[i]):<18}", f"{as_int(notebookD4[i]):^4d}", f"{str(user[i]):^12}",
                     f"{str(LC[i]):^8}", f"{str(proposal[i]):^10}", f"{sub:^30}"])

    total_width = sum(width + 3 for _, width in columns) + 1
    sep_line = "-" * total_width
    header_line = "| " + " | ".join(f"{name:^{width}}" for name, width in columns) + " |"
    table_text = "\n".join(
        [sep_line, header_line, sep_line]
        + ["| " + " | ".join(f"{val:<{width}}" for val, (_, width) in zip(row, columns)) + " |" for row in rows]
        + [sep_line])
    print(table_text)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(table_text + "\n")
        print(f"\nTable exported successfully to: {output_file}")
    if md_file:
        headers = [name for name, _ in columns]
        md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        md += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        print(f"Markdown table exported successfully to: {md_file}")


# -------------------------
# Command line: python nx_reader_D4.py FILE [FILE ...]
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Check how nx_reader_D4 reads D4 NeXus files.")
    p.add_argument("files", nargs="+", help=".nxs files")
    p.add_argument("--datasets", action="store_true", help="list every dataset of the first file")
    p.add_argument("--show", action="store_true", help="print show_single_numor for each file")
    a = p.parse_args()
    if a.datasets:
        list_datasets(a.files[0])
        print()
    check_file(a.files[0])
    if a.show:
        for fp in a.files:
            print()
            show_single_numor(read_nexus(fp))
