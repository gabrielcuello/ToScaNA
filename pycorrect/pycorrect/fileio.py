"""
fileio.py  --  Python translation of  fileio.f  (CORRECT 2.24, "lib" folder)
=============================================================================

Reading and writing of the multi-block data files used by CORRECT.

TEXT format (one file holds many "blocks" = spectra), as written by
APPEND_FILE and read by READ_FILE:

     (blank line)
     Block  1
     ========
     (blank line)
     Instrument: D4
     User      : ...
     Run number:        12345
     Spectrum  :            1
     Title     : ...
     Run date  : ...
     X caption : Scattering angle (degrees)
     Y caption : Counts
     Histogram :            F
     Points    :          500
       x1  y1  e1
       ...
       (one extra x value if Histogram = T: the upper bin edge)

DATA format: per block, a line with N, a title line, then N lines "x y"
(errors are set to 0.01 as in the Fortran).

GENIE (binary) format: VAX only, not supported on PC nor here.

Fortran -> Python correspondence
--------------------------------
READ_FILE   -> read_file()   returns a `Block` object
OPEN_FILE   -> DataWriter(...)        (constructor opens the file)
APPEND_FILE -> DataWriter.append(...)
CLOSE_FILE  -> DataWriter.close()     (also usable as a `with` context)
WRITE_FILE  -> write_file()
GET_INPUT   -> get_input()   (prompt and read a line from the keyboard)
PC_INQUIRE  -> pc_inquire()  (add default extension '.com' to a file name)
GET_SECTION -> not translated (VAX raw-file access, dead code on PC)

Standard library / NumPy used
-----------------------------
os.path.exists / splitext : file existence, split name and extension
open(..., 'r'/'w')        : text file access; `with` closes it automatically
numpy.array(..., float)   : convert lists to arrays
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

_CACHE: dict = {}       # parsed files, so that block n is not re-read n times


@dataclass
class Block:
    """One spectrum read from a file (all the output arguments of READ_FILE)."""
    x: np.ndarray
    y: np.ndarray
    e: np.ndarray
    histogram: bool = False          # Fortran: histogram_mode
    x_edge: float | None = None      # extra x value of a histogram (xx(nn+1))
    instrument: str = ""
    user: str = ""
    nrun: int = 0
    nspec: int = 0
    title: str = ""
    run_date: str = ""
    fnx: str = ""                    # X caption
    fny: str = ""                    # Y caption

    @property
    def npts(self) -> int:
        return len(self.y)

    def x_with_edge(self) -> np.ndarray:
        """x array of length npts+1 for histograms (xx(1..nn+1) in Fortran)."""
        if self.histogram and self.x_edge is not None:
            return np.append(self.x, self.x_edge)
        return self.x


def _value(line: str) -> str:
    """Value of a header line 'Label     : value'.  The Fortran reads fixed
    columns (format 13x,a).  Files produced by PREP2/append_data put the value
    one column earlier, so we take the text after the first ':' instead."""
    if ":" in line[:14]:
        return line.split(":", 1)[1].strip()
    return line[13:].strip()


def _parse_text(fname: str):
    """Parse a whole TEXT-format file into a list of Block."""
    with open(fname) as fh:
        lines = fh.read().splitlines()
    blocks, i, n = [], 0, len(lines)
    while i < n:
        # find the next 'Instrument' header line (4 lines after start of block)
        while i < n and not lines[i].lstrip().lower().startswith("instrument"):
            i += 1
        if i >= n:
            break
        hdr = [_value(lines[i + k]) for k in range(10)]
        i += 10
        hist = hdr[8].strip().upper().lstrip(".").startswith("T")
        npts = int(float(hdr[9]))
        data = []
        while len(data) < npts:
            vals = lines[i].replace(",", " ").split()
            i += 1
            if vals:
                data.append([float(v.replace("D", "E").replace("d", "e")) for v in vals[:3]])
        arr = np.array(data, float)
        edge = None
        if hist:
            while not lines[i].split():
                i += 1
            edge = float(lines[i].split()[0])
            i += 1
        e = arr[:, 2] if arr.shape[1] > 2 else np.full(npts, 0.01)
        blocks.append(Block(arr[:, 0], arr[:, 1], e, hist, edge,
                            hdr[0], hdr[1], int(float(hdr[2] or 0)),
                            int(float(hdr[3] or 0)), hdr[4], hdr[5], hdr[6], hdr[7]))
    return blocks


def _parse_data(fname: str):
    """Parse a DATA-format file (N / title / N lines 'x y')."""
    with open(fname) as fh:
        lines = [l for l in fh.read().splitlines()]
    blocks, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        npts = int(float(lines[i].split()[0]))
        title = lines[i + 1]
        i += 2
        xy = []
        while len(xy) < npts:
            v = lines[i].split()
            i += 1
            if v:
                xy.append([float(v[0]), float(v[1])])
        xy = np.array(xy)
        blocks.append(Block(xy[:, 0], xy[:, 1], np.full(npts, 0.01), title=title))
    return blocks


def read_file(fname: str, block: int, fmt: str = "TEXT") -> Block:
    """Read block number `block` (1-based, as in Fortran) of file `fname`.

    Fortran: SUBROUTINE READ_FILE(XX,YY,EE,NN,HISTOGRAM_MODE,INSTRUMENT,USER,
             NRUN,NSPEC,TITLE,RUN_DATE,FNX,FNY,FILE,N,FORMAT,IERROR)
    Errors raise exceptions instead of IERROR codes:
        FileNotFoundError (ierr=2), IndexError "no block n" (ierr=4),
        ValueError for an invalid format (ierr=6) or a reading error (ierr=5).
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No such file {fname}")
    fmt = fmt.upper()
    key = (os.path.abspath(fname), fmt, os.path.getmtime(fname))
    if key not in _CACHE:
        if fmt == "TEXT":
            _CACHE[key] = _parse_text(fname)
        elif fmt == "DATA":
            _CACHE[key] = _parse_data(fname)
        else:
            raise ValueError(f"{fmt} is not a valid format (GENIE is VAX only)")
    blocks = _CACHE[key]
    if block < 1 or block > len(blocks):
        raise IndexError(f"No block {block} in file {fname}")
    b = blocks[block - 1]
    # return a copy so that the caller may modify the arrays
    return Block(b.x.copy(), b.y.copy(), b.e.copy(), b.histogram, b.x_edge,
                 b.instrument, b.user, b.nrun, b.nspec, b.title, b.run_date,
                 b.fnx, b.fny)


class DataWriter:
    """Multi-block output file.

    Fortran: OPEN_FILE / APPEND_FILE / CLOSE_FILE with COMMON /write_data_file/
    (write_block, max_npts, file_format).  On the PC version the format is
    always forced to TEXT, and the program STOPS if the file already exists
    (here: FileExistsError unless overwrite=True).
    """

    def __init__(self, fname: str, fmt: str = "TEXT", overwrite: bool = False):
        if os.path.exists(fname) and not overwrite:
            raise FileExistsError(f"An output file already exists: {fname}")
        self.fmt = fmt.upper()
        self.block = 1                       # Fortran write_block
        self.fh = open(fname, "w")
        self.fname = fname

    def append(self, x, y, e, *, histogram=False, instrument="", user="",
               nrun=0, nspec=0, title="", run_date="", fnx="", fny=""):
        """Write the next block.  Fortran: SUBROUTINE APPEND_FILE.

        If histogram=True, `x` must have len(y)+1 values (bin edges); if it
        does not, the block is written in point mode.  (In the 2-axis Fortran
        the histogram flag of the sample was copied to the output with a
        garbage extra x value; writing point mode is the sane equivalent.)
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        e = np.asarray(e, float)
        n = len(y)
        if histogram and len(x) != n + 1:
            histogram = False
        fh = self.fh
        if self.fmt == "TEXT":
            fh.write(f"\n Block {self.block:2d}\n ========\n\n")
            fh.write(f" Instrument: {instrument.strip()}\n")
            fh.write(f" User      : {user.strip()}\n")
            fh.write(f" Run number: {nrun:12d}\n")
            fh.write(f" Spectrum  : {nspec:12d}\n")
            fh.write(f" Title     : {title.strip()}\n")
            fh.write(f" Run date  : {run_date.strip()}\n")
            fh.write(f" X caption : {fnx.strip()}\n")
            fh.write(f" Y caption : {fny.strip()}\n")
            fh.write(f" Histogram : {'T' if histogram else 'F':>12s}\n")
            fh.write(f" Points    : {n:12d}\n")
            for i in range(n):        # Fortran list-directed WRITE(4,*)
                fh.write(f"  {x[i]:15.7E}  {y[i]:15.7E}  {e[i]:15.7E}\n")
            if histogram:
                fh.write(f"  {x[n]:15.7E}\n")
        elif self.fmt == "DATA":
            fh.write(f" {n}\n{title}\n")
            for i in range(n):
                fh.write(f"  {x[i]:15.7E}  {y[i]:15.7E}\n")
        self.block += 1

    def close(self):
        """Fortran: SUBROUTINE CLOSE_FILE."""
        if not self.fh.closed:
            self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_file(fname, x, y, e, fmt="TEXT", overwrite=False, **header):
    """Single-block file.  Fortran: SUBROUTINE WRITE_FILE."""
    with DataWriter(fname, fmt, overwrite) as w:
        w.append(x, y, e, **header)


def get_input(prompt: str) -> str:
    """Fortran: SUBROUTINE GET_INPUT (PC replacement of VAX LIB$GET_INPUT)."""
    try:
        return input(" " + prompt)
    except EOFError:
        return "quit"


def pc_inquire(name: str, default_ext: str = ".com") -> str:
    """Fortran: SUBROUTINE PC_INQUIRE -- append default_ext if the name has no
    extension."""
    name = name.strip().split()[0] if name.strip() else name
    return name if os.path.splitext(name)[1] else name + default_ext
