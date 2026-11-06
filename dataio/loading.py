"""
Readers of simple column ASCII data files.

Position in the processing chain
--------------------------------
Used after the regrouping, to read back x-y-error files: diffractograms in angle or
Q (e.g. the ``.adat``/``.qdat`` files written by ``d4creg_outputs``, the output of
CORRECT), S(Q) files, and any other 2- or 3-column file.

Functions
---------
read_xye
    Read a 2- or 3-column file into three 1-D arrays ``(x, y, e)``.
read_3col
    Read a 2- or 3-column file into one ``(n, 3)`` array.

Conventions
-----------
- Any line that contains ``#`` ANYWHERE is skipped entirely (header lines; but
  also a data line with a trailing ``# comment``).
- Empty (or white-space only) lines are skipped.
- Columns are separated by any white space. Column 1 is x, column 2 is y; the
  error is read from column 3 only when the line has exactly 3 columns. A line
  with 2 columns, or with 4 or more columns, gets an error of 0.
- A missing third column, or a third column equal to ``"i"`` or ``"o"`` (flags
  used in some ILL/CORRECT files), gives an error of 0.
- No units are assumed; the values are returned as read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def read_xye(
    filename: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Read an ASCII file with 2 or 3 columns into three 1-D arrays (x, y, e).

    Legacy `read_xye` implementation.

    The file is read line by line. A line is skipped if it contains ``#``
    anywhere, or if it is empty after stripping white space. Otherwise it is split
    on white space and:

    - column 1 -> x, column 2 -> y (converted with ``float``);
    - if the line has exactly 3 columns: the error is ``float(column 3)``, or 0.0
      if column 3 is the flag ``"i"`` or ``"o"``;
    - if the line has any other number of columns (2, but also 4 or more), the
      error is 0.0 (extra columns are ignored).

    Parameters
    ----------
    filename : str
        Path of the file to read (opened with the platform default encoding).

    Returns
    -------
    x, y, e : numpy.ndarray of float64, shape (n,)
        Abscissas, ordinates and errors of the ``n`` data lines, in the units of
        the file.

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist.
    ValueError
        If a data line has a non-numeric x, y or error (other than ``i``/``o``).
    IndexError
        If a data line (without ``#``) has a single column.

    Notes
    -----
    Prints ``"The data file <filename> read with no errors. Number of data = <n>"``.

    Changes from the legacy code (audit 2026-11, C27): legacy removed the last
    character of each line (``[:-1]``) assuming it was a newline, so a final line
    without newline lost its last digit. Lines are now stripped.

    See Also
    --------
    read_3col : same parsing, returned as one (n, 3) array.
    toscana.dataio.saving.saveFile_xye : writes files in this format.
    """
    with open(filename, "r") as data:
        x: list[float] = []
        y: list[float] = []
        e: list[float] = []

        for dataline in data.readlines():
            if "#" not in dataline:
                # Bug fix vs legacy: `[:-1]` dropped the last character of a
                # final line without a trailing newline.
                row = dataline.strip()
                if len(row) > 0:
                    columns = row.split()
                    x.append(float(columns[0]))
                    y.append(float(columns[1]))
                    if len(columns) == 3:
                        if (columns[2] == "i") or (columns[2] == "o"):
                            e.append(float(0.0))
                        else:
                            e.append(float(columns[2]))
                    else:
                        e.append(float(0.0))

    print(
        "The data file {} read with no errors. Number of data = {}".format(
            filename, len(x)
        )
    )

    xa: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    ya: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
    ea: NDArray[np.float64] = np.asarray(e, dtype=np.float64)
    return xa, ya, ea


def read_3col(filename: str) -> NDArray[np.float64]:
    """
    Read an ASCII file with 2 or 3 columns into one (n, 3) array.

    read_3col function

    Opens a file and read it line by line. This function assumes it is a 3-columns
    file. The lines containig the symbol # are ignored. Be careful, # could be
    anywhere in the line! The empty lines are also ignored.

    The parsing rules are those of `read_xye`: column 1 -> x, column 2 -> y; the
    error is column 3 if the line has exactly 3 columns (0.0 if that column is
    ``"i"`` or ``"o"``), and 0.0 otherwise (2 columns, or more than 3 columns).

    Parameters
    ----------
    filename : str
        The path to the file to read (platform default encoding).

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        A 3-column matrix: column 0 = x (abscissas), column 1 = y (ordinates),
        column 2 = error of y; units as in the file. For an empty file the array
        has shape (0, 3). The dtype is float64.

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist.
    ValueError
        If a data line has a non-numeric x, y or error (other than ``i``/``o``).
    IndexError
        If a data line (without ``#``) has a single column.

    Notes
    -----
    Unlike `read_xye`, prints nothing.

    See Also
    --------
    read_xye : same parsing, returned as three 1-D arrays.
    toscana.dataio.saving.saveFile_3col : writes files in this format.
    """
    with open(filename, "r") as data:
        # Creating the lists that will contain abcissas, ordinates and errors
        x = []
        y = []
        e = []

        # Reading the file line by line
        for dataline in data.readlines():
            if "#" not in dataline:  # Only the lines without # are treated.
                row = dataline.strip(" ")
                # Handle potential \n at the end if not handled by strip
                row = row.strip()
                if len(row) > 0:  # Only the no-empty lines are treated
                    columns = row.split()  # This method split the line using the spaces
                    x.append(float(columns[0]))
                    y.append(float(columns[1]))
                    if len(columns) == 3:
                        if (columns[2] == "i") or (columns[2] == "o"):
                            e.append(float(0.0))
                        else:
                            e.append(float(columns[2]))
                    else:
                        e.append(float(0.0))

    # Converts the lists in arrays
    xa = np.array(x)
    ya = np.array(y)
    ea = np.array(e)
    # Reshapes the arrays as 2D arrays, but with 1 column
    xa = xa.reshape(xa.shape[0], 1)
    ya = ya.reshape(ya.shape[0], 1)
    ea = ea.reshape(ea.shape[0], 1)
    # Concatenates the arrays to have a 3-column matrix
    data_out = np.concatenate((xa, ya), axis=1)
    data_out = np.concatenate((data_out, ea), axis=1)
    return data_out
