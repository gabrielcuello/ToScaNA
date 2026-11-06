"""
Containers for ASCII x-y(-e) data files, with simple plotting.

Used in the notebooks to look at regrouped diffractograms (``.reg``,
``.adat``, ``.qdat``...), structure factors and real-space functions, and by
the Ni calibration (`DataXYE` performs an automatic peak search in
x ∈ [5, 40], the 2θ range of the Ni reflections).

Public names
------------
DataXYE
    Reads a 2- or 3-column file with header lines; statistics, peak search,
    ``plot``, ``show``, ``header``.
DataXYE_old
    Subclass of `DataXYE` without changes, kept for the legacy name.
DataXYE2
    Minimal container read with ``np.loadtxt``; ``plot`` with axis type and
    overlay option.
DataXYE0
    Container for 3-column files read with ``np.loadtxt``; ``plot``,
    ``show``, ``save``.

Conventions
-----------
Column 0 is the abscissa x (2θ in degrees, Q in 1/Å or r in Å), column 1 the
ordinate y, column 2 its error e. matplotlib is imported only inside the
plotting methods. Plots are shown with ``plt.show()``, which blocks or
closes the figure depending on the back-end.

Changes from the legacy code (audit 2026-11, C22): `DataXYE` no longer
crashes on files without points in [5, 40], and ``DataXYE.plot`` saves the
figure before ``plt.show()``.
"""

from __future__ import annotations

import os
import statistics as st
from typing import TYPE_CHECKING

import numpy as np

from toscana.numerics.peak_finding import find_minimum_within_range, find_peaks_in_range

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DataXYE:
    """
    Read ASCII data files with 2 (X Y) or 3 (X Y E) columns (legacy behavior).

    Each line is stripped of surrounding whitespace. Empty lines are skipped;
    lines whose first character is ``#`` or ``!`` are header lines, stored
    (stripped) in `head`; every other line must have 2 or 3
    whitespace-separated numbers. A 2-column line gets the error -1.0 (a
    marker for "no error"). Comments at the end of a data line are not
    allowed.

    After reading, an automatic peak search is done in x ∈ [5, 40]
    (inclusive; the 2θ range of the Ni calibration peaks), see `peaks_x`.

    Parameters
    ----------
    filename : str
        Path of the file to read.

    Attributes
    ----------
    filename : str
        As given.
    basename : str
        `filename` without its extension (directories included), used to
        name saved figures.
    ext : str
        Extension without the dot (``"reg"``), selects the axis label.
    x, y, e : ndarray of float64, shape (N,)
        Abscissa, ordinate and error (-1.0 for 2-column lines).
    head : list of str
        Header lines, in file order.
    xave, xmin, xmax, yave, ymin, ymax, eave, emin, emax : float
        Mean, minimum and maximum of each column (the error statistics
        include the -1.0 markers).
    peaks_x : ndarray of float64
        Positions of the local maxima of y with x in [5, 40]
        (``scipy.signal.find_peaks``, default settings) whose height
        relative to the minimum of y in that range exceeds 3.5.
    peaks_y : ndarray of float64
        For those peaks, the RATIO y_peak / yminr (not the intensity).
    xminr, yminr : numpy.float64
        Position and value of the minimum of y with x in [5, 40].

    If no point has x in [5, 40], `peaks_x` and `peaks_y` are empty and
    `xminr`, `yminr` are NaN.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If a data line has a number of columns other than 2 or 3 ("Wrong
        file format: expected 2 or 3 columns."; legacy printed and exited),
        or a field is not a number; also (from ``min``) if the file has no
        data line.
    statistics.StatisticsError
        If the file has no data line (raised first, by the mean).

    Notes
    -----
    If the minimum of y in [5, 40] is 0 or negative, the ratios in
    `peaks_y` are infinite or negative and the 3.5 threshold is meaningless.

    This mirrors the final `DataXYE` definition in `legacy/lib/toscana.py`, which
    is the effective one at runtime (the legacy file defines `DataXYE` twice).

    Bug fix vs legacy (audit 2026-11, C22): files with no point in [5, 40]
    (Q- or r-space files) crashed with "argmin of an empty sequence"; the
    figure was saved after ``plt.show()`` (empty images).
    """

    filename: str
    basename: str
    ext: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    e: NDArray[np.float64]
    head: list[str]

    xave: float
    xmin: float
    xmax: float
    yave: float
    ymin: float
    ymax: float
    eave: float
    emin: float
    emax: float

    peaks_x: NDArray[np.float64]
    peaks_y: NDArray[np.float64]
    xminr: np.float64
    yminr: np.float64

    def __init__(self, filename: str):
        """
        Read `filename`; see the class docstring.

        Parameters
        ----------
        filename : str
            Path of the 2- or 3-column file.
        """
        self.filename = filename
        self.basename = os.path.splitext(filename)[0]
        self.ext = os.path.splitext(filename)[1][1:]

        x_list: list[float] = []
        y_list: list[float] = []
        e_list: list[float] = []
        head: list[str] = []

        with open(self.filename, "r") as data:
            for dataline in data.readlines():
                row = dataline.strip(" ").strip()
                if not row:
                    continue
                if row[0] in {"#", "!"}:
                    head.append(row)
                    continue

                columns = row.split()
                if len(columns) == 2:
                    x_list.append(float(columns[0]))
                    y_list.append(float(columns[1]))
                    e_list.append(-1.0)
                elif len(columns) == 3:
                    x_list.append(float(columns[0]))
                    y_list.append(float(columns[1]))
                    e_list.append(float(columns[2]))
                else:
                    # Legacy prints and exits; prefer CI-safe exception behavior.
                    raise ValueError("Wrong file format: expected 2 or 3 columns.")

        self.head = head
        self.x = np.asarray(x_list, dtype=np.float64)
        self.y = np.asarray(y_list, dtype=np.float64)
        self.e = np.asarray(e_list, dtype=np.float64)

        self.xave = float(st.mean(self.x))
        self.xmin = float(min(self.x))
        self.xmax = float(max(self.x))

        self.yave = float(st.mean(self.y))
        self.ymin = float(min(self.y))
        self.ymax = float(max(self.y))

        self.eave = float(st.mean(self.e))
        self.emin = float(min(self.e))
        self.emax = float(max(self.e))

        # Peak search in 5..40 (2θ for Ni calibration). Bug fix vs legacy:
        # files with no point in that range (e.g. Q- or r-space files)
        # crashed with "argmin of an empty sequence".
        if np.any((self.x >= 5.0) & (self.x <= 40.0)):
            self.peaks_x, self.peaks_y = find_peaks_in_range(self.x, self.y, 5.0, 40.0)
            self.xminr, self.yminr = find_minimum_within_range(self.x, self.y, 5.0, 40.0)
        else:
            self.peaks_x = np.empty(0, dtype=np.float64)
            self.peaks_y = np.empty(0, dtype=np.float64)
            self.xminr, self.yminr = np.float64(np.nan), np.float64(np.nan)

    def plot(
        self,
        file_format: str | int = 0,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> None:
        """
        Plot y versus x in a new figure, optionally save it, then show it.

        Red line with ``+`` markers (``"r-+"``), legend = file name, title
        "Data in <filename>", ordinate "Intensity (arb. units)", grid,
        figure size 9 x 6 inches.
    The abscissa label is chosen from the file extension: ``adat``,
    ``Adat``, ``reg`` -> 2θ (degrees); ``qdat``, ``Qdat``, ``Qreg``, ``soq``,
    ``SoQ`` -> Q (1/Å); ``pcf``, ``pdf``, ``tor``, ``rdf``, ``run`` -> R (Å);
    anything else -> "Abscissa".

        Parameters
        ----------
        file_format : str or int, optional
            Image format. If one of ``"png"``, ``"pdf"``, ``"jpg"``,
            ``"tiff"``, ``"svg"``, ``"jpeg"``, ``"ps"``, ``"eps"``, the
            figure is saved to ``"../regdata/" + basename + "." +
            file_format`` (relative to the current working directory) and
            "Figure saved on <path>" is printed. Any other value (default 0)
            means no file.
        xmin, xmax, ymin, ymax : float or None, optional
            Axis limits; None (default) keeps matplotlib's automatic limit.

        Raises
        ------
        FileNotFoundError
            If saving and the directory ``../regdata`` (or a sub-directory
            implied by `basename`) does not exist.

        Notes
        -----
        `basename` contains the directory part of `filename`, so for
        ``"data/x.reg"`` the figure goes to ``../regdata/data/x.png``.
        Bug fix vs legacy (C22): the figure is saved BEFORE ``plt.show()``.
        """
        # Lazy import so using the data model doesn't require matplotlib at import time.
        import matplotlib.pyplot as plt

        a_ext = ["adat", "Adat", "reg"]
        q_ext = ["qdat", "Qdat", "Qreg", "soq", "SoQ"]
        r_ext = ["pcf", "pdf", "tor", "rdf", "run"]

        plt.figure(figsize=(9, 6))
        plt.plot(self.x, self.y, "r-+", label=self.filename)

        plt.legend(loc="best")
        plt.title("Data in " + self.filename)
        plt.xlabel("Abscissa")
        if self.ext in a_ext:
            plt.xlabel(r"$2\theta$ (˚)")
        elif self.ext in q_ext:
            plt.xlabel(r"$Q$ (Å$^{-1}$)")
        elif self.ext in r_ext:
            plt.xlabel("$R$ (Å)")
        plt.ylabel("Intensity (arb. units)")
        plt.axis([xmin, xmax, ymin, ymax])  # type: ignore[arg-type]
        plt.grid(True)
        plt.tight_layout()

        # Bug fix vs legacy: save BEFORE show(); after an interactive show()
        # the figure is closed and an empty one was saved.
        if file_format in ["png", "pdf", "jpg", "tiff", "svg", "jpeg", "ps", "eps"]:
            file_fig = "../regdata/" + self.basename + "." + str(file_format)
            plt.savefig(file_fig, format=str(file_format))
            print(f"Figure saved on {file_fig}")
        plt.show()

    def show(self) -> None:
        """
        Print the data as a table, one line per point.

        Header by extension (same groups as `plot`): "Ang Intensity Error",
        "Q Intensity Error", "R Intensity Ignore", or "x y e"; then x, y, e
        right-aligned in fields of width 6, 12, 12 (full precision, so long
        numbers overflow the widths).
        """
        a_ext = ["adat", "Adat", "reg"]
        q_ext = ["qdat", "Qdat", "Qreg", "soq", "SoQ"]
        r_ext = ["pcf", "pdf", "tor", "rdf", "run"]

        if self.ext in a_ext:
            print(f"{'Ang':>6}{'Intensity':>12}{'Error':>12}")
        elif self.ext in q_ext:
            print(f"{'Q':>6}{'Intensity':>12}{'Error':>12}")
        elif self.ext in r_ext:
            print(f"{'R':>6}{'Intensity':>12}{'Ignore':>12}")
        else:
            print(f"{'x':>6}{'y':>12}{'e':>12}")

        for i in range(len(self.x)):
            print(f"{self.x[i]:>6}{self.y[i]:>12}{self.e[i]:>12}")

    def header(self, lines: int = 1000) -> None:
        """
        Print the first header lines, numbered from 1.

        Parameters
        ----------
        lines : int, optional
            Maximum number of header lines to print. Default 1000.
        """
        for i in range(min(lines, len(self.head))):
            print(f"{i + 1:>3} {self.head[i]}")


class DataXYE_old(DataXYE):
    """
    Legacy-compatible alias of `DataXYE` (kept for parity with legacy API).

    A subclass that adds nothing: same constructor, attributes and methods
    as `DataXYE`. (In the legacy file ``DataXYE_old`` was the first, older
    definition; here it is simply the current `DataXYE`.)
    """


class DataXYE2:
    """
    Minimal container of a numeric table read with ``np.loadtxt`` (legacy).

    Parameters
    ----------
    file : str
        Path of a whitespace-separated numeric file with at least 2 columns
        and 2 rows (``#`` starts a comment, NumPy default).

    Attributes
    ----------
    data : ndarray of float64, shape (N, M)
        The table.
    file, filename : str
        Both equal to `file`.
    title : str
        ``"Plotting <file>"``, default plot title.
    labelx, labely : str
        ``"Axis X"``, ``"Axis Y"``: labels used when the plot scale is not
        a, q or r.
    color : str
        ``"blue"``, default line colour.
    curve : str
        ``"-"``, default line style.
    scale : str
        ``"a"``; stored only, `plot` uses its own argument.
    superplot : bool
        False; stored only, `plot` uses its own argument.

    Raises
    ------
    OSError, ValueError
        From ``np.loadtxt`` (missing file, non-numeric data).
    """

    def __init__(self, file: str):
        """
        Load `file`; see the class docstring.

        Parameters
        ----------
        file : str
            Path of the numeric file.
        """
        self.data: NDArray[np.float64] = np.loadtxt(file, dtype=np.float64)
        self.file = file
        # Legacy uses `self.filename` in `plot()` but only sets `self.file`.
        # Defining both keeps behavior predictable.
        self.filename = file

        self.title = "Plotting " + self.file
        self.labelx = "Axis X"
        self.labely = "Axis Y"
        self.color = "blue"
        self.curve = "-"
        self.scale = "a"
        self.superplot = False

    def plot(
        self,
        scale: str = "a",
        superplot: bool = False,
        title: str | None = None,
        labelx: str | None = None,
        labely: str | None = None,
        color: str | None = None,
        curve: str | None = None,
    ) -> None:
        """
        Plot column 1 versus column 0.

        Parameters
        ----------
        scale : str, optional
            ``"a"``/``"A"`` (default): abscissa 2θ (degrees); ``"q"``/``"Q"``:
            Q (1/Å); ``"r"``/``"R"``: R (Å). The ordinate label is
            "Intensity (arb. units)". Any other value: labels `labelx` and
            `labely`.
        superplot : bool, optional
            False (default): open a new 9 x 6 figure and call ``plt.show()``
            at the end. True: draw on the current figure and do not show, to
            overlay several data sets (call ``plt.show()`` yourself).
        title, labelx, labely, color, curve : str or None, optional
            Title, axis labels, line colour and line style; None (default)
            uses the attributes of the same name.

        Raises
        ------
        IndexError
            If `data` has fewer than 2 columns or is 1-D (single-row file).

        Notes
        -----
        The legend label is `filename`. The axis limits are set to the data
        range extended by 10 % of the range on each side (computed from this
        data set only, also when overlaying). Because `title` has already
        been replaced by ``self.title`` when None, the "Data in <file>"
        fallback in the code is never used.
        """
        import matplotlib.pyplot as plt

        title = title if title is not None else self.title
        labelx = labelx if labelx is not None else self.labelx
        labely = labely if labely is not None else self.labely
        color = color if color is not None else self.color
        curve = curve if curve is not None else self.curve

        if not superplot:
            plt.figure(figsize=(9, 6))

        plt.plot(
            self.data[:, 0],
            self.data[:, 1],
            color=color,
            linestyle=curve,
            label=self.filename,
        )

        plt.legend(loc="best")
        plt.title("Data in " + self.file if title is None else title)
        plt.ylabel("Intensity (arb. units)")
        if scale == "Q" or scale == "q":
            plt.xlabel(r"$Q$ (Å$^{-1}$)")
        elif scale == "A" or scale == "a":
            plt.xlabel(r"$2\theta$ (˚)")
        elif scale == "R" or scale == "r":
            plt.xlabel("$R$ (Å)")
        else:
            plt.xlabel(labelx)
            plt.ylabel(labely)

        x_range = np.ptp(self.data[:, 0])
        y_range = np.ptp(self.data[:, 1])

        plt.axis(
            [  # type: ignore[arg-type]
                float(np.min(self.data[:, 0]) - 0.1 * x_range),
                float(np.max(self.data[:, 0]) + 0.1 * x_range),
                float(np.min(self.data[:, 1]) - 0.1 * y_range),
                float(np.max(self.data[:, 1]) + 0.1 * y_range),
            ]
        )
        plt.grid(True)
        plt.tight_layout()
        if not superplot:
            plt.show()


class DataXYE0:
    """
    Container of a 3-column (x, y, e) file read with ``np.loadtxt`` (legacy).

    Parameters
    ----------
    filename : str
        Path of a numeric file with at least 3 columns and 2 rows (``#``
        starts a comment).
    symbol : str, optional
        matplotlib format string used by `plot`. Default ``"r-+"``.
    save : bool, optional
        Legacy argument, ignored. Default False.
    scale : str, optional
        Axis type for `plot`: ``"a"`` (default, 2θ in degrees), ``"q"`` (Q
        in 1/Å), ``"r"`` (R in Å), upper or lower case; anything else gives
        the label "Abscissa".
    superplot : bool, optional
        Legacy argument, ignored here (see `plot`). Default False.
    xmin, xmax, ymin, ymax : float or None, optional
        Stored as `minx`, `maxx`, `miny`, `maxy`; not used by `plot`.

    Attributes
    ----------
    data : ndarray of float64, shape (N, M)
        The table.
    filename : str
        As given.
    ext : str
        Extension without the dot (stored only).
    x, y, e : ndarray
        Views of columns 0, 1, 2 of `data`.
    xave, xmin, xmax, yave, ymin, ymax, eave, emin, emax : float
        Mean, minimum and maximum of each column.
    scale, symbol : str
        As given.
    minx, maxx, miny, maxy : float or None
        The `xmin`... arguments.

    Raises
    ------
    OSError, ValueError
        From ``np.loadtxt``.
    IndexError
        If the table has fewer than 3 columns or a single row.
    """

    def __init__(
        self,
        filename: str,
        symbol: str = "r-+",
        save: bool = False,  # noqa: FBT001 (legacy API)
        scale: str = "a",
        superplot: bool = False,  # noqa: FBT001 (legacy API)
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
    ):
        """
        Load `filename` and compute the column statistics.

        See the class docstring for the parameters.
        """
        self.data: NDArray[np.float64] = np.loadtxt(filename, dtype=np.float64)
        self.filename = filename
        self.ext = os.path.splitext(filename)[1][1:]
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        self.e = self.data[:, 2]

        self.xave = float(st.mean(self.x))
        self.xmin = float(min(self.x))
        self.xmax = float(max(self.x))
        self.yave = float(st.mean(self.y))
        self.ymin = float(min(self.y))
        self.ymax = float(max(self.y))
        self.eave = float(st.mean(self.e))
        self.emin = float(min(self.e))
        self.emax = float(max(self.e))

        self.scale = scale
        self.symbol = symbol
        self.minx = xmin
        self.maxx = xmax
        self.miny = ymin
        self.maxy = ymax

        # Preserve legacy "save" parameter (no side effects implemented here).
        _ = save
        _ = superplot

    def plot(
        self,
        symbol: str,
        superplot: bool = False,  # noqa: FBT001 (legacy API)
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
    ) -> None:
        """
        Plot y versus x with the format string `self.symbol`.

        Prints ``"*** <self.symbol>"``, then plots with legend label
        `filename`, title "Data in <filename>", abscissa label from
        `self.scale` (see the class docstring), ordinate "Intensity (arb.
        units)", grid.

        Parameters
        ----------
        symbol : str
            Required by the legacy signature but IGNORED: `self.symbol`
            (set in the constructor) is used.
        superplot : bool, optional
            False (default): new 9 x 6 figure and ``plt.show()`` at the end.
            True: draw on the current figure without showing (overlay).
        xmin, xmax, ymin, ymax : float or None, optional
            Axis limits; None (default) keeps the automatic limit. The
            values given to the constructor are not used.
        """
        import matplotlib.pyplot as plt

        if not superplot:
            plt.figure(figsize=(9, 6))
        _ = symbol  # legacy arg is unused (self.symbol is used instead)

        print("***", self.symbol)
        plt.plot(self.x, self.y, self.symbol, label=self.filename)

        plt.legend(loc="best")
        plt.title("Data in " + self.filename)
        if self.scale == "Q" or self.scale == "q":
            plt.xlabel(r"$Q$ (Å$^{-1}$)")
        elif self.scale == "A" or self.scale == "a":
            plt.xlabel(r"$2\theta$ (˚)")
        elif self.scale == "R" or self.scale == "r":
            plt.xlabel("$R$ (Å)")
        else:
            plt.xlabel("Abscissa")
        plt.ylabel("Intensity (arb. units)")
        plt.axis([xmin, xmax, ymin, ymax])  # type: ignore[arg-type]
        plt.grid(True)
        plt.tight_layout()

        if not superplot:
            plt.show()

    def show(self) -> None:
        """
        Print x, y, e, one point per line (widths 6, 12, 12, no header).
        """
        for i in range(len(self.x)):
            print(f"{self.x[i]:>6}{self.y[i]:>12}{self.e[i]:>12}")

    def save(self, filename: str) -> None:
        """
        Write columns 0, 1, 2 of `data` to a text file.

        Each line is ``"{: 9.3f} {:18.6f} {:18.6f}"`` (x, y, e), no header.

        Parameters
        ----------
        filename : str
            Output path; created or OVERWRITTEN.

        Raises
        ------
        OSError
            If the file cannot be written.

        Notes
        -----
        Prints "File <filename> saved".
        """
        x = self.data[:, 0]
        y = self.data[:, 1]
        e = self.data[:, 2]
        with open(filename, "w") as datafile:
            for i in range(len(x)):
                datafile.write(
                    "{: 9.3f}".format(x[i])
                    + " "
                    + "{:18.6f}".format(y[i])
                    + " "
                    + "{:18.6f}".format(e[i])
                    + "\n"
                )
            print("File " + filename + " saved")
