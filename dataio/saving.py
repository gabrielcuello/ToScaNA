"""
Writers of ASCII data files (x-y-e, real-space functions, single numors).

Position in the processing chain
--------------------------------
Output side of the package:

- ``saveOneNumor`` is called by ``acquisition.getOneNumor`` during the regrouping
  (one ``.cdat`` file per numor when ``runInfo["writeNumor"] == 1``);
- ``saveFile_xye`` / ``saveFile_3col`` write any intermediate x-y-e result
  (diffractograms, S(Q), ...);
- ``saveRSCF`` and ``saveCorrelations`` write the real-space functions G(r),
  g(r), RDF(r), T(r), N(r) obtained by the Fourier transform
  (``numerics.fourier.sineFT``).

Functions
---------
saveFile_xye
    Write a 3-column file from three sequences x, y, e.
saveFile_3col
    Write a 3-column file from an (n, 3) array.
saveCorrelations
    Write a 6-column file r, G, g, RDF, T, N from six separate sequences.
saveOneNumor
    Write the corrected cells of one numor (angle, counts, error, Q, cell).
saveRSCF
    Write the (n, 6) or (n, 11) output of ``sineFT`` with a column description.

Private helpers
---------------
_RSCF_COLUMNS
    Names and descriptions of the six real-space columns.
_rscf_column_header
    Header lines describing the columns of ``saveRSCF``.

Conventions
-----------
- Every file starts with ``# <filename>``, then the user ``heading`` lines, each
  prefixed with ``# ``; data lines follow without ``#`` (readable by
  ``loading.read_xye`` / ``read_3col``).
- Files are written with the platform default encoding (the headers of
  ``saveOneNumor`` and ``saveRSCF`` contain non-ASCII characters such as Å, σ, ρ),
  in the path given (relative to the current working directory if not absolute),
  overwriting existing files.
- No units are imposed except where stated (r in Å, angles in degrees, Q in 1/Å).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def saveFile_xye(  # noqa: N802 (legacy API)
    filename: str,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    e: Sequence[float] | np.ndarray,
    heading: Sequence[str],
) -> None:
    """
    Create a 3-column ASCII file with x, y, error (legacy behavior).

    File layout: ``# <filename>``, one line ``# <text>`` per element of
    ``heading``, then ``len(x)`` lines ``x y e`` with formats ``{: 9.3f}`` (x,
    space for the sign), ``{:18.6f}`` (y), ``{:18.6f}`` (e), separated by one
    space.

    Parameters
    ----------
    filename : str
        Path of the output file (overwritten).
    x, y, e : sequence of float or numpy.ndarray
        Abscissas, ordinates and errors. Only the first ``len(x)`` elements of
        ``y`` and ``e`` are written.
    heading : sequence of str
        Header lines (without ``#`` and without newline).

    Returns
    -------
    None

    Raises
    ------
    IndexError
        If ``y`` or ``e`` is shorter than ``x``.

    Notes
    -----
    Prints ``"File <filename> saved"``. x is written with only 3 decimals.

    Parity-first copy of `legacy/lib/toscana.py::saveFile_xye`, including
    formatting and print message.
    """
    with open(filename, "w") as datafile:
        datafile.write("# " + filename + "\n")
        for i in range(len(heading)):
            datafile.write("# " + heading[i] + "\n")
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
    return


def saveFile_3col(  # noqa: N802 (legacy API)
    filename: str,
    data: np.ndarray,
    heading: Sequence[str],
) -> None:
    """
    Create a 3-column ASCII file with x, y, error from a Nx3 array (legacy behavior).

    Same layout and formats as `saveFile_xye`, with x, y, e taken from columns 0,
    1, 2 of ``data`` (further columns are ignored).

    Parameters
    ----------
    filename : str
        Path of the output file (overwritten).
    data : numpy.ndarray, shape (n, >=3)
        Columns x, y, error.
    heading : sequence of str
        Header lines (without ``#`` and without newline).

    Returns
    -------
    None

    Notes
    -----
    Prints ``"File <filename> saved"``.
    """
    x = data[:, 0]
    y = data[:, 1]
    e = data[:, 2]
    with open(filename, "w") as datafile:
        datafile.write("# " + filename + "\n")
        for i in range(len(heading)):
            datafile.write("# " + heading[i] + "\n")
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
    return


def saveCorrelations(  # noqa: N802 (legacy API)
    filename: str,
    ar: Sequence[float] | np.ndarray,
    pcf: Sequence[float] | np.ndarray,
    pdf: Sequence[float] | np.ndarray,
    rdf: Sequence[float] | np.ndarray,
    tor: Sequence[float] | np.ndarray,
    run: Sequence[float] | np.ndarray,
    heading: Sequence[str],
) -> None:
    """
    Create a 6-column ASCII file with real-space correlation functions (legacy behavior).

    File layout: ``# <filename>``, ``# <text>`` for each heading line, then one
    line per r value with format ``{:9.3f}`` for r and ``{:12.6f}`` for the five
    functions. The number of lines is that of the SHORTEST input (``zip``).

    Parameters
    ----------
    filename : str
        Path of the output file (overwritten).
    ar : sequence of float
        r in Å.
    pcf : sequence of float
        G(r) = 4πρr[g(r) − 1], Å⁻².
    pdf : sequence of float
        g(r), dimensionless.
    rdf : sequence of float
        RDF(r) = 4πρr²g(r), atoms/Å.
    tor : sequence of float
        T(r) = 4πρr·g(r), Å⁻².
    run : sequence of float
        N(r), running coordination number (atoms).
    heading : sequence of str
        Header lines (without ``#`` and without newline).

    Returns
    -------
    None

    Notes
    -----
    Prints ``"File <filename> saved"``. Unlike `saveRSCF`, no column
    description is written. (The names pcf/pdf follow the legacy code: ``pcf``
    holds G(r) and ``pdf`` holds g(r).)

    See Also
    --------
    saveRSCF : writes the same functions from the array returned by ``sineFT``.
    """
    with open(filename, "w") as datafile:
        datafile.write("# " + filename + "\n")
        datafile.writelines("# " + line + "\n" for line in heading)
        for vals in zip(ar, pcf, pdf, rdf, tor, run):
            datafile.write(
                "{:9.3f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".format(*vals)
            )
    print("File " + filename + " saved")
    return


def saveOneNumor(  # noqa: N802 (legacy API)
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    e: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
    c: Sequence[float] | np.ndarray,
    head: dict[str, Any],
    runInfo: dict[str, Any],
) -> None:
    """
    Save an individual file for a single numor (legacy behavior).

    File layout: ``# <filename>``, ``# Sample: <subtitle>``,
    ``# Starts: <startTime>  Ends: <endTime>``, counting time (s), monitor and
    detector counts with their rates (counts / counting time), a dashed line,
    the column titles ``Angle(deg) Counts Error Q(1/Å) Cell``, then one line per
    cell with formats ``{: 9.3f} {:18.6f} {:18.6f} {:9.3f} {:9.0f}``.

    Parameters
    ----------
    x : sequence of float
        2θ of every cell, degrees (as given by ``getOneNumor``: zero angle NOT
        subtracted).
    y : sequence of float
        Normalised counts.
    e : sequence of float
        Their errors.
    q : sequence of float
        Q in 1/Å.
    c : sequence of float
        Cell identifiers ``1000·bank + cell``.
    head : dict
        Metadata of the numor. Keys read: ``numor``, ``subtitle``, ``startTime``,
        ``endTime``, ``CntTime (sec)``, ``MonitorCnts``, ``TotalCnts`` (the
        latter two are dead-time corrected when called from ``getOneNumor``).
    runInfo : dict
        Key read: ``ext`` (index 3, default ``".cdat"``).

    Returns
    -------
    None

    Raises
    ------
    ZeroDivisionError
        If ``head["CntTime (sec)"]`` is a Python 0 (a NumPy scalar 0, as read
        from NeXus, gives inf with a warning instead).

    Notes
    -----
    Writes `{head["numor"]}{runInfo["ext"][3]}` in the current working directory
    (e.g. ``387229.cdat``), overwriting it, with the platform default encoding.
    Prints ``"File <filename> saved."``. NaN values (excluded cells) are written
    as ``nan``.
    """
    filename = str(head["numor"]) + str(runInfo["ext"][3])
    # Legacy uses the platform default encoding; preserve for byte-level parity.
    with open(filename, "w") as datafile:
        datafile.write("# {} \n".format(filename))
        datafile.write("# Sample: {}\n".format(head["subtitle"]))
        datafile.write(
            "# Starts: {}  Ends: {} \n".format(head["startTime"], head["endTime"])
        )
        datafile.write("# Counting time: {:8.4f} s\n".format(head["CntTime (sec)"]))
        datafile.write(
            "# Monitor: {:10.2f} counts ({:8.2f} c/s)\n".format(
                head["MonitorCnts"], head["MonitorCnts"] / head["CntTime (sec)"]
            )
        )
        datafile.write(
            "# Detector: {:10.2f} counts ({:8.2f} c/s)\n".format(
                head["TotalCnts"], head["TotalCnts"] / head["CntTime (sec)"]
            )
        )
        datafile.write("#" + 80 * "-" + "\n")
        datafile.write(
            "# Angle(deg)"
            + 6 * " "
            + "Counts"
            + 14 * " "
            + "Error"
            + 8 * " "
            + "Q(1/Å)"
            + 6 * " "
            + "Cell \n"
        )
        for i in range(len(x)):
            datafile.write(
                "{: 9.3f} {:18.6f} {:18.6f} {:9.3f} {:9.0f}".format(
                    x[i], y[i], e[i], q[i], c[i]
                )
                + "\n"
            )
        print("File {} saved.".format(filename))
    return


# Name and description of the six real-space columns written by `saveRSCF`
# (the error columns, when present, are generated by `_rscf_column_header`).
_RSCF_COLUMNS = [
    ("r", "distance (Å)"),
    ("G", "G(r) = 4πρr[g(r) - 1], pair distribution function (Å^-2)"),
    ("g", "g(r), pair correlation function (dimensionless)"),
    ("RDF", "RDF(r) = 4πρr²g(r), radial distribution function (atoms/Å)"),
    ("T", "T(r) = 4πρr·g(r), total correlation function (Å^-2)"),
    ("N", "N(r) = ∫ RDF dr' from the first r, running coordination number (atoms)"),
]


def _rscf_column_header(with_errors: bool) -> list[str]:
    """
    Header lines describing the columns written by `saveRSCF`.

    Parameters
    ----------
    with_errors : bool
        False: 6 columns r, G, g, RDF, T, N. True: 11 columns, each function
        followed by its 1σ error (dG, dg, dRDF, dT, dN), plus a paragraph on the
        meaning and limits of the errors.

    Returns
    -------
    list of str
        Lines without ``#`` and without newline: ``"Columns:"``, one line per
        column (number, name, description from ``_RSCF_COLUMNS``), the error
        paragraph if ``with_errors``, and finally one line with the column names
        right-aligned to match the data (width 7 for r, 12 for the others).
    """
    lines = ["Columns:"]
    names: list[str] = []
    col = 1
    for i, (name, text) in enumerate(_RSCF_COLUMNS):
        lines.append(f"  {col:2d}  {name:<6} {text}")
        names.append(name)
        col += 1
        if with_errors and i > 0:
            lines.append(f"  {col:2d}  d{name:<5} 1σ error of {name}(r), same units")
            names.append("d" + name)
            col += 1
    if with_errors:
        lines += [
            "Errors: 1σ statistical, propagated exactly through the (linear) Fourier",
            "  transform from the S(Q) error column, assuming uncorrelated S(Q) points.",
            "  Values at different r are strongly correlated with each other.",
            "  Systematic errors (normalisation, density, self-scattering,",
            "  truncation) are not included.",
        ]
    lines.append(" ".join(f"{n:>12}" if k else f"{n:>7}" for k, n in enumerate(names)))
    return lines


def saveRSCF(  # noqa: N802 (legacy API)
    filename: str,
    fou: Sequence[Sequence[float]] | np.ndarray,
    heading: Sequence[str],
) -> None:
    """
    Write real-space correlation functions from `sineFT` to an ASCII file.

    `fou` is either the (N, 6) array r, G, g, RDF, T, N, or the (N, 11) array
    from `sineFT(..., with_errors=True)`, where each function is followed by
    its 1σ error: r, G, dG, g, dg, RDF, dRDF, T, dT, N, dN. After the user
    `heading`, a block describing every column is written.

    File layout: ``# <filename>``; ``# <text>`` for each heading line; the
    column block of `_rscf_column_header` (each line prefixed with ``# ``),
    ending with a line of column names; then one line per r with format
    ``{: 9.3f}`` for r and `` {:12.6f}`` for every other column.

    Parameters
    ----------
    filename : str
        Path of the output file (overwritten).
    fou : array_like, shape (N, 6) or (N, 11)
        Output of ``numerics.fourier.sineFT``. Units: r in Å, G and T in Å⁻²,
        g dimensionless, RDF in atoms/Å, N in atoms.
    heading : sequence of str
        Header lines (without ``#`` and without newline).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``fou`` is not 2-D or does not have 6 or 11 columns (message
        ``"saveRSCF: `fou` must have 6 or 11 columns."``), or cannot be converted
        to a float array.

    Notes
    -----
    Prints ``"File <filename> saved"``. Platform default encoding; the header
    contains non-ASCII characters (Å, π, ρ, σ, ², ·, ∫).
    """
    data = np.asarray(fou, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] not in (6, 11):
        raise ValueError("saveRSCF: `fou` must have 6 or 11 columns.")
    with_errors = data.shape[1] == 11

    with open(filename, "w") as datafile:
        datafile.write("# " + filename + "\n")
        for line in heading:
            datafile.write("# " + line + "\n")
        for line in _rscf_column_header(with_errors):
            datafile.write("# " + line + "\n")
        for row in data:
            datafile.write(
                "{: 9.3f}".format(row[0])
                + "".join(" {:12.6f}".format(v) for v in row[1:])
                + "\n"
            )
        print("File " + filename + " saved")
    return
