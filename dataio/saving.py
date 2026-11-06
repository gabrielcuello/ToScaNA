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

    Notes
    -----
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

    Notes
    -----
    Writes `{head["numor"]}{runInfo["ext"][3]}` in the current working directory.
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


_RSCF_COLUMNS = [
    ("r", "distance (Å)"),
    ("G", "G(r) = 4πρr[g(r) - 1], pair distribution function (Å^-2)"),
    ("g", "g(r), pair correlation function (dimensionless)"),
    ("RDF", "RDF(r) = 4πρr²g(r), radial distribution function (atoms/Å)"),
    ("T", "T(r) = 4πρr·g(r), total correlation function (Å^-2)"),
    ("N", "N(r) = ∫ RDF dr' from the first r, running coordination number (atoms)"),
]


def _rscf_column_header(with_errors: bool) -> list[str]:
    """Header lines describing the columns written by `saveRSCF`."""
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
