"""
Reader of the parameter file of the data-treatment notebooks.

Position in the processing chain
--------------------------------
Used after the regrouping, for the data treatment (corrections, normalisation,
S(Q), Fourier transform). ``getRunningParams`` reads the description of the
experiment (instrument, beam, grids, container, absorber, normaliser, environment,
sample, data files) into the dictionary ``inputPar``, which is the input of
``toscana.experiment.Measurement``. It is distinct from
``dataio.parameters.readParam`` (the d4creg file), although the syntax is similar.

Functions
---------
getRunningParams
    Read the notebook parameter file into ``inputPar`` (tag -> (label, values...)).

Conventions
-----------
- Every value of ``inputPar`` is a tuple ``(label, value1, value2, ...)``; the label
  is a human-readable description used for printing tables.
- Units are those of the labels: angles in degrees, wavelength in Å, slits and
  flags in mm, Q in 1/Å, r in Å, temperature in K, density in g/cm³, mass in g
  (shape dimensions as given in the file).
- No file is written and nothing is logged.
"""

from __future__ import annotations

from typing import Any


def getRunningParams(parfile: str) -> dict[str, Any]:
    """
    Read the parameters file for data treatment notebooks.

    Line syntax (same rules as the corrected ``readParam`` parser, C25):

    - a line of length <= 1 (only the newline) is skipped; a line starting with
      ``#`` or ``!`` is a comment;
    - any other line must start with ``<``. It is cut at the first ``#``
      (inline comment, optional) and split on any whitespace: token 0 is the
      tag, tokens 1, 2, ... the values. Values never contain the newline, and
      several spaces or tabs between values are allowed;
    - unknown tags are silently ignored.

    Tags read (``inputPar[tag] = (label, ...)``; values are strings unless
    "float" is stated):

    Experiment
        ``<ins>`` instrument; ``<pro>`` proposal; ``<mpr>`` main proposer;
        ``<lco>`` local contact; ``<cyc>`` cycle; ``<env>`` environment code;
        ``<dte>`` dates ``start-end``, also gives ``<sta>`` (text before the
        first ``-``) and ``<end>`` (text after it); ``<log>`` logbook
        ``notebook-page``, also gives ``<nbk>`` and ``<pag>`` the same way.
    Beam
        ``<zac>`` zero angle, deg (float); ``<wle>`` wavelength, Å (float);
        ``<sli>`` slits: ``(label, first value as str)`` plus ``<lft>`` and
        ``<rgt>`` (floats, mm, values 1 and 2); ``<fla>`` flags: same with
        ``<top>`` and ``<bot>``; ``<apr>`` angular precision, deg (float).
    Grids
        ``<asc>`` angular scale (deg), ``<qsc>`` Q scale (1/Å), ``<rsc>`` r scale
        (Å): ``(label, min, max, step)`` as floats.
    Container, absorber, normaliser, environment
        ``<Cmater>``, ``<Amater>``, ``<Nmater>``, ``<Emater>``: material; a
        warning is printed if it is not in the accepted list (container: SiO2,
        V, TiZr, Nb, None; absorber: B, None; normaliser: V, None; environment:
        V, A, None). ``<Cshape>``, ``<Ashape>``, ``<Nshape>``, ``<Eshape>``:
        ``(label, type, inner diameter, outer diameter, height)`` as STRINGS;
        a warning is printed if the type is not ``cylinder``.
    Sample
        ``<Sdescr>`` description: the whole text after the tag up to the first
        ``#``, with surrounding spaces removed;
        ``<StempK>`` temperature, K (float); ``<SatoNN>`` (any tag starting
        with ``<Sato``): the values after the tag, giving
        ``("Atom kk: ", symbol, number, ...)`` with kk the running count of such
        lines, and ``<SNelem>`` = ``("Number of elements:", count)``;
        ``<Sdegcc>`` density, g/cm³ (float); ``<Smassc>`` mass in the can, g
        (float); ``<Sshape>`` shape as for the container; ``<Sfulln>`` fullness
        (float); ``<Stitle>`` title.
    Data files
        ``<CoFile>`` container, ``<AbFile>`` absorber, ``<NoFile>`` vanadium
        (normaliser), ``<EnFile>`` environment, ``<SaFile>`` sample.

    Parameters
    ----------
    parfile : str
        Path of the parameter file (platform default encoding).

    Returns
    -------
    dict
        ``inputPar``: tag (string including ``<>``) -> tuple ``(label, values...)``.
        Always contains ``"<par>"`` = ``("Parameter file   ", parfile)``.

    Raises
    ------
    ValueError
        ``"Wrong input on line <i> in file <name>"`` if a line longer than one
        character does not start with ``#``, ``!`` or ``<`` (checked for the
        whole file first; the message is also printed); a numeric value that
        cannot be converted to float.
    IndexError
        If a tag line has fewer values than the tag needs.
    FileNotFoundError
        If ``parfile`` does not exist.

    Notes
    -----
    This is a legacy notebook-oriented parser distinct from
    `toscana.dataio.parameters.readParam`. It intentionally preserves the legacy
    dictionary shape (keys like `<ins>`, `<asc>`, ...).

    Prints warnings for unknown materials and shapes.

    Changes from the legacy code (audit 2026-11, same defect as C25 in
    ``readParam``): lines were split with ``split(" ")``, so a value that was
    the last word of its line kept the newline (``<Cmater> V`` gave ``"V\\n"``
    and a false "not available" warning; titles and file names ended with a
    newline), repeated spaces produced empty values, and ``<Sdescr>`` /
    ``<SatoNN>`` were read from fixed character positions (without ``#`` the
    last character was dropped). Files written with a trailing comment on
    every line, as the old parser required, give the same result as before,
    except that the description no longer keeps the spaces before ``#``.

    See Also
    --------
    toscana.experiment.Measurement : consumes ``inputPar``.
    """
    shapes = ["cylinder"]
    containers = ["SiO2", "V", "TiZr", "Nb", "None"]
    absorbers = ["B", "None"]
    normalisers = ["V", "None"]
    environments = ["V", "A", "None"]
    Nelements = 0  # Number of elements (or chemical species) in the sample

    inputPar: dict[str, Any] = {}
    inputPar["<par>"] = ("Parameter file   ", parfile)
    with open(parfile, "r") as f:
        lines = f.readlines()

    # Testing the first character of each non-blank line.
    # If this character is not #, ! or <, stops the program.
    for i in range(len(lines)):
        if len(lines[i]) > 1:
            first = (
                (lines[i][0] != "#") and (lines[i][0] != "!") and (lines[i][0] != "<")
            )
            if first:
                print("Wrong input on line: ", i + 1, " in file: ", parfile)
                raise ValueError(f"Wrong input on line {i + 1} in file {parfile}")

    # Loop over all lines in the input file
    for i in range(len(lines)):
        if (lines[i][0] == "#") or (lines[i][0] == "!"):
            pass
        elif len(lines[i]) == 1:
            pass
        elif lines[i][0] == "<":
            # Bug fix vs legacy (audit 2026-11, same defect as C25 in
            # readParam): legacy used lines[i].split(" "), so a value ending
            # its line kept the newline ("V\n" -> false "not available"
            # warning, title with a newline) and repeated spaces gave empty
            # values. The line is now cut at "#" (inline comment) and split on
            # any whitespace; `rest` is the text after the tag (descriptions).
            content = lines[i].split("#")[0].rstrip()
            line = content.split()
            rest = content[len(line[0]):].strip()

            if line[0] == "<ins>":
                inputPar[line[0]] = ("Instrument:", line[1])

            if line[0] == "<pro>":
                inputPar[line[0]] = ("Proposal:", line[1])

            if line[0] == "<mpr>":
                inputPar[line[0]] = ("Main Proposer:", line[1])

            if line[0] == "<lco>":
                inputPar[line[0]] = ("Local Contact:", line[1])

            if line[0] == "<dte>":
                inputPar[line[0]] = ("Dates:", line[1])
                hyphen = line[1].find("-")
                inputPar["<sta>"] = ("Starting date:", line[1][0:hyphen])
                inputPar["<end>"] = ("Ending date:", line[1][hyphen + 1 :])

            if line[0] == "<cyc>":
                inputPar[line[0]] = ("Cycle:", line[1])

            if line[0] == "<log>":
                inputPar[line[0]] = ("Logbook:", line[1])
                hyphen = line[1].find("-")
                inputPar["<nbk>"] = ("Notebook #:", line[1][0:hyphen])
                inputPar["<pag>"] = ("Page:", line[1][hyphen + 1 :])

            if line[0] == "<env>":
                inputPar[line[0]] = ("Environment:", line[1])

            if line[0] == "<zac>":
                inputPar[line[0]] = ("Zero angle (deg):", float(line[1]))

            if line[0] == "<wle>":
                inputPar[line[0]] = ("Wavelength (Å):", float(line[1]))

            if line[0] == "<sli>":
                inputPar[line[0]] = ("Slits (mm):", line[1])
                inputPar["<lft>"] = ("Left (mm):", float(line[1]))
                inputPar["<rgt>"] = ("Right (mm):", float(line[2]))

            if line[0] == "<fla>":
                inputPar[line[0]] = ("Flags (mm):", line[1])
                inputPar["<top>"] = ("Top (mm):", float(line[1]))
                inputPar["<bot>"] = ("Bottom (mm):", float(line[2]))

            if line[0] == "<apr>":
                inputPar[line[0]] = ("Angular precision (deg):", float(line[1]))

            if line[0] == "<asc>":
                inputPar[line[0]] = (
                    "Angular scale (deg):",
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                )

            if line[0] == "<qsc>":
                inputPar[line[0]] = (
                    "Q scale (1/Å):",
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                )

            if line[0] == "<rsc>":
                inputPar[line[0]] = (
                    "r scale (Å):",
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                )

            if line[0] == "<Cmater>":
                inputPar[line[0]] = ("Container material:", line[1])
                if line[1] in containers:
                    pass
                else:
                    print(
                        "The container {} is not available".format(inputPar[line[0]][1])
                    )

            if line[0] == "<Cshape>":
                inputPar[line[0]] = (
                    "Container shape:",
                    line[1],
                    line[2],
                    line[3],
                    line[4],
                )
                if line[1] in shapes:
                    pass
                else:
                    print("The shape {} is not available".format(inputPar[line[0]][1]))

            if line[0] == "<Amater>":
                inputPar[line[0]] = ("Absorber material:", line[1])
                if line[1] in absorbers:
                    pass
                else:
                    print(
                        "The absorber {} is not available".format(inputPar[line[0]][1])
                    )

            if line[0] == "<Ashape>":
                inputPar[line[0]] = (
                    "Absorber shape:",
                    line[1],
                    line[2],
                    line[3],
                    line[4],
                )
                if line[1] in shapes:
                    pass
                else:
                    print("The shape {} is not available".format(inputPar[line[0]][1]))

            if line[0] == "<Nmater>":
                inputPar[line[0]] = ("Normaliser material:", line[1])
                if line[1] in normalisers:
                    pass
                else:
                    print(
                        "The normaliser {} is not available".format(
                            inputPar[line[0]][1]
                        )
                    )

            if line[0] == "<Nshape>":
                inputPar[line[0]] = (
                    "Normaliser shape:",
                    line[1],
                    line[2],
                    line[3],
                    line[4],
                )
                if line[1] in shapes:
                    pass
                else:
                    print("The shape {} is not available".format(inputPar[line[0]][1]))

            if line[0] == "<Emater>":
                inputPar[line[0]] = ("Environment material:", line[1])
                if line[1] in environments:
                    pass
                else:
                    print(
                        "The environment {} is not available".format(
                            inputPar[line[0]][1]
                        )
                    )

            if line[0] == "<Eshape>":
                inputPar[line[0]] = (
                    "Environment shape:",
                    line[1],
                    line[2],
                    line[3],
                    line[4],
                )
                if line[1] in shapes:
                    pass
                else:
                    print(
                        "The Environment {} is not available".format(
                            inputPar[line[0]][1]
                        )
                    )

            if line[0] == "<Sdescr>":
                # Free text after the tag, up to "#" (legacy took characters
                # 9.. and, without "#", dropped the last character).
                inputPar[line[0]] = ("Sample description:  ", rest)

            if line[0] == "<StempK>":
                inputPar[line[0]] = ("Sample temperature:  ", float(line[1]))

            if line[0][0:5] == "<Sato":
                Nelements += 1
                # Values after the tag: symbol, number of atoms, ...
                atompar = line[1:]
                inputPar[line[0]] = ("Atom " + str(Nelements).zfill(2) + ": ", *atompar)
                inputPar["<SNelem>"] = ("Number of elements:", Nelements)

            if line[0] == "<Sdegcc>":
                inputPar[line[0]] = ("Sample density:      ", float(line[1]))

            if line[0] == "<Smassc>":
                inputPar[line[0]] = ("Sample mass in can:  ", float(line[1]))

            if line[0] == "<Sshape>":
                inputPar[line[0]] = (
                    "Sample shape:",
                    line[1],
                    line[2],
                    line[3],
                    line[4],
                )
                if line[1] in shapes:
                    pass
                else:
                    print("The shape {} is not available".format(inputPar[line[0]][1]))

            if line[0] == "<Sfulln>":
                inputPar[line[0]] = ("Sample fullness:", float(line[1]))

            if line[0] == "<Stitle>":
                inputPar[line[0]] = ("Sample title:        ", line[1])

            if line[0] == "<CoFile>":
                inputPar[line[0]] = ("Container file:      ", line[1])

            if line[0] == "<AbFile>":
                inputPar[line[0]] = ("Absorber file:       ", line[1])

            if line[0] == "<NoFile>":
                inputPar[line[0]] = ("Vanadium file:       ", line[1])

            if line[0] == "<EnFile>":
                inputPar[line[0]] = ("Environment file:    ", line[1])

            if line[0] == "<SaFile>":
                inputPar[line[0]] = ("Sample file:         ", line[1])
        else:
            print("Input error in line: ", i + 1, " file: ", parfile)
            raise ValueError(f"Input error in line {i + 1} file {parfile}")
    return inputPar
