"""
Bookkeeping dictionaries describing one D4 experiment (legacy notebook API).

The data-treatment notebooks start by describing the experiment: who and when,
the beam, the container, the binning grids, the numor ranges, the vanadium
standard and the sample. Each ``set*Info`` function below takes these values
as keyword arguments, derives a few quantities (beam width and height, wall
thickness, number of grid points, volumes, densities, averaged cross
sections...), PRINTS a human-readable summary to stdout and returns everything
in a plain ``dict``. The dictionaries are later passed around the notebook
(e.g. the vanadium and sample dictionaries feed the normalisation).

They are an alternative to `toscana.experiment.measurement.Measurement`, which
builds similar information from a parameter file.

Public names
------------
VANADIUM_DENSITY_GCC
    Macroscopic density of the vanadium standard, 6.11 g/cm³.
VANADIUM_MOLAR_MASS
    Molar mass of vanadium, 50.9415 g/mol.
VANADIUM_ATOMIC_DENSITY
    Atomic density derived from the two constants above,
    6.11 × 0.602214076 / 50.9415 = 0.0722305 atoms/Å³.
setExpInfo
    Experiment identification (proposal, people, dates, logbook).
setBeamInfo
    Wavelength, zero angle, slits and flags; beam width and height.
setCanInfo
    Container material, shape, diameters, height and wall thickness.
setBinInfo
    Angular, Q and r grids and their numbers of points.
setNumorInfo
    Numor ranges of each measured object.
setVanaInfo
    Vanadium standard: cross sections, scattering length, density, volume.
setSampleInfo
    Sample: geometry, densities, averaged cross sections, ratios to vanadium.

Conventions and units
---------------------
Lengths of samples, containers, slits and flags in mm; volumes in cm³;
macroscopic densities in g/cm³; atomic densities in atoms/Å³; molar masses in
g/mol (per atom for averages); cross sections in barns (per atom); scattering
lengths in fm; angles in degrees; wavelength in Å; Q in 1/Å; r in Å.
Mass numbers ``A`` are dimensionless (see the remark in `setSampleInfo`).

Changes from the legacy code (audit 2026-11)
--------------------------------------------
- P6: the vanadium density is 6.11 g/cm³ (legacy 6.51 g/cm³ in `setVanaInfo`
  and 0.0769591 atoms/Å³ typed separately in `setSampleInfo`); both defaults
  now derive from `VANADIUM_DENSITY_GCC`, so the default ``DenRatioSV`` of
  `setSampleInfo` increases by 6.5 %.
- C16: vanadium b_coh = -0.443 fm everywhere, σ_coh = 4π(0.443)²/100 =
  0.02466 b and σ_sca = σ_coh + σ_inc = 5.10466 b (legacy -0.3824 fm,
  0.01838 b, 5.09838 b).
- C30: absorption is scaled with the thermal wavelength 1.798 Å
  (`toscana.physics.properties.THERMAL_WAVELENGTH`; legacy 1.8 Å here).
- P9: free-atom cross sections of the sample are computed element by element
  (legacy applied the factor once with the average mass).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from toscana.physics.geometry import getCylVolume
from toscana.physics.properties import (
    AtomicAvg,
    getAbsXS,
    getAtomicDensity,
    getFreeXS,
    getMassNumber,
)
from toscana.utils.dicts import extractAttr

# Vanadium standard: macroscopic density (g/cm3) and molar mass (g/mol).
# Legacy used 6.51 g/cm3 (0.0769591 atoms/A3); the vanadium density is 6.11.
# (audit 2026-11, P6). Atomic density in atoms/Å³:
#   ρ = density [g/cm³] × N_A × 10⁻²⁴ / M [g/mol] = 6.11 × 0.602214076 / 50.9415
#     = 0.0722305 atoms/Å³.
VANADIUM_DENSITY_GCC = 6.11
VANADIUM_MOLAR_MASS = 50.9415
VANADIUM_ATOMIC_DENSITY = VANADIUM_DENSITY_GCC * 0.602214076 / VANADIUM_MOLAR_MASS


def setExpInfo(
    Proposal: str = "6-01-000",
    mainProposer: str = "nobody",
    experimenters: str = "nobody",
    LC: str = "Cuello",
    otherUsers: str = "nobody",
    startDate: str = "01/01/2000",
    endDate: str = "02/01/2000",
    environment: str = "A",
    logBook: int = 0,
    logPage: int = 0,
    instrument: str = "D4",
) -> dict[str, Any]:
    """
    Create a dictionary with information about the experiment (legacy behavior).

    The arguments are stored unchanged (no validation, no conversion) and a
    summary is printed.

    Parameters
    ----------
    Proposal : str, optional
        ILL proposal number. Default ``"6-01-000"``.
    mainProposer : str, optional
        Name of the main proposer. Default ``"nobody"``.
    experimenters : str, optional
        Users present on site. Default ``"nobody"``.
    LC : str, optional
        Local contact. Default ``"Cuello"``.
    otherUsers : str, optional
        Other users of the proposal. Default ``"nobody"``.
    startDate, endDate : str, optional
        Dates of the experiment, free text (defaults ``"01/01/2000"`` and
        ``"02/01/2000"``).
    environment : str, optional
        Code of the sample environment. Default ``"A"``.
    logBook : int, optional
        Number of the D4 logbook (notebook). Default 0.
    logPage : int, optional
        Page in the logbook. Default 0.
    instrument : str, optional
        Instrument name. Default ``"D4"``.

    Returns
    -------
    dict
        Keys and the argument they come from:

        - ``"Proposal"``: `Proposal`
        - ``"MP"``: `mainProposer`
        - ``"experimenters"``: `experimenters`
        - ``"LC"``: `LC`
        - ``"otherUsers"``: `otherUsers`
        - ``"instr"``: `instrument`
        - ``"startDate"``, ``"endDate"``: `startDate`, `endDate`
        - ``"environment"``: `environment`
        - ``"logBook"``, ``"logPage"``: `logBook`, `logPage`

    Notes
    -----
    Prints a block headed "Experiment" (30 dashes first, a blank line after)
    with one line per item.
    """
    experiment: dict[str, Any] = {"Proposal": Proposal}
    experiment["MP"] = mainProposer
    experiment["experimenters"] = experimenters
    experiment["LC"] = LC
    experiment["otherUsers"] = otherUsers
    experiment["instr"] = instrument
    experiment["startDate"] = startDate
    experiment["endDate"] = endDate
    experiment["environment"] = environment
    experiment["logBook"] = logBook
    experiment["logPage"] = logPage

    print(30 * "-")
    print("Experiment")
    print(4 * " ", f"Instrument: {experiment['instr']}")
    print(4 * " ", f"Proposal: {experiment['Proposal']}")
    print(4 * " ", f"Main proposer: {experiment['MP']}")
    print(4 * " ", f"Other users: {experiment['otherUsers']}")
    print(4 * " ", f"On-site users: {experiment['experimenters']}")
    print(4 * " ", f"Local contact: {experiment['LC']}")
    print(
        4 * " ",
        "Starting date: {0} ---->  Ending date: {1}".format(
            experiment["startDate"], experiment["endDate"]
        ),
    )
    print(4 * " ", f"Sample environment: {experiment['environment']}")
    print(
        4 * " ",
        "D4 notebook: {0}  Page: {1}".format(
            experiment["logBook"], experiment["logPage"]
        ),
    )
    print()
    return experiment


def setBeamInfo(
    zeroAngle: float = 0,
    wavelength: float = 0.5,
    LohenSlit: float = 2.5,
    GamsSlit: float = -2.5,
    topFlag: float = 25.0,
    bottomFlag: float = -25.0,
) -> dict[str, Any]:
    """
    Create a dictionary with information about the beam (legacy behavior).

    The beam size is defined by two vertical slits (left and right, legacy
    names ``LohenSlit`` and ``GamsSlit``) and two horizontal flags (top and
    bottom), all positions in mm relative to the beam axis::

        width  = LohenSlit - GamsSlit     (mm)
        height = topFlag   - bottomFlag   (mm)

    Parameters
    ----------
    zeroAngle : float, optional
        Zero-angle correction 2θ₀ of the detector, in degrees. Default 0.
    wavelength : float, optional
        Incident wavelength, in Å. Default 0.5.
    LohenSlit : float, optional
        Position of the left vertical slit, in mm. Default 2.5.
    GamsSlit : float, optional
        Position of the right vertical slit, in mm (negative on the other
        side of the axis). Default -2.5.
    topFlag : float, optional
        Position of the top flag, in mm. Default 25.0.
    bottomFlag : float, optional
        Position of the bottom flag, in mm. Default -25.0.

    Returns
    -------
    dict
        - ``"zero"``: zero angle (deg)
        - ``"wlength"``: wavelength (Å)
        - ``"LohenSlit"``, ``"GamsSlit"``: slit positions (mm)
        - ``"width"``: beam width = LohenSlit - GamsSlit (mm)
        - ``"topFlag"``, ``"botFlag"``: flag positions (mm)
        - ``"height"``: beam height = topFlag - bottomFlag (mm)

        With the defaults: width 5 mm, height 50 mm.

    Notes
    -----
    Prints a block "Beam characteristics" with the wavelength, the zero angle
    and the beam dimensions. No check that width or height are positive.
    """
    beam_width = LohenSlit - GamsSlit
    beam_height = topFlag - bottomFlag

    beam: dict[str, Any] = {"zero": zeroAngle}
    beam["wlength"] = wavelength
    beam["LohenSlit"] = LohenSlit
    beam["GamsSlit"] = GamsSlit
    beam["width"] = beam_width
    beam["topFlag"] = topFlag
    beam["botFlag"] = bottomFlag
    beam["height"] = beam_height

    print(30 * "-")
    print("Beam characteristics")
    print(4 * " ", "Wavelength = {:8.6g} A".format(beam["wlength"]))
    print(4 * " ", "Zero angle = {:8.6g} deg".format(beam["zero"]))
    print(
        4 * " ",
        "Dimensions: (Width = {0:.6g} mm)x(Height = {1:.6g} mm)".format(
            beam["width"], beam["height"]
        ),
    )
    print()
    return beam


def setCanInfo(
    material: str = "Vanadium",
    shape: str = "Cylinder",
    outerDiam: float = 5,
    innerDiam: float = 4.8,
    height: float = 60.0,
) -> dict[str, Any]:
    """
    Create a dictionary with information about the container (legacy behavior).

    The wall thickness is computed as ``(outerDiam - innerDiam) / 2`` (mm).

    Parameters
    ----------
    material : str, optional
        Container material (free text). Default ``"Vanadium"``.
    shape : str, optional
        Container shape (free text). Default ``"Cylinder"``.
    outerDiam : float, optional
        Outer diameter, in mm. Default 5.
    innerDiam : float, optional
        Inner diameter, in mm. Default 4.8.
    height : float, optional
        Height, in mm. Default 60.0.

    Returns
    -------
    dict
        - ``"material"``, ``"shape"``: as given
        - ``"outerDiam"``, ``"innerDiam"``: diameters (mm)
        - ``"height"``: height (mm)
        - ``"wallThickness"``: (outerDiam - innerDiam)/2 (mm); 0.1 mm with
          the defaults

    Notes
    -----
    Prints a block "Container" with type, diameters, wall thickness and
    height. No check that innerDiam < outerDiam.
    """
    wall_thickness = (outerDiam - innerDiam) / 2.0

    can: dict[str, Any] = {"material": material}
    can["shape"] = shape
    can["outerDiam"] = outerDiam
    can["innerDiam"] = innerDiam
    can["height"] = height
    can["wallThickness"] = wall_thickness

    print(30 * "-")
    print("Container")
    print(4 * " ", "Type: {0} {1}".format(can["material"], can["shape"]))
    print(4 * " ", f"Outer diameter = {can['outerDiam']} mm")
    print(4 * " ", f"Inner diameter = {can['innerDiam']} mm")
    print(4 * " ", "Wall thickness = {:.3g} mm".format(can["wallThickness"]))
    print(4 * " ", f"Height = {can['height']} mm")
    print()
    return can


def setBinInfo(
    AngularResolution: float = 0.125,
    AMin: float = 0.0,
    AMax: float = 140.0,
    AStep: float = 0.125,
    QMin: float = 0.0,
    QMax: float = 23.5,
    QStep: float = 0.02,
    RMin: float = 0.0,
    RMax: float = 20.0,
    RStep: float = 0.01,
) -> dict[str, Any]:
    """
    Create a dictionary with information about the binnings (legacy behavior).

    Each grid is a tuple ``(min, max, step)`` and its number of points is
    computed as::

        NbrPoints = int((max - min) / step)

    i.e. truncated toward zero, the end point ``max`` not included. Because
    of floating-point division this can be one less than expected when the
    step divides the range exactly only in decimal (e.g. a quotient computed
    as 1174.9999 gives 1174). This count is only informative here; it is not
    used to build arrays in this module.

    Parameters
    ----------
    AngularResolution : float, optional
        Angular width of one detector channel (cell), in degrees.
        Default 0.125.
    AMin, AMax, AStep : float, optional
        Angular grid 2θ, in degrees. Defaults 0.0, 140.0, 0.125.
    QMin, QMax, QStep : float, optional
        Momentum-transfer grid, in 1/Å. Defaults 0.0, 23.5, 0.02.
    RMin, RMax, RStep : float, optional
        Real-space grid, in Å. Defaults 0.0, 20.0, 0.01.

    Returns
    -------
    dict
        - ``"Ares"``: `AngularResolution` (deg)
        - ``"Abin"``: (AMin, AMax, AStep) (deg)
        - ``"Qbin"``: (QMin, QMax, QStep) (1/Å)
        - ``"Rbin"``: (RMin, RMax, RStep) (Å)
        - ``"NbrPointsA"``, ``"NbrPointsQ"``, ``"NbrPointsR"``: int, number
          of points of each grid as above (1120, 1175, 2000 with the
          defaults)

    Raises
    ------
    ZeroDivisionError
        If a step is 0.

    Notes
    -----
    Prints a block "Binning" with the channel width and the three grids.
    """
    Abin = (AMin, AMax, AStep)
    Qbin = (QMin, QMax, QStep)
    Rbin = (RMin, RMax, RStep)

    NbrPointsA = int((Abin[1] - Abin[0]) / Abin[2])
    NbrPointsQ = int((Qbin[1] - Qbin[0]) / Qbin[2])
    NbrPointsR = int((Rbin[1] - Rbin[0]) / Rbin[2])

    binning: dict[str, Any] = {"Ares": AngularResolution}
    binning["Abin"] = Abin
    binning["Qbin"] = Qbin
    binning["Rbin"] = Rbin
    binning["NbrPointsA"] = NbrPointsA
    binning["NbrPointsQ"] = NbrPointsQ
    binning["NbrPointsR"] = NbrPointsR

    print(30 * "-")
    print("Binning")
    print(4 * " ", "Angular channel width = {:.3g} deg".format(binning["Ares"]))
    print(
        4 * " ",
        "In angle: from {0:.3g} deg to {1:.3g} deg, in steps of {2:.3f} deg, thus {3:.5g} points.".format(
            binning["Abin"][0],
            binning["Abin"][1],
            binning["Abin"][2],
            binning["NbrPointsA"],
        ),
    )
    print(
        4 * " ",
        "In Q: from {0:.3g} 1/A to {1:.3g} 1/A, in steps of {2:.3f} 1/A, thus {3:.5g} points.".format(
            binning["Qbin"][0],
            binning["Qbin"][1],
            binning["Qbin"][2],
            binning["NbrPointsQ"],
        ),
    )
    print(
        4 * " ",
        "In R: from {0:.3g} A to {1:.3g} A, in steps of {2:.3f} A, thus {3:.5g} points.".format(
            binning["Rbin"][0],
            binning["Rbin"][1],
            binning["Rbin"][2],
            binning["NbrPointsR"],
        ),
    )
    print()
    return binning


def setNumorInfo(
    totalNumors: tuple[str, int, int] = ("Exp", 0, 1),
    containerNumors: tuple[str, int, int] = ("Can", 0, 1),
    environmentNumors: tuple[str, int, int] = ("Env", 0, 1),
    nickelNumors: tuple[str, int, int] = ("Ni5", 0, 1),
    vanadiumNumors: tuple[str, int, int] = ("Van", 0, 1),
    absorberNumors: tuple[str, int, int] = ("Abs", 0, 1),
    sampleNumors: tuple[str, int, int] = ("S01", 0, 1),
) -> dict[str, Any]:
    """
    Create a dictionary with information about the numors (legacy behavior).

    A numor is the ILL run number of one measurement file. Each argument is a
    tuple ``(label, first, last)`` describing a consecutive range of numors.

    Parameters
    ----------
    totalNumors : tuple of (str, int, int), optional
        All numors of the experiment. Default ``("Exp", 0, 1)``.
    containerNumors : tuple, optional
        Empty container. Default ``("Can", 0, 1)``.
    environmentNumors : tuple, optional
        Sample environment (empty instrument / furnace...). Default
        ``("Env", 0, 1)``.
    nickelNumors : tuple, optional
        Nickel powder used for calibration. Default ``("Ni5", 0, 1)``.
    vanadiumNumors : tuple, optional
        Vanadium standard. Default ``("Van", 0, 1)``.
    absorberNumors : tuple, optional
        Absorber (e.g. boron) measurement. Default ``("Abs", 0, 1)``.
    sampleNumors : tuple, optional
        Sample. Default ``("S01", 0, 1)``.

    Returns
    -------
    dict
        The tuples, unchanged, under the keys ``"experiment"``,
        ``"container"``, ``"environment"``, ``"nickel"``, ``"vanadium"``,
        ``"absorber"`` and ``"sample"``.

    Raises
    ------
    ValueError, TypeError
        If a range to be printed has ``first`` or ``last`` that ``int()``
        cannot convert.

    Notes
    -----
    Prints "Numors" and then, for each range whose ``first`` is not 0, a
    line ``"<label>: <first> -<last>, <n> numors."`` with
    ``n = int(last) - int(first) + 1`` (the experiment and sample lines are
    prefixed by "Total" and "Sample"). A ``first`` equal to 0 means "not
    measured" and the range is not printed (it is still returned). Unlike the
    other ``set*Info`` functions, no dashed separator is printed.
    """
    numors: dict[str, Any] = {
        "experiment": totalNumors,
        "container": containerNumors,
        "environment": environmentNumors,
        "nickel": nickelNumors,
        "vanadium": vanadiumNumors,
        "absorber": absorberNumors,
        "sample": sampleNumors,
    }

    print("Numors")

    if totalNumors[1] != 0:
        print(
            "{}Total {}: {} -{}, {} numors.".format(
                4 * " ",
                *numors["experiment"],
                int(numors["experiment"][2]) - int(numors["experiment"][1]) + 1,
            )
        )
    if containerNumors[1] != 0:
        print(
            "{}{}: {} -{}, {} numors.".format(
                4 * " ",
                *numors["container"],
                int(numors["container"][2]) - int(numors["container"][1]) + 1,
            )
        )
    if environmentNumors[1] != 0:
        print(
            "{}{}: {} -{}, {} numors.".format(
                4 * " ",
                *numors["environment"],
                int(numors["environment"][2]) - int(numors["environment"][1]) + 1,
            )
        )
    if nickelNumors[1] != 0:
        print(
            "{}{}: {} -{}, {} numors.".format(
                4 * " ",
                *numors["nickel"],
                int(numors["nickel"][2]) - int(numors["nickel"][1]) + 1,
            )
        )
    if vanadiumNumors[1] != 0:
        print(
            "{}{}: {} -{}, {} numors.".format(
                4 * " ",
                *numors["vanadium"],
                int(numors["vanadium"][2]) - int(numors["vanadium"][1]) + 1,
            )
        )
    if absorberNumors[1] != 0:
        print(
            "{}{}: {} -{}, {} numors.".format(
                4 * " ",
                *numors["absorber"],
                int(numors["absorber"][2]) - int(numors["absorber"][1]) + 1,
            )
        )
    if sampleNumors[1] != 0:
        print(
            "{}Sample {}: {} -{}, {} numors.".format(
                4 * " ",
                *numors["sample"],
                int(numors["sample"][2]) - int(numors["sample"][1]) + 1,
            )
        )
    return numors


def setVanaInfo(
    IncXS: float = 5.08,
    CohXS: float = 0.02466,  # 4π(0.443 fm)²/100 (legacy 0.0184 for -0.3824 fm)
    ScaXS: float = 5.10466,  # CohXS + IncXS (legacy 5.0984)
    AbsXS: float = 5.08,
    CohSL: float = -0.443,  # legacy -0.3824; matches isotopes database "V"
    molarM: float = 50.9415,
    NAtoms: float = 1.0,
    Diam: float = 6.08,
    Height: float = 50.0,
    density: float = VANADIUM_DENSITY_GCC,
) -> dict[str, Any]:
    """
    Create a dictionary with information about the vanadium (legacy behavior).

    The vanadium standard is a solid cylinder used to normalise the sample
    data. From the arguments the function derives::

        SelfQ0    = IncXS / (4π)                             barns/sr/atom
        den_aac   = density × 0.602214076 / molarM           atoms/Å³
        A         = molarM / 1.0086649                       (mass ratio m/m_n)
        FreeIncXS = IncXS × A² / (1 + A)²                    barns/atom
        volume    = π (Diam/2)² × Height / 1000              cm³

    using `getAtomicDensity`, `getMassNumber`, `getFreeXS` and
    `getCylVolume`. The cross sections are stored as given: ``ScaXS`` is NOT
    recomputed from ``CohXS + IncXS`` and ``CohXS`` not from ``CohSL``, so
    keep them consistent if you override one of them.

    Parameters
    ----------
    IncXS : float, optional
        Bound incoherent cross section, barns/atom. Default 5.08.
    CohXS : float, optional
        Bound coherent cross section, barns/atom. Default 0.02466
        = 4π(0.443 fm)²/100.
    ScaXS : float, optional
        Bound total scattering cross section, barns/atom. Default 5.10466
        = CohXS + IncXS.
    AbsXS : float, optional
        Absorption cross section at the thermal wavelength (1.798 Å),
        barns/atom. Default 5.08. Stored only; not scaled to the wavelength.
    CohSL : float, optional
        Coherent scattering length, fm. Default -0.443 (the value of the
        isotope table, ``Element("V").re_bcoh``).
    molarM : float, optional
        Molar mass, g/mol. Default 50.9415.
    NAtoms : float, optional
        Number of atoms in the formula unit. Default 1.0. Stored only.
    Diam : float, optional
        Diameter of the vanadium cylinder, mm. Default 6.08.
    Height : float, optional
        Height of vanadium seen by the beam, mm. Default 50.0. Only used for
        the volume; it is not stored in the dictionary.
    density : float, optional
        Macroscopic density, g/cm³. Default `VANADIUM_DENSITY_GCC` = 6.11.
        If ``density <= 0``, `getAtomicDensity` prints a warning and returns
        the water-like value 0.1 atoms/Å³ for ``den_aac``.

    Returns
    -------
    dict
        - ``"IncXS"``, ``"CohXS"``, ``"ScaXS"``, ``"AbsXS"``: as given
          (barns/atom)
        - ``"CohSL"``: as given (fm)
        - ``"SelfQ0"``: σ_inc/4π, self scattering at Q = 0 (barns/sr/atom);
          0.40425 with the defaults
        - ``"diam"``: `Diam` (mm)
        - ``"NAtoms"``: `NAtoms`
        - ``"molarM"``: `molarM` (g/mol)
        - ``"den_gcc"``: `density` (g/cm³)
        - ``"den_aac"``: atomic density (atoms/Å³); 0.0722305 by default
        - ``"A"``: mass ratio molarM/1.0086649 (dimensionless); 50.5039
        - ``"FreeIncXS"``: free incoherent cross section (barns/atom);
          4.88465 with the defaults
        - ``"volume"``: volume of the cylinder (cm³); 1.45167 by default

    Notes
    -----
    Prints a block "Standard of vanadium": diameter, volume, bound and free
    incoherent cross sections, σ/4π at Q = 0 and at Q → ∞ (FreeIncXS/4π),
    molar mass, mass number, density and atomic density.

    Changes from the legacy code (audit 2026-11):

    - C16: ``CohSL`` default -0.443 fm (legacy -0.3824 fm), ``CohXS``
      0.02466 b (legacy 0.0184), ``ScaXS`` 5.10466 b (legacy 5.0984); the
      defaults now equal the isotope table values for "V" (checked by a
      test).
    - P6: ``density`` default 6.11 g/cm³ (legacy 6.51 g/cm³).

    See Also
    --------
    setSampleInfo : uses ``volume`` (as `vanavol`) and ``den_aac`` (as
        `vanadens`) to form the sample/vanadium ratios.
    """
    SelfQ0 = IncXS / 4.0 / np.pi
    den_aac = getAtomicDensity(density=density, molarMass=molarM)
    A = getMassNumber(molarMass=molarM)
    FreeIncXS = getFreeXS(BoundXS=IncXS, A=A)
    volume = getCylVolume(diameter=Diam, height=Height) / 1000.0

    vana: dict[str, Any] = {}
    vana["IncXS"] = IncXS
    vana["CohXS"] = CohXS
    vana["ScaXS"] = ScaXS
    vana["AbsXS"] = AbsXS
    vana["CohSL"] = CohSL
    vana["SelfQ0"] = SelfQ0
    vana["diam"] = Diam
    vana["NAtoms"] = NAtoms
    vana["molarM"] = molarM
    vana["den_gcc"] = density
    vana["den_aac"] = den_aac
    vana["A"] = A
    vana["FreeIncXS"] = FreeIncXS
    vana["volume"] = volume

    print(30 * "-")
    print("Standard of vanadium")
    print(
        4 * " ",
        "The standard is a cylinder of {:.3g} mm of diameter.".format(vana["diam"]),
    )
    print(4 * " ", "Volume in the beam = {:.3g} cm3.".format(vana["volume"]))
    print()
    print(
        4 * " ",
        "Bound incoherent cross section {:.6g} barns/atom.".format(vana["IncXS"]),
    )
    print(
        4 * " ",
        "Free incoherent cross section",
        "{:.6g} barns/atom.".format(vana["FreeIncXS"]),
    )
    print()
    print(
        4 * " ",
        "b**2 = sigma/4/pi at Q=0 (self) is",
        "{:.6g} barns/sterad/atom.".format(vana["SelfQ0"]),
    )
    print(
        4 * " ",
        "b**2 = sigma/4/pi at Q=infty is",
        "{:.6g} barns/sterad/atom".format(vana["FreeIncXS"] / 4.0 / np.pi),
    )
    print()
    print(4 * " ", "The molar mass is", "{:.6g} g/mol.".format(vana["molarM"]))
    print(
        4 * " ", "Mass number, A = ", "{:.6g} (= mass/neutron_mass)".format(vana["A"])
    )
    print()
    print(4 * " ", "Density = ", "{:.6g} g/cm3.".format(vana["den_gcc"]))
    print(4 * " ", "Atomic density = {:.6g} atoms/A3.".format(vana["den_aac"]))
    print()
    return vana


def setSampleInfo(
    comp_sample: Mapping[str, float],
    atoms_sample: Mapping[str, float],
    natoms_sample: float,
    elements: Mapping[str, Any],
    c_sample: Mapping[str, float],
    wavelength: float = 0.5,
    beamH: float = 50.0,
    vanavol: float = 1.0,
    vanadens: float = VANADIUM_ATOMIC_DENSITY,
    height: float = 60.0,
    diameter: float = 5.0,
    mass: float = 1.0,
    density: float = 1.0,
    title: str = "sample",
) -> dict[str, Any]:
    """
    Create a dictionary with information about the sample (legacy behavior).

    The sample is treated as a SOLID cylinder of diameter `diameter` (no
    inner diameter; for annular samples use `Measurement`). With
    c_i = ``c_sample[i]`` the atomic concentrations and the element data
    taken from ``elements[i]`` (an `Element`), the function computes::

        heightInBeam = min(height, beamH)                              mm
        volume       = π (diameter/2)² height / 1000                   cm³
        volumeInBeam = π (diameter/2)² heightInBeam / 1000             cm³
        effden_gcc   = mass / volume                                   g/cm³
        packing      = effden_gcc / density
        filling      = heightInBeam / beamH
        molarM       = Σ c_i weight_i                                  g/mol/atom
        den_aac      = density    × 0.602214076 / molarM               atoms/Å³
        effden_aac   = effden_gcc × 0.602214076 / molarM               atoms/Å³
        CohXS, IncXS, ScaXS, AbsXS = Σ c_i σ_i (sig_coh, sig_inc,
                                     sig_sca, sig_abs)                 barns/atom
        AbsWW        = AbsXS × wavelength / 1.798                      barns/atom
        CohSL        = Σ c_i Re(b_coh,i)                               fm
        A            = Σ c_i A_i   (A_i = Element.A, nucleon number)
        Free<X>XS    = Σ c_i σ_X,i [a_i/(a_i+1)]²,  a_i = weight_i/1.0086649
        SelfQ0       = IncXS / (4π)                                    barns/sr/atom
        VolRatioSV   = volumeInBeam / vanavol
        DenRatioSV   = effden_aac / vanadens

    ``AbsWW`` uses the 1/v law (`getAbsXS`) with the thermal wavelength
    1.798 Å. The free cross sections use `getFreeXS` and `getMassNumber`
    element by element.

    Parameters
    ----------
    comp_sample : Mapping[str, float]
        Composition in "units" (e.g. ``{"H2O": 1}``). Only printed.
    atoms_sample : Mapping[str, float]
        Number of atoms of each element in one unit (e.g.
        ``{"H": 2, "O": 1}``). Only printed.
    natoms_sample : float
        Number of atoms in one unit. Stored as ``"NAtoms"`` and printed.
    elements : Mapping[str, Element]
        Element symbol -> `toscana.isotopes.Element` (any object with the
        attributes ``weight``, ``sig_coh``, ``sig_inc``, ``sig_sca``,
        ``sig_abs``, ``re_bcoh`` and ``A`` works). Must contain every key of
        `c_sample`.
    c_sample : Mapping[str, float]
        Atomic concentration of each element (fractions, normally summing to
        1, e.g. from `getConcentrations`). Defines which elements enter all
        the averages.
    wavelength : float, optional
        Incident wavelength, Å, for ``AbsWW``. Default 0.5.
    beamH : float, optional
        Beam height, mm. Default 50.0.
    vanavol : float, optional
        Volume of vanadium in the beam, cm³ (``setVanaInfo()["volume"]``).
        Default 1.0.
    vanadens : float, optional
        Atomic density of vanadium, atoms/Å³. Default
        `VANADIUM_ATOMIC_DENSITY` = 0.0722305.
    height : float, optional
        Sample height, mm. Default 60.0.
    diameter : float, optional
        Sample diameter, mm. Default 5.0.
    mass : float, optional
        Sample mass, g. Default 1.0.
    density : float, optional
        Macroscopic (bulk material) density, g/cm³. Default 1.0. If
        ``<= 0``, `getAtomicDensity` prints a warning and returns 0.1
        atoms/Å³ for ``den_aac`` (``packing`` still divides by it).
    title : str, optional
        Sample name. Default ``"sample"``.

    Returns
    -------
    dict
        - ``"Title"``: `title`
        - ``"height"``, ``"heightInBeam"``, ``"diam"``: mm
        - ``"mass"``: g
        - ``"NAtoms"``: `natoms_sample`
        - ``"molarM"``: average molar mass, g/mol/atom
        - ``"volume"``, ``"volumeInBeam"``: cm³
        - ``"den_gcc"``: `density`, g/cm³
        - ``"effden_gcc"``: effective density mass/volume, g/cm³
        - ``"packing"``: packing fraction effden_gcc/density
        - ``"filling"``: fraction of the beam height filled, heightInBeam/beamH
        - ``"den_aac"``, ``"effden_aac"``: atomic and effective atomic
          densities, atoms/Å³
        - ``"CohXS"``, ``"IncXS"``, ``"ScaXS"``, ``"AbsXS"``: bound averaged
          cross sections, barns/atom (AbsXS at the thermal wavelength)
        - ``"AbsWW"``: absorption at `wavelength`, barns/atom
        - ``"CohSL"``: average coherent scattering length (real part, sign
          kept), fm
        - ``"A"``: average nucleon number Σ c_i A_i (dimensionless)
        - ``"FreeCohXS"``, ``"FreeIncXS"``, ``"FreeScaXS"``: free-atom cross
          sections, barns/atom
        - ``"SelfQ0"``: self scattering at Q = 0, IncXS/4π, barns/sr/atom
        - ``"VolRatioSV"``: sample/vanadium volume ratio (in the beam)
        - ``"DenRatioSV"``: sample/vanadium atomic-density ratio (effective)

    Raises
    ------
    KeyError
        If a key of `c_sample` is missing from `elements` (raised by
        `AtomicAvg` or by the free cross-section sum).
    ZeroDivisionError
        If ``volume``, ``density``, ``molarM``, ``beamH``, ``vanavol`` or
        ``vanadens`` is 0.

    Notes
    -----
    Prints a long block (80 dashes first): composition, atomic composition,
    concentrations, average molar mass and mass number, bound and free cross
    sections, absorption (thermal and at `wavelength`), scattering length,
    self scattering, densities, cylinder geometry, volume and density
    ratios, packing and filling fractions.

    ``"A"`` here is the concentration average of the integer nucleon numbers
    of the table (``Element.A``, e.g. 59 for natural Ni), printed as "Mass
    number ... amu", whereas ``setVanaInfo()["A"]`` and the free cross
    sections use the mass ratio weight/m_n. ``"A"`` is not used in any
    computation of this function.

    Changes from the legacy code (audit 2026-11):

    - P9: ``FreeCohXS``, ``FreeIncXS``, ``FreeScaXS`` are summed element by
      element, Σ c_i σ_i [a_i/(a_i+1)]². Legacy applied the factor once with
      the AVERAGE mass number, badly wrong for samples with light atoms
      (H₂O: 14.908 b instead of 41.116 b for FreeScaXS).
    - C30: ``AbsWW`` = AbsXS × λ / 1.798 (legacy AbsXS × λ / 1.8).
    - P6: default `vanadens` 0.0722305 atoms/Å³ (legacy 0.0769591, which
      corresponded to 6.51 g/cm³), so ``DenRatioSV`` increases by 6.5 %.

    See Also
    --------
    setVanaInfo, toscana.experiment.measurement.Measurement,
    toscana.physics.properties.getConcentrations
    """
    heightInBeam = min(height, beamH)
    volume = getCylVolume(diameter, height) / 1000.0
    volumeInBeam = getCylVolume(diameter=diameter, height=heightInBeam) / 1000.0
    effden_gcc = mass / volume
    packing = effden_gcc / density
    filling = heightInBeam / beamH

    molarM = AtomicAvg(c_sample, extractAttr(elements, "weight"))
    den_aac = getAtomicDensity(density=density, molarMass=molarM)
    effden_aac = getAtomicDensity(density=effden_gcc, molarMass=molarM)

    CohXS = AtomicAvg(c_sample, extractAttr(elements, "sig_coh"))
    IncXS = AtomicAvg(c_sample, extractAttr(elements, "sig_inc"))
    ScaXS = AtomicAvg(c_sample, extractAttr(elements, "sig_sca"))
    AbsXS = AtomicAvg(c_sample, extractAttr(elements, "sig_abs"))
    AbsWW = getAbsXS(AbsXS, wavelength)  # was AbsXS * wavelength / 1.8
    CohSL = AtomicAvg(c_sample, extractAttr(elements, "re_bcoh"))
    A = AtomicAvg(c_sample, extractAttr(elements, "A"))

    # Free-atom cross sections element by element:
    #   σ_free = Σ c_i σ_i (A_i/(A_i+1))²,  A_i = m_i / m_n.
    # Bug fix vs legacy: legacy applied the factor once with the AVERAGE mass
    # number, which is badly wrong for samples with light atoms (H, D, Li).
    def _free(attr: str) -> float:
        """
        Free-atom average Σ c_i σ_i [a_i/(a_i+1)]² of one cross section.

        Parameters
        ----------
        attr : str
            Name of the bound cross-section attribute of the elements
            (``"sig_coh"``, ``"sig_inc"`` or ``"sig_sca"``), in barns.

        Returns
        -------
        float
            Free cross section, barns/atom, with a_i = weight_i / 1.0086649
            (`getMassNumber`) and the factor applied by `getFreeXS`.
        """
        return float(
            sum(
                c_sample[key]
                * getFreeXS(
                    BoundXS=float(getattr(elements[key], attr)),
                    A=getMassNumber(float(elements[key].weight)),
                )
                for key in c_sample
            )
        )

    FreeCohXS = _free("sig_coh")
    FreeIncXS = _free("sig_inc")
    FreeScaXS = _free("sig_sca")
    SelfQ0 = IncXS / 4.0 / np.pi

    VolRatioSV = volumeInBeam / vanavol
    DenRatioSV = effden_aac / vanadens

    sample: dict[str, Any] = {"Title": title}
    sample["height"] = height
    sample["heightInBeam"] = heightInBeam
    sample["diam"] = diameter
    sample["mass"] = mass
    sample["NAtoms"] = natoms_sample
    sample["molarM"] = molarM
    sample["volume"] = volume
    sample["volumeInBeam"] = volumeInBeam
    sample["den_gcc"] = density
    sample["effden_gcc"] = effden_gcc
    sample["packing"] = packing
    sample["filling"] = filling
    sample["den_aac"] = den_aac
    sample["effden_aac"] = effden_aac
    sample["CohXS"] = CohXS
    sample["IncXS"] = IncXS
    sample["ScaXS"] = ScaXS
    sample["AbsXS"] = AbsXS
    sample["AbsWW"] = AbsWW
    sample["CohSL"] = CohSL
    sample["A"] = A
    sample["FreeCohXS"] = FreeCohXS
    sample["FreeIncXS"] = FreeIncXS
    sample["FreeScaXS"] = FreeScaXS
    sample["SelfQ0"] = SelfQ0
    sample["VolRatioSV"] = VolRatioSV
    sample["DenRatioSV"] = DenRatioSV

    print(80 * "-")
    print("Sample", sample["Title"])

    text = " "
    print()
    print("Composition:")
    for key, value in comp_sample.items():
        text += "+ " + str(value) + " of " + key + " "
    print(4 * " ", 'A "unit" is', text[3:])

    text = " "
    print()
    print("Atomic composition:")
    for key, value in atoms_sample.items():
        text += "+ " + str(value) + " " + key + " atoms "
    print(4 * " ", 'A "unit" has', text[3:])
    print(4 * " ", "Number of atoms in one unit = {} atoms".format(sample["NAtoms"]))

    print()
    print("Atomic concentrations:")
    for key, value in c_sample.items():
        print(8 * " ", key, "{:.6f}".format(value))

    print()
    print("Average molar mass: {:.6g} g/mol/atom".format(sample["molarM"]))
    print("Mass number: {:.6g} amu".format(sample["A"]))
    print()
    print("Coherent cross section: {:.6g} barns/atom".format(sample["CohXS"]))
    print("Incoherent cross section: {:.6g} barns/atom".format(sample["IncXS"]))
    print("Scattering cross section: {:.6g} barns/atom".format(sample["ScaXS"]))
    print()
    print("Free coherent cross section: {:.6g} barns/atom".format(sample["FreeCohXS"]))
    print(
        "Free incoherent cross section: {:.6g} barns/atom".format(sample["FreeIncXS"])
    )
    print(
        "Free scattering cross section: {:.6g} barns/atom".format(sample["FreeScaXS"])
    )
    print()
    print("Absorption cross section: {:.6g} barns/atom".format(sample["AbsXS"]))
    print(
        "Absorption cross section at {:.6g} A: {:.6g} barns/atom".format(
            wavelength, sample["AbsWW"]
        )
    )
    print()
    print("Coherent scattering length: {:.6g} fm".format(sample["CohSL"]))
    print(
        "Self scattering cross section at Q=0: {:.6g} barns/steradian/atom".format(
            sample["SelfQ0"]
        )
    )
    print()
    print("Density: {:.6g} g/cm3".format(sample["den_gcc"]))
    print("Atomic density: {:.6g} atoms/A3".format(sample["den_aac"]))
    print()
    print("Cylindrical sample")
    print(4 * " ", "Diameter: {:.6g} mm".format(sample["diam"]))
    print(4 * " ", "Height: {:.6g} mm".format(sample["height"]))
    print(4 * " ", "Volume: {:.6g} cm3".format(sample["volume"]))
    print(4 * " ", "Mass: {:.6g} g".format(sample["mass"]))
    print(4 * " ", "Effective density: {:.6g} g/cm3".format(sample["effden_gcc"]))
    print(
        4 * " ",
        "Effective atomic density: {:.6g} atoms/A3".format(sample["effden_aac"]),
    )
    print()
    print(
        4 * " ", "Sample/Vanadium volume fraction: {:.6g}".format(sample["VolRatioSV"])
    )
    print(
        4 * " ", "Sample/Vanadium density fraction: {:.6g}".format(sample["DenRatioSV"])
    )
    print(4 * " ", "Packing fraction: {:.6g}".format(sample["packing"]))
    print(4 * " ", "Filling fraction: {:.6g}".format(sample["filling"]))

    return sample
