"""
The `Measurement` class: everything about one sample measurement.

In the data-treatment notebooks, the analysis parameter file is read with
`toscana.dataio.running_params.getRunningParams`, and the resulting
dictionary is passed to `Measurement`. The object then holds, as plain
attributes, the experiment and beam information, the binning grids, the
geometry of the container, normaliser (vanadium), absorber and sample
environment, the sample geometry, mass and densities, its composition with
the averaged nuclear data (from `toscana.isotopes.Element`), and the five
data sets (sample, environment, container, normaliser, absorber) read from
the files named in the parameter file.

Public names
------------
Measurement
    Container built from the ``inputPar`` dictionary; method ``showTable``
    prints the main parameters.

Conventions and units
---------------------
Lengths in mm, volumes in cm³, masses in g, macroscopic densities in g/cm³,
atomic densities in atoms/Å³, cross sections in barns/atom, scattering
lengths in fm, angles in degrees, wavelength in Å, Q in 1/Å, r in Å,
temperatures in K (and °C for ``TempC``).

Changes from the legacy code (audit 2026-11)
--------------------------------------------
- C14: annulus volume π h (r_o² - r_i²) instead of π h (r_o - r_i)².
- P9: free-atom cross section computed element by element.
- P13: ``showTable`` uses ``contextlib.redirect_stdout`` and closes its file.
- C30: absorption at the wavelength uses the thermal wavelength 1.798 Å
  (through `getAbsXS`).
"""

from __future__ import annotations

import cmath
import contextlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from toscana.dataio.loading import read_3col
from toscana.isotopes.core import Element
from toscana.physics.properties import getAbsXS, getAtomicDensity, getMassNumber

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Measurement:
    """
    Legacy-compatible measurement container.

    Collects everything about a measurement from the dictionary returned by
    `toscana.dataio.running_params.getRunningParams`, computes the derived
    sample quantities and reads the five data files.

    Parameters
    ----------
    inputPar : Mapping[str, Sequence]
        Dictionary ``{"<key>": (label, value1, value2, ...)}`` as produced by
        ``getRunningParams``: element 0 of every value is a descriptive
        label and is ignored here (except by `showTable`); the data start at
        element 1. Values may be strings or numbers; numeric fields are
        converted with ``float()`` (``int()`` for ``<SNelem>``). Keys read
        (field index in brackets):

        Experiment
            ``<par>`` [1] parameter-file name; ``<ins>`` [1] instrument;
            ``<pro>`` [1] proposal; ``<mpr>`` [1] main proposer; ``<lco>``
            [1] local contact; ``<sta>``, ``<end>`` [1] start and end dates;
            ``<cyc>`` [1] reactor cycle; ``<nbk>``, ``<pag>`` [1] logbook
            number and page; ``<env>`` [1] sample-environment code.
        Beam
            ``<zac>`` [1] zero angle (deg); ``<wle>`` [1] wavelength (Å);
            ``<lft>``, ``<rgt>`` [1] left and right slits (mm); ``<top>``,
            ``<bot>`` [1] top and bottom flags (mm).
        Grids
            ``<apr>`` [1] angular precision (deg); ``<asc>`` [1..3] angular
            grid min, max, step (deg); ``<qsc>`` [1..3] Q grid (1/Å);
            ``<rsc>`` [1..3] r grid (Å).
        Container, normaliser, absorber, environment
            ``<Cmater>``, ``<Amater>``, ``<Emater>`` [1] materials;
            ``<Cshape>``, ``<Nshape>``, ``<Ashape>``, ``<Eshape>`` [1] shape,
            [2] inner diameter (mm), [3] outer diameter (mm), [4] height
            (mm). ``<Nmater>`` is NOT read: the normaliser material is
            always ``"Vanadium"``.
        Sample
            ``<Sdescr>`` [1] description; ``<StempK>`` [1] temperature (K);
            ``<SNelem>`` [1] number of elements; ``<Smassc>`` [1] mass in
            the can (g); ``<Sdegcc>`` [1] density (g/cm³); ``<Sshape>`` [1]
            shape, [2] inner diameter, [3] outer diameter, [4] height (mm);
            ``<Sfulln>`` [1] fullness; ``<Stitle>`` [1] title;
            ``<Sato01>``, ``<Sato02>``, ... ``<SatoNN>`` (NN = 01 to
            ``<SNelem>``, two digits) [1] element symbol (a key of the
            isotope table, e.g. ``"H2"`` for deuterium), [2] number of atoms
            of that element in the formula unit.
        Data files
            ``<SaFile>``, ``<EnFile>``, ``<CoFile>``, ``<NoFile>``,
            ``<AbFile>`` [1] paths of the sample, environment, container,
            normaliser and absorber files (surrounding whitespace removed;
            relative paths are relative to the current working directory).

        Keys written by ``getRunningParams`` but not used: ``<dte>``,
        ``<log>``, ``<sli>``, ``<fla>``, ``<Nmater>``.

    Attributes
    ----------
    inputPar : Mapping
        The input dictionary (kept for `showTable`).

    Experiment (strings, as in the file):
        ``parfile`` (<par>), ``instrument`` (<ins>), ``proposal`` (<pro>),
        ``mainProposer`` (<mpr>), ``localContact`` (<lco>), ``startDate``
        (<sta>), ``endDate`` (<end>), ``cycle`` (<cyc>), ``notebook``
        (<nbk>), ``page`` (<pag>), ``envCode`` (<env>).

    Beam:
        ``zeroAngle`` : float, zero angle 2θ₀, deg.
        ``wavelength`` : float, Å.
        ``vslits`` : (left, right) slit positions, mm.
        ``hslits`` : (top, bottom) flag positions, mm.
        ``beamHeight`` : top - bottom, mm.
        ``beamWidth`` : left - right, mm.

    Grids:
        ``angularPrecision`` : float, deg.
        ``aScale`` : (min, max, step), deg.
        ``qScale`` : (min, max, step), 1/Å.
        ``rScale`` : (min, max, step), Å.

    Container, normaliser, absorber, environment (5-tuples
    ``(material, shape, inner diameter mm, outer diameter mm, height mm)``):
        ``container``, ``normaliser`` (material always ``"Vanadium"``),
        ``absorber``, ``environ``.

    Sample description and geometry:
        ``Description`` : str.
        ``Title`` : str.
        ``TempK`` : float, K.
        ``TempC`` : float, TempK - 273.15, °C.
        ``NbrElements`` : int, number of ``<SatoNN>`` lines read.
        ``Shape`` : (shape, inner diameter, outer diameter, height), mm.
        ``Fullness`` : float, as given (stored only, not used).
        ``Height`` : float, Shape[3], mm.
        ``InnerDiam``, ``OuterDiam`` : float, Shape[1], Shape[2], mm.
        ``InnerRadius``, ``OuterRadius`` : float, half the diameters, mm.
        ``Volume`` : float, π h (r_o² - r_i²) / 1000, cm³ (annulus; solid
        cylinder when InnerDiam = 0).

    Sample mass and densities:
        ``Mass`` : float, g.
        ``Density`` : float, macroscopic density of the material, g/cm³.
        ``EffDensity`` : float, Mass / Volume, g/cm³.
        ``PackingFraction`` : float, EffDensity / Density.
        ``AtomicDensity`` : float, Density × 0.602214076 / molarMass,
        atoms/Å³ (`getAtomicDensity`).
        ``EffAtomicDensity`` : float, same with EffDensity, atoms/Å³.

    Composition and averaged nuclear data (c_i = n_i / Σ n_j; element data
    from ``Element(symbol)``):
        ``atoms`` : dict symbol -> number of atoms n_i in the unit.
        ``natoms`` : float, Σ n_i (sum over the ``<SatoNN>`` lines).
        ``symbols`` : list of the symbols, in file order.
        ``conc`` : dict symbol -> c_i.
        ``nuclei`` : dict symbol -> tuple ``(symbol, n_i, c_i, weight (u),
        bcoh (complex, fm), binc (complex, fm), sig_coh, sig_inc, sig_sca,
        sig_abs (barns))``.
        ``molarMass`` : Σ c_i weight_i, g/mol per atom.
        ``scoh``, ``sinc``, ``ssca``, ``sabs`` : Σ c_i σ_i, barns/atom
        (bound; sabs at the thermal wavelength 1.798 Å).
        ``bcoh``, ``binc`` : float, MODULI |Σ c_i b_i| of the averaged
        complex scattering lengths, fm (the sign is lost; P10 in the
        appendix proposes keeping it).
        ``free`` : Σ c_i σ_sca,i [a_i/(a_i+1)]², a_i = weight_i/1.0086649
        (`getMassNumber`), free-atom scattering cross section, barns/atom.
        ``b2lowQ`` : ssca / 4π, barns/sr/atom (self scattering, low Q).
        ``b2highQ`` : free / 4π, barns/sr/atom (self scattering, high Q).
        ``sabswl`` : sabs × wavelength / 1.798, absorption at the
        wavelength, barns/atom (`getAbsXS`).

    Data arrays (``(N, 3)`` float arrays with columns x, y, error, read with
    `toscana.dataio.loading.read_3col`):
        ``Data`` (sample, ``File``), ``envData`` (``envFile``), ``conData``
        (``conFile``), ``norData`` (``norFile``), ``absData`` (``absFile``).
        The file names are kept in the attributes in brackets. ``read_3col``
        skips empty lines and every line containing ``#`` anywhere, and sets
        the error to 0 for 2-column lines or an error field ``i``/``o``.

    Raises
    ------
    KeyError
        If a required key of `inputPar` is missing (including a
        ``<SatoNN>`` up to ``<SNelem>``).
    IndexError
        If an entry has fewer fields than read.
    ValueError
        If a numeric field cannot be converted, if a file has a non-numeric
        data line (from ``read_3col``), or if an element symbol is not in
        the isotope table (from `Element`).
    OSError
        If one of the five data files cannot be opened.
    ZeroDivisionError
        If the sample volume, the density, the total number of atoms or the
        average molar mass is 0 (e.g. no ``<SatoNN>`` lines).

    Notes
    -----
    Side effects: reads the five data files; prints a two-line warning if
    the container inner diameter (``<Cshape>`` [2]) differs from the sample
    outer diameter (``<Sshape>`` [3]); `getAtomicDensity` prints a warning
    if a density is <= 0.

    If the same symbol appears in two ``<SatoNN>`` lines, ``atoms`` keeps
    the last count but ``natoms`` adds both, so the concentrations do not
    sum to 1.

    This mirrors `legacy/lib/toscana.py::Measurement` (parity-first) and is
    intentionally attribute-heavy to match legacy call sites.

    Changes from the legacy code (audit 2026-11):

    - C14: ``Volume`` = π h (r_o² - r_i²)/1000. Legacy used π h (r_o - r_i)²,
      only right for solid cylinders; for a 4/6 mm annulus the volume was 5
      times too small and ``PackingFraction`` 5 times too large.
    - P9: ``free`` is summed element by element with each element's mass
      ratio. Legacy applied (M̄/(M̄+m_n))² with the MEAN molar mass, badly
      wrong for light atoms (H₂O: 14.908 b instead of 41.116 b).
    - P13: `showTable` no longer replaces ``sys.stdout``.

    Usage::

        from toscana.dataio.running_params import getRunningParams
        from toscana.experiment import Measurement

        m = Measurement(getRunningParams("sample.par"))
        print(m.Volume, m.EffDensity, m.free)
        m.showTable()
    """

    def __init__(self, inputPar: Mapping[str, Sequence[str]]):
        """
        Build the measurement from `inputPar`.

        See the class docstring for the keys read, the attributes created
        and the exceptions raised.

        Parameters
        ----------
        inputPar : Mapping[str, Sequence]
            Dictionary returned by
            `toscana.dataio.running_params.getRunningParams`.
        """
        self.inputPar = inputPar

        self.parfile = inputPar["<par>"][1]
        self.instrument = inputPar["<ins>"][1]
        self.proposal = inputPar["<pro>"][1]
        self.mainProposer = inputPar["<mpr>"][1]
        self.localContact = inputPar["<lco>"][1]
        self.startDate = inputPar["<sta>"][1]
        self.endDate = inputPar["<end>"][1]
        self.cycle = inputPar["<cyc>"][1]
        self.notebook = inputPar["<nbk>"][1]
        self.page = inputPar["<pag>"][1]
        self.envCode = inputPar["<env>"][1]

        self.zeroAngle = float(inputPar["<zac>"][1])
        self.wavelength = float(inputPar["<wle>"][1])

        self.vslits = (float(inputPar["<lft>"][1]), float(inputPar["<rgt>"][1]))
        self.hslits = (float(inputPar["<top>"][1]), float(inputPar["<bot>"][1]))
        self.beamHeight = self.hslits[0] - self.hslits[1]
        self.beamWidth = self.vslits[0] - self.vslits[1]

        self.angularPrecision = float(inputPar["<apr>"][1])
        self.aScale = (
            float(inputPar["<asc>"][1]),
            float(inputPar["<asc>"][2]),
            float(inputPar["<asc>"][3]),
        )
        self.qScale = (
            float(inputPar["<qsc>"][1]),
            float(inputPar["<qsc>"][2]),
            float(inputPar["<qsc>"][3]),
        )
        self.rScale = (
            float(inputPar["<rsc>"][1]),
            float(inputPar["<rsc>"][2]),
            float(inputPar["<rsc>"][3]),
        )

        self.container = (
            inputPar["<Cmater>"][1],
            inputPar["<Cshape>"][1],
            float(inputPar["<Cshape>"][2]),
            float(inputPar["<Cshape>"][3]),
            float(inputPar["<Cshape>"][4]),
        )
        self.normaliser = (
            "Vanadium",
            inputPar["<Nshape>"][1],
            float(inputPar["<Nshape>"][2]),
            float(inputPar["<Nshape>"][3]),
            float(inputPar["<Nshape>"][4]),
        )
        self.absorber = (
            inputPar["<Amater>"][1],
            inputPar["<Ashape>"][1],
            float(inputPar["<Ashape>"][2]),
            float(inputPar["<Ashape>"][3]),
            float(inputPar["<Ashape>"][4]),
        )
        self.environ = (
            inputPar["<Emater>"][1],
            inputPar["<Eshape>"][1],
            float(inputPar["<Eshape>"][2]),
            float(inputPar["<Eshape>"][3]),
            float(inputPar["<Eshape>"][4]),
        )

        self.Description = inputPar["<Sdescr>"][1]
        self.TempK = float(inputPar["<StempK>"][1])
        self.TempC = float(inputPar["<StempK>"][1]) - 273.15

        self.NbrElements = int(inputPar["<SNelem>"][1])
        self.Mass = float(inputPar["<Smassc>"][1])
        self.Density = float(inputPar["<Sdegcc>"][1])
        self.Shape = (
            inputPar["<Sshape>"][1],
            float(inputPar["<Sshape>"][2]),
            float(inputPar["<Sshape>"][3]),
            float(inputPar["<Sshape>"][4]),
        )
        self.Fullness = float(inputPar["<Sfulln>"][1])
        self.Height = self.Shape[3]

        self.InnerDiam = self.Shape[1]
        self.OuterDiam = self.Shape[2]
        self.InnerRadius = self.InnerDiam / 2.0
        self.OuterRadius = self.OuterDiam / 2.0

        # Legacy warning check (keep).
        if float(self.container[2]) != self.OuterDiam:
            print(
                "WARNING! Container inner diameter should be equal",
                "to sample outer diameter",
            )
            msg = "    Container id = {} mm != Sample od = {} mm"
            print(msg.format(float(self.container[2]), self.OuterDiam))

        self.Title = inputPar["<Stitle>"][1]

        # Sample volume (cm3) of a cylinder or annulus: π h (ro² - ri²), mm³/1000.
        # Bug fix vs legacy: legacy used (ro - ri)², which is only right for
        # solid cylinders (ri = 0).
        self.Volume = (
            np.pi
            * self.Height
            / 1000.0
            * (self.OuterRadius**2 - self.InnerRadius**2)
        )
        self.EffDensity = self.Mass / float(self.Volume)
        self.PackingFraction = self.EffDensity / self.Density

        self.conFile = inputPar["<CoFile>"][1].strip()
        self.norFile = inputPar["<NoFile>"][1].strip()
        self.envFile = inputPar["<EnFile>"][1].strip()
        self.File = inputPar["<SaFile>"][1].strip()
        self.absFile = inputPar["<AbFile>"][1].strip()

        # Extra concentric layers of <LayNN> tags (sorted by tag): dicts with
        # name, role, material, inner_diam, outer_diam, height (mm). Used by
        # toscana.corrections.setup.setups_from_measurement.
        self.extra_layers = [
            {
                "name": inputPar[key][1],
                "role": inputPar[key][2],
                "material": inputPar[key][3],
                "inner_diam": float(inputPar[key][4]),
                "outer_diam": float(inputPar[key][5]),
                "height": float(inputPar[key][6]),
            }
            for key in sorted(k for k in inputPar if k.startswith("<Lay"))
        ]

        self.Data: NDArray[np.float64] = read_3col(self.File)
        self.envData: NDArray[np.float64] = read_3col(self.envFile)
        self.conData: NDArray[np.float64] = read_3col(self.conFile)
        self.norData: NDArray[np.float64] = read_3col(self.norFile)
        self.absData: NDArray[np.float64] = read_3col(self.absFile)

        atoms: dict[str, float] = {}
        natoms = 0.0
        for elem_idx in range(self.NbrElements):
            ordatom = str(elem_idx + 1).zfill(2)
            label = "<Sato" + ordatom + ">"
            natoms += float(inputPar[label][2])
            atoms[inputPar[label][1]] = float(inputPar[label][2])

        self.natoms = natoms
        self.atoms = atoms
        self.symbols = list(atoms.keys())

        conc: dict[str, float] = {}
        for symbol in list(atoms.keys()):
            conc[symbol] = atoms[symbol] / natoms
        self.conc = conc

        self.nuclei: dict[str, tuple[object, ...]] = {}
        for symbol in list(atoms.keys()):
            iso = Element(symbol)
            self.nuclei[symbol] = (
                symbol,
                atoms[symbol],
                conc[symbol],
                iso.weight,
                iso.bcoh,
                iso.binc,
                iso.sig_coh,
                iso.sig_inc,
                iso.sig_sca,
                iso.sig_abs,
            )

        self.molarMass = 0.0
        bcoh_acc: complex = 0j
        binc_acc: complex = 0j
        self.scoh = 0.0
        self.sinc = 0.0
        self.ssca = 0.0
        self.sabs = 0.0
        # Free-atom scattering cross section, element by element:
        #   σ_free = Σ c_i σ_sca,i (A_i / (A_i + 1))²,  A_i = m_i / m_n.
        # Bug fix vs legacy: legacy applied (M̄/(M̄+m_n))² with the MEAN molar
        # mass, which is badly wrong for samples with light atoms (H, D, Li).
        self.free = 0.0
        for symbol in list(atoms.keys()):
            iso = Element(symbol)
            self.molarMass += iso.weight * conc[symbol]
            bcoh_acc += iso.bcoh * conc[symbol]
            binc_acc += iso.binc * conc[symbol]
            self.scoh += iso.sig_coh * conc[symbol]
            self.sinc += iso.sig_inc * conc[symbol]
            self.ssca += iso.sig_sca * conc[symbol]
            self.sabs += iso.sig_abs * conc[symbol]
            mass_ratio = getMassNumber(iso.weight)  # m_i / m_n
            self.free += conc[symbol] * iso.sig_sca * (mass_ratio / (mass_ratio + 1.0)) ** 2

        self.bcoh = float(cmath.polar(bcoh_acc)[0])
        self.binc = float(cmath.polar(binc_acc)[0])
        self.b2lowQ = self.ssca / 4 / np.pi
        self.b2highQ = self.free / 4 / np.pi
        self.sabswl = getAbsXS(self.sabs, self.wavelength)

        self.AtomicDensity = getAtomicDensity(
            density=self.Density, molarMass=self.molarMass
        )
        self.EffAtomicDensity = getAtomicDensity(
            density=self.EffDensity, molarMass=self.molarMass
        )

    def showTable(self, outfile: str | None = None) -> None:  # noqa: N802 (legacy API)
        """
        Print the main input parameters, to stdout or to `outfile`.

        Prints, as label/value pairs taken from `inputPar`, the parameter
        file name, instrument, cycle, environment, proposal, main proposer,
        local contact, start date, logbook number and page, wavelength, zero
        angle, angular precision, and the beam flags and slits with the
        computed beam height and width (mm). See `_print_table`. The sample,
        container and composition are NOT printed.

        Parameters
        ----------
        outfile : str or None, optional
            If given (non-empty), the table is written to this file, which is
            created or OVERWRITTEN; otherwise it is printed to the current
            ``sys.stdout``. Default None.

        Raises
        ------
        OSError
            If `outfile` cannot be opened for writing.
        IndexError, KeyError
            From `_print_table` if an entry is missing or too short.

        Notes
        -----
        Bug fix vs legacy (audit 2026-11, P13): legacy replaced `sys.stdout`
        with an unclosed file and then set it to `sys.__stdout__`, which broke
        Jupyter output and leaked the file on errors. Now
        ``contextlib.redirect_stdout`` is used inside a ``with open(...)``
        block, so the previous output stream is restored and the file is
        closed even on errors.
        """
        if outfile:
            with open(outfile, "w") as fh, contextlib.redirect_stdout(fh):
                self._print_table()
        else:
            self._print_table()

    def _print_table(self) -> None:
        """
        Print the parameter table to ``sys.stdout`` (helper of `showTable`).

        Layout: a line of 120 dashes; "Input parameters from <par>"; an empty
        line; then four rows of three ``(label, value)`` pairs, each entry of
        `inputPar` unpacked with ``*``:

        1. ``<ins>``, ``<cyc>``, ``<env>``
        2. ``<pro>``, ``<mpr>``, ``<lco>``
        3. ``<sta>``, ``<nbk>``, ``<pag>``
        4. ``<wle>``, ``<zac>``, ``<apr>``

        then "Beam dimensions:" and one row with ``<top>``, ``<bot>``,
        "Height (mm)=" `beamHeight`, ``<lft>``, ``<rgt>``, "Width (mm)="
        `beamWidth`, and a final empty line.

        Raises
        ------
        IndexError
            If the entries of one row have fewer elements in total than the
            row has fields (6, or 12 for the beam row). The format strings
            assume exactly (label, value) per entry: an entry with more or
            fewer elements shifts the following fields without error, and
            surplus values at the end of a row are silently dropped.
        KeyError
            If one of these keys is missing.
        """
        print(120 * "-")
        print(f"Input parameters from {self.inputPar['<par>'][1]}")
        print("".format())
        print(
            " {:<15} {:<15} {:<20} {:<20} {:<25} {:<20}".format(
                *self.inputPar["<ins>"],
                *self.inputPar["<cyc>"],
                *self.inputPar["<env>"],
            )
        )
        print(
            " {:<15} {:<15} {:<20} {:<20} {:<25} {:<20}".format(
                *self.inputPar["<pro>"],
                *self.inputPar["<mpr>"],
                *self.inputPar["<lco>"],
            )
        )
        print(
            " {:<15} {:<15} {:<20} {:<20} {:<25} {:<20}".format(
                *self.inputPar["<sta>"],
                *self.inputPar["<nbk>"],
                *self.inputPar["<pag>"],
            )
        )
        print(
            " {:<15} {:<15} {:<20} {:<20} {:<25} {:<15}".format(
                *self.inputPar["<wle>"],
                *self.inputPar["<zac>"],
                *self.inputPar["<apr>"],
            )
        )
        print(" {}".format("Beam dimensions:"))
        print(
            " {:<9} {:<5} {:<12} {:<5} | {:<7} {:<5} | {:<10} {:<5} {:<11} {:<5} | {:<7} {:<5}".format(
                *self.inputPar["<top>"],
                *self.inputPar["<bot>"],
                "Height (mm)=  ",
                self.beamHeight,
                *self.inputPar["<lft>"],
                *self.inputPar["<rgt>"],
                "Width (mm)=  ",
                self.beamWidth,
            )
        )
        print("".format())
