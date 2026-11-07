"""
nuclear_data.py - Atomic masses and neutron cross sections of the elements and
isotopes, from tables.

What is available and where it comes from
-----------------------------------------
* ATOMIC MASSES (needed to convert a density into a number of atoms per cm^3):
    1. the optional package `periodictable` if it is installed
       (pip install periodictable): masses of elements AND isotopes;
    2. otherwise a small built-in table in this file: standard atomic weights
       of the natural elements (4-5 significant figures, enough for
       attenuation estimates) and the masses of some isotopes that are common
       in neutron work. An isotope that is in neither is approximated by its
       mass number (the error is below 0.1 %) and a warning is issued.
* NEUTRON CROSS SECTIONS (scattering and absorption, in barn per atom):
    only from the optional package `periodictable`, which contains the
    tabulated values for the natural elements and for many isotopes, at the
    thermal reference velocity 2200 m/s (wavelength 1.798 Angstrom, energy
    25.3 meV). There is NO built-in table of cross sections on purpose: a
    wrong number would silently give a wrong result. If the package is not
    installed you provide the cross sections yourself (see layer_builder.py).

Using another database
----------------------
`AtomicData` is a small class with two methods, `mass(species)` and
`neutron_xs(species)`. To use another source (a file of your own, another
package), write a class with those two methods and give an object of it to
`layer_builder.make_layer_tuple(..., data=your_object)`.

Units: masses in g/mol (= atomic mass units u), cross sections in barn
(1 barn = 1e-24 cm^2) per atom.
"""

import warnings

# `Species` describes "natural Fe" or "the isotope 56Fe" (see formula.py).
from formula import Species


class TableError(LookupError):
    """A value is not available in the tables."""


# --------------------------------------------------------------------------
# Built-in table 1: standard atomic weights of the natural elements (g/mol)
# --------------------------------------------------------------------------
# For elements with no stable isotope the mass number of the longest-lived
# isotope is given (Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Np, Pu, Am, Cm).
NATURAL_MASSES = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04,
    "Pa": 231.04, "U": 238.03, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0,
}

# --------------------------------------------------------------------------
# Built-in table 2: masses of some isotopes (g/mol), key = (symbol, mass number)
# --------------------------------------------------------------------------
ISOTOPE_MASSES = {
    ("H", 1): 1.007825, ("H", 2): 2.014102, ("H", 3): 3.016049,
    ("He", 3): 3.016029, ("He", 4): 4.002603,
    ("Li", 6): 6.015123, ("Li", 7): 7.016003,
    ("B", 10): 10.012937, ("B", 11): 11.009306,
    ("C", 12): 12.000000, ("C", 13): 13.003355,
    ("N", 14): 14.003074, ("N", 15): 15.000109,
    ("O", 16): 15.994915, ("O", 17): 16.999132, ("O", 18): 17.999160,
    ("Fe", 56): 55.934936, ("Fe", 57): 56.935393,
    ("Ni", 58): 57.935343, ("Ni", 62): 61.928345, ("Ni", 64): 63.927966,
    ("Cd", 113): 112.904401,
    ("Gd", 155): 154.922622, ("Gd", 157): 156.923960,
    ("U", 235): 235.043930, ("U", 238): 238.050788,
}


# --------------------------------------------------------------------------
# The data object used by layer_builder.py
# --------------------------------------------------------------------------
class AtomicData:
    """Access to atomic masses and neutron cross sections.

    use_periodictable : if True (default) the package `periodictable` is used
        when it is installed. Set it to False to force the built-in masses
        (the cross sections are then not available).
    """

    def __init__(self, use_periodictable=True):
        self._pt = None
        if use_periodictable:
            try:
                import periodictable           # optional dependency
                self._pt = periodictable
            except ImportError:
                self._pt = None                # not installed: built-in table only

    @property
    def has_neutron_data(self):
        """True if tabulated neutron cross sections can be used."""
        return self._pt is not None

    # ---- internal -------------------------------------------------------
    def _pt_item(self, species):
        """The element (natural) or isotope object of `periodictable`."""
        element = self._pt.elements.symbol(species.symbol)
        if species.mass_number is None:
            return element
        return element[species.mass_number]    # periodictable: Fe[56] is the isotope 56Fe

    # ---- masses ---------------------------------------------------------
    def mass(self, species):
        """Mass in g/mol of one atom (mole of atoms) of `species`."""
        # 1) the package periodictable, if available
        if self._pt is not None:
            try:
                return float(self._pt_item(species).mass)
            except Exception:
                pass                           # not found there: use the built-in table

        # 2) built-in tables
        if species.mass_number is None:
            if species.symbol in NATURAL_MASSES:
                return NATURAL_MASSES[species.symbol]
            raise TableError(
                f"No mass for element {species.symbol}. Give it with "
                f"masses={{'{species.symbol}': value_in_g_per_mol}}.")
        key = (species.symbol, species.mass_number)
        if key in ISOTOPE_MASSES:
            return ISOTOPE_MASSES[key]

        # 3) isotope not tabulated: use the mass number (error < 0.1 %)
        warnings.warn(f"Mass of {species} not tabulated: using the mass number "
                      f"{species.mass_number}. Give an exact value with "
                      f"masses={{'{species.key}': value}} if needed.")
        return float(species.mass_number)

    # ---- neutron cross sections ----------------------------------------
    def neutron_xs(self, species):
        """Tabulated neutron cross sections of ONE atom of `species`, at the
        thermal velocity 2200 m/s, in barn.

        Returns (sigma_scattering, sigma_absorption), where the scattering
        cross section is coherent + incoherent.
        """
        if self._pt is None:
            raise TableError(
                f"No tabulated cross sections for {species}: the package "
                "'periodictable' is not installed (pip install periodictable). "
                "Give the values yourself with sigma_scat/sigma_abs or "
                f"cross_sections={{'{species.key}': (scattering, absorption)}}.")
        try:
            neutron = self._pt_item(species).neutron
            coherent, incoherent = neutron.coherent, neutron.incoherent
            absorption = neutron.absorption
        except Exception as error:             # e.g. isotope unknown to the package
            raise TableError(f"periodictable has no neutron data for {species}: "
                             f"{error}") from error
        if coherent is None or incoherent is None or absorption is None:
            raise TableError(f"The neutron data of {species} are incomplete in "
                             "periodictable: give the values yourself.")
        return float(coherent) + float(incoherent), float(absorption)
