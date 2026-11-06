"""
Nuclear data of elements and isotopes: the `Element` class.

`Element` reads one row of the table `toscana.isotopes.database.ISOTOPE_DICT`
(values from "Bound Scattering Lengths and Cross Sections of the Elements and
Their Isotopes", Neutron Scattering Fundamentals, vol. 44, App. 2) and
exposes it as attributes, with derived complex scattering lengths and bound
cross sections. `toscana.experiment` uses it to average cross sections and
masses over the sample composition.

Public names
------------
Element
    Nuclear data of one element or isotope (``Element("Ni")``,
    ``Element("H2")``); ``Element.getIsotope`` lists table entries.
getFree
    Free-atom value [A/(A+1)]² × bound value for an `Element`.
Positive, Negative, SemiWhole, Top, Int, String, Complex, dummyOk
    Validators ``f(key, value)`` that PRINT a warning when an override
    passed to `Element` is implausible (they never reject it).
myCorr
    Dict attribute name -> list of validators applied to overrides.
Validator
    Type alias ``Callable[[str, Any], None]``.
Elemento, elemento
    Deprecated aliases of `Element` (DeprecationWarning), resolved by the
    module ``__getattr__`` from `_DEPRECATED_ALIASES`.

Units
-----
Scattering lengths in fm; cross sections in barns (1 b = 100 fm²);
atomic masses in u; abundances in %. The column names and units are
`toscana.isotopes.database.NOMBRES` and `UNITS`.

Changes from the legacy code (audit 2026-11): C15 (bound cross sections in
barns), P15 (unknown names raise ``ValueError``), class renamed from
``Elemento``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List

import numpy as np

from toscana.isotopes.database import ISOTOPE_DICT, NOMBRES, UNITS


def Positive(key: str, number: Any) -> None:
    """
    Warn if ``float(number)`` is not strictly positive.

    Parameters
    ----------
    key : str
        Name of the parameter (only used in the message).
    number : Any
        Value to check; converted with ``float()``.

    Raises
    ------
    ValueError, TypeError
        If `number` cannot be converted to float (`Element` catches these).

    Notes
    -----
    Prints ``" WARNING. <key> parameter must have a positive value"``; never
    raises for a numeric value.
    """
    if not (float(number) > 0):
        print(f" WARNING. {key} parameter must have a positive value")


def Negative(key: str, number: Any) -> None:
    """
    Warn if ``float(number)`` is not strictly negative.

    Parameters
    ----------
    key : str
        Name of the parameter (only used in the message).
    number : Any
        Value to check; converted with ``float()``.

    Raises
    ------
    ValueError, TypeError
        If `number` cannot be converted to float.

    Notes
    -----
    Prints ``" WARNING. <key> parameter must have a negative value"``. Not
    used by any entry of `myCorr`.
    """
    if not (float(number) < 0):
        print(f" WARNING. {key} parameter must have a negative value")


def SemiWhole(key: str, number: Any) -> None:
    """
    Warn if a numeric value is not a multiple of 1/2 (nuclear spin).

    Parameters
    ----------
    key : str
        Name of the parameter (only used in the message).
    number : Any
        Value to check. Values that ``float()`` cannot convert (such as the
        table strings ``"1/2"`` or ``"NULL"``) are silently accepted.

    Notes
    -----
    Prints ``" WARNING. <key> parameter must be semi-whole"`` when
    ``float(number) % 0.5 != 0``. Never raises.
    """
    # Handle 'NULL' or non-numeric strings that might have been passed
    try:
        val = float(number)
        if (val % (1 / 2)) != 0:
            print(f" WARNING. {key} parameter must be semi-whole")
    except (ValueError, TypeError):
        pass


def Top(key: str, value: Any) -> None:
    """
    Warn if a value exceeds the upper bound of its parameter.

    Bounds: ``weight`` 248.072 (u), ``abundance`` 100.0 (%), ``Z`` 96.0;
    any other key has bound 0.0 (so every positive value warns).

    Parameters
    ----------
    key : str
        Name of the parameter; selects the bound.
    value : Any
        Value to check; converted with ``float()``.

    Raises
    ------
    ValueError, TypeError
        If `value` cannot be converted to float.

    Notes
    -----
    Prints ``" WARNING. <key> parameter can't be bigger than <top>"``.
    In `myCorr` it is used for ``weight`` and ``abundance`` only.
    """
    top = 0.0
    if key == "weight":
        top = 248.072
    elif key == "abundance":
        top = 100.0
    elif key == "Z":
        top = 96.0

    if float(value) > top:
        print(f" WARNING. {key} parameter can't be bigger than {top}")


def Int(key: str, value: Any) -> None:
    """
    Warn if ``float(value)`` is not a whole number.

    Parameters
    ----------
    key : str
        Name of the parameter (only used in the message).
    value : Any
        Value to check; converted with ``float()``.

    Raises
    ------
    ValueError, TypeError
        If `value` cannot be converted to float.

    Notes
    -----
    Prints ``" WARNING. <key> parameter has to be a whole number"``.
    """
    if not float(value).is_integer():
        print(f" WARNING. {key} parameter has to be a whole number")


def String(key: str, value: Any) -> None:
    """
    Warn if `value` is not a ``str``.

    Parameters
    ----------
    key : str
        Name of the parameter (only used in the message).
    value : Any
        Value to check.

    Notes
    -----
    Prints ``" WARNING. <key> parameter has to be a string"``. Never raises.
    """
    if not isinstance(value, str):
        print(f" WARNING. {key} parameter has to be a string")


def Complex(key: str, value: Any) -> None:
    """
    Warn if `value` is not a Python ``complex``.

    Parameters
    ----------
    key : str
        Name of the parameter (only used in the message).
    value : Any
        Value to check (a float such as ``-0.443`` also warns).

    Notes
    -----
    Prints ``" WARNING. <key> parameter has to be a complex number"``.
    Never raises.
    """
    if not isinstance(value, complex):
        print(f" WARNING. {key} parameter has to be a complex number")


def dummyOk(key: str, value: Any) -> None:
    """
    Accept any value (no-op validator, used for ``re_bcoh``).

    Parameters
    ----------
    key : str
        Ignored.
    value : Any
        Ignored.
    """
    return


def getFree(isot: Element, key: complex) -> complex:
    """
    Free-atom value of a bound quantity of an isotope.

    Computes::

        s = [A/(A+1)]² × key

    separately for the real and imaginary parts, each rounded to 4
    decimals, with A = ``isot.A``.

    Parameters
    ----------
    isot : Element
        Isotope or element; only its attribute ``A`` is used.
    key : complex
        Bound value to convert, typically a cross section in barns such as
        ``isot.scoh_bound`` (any number with ``.real`` and ``.imag`` works).

    Returns
    -------
    complex
        The free value, same units as `key`.

    Raises
    ------
    ZeroDivisionError
        If ``isot.A == -1``.

    Notes
    -----
    ``isot.A`` is the integer nucleon number from the table (59 for natural
    Ni), not the mass ratio m/m_n = weight/1.0086649 used by
    `toscana.physics.properties.getMassNumber` (58.19 for Ni) in
    `Measurement` and `setSampleInfo`. Not used elsewhere in the package.
    """
    s = complex(
        round(math.pow((isot.A / (isot.A + 1)), 2) * key.real, 4),
        round(math.pow((isot.A / (isot.A + 1)), 2) * key.imag, 4),
    )
    return s


# A validator receives (attribute name, proposed value) and only prints.
Validator = Callable[[str, Any], None]

# Validators run on each override passed to Element(**kwargs), by key.
# Keys without an entry (e.g. sig_coh, re_binc) are accepted unchecked.

myCorr: Dict[str, List[Validator]] = {
    "symbol": [String],
    "iso": [String],
    "name": [String],
    "A": [Int, Positive],
    "Z": [Int, Positive],
    "weight": [Top, Positive],
    "spin": [SemiWhole],
    "abundance": [Top, Positive],
    "life": [Positive, String],
    "re_bcoh": [dummyOk],
    "neut": [Int, Positive],
    "bcoh": [Complex],
    "binc": [Complex],
    "scoh_bound": [Complex],
    "sinc_bound": [Complex],
}


class Element:
    """
    Class that stores nuclear values for an isotope or element.

    Nuclear values are sourced from "Bound Scattering Lengths and Cross Sections
    of the Elements and Their Isotopes" in Appendix 2 of Neutron Scattering
    Fundamentals, volume 44 (table `toscana.isotopes.database.ISOTOPE_DICT`).

    Parameters
    ----------
    iso_name : str
        Key of ``ISOTOPE_DICT``: an element with natural abundance (``"Ni"``,
        ``"H"``, ``"V"``), an isotope (``"Ni58"``, ``"Li6"``), or the
        special entries ``"H1"`` (proton), ``"H2"`` (deuteron, use it for
        D), ``"H3"`` (triton), ``"n1"`` (neutron). Case sensitive.
    table : int, optional
        0 (default): print nothing. 1: print the table of this entry (with
        the overrides). 2: print the tables of all entries of the table with
        the same chemical symbol (natural element and its isotopes, built
        from the table WITHOUT the overrides). Any other value: as 0.
    **kwargs : Any
        Overrides of table values, by attribute name (e.g.
        ``Element("V", sig_inc=5.2)``). See Notes.

    Attributes
    ----------
    symbol : str
        Chemical symbol (``"Ni"``; ``"H"`` also for H2, H3).
    iso : str
        ``"nat"`` for the natural element, otherwise the mass number as a
        string (``"58"``).
    name : str
        Name (``"Nickel"``; ``"deuteron"`` for H2).
    A : int
        Nucleon (mass) number; for natural elements the table gives a rounded
        value (59 for Ni, 1 for H).
    Z : int
        Atomic number.
    weight : float
        Atomic mass, u (g/mol); for natural elements the standard atomic
        weight (58.6934 for Ni).
    excess : float
        Mass excess as tabulated (MeV; e.g. -60.2277 for Ni58), 0 if absent.
    spin : str
        Nuclear spin as tabulated (``"1/2"``); ``"0"`` when absent ("NULL").
    parity : str
        Parity (``"+"``/``"-"``); ``"0"`` when absent.
    abundance : float
        Natural abundance, % (100 for natural elements).
    life : str
        Half-life as tabulated (``"inf"`` for stable, ``"10.24m"``...).
    re_bcoh, im_bcoh : float
        Real and imaginary parts of the bound coherent scattering length, fm.
    bplus, bminus : float
        Bound scattering lengths of the I + 1/2 and I - 1/2 spin states, fm
        (0 if absent).
    re_binc, im_binc : float
        Real and imaginary parts of the bound incoherent scattering length,
        fm (0 if absent).
    sig_coh, sig_inc, sig_sca : float
        Bound coherent, incoherent and total scattering cross sections, barns.
    sig_abs : float
        Absorption cross section for 2200 m/s neutrons (λ = 1.798 Å), barns.
    contrast : float
        The "contrast" column of the table, as tabulated (0 if absent).
    bcoh, binc : complex
        Complex scattering lengths re + i·im, fm (or the ``bcoh``/``binc``
        overrides).
    scoh_bound, sinc_bound : complex
        4π |b|² / 100 from `bcoh` and `binc`, barns (imaginary part 0),
        each part rounded to 4 decimals. For Ni 4π(10.3)²/100 = 13.3317 b,
        consistent with ``sig_coh`` = 13.3 b. ``sinc_bound`` is 0 when the
        table has no incoherent length (e.g. natural V, although
        ``sig_inc`` = 5.08 b): use ``sig_inc`` for the cross section.
    neut : int
        Number of neutrons A - Z (0 if A or Z is not an integer).
    data : dict
        All values by column name (`toscana.isotopes.database.NOMBRES`), raw
        table values with ``"NULL"`` replaced by 0 and the overrides applied,
        plus the derived ``bcoh``, ``binc``, ``scoh_bound``, ``sinc_bound``,
        ``neut``. The attributes above are typed copies (``int``, ``float``,
        ``str``, ``complex``; unconvertible values become 0, 0.0 or the
        string form).

    Raises
    ------
    ValueError
        If `iso_name` is not a key of ``ISOTOPE_DICT`` (audit 2026-11, P15;
        legacy printed a message and returned a half-built object).

    Notes
    -----
    Overrides (`kwargs`):

    - If the key has validators in `myCorr` they are run first; they only
      PRINT warnings for implausible values (``A``, ``Z``: whole and
      positive; ``weight``: positive and <= 248.072; ``abundance``: positive
      and <= 100; ``spin``: multiple of 1/2; ``symbol``, ``iso``, ``name``:
      str; ``life``: str (a numeric value warns); ``bcoh``, ``binc``, ``scoh_bound``,
      ``sinc_bound``: complex; ``neut``: whole and positive). Their
      ValueError/TypeError are swallowed. The value is always accepted.
    - The value replaces the table value if the key is a column of the
      table or a derived key; any other key (e.g. a misspelt one) is
      silently ignored.
    - ``bcoh`` and ``binc`` overrides replace the complex lengths built from
      re/im (the ``re_bcoh``... attributes are NOT updated). Overrides of
      ``scoh_bound``, ``sinc_bound`` and ``neut`` have no effect: these are
      recomputed. Cross sections and scattering lengths are not made
      consistent with each other (overriding ``sig_coh`` does not change
      ``bcoh``, and vice versa).

    Changes from the legacy code (audit 2026-11):

    - C15: ``scoh_bound`` and ``sinc_bound`` are in barns (divided by 100);
      legacy gave 4π|b|² in fm² labelled barns (Ni: 1333 instead of 13.3).
    - P15: unknown names raise ``ValueError``.
    - Renamed from ``Elemento``; ``Elemento`` and ``elemento`` remain as
      deprecated aliases (module ``__getattr__``, DeprecationWarning).
    - C16: the table value of V is b_coh = -0.443 fm, σ_coh = 0.02466 b.

    Usage::

        from toscana.isotopes import Element

        ni = Element("Ni")
        ni.sig_coh, ni.bcoh, ni.scoh_bound   # 13.3, (10.3+0j), (13.3317+0j)
        d = Element("H2")                     # deuterium
        Element("Ni", table=2)                # print all Ni isotopes
    """

    # Commonly accessed fields (explicit for `mypy --strict`).
    symbol: str
    iso: str
    name: str
    A: int
    Z: int
    weight: float
    excess: float
    spin: str
    parity: str
    abundance: float
    life: str
    re_bcoh: float
    im_bcoh: float
    bplus: float
    bminus: float
    re_binc: float
    im_binc: float
    sig_coh: float
    sig_inc: float
    sig_sca: float
    sig_abs: float
    contrast: float
    neut: int
    bcoh: complex
    binc: complex
    scoh_bound: complex
    sinc_bound: complex

    def __init__(self, iso_name: str, table: int = 0, **kwargs: Any):
        """
        Initialize an Element object.

        Steps: look up `iso_name`; map the table tuple to `data` by column
        name; replace ``"NULL"`` by 0; apply and validate the overrides;
        compute ``bcoh``, ``binc``, ``scoh_bound``, ``sinc_bound``, ``neut``;
        copy everything to typed attributes; print the table if requested.
        See the class docstring for details.

        Parameters
        ----------
        iso_name : str
            Name of the isotope (e.g., 'Li', 'Li6', 'Li7').
        table : int, optional
            0: no display, 1: display table for isotope, 2: display all isotopes
            of the same element. Default is 0.
        **kwargs : Any
            Override default values from ISOTOPE_DICT.

        Raises
        ------
        ValueError
            If `iso_name` is not in ``ISOTOPE_DICT``.

        Notes
        -----
        With ``table=1`` prints `iso_name` and a three-column table
        PARAMETER / VALUE / UNITS of `data` (units from
        `toscana.isotopes.database.UNITS`, ``"-"`` if missing). With
        ``table=2`` prints, for every entry with the same symbol (in table
        order), its key in upper case and the same table, building a new
        `Element` for each.
        """
        if iso_name not in ISOTOPE_DICT:
            # Bug fix vs legacy: legacy printed a message and returned a
            # half-initialised object (AttributeError later on).
            raise ValueError(
                f"The isotope {iso_name!r} does not exist in this dictionary."
            )

        # Load data from dictionary
        data_tuple = ISOTOPE_DICT[iso_name]
        self.data: Dict[str, Any] = dict(zip(NOMBRES, data_tuple))

        # Replace 'NULL' with 0 and initialize derived fields
        for key in self.data:
            if self.data[key] == "NULL":
                self.data[key] = 0

        self.data.update(
            {"bcoh": 0j, "binc": 0j, "scoh_bound": 0j, "sinc_bound": 0j, "neut": 0}
        )

        # Apply overrides from kwargs and validate
        for key, value in kwargs.items():
            if key in myCorr:
                for func in myCorr[key]:
                    try:
                        func(key, value)
                    except (ValueError, TypeError):
                        pass
            if key in self.data:
                self.data[key] = value

        # Calculate derived physical properties
        bcoh = complex(
            float(self.data.get("re_bcoh", 0)), float(self.data.get("im_bcoh", 0))
        )
        binc = complex(
            float(self.data.get("re_binc", 0)), float(self.data.get("im_binc", 0))
        )

        # Allow overrides for complex Scattering Lengths
        if "bcoh" in kwargs:
            bcoh = kwargs["bcoh"]
        if "binc" in kwargs:
            binc = kwargs["binc"]

        self.data["bcoh"] = bcoh
        self.data["binc"] = binc

        # Bound cross sections in barns: 4π|b|^2 with b in fm gives fm^2, and
        # 1 barn = 100 fm^2. (Bug fix vs legacy: no /100, values were 100x.)
        scoh_bound = 4 * np.pi * bcoh * bcoh.conjugate() / 100.0
        sinc_bound = 4 * np.pi * binc * binc.conjugate() / 100.0

        # Rounding logic from legacy content
        self.data["scoh_bound"] = complex(
            round(scoh_bound.real, 4), round(scoh_bound.imag, 4)
        )
        self.data["sinc_bound"] = complex(
            round(sinc_bound.real, 4), round(sinc_bound.imag, 4)
        )

        # Number of neutrons = A - Z
        try:
            self.data["neut"] = int(self.data["A"]) - int(self.data["Z"])
        except (ValueError, TypeError):
            self.data["neut"] = 0

        # Handle second pass of overrides for properties like s_bound (though not in NOMBRES)
        # For legacy compatibility, we keep this pattern if needed, but here we just map to attributes.

        # Map internal dictionary to object attributes for easy access (e.g., element.sig_sca)
        for key, value in self.data.items():
            setattr(self, key, value)

        # Ensure a typed view for commonly-used attributes (legacy data can contain mixed types).
        def _as_int(value: Any) -> int:
            """Convert to int: int(value), else int(float(value)), else 0."""
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return 0

        def _as_float(value: Any) -> float:
            """Convert to float: float(value), else 0.0."""
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        self.symbol = str(self.data.get("symbol", ""))
        self.iso = str(self.data.get("iso", ""))
        self.name = str(self.data.get("name", ""))
        self.A = _as_int(self.data.get("A", 0))
        self.Z = _as_int(self.data.get("Z", 0))
        self.weight = _as_float(self.data.get("weight", 0.0))
        self.excess = _as_float(self.data.get("excess", 0.0))
        self.spin = str(self.data.get("spin", ""))
        self.parity = str(self.data.get("parity", ""))
        self.abundance = _as_float(self.data.get("abundance", 0.0))
        self.life = str(self.data.get("life", ""))
        self.re_bcoh = _as_float(self.data.get("re_bcoh", 0.0))
        self.im_bcoh = _as_float(self.data.get("im_bcoh", 0.0))
        self.bplus = _as_float(self.data.get("bplus", 0.0))
        self.bminus = _as_float(self.data.get("bminus", 0.0))
        self.re_binc = _as_float(self.data.get("re_binc", 0.0))
        self.im_binc = _as_float(self.data.get("im_binc", 0.0))
        self.sig_coh = _as_float(self.data.get("sig_coh", 0.0))
        self.sig_inc = _as_float(self.data.get("sig_inc", 0.0))
        self.sig_sca = _as_float(self.data.get("sig_sca", 0.0))
        self.sig_abs = _as_float(self.data.get("sig_abs", 0.0))
        self.contrast = _as_float(self.data.get("contrast", 0.0))
        self.neut = _as_int(self.data.get("neut", 0))
        self.bcoh = complex(self.data.get("bcoh", 0j))
        self.binc = complex(self.data.get("binc", 0j))
        self.scoh_bound = complex(self.data.get("scoh_bound", 0j))
        self.sinc_bound = complex(self.data.get("sinc_bound", 0j))

        # Table display logic
        if table == 1:
            print("{:<15}".format(iso_name))
            print("{:<10} {:<10} {:<10}".format("PARAMETER", "VALUE", "UNITS"))
            for key in self.data:
                val = self.data[key]
                unit = UNITS.get(key, "-")
                print("{:<10} {:<10} {:<10}".format(key, str(val), unit))

        if table == 2:
            symbol = self.data["symbol"]
            for key in ISOTOPE_DICT:
                if ISOTOPE_DICT[key][0] == symbol:
                    iso_obj = Element(key)
                    print("\n")
                    print("{:<15}".format(key.upper()))
                    print("\n")
                    print("{:<10} {:<10} {:<10}".format("PARAMETER", "VALUE", "UNITS"))
                    for k in iso_obj.data:
                        val = iso_obj.data[k]
                        unit = UNITS.get(k, "-")
                        print("{:<10} {:<10} {:<10}".format(k, str(val), unit))

    @classmethod
    def getIsotope(cls, iso_name: str) -> List[str]:
        """
        List the table entries of an element, or all element names.

        Parameters
        ----------
        iso_name : str
            Element NAME as in the table (``"Nickel"``, not ``"Ni"``), or
            ``"All"``.

        Returns
        -------
        list of str
            For an element name: the names usable with `Element` of all
            entries with that name, in table order, built as the symbol for
            the natural element (iso ``"nat"``) and symbol + mass number
            otherwise, e.g. ``["Ni58", "Ni60", "Ni61", "Ni62", "Ni64",
            "Ni"]``. Note that H1, H2, H3 have the names ``"proton"``,
            ``"deuteron"``, ``"triton"``, so ``getIsotope("Hydrogen")`` gives
            ``["H"]`` only. An unknown name gives an empty list.

            For ``"All"``: the names of the table in order, consecutive
            duplicates removed (``["neutron", "proton", "deuteron",
            "triton", "Hydrogen", ...]``).
        """
        lista = []
        if iso_name == "All":
            xName = ""
            # Sort by name to match legacy behavior of tracking name changes
            # Though ISOTOPE_DICT is already somewhat ordered
            for key in ISOTOPE_DICT:
                name = ISOTOPE_DICT[key][2]
                if name != xName:
                    lista.append(name)
                xName = name
        else:
            for key in ISOTOPE_DICT:
                if ISOTOPE_DICT[key][2] == iso_name:
                    if ISOTOPE_DICT[key][1] == "nat":
                        lista.append(ISOTOPE_DICT[key][0])
                    else:
                        lista.append(ISOTOPE_DICT[key][0] + ISOTOPE_DICT[key][1])
        return lista

    def __repr__(self) -> str:
        """Return ``"Element('<symbol>', iso='<iso>', name='<name>')"``."""
        return f"Element('{self.symbol}', iso='{self.iso}', name='{self.name}')"


# Deprecated Spanish names, kept so existing notebooks keep working.
_DEPRECATED_ALIASES = {"Elemento": "Element", "elemento": "Element"}


def __getattr__(name: str) -> Any:
    """
    Resolve the deprecated names ``Elemento`` and ``elemento`` (PEP 562).

    Parameters
    ----------
    name : str
        Attribute requested from the module.

    Returns
    -------
    type
        `Element`, for a name in `_DEPRECATED_ALIASES`.

    Raises
    ------
    AttributeError
        For any other unknown name.

    Notes
    -----
    Emits a ``DeprecationWarning`` "`Elemento` is deprecated; use `Element`
    instead." pointing to the caller.
    """
    if name in _DEPRECATED_ALIASES:
        new = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"`{name}` is deprecated; use `{new}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
