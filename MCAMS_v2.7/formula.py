"""
formula.py - Read a chemical formula such as "H2O", "Fe2O3Na2Li" or "[2H]2O".

The result is a dictionary that says how many atoms of each SPECIES the
formula contains. A species is either a natural element ("Fe": the natural
mixture of isotopes) or one specific isotope ("[56Fe]").

Notation (the proposal)
-----------------------
    Natural element       : its symbol, capital first letter     Fe   O   Na   Li
    Number of atoms       : number after the symbol (default 1)  H2O   NH3   Fe2O3
    Fractional numbers    : decimals are allowed                 La0.7Sr0.3MnO3
    ISOTOPE               : mass number + symbol in [ ] brackets [2H]   [56Fe]   [10B]
    Deuterium, tritium    : the usual shorthands D and T         D2O   =   [2H]2O
    Groups                : round parentheses, with a number     Ca(OH)2   Al2(SO4)3
    Hydrates              : separate with * (or the dot ·)       CuSO4*5H2O
    Mixture of isotopes   : list each isotope with its amount    [6Li]0.5[7Li]0.5F
    Enriched compound     : isotope inside the formula           [10B]4C   Fe2[17O]3

Why square brackets for isotopes? A number placed before a symbol is
ambiguous in a formula ("H22H": H22 and H, or H2 and 2H?). Brackets remove
every doubt, and a bracket can never be confused with a number of atoms or
with a group (groups use round parentheses).

Examples
--------
    >>> parse_formula("H2O")
    {Species(symbol='H', mass_number=None): 2.0, Species(symbol='O', mass_number=None): 1.0}
    >>> parse_formula("D2O") == parse_formula("[2H]2O")
    True

What this file contains
-----------------------
    ELEMENT_SYMBOLS   : the symbols of the 118 elements, used to validate the text.
    Species           : an element, natural or a given isotope.
    parse_formula     : text -> {Species: number of atoms}.
    parse_species     : text "Fe", "56Fe", "[56Fe]", "D" -> Species.
    FormulaError      : the error raised for an incorrect formula.

No NumPy is needed here: it is plain Python text processing, using the
`re` (regular expressions) module of the standard library. A regular
expression is a pattern that describes pieces of text; the one below
describes the different "words" (tokens) that can appear in a formula.
"""

import re
from dataclasses import dataclass
from typing import Optional

# The 118 chemical elements, in order of atomic number.
ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og")

# Shorthand names of the hydrogen isotopes: D = 2H (deuterium), T = 3H (tritium).
_ALIASES = {"D": ("H", 2), "T": ("H", 3)}


class FormulaError(ValueError):
    """The chemical formula (or a species name) is not valid."""


# --------------------------------------------------------------------------
# A species: natural element or one isotope
# --------------------------------------------------------------------------
@dataclass(frozen=True)       # frozen -> immutable, so it can be a dictionary key
class Species:
    """A kind of atom.

    symbol      : chemical symbol, e.g. "Fe".
    mass_number : None for the natural element (natural mixture of isotopes),
                  or the number of nucleons of the isotope, e.g. 56.
    """
    symbol: str
    mass_number: Optional[int] = None

    @property
    def key(self):
        """Short text without brackets: "Fe" for natural iron, "56Fe" for the
        isotope. It is the form used for the keys of the dictionaries that
        you can give to layer_builder (masses, cross sections)."""
        if self.mass_number is None:
            return self.symbol
        return f"{self.mass_number}{self.symbol}"

    def __str__(self):
        """The way it is written inside a formula: Fe or [56Fe]."""
        return self.symbol if self.mass_number is None else f"[{self.key}]"


def _make_species(symbol, mass_number=None):
    """Create a Species after checking the symbol (and the D/T shorthands)."""
    if mass_number is None and symbol in _ALIASES:
        symbol, mass_number = _ALIASES[symbol]
    if symbol not in ELEMENT_SYMBOLS:
        raise FormulaError(f"'{symbol}' is not a chemical element symbol.")
    if mass_number is not None and mass_number < 1:
        raise FormulaError(f"Invalid mass number {mass_number} for {symbol}.")
    return Species(symbol, mass_number)


# One species written alone: "Fe", "56Fe", "[56Fe]", "D".
_SPECIES_RE = re.compile(r"^\[?(\d+)?([A-Z][a-z]?)\]?$")


def parse_species(text):
    """Convert the text of one species ("Fe", "56Fe", "[56Fe]", "D") into a
    `Species`. Used to read the keys of user dictionaries."""
    m = _SPECIES_RE.match(text.strip())
    if m is None:
        raise FormulaError(f"Cannot read '{text}' as an element or isotope "
                           "(examples: Fe, 56Fe, [56Fe], D).")
    mass_number = int(m.group(1)) if m.group(1) else None
    return _make_species(m.group(2), mass_number)


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------
# The "words" (tokens) of a formula. Each alternative has a name (?P<name>...):
#   isotope : [56Fe]      -> "[" digits  Capital+lowercase? "]"
#   element : Fe, O, D    -> a capital letter, optionally followed by 1 lowercase letter
#   number  : 2, 0.5      -> digits, optionally with a decimal part
#   open / close : ( )    -> a group
_TOKEN_RE = re.compile(r"""
      (?P<isotope>\[(?P<A>\d+)(?P<isym>[A-Z][a-z]?)\])
    | (?P<element>[A-Z][a-z]?)
    | (?P<number>\d+(?:\.\d+)?)
    | (?P<open>\()
    | (?P<close>\))
""", re.VERBOSE)


def _add(counts, species, amount):
    """Add `amount` atoms of `species` to the dictionary `counts`."""
    counts[species] = counts.get(species, 0.0) + amount


def _parse_part(text, original):
    """Parse a formula without hydrate separators. Returns {Species: amount}.

    Method: read the text token by token, keeping a STACK of dictionaries.
    The top of the stack is the group being read (the whole formula at the
    start). "(" opens a new group; ")" closes it and adds it to the group
    below. A number multiplies the last thing read (an atom or a group).
    """
    stack = [{}]          # stack[-1] = dictionary of the group being read
    last = None           # what the next number multiplies: ("atom", sp) or ("group", dict)
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise FormulaError(f"Unexpected character '{text[pos]}' at position "
                               f"{pos + 1} of '{original}'.")
        pos = m.end()

        if m.group("isotope") or m.group("element"):
            if m.group("isotope"):
                sp = _make_species(m.group("isym"), int(m.group("A")))
            else:
                sp = _make_species(m.group("element"))
            _add(stack[-1], sp, 1.0)             # one atom, a number may change it
            last = ("atom", sp)

        elif m.group("number"):
            if last is None:
                raise FormulaError(f"A number must follow an element or a "
                                   f"closing parenthesis ('{original}').")
            value = float(m.group("number"))
            if value <= 0:
                raise FormulaError(f"Amounts must be positive ('{original}').")
            kind, item = last
            if kind == "atom":
                _add(stack[-1], item, value - 1.0)       # 1 atom was already counted
            else:
                for sp, amount in item.items():
                    _add(stack[-1], sp, amount * (value - 1.0))
            last = None

        elif m.group("open"):
            stack.append({})
            last = None

        else:  # closing parenthesis
            if len(stack) == 1:
                raise FormulaError(f"')' without matching '(' in '{original}'.")
            group = stack.pop()
            if not group:
                raise FormulaError(f"Empty parentheses in '{original}'.")
            for sp, amount in group.items():
                _add(stack[-1], sp, amount)
            last = ("group", group)

    if len(stack) != 1:
        raise FormulaError(f"'(' without matching ')' in '{original}'.")
    return stack[0]


def parse_formula(formula):
    """Convert a chemical formula into {Species: number of atoms}.

    See the top of this file for the notation. Raises `FormulaError` with a
    message that points at the problem if the formula cannot be read.
    """
    text = formula.replace(" ", "")
    if not text:
        raise FormulaError("The formula is empty.")

    # Hydrates: "CuSO4*5H2O" -> parts "CuSO4" and "5H2O" (a number in front of
    # a part multiplies it). The dot "." is NOT a separator because it is used
    # in decimals; use "*" or the middle dot "·".
    parts = re.split(r"[*·]", text)

    total = {}
    for i, part in enumerate(parts):
        coefficient = 1.0
        lead = re.match(r"\d+(?:\.\d+)?", part)
        if lead:
            if i == 0:
                raise FormulaError(
                    f"A formula cannot start with a number ('{formula}'). "
                    "For an isotope write the mass number inside brackets, "
                    "e.g. [2H] or [56Fe].")
            coefficient = float(lead.group())
            part = part[lead.end():]
        if not part:
            raise FormulaError(f"Empty part in '{formula}'.")
        for sp, amount in _parse_part(part, formula).items():
            _add(total, sp, amount * coefficient)
    return total


def format_composition(composition):
    """Readable text of a composition, e.g. "[2H]:2  O:1"."""
    return "  ".join(f"{sp}:{amount:g}" for sp, amount in composition.items())


# --------------------------------------------------------------------------
# Demo:  python formula.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    for f in ("H2O", "NH3", "Fe2O3Na2Li", "D2O", "[2H]2O", "[56Fe]2O3",
              "Ca(OH)2", "Al2(SO4)3", "La0.7Sr0.3MnO3", "CuSO4*5H2O",
              "[10B]4C", "[6Li]0.5[7Li]0.5F"):
        print(f"{f:<20} -> {format_composition(parse_formula(f))}")

    print("\nExamples of incorrect formulas:")
    for f in ("Xx2", "H2(", "h2o", "2H2O", "Ca(OH2", "H2O)"):
        try:
            parse_formula(f)
        except FormulaError as error:
            print(f"{f:<8} -> {error}")
