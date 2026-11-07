"""
parse_line.py  --  Python translation of  parse_line.f
========================================================

CORRECT is driven by VMS-DCL-like command lines, e.g.

    sample  vsample.dat 0.3 /density=0.0723 /temperature=300
    vanadium van.dat 0.32 /smoothing=-5
    component 1 v
    execute /nolist

Grammar (as implemented by the Fortran PARSE_LINE):
  * blank line or a line starting with '!'  -> comment, ignored
  * Fortran converts everything to lower case except text between double
    quotes; the Python version lower-cases only verbs and qualifier names, so
    that file names keep their case on Linux/macOS
  * the line is split into elements separated by blanks or by '/'
  * element 1 is the VERB; it may be abbreviated as long as it is unambiguous
  * elements starting with '/' are QUALIFIERS, optionally '/name=value';
    abbreviations are allowed (the Fortran does a substring search in the
    verb definition string, we do the same)
  * other elements are PARAMETERS
  * every parameter and qualifier value is also converted to a number when
    possible (Fortran: READ with format '(fN.0)'); 0.0 when it is not a number.

Python standard library used
----------------------------
re.split / re.findall : regular expressions (module `re`) used to cut the
                        line into elements while respecting double quotes.
str.lower, str.strip, str.startswith : string methods.
dataclasses.dataclass : simple record type for the parsed result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Fortran: DATA VERBS / ... / in PROGRAM CORRECT.  Each entry lists the verb
# followed by its allowed qualifiers.
# 'faithful' is a Python addition (switch to reproduce CORRECT 2.24 bugs).
VERBS = [
    "background /furnace /container /sample /vanadium",
    "beam",
    "cadmium",
    "calibration",
    "collimator",
    "component",
    "container /type /density /absorption /scattering /multiplier",
    "execute /nolist /output",
    "faithful",
    "furnace /density /absorption /scattering",
    "geometry",
    "instrument",
    "list",
    "low_angle /furnace /container /vanadium /background",
    "monitor",
    "output",
    "placzek",
    "quit",
    "range",
    "sample /density /temperature /fullness /cs_file",
    "spectrum /detector /default /to",
    "title",
    "vanadium /density /absorption /scattering /smoothing /furnace /multiplier /container",
    "wavelength",
    "write /output",
    "xin",
    "xout",
]


@dataclass
class ParsedLine:
    """Result of parsing one command line.

    Fortran arguments of PARSE_LINE  ->  attributes
        verb            -> verb ('' if empty/comment/error)
        qual(*)         -> qual   (full qualifier names, e.g. '/density')
        qval(*)         -> qval   (text after '=', '' if none)
        rqval(*)        -> rqval  (numeric value of qval, 0.0 if not a number)
        nqual           -> len(qual)
        param(*)        -> param
        rparam(*)       -> rparam
        nparams         -> len(param)
    """
    verb: str = ""
    qual: List[str] = field(default_factory=list)
    qval: List[str] = field(default_factory=list)
    rqval: List[float] = field(default_factory=list)
    param: List[str] = field(default_factory=list)
    rparam: List[float] = field(default_factory=list)


def _to_float(s: str) -> float:
    """Fortran 'READ(string,'(fN.0)') x' with ERR= -> 0.  Fortran also accepts
    D exponents (1.0d-3), so 'd' is translated to 'e'."""
    try:
        return float(s.replace("d", "e").replace("D", "e"))
    except ValueError:
        return 0.0


def _split_elements(buf: str, allowed: List[str]) -> List[str]:
    """Split a command line into elements.

    Fortran: an element ends at a blank or at any '/' (VMS/Windows file names
    contain no '/').  On Linux/macOS paths contain '/', so here a '/' starts a
    new element (a qualifier) only if the word after it is the beginning of
    one of the verb's qualifiers (`allowed`), e.g. 'file.dat/dens=0.07'.
    Anything else containing '/' (e.g. /home/me/run1.dat) is a parameter.
    Quote a path if one of its directories is named like a qualifier.
    """
    words = re.findall(r'(?:"[^"]*"|[^\s"])+', buf)
    elements = []
    for w in words:
        if w.startswith('"'):
            elements.append(w)
            continue
        cuts = [0]
        for m in re.finditer("/", w):
            name = re.match(r"/([^/=]*)", w[m.start():]).group(1).lower()
            if name and any(q[1:].startswith(name) for q in allowed):
                cuts.append(m.start())
        cuts.append(len(w))
        elements += [w[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if w[a:b]]
    return elements


def parse_line(line: str, verbs: List[str] = VERBS) -> ParsedLine:
    """Parse one command line.  Fortran: SUBROUTINE PARSE_LINE."""
    out = ParsedLine()
    s = line.strip()
    if not s or s.startswith("!"):
        return out
    if s.count('"') % 2:
        print(" Unbalanced quotation marks")
        return out
    # Fortran lower-cases the whole line (outside quotes).  On Windows/VMS
    # file names are case-insensitive so that was harmless; on Linux/macOS it
    # is not.  We therefore lower-case only the verb and the qualifier NAMES,
    # and keep parameters and qualifier values as typed.  obey_line() compares
    # keyword parameters (e.g. 'geometry Cyl') in lower case itself.
    first = s.split()[0]
    head = re.split(r"[/\s]", first)[0].lower()
    # ---- verb (abbreviation allowed, must be unique) ---------------------
    matches = [v for v in verbs if v.split()[0].startswith(head)]
    exact = [v for v in matches if v.split()[0] == head]
    if exact:
        matches = exact
    if len(matches) > 1:
        print(" Ambiguous verb", head)
        return out
    if not matches:
        print(" Unrecognised command verb", head)
        return out
    vdef = matches[0]
    out.verb = vdef.split()[0]
    allowed = vdef.split()[1:]
    elements = _split_elements(s, allowed)[1:]          # drop the verb itself

    def qualifier_name(el):
        """'/dens=0.07' -> '/dens' if it abbreviates an allowed qualifier."""
        if not el.startswith("/") or len(el) < 2:
            return None
        name = el.partition("=")[0].lower()
        return name if any(q.startswith(name) for q in allowed) else None

    # ---- qualifiers and parameters ----------------------------------------
    for el in elements:
        name = qualifier_name(el)
        if name:
            val = el.partition("=")[2]
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            cand = [q for q in allowed if q.startswith(name)]
            if len(cand) > 1 and name not in cand:
                print(" Ambiguous qualifier", el)
                return ParsedLine()
            q = name if name in cand else cand[0]
            out.qual.append(q)
            out.qval.append(val)
            out.rqval.append(_to_float(val) if val else 0.0)
        else:
            p = el[1:-1] if (el.startswith('"') and el.endswith('"')) else el
            out.param.append(p)
            # Fortran converts only if length <= 9 characters
            out.rparam.append(_to_float(p) if len(p) <= 9 else 0.0)
    return out
