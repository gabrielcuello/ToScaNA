"""
Helpers to reshape dictionaries of records.

Public names
------------
extractDict
    ``{k: dictionary[k][key]}`` from a dictionary of dictionaries.
extractAttr
    ``{k: getattr(obj, attr)}`` from a dictionary of objects, e.g.
    ``extractAttr(elements, "sig_sca")`` with ``elements`` mapping symbols to
    `toscana.isotopes.Element`.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

T = TypeVar("T")


def extractDict(dictionary: Mapping[str, Mapping[str, T]], key: str) -> dict[str, T]:
    """
    Extract a sub-dictionary from a dictionary of dictionaries (legacy behavior).

    Parameters
    ----------
    dictionary : Mapping[str, Mapping[str, T]]
        Outer dictionary whose values are dictionaries.
    key : str
        Key to extract from every inner dictionary.

    Returns
    -------
    dict
        ``{k: dictionary[k][key] for k in dictionary}``, in the order of
        `dictionary`.

    Raises
    ------
    KeyError
        If an inner dictionary lacks `key`.

    Usage::

        extractDict({"a": {"x": 1}, "b": {"x": 2}}, "x")   # {"a": 1, "b": 2}
    """
    result: dict[str, T] = {}
    for key1, _value in dictionary.items():
        result[key1] = dictionary[key1][key]
    return result


def extractAttr(Dict: Mapping[str, Any], attr: str) -> dict[str, Any]:
    """
    Extract an attribute from a dictionary containing class instances (legacy behavior).

    Preserves legacy printing when an attribute is missing.

    Parameters
    ----------
    Dict : Mapping[str, Any]
        Dictionary of objects, e.g. symbol -> `Element`.
    attr : str
        Attribute name, e.g. ``"weight"`` or ``"sig_sca"``.

    Returns
    -------
    dict
        ``{k: getattr(obj, attr)}`` for every object that has `attr`.
        Objects without it are SKIPPED (their key is absent from the result)
        after printing "Attribute <attr> not found in <obj>." Never raises
        for a missing attribute, so a later lookup by key may raise
        ``KeyError``.
    """
    result: dict[str, Any] = {}
    for key, val in Dict.items():
        if hasattr(val, attr):
            result[key] = getattr(val, attr)
        else:
            print(f"Attribute {attr} not found in {val}.")
    return result
