"""
Make THIS ToScaNA (the parent folder of ``example``) importable as ``toscana``.

The project folder is registered under the name ``toscana`` without adding
it to ``sys.path``. This takes precedence over any other ``toscana``
installed in the Python environment (e.g. an older version), so the example
always runs with the code it was written for. Import this module before any
``toscana`` import.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

_loaded = sys.modules.get("toscana")
if _loaded is None or pathlib.Path(getattr(_loaded, "__file__", "")).resolve().parent != ROOT:
    for _name in [n for n in sys.modules if n == "toscana" or n.startswith("toscana.")]:
        del sys.modules[_name]
    _spec = importlib.util.spec_from_file_location(
        "toscana", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["toscana"] = _mod
    _spec.loader.exec_module(_mod)
