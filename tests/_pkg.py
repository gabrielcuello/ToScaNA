"""
Make the project importable as `toscana` without installing it.

The project directory is named `ToScaNA-pilot`, while the code imports itself
as `toscana`. This helper registers the package under that name. It also keeps
the project root OFF `sys.path`: the subpackages `math/` and `io/` would
otherwise shadow the standard-library modules of the same name.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.dont_write_bytecode = True
sys.path[:] = [p for p in sys.path if p not in ("", ".") and pathlib.Path(p).resolve() != ROOT]

if "toscana" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "toscana", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["toscana"] = _mod
    _spec.loader.exec_module(_mod)
