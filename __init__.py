"""
ToScaNA: reduction and analysis of neutron total-scattering data from D4 (ILL).

The project directory is the package, imported as ``toscana``. Importing it
imports the subpackages listed below, defines ``__version__`` and installs
the deprecated import names (see `toscana._aliases`).

Subpackages imported by ``import toscana``
------------------------------------------
data
    ASCII x-y-e data containers with plotting.
dataio
    Raw data (numors), detector corrections, regrouping, parameter files,
    output files.
experiment
    Experiment, beam, container, vanadium and sample bookkeeping
    (``Measurement``, ``set*Info``).
isotopes
    Nuclear data table and the ``Element`` class.
models
    Model functions: vanadium, self scattering, cross sections, Ni.
numerics
    Binning, Fourier transforms, windows, fitting, line shapes.
physics
    Unit conversions, geometry, crystallography, densities, cross sections.
plotting
    matplotlib helpers.
utils
    Dates, dictionaries, subprocess, interactive stop.

Not imported automatically (import them explicitly): ``calibration`` (Ni
calibration of wavelength and zero angle), ``chemistry`` (compatibility
re-exports), ``workflows`` (the d4creg regrouping workflow).

Other names
-----------
__version__
    ``"0.1.0"``.
math, io
    Deprecated attribute names for ``numerics`` and ``dataio``: accessing
    ``toscana.math`` returns ``toscana.numerics`` with a
    ``DeprecationWarning`` (module ``__getattr__``). Dotted imports such as
    ``import toscana.math.fourier`` are handled by `toscana._aliases`.

Units used throughout the package: lengths of samples and slits in mm,
wavelength and r in Å, Q in 1/Å, angles in degrees, cross sections in barns,
scattering lengths in fm, densities in g/cm³ or atoms/Å³.
"""

from typing import Any

from . import (
    data,
    dataio,
    experiment,
    isotopes,
    models,
    numerics,
    physics,
    plotting,
    utils,
)
from . import _aliases

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "data",
    "dataio",
    "experiment",
    "isotopes",
    "models",
    "numerics",
    "physics",
    "plotting",
    "utils",
]

# Deprecated names `toscana.math` / `toscana.io` (renamed to avoid clashing
# with the standard library); see _aliases.py.
_aliases.install()


def __getattr__(name: str) -> Any:
    """
    Resolve the deprecated attribute names ``toscana.math`` / ``toscana.io``.

    Module-level ``__getattr__`` (PEP 562), called only for names not found
    in the module.

    Parameters
    ----------
    name : str
        Attribute requested.

    Returns
    -------
    module
        ``toscana.numerics`` for ``"math"``, ``toscana.dataio`` for ``"io"``
        (the already imported module objects), after a
        ``DeprecationWarning`` "`toscana.math` is deprecated; use
        `toscana.numerics` instead.".

    Raises
    ------
    AttributeError
        For any other unknown name.
    """
    if name in _aliases.RENAMED:
        new = _aliases.RENAMED[name]
        _aliases.warn_renamed(f"{__name__}.{name}", f"{__name__}.{new}", stacklevel=3)
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
