"""
ASCII x-y(-e) data containers with plotting.

Public names
------------
DataXYE
    2- or 3-column file with header lines; statistics, automatic peak search
    in x ∈ [5, 40], ``plot`` (optionally saved to ``../regdata``), ``show``,
    ``header``.
DataXYE_old
    Legacy name; subclass of `DataXYE` without changes.
DataXYE0
    3-column file read with ``np.loadtxt``; ``plot``, ``show``, ``save``.
DataXYE2
    Numeric table read with ``np.loadtxt``; ``plot`` with axis type and
    overlay option.

See `toscana.data.xy` for details.
"""

from .xy import DataXYE, DataXYE0, DataXYE2, DataXYE_old

__all__ = ["DataXYE", "DataXYE0", "DataXYE2", "DataXYE_old"]
