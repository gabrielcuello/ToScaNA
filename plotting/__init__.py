"""
matplotlib helpers.

Public names
------------
plotQ, plotA, plotR
    Set title, axis labels, limits, grid and legend of the current plot for
    a Q (1/Å), angle 2θ (degrees) or r (Å) abscissa (`toscana.plotting.basic`).
defColors
    The fixed list of 20 CSS4 colours used by the d4creg plots.
makePlotsA, makePlotsQ
    Diagnostic plots of the d4creg regrouping workflow, in angle and in Q
    (`toscana.plotting.workflows`).

matplotlib is imported only inside the functions that need it.
"""

from .basic import plotA, plotQ, plotR
from .workflows import defColors, makePlotsA, makePlotsQ

__all__ = ["defColors", "makePlotsA", "makePlotsQ", "plotA", "plotQ", "plotR"]
