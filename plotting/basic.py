"""
Decorate the current matplotlib plot for Q, angle or r data.

The functions receive the ``matplotlib.pyplot`` module (or any object with
the same methods, see `PlotLike`) as their first argument, plus a title,
limits and labels, and apply them to the current axes. They do not create,
save or show figures.

Public names
------------
PlotLike
    Protocol of the methods used (``title``, ``xlabel``, ``ylabel``,
    ``axis``, ``grid``, ``legend``).
plotQ
    Abscissa "Q (Å⁻¹)", default range 0-25 1/Å.
plotA
    Abscissa "2θ (°)", default range 0-140 degrees.
plotR
    Free labels, default range 0-20 Å.

Usage::

    import matplotlib.pyplot as plt
    from toscana.plotting import plotQ

    plt.plot(q, sq, label="S(Q)")
    plotQ(plt, title="Sample", xmax=20, ylabel="S(Q)")
    plt.show()
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence


class PlotLike(Protocol):
    """
    Structural type of the plotting object passed to `plotQ`, `plotA`, `plotR`.

    ``matplotlib.pyplot`` satisfies it. Only used for type checking.
    """

    def title(self, label: Optional[str] = ...) -> object: ...

    def xlabel(self, label: Optional[str] = ...) -> object: ...

    def ylabel(self, label: Optional[str] = ...) -> object: ...

    def axis(self, limits: Sequence[Optional[float]]) -> object: ...

    def grid(self, visible: bool = ...) -> object: ...

    def legend(self, loc: str = ...) -> object: ...


def plotQ(
    plt: PlotLike,
    title: Optional[str] = None,
    xmin: float = 0,
    xmax: float = 25,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    ylabel: str = "Intensity (arb. units)",
) -> None:
    """
    Decorate the current plot for a Q abscissa.

    Calls, in order: ``plt.title(title)``, ``plt.xlabel`` with the
    mathtext label "Q (Å⁻¹)", ``plt.ylabel(ylabel)``,
    ``plt.axis([xmin, xmax, ymin, ymax])``, ``plt.grid(True)`` and
    ``plt.legend(loc="best")``.

    Parameters
    ----------
    plt : PlotLike
        Usually the ``matplotlib.pyplot`` module.
    title : str or None, optional
        Plot title; None (default) gives an empty title.
    xmin, xmax : float, optional
        Abscissa limits, 1/Å. Defaults 0 and 25.
    ymin, ymax : float or None, optional
        Ordinate limits; None (default) keeps the automatic limit.
    ylabel : str, optional
        Ordinate label. Default "Intensity (arb. units)".

    Notes
    -----
    ``legend`` warns (matplotlib "No artists with labels found") if no
    curve has a label.
    """
    plt.title(title)
    plt.xlabel(r"$Q ({\rm \AA}^{-1}$)")
    plt.ylabel(ylabel)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True)
    plt.legend(loc="best")


def plotA(
    plt: PlotLike,
    title: Optional[str] = None,
    xmin: float = 0,
    xmax: float = 140,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    ylabel: str = "Intensity (arb. units)",
) -> None:
    """
    Decorate the current plot for a scattering-angle abscissa.

    Same as `plotQ` but with the abscissa label "2θ (°)" (mathtext
    ``$2\\theta$ ($^o$)``).

    Parameters
    ----------
    plt : PlotLike
        Usually the ``matplotlib.pyplot`` module.
    title : str or None, optional
        Plot title; None (default) gives an empty title.
    xmin, xmax : float, optional
        Abscissa limits, degrees. Defaults 0 and 140.
    ymin, ymax : float or None, optional
        Ordinate limits; None (default) keeps the automatic limit.
    ylabel : str, optional
        Ordinate label. Default "Intensity (arb. units)".
    """
    plt.title(title)
    plt.xlabel(r"$2\theta$ ($^o$)")
    plt.ylabel(ylabel)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True)
    plt.legend(loc="best")


def plotR(
    plt: PlotLike,
    title: Optional[str] = None,
    xmin: float = 0,
    xmax: float = 20,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Decorate the current plot for a real-space (r) abscissa.

    Same sequence of calls as `plotQ`, but both axis labels are given by
    the caller (no default r label).

    Parameters
    ----------
    plt : PlotLike
        Usually the ``matplotlib.pyplot`` module.
    title : str or None, optional
        Plot title; None (default) gives an empty title.
    xmin, xmax : float, optional
        Abscissa limits, Å. Defaults 0 and 20.
    ymin, ymax : float or None, optional
        Ordinate limits; None (default) keeps the automatic limit.
    xlabel, ylabel : str or None, optional
        Axis labels; None (default) gives empty labels.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True)
    plt.legend(loc="best")
