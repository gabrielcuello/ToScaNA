"""
Simple sample geometry.

Public functions
----------------
getCylVolume(diameter=5.0, height=50.0)
    Volume of a solid cylinder, V = π (d/2)² h.

Conventions and units
---------------------
No unit conversion is done: the volume is in the cube of the length unit of
the inputs (mm³ for diameters and heights in mm, the usual case in ToScaNA;
divide by 1000 to obtain cm³). For annular samples (hollow cylinders) see
``toscana.experiment`` (``Measurement``), which uses π h (r_o² - r_i²).
"""

from __future__ import annotations

import numpy as np


def getCylVolume(diameter: float = 5.0, height: float = 50.0) -> float:
    """
    Calculate the volume of a cylinder.

    V = π (diameter/2)² · height, for a solid (full) cylinder.

    Parameters
    ----------
    diameter : float, optional
        The diameter of the cylinder. Default is 5.0 (mm in ToScaNA usage).
    height : float, optional
        The height of the cylinder. Default is 50.0 (mm in ToScaNA usage).

    Returns
    -------
    float
        The volume of the cylinder, in the cube of the input length unit
        (mm³ for mm input). No validation: negative diameters give a positive
        volume (squared), negative heights a negative volume.

    Examples
    --------
    >>> round(getCylVolume(5.0, 50.0), 4)
    981.7477
    """
    radius = diameter / 2.0
    return float(np.pi * radius**2 * height)
