"""
sample_factory.py - MCAMS: build a sample of the chosen geometry ("slab",
"cylinder", "disc" (version 2.3) or "sphere" (version 2.4)).

Why a "factory"?
----------------
The simulations (transmission_vs_angle.py, multiple_scattering.py) work with
any object that follows the `LayeredSample` interface (sample_base.py). A
factory function creates the right kind of object from a simple text choice,
so a script only has to change ONE word to switch from one geometry to the
other:

    sample = make_sample("slab",     layers, height=20, width=20)
    sample = make_sample("cylinder", layers, height=20)
    sample = make_sample("disc",     layers, diameter=20)      # version 2.3
    sample = make_sample("sphere",   layers)                   # version 2.4

How the `layers` are read depends on the geometry (see the modules):
    slab     : from the FRONT face to the back; thickness along the normal.
    disc     : as the slab (the disc is a slab with a circular section).
    cylinder : from the INNER cylinder outwards; the thickness of the first
               layer is its DIAMETER, the others are radial thicknesses.
    sphere   : as the cylinder (inner sphere: DIAMETER; shells: radial).
Each layer is a tuple (name, thickness_mm, mu_per_mm[, scatter_fraction[,
fill_height_mm]]), a dict with the fields of `Layer` or a `Layer`.

Vertical extent of the layers (version 2.2): each layer can have its own
`bottom` and `height` (mm, lab y measured from the centre of the beam); its
material lies between bottom and bottom + height, and the layer is empty
below and above. The sample height may then be None (envelope of the
layers). The fullness (fraction of the beam height covered by each layer) is
given by sample.beam_fullness() when the sample knows `beam_height`:

    sample = make_sample("cylinder",
                         [{"name": "powder", "thickness": 6, "mu": 0.25,
                           "bottom": -10, "height": 8},
                          {"name": "V", "thickness": 0.2, "mu": 0.04,
                           "bottom": -10, "height": 30}],
                         height=None, beam_height=8)
    sample.beam_fullness()      # -> [0.25, 1.0]: the powder reaches y = -2

The forms of version 2.0 (fill_height measured from the bottom of the
sample, or fullness measured from the bottom of the beam) are still
accepted.
"""

from .slab import create_slab_sample
from .cylinder import create_cylindrical_sample
from .disc import create_disc_sample
from .sphere import create_spherical_sample

# Names accepted for each geometry (lower case).
SLAB_NAMES = ("slab", "plate", "planar", "flat")
CYLINDER_NAMES = ("cylinder", "cylindrical", "cyl")
DISC_NAMES = ("disc", "disk", "circular", "circular plate", "round plate")
SPHERE_NAMES = ("sphere", "spherical", "ball")


def make_sample(geometry, layers, height=None, width=None, plane_angle_deg=90.0,
                y_bottom=None, beam_height=None, diameter=None):
    """Create a sample of the requested geometry.

    Parameters
    ----------
    geometry : str
        "slab", "cylinder", "disc" or "sphere".
    layers : list
        Layers: tuples (name, thickness_mm, mu_per_mm[, scatter_fraction[,
        fill_height_mm]]), dicts with the fields of `Layer`, or `Layer`s.
        Use dicts or Layers to give `bottom` and `height` (version 2.2).
    height : float or None
        Vertical size (mm): the height of the slab, or of the cylinders.
        None: the envelope of the layers given with bottom and height.
    width : float, optional
        Horizontal size (mm) of the slab. Required for "slab", ignored for
        "cylinder" (its width is set by its diameter).
    plane_angle_deg : float
        Slab and disc: angle between the beam and the plane of the sample (90
        = perpendicular). Ignored for "cylinder" (a vertical cylinder looks
        the same from every azimuth).
    y_bottom : float, optional
        Height (mm, lab y, 0 = beam centre) of the bottom of the sample;
        layers without their own `bottom` start here. Default None: sample
        centred on the beam (y_bottom = -height/2), or the lowest layer
        bottom when height is None.
    beam_height : float, optional
        Height of the beam (mm). Needed when a layer is given by its
        fullness; also shown in the summary.
    diameter : float, optional
        Disc only (required): diameter of the disc (mm). The disc is centred
        on the beam, so `height`, `width` and `y_bottom` are not used for it.
        The sphere (centred on the beam too) needs none of them: its size
        comes from the thicknesses of its layers, and plane_angle_deg is
        ignored (a sphere looks the same from every direction).

    Returns
    -------
    A `SlabSample` (slab), a `CylindricalSample` (cylinder), a
    `DiscSample` (disc) or a `SphericalSample` (sphere); all offer the same
    methods.
    """
    # .strip() removes spaces and .lower() converts to lower case, so that
    # " Cylinder " is accepted as well.
    name = geometry.strip().lower()

    if name in SLAB_NAMES:
        if width is None:
            raise ValueError("The slab geometry needs a `width` (mm).")
        return create_slab_sample(width, height, layers,
                                  plane_angle_deg=plane_angle_deg,
                                  y_bottom=y_bottom, beam_height=beam_height)

    if name in CYLINDER_NAMES:
        return create_cylindrical_sample(height, layers, y_bottom=y_bottom,
                                         beam_height=beam_height)

    if name in DISC_NAMES:
        if diameter is None:
            raise ValueError("The disc geometry needs a `diameter` (mm).")
        return create_disc_sample(diameter, layers,
                                  plane_angle_deg=plane_angle_deg,
                                  beam_height=beam_height)

    if name in SPHERE_NAMES:
        return create_spherical_sample(layers, beam_height=beam_height)

    raise ValueError(f"Unknown geometry '{geometry}'. "
                     "Use 'slab', 'cylinder', 'disc' or 'sphere'.")
