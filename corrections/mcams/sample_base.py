"""
sample_base.py - Ingredients shared by ALL sample geometries of MCAMS
(Monte Carlo Attenuation and Multiple Scattering).

Why does this file exist?
-------------------------
We have two kinds of sample: a stack of flat slabs (slab.py) and a set of
concentric cylinders (cylinder.py). The geometry is different, but the
programs that use the sample (transmission_vs_angle.py,
multiple_scattering.py, transmission_by_layer.py) must not care about it. To
make that possible, both samples follow the same "contract", written down
once here:

    * `Layer`          : the data of one layer (label, thickness, mu, ...,
                         and how high it is filled).
    * `LayeredSample`  : an abstract base class listing the methods that
                         every sample geometry must offer, plus the code that
                         is the same for all geometries (vertical extent of
                         the sample and filling of the layers).
    * `scattered_direction` : direction of a particle for a given angle 2theta.

Frames of reference
-------------------
LAB frame (the same for everybody):
    z = direction of the incident beam, x = horizontal, y = vertical.
    The origin is the centre of rotation of the instrument (the centre of
    the circle on which the detector moves). The beam is centred on y = 0.
SAMPLE frame:
    A frame attached to the sample, in which its geometry is simplest.
    Each geometry decides how it relates to the lab frame (a tilted slab
    needs a rotation around the VERTICAL axis; a vertical cylinder does not,
    its sample frame is the lab frame). In both cases the vertical
    coordinate is the same in the two frames: y' = y.

Vertical extent of the layers (version 2.2)
-------------------------------------------
A layer does not need to fill the whole height of the sample. Each layer i
contains material only between two heights (lab y, mm):

        bottom_i <= y <= top_i ,      top_i = bottom_i + height_i

and is EMPTY (no attenuation, no interaction) below and above. Heights are
measured from the CENTRE OF THE BEAM (y = 0; negative values are below it).
This describes, for instance,
  * a container partly filled with a powder or a liquid (material at the
    bottom of the container, empty headspace above it);
  * a sample whose bottom is not at the bottom of the beam (the beam may hit
    only its upper part, or pass below it);
  * containers with walls of different heights, inner tubes resting on
    different supports, ...
Particles can fly through the empty parts, so part of the beam can pass
below or above the material, and scattered particles can cross the empty
regions before entering another layer or leaving.

How to give the vertical extent of a layer (see `Layer`):
  * bottom, height : the RECOMMENDED form (version 2.2): level of the bottom
                     of the material and its height, in mm. Either may be
                     omitted: without `bottom` the material starts at the
                     bottom of the sample, without `height` it goes up to the
                     top of the sample.
  * fill_height    : (version 2.0) height of the material above the bottom
                     of the sample: the same as bottom = sample bottom,
                     height = fill_height.
  * fullness       : (version 2.0) fraction of the BEAM HEIGHT covered by
                     the material, starting at the bottom of the sample and
                     measured from the bottom of the beam.
Without any of them the layer fills the whole height of the sample, exactly
as in version 1.

The FULLNESS of every layer is always available as a result:
sample.beam_fullness() = (part of the beam height covered by the material) /
(beam height). It is not used as an input in the recommended form because,
once the bottom is free, one number cannot fix the geometry (f = 0.5 may be
the lower half of the beam, the upper half or a band in the middle) and it
cannot describe the material outside the beam, which matters for multiple
scattering.

The SAMPLE itself occupies y_bottom <= y <= y_top (the "box" of a slab, or
the length of the cylinders): it is given by `height` and `y_bottom` when
the sample is created, or, if `height` is None, it is the smallest range
that contains the material of all the layers given with bottom and height.

Units: all distances in mm, linear attenuation coefficients in 1/mm.

NumPy functions used in this file
---------------------------------
np.array(list)        : build a NumPy array (a fast table of numbers) from a list.
np.exp(a)             : e**a for every element of the array a.
np.sin, np.cos        : sine and cosine of every element (angles in radians).
np.radians(a)         : convert degrees to radians.
np.clip(a, lo, hi)    : limit the values of a to the interval [lo, hi].
np.unique(a)          : the different values of a, sorted.
a @ b                 : matrix product (for 2D arrays, rows times columns).
"""

# `abc` (Abstract Base Classes) is a module of the Python standard library.
# A class that inherits from ABC and has methods decorated with
# @abstractmethod cannot be used directly: it only defines WHAT methods a
# family of classes must have. Every subclass has to implement them.
from abc import ABC, abstractmethod

# `dataclass` writes the boring parts of a "record" class for us
# (__init__, __repr__, ...) from the list of fields.
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Small length (mm) used to forgive floating-point rounding errors when a
# point lies exactly on a surface (a face, a layer boundary, the level of
# the material in a partially filled layer).
GEOMETRY_TOL = 1e-7


# --------------------------------------------------------------------------
# One layer: just a small record of data
# --------------------------------------------------------------------------
@dataclass(frozen=True)          # frozen=True -> the values cannot be changed later
class Layer:
    """One layer of the sample.

    name      : label of the material (free text, e.g. "Fe").
    thickness : size of the layer (mm), must be > 0. What "size" means
                depends on the geometry:
                  * slab     : thickness along the beam-side normal of the layer.
                  * cylinder : radial thickness of the annulus, EXCEPT for the
                               innermost layer (a full cylinder), for which
                               it is the DIAMETER.
    mu        : TOTAL linear attenuation coefficient (1/mm), must be >= 0
                (scattering + absorption). Use 0 for an air/vacuum gap.
    scatter_fraction : fraction of the interactions in this layer that are
                scattering events (the rest is absorption), between 0 and 1.
                Default 1.0 (pure scatterer). It only matters for multiple
                scattering: an absorbed particle is lost.
    fill_height : height of the material (mm) measured from the BOTTOM of the
                sample, >= 0. None (default) = the layer is completely full.
                A value equal to or larger than the sample height also means
                completely full. Above this level the layer is empty.
    fullness  : alternative to fill_height: fraction of the BEAM HEIGHT that
                the material covers, between 0 and 1 (1 = covers the whole
                beam: the layer is then taken as completely full). The
                sample converts it into a fill height, so the sample must be
                created with its `beam_height`. Give fill_height OR fullness,
                not both.
    bottom    : (version 2.2, recommended) lab height y (mm) of the BOTTOM of
                the material, measured from the CENTRE OF THE BEAM (negative
                below it). None = the bottom of the sample.
    height    : (version 2.2, recommended) height of the material (mm), >= 0;
                top = bottom + height. None = up to the top of the sample.
                bottom/height cannot be combined with fill_height/fullness.

    Examples
    --------
        Layer("Fe", 1.0, 0.25)                        # full layer
        Layer("powder", 6.0, 0.25, 0.8, bottom=-6, height=9)
                                                      # material from y=-6 to y=+3 mm
        Layer("water", 3.0, 0.56, 1.0, fill_height=6) # 6 mm above the sample bottom
        Layer("water", 3.0, 0.56, fullness=0.5)       # covers the lower half of the beam
    In a list of layers a tuple is accepted as well; its 5th element is the
    fill height:  ("water", 3.0, 0.56, 1.0, 6.0); use a Layer or a dict
    ({"name": ..., "bottom": ..., "height": ...}) to give bottom and height.
    """
    name: str
    thickness: float
    mu: float
    scatter_fraction: float = 1.0
    fill_height: Optional[float] = None
    fullness: Optional[float] = None
    bottom: Optional[float] = None
    height: Optional[float] = None

    def __post_init__(self):
        # Called automatically right after the record is created: we use it
        # to refuse physically meaningless values early, with a clear message.
        if self.thickness <= 0:
            raise ValueError(f"Layer '{self.name}': thickness must be > 0 mm.")
        if self.mu < 0:
            raise ValueError(f"Layer '{self.name}': mu must be >= 0 (1/mm).")
        if not 0.0 <= self.scatter_fraction <= 1.0:
            raise ValueError(f"Layer '{self.name}': scatter_fraction must "
                             "be between 0 and 1.")
        if self.fill_height is not None and self.fullness is not None:
            raise ValueError(f"Layer '{self.name}': give fill_height OR "
                             "fullness, not both.")
        if self.fill_height is not None and self.fill_height < 0:
            raise ValueError(f"Layer '{self.name}': fill_height must be >= 0 mm.")
        if self.fullness is not None and not 0.0 <= self.fullness <= 1.0:
            raise ValueError(f"Layer '{self.name}': fullness must be "
                             "between 0 and 1.")
        if self.height is not None and self.height < 0:
            raise ValueError(f"Layer '{self.name}': height must be >= 0 mm.")
        if ((self.bottom is not None or self.height is not None)
                and (self.fill_height is not None or self.fullness is not None)):
            raise ValueError(f"Layer '{self.name}': give bottom/height OR "
                             "fill_height/fullness, not both.")

    @property
    def is_partial(self):
        """True if the layer was declared with a limited vertical extent
        (bottom, height, fill_height, or fullness < 1). Whether it really is
        partial also depends on the sample (see LayeredSample.partial)."""
        if self.fullness is not None:
            return self.fullness < 1.0
        return (self.fill_height is not None or self.bottom is not None
                or self.height is not None)


def make_layer(item):
    """Turn one item of a list of layers into a `Layer`.

    Accepted items:
      * a `Layer` (returned as it is);
      * a tuple (name, thickness, mu[, scatter_fraction[, fill_height]]);
      * a dict with the field names of Layer, e.g.
            {"name": "water", "thickness": 3, "mu": 0.56, "fullness": 0.5}
    """
    if isinstance(item, Layer):
        return item
    if isinstance(item, dict):
        return Layer(**item)          # ** unpacks the dict into keyword arguments
    return Layer(*item)               # *  unpacks the tuple into positional arguments


# --------------------------------------------------------------------------
# Direction of a scattered particle
# --------------------------------------------------------------------------
def scattered_direction(two_theta_deg, phi_deg=0.0):
    """Unit vector (LAB frame) of a particle scattered by `two_theta_deg`
    from the beam direction +z.

    two_theta_deg : angle between the scattered direction and +z (degrees)
    phi_deg       : azimuth around the beam, measured from +x towards +y
                    (default 0: detector in the horizontal plane, at +x).

    These are ordinary spherical coordinates whose polar axis is the beam:
        direction = (sin(2theta) cos(phi), sin(2theta) sin(phi), cos(2theta))
    Note that in a 2-axis instrument the detector moves in the horizontal
    plane, which is the plane y = 0, i.e. phi = 0 (or 180 degrees).

    For methods that work in the sample frame, convert the result first with
    `sample.to_sample_frame(...)`.
    """
    # np.radians converts degrees to radians, which np.sin / np.cos expect.
    t, f = np.radians(two_theta_deg), np.radians(phi_deg)
    return np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])


# --------------------------------------------------------------------------
# The common contract of all sample geometries
# --------------------------------------------------------------------------
class LayeredSample(ABC):
    """Abstract base class of the layered samples (slab, cylinder, ...).

    Every geometry stores the same layer data (thickness, mu, scatter
    fraction, filling) and the same vertical extent, and must implement the
    geometry-dependent methods marked @abstractmethod below. The
    simulations only use this interface, so they work for any geometry.
    """

    def __init__(self, layers, height=None, y_bottom=None, beam_height=None):
        """Store the layer data and the vertical extent of the sample.

        layers      : sequence of `Layer` (or tuples / dicts, see
                      `make_layer`), in the order defined by the geometry
                      (slab: from the front face to the back; cylinder: from
                      the axis outwards).
        height      : vertical size of the sample (mm), > 0. None (new in
                      2.2): the smallest range that contains the material of
                      the layers given with `bottom` and `height` (at least
                      one layer must have both).
        y_bottom    : height (lab y, mm) of the bottom of the sample. Default
                      None: centred on the beam (y_bottom = -height/2), or,
                      when `height` is None, the lowest bottom of the layers.
        beam_height : height of the beam (mm). Needed if a layer is given by
                      its `fullness`; also used by `summary` and
                      `beam_fullness` to give the fraction of the beam
                      covered by every layer.
        """
        if len(layers) == 0:
            raise ValueError("A sample needs at least one layer.")
        self.layers = tuple(make_layer(l) for l in layers)

        # One entry per layer in NumPy arrays: they let us compute with all
        # the layers at once (vectorised), with no Python loops.
        self.thicknesses = np.array([l.thickness for l in self.layers])
        self.mus = np.array([l.mu for l in self.layers])
        self.scatter_fractions = np.array(
            [l.scatter_fraction for l in self.layers])
        self.beam_height = None if beam_height is None else float(beam_height)
        if self.beam_height is not None and self.beam_height <= 0:
            raise ValueError("beam_height must be positive (mm).")

        # --- Vertical extent of the sample --------------------------------
        lo = None
        if height is None:
            # Envelope of the layers that give their full vertical extent.
            ext = [(l.bottom, l.bottom + l.height) for l in self.layers
                   if l.bottom is not None and l.height is not None]
            if not ext:
                raise ValueError("Give the sample height, or the bottom AND "
                                 "height of at least one layer.")
            lo = min(e[0] for e in ext) if y_bottom is None else float(y_bottom)
            hi = max(e[1] for e in ext)
            height = hi - lo
        if height <= 0:
            raise ValueError("The sample height must be positive (mm).")
        self.height = float(height)
        if y_bottom is None:
            y_bottom = lo if lo is not None else -self.height / 2
        self.y_bottom = float(y_bottom)
        self.y_top = self.y_bottom + self.height

        # --- Where is the material of each layer? ---------------------------
        # fill_bottoms[i] <= y <= fill_tops[i]  (lab = sample frame y, mm).
        # For a full layer they are the bottom and the top of the sample.
        levels = [self._material_range_of(l) for l in self.layers]
        self.fill_bottoms = np.array([b for b, t in levels])
        self.fill_tops = np.array([t for b, t in levels])
        # fill_heights[i] = height of the material of layer i (mm).
        self.fill_heights = np.maximum(self.fill_tops - self.fill_bottoms, 0.0)
        # A layer whose material lies completely outside the sample (or has
        # height 0) contains nothing: it is marked as empty.
        self.empty = self.fill_heights <= GEOMETRY_TOL
        # partial[i] is True when there is empty space below or above the
        # material of layer i.
        self.partial = (self.empty
                        | (self.fill_bottoms > self.y_bottom + GEOMETRY_TOL)
                        | (self.fill_tops < self.y_top - GEOMETRY_TOL))
        # The different levels (bottoms and tops of material) that lie
        # strictly inside the sample: they are the extra horizontal surfaces
        # that a particle may cross (np.unique also sorts them).
        inner = np.concatenate([self.fill_bottoms[~self.empty], self.fill_tops[~self.empty]])
        inner = inner[(inner > self.y_bottom + GEOMETRY_TOL) & (inner < self.y_top - GEOMETRY_TOL)]
        self.fill_levels = np.unique(inner)

    # ---- the vertical extent of the material of the layers ------------------
    def _material_range_of(self, layer):
        """(bottom, top) heights y (mm) of the material of one layer, limited
        to the sample [y_bottom, y_top]."""
        lo, hi = self.y_bottom, self.y_top
        if layer.bottom is not None or layer.height is not None:
            # Version 2.2: bottom and height, both relative to the beam centre.
            b = lo if layer.bottom is None else float(layer.bottom)
            t = hi if layer.height is None else b + float(layer.height)
        elif layer.fullness is not None:
            # Version 2.0: material from the bottom of the sample up to
            # beam_bottom + fullness * h_beam (the beam is centred on y = 0).
            if layer.fullness >= 1.0:
                return lo, hi                         # covers the beam: full
            if self.beam_height is None:
                raise ValueError(
                    f"Layer '{layer.name}' is given by its fullness: create "
                    "the sample with beam_height=... (mm) to convert it.")
            b = lo
            t = -self.beam_height / 2 + layer.fullness * self.beam_height
        elif layer.fill_height is not None:
            b, t = lo, lo + layer.fill_height         # version 2.0
        else:
            return lo, hi                             # nothing said: full
        # The material cannot be outside the sample.
        b = float(np.clip(b, lo, hi))
        t = float(np.clip(t, lo, hi))
        return b, max(b, t)

    def beam_fullness(self, beam_height=None):
        """Fraction of the beam height covered by the material of each layer.

        For a beam of height h_b centred on y = 0 (from -h_b/2 to +h_b/2) and
        a layer with material between y_b (bottom) and y_t (top):

            fullness = max(0, min(y_t, h_b/2) - max(y_b, -h_b/2)) / h_b

        i.e. the length of the overlap of the two intervals divided by the
        beam height: 1 if the material covers the whole beam, 0 if the beam
        passes completely above or below it. Returns one value per layer.
        """
        hb = self.beam_height if beam_height is None else float(beam_height)
        if hb is None:
            raise ValueError("Give the beam height (mm).")
        overlap = (np.minimum(self.fill_tops, hb / 2)
                   - np.maximum(self.fill_bottoms, -hb / 2))
        f = np.clip(overlap / hb, 0.0, 1.0)
        return np.where(self.empty, 0.0, f)

    @property
    def has_partial_layers(self):
        """True if at least one layer does not fill the whole sample height."""
        return bool(self.partial.any())

    def is_filled(self, layer_index, y, tol=GEOMETRY_TOL):
        """True where the height y (mm) is inside the MATERIAL of the given
        layer(s), i.e. between its bottom and its top. Both arguments may be
        arrays of the same shape."""
        y = np.asarray(y)
        return ((y >= self.fill_bottoms[layer_index] - tol)
                & (y <= self.fill_tops[layer_index] + tol)
                & ~self.empty[layer_index])

    def _fill_text(self, i):
        """Short text with the vertical extent of layer i, used by `summary`:
        "full", or "bottom to top" (mm) and, when the beam height is known,
        the fullness in parentheses."""
        if not self.partial[i]:
            return "full"
        if self.empty[i]:
            return "empty"
        text = f"{self.fill_bottoms[i]:.3f} to {self.fill_tops[i]:.3f}"
        if self.beam_height is not None:
            text += f" ({self.beam_fullness()[i]:.2f})"
        return text

    # ---- information shared by all geometries ---------------------------
    @property                       # a property is used like an attribute: sample.n_layers
    def n_layers(self):
        """Number of layers."""
        return len(self.layers)

    def transmission(self, path_lengths):
        """Fraction of particles surviving the given paths: exp(-sum mu_i*L_i).

        This is the Beer-Lambert law. `path_lengths` has shape
        (N particles, n_layers): row j holds the distance travelled by
        particle j inside the MATERIAL of each layer (mm). (The paths in the
        empty part of partially filled layers are not included: they do not
        attenuate.)
        """
        # np.atleast_2d makes sure we have a 2D table even if one particle
        # (a 1D array) was given.
        # `@ self.mus` multiplies each row by the vector of mu's and adds up:
        # for every particle it gives sum_i L_i * mu_i (the optical depth).
        # np.exp then turns the optical depth into a survival probability.
        return np.exp(-np.atleast_2d(path_lengths) @ self.mus)

    # ---- geometry-dependent methods: every subclass must write them -----
    @abstractmethod
    def summary(self):
        """Return a text table describing the sample."""

    @abstractmethod
    def to_sample_frame(self, vectors):
        """Convert vectors (directions) from the lab frame to the sample
        frame. Input (3,) or (N, 3); output (N, 3)."""

    @abstractmethod
    def to_lab_frame(self, vectors):
        """Inverse of `to_sample_frame`."""

    @abstractmethod
    def contains(self, points, tol=1e-9):
        """True for each point (SAMPLE frame, shape (N, 3)) inside the
        volume of the sample (including the empty part of partially filled
        layers)."""

    @abstractmethod
    def layer_of_points(self, points):
        """Index of the layer whose MATERIAL contains each point (SAMPLE
        frame, shape (N, 3)); -1 for points outside the sample or in the
        empty part above a partially filled layer."""

    @abstractmethod
    def outgoing_path_lengths(self, points, directions):
        """Distance travelled in the material of each layer by particles
        that start at `points` and fly along `directions` (both in the
        SAMPLE frame) until they leave the sample. Returns an array
        (N, n_layers) in mm."""

    @abstractmethod
    def next_interaction(self, points, directions, rng=None):
        """For particles flying from `points` along `directions` (SAMPLE
        frame, inside the sample): where is their next interaction?
        Returns (new_points (N, 3), interacts (N,) boolean). Where
        `interacts` is False the particle leaves the sample instead."""

    @abstractmethod
    def sample_interaction_points(self, x, y, rng=None):
        """First interaction of beam particles.

        The particles travel along +z in the lab frame; (x, y) is their
        position in the plane perpendicular to the beam that contains the
        origin (these are the numbers given by beam.sample_beam_point).

        Returns (points (M, 3) in the SAMPLE frame, interacted (N,) boolean).
        """
