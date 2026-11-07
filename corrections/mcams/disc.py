"""
disc.py - DISC geometry of MCAMS (version 2.3): a flat sample with a
CIRCULAR section, made of layers stacked along its normal, centred on the
beam. Like the slab, it can be rotated around the vertical axis.

(The rectangular slab is in slab.py and the concentric cylinders in
cylinder.py; all three follow the common interface of sample_base.py.)

What the sample looks like
--------------------------
Think of a coin, or of a stack of coins of different materials glued
together: a circular plate of diameter D whose layers are thin discs, one
behind the other along the normal of the plate. For example a flat circular
cell (window / liquid / window), a pressed pellet, or a powder between two
circular windows.

    * every layer is a disc of the SAME diameter D;
    * the centre of the discs is on the beam axis (x = y = 0 of the lab);
    * by default the plane of the disc is perpendicular to the beam; it can be
      rotated around the vertical axis exactly as the slab
      (`plane_angle_deg`, see slab.py);
    * each layer has the same properties as a layer of the slab or of the
      cylinder: label, thickness, mu, scatter fraction and vertical extent of
      its material (bottom and height, version 2.2). A disc-shaped cell half
      filled with a liquid is therefore described exactly as a slab cell.

Conventions
-----------
* All distances in mm, linear attenuation coefficients in 1/mm.
* LAB frame: the beam travels along +z, x is horizontal, y is vertical; the
  beam is centred on y = 0.
* SAMPLE frame: the same as for the slab. z' is the normal of the layers,
  x' (horizontal) and y' (vertical, y' = y) lie in the plane of the layers.
  Layer i spans boundaries[i] <= z' <= boundaries[i+1]; the front face is at
  z' = z_front (default 0). The section is the circle
          x'^2 + y'^2 <= R^2 ,      R = D / 2,
  so the sample goes from y = -R to y = +R (y_bottom = -R, height = D).
* ORIENTATION: `plane_angle_deg` = angle between the beam and the plane of
  the disc (90 = perpendicular, the default). The rotation axis is the
  vertical line through the centre of the front face (on the beam axis).

How the paths are computed
--------------------------
Exactly as for the slab (read the docstring of slab.py first): the material
of layer i is the intersection of three regions, and along a straight line
p + t*d each of them is an interval of t:
    * the SECTION of the sample: the infinite cylinder x'^2 + y'^2 <= R^2,
      whose axis is the normal z' of the disc, cut by the front and back
      faces (z_front <= z' <= z_back);
    * the slice of the layer (boundaries[i] <= z' <= boundaries[i+1]);
    * the band of heights of its material (bottom_i <= y' <= top_i).
The only difference with the slab is the first region: a circle instead of a
rectangle. A cylinder is convex, so a straight line enters it at most once
and leaves it at most once: its intersection with the line is ONE interval,
obtained by solving a quadratic equation (see `_cylinder_interval`). The
rest of the calculation (sorting the intervals, accumulating the optical
depth, finding the next interaction) is inherited from the slab without any
change: `DiscSample` is a `SlabSample` whose section is round.

Typical use
-----------
    from disc import create_disc_sample
    from sample_base import Layer
    sample = create_disc_sample(
        diameter=20,
        layers=[("Al", 1.0, 0.010),                       # front window
                Layer("water", 2.0, 0.56, 1.0, bottom=-10, height=12),  # up to y = +2
                ("Al", 1.0, 0.010)],                      # back window
        plane_angle_deg=90, beam_height=8)
    print(sample.summary())

NumPy functions used in this file
---------------------------------
np.sqrt(a)            : square root of every element.
np.maximum, np.minimum: element by element larger / smaller of two arrays.
np.where(cond, a, b)  : element by element, `a` where cond is True, else `b`.
np.hypot(x, y)        : sqrt(x**2 + y**2) element by element (no overflow).
np.atleast_2d(a)      : make one point (3,) into a table (1, 3).
np.clip(a, lo, hi)    : limit the values of a to [lo, hi].
np.linspace(a, b, n)  : n equally spaced numbers from a to b.
np.trapezoid(y, x)    : integral of the curve y(x) by the trapezoidal rule
                        (called np.trapz in NumPy < 2.0).
np.errstate(...)      : silence warnings (divisions handled with np.where).
"""

import numpy as np

from .sample_base import (Layer, scattered_direction,          # noqa: F401
                         make_layer, GEOMETRY_TOL)
from .slab import SlabSample, _slice_interval

# np.trapezoid is the NumPy >= 2.0 name of np.trapz.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# A helper: when is a ray inside an infinite cylinder?
# --------------------------------------------------------------------------
def _cylinder_interval(px, py, dx, dy, radius):
    """Distances t at which rays p + t*d are inside the infinite cylinder
    x^2 + y^2 <= radius^2 (axis along z). Only the x and y components of the
    points and directions matter.

    px, py, dx, dy : (N,) arrays.

    The ray is on the surface when (px + t dx)^2 + (py + t dy)^2 = R^2, i.e.

        a t^2 + 2 b t + c = 0   with   a = dx^2 + dy^2,
                                       b = px dx + py dy,
                                       c = px^2 + py^2 - R^2 .

    Its two roots t = (-b -+ sqrt(b^2 - a c)) / a are where the ray enters
    and leaves the cylinder. Special cases:
      * b^2 - a c < 0 : the line misses the cylinder -> empty interval;
      * a = 0         : the ray is parallel to the axis (moves only along z):
                        inside for ever if c <= 0, never otherwise.

    Returns (t_lo, t_hi); the empty interval is coded as (inf, -inf), "for
    ever" as (-inf, inf), the same conventions as `_slice_interval`.
    """
    a = dx * dx + dy * dy
    b = px * dx + py * dy
    c = px * px + py * py - radius * radius
    along_axis = a < 1e-24                    # direction (practically) parallel to z
    safe_a = np.where(along_axis, 1.0, a)     # avoid dividing by zero
    disc = b * b - a * c                      # the discriminant (divided by 4)
    meets = disc >= 0.0
    root = np.sqrt(np.where(meets, disc, 0.0))
    t_lo = np.where(meets, (-b - root) / safe_a, np.inf)
    t_hi = np.where(meets, (-b + root) / safe_a, -np.inf)
    # Rays parallel to the axis: inside (whole line) or outside (never).
    inside = c <= 2.0 * radius * GEOMETRY_TOL
    t_lo = np.where(along_axis, np.where(inside, -np.inf, np.inf), t_lo)
    t_hi = np.where(along_axis, np.where(inside, np.inf, -np.inf), t_hi)
    return t_lo, t_hi


# --------------------------------------------------------------------------
# The disc sample
# --------------------------------------------------------------------------
class DiscSample(SlabSample):
    """Multilayer disc (circular plate) centred on the beam. Create it with
    `create_disc_sample`.

    It inherits everything from `SlabSample` (frames, rotation, layers along
    the normal, interaction points, multiple scattering, ...) and only
    changes what depends on the shape of the section: the path inside the
    sample, the test "is this point inside?", the clipping of rounding
    errors, the summary and the copy with another orientation.
    """

    def __init__(self, diameter, layers, z_front=0.0, plane_angle_deg=90.0,
                 beam_height=None):
        if diameter <= 0:
            raise ValueError("The disc diameter must be positive (mm).")
        self.diameter = float(diameter)
        self.radius = self.diameter / 2.0
        # The disc occupies -R <= y <= R: for the slab code it is a square
        # D x D whose corners are cut away. The square (_lo, _hi) is kept as
        # the BOUNDING BOX, used for instance to know how far upstream the
        # beam particles must start.
        super().__init__(width=self.diameter, height=self.diameter,
                         layers=layers, z_front=z_front,
                         plane_angle_deg=plane_angle_deg,
                         y_bottom=-self.radius, beam_height=beam_height)

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------
    def summary(self):
        """Return a small text table describing the sample."""
        lines = [f"Disc sample, diameter {self.diameter:g} mm "
                 f"(y from {self.y_bottom:g} to {self.y_top:g}), "
                 f"{self.n_layers} layer(s), thickness {self.total_thickness:g} mm, "
                 f"angle beam/plane = {self.plane_angle_deg:g} deg",
                 f"{'#':>2} {'name':<10} {'z_start':>9} {'z_end':>9} "
                 f"{'thick':>8} {'mu(1/mm)':>9} {'scat.frac':>9}  material y (mm)"]
        for i, l in enumerate(self.layers):
            lines.append(f"{i:>2} {l.name:<10} {self.boundaries[i]:9.3f} "
                         f"{self.boundaries[i + 1]:9.3f} {l.thickness:8.3f} "
                         f"{l.mu:9.4f} {l.scatter_fraction:9.2f}  "
                         f"{self._fill_text(i)}")
        return "\n".join(lines)

    def with_plane_angle(self, plane_angle_deg):
        """Return a copy of this disc with another orientation."""
        return DiscSample(self.diameter, self.layers, self.z_front,
                          plane_angle_deg, self.beam_height)

    def face_area_in_beam(self, beam_width, beam_height):
        """Area (mm^2) of the part of the beam that hits the FRONT FACE of the
        disc, seen along the beam (uniform rectangular beam centred on the
        disc).

        Seen along the beam, a disc tilted by `tilt` = 90 - plane_angle looks
        like an ellipse with vertical semi-axis R and horizontal semi-axis
        R cos(tilt). At height y the ellipse is 2 R cos(tilt) sqrt(1 - y^2/R^2)
        wide; the beam is beam_width wide. The area is the integral over the
        height of the beam of the smaller of the two widths:

            area = integral_{|y| < min(R, h_b/2)} min(ellipse width, w_b) dy

        computed here with the trapezoidal rule on a fine grid. The thickness
        of the disc is ignored (exact for a thin disc). area / (w_b h_b) is
        the fraction of the beam particles that hit the face.
        """
        y_max = min(self.radius, beam_height / 2.0)
        y = np.linspace(-y_max, y_max, 20001)
        half = self.radius * self._cos * np.sqrt(np.clip(1.0 - (y / self.radius) ** 2, 0.0, 1.0))
        width = np.minimum(2.0 * half, beam_width)
        return float(_trapezoid(width, y))

    # ------------------------------------------------------------------
    # Where is a point? (sample frame)
    # ------------------------------------------------------------------
    def contains(self, points, tol=1e-9):
        """True for every point (x', y', z') inside the volume of the disc
        (empty parts of partially filled layers included): distance from the
        axis <= R and depth between the front and back faces."""
        p = np.atleast_2d(points)
        r = np.hypot(p[:, 0], p[:, 1])               # distance from the disc axis
        return ((r <= self.radius + tol)
                & (p[:, 2] >= self.z_front - tol) & (p[:, 2] <= self.z_back + tol))

    def _clip_inside(self, points):
        """Push points that rounding errors put just outside the disc back
        onto its surface: depth limited to the faces, and points slightly
        beyond the rim moved radially onto it."""
        p = np.array(points, dtype=float, copy=True)
        p[:, 2] = np.clip(p[:, 2], self.z_front, self.z_back)
        r = np.hypot(p[:, 0], p[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(r > self.radius, self.radius / r, 1.0)
        p[:, 0] *= scale
        p[:, 1] *= scale
        return p

    # ------------------------------------------------------------------
    # The only geometric difference with the slab: the section is a circle
    # ------------------------------------------------------------------
    def _section_interval(self, p, d):
        """Interval of t during which rays p + t*d are inside the disc
        (all layers together): inside the cylinder x'^2 + y'^2 <= R^2 AND
        between the front and back faces. Both are intervals of t; the ray is
        in the disc during their intersection (largest start, smallest end).
        Returns (t_in, t_out), two (N,) arrays (t_in > t_out = missed)."""
        c_lo, c_hi = _cylinder_interval(p[:, 0], p[:, 1], d[:, 0], d[:, 1], self.radius)
        z_lo, z_hi = _slice_interval(p[:, 2], d[:, 2], self.z_front, self.z_back)
        return np.maximum(c_lo, z_lo), np.minimum(c_hi, z_hi)


# --------------------------------------------------------------------------
# Public helper to build a disc sample
# --------------------------------------------------------------------------
def create_disc_sample(diameter, layers, z_front=0.0, plane_angle_deg=90.0,
                       beam_height=None):
    """Build a layered disc (circular plate) centred on the beam.

    Parameters
    ----------
    diameter : float
        Diameter of the disc (mm), the same for all the layers. The disc goes
        from y = -diameter/2 to +diameter/2 (it is centred on the beam).
    layers : list
        Layers in the order the beam meets them (front to back). Each item is
        a `Layer`, a tuple (name, thickness_mm, mu_per_mm[, scatter_fraction[,
        fill_height_mm]]) or a dict with the fields of `Layer`. As for the
        other geometries a layer can be partially filled: give `bottom` and
        `height` (mm from the beam centre; the material is cut at the rim of
        the disc), or the forms of version 2.0 (fill_height measured from the
        lowest point of the disc, y = -diameter/2, or fullness).
    z_front : float
        Position of the front face along the normal (mm). Default 0.
    plane_angle_deg : float
        Angle between the beam and the plane of the disc (0 < angle < 180).
        90 (default) = perpendicular to the beam. Rotation around the
        vertical axis through the centre of the front face.
    beam_height : float, optional
        Height of the beam (mm): needed if some layer is given by its
        fullness; also used by `summary` and `beam_fullness`.

    Returns
    -------
    DiscSample
    """
    built = [make_layer(l) for l in layers]
    return DiscSample(diameter, built, z_front, plane_angle_deg, beam_height)


# --------------------------------------------------------------------------
# Demo:  python disc.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from .beam import sample_beam_point

    # A circular cell of diameter 20 mm: two Al windows of 1 mm and 2 mm of
    # water between them, filled up to y = +2 mm (the beam goes from -4 to
    # +4 mm, so the water covers 6 of its 8 mm: fullness 0.75). The beam,
    # 8 x 8 mm, falls entirely inside the disc.
    beam_w, beam_h = 8.0, 8.0
    sample = create_disc_sample(
        20.0, [("Al", 1.0, 0.010),
               Layer("water", 2.0, 0.56, 1.0, bottom=-10.0, height=12.0),
               ("Al", 1.0, 0.010)],
        plane_angle_deg=90, beam_height=beam_h)
    print(sample.summary())

    n = 100000
    x, y = sample_beam_point(beam_w, beam_h, n=n, rng=1)
    points, hit = sample.sample_interaction_points(x, y, rng=2)
    layer = sample.layer_of_points(points)
    print("\nFraction interacting per layer:",
          np.round(np.bincount(layer, minlength=sample.n_layers) / n, 4))

    # A large beam (30 x 30 mm) on the same disc: only the particles that hit
    # the circular face can interact. Compare the fraction that crosses the
    # disc with the area of the face.
    x, y = sample_beam_point(30.0, 30.0, n=n, rng=3)
    start = sample.to_sample_frame(np.column_stack([x, y, np.full(n, -50.0)]))
    t_in, t_out = sample._section_interval(start, np.tile(sample.to_sample_frame([0, 0, 1.0]), (n, 1)))
    print("fraction of a 30 x 30 mm beam hitting the disc:", round(np.mean(t_out > t_in), 4),
          " expected pi R^2 / (30*30):", round(sample.face_area_in_beam(30, 30) / 900, 4))

    # Transmission towards a detector at 2theta = 60 deg
    d = sample.to_sample_frame(scattered_direction(60.0))
    print("Mean survival to the detector (2theta = 60 deg):",
          round(sample.transmission(sample.outgoing_path_lengths(points, d)).mean(), 4))
