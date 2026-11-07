"""
sphere.py - SPHERICAL geometry of MCAMS (version 2.4): concentric spheres,
an innermost full sphere surrounded by spherical shells, centred on the beam.

(The other geometries are in slab.py, disc.py and cylinder.py; all follow the
common interface of sample_base.py, so every simulation works with them.)

What the sample looks like
--------------------------
Think of an onion, or of a ball inside a spherical container: a full sphere
of one material, surrounded by shells of other materials (a container, a
heat shield, a gap...). For example a spherical powder sample in a thin
vanadium sphere, a droplet, or a ball of a single material.

    * the centre of all the spheres is the centre of the instrument, on the
      beam axis (x = y = z = 0 of the lab frame);
    * the LAYERS are given from the INNER one outwards, exactly as for the
      cylinder:
          - 1st layer = full sphere. Its "thickness" is its DIAMETER.
          - next layers = shells. Their thickness is the RADIAL thickness.
      Example: thicknesses (6.0, 1.0, 0.5) mm give radii 3.0, 4.0 and 4.5 mm
      (outer diameter 9 mm);
    * each layer has the same properties as in the other geometries: label,
      thickness, mu, scatter fraction and vertical extent of its material
      (bottom and height, version 2.2). A spherical flask half filled with a
      liquid is a full shell (the glass) around an inner sphere whose material
      goes from the bottom of the sphere up to the level of the liquid;
    * a sphere looks the same from every direction: there is no orientation
      parameter, and the sample frame is the lab frame.

Conventions
-----------
* All distances in mm, linear attenuation coefficients in 1/mm.
* LAB frame (= sample frame): beam along +z, x horizontal, y vertical; the
  beam is centred on y = 0.
* The sample occupies |r| <= R_out, so it goes from y = -R_out to +R_out
  (y_bottom = -R_out, height = 2 R_out). The bands of material of the layers
  are cut at the outer sphere. In the forms of version 2.0, fill_height is
  measured from the lowest point of the sphere (y = -R_out).

How the paths are computed
--------------------------
Exactly as for the cylinder (read the docstring of cylinder.py first): the
straight path p + t*d is cut into SEGMENTS at every place where the material
may change, and inside one segment the material is always the same. The
places are
    * where the path crosses the sphere of radius R_i of every layer: the
      solutions of |p + t d|^2 = R_i^2, a quadratic equation (see
      `_shell_interval`);
    * where it crosses the horizontal planes that limit the material of the
      partially filled layers (y = bottom_i or top_i).
The layer of a segment is given by the distance of its midpoint to the
CENTRE (not to an axis), and whether it is in the material by the height of
its midpoint. Then the path length in each layer and the next interaction are
found by walking along the segments, as for the cylinder.

So `SphericalSample` is a `CylindricalSample` in which three small methods
are different: the distance that decides the layer (`_radius_of`), the
crossings of a layer boundary (`_shell_interval`) and the outline of the
whole sample (`_outline_interval`: the outer sphere; a sphere has no flat
ends). The rest is inherited.

Typical use
-----------
    from sphere import create_spherical_sample
    sample = create_spherical_sample(
        [("Ni", 6.0, 0.180, 0.94),      # inner sphere: DIAMETER 6 mm
         ("V",  0.2, 0.046, 0.78)],     # shell: radial thickness 0.2 mm
        beam_height=8)
    print(sample.summary())

NumPy functions used in this file
---------------------------------
np.cumsum(a)          : running sum: [3, 0.2] -> [3, 3.2] (radii from thicknesses).
np.sqrt(a)            : square root of every element.
np.where(cond, a, b)  : element by element, `a` where cond is True, else `b`.
np.full(n, value)     : array of n copies of `value`.
np.atleast_2d(a)      : make one point (3,) into a table (1, 3).
np.linalg.norm(v, axis=1) : length of each row vector.
np.linspace(a, b, n)  : n equally spaced numbers from a to b.
np.minimum            : element by element smaller of two arrays.
np.trapezoid(y, x)    : integral by the trapezoidal rule (np.trapz in NumPy < 2.0).
"""

import numpy as np

from sample_base import Layer, scattered_direction, make_layer   # noqa: F401
from cylinder import CylindricalSample

# np.trapezoid is the NumPy >= 2.0 name of np.trapz.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class SphericalSample(CylindricalSample):
    """Concentric spheres centred on the beam. Create it with
    `create_spherical_sample`.

    It inherits from `CylindricalSample` the way paths are cut into segments,
    the interaction points, the path lengths and the multiple scattering,
    and redefines only what depends on the shape: the distance that tells
    the layer of a point, the crossings of the spheres, the outline of the
    sample, the test "is this point inside?" and the summary.
    """

    def __init__(self, layers, beam_height=None):
        # The outer radius is needed BEFORE the common initialisation, because
        # it fixes the vertical extent of the sample (-R_out to +R_out). It is
        # computed here from the thicknesses (first one = diameter); the parent
        # class computes the radii again, in the same way.
        built = [make_layer(l) for l in layers]
        if len(built) == 0:
            raise ValueError("A sample needs at least one layer.")
        steps = np.array([l.thickness for l in built], dtype=float)
        steps[0] = steps[0] / 2.0                    # diameter -> radius
        r_out = float(np.cumsum(steps)[-1])
        super().__init__(height=2.0 * r_out, layers=built, y_bottom=-r_out,
                         beam_height=beam_height)
        self.diameter = 2.0 * self.outer_radius

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------
    def summary(self):
        """Return a small text table describing the sample."""
        lines = [f"Spherical sample, outer diameter {self.diameter:g} mm "
                 f"(centred on the beam, y from {self.y_bottom:g} to {self.y_top:g}), "
                 f"{self.n_layers} layer(s)",
                 f"{'#':>2} {'name':<10} {'r_in':>8} {'r_out':>8} "
                 f"{'thick':>8} {'mu(1/mm)':>9} {'scat.frac':>9}  material y (mm)"]
        for i, l in enumerate(self.layers):
            lines.append(f"{i:>2} {l.name:<10} {self.inner_radii[i]:8.3f} "
                         f"{self.radii[i]:8.3f} {l.thickness:8.3f} "
                         f"{l.mu:9.4f} {l.scatter_fraction:9.2f}  "
                         f"{self._fill_text(i)}")
        return "\n".join(lines)

    def section_area_in_beam(self, beam_width, beam_height):
        """Area (mm^2) of the part of a uniform rectangular beam (centred on
        the sphere) that hits the sphere: the circle of radius R_out seen
        along the beam, inside the beam rectangle.

        At height y the circle is 2 sqrt(R^2 - y^2) wide; the beam is
        beam_width wide. The area is the integral over the height of the beam
        of the smaller of the two widths:

            area = integral_{|y| < min(R, h_b/2)} min(2 sqrt(R^2 - y^2), w_b) dy

        computed with the trapezoidal rule on a fine grid (pi R^2 for a beam
        larger than the sphere). area / (w_b h_b) is the fraction of the beam
        particles that meet the sample.
        """
        R = self.outer_radius
        y_max = min(R, beam_height / 2.0)
        y = np.linspace(-y_max, y_max, 20001)
        chord = 2.0 * np.sqrt(np.clip(R * R - y * y, 0.0, None))
        return float(_trapezoid(np.minimum(chord, beam_width), y))

    # ------------------------------------------------------------------
    # Where is a point?
    # ------------------------------------------------------------------
    def contains(self, points, tol=1e-9):
        """True for every point (x, y, z) inside the outer sphere (the empty
        part of partially filled layers included)."""
        p = np.atleast_2d(points)
        return np.linalg.norm(p, axis=1) <= self.outer_radius + tol

    # ------------------------------------------------------------------
    # The three shape-dependent ingredients of the path calculation
    # ------------------------------------------------------------------
    @staticmethod
    def _radius_of(x, y, z):
        """Distance of the points to the CENTRE: it tells the layer."""
        return np.sqrt(x * x + y * y + z * z)

    def _shell_interval(self, p, d, R):
        """Distances t at which the rays p + t*d enter and leave the sphere of
        radius R centred on the origin.

        A point of the ray is on the sphere when |p + t d|^2 = R^2, i.e.

            a t^2 + 2 b t + c = 0   with   a = d.d (= 1 for unit d),
                                           b = p.d ,
                                           c = p.p - R^2 ,

        whose solutions t = (-b -+ sqrt(b^2 - a c)) / a are where the ray
        enters and leaves the sphere. If b^2 - a c <= 0 the line misses the
        sphere (or only touches it): empty interval, coded (+inf, -inf).
        Unlike the cylinder there is no special direction: every straight
        line either crosses a sphere or misses it.
        Returns (t_lo, t_hi), two (N,) arrays.
        """
        a = np.sum(d * d, axis=1)
        b = np.sum(p * d, axis=1)
        c = np.sum(p * p, axis=1) - R * R
        disc = b * b - a * c                          # discriminant (divided by 4)
        crosses = disc > 0.0
        root = np.sqrt(np.where(crosses, disc, 0.0))  # sqrt only where valid
        t_lo = np.where(crosses, (-b - root) / a, np.inf)
        t_hi = np.where(crosses, (-b + root) / a, -np.inf)
        return t_lo, t_hi

    def _outline_interval(self, p, d):
        """Distances t during which the rays are inside the whole sample:
        the outer sphere (a sphere has no flat ends)."""
        return self._shell_interval(p, d, self.outer_radius)


# --------------------------------------------------------------------------
# Public helper to build a spherical sample
# --------------------------------------------------------------------------
def create_spherical_sample(layers, beam_height=None):
    """Build a sample of concentric spheres centred on the beam.

    Parameters
    ----------
    layers : list
        Layers from the INNER one outwards. Each item is a `Layer`, a tuple
        (name, thickness_mm, mu_per_mm[, scatter_fraction[, fill_height_mm]])
        or a dict with the fields of `Layer`. The thickness of the first
        (innermost, full sphere) layer is its DIAMETER; for the others it is
        the radial thickness of the shell. As for the other geometries a layer
        can be partially filled: give `bottom` and `height` (mm from the beam
        centre; the material is cut at the outer sphere), or the forms of
        version 2.0 (fill_height measured from the lowest point of the
        sphere, y = -R_out, or fullness).
    beam_height : float, optional
        Height of the beam (mm): needed if some layer is given by its
        fullness; also used by `summary` and `beam_fullness`.

    Returns
    -------
    SphericalSample
    """
    return SphericalSample(layers, beam_height)


# --------------------------------------------------------------------------
# Demo:  python sphere.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from beam import sample_beam_point

    # A spherical vanadium container (wall 0.2 mm) of inner diameter 10 mm,
    # half filled with a powder: the powder goes from the bottom of the
    # sphere up to y = 0 (the centre of the beam). Beam 8 x 8 mm.
    beam_w, beam_h = 8.0, 8.0
    sample = create_spherical_sample(
        [Layer("powder", 10.0, 0.25, 0.8, bottom=-5.0, height=5.0),
         ("V", 0.2, 0.046, 0.78)],
        beam_height=beam_h)
    print(sample.summary())

    n = 100000
    x, y = sample_beam_point(beam_w, beam_h, n=n, rng=1)
    points, hit = sample.sample_interaction_points(x, y, rng=2)
    layer = sample.layer_of_points(points)
    print("\nFraction interacting per layer:",
          np.round(np.bincount(layer, minlength=sample.n_layers) / n, 4))
    print("Fraction of the beam that meets the sphere:",
          round(sample.section_area_in_beam(beam_w, beam_h) / (beam_w * beam_h), 4))

    d = sample.to_sample_frame(scattered_direction(60.0))
    print("Mean survival to the detector (2theta = 60 deg):",
          round(sample.transmission(sample.outgoing_path_lengths(points, d)).mean(), 4))
