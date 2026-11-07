"""
cylinder.py - CYLINDRICAL geometry of MCAMS: concentric cylinders (an
innermost full cylinder surrounded by annular layers), with the common axis
VERTICAL. Each layer can be only partially filled: its material lies between
its own bottom and top (version 2.2), with empty space below and above it, as
in a partially filled container or a sample not centred on the beam.

(The slab geometry is in slab.py; both follow the common interface defined
in sample_base.py, so the simulations work with either.)

Geometry and conventions
------------------------
* LAB frame: beam along +z, x horizontal, y VERTICAL (= axis of the cylinders).
* The axis of the cylinders passes through the origin: the centre of the
  circle on which the detector moves (2-axis instrument). The detector moves
  in the horizontal plane y = 0.
* The cylinders span heights y_bottom <= y <= y_top = y_bottom + height (all
  the same height; default centred on the beam, y_bottom = -height/2). Their
  ends are flat; particles can leave through them.
* VERTICAL EXTENT: layer i contains material only for
  fill_bottoms[i] <= y <= fill_tops[i] (see sample_base.py: `bottom` and
  `height` of each Layer, from the centre of the beam); below and above that
  range the annulus (or the inner cylinder) is empty.
* LAYERS are given from the INNER one outwards:
      - 1st layer = full cylinder. Its "thickness" is its DIAMETER.
      - next layers = annuli. Their thickness is the RADIAL thickness.
  Example: thicknesses (6.0, 1.0, 0.5) mm give radii 3.0, 4.0 and 4.5 mm
  (outer diameter 9 mm).
* SAMPLE frame = LAB frame: rotating a vertical cylinder around its own axis
  changes nothing, so no orientation parameter is needed.

How the particle paths are computed (the key idea)
--------------------------------------------------
A particle moves along a straight line  position(t) = p + t*d  (d = unit
direction, t = distance travelled). Looking from above, a cylinder is a circle
(x^2 + z^2 = R^2), so the line meets each circle at two values of t (found by
solving a quadratic equation). Together with the places where the line leaves
the top/bottom faces, and the places where it crosses the level of the
material of a partially filled layer (horizontal planes y = bottom or top),
these values cut the path into SEGMENTS. Inside one segment the material is
always the same, so we only need to know which layer each segment belongs to
(we test the radius of its midpoint) and whether it is in the material or in
the empty space below or above it (we test the height of its midpoint).
Everything we need follows from the list of segments of every particle:
  * path length in each layer      -> add up the lengths of its FILLED segments,
  * where the next interaction is  -> walk along the segments accumulating
                                      the optical depth (mu * length; mu = 0
                                      in the empty segments).

NumPy functions used in this file (short glossary)
--------------------------------------------------
np.hypot(a, b)        : sqrt(a**2 + b**2), computed safely.
np.sqrt, np.log       : square root / natural logarithm.
np.sort(a, axis=1)    : sort every row of a 2D table.
np.cumsum(a, axis=1)  : running sum along each row: [1,2,3] -> [1,3,6].
np.clip(a, lo, hi)    : limit values to the interval [lo, hi] (lo, hi may be arrays).
np.where(cond, a, b)  : element by element, `a` where cond is True, else `b`.
np.searchsorted(edges, v) : for each v, in which interval of the sorted array `edges`
                        it falls (used to find in which layer a radius lies).
np.column_stack       : put 1D arrays side by side as columns of a 2D table.
np.full(n, value)     : array of n copies of `value`.
np.tile(a, (n, 1))    : repeat the row `a` n times.
np.broadcast_to       : repeat an array virtually to a bigger shape (no copy).
np.inf                : infinity; "t = -inf ... +inf" means "for ever".
np.errstate           : temporarily silence warnings (divisions we protect ourselves).
np.arange(n)          : 0, 1, ..., n-1.
boolean masks         : a[mask] keeps only the elements where mask is True;
                        `~mask` is "not mask", `&` is "and".
"""

import numpy as np

from .sample_base import (Layer, LayeredSample, scattered_direction,  # noqa: F401
                         make_layer, GEOMETRY_TOL)


class CylindricalSample(LayeredSample):
    """Concentric vertical cylinders. Create it with `create_cylindrical_sample`."""

    # Number of particles processed at once in the first-interaction
    # calculation. Each particle needs a small table (one row per segment),
    # so very large numbers of particles are processed in chunks to limit the
    # memory used.
    _CHUNK = 200_000

    def __init__(self, height, layers, y_bottom=None, beam_height=None):
        # Common data (thicknesses, mus, scatter fractions, vertical extent
        # and filling of the layers) are stored by the base class.
        super().__init__(layers, height, y_bottom, beam_height)

        # --- Radii of the layers ---------------------------------------------
        # The thickness of the innermost layer is a DIAMETER: its radius is
        # half of it. For the other layers the thickness is radial. So, the
        # list of radial increments is [diameter/2, t_2, t_3, ...] and the outer
        # radius of each layer is the running sum (np.cumsum) of the increments.
        radial_steps = self.thicknesses.copy()
        radial_steps[0] = radial_steps[0] / 2.0
        self.radii = np.cumsum(radial_steps)             # outer radius of each layer
        # Inner radius of each layer = outer radius of the previous one
        # (0 for the innermost). np.concatenate joins the 0 and the radii.
        self.inner_radii = np.concatenate(([0.0], self.radii[:-1]))
        self.outer_radius = float(self.radii[-1])

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------
    def summary(self):
        """Return a small text table describing the sample.

        The column "fill" gives the height of the material above the bottom
        of the sample (mm), or "full"; when the beam height is known, the
        fullness (fraction of the beam covered) is shown in parentheses.
        """
        lines = [f"Cylindrical sample, height {self.height:g} mm "
                 f"(y from {self.y_bottom:g} to {self.y_top:g}), "
                 f"{self.n_layers} layer(s), outer diameter "
                 f"{2 * self.outer_radius:g} mm (axis vertical)",
                 f"{'#':>2} {'name':<10} {'r_in':>8} {'r_out':>8} "
                 f"{'thick':>8} {'mu(1/mm)':>9} {'scat.frac':>9}  material y (mm)"]
        for i, l in enumerate(self.layers):
            # For the inner layer the given thickness is a diameter; we show
            # its radius (r_out) and the thickness as given.
            lines.append(f"{i:>2} {l.name:<10} {self.inner_radii[i]:8.3f} "
                         f"{self.radii[i]:8.3f} {l.thickness:8.3f} "
                         f"{l.mu:9.4f} {l.scatter_fraction:9.2f}  "
                         f"{self._fill_text(i)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Frames: the sample frame is the lab frame
    # ------------------------------------------------------------------
    def to_sample_frame(self, vectors):
        """Lab -> sample frame. They coincide, so the vectors are only
        converted to an (N, 3) float array."""
        return np.atleast_2d(np.asarray(vectors, dtype=float))

    def to_lab_frame(self, vectors):
        """Sample -> lab frame (identical frames, see above)."""
        return np.atleast_2d(np.asarray(vectors, dtype=float))

    # ------------------------------------------------------------------
    # Where is a point?
    # ------------------------------------------------------------------
    def contains(self, points, tol=1e-9):
        """True for every point (x, y, z) inside the cylinder (the empty part
        of partially filled layers included).
        `tol` forgives floating-point rounding errors for points on a surface."""
        p = np.atleast_2d(points)
        r = np.hypot(p[:, 0], p[:, 2])            # distance to the vertical axis
        return ((r <= self.outer_radius + tol) & (p[:, 1] >= self.y_bottom - tol)
                & (p[:, 1] <= self.y_top + tol))

    def layer_of_points(self, points):
        """Index of the layer whose MATERIAL contains each point: -1 if the
        point is outside the sample or in the empty space below or above the
        material of a partially filled layer."""
        p = np.atleast_2d(points)
        r = self._radius_of(p[:, 0], p[:, 1], p[:, 2])   # distance to the axis
        # np.searchsorted tells in which interval of the sorted list of outer
        # radii r falls: r < radii[0] -> 0 (inner cylinder), radii[0] <= r <
        # radii[1] -> 1, and so on. A point exactly on the outer surface
        # gives n_layers, which we move back to the last layer.
        idx = np.minimum(np.searchsorted(self.radii, r, side="right"),
                         self.n_layers - 1)
        outside = ~self.contains(p, tol=GEOMETRY_TOL)
        empty = ~self.is_filled(idx, p[:, 1])     # below or above the material
        return np.where(outside | empty, -1, idx)

    # ------------------------------------------------------------------
    # The three shape-dependent ingredients of the path calculation.
    # (sphere.py redefines them; everything else is shared.)
    # ------------------------------------------------------------------
    @staticmethod
    def _radius_of(x, y, z):
        """Distance of the points (x, y, z) to the vertical axis: this number
        tells in which layer a point is."""
        return np.hypot(x, z)

    def _shell_interval(self, p, d, R):
        """Distances t at which the rays p + t*d enter and leave the
        (infinitely tall) cylinder of radius R around the vertical axis.

        Seen from above, the ray is the line (x + t dx, z + t dz) and the
        cylinder is the circle x^2 + z^2 = R^2. Substituting:
            a t^2 + 2 b t + c = 0
            a = dx^2 + dz^2 ,  b = x dx + z dz ,  c = x^2 + z^2 - R^2
        whose solutions are t = (-b -+ sqrt(b^2 - a c)) / a.
        An empty interval (the ray misses it) is coded as (+inf, -inf); a
        vertical ray (a = 0) is inside for ever if it starts inside, never
        otherwise. Returns (t_lo, t_hi), two (N,) arrays.
        """
        x, z, dx, dz = p[:, 0], p[:, 2], d[:, 0], d[:, 2]
        n = len(p)
        a = dx * dx + dz * dz
        b = x * dx + z * dz
        horizontal = a > 1e-14        # False for rays parallel to the axis (vertical)
        a_safe = np.where(horizontal, a, 1.0)     # avoid dividing by zero below
        c = x * x + z * z - R * R
        disc = b * b - a * c                       # discriminant
        crosses = horizontal & (disc > 0.0)        # the circle is really crossed
        root = np.sqrt(np.where(crosses, disc, 0.0))   # sqrt only where valid
        t_lo = np.full(n, np.inf)                  # default: empty interval
        t_hi = np.full(n, -np.inf)
        # Boolean-mask assignment: only where `crosses` is True.
        t_lo[crosses] = ((-b - root) / a_safe)[crosses]
        t_hi[crosses] = ((-b + root) / a_safe)[crosses]
        # A vertical ray never crosses the circle: it stays inside for ever
        # if it starts inside (c < 0), otherwise it never touches it.
        vertical_inside = (~horizontal) & (c < 0.0)
        t_lo[vertical_inside] = -np.inf
        t_hi[vertical_inside] = np.inf
        return t_lo, t_hi

    def _outline_interval(self, p, d):
        """Distances t during which the rays are inside the whole sample:
        inside the outer cylinder AND between the bottom and top faces
        (heights y_bottom and y_top). Returns (t_lo, t_hi) (empty if t_lo >= t_hi)."""
        y, dy = p[:, 1], d[:, 1]
        t_lo_c, t_hi_c = self._shell_interval(p, d, self.outer_radius)
        moving_v = np.abs(dy) > 1e-14                       # moves up or down?
        dy_safe = np.where(moving_v, dy, 1.0)
        t_a = (self.y_bottom - y) / dy_safe                 # distance to the bottom face
        t_b = (self.y_top - y) / dy_safe                    # distance to the top face
        inside_y = (y >= self.y_bottom - 1e-9) & (y <= self.y_top + 1e-9)
        # If the ray moves vertically, it is between the faces for t_a..t_b.
        # If it is horizontal, it is between them always (-inf..inf) when it
        # starts between them, never (empty interval) otherwise.
        t_y0 = np.where(moving_v, np.minimum(t_a, t_b),
                        np.where(inside_y, -np.inf, np.inf))
        t_y1 = np.where(moving_v, np.maximum(t_a, t_b),
                        np.where(inside_y, np.inf, -np.inf))
        return np.maximum(t_lo_c, t_y0), np.minimum(t_hi_c, t_y1)

    # ------------------------------------------------------------------
    # The heart of the geometry: cut every particle path into segments
    # ------------------------------------------------------------------
    def _trace(self, points, directions):
        """Cut the path of each particle inside the sample into segments.

        Each particle starts at `points[j]` (inside the sample OR outside it,
        e.g. upstream of it) and flies along `directions[j]`.

        Returns five arrays, with S = 2*n_layers + 1 + n_levels segments per
        particle (n_levels = number of different levels of material of the
        partially filled layers):
            starts : (N, S)  distance along the ray where each segment starts
            lens   : (N, S)  length of each segment (mm), >= 0
            layers : (N, S)  index of the layer of each segment
            filled : (N, S)  True if the segment is in the material, False if
                             it is in the empty space below or above it
            valid  : (N,)    False if the ray never goes through the sample
                             (then all its segments have length 0)
        """
        p = np.atleast_2d(np.asarray(points, dtype=float))
        d = np.atleast_2d(np.asarray(directions, dtype=float))
        d = np.broadcast_to(d, p.shape).astype(float)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)     # unit vectors
        n = len(p)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

        # --- Steps 1-3: the part of the ray inside the whole sample --------
        # (outermost cylinder, finite height, and only t >= 0 = forward). The
        # shape of the outline is computed by `_outline_interval`, so that
        # other round geometries (the sphere, sphere.py) can reuse _trace.
        t_lo_out, t_hi_out = self._outline_interval(p, d)
        t_in = np.maximum(t_lo_out, 0.0)                     # enters the sample
        t_out = t_hi_out                                     # leaves the sample
        valid = t_out > t_in
        # Rays that miss the sample get a zero-length path: t_in = t_out = 0.
        t_in = np.where(valid, t_in, 0.0)
        t_out = np.where(valid, t_out, 0.0)
        moving_v = np.abs(dy) > 1e-14                       # moves up or down?
        dy_safe = np.where(moving_v, dy, 1.0)

        # --- Step 4: breakpoints = every place where the material may change
        # For each layer boundary (cylinder, or sphere, of radius R_i) the ray enters and
        # leaves it. Clip these distances to [t_in, t_out] so that they lie
        # on the path inside the sample (np.clip accepts arrays as limits).
        breakpoints = [t_in]
        for R in self.radii:
            t_lo, t_hi = self._shell_interval(p, d, R)
            breakpoints.append(np.clip(t_lo, t_in, t_out))
            breakpoints.append(np.clip(t_hi, t_in, t_out))
        # The bottoms and tops of the material of the partially filled layers are
        # horizontal planes y = level: the ray crosses each of them once (at
        # t = (level - y)/dy), or never if it moves horizontally.
        for level in self.fill_levels:
            t_level = np.where(moving_v, (level - y) / dy_safe, t_in)
            breakpoints.append(np.clip(t_level, t_in, t_out))
        breakpoints.append(t_out)
        # One row per particle, one column per breakpoint, sorted along the
        # ray: consecutive breakpoints delimit the segments.
        bp = np.sort(np.column_stack(breakpoints), axis=1)

        starts = bp[:, :-1]
        lens = bp[:, 1:] - bp[:, :-1]                        # length of each segment

        # --- Step 5: which layer is each segment in? -----------------------
        # Look at the midpoint of the segment: its distance to the axis
        # (to the centre, for a sphere) tells in which layer it is. (starts[:, :] is (N,S); x[:, None]
        # makes x a column (N,1) so that both combine into an (N,S) table.)
        mid = starts + 0.5 * lens
        r_mid = self._radius_of(x[:, None] + dx[:, None] * mid,
                                y[:, None] + dy[:, None] * mid,
                                z[:, None] + dz[:, None] * mid)
        layers = np.minimum(np.searchsorted(self.radii, r_mid, side="right"),
                            self.n_layers - 1)
        # Is the segment in the material or in the empty space below/above it?
        # Compare the height of its midpoint with the level of its layer.
        y_mid = y[:, None] + dy[:, None] * mid
        filled = self.is_filled(layers, y_mid)
        return starts, lens, layers, filled, valid

    # ------------------------------------------------------------------
    # Path lengths
    # ------------------------------------------------------------------
    def outgoing_path_lengths(self, points, directions):
        """Distance travelled in the MATERIAL of each layer by particles that
        start at `points` (inside the cylinder) and fly along `directions`
        until they leave the sample (through the side or through the
        top/bottom faces). The empty space below or above the material is
        not counted (it does not attenuate).

        Returns an array (N, n_layers) in mm.
        """
        p = self._check_points(points)
        starts, lens, layers, filled, valid = self._trace(p, directions)
        lengths = np.zeros((len(p), self.n_layers))
        for i in range(self.n_layers):
            # Add up the length of the filled segments that belong to layer
            # i: np.where keeps `lens` only where both conditions hold.
            lengths[:, i] = np.sum(np.where((layers == i) & filled, lens, 0.0),
                                   axis=1)
        return lengths

    # ------------------------------------------------------------------
    # Where does the particle interact?
    # ------------------------------------------------------------------
    def _first_interaction(self, points, directions, rng):
        """Find the first interaction of particles flying from `points` along
        `directions` (the points may be inside the sample or upstream of it).

        Returns (new_points (N, 3), interacts (N,) boolean).
        """
        p_all = np.atleast_2d(np.asarray(points, dtype=float))
        d_all = np.atleast_2d(np.asarray(directions, dtype=float))
        d_all = np.broadcast_to(d_all, p_all.shape).astype(float)
        d_all = d_all / np.linalg.norm(d_all, axis=1, keepdims=True)

        new_points = np.empty_like(p_all)
        interacts = np.zeros(len(p_all), dtype=bool)

        # Process the particles in chunks to limit the memory used.
        for i0 in range(0, len(p_all), self._CHUNK):
            sl = slice(i0, i0 + self._CHUNK)           # this chunk of particles
            p, d = p_all[sl], d_all[sl]
            n = len(p)

            # Segments of the path inside the sample.
            starts, lens, layers, filled, valid = self._trace(p, d)

            # Linear attenuation coefficient of every segment:
            # self.mus[layers] picks, for every segment, the mu of its layer
            # ("fancy indexing" with an integer table); the empty segments
            # (outside the band of the material) get mu = 0.
            mu_seg = np.where(filled, self.mus[layers], 0.0)
            # Optical depth (mu * length) of every segment and its running
            # total along the path (np.cumsum along each row).
            tau_seg = lens * mu_seg
            cum = np.cumsum(tau_seg, axis=1)

            # Beer-Lambert: the optical depth at which a particle interacts is
            # exponentially distributed: tau = -ln(1-u) with u uniform in [0,1).
            tau = -np.log(1.0 - rng.uniform(size=n))
            hit = valid & (tau < cum[:, -1])      # it interacts inside the sample

            # Segment in which tau is reached = number of segments whose
            # cumulative optical depth is still <= tau.
            k = np.minimum((cum <= tau[:, None]).sum(axis=1), cum.shape[1] - 1)
            rows = np.arange(n)
            mu_k = mu_seg[rows, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                extra = (tau - (cum[rows, k] - tau_seg[rows, k])) / mu_k
            dist = np.where(hit, starts[rows, k] + extra, 0.0)

            new_points[sl] = p + d * dist[:, None]
            interacts[sl] = hit
        return new_points, interacts

    def next_interaction(self, points, directions, rng=None):
        """Next interaction of particles flying inside the sample (used for
        multiple scattering). See `LayeredSample.next_interaction`."""
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        return self._first_interaction(points, directions, rng)

    def sample_interaction_points(self, x, y, rng=None):
        """First interaction of beam particles.

        The particles travel along +z. (x, y) is their horizontal and
        vertical position in the plane perpendicular to the beam through the
        axis of the cylinders (the numbers given by `sample_beam_point`).
        A particle with |x| larger than the outer radius, or with y outside
        [y_bottom, y_top], misses the sample. A particle passing above the
        material of a partially filled layer crosses its empty part without
        interacting there.

        Returns (points (M, 3), interacted (N,) boolean); the points are in
        the sample (= lab) frame.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # The beam plane z = 0 goes through the axis, i.e. INSIDE the sample.
        # So we start the particles upstream, just outside the cylinder
        # (z = -(R_out + 1)), and let them fly along +z.
        start = np.column_stack([x, y, np.full(len(x), -(self.outer_radius + 1.0))])
        direction = np.array([0.0, 0.0, 1.0])
        new_points, interacted = self._first_interaction(start, direction, rng)
        return new_points[interacted], interacted

    # ------------------------------------------------------------------
    def _check_points(self, points):
        """Return points as an (N, 3) float array; raise if outside sample."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        if p.shape[1] != 3:
            raise ValueError("Points must have 3 coordinates (x, y, z).")
        if not np.all(self.contains(p)):
            raise ValueError("Some points are outside the sample.")
        return p


# --------------------------------------------------------------------------
# Public helper to build a cylindrical sample
# --------------------------------------------------------------------------
def create_cylindrical_sample(height, layers, y_bottom=None, beam_height=None):
    """Build a sample of concentric vertical cylinders.

    Parameters
    ----------
    height : float or None
        Height of the sample (mm): the cylinders go from y_bottom to
        y_bottom + height. None: the smallest range containing the material
        of the layers given with `bottom` and `height` (version 2.2).
    layers : list
        Layers from the INNER one outwards. Each item is a `Layer`, a tuple
        (name, thickness_mm, mu_per_mm[, scatter_fraction[, fill_height_mm]])
        or a dict with the fields of `Layer` (use a dict or a Layer to give
        `bottom` and `height`, or the `fullness`).
        The thickness of the first (innermost, full cylinder) layer is its
        DIAMETER; for the others it is the radial thickness of the annulus.
    y_bottom : float, optional
        Height of the bottom of the sample (mm, lab y, 0 = beam centre).
        Default: centred on the beam (y_bottom = -height/2), or the lowest
        bottom of the layers when height is None. Layers without `bottom`
        start here.
    beam_height : float, optional
        Height of the beam (mm): needed if some layer is given by its
        fullness; also shown in the summary.

    Returns
    -------
    CylindricalSample
    """
    built = [make_layer(l) for l in layers]
    return CylindricalSample(height, built, y_bottom, beam_height)


# --------------------------------------------------------------------------
# Demo:  python cylinder.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from .beam import sample_beam_point

    # A vanadium can (wall 0.2 mm, 20 mm high, from y = -12 to +8 mm) of
    # inner diameter 6 mm. The powder rests on the bottom of the can, 8 mm
    # below the centre of the 8 mm beam, and is 10.4 mm high: it reaches
    # y = -1.6 mm, so it covers 0.3 of the beam (beam from -4 to +4 mm).
    beam_w, beam_h = 8.0, 8.0
    sample = create_cylindrical_sample(
        None, [Layer("powder", 6.0, 0.250, 0.8, bottom=-12.0, height=10.4),
               Layer("V", 0.2, 0.040, 0.1, bottom=-12.0, height=20.0)],
        beam_height=beam_h)
    print(sample.summary())

    n = 100000
    x, y = sample_beam_point(beam_w, beam_h, n=n, rng=1)
    points, hit = sample.sample_interaction_points(x, y, rng=2)
    layer = sample.layer_of_points(points)
    print("\nFraction interacting per layer:",
          np.round(np.bincount(layer, minlength=sample.n_layers) / n, 4))
    print("Fraction crossing without interaction (or missing):",
          round(1 - hit.mean(), 4))

    d = sample.to_sample_frame(scattered_direction(60.0))
    L_out = sample.outgoing_path_lengths(points, d)
    print("Mean survival probability to the detector (2theta=60 deg):",
          round(sample.transmission(L_out).mean(), 4))
