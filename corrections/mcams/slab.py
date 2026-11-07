"""
slab.py - SLAB geometry of MCAMS: a rectangular sample made of flat layers
perpendicular to the sample normal, which can be rotated around the vertical
axis with respect to the incident beam. Each layer can be only partially
filled: its material lies between its own bottom and top, with empty space
below and above it (version 2.2).

(The cylindrical geometry is in cylinder.py; both follow the common
interface defined in sample_base.py.)

Conventions
-----------
* All distances in mm, so linear attenuation coefficients are in 1/mm.
* LAB frame: the beam travels along +z, x is horizontal, y is vertical.
* SAMPLE frame: attached to the sample. z' is the normal of the layers, x'
  and y' lie in the plane of the layers (y' = y, vertical). Layer i spans
  boundaries[i] <= z' <= boundaries[i+1]; the front face is at z' = z_front
  (default 0); horizontally the section is centred: |x'| <= width/2;
  vertically the sample goes from y_bottom to y_top = y_bottom + height
  (default: centred on the beam, y_bottom = -height/2).
* VERTICAL EXTENT (version 2.2): layer i contains material only for
  fill_bottoms[i] <= y' <= fill_tops[i] (see sample_base.py: `bottom` and
  `height` of each Layer, measured from the centre of the beam); below and
  above that range the layer is empty.
* ORIENTATION: `plane_angle_deg` is the angle between the incident beam and
  the plane of the sample (the layers):
      90  -> plane perpendicular to the beam (normal incidence; lab frame and
             sample frame coincide)
      <90 -> sample rotated around the vertical axis y, normal tilted
             towards +x (tilt = 90 - plane_angle)
      >90 -> same, but tilted towards -x
  The rotation axis is the vertical line through the origin of the sample
  frame, which is the centre of the front face (when z_front = 0), on the
  beam axis. Values very close to 0 or 180 (beam parallel to the plane) are
  not allowed.

How the paths are computed (the key idea)
-----------------------------------------
A particle moves along a straight line  position(t) = p + t*d  (d = unit
direction, t = distance travelled). The MATERIAL of layer i is the
intersection of three simple regions:
    * the box of the sample  (|x'| <= w/2, y_bottom <= y' <= y_top,
                              z_front <= z' <= z_back),
    * the slice of the layer (boundaries[i] <= z' <= boundaries[i+1]),
    * the band of heights of its material (fill_bottoms[i] <= y' <= fill_tops[i]).
Along a straight line each of these regions is an INTERVAL of t, so their
intersection is an interval too: [t0_i, t1_i]. Its length t1_i - t0_i is the
path in the material of layer i. Because the layers follow each other along
z', these intervals do not overlap, and sorting them by t0 gives the order in
which the particle meets the material. Everything follows from them:
  * path length in each layer  -> t1 - t0,
  * where the next interaction is -> walk along the intervals in order,
                                     accumulating the optical depth mu*length.
For full layers this gives exactly the same results as before.

What the SlabSample object gives you (the interface of LayeredSample)
-----------------------------------------------------------------
    sample.sample_interaction_points(x, y)  -> where beam particles interact
    sample.next_interaction(points, dirs)   -> next interaction of a flying particle
    sample.outgoing_path_lengths(points, d) -> path in each layer, point -> exit
    sample.transmission(lengths)            -> exp(-sum(mu_i * L_i))
    sample.layer_of_points(points)          -> which layer contains each point
    sample.to_sample_frame(vectors)         -> lab vectors -> sample frame
    sample.to_lab_frame(vectors)            -> sample frame -> lab vectors
    sample.with_plane_angle(angle)          -> same sample, other orientation
    sample.beam_fullness(beam_height)       -> fraction of the beam covered by each layer

Typical use
-----------
    sample = create_slab_sample(
        width=20, height=20,
        layers=[("Al", 2.0, 0.020),          # (name, thickness mm, mu 1/mm,
                ("water", 3.0, 0.560, 1.0, 12.0),  # [scatter fraction, fill height mm])
                # or, with an explicit band (version 2.2):
                # Layer("water", 3.0, 0.560, 1.0, bottom=-6.0, height=9.0)
                ("Al", 2.0, 0.020)],
        plane_angle_deg=60)                  # angle default: 90
    print(sample.summary())

NumPy functions used in this file (short glossary)
--------------------------------------------------
np.array / np.asarray : build an array / convert to float array without copying if possible.
np.atleast_2d(a)      : make a (3,) vector into a (1, 3) table, leave (N, 3) as it is.
np.broadcast_to(a, shape) : repeat `a` virtually to a bigger shape (no memory copy).
np.cumsum(a, axis)    : running sum along an axis: [1,2,3] -> [1,3,6].
np.concatenate        : join arrays one after the other.
np.clip(a, lo, hi)    : limit values to [lo, hi].
np.where(cond, a, b)  : element by element, take `a` where cond is True, else `b`.
np.searchsorted(edges, v) : for each v, the position where it would be inserted in
                        the sorted array `edges` (a fast way to find in which interval it falls).
np.argsort(a, axis=1) : for every row, the order of the columns that sorts it.
np.take_along_axis(a, idx, axis=1) : reorder every row of `a` with the indices `idx`.
np.minimum / np.maximum : element by element smaller / larger of two arrays.
np.log, np.exp        : natural logarithm / exponential.
np.linalg.norm(v, axis=1) : length of each row vector.
np.errstate(...)      : temporarily silence warnings (here: divisions by zero that we
                        handle ourselves with np.where).
np.column_stack       : put 1D arrays side by side as the columns of a 2D table.
np.inf                : infinity; an interval (-inf, inf) means "always".
a[:, None]            : add an axis to a 1D array, (N,) -> (N, 1), needed to combine
                        it with a (1, L) array into an (N, L) table ("broadcasting").
"""

import numpy as np

# Layer, scattered_direction and the abstract base class come from
# sample_base.py. Layer and scattered_direction are imported here as well, so
# they can also be imported from this module: "from slab import Layer".
from .sample_base import (Layer, LayeredSample, scattered_direction,  # noqa: F401
                         make_layer, GEOMETRY_TOL)


# --------------------------------------------------------------------------
# A small helper: the interval of t during which a ray is inside a "slice"
# --------------------------------------------------------------------------
def _slice_interval(p, d, lo, hi):
    """Distances t at which rays p + t*d are between two parallel planes.

    p, d   : (N,) or (N, 1) arrays, one coordinate of the start point and of
             the direction (e.g. the z' coordinate).
    lo, hi : limits of the slice in that coordinate (numbers or arrays that
             broadcast with p).

    Returns (t_lo, t_hi). If the ray does not move in that coordinate
    (d = 0) it is inside for ever (-inf, inf) when lo <= p <= hi, and never
    otherwise (coded as the empty interval (inf, -inf)).
    """
    moving = d != 0.0
    safe_d = np.where(moving, d, 1.0)            # avoid dividing by zero
    t_a = (lo - p) / safe_d                      # reaches the plane "lo"
    t_b = (hi - p) / safe_d                      # reaches the plane "hi"
    inside = (p >= lo - GEOMETRY_TOL) & (p <= hi + GEOMETRY_TOL)
    t_lo = np.where(moving, np.minimum(t_a, t_b),
                    np.where(inside, -np.inf, np.inf))
    t_hi = np.where(moving, np.maximum(t_a, t_b),
                    np.where(inside, np.inf, -np.inf))
    return t_lo, t_hi


# --------------------------------------------------------------------------
# The slab sample: layers + geometry + the calculations we will need
# --------------------------------------------------------------------------
class SlabSample(LayeredSample):
    """Rectangular multilayer slab sample. Create it with `create_slab_sample`."""

    # Number of particles processed at once (limits the memory used).
    _CHUNK = 500_000

    def __init__(self, width, height, layers, z_front=0.0,
                 plane_angle_deg=90.0, y_bottom=None, beam_height=None):
        # Common data (layers, thicknesses, mus, scatter fractions, vertical
        # extent and filling of the layers) are stored by the base class.
        super().__init__(layers, height, y_bottom, beam_height)

        # --- Check the inputs ---------------------------------------------
        if width <= 0:
            raise ValueError("Sample width must be positive (mm).")
        if not 0.0 < plane_angle_deg < 180.0:
            raise ValueError("plane_angle_deg must be between 0 and 180.")

        self.width = float(width)
        self.z_front = float(z_front)
        self.plane_angle_deg = float(plane_angle_deg)

        # --- Layer boundaries along z' ---------------------------------------
        # N layers -> N+1 boundaries. np.cumsum adds the thicknesses one after
        # the other: [2, 1, 0.5] -> [2, 3, 3.5]; we put a 0 in front.
        self.boundaries = self.z_front + np.concatenate(
            ([0.0], np.cumsum(self.thicknesses)))
        self.z_back = float(self.boundaries[-1])

        # Optical depth (integral of mu dz') at each boundary, measured from
        # the front face, for a path along the layer normal (full layers).
        self._tau_edges = np.concatenate(
            ([0.0], np.cumsum(self.mus * self.thicknesses)))

        # Corners of the box of the sample (sample frame).
        self._lo = np.array([-self.width / 2, self.y_bottom, self.z_front])
        self._hi = np.array([self.width / 2, self.y_top, self.z_back])

        # --- Orientation --------------------------------------------------
        # tilt = rotation of the sample normal away from the beam direction.
        tilt = np.radians(90.0 - self.plane_angle_deg)
        self._cos = float(np.cos(tilt))      # cos of angle between beam and normal
        self._sin = float(np.sin(tilt))
        if self._cos < 1e-6:
            raise ValueError("Beam parallel to the sample plane: not allowed.")
        # Unit vectors x', y', z' of the sample frame written in the lab
        # frame (one per ROW). For tilt = 0 this is the identity matrix.
        self._axes = np.array([[self._cos, 0.0, -self._sin],
                               [0.0,       1.0,  0.0],
                               [self._sin, 0.0,  self._cos]])

        # A distance (mm) larger than the size of the sample seen from the
        # origin: beam particles start that far upstream (see
        # sample_interaction_points).
        self._far = float(np.linalg.norm(np.maximum(np.abs(self._lo),
                                                    np.abs(self._hi)))) + 1.0

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------
    @property
    def total_thickness(self):
        """Total thickness of the stack along the normal (mm)."""
        return self.z_back - self.z_front

    def summary(self):
        """Return a small text table describing the sample.

        The column "fill" gives the height of the material above the bottom
        of the sample (mm), or "full"; when the beam height is known, the
        fullness (fraction of the beam covered) is shown in parentheses.
        """
        lines = [f"Slab sample {self.width:g} x {self.height:g} mm "
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
        """Return a copy of this sample with another orientation.

        Handy to scan the angle:  [sample.with_plane_angle(a) for a in ...]
        """
        return SlabSample(self.width, self.height, self.layers, self.z_front,
                          plane_angle_deg, self.y_bottom, self.beam_height)
        # (DiscSample, in disc.py, has its own version of this method.)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def to_sample_frame(self, vectors):
        """Convert vectors (e.g. directions) from the lab frame to the sample
        frame. Accepts one vector (3,) or an array (N, 3); returns (N, 3).

        Use it for the detector direction before calling
        `outgoing_path_lengths`. Because the two frames have the same origin,
        it converts POSITIONS as well.
        """
        # The coordinates of a vector v in the sample frame are its dot
        # products with the sample axes: v' = (v.x', v.y', v.z').
        # For a table of vectors (one per row) that is `vectors @ axes.T`
        # (".T" = transposed matrix: rows and columns exchanged).
        return np.atleast_2d(np.asarray(vectors, dtype=float)) @ self._axes.T

    def to_lab_frame(self, vectors):
        """Inverse of `to_sample_frame`: sample-frame vectors -> lab frame.
        Accepts one vector (3,) or an array (N, 3); returns (N, 3)."""
        # The axes form an orthogonal (rotation) matrix, so the inverse
        # rotation is simply the multiplication by the matrix itself.
        return np.atleast_2d(np.asarray(vectors, dtype=float)) @ self._axes

    # ------------------------------------------------------------------
    # Where is a point? (sample frame)
    # ------------------------------------------------------------------
    def contains(self, points, tol=1e-9):
        """True for every point (x', y', z') that lies inside the box of the
        sample (the empty parts of partially filled layers included).

        `tol` is a tiny tolerance (mm) so that points lying exactly on a face,
        but displaced by floating-point rounding errors, still count as inside.
        """
        p = np.atleast_2d(points)
        # All the conditions must hold; np.all(..., axis=1) checks the three
        # coordinates of each point at once.
        return np.all((p >= self._lo - tol) & (p <= self._hi + tol), axis=1)

    def layer_index(self, z):
        """Index of the layer containing depth z' (-1 if outside the sample).

        Works for a single number or an array of z' values. Only the depth
        is considered (not the filling): see `layer_of_points`.
        """
        z = np.asarray(z, dtype=float)
        # np.searchsorted finds, for each z, where it would be inserted in the
        # sorted list of boundaries. With side="right", a z lying inside layer
        # i gives i+1, so we subtract 1 to get i.
        idx = np.searchsorted(self.boundaries, z, side="right") - 1
        # A point exactly on the back face would give an index one too large:
        # we put it in the last layer.
        idx = np.where(z == self.z_back, self.n_layers - 1, idx)
        # Points in front of the front face or behind the back face: -1.
        outside = (z < self.z_front) | (z > self.z_back)
        return np.where(outside, -1, idx)

    def layer_of_points(self, points):
        """Layer index of each point (x', y', z') of an (N, 3) array.

        The layer follows from the depth z' (the layers are perpendicular to
        z'). The result is -1 for points outside the sample AND for points in
        the empty space below or above the material of a partially filled layer.
        """
        p = np.atleast_2d(points)
        idx = self.layer_index(p[:, 2])
        inside = self.contains(p, tol=GEOMETRY_TOL)
        filled = self.is_filled(np.maximum(idx, 0), p[:, 1])
        return np.where(inside & (idx >= 0) & filled, idx, -1)

    # ------------------------------------------------------------------
    # The heart of the geometry: where is the ray inside the material?
    # ------------------------------------------------------------------
    def _section_interval(self, p, d):
        """Interval of t during which rays p + t*d are inside the OUTLINE of
        the sample (all layers together, empty parts included).

        Slab: the box |x'| <= w/2, y_bottom <= y' <= y_top, z_front <= z' <=
        z_back. For each coordinate the ray is between the two faces during an
        interval of t; it is inside the box when it is inside the three
        slices at the same time: the intersection (largest start, smallest
        end). Returns (t_in, t_out), two (N,) arrays (t_in > t_out = missed).

        This is the only place where the shape of the section enters the
        path calculation: the disc geometry (disc.py) replaces this method.
        """
        t_lo, t_hi = _slice_interval(p, d, self._lo, self._hi)        # (N, 3)
        return t_lo.max(axis=1), t_hi.min(axis=1)

    def _clip_inside(self, points):
        """Push points that rounding errors put just outside the sample back
        onto its surface (here: limit each coordinate to the box)."""
        return np.clip(points, self._lo, self._hi)

    def _material_intervals(self, points, directions):
        """For rays p + t*d (t >= 0), the interval [t0_i, t1_i] spent in the
        MATERIAL of each layer i.

        The start points may be inside the sample or outside it (e.g.
        upstream of it); directions must be unit vectors. Both in the SAMPLE
        frame, shape (N, 3).

        Returns (t0, t1), two (N, n_layers) arrays with t1 >= t0; a layer that
        is not crossed has t0 = t1 (zero length).
        """
        p, d = points, directions

        # --- Step 1: part of the ray inside the sample [t_in, t_out] -------
        # (the box of a slab, or the cylinder of a disc: see
        # `_section_interval`). We only look forward: t >= 0.
        t_in, t_out = self._section_interval(p, d)                     # (N,), (N,)
        t_in = np.maximum(t_in, 0.0)

        # --- Step 2: interval inside the slice of each layer ---------------
        # Same idea with the planes z' = boundaries[i] and boundaries[i+1].
        # p[:, 2:3] keeps a column (N, 1) that broadcasts with the (L,) limits.
        t_za, t_zb = _slice_interval(p[:, 2:3], d[:, 2:3],
                                     self.boundaries[:-1], self.boundaries[1:])
        # A ray parallel to the layers (d_z' = 0) that starts exactly on a
        # boundary would be "inside" two layers; keep only the layer given by
        # layer_index, as in the full-layer version.
        flat = d[:, 2] == 0.0
        if flat.any():
            own = (self.layer_index(p[flat, 2])[:, None]
                   == np.arange(self.n_layers)[None, :])
            t_za[flat] = np.where(own, -np.inf, np.inf)
            t_zb[flat] = np.where(own, np.inf, -np.inf)

        # --- Step 3: interval inside the band of the material ---------------
        # Region fill_bottoms[i] <= y' <= fill_tops[i]: one more "slice",
        # between the bottom and the top of the material of each layer.
        t_fa, t_fb = _slice_interval(p[:, 1:2], d[:, 1:2],
                                     self.fill_bottoms[None, :],
                                     self.fill_tops[None, :])

        # --- Step 4: intersection of the three intervals --------------------
        t0 = np.maximum(np.maximum(t_in[:, None], t_za), t_fa)
        t1 = np.minimum(np.minimum(t_out[:, None], t_zb), t_fb)
        t1 = np.maximum(t1, t0)                      # empty -> zero length
        # Non-crossed layers can have infinite limits: put them at 0 length 0.
        empty = ~np.isfinite(t0) | (t1 <= t0)
        t0 = np.where(empty, 0.0, t0)
        t1 = np.where(empty, 0.0, t1)
        return t0, t1

    @staticmethod
    def _unit(points, directions):
        """Points and unit directions as (N, 3) float arrays (one direction
        is repeated for all the points)."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        d = np.atleast_2d(np.asarray(directions, dtype=float))
        # One direction for all the points is allowed: np.broadcast_to
        # repeats it N times (without copying memory) to match the points.
        d = np.broadcast_to(d, p.shape).astype(float)
        # Normalise: divide every row by its length (np.linalg.norm with
        # axis=1 computes the length of each row; keepdims keeps it as a column).
        return p, d / np.linalg.norm(d, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Path lengths (sample frame)
    # ------------------------------------------------------------------
    def outgoing_path_lengths(self, points, directions):
        """Path travelled in the material of each layer by a particle that
        starts at each point and flies along `directions` until it leaves the
        sample (through the front, back or any side face, including the top).
        The empty space below or above the material is not counted (it
        does not attenuate).

        points     : (N, 3) array or one point, in the SAMPLE frame
        directions : (N, 3) array or one vector, in the SAMPLE frame (use
                     `to_sample_frame` on lab directions); need not be
                     normalised. See `scattered_direction`.

        Returns an array of shape (N, n_layers) in mm.
        """
        p = self._check_points(points)
        p, d = self._unit(p, directions)
        t0, t1 = self._material_intervals(p, d)
        return t1 - t0

    def incident_path_lengths(self, points):
        """Path travelled in the material of each layer by the incident beam
        (direction +z of the lab frame) from where it enters the sample up
        to each point (sample frame), in mm, shape (N, n_layers).

        It is computed by following the beam BACKWARDS from the point until
        it leaves the sample, so it is correct also for particles entering
        through a side face or through the empty part of a layer. It should
        not be combined with the interaction points of
        `sample_interaction_points` to compute a transmission: those already
        include the attenuation of the incident beam.
        """
        back = -self.to_sample_frame([0.0, 0.0, 1.0])
        return self.outgoing_path_lengths(points, back)

    # ------------------------------------------------------------------
    # Where does the particle interact?
    # ------------------------------------------------------------------
    def _first_interaction(self, points, directions, rng):
        """Find the first interaction of particles flying from `points` along
        `directions` (SAMPLE frame; the points may be inside the sample or
        upstream of it).

        Returns (new_points (N, 3), interacts (N,) boolean).
        """
        p_all, d_all = self._unit(points, directions)
        new_points = p_all.copy()
        interacts = np.zeros(len(p_all), dtype=bool)

        # Process the particles in chunks to limit the memory used.
        for i0 in range(0, len(p_all), self._CHUNK):
            sl = slice(i0, i0 + self._CHUNK)          # this chunk of particles
            p, d = p_all[sl], d_all[sl]
            n = len(p)

            # --- Step 1: material intervals, in the order met by the ray ---
            t0, t1 = self._material_intervals(p, d)
            order = np.argsort(t0, axis=1)            # sort the layers by t0
            t0 = np.take_along_axis(t0, order, axis=1)
            seg = np.take_along_axis(t1, order, axis=1) - t0   # lengths
            mu = self.mus[order]                      # mu of each sorted piece

            # --- Step 2: optical depth of each piece and running total -----
            tau_seg = seg * mu
            cum = np.cumsum(tau_seg, axis=1)

            # --- Step 3: random optical depth to the next interaction ------
            # Beer-Lambert: the optical depth at which a particle interacts is
            # exponentially distributed: tau = -ln(1-u), u uniform in [0,1).
            # If it is larger than the optical depth to the exit, the particle
            # leaves the sample without interacting.
            tau = -np.log(1.0 - rng.uniform(size=n))
            hit = tau < cum[:, -1]

            # --- Step 4: piece in which tau is reached ---------------------
            # Counting the pieces whose cumulative optical depth is still
            # <= tau gives the index of the piece in which tau is reached.
            k = np.minimum((cum <= tau[:, None]).sum(axis=1), self.n_layers - 1)
            rows = np.arange(n)      # 0, 1, ..., n-1: "fancy indexing" picks one element per row
            with np.errstate(divide="ignore", invalid="ignore"):
                extra = (tau - (cum[rows, k] - tau_seg[rows, k])) / mu[rows, k]
            # Distance = start of that piece + the extra distance inside it.
            dist = np.where(hit, t0[rows, k] + extra, 0.0)

            # --- Step 5: the new position ----------------------------------
            new_points[sl] = p + d * dist[:, None]
            interacts[sl] = hit
        return new_points, interacts

    def _depth_from_tau(self, tau_normal):
        """Convert an optical depth measured along the layer normal into the
        depth z' where it is reached. NaN if it lies beyond the back face.
        (Full layers, infinite section: used by `sample_interaction_depth`.)

        Idea: the optical depth grows piece-wise linearly with z' (slope = mu
        of the layer). We find the layer where the wanted value is reached and
        invert the straight line inside it.
        """
        tau_edges = self._tau_edges
        # The particle interacts only if its random optical depth is smaller
        # than the total optical depth of the whole sample.
        interacts = tau_normal < tau_edges[-1]
        # Layer where tau is reached (same searchsorted trick as layer_index).
        idx = np.clip(np.searchsorted(tau_edges, tau_normal, side="right") - 1,
                      0, self.n_layers - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.boundaries[idx] + (tau_normal - tau_edges[idx]) / self.mus[idx]
        return np.where(interacts, z, np.nan)    # NaN = no interaction

    def sample_interaction_depth(self, n=1, rng=None):
        """Draw the depth z' (mm) at which `n` beam particles entering through
        the front face interact (Beer-Lambert law, oblique incidence taken
        into account). NaN = crossed the sample without interacting.

        This is a quick estimate that ignores the finite width and height of
        the sample and the filling of the layers (all taken as full); use
        `sample_interaction_points` for the complete calculation.
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        # Inverse-transform sampling: if u is uniform in [0,1), then
        # tau = -ln(1-u) is exponential. Along the layer normal the optical
        # depth is smaller by the factor cos(angle between beam and normal).
        tau = -np.log(1.0 - rng.uniform(size=n))
        return self._depth_from_tau(tau * self._cos)

    def sample_interaction_points(self, x, y, rng=None):
        """Find where beam particles interact inside the sample.

        The particles travel along +z in the lab frame. They are described by
        their position (x, y) in a plane perpendicular to the beam that goes
        through the rotation axis (the plane z = 0 of the lab frame): these
        are exactly the (x, y) given by `sample_beam_point`.

        The calculation takes into account that
          * the beam is oblique to the sample plane (longer path in the
            material, footprint stretched by 1/cos(tilt)),
          * a particle can enter through a side face, or leave through it
            before interacting, since the sample has a finite section,
          * a particle can pass OVER or UNDER the material of a partially filled
            layer (through its empty part) and interact in the next layer,
          * particles missing the sample or crossing it without interacting
            give no interaction point.

        Returns
        -------
        points : (M, 3) array
            Interaction points in the SAMPLE frame (x', y', z').
        interacted : (N,) boolean array
            True for the particles that interacted (the M rows of `points`).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # The particles start far upstream, outside the sample, on the line
        # of the beam (lab frame: (x, y, -far), direction +z), and fly into
        # it. The two frames share the origin, so the start points are
        # converted with the same rotation as the directions.
        start_lab = np.column_stack([x, y, np.full(len(x), -self._far)])
        start = self.to_sample_frame(start_lab)
        direction = self.to_sample_frame([0.0, 0.0, 1.0])
        new_points, interacted = self._first_interaction(start, direction, rng)
        return self._clip_inside(new_points[interacted]), interacted

    def next_interaction(self, points, directions, rng=None):
        """Follow particles that fly from `points` along `directions`
        (both in the SAMPLE frame, inside the sample) and find where each one
        has its next interaction, if it has one before leaving the sample.

        This is what multiple scattering needs: after a scattering event the
        particle flies in a new direction through layers of different mu,
        possibly through the empty part of partially filled layers.

        Returns
        -------
        new_points : (N, 3) array
            Position of the next interaction (meaningless where the particle
            leaves the sample).
        interacts : (N,) boolean array
            True if the particle interacts before leaving the sample; False
            if it leaves the sample (then its direction is its final one).
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        new_points, interacts = self._first_interaction(points, directions, rng)
        # Keep the points inside the sample despite rounding errors.
        return self._clip_inside(new_points), interacts

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
# Public helper to build a slab sample
# --------------------------------------------------------------------------
def create_slab_sample(width, height, layers, z_front=0.0,
                       plane_angle_deg=90.0, y_bottom=None, beam_height=None):
    """Build a layered rectangular (slab) sample.

    Parameters
    ----------
    width, height : float
        Section of the sample in its own plane (mm): width horizontal,
        height vertical. height may be None (version 2.2): the smallest
        range containing the material of the layers given with `bottom` and
        `height`.
    layers : list
        Layers in the order the beam meets them. Each item is a `Layer`, a
        tuple (name, thickness_mm, mu_per_mm[, scatter_fraction[,
        fill_height_mm]]) or a dict with the fields of `Layer` (use a dict or
        a Layer to give `bottom` and `height`, or the `fullness`).
    z_front : float
        Position of the front face along the layer normal (mm). Default 0.
    plane_angle_deg : float
        Angle between the incident beam and the plane of the sample, in
        degrees (0 < angle < 180). 90 (default) = perpendicular to the beam.
        The sample is rotated around the vertical axis (see module docstring).
    y_bottom : float, optional
        Height of the bottom of the sample (mm, lab y, 0 = beam centre).
        Default: centred on the beam (-height/2), or the lowest bottom of the
        layers when height is None. Layers without `bottom` start here.
    beam_height : float, optional
        Height of the beam (mm): needed if some layer is given by its
        fullness; also shown in the summary.

    Returns
    -------
    SlabSample
    """
    built = [make_layer(l) for l in layers]
    return SlabSample(width, height, built, z_front, plane_angle_deg,
                      y_bottom, beam_height)


# --------------------------------------------------------------------------
# Demo:  python slab.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from .beam import sample_beam_point            # the beam function

    # A flat container 20 mm high whose bottom is 6 mm below the centre of
    # an 8 mm beam (the beam goes from -4 to +4 mm): two Al walls and water
    # from the bottom of the cell (y = -6) up to y = 0, i.e. 6 mm of water
    # that cover the lower half of the beam (fullness 0.5).
    beam_w, beam_h = 8.0, 8.0
    sample = create_slab_sample(
        20, 20, [("Al", 1.0, 0.010),
                 {"name": "water", "thickness": 3.0, "mu": 0.56, "bottom": -6.0, "height": 6.0},
                 ("Al", 1.0, 0.010)],
        plane_angle_deg=60, y_bottom=-6.0, beam_height=beam_h)
    print(sample.summary())

    # 1) launch particles with the beam function, 2) find where they interact
    n = 100000
    x, y = sample_beam_point(beam_w, beam_h, n=n, rng=1)
    points, hit = sample.sample_interaction_points(x, y, rng=2)
    layer = sample.layer_of_points(points)
    # np.bincount counts how many times each layer index appears.
    print("\nFraction interacting per layer:",
          np.round(np.bincount(layer, minlength=sample.n_layers) / n, 4))
    print("Fraction crossing without interaction (or missing):",
          round(1 - hit.mean(), 4))

    # 3) path inside the sample towards a detector at 2theta = 60 deg.
    # The detector direction is given in the lab frame: convert it first.
    d = sample.to_sample_frame(scattered_direction(60.0))
    L_out = sample.outgoing_path_lengths(points, d)
    # The interaction points already include the attenuation of the incident
    # beam, so only the outgoing path is attenuated here.
    survive = sample.transmission(L_out)
    print("Mean survival probability to the detector (2theta=60 deg):",
          round(survive.mean(), 4))
