"""
multiple_scattering.py - Count the particles that leave the sample after
exactly one scattering (single) or two or more (multiple), as a function of
the scattering angle 2theta. It works with ANY sample geometry (slab or
cylinder), because it only uses the common interface of sample_base.py.

How it works (analog Monte Carlo, particle by particle)
-------------------------------------------------------
1. Each particle starts in the beam (beam.py) and flies to its first
   interaction point in the sample.
2. At every interaction the particle is either
       - scattered (probability = `scatter_fraction` of the layer), or
       - absorbed (it is lost).
3. A scattered particle gets a new direction (angular distribution below),
   flies on through the layers and either interacts again (next scattering
   or absorption) or leaves the sample.
4. When a scattered particle leaves the sample, it is counted in the bin of
   its final scattering angle 2theta = angle between its final direction and
   the incident beam direction (+z), together with its number of scatterings.

Particles that never interact (they cross the sample or miss it) are not
"scattered" and are not counted in the angular distributions.

Choices you can change
----------------------
* Angular distribution of ONE scattering event: `deflection_sampler`, a
  function (n, rng) -> array of n values of cos(chi), where chi is the
  deflection angle relative to the direction before the event. Default:
  isotropic. `henyey_greenstein(g)` gives a forward (g > 0) or backward
  (g < 0) peaked example. The same distribution is used in all layers.
* Detector window: all azimuths by default; give `phi_deg` and
  `phi_halfwidth_deg` to accept only a vertical strip around one azimuth
  (azimuth measured from +x towards +y in the lab frame). For a 2-axis
  instrument (detector in the horizontal plane), use phi_deg=0 and a small
  half-width.
* Absorption: `scatter_fraction` of each layer.

Usage
-----
    python multiple_scattering.py        # runs the example at the bottom

or, from your own code:
    res = simulate_scattering_orders(sample, 8, 8, n_particles=200000)
    res.angles_deg, res.single, res.multiple, res.total    # counts per bin
    plot_scattering_orders(res)

NumPy functions used in this file (short glossary)
--------------------------------------------------
rng.uniform(a, b, size=n) : n random numbers uniformly distributed in [a, b).
np.sqrt, np.cos, np.sin   : element by element; np.clip(a, lo, hi) limits values.
np.cross(a, b)            : vector (cross) product of the vectors in the rows.
np.linalg.norm(v, axis=1) : length of each row vector.
np.where(cond, a, b)      : take `a` where cond is True, else `b`.
np.tile(a, (n, 1))        : repeat the row `a` n times to make an (n, 3) table.
np.concatenate            : join arrays (here the exit directions of every round).
np.arccos, np.arctan2     : inverse cosine / angle of a point (x, y) from the x axis.
np.degrees / np.radians   : convert radians <-> degrees.
np.histogram(a, bins=edges) : count how many values of `a` fall in each bin
                            (returns counts and edges; we keep the counts).
np.zeros(shape)           : array of zeros.
boolean masks             : a[mask] keeps the elements where mask is True; `~mask`
                            is "not mask".
"""

from dataclasses import dataclass

import numpy as np

from .beam import sample_beam_point
from .slab import create_slab_sample


# --------------------------------------------------------------------------
# Angular distribution of one scattering event
# --------------------------------------------------------------------------
def isotropic_cos(n, rng):
    """Isotropic scattering: cos(chi) uniform between -1 and 1.
    (Uniform in cos(chi), not in chi, gives equal probability per unit of
    solid angle.)"""
    return rng.uniform(-1.0, 1.0, size=n)


def henyey_greenstein(g):
    """Return a deflection sampler with the Henyey-Greenstein distribution.

    g = 0 isotropic, g > 0 forward peaked, g < 0 backward peaked
    (g is the mean cosine of the deflection angle). It is a standard,
    one-parameter model of anisotropic scattering.
    """
    def sampler(n, rng):
        u = rng.uniform(size=n)
        if abs(g) < 1e-6:
            return 1.0 - 2.0 * u
        # Inverse of the cumulative distribution of the HG function.
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
        return np.clip((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0)
    return sampler


def rotate_directions(d, cos_chi, rng):
    """New directions that make an angle chi (cos_chi given) with the unit
    vectors `d` (shape (N, 3)), with a random azimuth around `d`.

    Idea: build two unit vectors u and v perpendicular to d (and to each
    other). Then the new direction is
        cos(chi) d + sin(chi) (cos(psi) u + sin(psi) v)
    where psi is a random angle between 0 and 2 pi.
    """
    psi = rng.uniform(0.0, 2.0 * np.pi, size=len(d))        # random azimuth around d
    sin_chi = np.sqrt(np.clip(1.0 - cos_chi**2, 0.0, None))  # clip avoids sqrt of -1e-17

    # A helper vector that is not parallel to d: the z axis, unless d is
    # nearly along z, then the x axis.
    helper = np.where((np.abs(d[:, 2]) < 0.9)[:, None],
                      np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    # The cross product d x helper is perpendicular to both: normalise it -> u.
    u = np.cross(d, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # v perpendicular to d and u.
    v = np.cross(d, u)

    # [:, None] turns the 1D arrays into columns so that they multiply each
    # row (vector) of d, u and v.
    new = (cos_chi[:, None] * d
           + sin_chi[:, None] * (np.cos(psi)[:, None] * u
                                 + np.sin(psi)[:, None] * v))
    return new / np.linalg.norm(new, axis=1, keepdims=True)   # remove rounding errors


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class ScatteringResult:
    """Result of `simulate_scattering_orders`.

    angle_edges_deg : bin edges of the scattering angle 2theta (degrees)
    angles_deg      : bin centres
    solid_angle     : solid angle (sr) covered by each bin and detector window
    counts_by_order : array (K, n_bins); row k-1 = particles scattered
                      exactly k times that left the sample in each bin
    n_incident      : number of incident particles
    n_no_interaction: particles that crossed the sample or missed it
    n_absorbed      : particles absorbed in the sample
    n_exited        : scattered particles that left the sample (all angles
                      and azimuths, also outside the detector window)
    n_unfinished    : particles still inside after `max_orders` events
                      (should be 0)
    """
    angle_edges_deg: np.ndarray
    angles_deg: np.ndarray
    solid_angle: np.ndarray
    counts_by_order: np.ndarray
    n_incident: int
    n_no_interaction: int
    n_absorbed: int
    n_exited: int
    n_unfinished: int

    @property
    def single(self):
        """Counts per bin of particles scattered exactly once."""
        return self.counts_by_order[0]

    @property
    def multiple(self):
        """Counts per bin of particles scattered two or more times."""
        return self.counts_by_order[1:].sum(axis=0)

    @property
    def total(self):
        """Counts per bin of all scattered particles (single + multiple)."""
        return self.counts_by_order.sum(axis=0)

    def per_steradian(self, counts):
        """Convert counts per bin into particles per incident particle per
        steradian, which does not depend on the bin width."""
        return counts / (self.n_incident * self.solid_angle)


# --------------------------------------------------------------------------
# Main simulation
# --------------------------------------------------------------------------
def simulate_scattering_orders(sample, beam_width, beam_height, H=None, V=None,
                               n_particles=200000, angle_edges_deg=None,
                               phi_deg=0.0, phi_halfwidth_deg=None,
                               deflection_sampler=None, max_orders=200,
                               rng=None):
    """
    Simulate particles through the sample and count them by scattering angle
    and by number of scatterings (single / multiple).

    Parameters
    ----------
    sample : LayeredSample
        Slab or cylindrical sample (see sample_factory.make_sample);
        geometry and `scatter_fraction` of the layers are taken into account.
    beam_width, beam_height : float
        Beam size (mm); H, V optional beam profiles (see beam.py).
    n_particles : int
        Number of incident particles. Multiple scattering is rare, so use
        many particles (10^5 to 10^6) for a smooth curve.
    angle_edges_deg : array-like, optional
        Bin edges of 2theta in degrees. Default: 90 bins of 2 degrees, 0-180.
    phi_deg, phi_halfwidth_deg : float, optional
        Detector window in azimuth: only particles leaving with an azimuth
        within +-phi_halfwidth_deg of phi_deg are counted. Default None: all
        azimuths.
    deflection_sampler : callable, optional
        (n, rng) -> cos(chi) values; default isotropic (see above).
    max_orders : int
        Safety limit on the number of scatterings followed per particle.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible results.

    Returns
    -------
    ScatteringResult
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    sampler = deflection_sampler or isotropic_cos       # `or`: use the default if None
    if angle_edges_deg is None:
        angle_edges_deg = np.linspace(0.0, 180.0, 91)
    edges = np.asarray(angle_edges_deg, dtype=float)

    # --- Step 1: start points in the beam and first interaction points -------
    x, y = sample_beam_point(beam_width, beam_height, H, V,
                             n=max(n_particles, 2), rng=rng)
    n_incident = len(x)
    pos, _ = sample.sample_interaction_points(x, y, rng=rng)
    n_no_interaction = n_incident - len(pos)

    # State of the particles still inside the sample, all in the SAMPLE frame:
    #   pos   : position of their current interaction
    #   dirn  : direction they had when arriving at that interaction
    #   n_sc  : number of scatterings suffered BEFORE this interaction
    # They all arrive travelling along +z of the lab frame; we convert that
    # direction to the sample frame and repeat it for every particle.
    dirn = np.tile(sample.to_sample_frame([0.0, 0.0, 1.0]), (len(pos), 1))
    n_sc = np.zeros(len(pos), dtype=int)

    exit_dirs, exit_orders = [], []     # final direction and order of exits
    n_absorbed = 0

    # --- The loop: one round = one scattering event for every particle that
    #     is still inside the sample --------------------------------------
    for _ in range(max_orders):
        if len(pos) == 0:               # nobody left inside: finished
            break

        # Step 2: at each interaction, scattering or absorption?
        # The probability depends on the layer where the interaction happens.
        layer = sample.layer_of_points(pos)
        scatters = rng.uniform(size=len(pos)) < sample.scatter_fractions[layer]
        n_absorbed += int((~scatters).sum())
        # Keep only the scattered particles (boolean mask) and count the event.
        pos, dirn, n_sc = pos[scatters], dirn[scatters], n_sc[scatters] + 1
        if len(pos) == 0:
            break

        # Step 3: new direction after the scattering event.
        dirn = rotate_directions(dirn, sampler(len(pos), rng), rng)

        # Step 4: next interaction inside the sample, or exit.
        new_pos, interacts = sample.next_interaction(pos, dirn, rng=rng)
        # Particles that do NOT interact again leave the sample: store their
        # final direction and their number of scatterings.
        exit_dirs.append(dirn[~interacts])
        exit_orders.append(n_sc[~interacts])
        # The others go on to the next round from their new interaction point.
        pos, dirn, n_sc = new_pos[interacts], dirn[interacts], n_sc[interacts]

    n_unfinished = len(pos)             # should be 0 (see max_orders)

    # --- Step 5: final directions -> angle 2theta and azimuth (LAB frame) ----
    if exit_dirs:
        # np.concatenate joins the lists of all the rounds into single arrays.
        d_lab = sample.to_lab_frame(np.concatenate(exit_dirs))
        orders = np.concatenate(exit_orders)
    else:
        d_lab, orders = np.empty((0, 3)), np.empty(0, dtype=int)
    # 2theta = angle between the final direction and +z: cos(2theta) = d_z.
    two_theta = np.degrees(np.arccos(np.clip(d_lab[:, 2], -1.0, 1.0)))
    # Azimuth around the beam, measured from +x towards +y.
    phi = np.degrees(np.arctan2(d_lab[:, 1], d_lab[:, 0]))

    # Detector window in azimuth.
    if phi_halfwidth_deg is None:
        in_window = np.ones(len(phi), dtype=bool)
        d_phi = 2.0 * np.pi                        # full circle
    else:
        # Difference of azimuths wrapped into [-180, 180) degrees.
        delta = (phi - phi_deg + 180.0) % 360.0 - 180.0
        in_window = np.abs(delta) <= phi_halfwidth_deg
        d_phi = min(2.0 * np.radians(phi_halfwidth_deg), 2.0 * np.pi)

    # --- Step 6: histogram, one row per number of scatterings ---------------
    # (at least 2 rows, so that "single" and "multiple" always exist).
    max_order = max(2, int(orders.max()) if len(orders) else 2)
    counts = np.zeros((max_order, len(edges) - 1))
    for k in range(1, max_order + 1):
        sel = in_window & (orders == k)            # particles scattered exactly k times
        # np.histogram returns (counts per bin, edges); we keep the counts [0].
        counts[k - 1] = np.histogram(two_theta[sel], bins=edges)[0]

    # Solid angle of each bin: the band between polar angles a and b, over an
    # azimuth range d_phi, has solid angle d_phi * (cos a - cos b).
    solid_angle = d_phi * (np.cos(np.radians(edges[:-1]))
                           - np.cos(np.radians(edges[1:])))
    return ScatteringResult(
        angle_edges_deg=edges,
        angles_deg=0.5 * (edges[:-1] + edges[1:]),
        solid_angle=solid_angle,
        counts_by_order=counts,
        n_incident=n_incident,
        n_no_interaction=n_no_interaction,
        n_absorbed=n_absorbed,
        n_exited=int(len(orders)),
        n_unfinished=n_unfinished)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_scattering_orders(result, per_steradian=True, logy=False, title=None,
                           save_as=None, show=True):
    """
    Plot total, single and multiple scattering as a function of 2theta.

    per_steradian : True  -> particles per incident particle per steradian
                             (independent of the bin width, flat for
                             isotropic scattering without attenuation)
                    False -> raw counts per bin
    logy          : logarithmic vertical axis (useful: multiple scattering
                    is usually much weaker than single scattering)
    save_as       : file name to save the figure, or None.
    show          : open the plot window (False to only save).
    """
    import matplotlib.pyplot as plt

    curves = {"total": result.total, "single scattering": result.single,
              "multiple scattering (2 or more)": result.multiple}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, counts in curves.items():
        values = result.per_steradian(counts) if per_steradian else counts
        if logy:                               # avoid log(0): zeros become NaN (not drawn)
            values = np.where(values > 0, values, np.nan)
        ax.plot(result.angles_deg, values, marker="o", ms=3, lw=1.5,
                label=label)

    ax.set_xlabel("Scattering angle 2$\\theta$ (degrees)")
    ax.set_ylabel("Particles per incident particle per steradian"
                  if per_steradian else "Counts per bin")
    ax.set_xlim(0, 180)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title or "Single and multiple scattering vs angle")
    fig.tight_layout()

    if save_as:
        fig.savefig(save_as, dpi=150)
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Example:  python multiple_scattering.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Slab: layers (name, thickness mm, mu 1/mm, scatter fraction).
    # The scatter fractions are only illustrative: use your own values.
    sample = create_slab_sample(20, 20, [("Al", 2.0, 0.020, 1.0),
                                    ("Fe", 1.0, 0.250, 0.6),
                                    ("Cu", 0.5, 0.300, 0.5)],
                           plane_angle_deg=90)
    print(sample.summary())

    res = simulate_scattering_orders(sample, beam_width=8, beam_height=8,
                                     n_particles=1000000, rng=1)

    print(f"\nIncident particles          : {res.n_incident}")
    print(f"No interaction (cross/miss) : {res.n_no_interaction}")
    print(f"Absorbed                    : {res.n_absorbed}")
    print(f"Scattered and exited        : {res.n_exited}")
    print(f"Unfinished (should be 0)    : {res.n_unfinished}")
    n_single, n_multi = res.single.sum(), res.multiple.sum()
    print(f"\nExited after 1 scattering   : {n_single:.0f}")
    print(f"Exited after 2 or more      : {n_multi:.0f} "
          f"({100 * n_multi / (n_single + n_multi):.1f} % of scattered)")

    plot_scattering_orders(res, logy=True, save_as="scattering_orders.png")
