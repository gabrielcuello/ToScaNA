"""
transmission_by_layer.py - Transmission vs scattering angle, separated by the
LAYER in which the particle had its LAST scattering before leaving the sample.

This is a NEW module: it uses the other modules but does not change any of
them. It works with any geometry (slab or cylinder, see sample_base.py).

The question it answers
-----------------------
A particle that leaves the sample towards the detector at angle 2theta had a
last scattering somewhere, in some layer k. From that point it flies to the
detector and crosses several layers on the way out, each one attenuating it.
For every layer k, this module gives a vector (one value per angle):

        T_k(2theta) = average over the particles whose last scattering was in
                      layer k of   exp( - sum_i mu_i * L_i )

where L_i is the distance travelled in layer i (ALL layers crossed, not only
layer k) from the scattering point to the exit of the sample, along the
direction of the detector. This is the Beer-Lambert law, as in
transmission_vs_angle.py.

How "last scattering" is defined (point estimator)
--------------------------------------------------
We follow the particles one by one (random walk, as in
multiple_scattering.py): first interaction, scattering or absorption, new
direction, next interaction, ... At EVERY scattering event we ask: "if this
were the last scattering and the particle went to the detector, how much
would it be attenuated?". That is computed for all the angles at once, with
the exact direction of the detector, which gives smooth curves.
The event is classified by
  * the layer where it happens (= the layer of the last scattering),
  * its order: "single" = first scattering of the particle,
               "multiple" = second or later scattering.
"total" is single + multiple together.

Notes
-----
* T_k is the pure attenuation (between 0 and 1). The result also gives how
  many scattering events took place in each layer, to weight the layers:
  events(kind)[k] / n_incident = events in layer k per incident particle.
* The probability of going towards the detector after a scattering (the
  angular distribution of the scattering, e.g. 1/(4 pi) if isotropic) is
  NOT included: it is a common factor, as in transmission_vs_angle.py.
* A layer without scattering events (for instance an air gap with mu = 0)
  gets NaN ("not a number") and is not drawn in the plots.
* With scatter_fraction = 1 in all layers, the "single" result averaged over
  the layers is exactly the transmission_vs_angle.py result.

Usage
-----
    res = transmission_by_last_layer(sample, 8, 8, n_particles=50000)
    res.transmission[k]            # vector vs angle, last scattering in layer k
    res.transmission_single[k]     # same, only first scatterings
    res.transmission_multiple[k]   # same, only second and later scatterings
    plot_transmission_by_layer(res)

NumPy functions used in this file (short glossary)
--------------------------------------------------
np.linspace(a, b, n)    : n equally spaced numbers from a to b.
np.vstack(list)         : stack arrays one on top of the other (as rows).
np.zeros(shape)         : array filled with zeros.
np.bincount(idx, weights, minlength) : for every integer value v, adds up the
                          `weights` of the elements whose idx equals v (or
                          counts them if no weights): a fast "sum by category".
a.reshape(r, c)         : same numbers, arranged as r rows and c columns.
np.clip(a, lo, hi)      : limit the values to [lo, hi].
np.where(cond, a, b)    : take `a` where cond is True, else `b`.
np.nan                  : "not a number", used for "no data".
np.errstate             : temporarily silence warnings (divisions we protect).
boolean masks           : a[mask] keeps the elements where mask is True; `~mask`
                          is "not mask".
"""

from dataclasses import dataclass

import numpy as np

from .beam import sample_beam_point
from .sample_base import scattered_direction
from .multiple_scattering import isotropic_cos, rotate_directions


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
# Positions in the first axis of the stored arrays:
_SINGLE, _MULTIPLE = 0, 1


@dataclass
class LastLayerResult:
    """Result of `transmission_by_last_layer`.

    angles_deg      : scattering angles 2theta (degrees), shape (A,)
    layer_names     : names of the layers, in the order of the sample
    sum_attenuation : sum of the attenuation factors, shape (2, L, A):
                      [order class (single, multiple), layer, angle]
    n_events        : number of scattering events, shape (2, L)
    n_incident      : number of incident particles

    Use the methods and properties below instead of these raw sums.
    """
    angles_deg: np.ndarray
    layer_names: tuple
    sum_attenuation: np.ndarray
    n_events: np.ndarray
    n_incident: int

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _classes(kind):
        """Which order classes belong to `kind` ("single", "multiple", "total")."""
        if kind == "single":
            return [_SINGLE]
        if kind == "multiple":
            return [_MULTIPLE]
        if kind == "total":
            return [_SINGLE, _MULTIPLE]
        raise ValueError("kind must be 'single', 'multiple' or 'total'.")

    def events(self, kind="total"):
        """Number of scattering events in each layer, shape (L,)."""
        return self.n_events[self._classes(kind)].sum(axis=0)

    def transmission_of(self, kind="total"):
        """Transmission vs angle for every layer of last scattering,
        shape (L, A). NaN for layers without events."""
        sums = self.sum_attenuation[self._classes(kind)].sum(axis=0)    # (L, A)
        n = self.events(kind)[:, None]                                  # (L, 1)
        # average = sum / number of events; where there are no events we
        # avoid the division by zero and put NaN instead.
        return np.where(n > 0, sums / np.where(n > 0, n, 1), np.nan)

    def mean_transmission(self, kind="total"):
        """Transmission vs angle averaged over ALL the layers, weighted by
        the number of events of each layer, shape (A,)."""
        sums = self.sum_attenuation[self._classes(kind)].sum(axis=0)    # (L, A)
        return sums.sum(axis=0) / self.events(kind).sum()

    # ---- the three vectors most often wanted ---------------------------------
    @property
    def transmission(self):
        """(L, A): last scattering in layer k, all scattering orders."""
        return self.transmission_of("total")

    @property
    def transmission_single(self):
        """(L, A): last scattering in layer k, particles scattered once."""
        return self.transmission_of("single")

    @property
    def transmission_multiple(self):
        """(L, A): last scattering in layer k, particles scattered 2+ times."""
        return self.transmission_of("multiple")


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def transmission_by_last_layer(sample, beam_width, beam_height, H=None, V=None,
                               n_particles=50000, angles_deg=None, phi_deg=0.0,
                               deflection_sampler=None, max_orders=200,
                               rng=None):
    """
    Transmission vs scattering angle, with one vector per layer in which the
    last scattering took place (see the top of this file).

    Parameters
    ----------
    sample : LayeredSample
        Slab or cylindrical sample (see sample_factory.make_sample).
    beam_width, beam_height : float
        Beam size (mm); H, V optional beam profiles (see beam.py).
    n_particles : int
        Number of incident particles. The cost is proportional to this number
        times the number of angles, so 20000-100000 is a good start.
    angles_deg : array-like, optional
        Scattering angles 2theta in degrees. Default: 0 to 180, step 1.
    phi_deg : float
        Azimuth of the detector around the beam (0 = towards +x, in the
        horizontal plane).
    deflection_sampler : callable, optional
        (n, rng) -> cos(chi) of ONE scattering event; default isotropic.
        It decides where the later scatterings of a particle happen.
    max_orders : int
        Safety limit on the number of scatterings followed per particle.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible results.

    Returns
    -------
    LastLayerResult
    """
    if angles_deg is None:
        angles_deg = np.linspace(0.0, 180.0, 181)
    angles_deg = np.asarray(angles_deg, dtype=float)
    n_angles = len(angles_deg)
    n_layers = sample.n_layers

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    sampler = deflection_sampler or isotropic_cos       # `or`: default if None

    # --- Step 1: the direction of the detector for every angle ---------------
    # scattered_direction gives the unit vector in the LAB frame; the sample
    # works in its own frame, so we convert them. np.vstack stacks the
    # (1, 3) results into one table with a row per angle: shape (A, 3).
    detector_dirs = np.vstack([sample.to_sample_frame(scattered_direction(a, phi_deg))
                               for a in angles_deg])

    # --- Step 2: start points in the beam and first interaction points -------
    x, y = sample_beam_point(beam_width, beam_height, H, V,
                             n=max(n_particles, 2), rng=rng)
    n_incident = len(x)
    pos, _ = sample.sample_interaction_points(x, y, rng=rng)

    # State of the particles still inside the sample (SAMPLE frame):
    #   pos  : position of their current interaction
    #   dirn : direction they had when arriving there
    #   n_sc : number of scatterings suffered BEFORE this interaction
    dirn = np.tile(sample.to_sample_frame([0.0, 0.0, 1.0]), (len(pos), 1))
    n_sc = np.zeros(len(pos), dtype=int)

    # Accumulators filled in the loop below:
    #   sums[c, k, a]  : sum of attenuation factors (class c, layer k, angle a)
    #   counts[c, k]   : number of scattering events (class c, layer k)
    sums = np.zeros((2, n_layers, n_angles))
    counts = np.zeros((2, n_layers), dtype=np.int64)

    # --- The loop: one round = one scattering event for every particle still
    #     inside the sample ---------------------------------------------------
    for _ in range(max_orders):
        if len(pos) == 0:                # nobody left inside: finished
            break

        # Step 3: at each interaction: scattering or absorption?
        layer = sample.layer_of_points(pos)
        # (A point flagged "outside" by rounding errors, -1, goes to layer 0
        # so that the bookkeeping below never fails.)
        layer = np.clip(layer, 0, n_layers - 1)
        scatters = rng.uniform(size=len(pos)) < sample.scatter_fractions[layer]
        # Keep only the particles that scatter (boolean mask).
        pos, dirn, layer = pos[scatters], dirn[scatters], layer[scatters]
        n_sc = n_sc[scatters] + 1
        if len(pos) == 0:
            break

        # Step 4: RECORD. Each of these scattering events could be the LAST
        # one of its particle. Category of every event = (order class, layer):
        # class 0 if it is the first scattering (single), class 1 otherwise.
        order_class = (n_sc > 1).astype(int)
        category = order_class * n_layers + layer       # one integer per event
        # np.bincount counts the events of each category; reshape arranges
        # the 2*L counts as a table (class, layer).
        counts += np.bincount(category, minlength=2 * n_layers).reshape(2, n_layers)

        # For every angle: path from the scattering point to the exit along
        # the detector direction (distance in each layer), then the
        # Beer-Lambert attenuation exp(-sum(mu*L)) of every event, and add
        # them up by category with np.bincount(weights=...).
        for a in range(n_angles):
            path = sample.outgoing_path_lengths(pos, detector_dirs[a])
            attenuation = sample.transmission(path)
            sums[:, :, a] += np.bincount(category, weights=attenuation,
                                         minlength=2 * n_layers
                                         ).reshape(2, n_layers)

        # Step 5: the walk goes on. New direction after the scattering...
        dirn = rotate_directions(dirn, sampler(len(pos), rng), rng)
        # ...and next interaction inside the sample, or exit (then the
        # particle is finished: it has already been recorded above).
        new_pos, interacts = sample.next_interaction(pos, dirn, rng=rng)
        pos, dirn, n_sc = new_pos[interacts], dirn[interacts], n_sc[interacts]

    return LastLayerResult(
        angles_deg=angles_deg,
        layer_names=tuple(l.name for l in sample.layers),
        sum_attenuation=sums,
        n_events=counts,
        n_incident=n_incident)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_transmission_by_layer(result, kind="all", title=None, save_as=None,
                               show=True):
    """
    Plot the transmission vs angle, one curve per layer of last scattering.

    kind    : "total", "single", "multiple", or "all" (default): three panels
              side by side (total / single / multiple).
    save_as : file name to save the figure, or None.
    show    : open the plot window (False to only save).

    The legend gives, for each layer, the number of scattering events in it
    per incident particle (how important that layer is).
    """
    import matplotlib.pyplot as plt

    kinds = ["total", "single", "multiple"] if kind == "all" else [kind]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.5),
                             sharey=True, squeeze=False)

    for ax, kd in zip(axes[0], kinds):
        T = result.transmission_of(kd)          # (L, A)
        n_events = result.events(kd)            # (L,)
        for k, name in enumerate(result.layer_names):
            if n_events[k] == 0:                # no events: nothing to draw
                continue
            ax.plot(result.angles_deg, T[k], lw=2,
                    label=f"last scattering in {name} "
                          f"({n_events[k] / result.n_incident:.3f}/incident)")
        ax.set_title(kd if kind == "all" else (title or kd))
        ax.set_xlabel("Scattering angle 2$\\theta$ (degrees)")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Transmission")
    if kind == "all":
        fig.suptitle(title or "Transmission vs angle by layer of last scattering")
    fig.tight_layout()

    if save_as:
        fig.savefig(save_as, dpi=150)
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Example:  python transmission_by_layer.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from .slab import create_slab_sample

    # Slab with (name, thickness mm, mu 1/mm, scatter fraction): illustrative.
    sample = create_slab_sample(20, 20, [("Al", 2.0, 0.020, 1.0),
                                         ("Fe", 1.0, 0.250, 0.6),
                                         ("Cu", 0.5, 0.300, 0.5)])
    print(sample.summary())

    res = transmission_by_last_layer(sample, 8, 8, n_particles=50000, rng=1)

    print("\nScattering events per incident particle, by layer of scattering:")
    for kind in ("single", "multiple"):
        per_incident = res.events(kind) / res.n_incident
        # float(v) turns NumPy numbers into plain Python numbers for printing.
        print(f"  {kind:<9}", {name: round(float(v), 4)
                              for name, v in zip(res.layer_names, per_incident)})

    print("\nTransmission at 2theta = 60 and 120 degrees (last scattering in ...):")
    for k, name in enumerate(res.layer_names):
        print(f"  {name}:  {res.transmission[k, 60]:.4f}   {res.transmission[k, 120]:.4f}")

    plot_transmission_by_layer(res, save_as="transmission_by_layer.png")
