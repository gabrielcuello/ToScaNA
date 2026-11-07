"""
direct_attenuation.py - SINGLE-SCATTERING attenuation factors computed by
"direct sampling" (deterministic weights), as a fast and precise REFERENCE
for the Monte Carlo results of MCAMS.

This is a NEW module: it uses the common sample interface (sample_base.py)
and does not change any other module. It works with any geometry (slab or
cylinder), with partially filled layers and with beam profiles.

What it computes
----------------
For every layer k and scattering angle 2theta, the attenuation factor of
the neutrons scattered ONCE in layer k (Paalman & Pings factor):

    A_k(2theta) = < exp( -sum_i mu_i [ L_in,i(r) + L_out,i(r, 2theta) ] ) >

The average < > is taken over all the points r of the MATERIAL of layer k
that are lit by the beam, each point weighted by the beam intensity
H(x) V(y) there. L_in,i is the path in layer i travelled by the incident
beam to reach r, L_out,i the path from r to the exit of the sample towards
the detector. Without any attenuation A_k = 1.

How (the "direct" or "expected-value" method)
---------------------------------------------
The analog Monte Carlo of MCAMS (transmission_by_layer.py) shoots particles
and lets them interact at random places: the attenuation of the incident
beam is reproduced by WHERE the particles happen to interact. Here, instead:

  1. Points are drawn uniformly in the volume lit by the beam: (x, y) from
     the beam profile (beam.py) and z uniformly over the extent of the sample
     along the beam. Points outside the material are discarded; the others
     are sorted by layer, until every layer has `n_points` points (thin
     layers then get as many points as thick ones).
  2. For each point the incident path is computed EXACTLY, by following the
     beam backwards from the point to the surface of the sample
     (outgoing_path_lengths with the direction -z), and the outgoing path
     towards the detector is computed exactly as well.
  3. The point contributes exp(-sum mu (L_in + L_out)): its attenuation is a
     WEIGHT, not a random event. A_k is the mean of the weights of layer k;
     its statistical error is (standard deviation) / sqrt(number of points).

Because no particle is "lost" in a random interaction, the variance is much
smaller than in the analog Monte Carlo for the same number of points.

What it does NOT do (use the Monte Carlo of MCAMS for that)
-----------------------------------------------------------
* MULTIPLE scattering: only the first (single) scattering is included.
* Anything that depends on the angular distribution of the scattering
  (henyey_greenstein, ...): the attenuation factors do not depend on it.
* Counting by number of scatterings, histograms in angle with an azimuth
  window, last layer of a multiply scattered particle, ...
* For VERY absorbing layers (mu x size >> 1) the weights span many orders of
  magnitude and only the points near the entrance surface matter: the Monte
  Carlo, which places its interactions there automatically, can then be
  MORE efficient (see the manual, chapter "Direct method").

Main functions
--------------
    attenuation_factors_direct(sample, beam_width, beam_height, ...)  -> AttenuationResult
    attenuation_factors_mc(sample, beam_width, beam_height, ...)      -> AttenuationResult
        (the same quantity from the Monte Carlo of MCAMS, for comparison)
    mean_material_paths(sample, beam_width, beam_height, ...)
        (mean path of the beam in each layer: needed to normalise the MC)
    compare_results(reference, other)        -> text table of the differences
    plot_attenuation_factors(results, ...)   -> figure

The result also gives the SINGLE-SCATTERING INTENSITY per incident particle
and per steradian (isotropic scattering), result.single_per_steradian(),
which is the exact reference for the "single" curve of
multiple_scattering.simulate_scattering_orders (with a narrow azimuth window).

NumPy functions used in this file (short glossary)
--------------------------------------------------
np.random.default_rng(seed)   : random number generator.
rng.uniform(a, b, size=n)     : n random numbers uniform in [a, b).
np.column_stack([a, b, c])    : put 1D arrays side by side as columns (N, 3).
np.bincount(i, weights, minlength) : sum of `weights` for every value of the
                                integer array i (a fast "sum by category").
np.exp, np.sqrt               : element by element.
a @ b                         : matrix product; (N, L) @ (L,) = sum_i a[:, i] b[i].
np.full(n, v), np.zeros, np.empty : arrays filled with v / zeros / not filled.
np.vstack(list)               : stack rows into a table.
np.where(cond, a, b)          : a where cond is True, else b.
np.nan                        : "not a number", used for "no result".
np.std(a, ddof=1)             : sample standard deviation (divides by n-1).
copy.copy(obj)                : shallow copy of an object (standard library).
time.perf_counter()           : a precise clock, to measure run times.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from beam import sample_beam_point
from sample_base import scattered_direction


# --------------------------------------------------------------------------
# Result container (the same for the direct method and the Monte Carlo)
# --------------------------------------------------------------------------
@dataclass
class AttenuationResult:
    """Single-scattering attenuation factors, one row per layer.

    angles_deg       : scattering angles 2theta (degrees), shape (A,)
    layer_names      : names of the layers
    factors          : A_k(2theta), shape (L, A); NaN for layers without
                       material in the beam or with mu = 0
    errors           : statistical error (one standard deviation), (L, A)
    mean_path        : mean path of the beam in the material of each layer
                       (mm, averaged over the beam, including the parts of
                       the beam that miss the layer), shape (L,)
    mean_path_error  : its statistical error, shape (L,)
    mus              : mu of each layer (1/mm)
    scatter_fractions: scatter fraction of each layer
    method           : "direct" or "monte carlo"
    n_samples        : points (direct) or incident particles (MC) used
    seconds          : run time
    """
    angles_deg: np.ndarray
    layer_names: tuple
    factors: np.ndarray
    errors: np.ndarray
    mean_path: np.ndarray
    mean_path_error: np.ndarray
    mus: np.ndarray
    scatter_fractions: np.ndarray
    method: str
    n_samples: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seconds: float = 0.0

    def layer(self, name):
        """Index of the layer called `name`."""
        return list(self.layer_names).index(name)

    def single_per_steradian(self, isotropic=True):
        """Singly scattered particles per incident particle and per steradian
        that leave the sample towards each angle (sum over the layers):

            sum_k  f_k mu_k <L_k> A_k(2theta) / (4 pi)

        f_k mu_k <L_k> is the probability that an incident particle is
        scattered in layer k if there were no attenuation, A_k the
        attenuation, 1/(4 pi) the isotropic angular distribution. This is the
        exact expected value of the "single" curve of
        multiple_scattering.simulate_scattering_orders(...).per_steradian
        (for a narrow azimuth window around phi; see the manual).
        """
        if not isotropic:
            raise NotImplementedError("only isotropic scattering")
        w = self.scatter_fractions * self.mus * self.mean_path / (4.0 * np.pi)   # (L,)
        A = np.where(np.isfinite(self.factors), self.factors, 0.0)
        return w @ A                                                           # (A,)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _lab_z_range(sample):
    """Interval of lab z (beam direction) that contains the whole sample.

    Cylinder: the axis is on the beam line, so |z| <= outer radius.
    Slab: the 8 corners of its box (sample frame) are converted to the lab
    frame (the two frames share the origin, so the rotation applies to
    points as well) and their z range is taken.
    """
    if hasattr(sample, "outer_radius"):
        r = float(sample.outer_radius)
        return -r, r
    lo, hi = sample._lo, sample._hi
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    zs = sample.to_lab_frame(corners)[:, 2]
    return float(zs.min()), float(zs.max())


def _draw_points(sample, beam_width, beam_height, H, V, n_target, max_draws,
                 chunk, rng):
    """Draw points uniformly in the volume lit by the beam and keep, for
    every layer, up to `n_target` of those that fall in its MATERIAL.

    Returns
    -------
    points  : (M, 3) points in the SAMPLE frame
    layer   : (M,) layer index of each point
    counts  : (L,) number of draws that fell in each layer (all draws, also
              the ones not kept): gives the mean path of the beam
    n_draws : total number of draws
    z_len   : length of the z interval sampled (mm)
    """
    L = sample.n_layers
    z0, z1 = _lab_z_range(sample)
    kept_pts = [[] for _ in range(L)]
    kept_n = np.zeros(L, dtype=int)
    counts = np.zeros(L, dtype=np.int64)
    # Layers that cannot scatter (mu = 0) do not need points.
    needed = sample.mus > 0
    n_draws = 0
    while n_draws < max_draws:
        n = int(min(chunk, max_draws - n_draws))
        n = max(n, 2)                                       # sample_beam_point needs n >= 2 for arrays
        x, y = sample_beam_point(beam_width, beam_height, H, V, n=n, rng=rng)
        z = rng.uniform(z0, z1, size=n)
        # Lab -> sample frame (same origin: the rotation applies to points).
        pts = sample.to_sample_frame(np.column_stack([x, y, z]))
        lay = sample.layer_of_points(pts)                  # -1 = not in material
        # Points within rounding distance (1e-7 mm) of the outer surface can
        # be given a layer but be refused by outgoing_path_lengths: drop them.
        lay = np.where(sample.contains(pts), lay, -1)
        n_draws += n
        inside = lay >= 0
        counts += np.bincount(lay[inside], minlength=L)
        for k in range(L):
            if kept_n[k] < n_target:
                sel = pts[lay == k][: n_target - kept_n[k]]
                kept_pts[k].append(sel)
                kept_n[k] += len(sel)
        if np.all(kept_n[needed] >= n_target):
            break
    layer = np.concatenate([np.full(kept_n[k], k) for k in range(L)])
    points = np.concatenate([np.concatenate(p) if p else np.empty((0, 3))
                             for p in kept_pts]) if layer.size else np.empty((0, 3))
    return points, layer, counts, n_draws, z1 - z0


def _chord_ratio(sample, pts):
    """For points in the material, the FULL path of their beam line in each
    layer (backwards along -z plus forwards along +z, lab frame), divided by
    the total path in all layers: array (N, L), each row sums to 1."""
    fwd = sample.to_sample_frame([0.0, 0.0, 1.0])[0]
    chord = (sample.outgoing_path_lengths(pts, fwd)
             + sample.outgoing_path_lengths(pts, -fwd))           # (N, L)
    return chord / chord.sum(axis=1, keepdims=True)


def _mean_paths_from_draws(sample, counts, n_draws, zlen, pts_uniform):
    """Mean path <L_k> of the beam in each layer, with its error.

    Plain estimate: <L_k> = zlen * (fraction of the draws in layer k). For a
    thin layer few draws fall in it and this is noisy. A better ("ratio")
    estimate uses the full beam line through each point drawn in the
    material:
        <L_k> = zlen * p_mat * < L_k(line) / L_tot(line) >
    where p_mat is the fraction of draws in ANY layer and the average is
    over points drawn uniformly in the material (lines are then picked in
    proportion to L_tot, which the ratio corrects). It is exact on average
    and much less noisy for thin layers.
    """
    p_mat = counts.sum() / n_draws
    ratio = _chord_ratio(sample, pts_uniform)                         # (N, L)
    r = ratio.mean(axis=0)
    r_err = ratio.std(axis=0, ddof=1) / np.sqrt(len(ratio))
    mean = zlen * p_mat * r
    rel_p = np.sqrt((1 - p_mat) / (p_mat * n_draws))
    with np.errstate(invalid="ignore", divide="ignore"):
        err = mean * np.sqrt(rel_p ** 2 + np.where(r > 0, (r_err / r) ** 2, 0.0))
    return mean, err


def mean_material_paths(sample, beam_width, beam_height, H=None, V=None,
                        n_draws=1_000_000, n_lines=200_000, rng=None):
    """Mean path (mm) of the incident beam in the material of each layer,
    averaged over the whole beam (H, V profiles, including the parts of the
    beam that miss the layer), with its statistical error.

    mu_k * <L_k> is the probability that an incident particle would interact
    in layer k if there were no attenuation. See _mean_paths_from_draws for
    the method.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    z0, z1 = _lab_z_range(sample)
    x, y = sample_beam_point(beam_width, beam_height, H, V, n=max(n_draws, 2), rng=rng)
    z = rng.uniform(z0, z1, size=len(x))
    pts = sample.to_sample_frame(np.column_stack([x, y, z]))
    lay = sample.layer_of_points(pts)
    lay = np.where(sample.contains(pts), lay, -1)          # see _draw_points
    counts = np.bincount(lay[lay >= 0], minlength=sample.n_layers)
    inside = pts[lay >= 0][:n_lines]
    return _mean_paths_from_draws(sample, counts, len(x), z1 - z0, inside)


# --------------------------------------------------------------------------
# The direct method
# --------------------------------------------------------------------------
def attenuation_factors_direct(sample, beam_width, beam_height, H=None, V=None,
                               angles_deg=None, phi_deg=0.0, n_points=200_000,
                               max_draws=None, chunk=500_000, rng=None):
    """
    Single-scattering attenuation factors A_k(2theta) by direct sampling.

    Parameters
    ----------
    sample : LayeredSample
        Slab or cylinder (sample_factory.make_sample), partial layers allowed.
    beam_width, beam_height : float
        Beam size (mm); H, V optional beam profiles (beam.py).
    angles_deg : array-like, optional
        Scattering angles 2theta in degrees. Default 0 to 180, step 1.
    phi_deg : float
        Azimuth of the detector (0 = towards +x, horizontal plane).
    n_points : int
        Number of points kept IN EACH LAYER. The error of A_k decreases as
        1/sqrt(n_points). 1e5 gives typically 0.05-0.2 %.
    max_draws : int, optional
        Safety limit on the number of points drawn (default 300 * n_points).
        A layer that is not reached (outside the beam, empty part, ...)
        gets NaN.
    chunk : int
        Points drawn at once (memory use).
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible results.

    Returns
    -------
    AttenuationResult (method = "direct")
    """
    t_start = time.perf_counter()
    if angles_deg is None:
        angles_deg = np.linspace(0.0, 180.0, 181)
    angles = np.asarray(angles_deg, dtype=float)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if max_draws is None:
        max_draws = 300 * n_points
    L = sample.n_layers

    # --- Step 1: points in the material of each layer -----------------------
    pts, lay, counts, n_draws, zlen = _draw_points(
        sample, beam_width, beam_height, H, V, n_points, max_draws, chunk, rng)
    # Mean path of the beam in each layer (to normalise the Monte Carlo and
    # for the single-scattering intensity): an independent set of uniform
    # points in the material gives the ratio estimate.
    mean_path, mean_path_err = mean_material_paths(
        sample, beam_width, beam_height, H, V,
        n_draws=max(1_000_000, min(n_draws, 5_000_000)), n_lines=min(n_points, 200_000), rng=rng)

    # --- Step 2: incident path of every point -------------------------------
    # Follow the beam BACKWARDS from the point (direction -z of the lab
    # frame, converted to the sample frame) until it leaves the sample: the
    # path in each layer is exactly the path of the incident beam.
    back = -sample.to_sample_frame([0.0, 0.0, 1.0])[0]
    tau_in = sample.outgoing_path_lengths(pts, back) @ sample.mus if len(pts) else np.zeros(0)
    w_in = np.exp(-tau_in)                                  # attenuation on the way in

    # --- Step 3: outgoing path for every angle, mean per layer --------------
    sums = np.zeros((L, len(angles)))
    sums2 = np.zeros((L, len(angles)))
    n_k = np.bincount(lay, minlength=L).astype(float)
    for a, angle in enumerate(angles):
        d = sample.to_sample_frame(scattered_direction(angle, phi_deg))[0]
        w = w_in * sample.transmission(sample.outgoing_path_lengths(pts, d))
        sums[:, a] = np.bincount(lay, weights=w, minlength=L)
        sums2[:, a] = np.bincount(lay, weights=w * w, minlength=L)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / n_k[:, None]
        var = (sums2 / n_k[:, None] - mean ** 2) * n_k[:, None] / (n_k[:, None] - 1)
        err = np.sqrt(np.maximum(var, 0.0) / n_k[:, None])
    bad = (n_k < 2) | (sample.mus <= 0)
    mean[bad], err[bad] = np.nan, np.nan

    return AttenuationResult(
        angles_deg=angles, layer_names=tuple(l.name for l in sample.layers),
        factors=mean, errors=err, mean_path=mean_path, mean_path_error=mean_path_err,
        mus=sample.mus.copy(), scatter_fractions=sample.scatter_fractions.copy(),
        method="direct", n_samples=n_k.astype(int),
        seconds=time.perf_counter() - t_start)


# --------------------------------------------------------------------------
# The same quantity from the Monte Carlo of MCAMS
# --------------------------------------------------------------------------
def attenuation_factors_mc(sample, beam_width, beam_height, H=None, V=None,
                           angles_deg=None, phi_deg=0.0, n_particles=200_000,
                           n_batches=5, mean_path=None, rng=None):
    """
    The same A_k(2theta), computed with the analog Monte Carlo of MCAMS
    (transmission_by_layer.transmission_by_last_layer, first scatterings
    only), for comparison with the direct method.

    The Monte Carlo gives, for the first scattering events in layer k, the
    sum S_k(2theta) of their outgoing attenuations. Its expected value per
    incident particle is  f_k mu_k <L_k> A_k(2theta)  (the incident
    attenuation is in the positions of the interactions), so

        A_k = S_k / N_incident / (f_k mu_k <L_k>).

    Here the scatter fractions are temporarily set to 1 (it does not change
    A_k and gives more events). <L_k> (mean_path) is taken from a direct
    result if given (`mean_path=direct_result.mean_path`), otherwise
    computed with mean_material_paths.

    The error is the standard error of the mean of `n_batches` independent
    runs of `n_particles` particles each.
    Returns AttenuationResult (method = "monte carlo").
    """
    # Imported here so that this module can be used without the MC modules.
    from transmission_by_layer import transmission_by_last_layer

    t_start = time.perf_counter()
    if angles_deg is None:
        angles_deg = np.linspace(0.0, 180.0, 181)
    angles = np.asarray(angles_deg, dtype=float)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if mean_path is None:
        mean_path, _ = mean_material_paths(sample, beam_width, beam_height, H, V, rng=rng)
    mean_path = np.asarray(mean_path, dtype=float)

    # A shallow copy of the sample in which every interaction is a scattering.
    pure = copy.copy(sample)
    pure.scatter_fractions = np.ones(sample.n_layers)

    runs = []
    for _ in range(n_batches):
        res = transmission_by_last_layer(pure, beam_width, beam_height, H, V,
                                         n_particles=n_particles, angles_deg=angles,
                                         phi_deg=phi_deg, max_orders=1, rng=rng)
        S = res.sum_attenuation[0] / res.n_incident            # (L, A)
        with np.errstate(invalid="ignore", divide="ignore"):
            runs.append(S / (sample.mus * mean_path)[:, None])
    runs = np.array(runs)
    mean = runs.mean(axis=0)
    err = runs.std(axis=0, ddof=1) / np.sqrt(n_batches) if n_batches > 1 else np.full_like(mean, np.nan)
    bad = (sample.mus <= 0) | (mean_path <= 0)
    mean[bad], err[bad] = np.nan, np.nan
    return AttenuationResult(
        angles_deg=angles, layer_names=tuple(l.name for l in sample.layers),
        factors=mean, errors=err, mean_path=mean_path,
        mean_path_error=np.zeros_like(mean_path), mus=sample.mus.copy(),
        scatter_fractions=sample.scatter_fractions.copy(), method="monte carlo",
        n_samples=np.full(sample.n_layers, n_particles * n_batches),
        seconds=time.perf_counter() - t_start)


# --------------------------------------------------------------------------
# Comparison and plot
# --------------------------------------------------------------------------
def compare_results(reference, other, every=30):
    """Text table comparing two AttenuationResult (e.g. direct and MC):
    for each layer, the values every `every` degrees, the relative
    difference and the difference in units of the combined error (z).
    |z| < 2 for most angles means agreement within the statistics.
    Returns the text (and prints nothing)."""
    lines = [f"{'layer':<10}{'2theta':>7}{reference.method:>16}{other.method:>16}"
             f"{'diff %':>9}{'z':>7}"]
    for k, name in enumerate(reference.layer_names):
        if not np.all(np.isfinite(reference.factors[k])):
            continue
        for a in range(0, len(reference.angles_deg), every):
            r, o = reference.factors[k, a], other.factors[k, a]
            e = np.hypot(reference.errors[k, a], other.errors[k, a])
            lines.append(f"{name:<10}{reference.angles_deg[a]:7.1f}"
                         f"{r:9.5f}+-{reference.errors[k, a]:.5f}"
                         f"{o:9.5f}+-{other.errors[k, a]:.5f}"
                         f"{100 * (o / r - 1):9.3f}{(o - r) / e:7.2f}")
    return "\n".join(lines)


def plot_attenuation_factors(results, title=None, save_as=None, show=True):
    """Plot A_k(2theta) of one or several AttenuationResult.

    results : one result or a list; the first is drawn with lines, the others
              with points and error bars (typical use: [direct, mc]).
    When two results are given, a lower panel shows their relative
    difference in percent.
    """
    import matplotlib.pyplot as plt

    if isinstance(results, AttenuationResult):
        results = [results]
    ref = results[0]
    two = len(results) > 1
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(7, 5.5 if two else 4.2),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2.3, 1.2]} if two else None)
    ax = axes[0][0]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, name in enumerate(ref.layer_names):
        if not np.all(np.isfinite(ref.factors[k])):
            continue
        c = colours[k % len(colours)]
        ax.plot(ref.angles_deg, ref.factors[k], color=c, lw=2, label=f"{name} ({ref.method})")
        for other in results[1:]:
            ax.errorbar(other.angles_deg, other.factors[k], yerr=other.errors[k], fmt="o",
                        ms=3, mfc="white", color=c, errorevery=max(1, len(other.angles_deg) // 30),
                        markevery=max(1, len(other.angles_deg) // 30),
                        label=f"{name} ({other.method})")
            rel = 100 * (other.factors[k] / ref.factors[k] - 1)
            rel_e = 100 * np.hypot(other.errors[k], ref.errors[k]) / ref.factors[k]
            axes[1][0].fill_between(other.angles_deg, rel - rel_e, rel + rel_e, color=c, alpha=0.2, lw=0)
            axes[1][0].plot(other.angles_deg, rel, color=c, lw=1)
    ax.set_ylabel("Attenuation factor $A_k$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title or "Single-scattering attenuation factors")
    if two:
        axes[1][0].axhline(0, color="k", lw=0.8)
        axes[1][0].set_ylabel(f"{results[1].method} / {ref.method} - 1 (%)")
        axes[1][0].grid(alpha=0.3)
    axes[-1][0].set_xlabel("Scattering angle 2$\\theta$ (degrees)")
    axes[-1][0].set_xlim(0, 180)
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=150)
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Example:  python direct_attenuation.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from cylinder import create_cylindrical_sample

    # Ni rod (6 mm diameter) in a V can (0.2 mm wall), lambda = 0.5 A:
    # mu in 1/mm, scatter fractions mu_s/mu_t.
    sample = create_cylindrical_sample(50, [("Ni", 6.0, 0.1803, 0.937),
                                            ("V", 0.2, 0.0459, 0.779)])
    print(sample.summary())
    angles = np.arange(0, 181, 10.0)

    direct = attenuation_factors_direct(sample, 15, 20, angles_deg=angles,
                                        n_points=200_000, rng=1)
    mc = attenuation_factors_mc(sample, 15, 20, angles_deg=angles, n_particles=200_000,
                                n_batches=5, mean_path=direct.mean_path, rng=2)
    print(f"\ndirect: {direct.seconds:.1f} s   monte carlo: {mc.seconds:.1f} s\n")
    print(compare_results(direct, mc, every=3))
    plot_attenuation_factors([direct, mc], save_as="direct_vs_mc.png")
