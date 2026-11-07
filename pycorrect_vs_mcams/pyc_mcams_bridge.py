"""
pyc_mcams_bridge.py - Tools to compute with MCAMS (Monte Carlo) the same
quantities that pycorrect (CORRECT 2.24) computes analytically:

    pycorrect                          MCAMS equivalent (this module)
    ---------------------------------  -----------------------------------------
    atcors()  -> A_s,sch, A_c,sch, ...  mc_attenuation_factors()
                 (Paalman & Pings)        single-scattering point estimator of
                                          transmission_by_layer.py, normalised
    mscat()   -> delta (Blech &          mc_delta()  : multiple / single, all orders
                 Averbach, all orders)   mc_scattering_orders() : order by order,
                                          to compare with the double-scattering
                                          table delta_2 of doublescattering.dat

Only attenuation and multiple scattering are compared (no Placzek, no
vanadium normalisation), because MCAMS contains only these two effects.

Units
-----
pycorrect works in cm and cm^-1, MCAMS in mm and mm^-1. All the functions
of this module take pycorrect-style arguments (cm, cm^-1) and convert them
internally with the two helpers `cm_to_mm` and `per_cm_to_per_mm`, so that the
same numbers can be given to both programs.

How a Paalman & Pings factor is obtained from MCAMS
---------------------------------------------------
transmission_by_last_layer() records, for every scattering event in layer k,
the attenuation exp(-sum_i mu_i L_i) on the way out towards the detector.
The first interaction points are drawn with the Beer-Lambert law along the
incident beam, so they already contain the attenuation on the way IN.
Summing over the FIRST scattering events (order "single") and dividing by the
number of incident particles gives, per unit height of beam,

    sum_single[k] / N_inc = (f_k mu_k / W) * Integral_{S_k} exp(-sum mu (l_in + l_out)) dS

(W = beam width, f_k = scatter fraction, S_k = part of the cross-section of
layer k inside the beam). The Paalman & Pings factor is the same integral
divided by S_k, so

    A_k(2theta) = sum_single[k](2theta) / N_inc  /  (f_k mu_k S_k / W).

This is exact (no approximation) when the beam height is smaller than the
sample height and the detector is in the horizontal plane, which is the
geometry assumed by Paalman & Pings.

How delta is obtained from MCAMS
--------------------------------
delta = (intensity scattered 2 or more times) / (intensity scattered once),
both attenuated on their way out, i.e. exactly the ratio of the "multiple"
and "single" sums of transmission_by_last_layer(), summed over the layers.
The order-by-order version (mc_scattering_orders) gives delta_2 = order 2 /
order 1, the quantity tabulated by Blech & Averbach.

Statistical errors
------------------
Every Monte Carlo quantity is computed in `n_batches` independent runs
(different random seeds); the result is the mean and the error is the
standard error of the mean, std / sqrt(n_batches).

NumPy functions used here
-------------------------
np.asarray, np.zeros, np.vstack, np.tile : build arrays
np.mean(a, axis=0), np.std(a, axis=0, ddof=1) : mean and sample standard
        deviation over the batches (ddof=1 divides by n-1)
np.clip(a, lo, hi) : limit values
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# Locate the two packages.  MCAMS modules are imported as top-level modules
# (from beam import ...), so their folder must be on sys.path.  Set the
# environment variables MCAMS_DIR and PYCORRECT_DIR if they are elsewhere.
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MCAMS_DIR = os.environ.get("MCAMS_DIR", os.path.join(HERE, "..", "MCAMS"))
PYCORRECT_DIR = os.environ.get("PYCORRECT_DIR", os.path.join(HERE, "..", "pycorrect"))
for _p in (MCAMS_DIR, PYCORRECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))

from beam import sample_beam_point                                # noqa: E402  (MCAMS)
from cylinder import create_cylindrical_sample                    # noqa: E402  (MCAMS)
from sample_base import Layer, scattered_direction                # noqa: E402  (MCAMS)
from multiple_scattering import isotropic_cos, rotate_directions  # noqa: E402  (MCAMS)
from transmission_by_layer import transmission_by_last_layer      # noqa: E402  (MCAMS)

from pycorrect.ndplib import area                                 # noqa: E402  (pycorrect)


# ==========================================================================
#  Units
# ==========================================================================
def cm_to_mm(x):
    """Length: pycorrect (cm) -> MCAMS (mm)."""
    return 10.0 * x


def per_cm_to_per_mm(mu):
    """Linear attenuation coefficient: pycorrect (1/cm) -> MCAMS (1/mm)."""
    return mu / 10.0


def mu_total(rho, sigma_s, sigma_a, wavelength):
    """Total linear attenuation coefficient in 1/cm, exactly as pycorrect:
    mu = n (sigma_s + sigma_a * lambda / 1.798), n in atoms/A^3, sigma in barn.
    (1 barn/A^3 = 1/cm, so no other factor is needed.)"""
    return rho * (sigma_s + sigma_a * wavelength / 1.798)


# ==========================================================================
#  Geometry: the pycorrect cylinder (sample + can + furnace) in MCAMS
# ==========================================================================
def cylinder_layers_from_pycorrect(r1, r2=0.0, r3=0.0, r4=0.0,
                                   smu=0.0, cmu=0.0, hmu=0.0,
                                   s_frac=1.0, c_frac=1.0, h_frac=1.0):
    """List of MCAMS layers equivalent to the arguments of pycorrect.atcors.

    pycorrect (cm, 1/cm)                       MCAMS layer (mm, 1/mm)
    r1  sample radius                     ->   "sample": thickness = DIAMETER 2*r1
    r2  container outer radius            ->   "can": radial thickness r2 - r1
    r3  furnace inner radius              ->   "gap": radial thickness r3 - r2, mu = 0
    r4  furnace outer radius              ->   "furnace": radial thickness r4 - r3
    smu, cmu, hmu  total mu                ->   mu of each layer
    *_frac  scattering fraction mu_s/mu_t ->   scatter_fraction (only matters
                                               for multiple scattering)

    As in pycorrect, r2 <= r1 means "no container" and r4 <= r3 "no furnace".
    A layer with mu = 0 is allowed in MCAMS (it attenuates nothing and
    scatters nothing), which is how an empty can or an empty sample position
    is simulated.
    Returns (layers, names_of_layers_present).
    """
    layers = [Layer("sample", cm_to_mm(2 * r1), per_cm_to_per_mm(smu), s_frac)]
    r_out = r1
    if r2 > r1:
        layers.append(Layer("can", cm_to_mm(r2 - r1), per_cm_to_per_mm(cmu), c_frac))
        r_out = r2
    if r4 > r3:
        if r3 > r_out:
            layers.append(Layer("gap", cm_to_mm(r3 - r_out), 0.0, 1.0))
        layers.append(Layer("furnace", cm_to_mm(r4 - r3), per_cm_to_per_mm(hmu), h_frac))
    return layers, [l.name for l in layers]


def layer_area_in_beam(sample, beam_width_mm, k):
    """Area (mm^2) of the horizontal cross-section of layer k that lies inside
    the beam band |x| <= W/2.  Uses pycorrect.ndplib.area(r, d), the area of a
    circle of radius r inside a band of half-width d (Fortran FUNCTION AREA)."""
    d = beam_width_mm / 2.0
    r_out = sample.radii[k]
    r_in = sample.inner_radii[k]
    return area(r_out, d) - (area(r_in, d) if r_in > 0 else 0.0)


# ==========================================================================
#  Monte Carlo Paalman & Pings factors
# ==========================================================================
def mc_attenuation_factors(layers, beam_width_cm, angles_deg,
                           n_particles=200_000, n_batches=10,
                           sample_height_cm=5.0, beam_height_cm=2.0, seed=1):
    """Paalman & Pings factor of every layer, computed with MCAMS.

    For each layer k, A_k(2theta) is the attenuation (in + out) averaged over
    the part of the layer inside the beam, for neutrons scattered ONCE in
    layer k and attenuated by all the layers (see the module docstring).
    The scatter fractions are set to 1 here: they do not change A (the
    attenuation uses the total mu) but they maximise the number of events.

    Parameters
    ----------
    layers          : MCAMS layers (cylinder_layers_from_pycorrect)
    beam_width_cm   : full beam width (cm), as in pycorrect
    angles_deg      : scattering angles 2theta (degrees)
    n_particles     : incident particles PER BATCH
    n_batches       : independent runs used for the error bars
    sample_height_cm, beam_height_cm : vertical sizes; the beam must be
                      lower than the sample (2D geometry of Paalman & Pings)

    Returns
    -------
    mean : array (n_layers, n_angles)   A_k(2theta); NaN for layers with mu=0
    err  : array (n_layers, n_angles)   standard error of the mean
    """
    pure = [Layer(l.name, l.thickness, l.mu, 1.0) for l in layers]
    sample = create_cylindrical_sample(cm_to_mm(sample_height_cm), pure)
    W = cm_to_mm(beam_width_cm)
    angles = np.asarray(angles_deg, float)
    # Number of first-scattering events per incident particle that there
    # would be WITHOUT attenuation: mu_k * S_k / W  (per unit height of beam)
    norm = np.array([sample.mus[k] * layer_area_in_beam(sample, W, k) / W
                     for k in range(sample.n_layers)])
    runs = []
    for b in range(n_batches):
        res = transmission_by_last_layer(sample, W, cm_to_mm(beam_height_cm),
                                         n_particles=n_particles, angles_deg=angles,
                                         max_orders=1,           # first scattering only
                                         rng=seed + b)
        single = res.sum_attenuation[0] / res.n_incident        # (L, A)
        with np.errstate(divide="ignore", invalid="ignore"):
            runs.append(single / norm[:, None])
    runs = np.array(runs)
    return runs.mean(axis=0), runs.std(axis=0, ddof=1) / math.sqrt(n_batches)


# ==========================================================================
#  Monte Carlo multiple scattering
# ==========================================================================
def mc_scattering_orders(sample, beam_width_mm, beam_height_mm, angles_deg,
                         n_particles, max_orders=80, rng=None):
    """Point estimator resolved by scattering ORDER (built from MCAMS pieces).

    It follows exactly the random walk of transmission_by_last_layer()
    (first interaction from the beam, scattering or absorption with the
    scatter fraction of the layer, isotropic new direction, next interaction)
    but keeps one sum per order k = 1, 2, 3, ... instead of single/multiple:

        S[k-1, a] = sum over the k-th scattering events of exp(-sum mu L_out)
                    towards the detector at angle a.

    Then S[1]/S[0] = delta_2 (double / single, Blech & Averbach table) and
    S[1:].sum(0)/S[0] = delta (all orders).
    Arguments in mm (MCAMS units).  Returns S (max_orders, n_angles) and the
    number of incident particles.
    """
    rng = np.random.default_rng(rng)
    angles = np.asarray(angles_deg, float)
    dets = np.vstack([sample.to_sample_frame(scattered_direction(a)) for a in angles])
    x, y = sample_beam_point(beam_width_mm, beam_height_mm, n=n_particles, rng=rng)
    pos, _ = sample.sample_interaction_points(x, y, rng=rng)
    dirn = np.tile(sample.to_sample_frame([0.0, 0.0, 1.0]), (len(pos), 1))
    S = np.zeros((max_orders, len(angles)))
    for k in range(max_orders):
        if len(pos) == 0:
            break
        layer = np.clip(sample.layer_of_points(pos), 0, sample.n_layers - 1)
        keep = rng.uniform(size=len(pos)) < sample.scatter_fractions[layer]
        pos, dirn = pos[keep], dirn[keep]
        for a in range(len(angles)):
            S[k, a] += sample.transmission(sample.outgoing_path_lengths(pos, dets[a])).sum()
        dirn = rotate_directions(dirn, isotropic_cos(len(pos), rng), rng)
        new_pos, inside = sample.next_interaction(pos, dirn, rng=rng)
        pos, dirn = new_pos[inside], dirn[inside]
    return S, n_particles


def mc_delta(layers, beam_width_cm, beam_height_cm, sample_height_cm, angles_deg,
             n_particles=100_000, n_batches=8, seed=1, by_order=True):
    """Multiple-scattering ratios computed with MCAMS.

    Returns a dict of (mean, err) pairs, each an array over the angles:
        'delta'   : (orders >= 2) / (order 1)     -> pycorrect.mscat()
        'delta2'  : (order 2) / (order 1)         -> table doublescattering.dat * mu_s/mu_t
        'ratio32' : (order 3) / (order 2)         -> how fast the orders decrease
    With by_order=False only 'delta' is computed, with the standard MCAMS
    function transmission_by_last_layer (single / multiple sums).
    """
    sample = create_cylindrical_sample(cm_to_mm(sample_height_cm), layers)
    W, Hb = cm_to_mm(beam_width_cm), cm_to_mm(beam_height_cm)
    out = {"delta": [], "delta2": [], "ratio32": []}
    for b in range(n_batches):
        if by_order:
            S, _ = mc_scattering_orders(sample, W, Hb, angles_deg, n_particles, rng=seed + b)
            out["delta"].append(S[1:].sum(axis=0) / S[0])
            out["delta2"].append(S[1] / S[0])
            out["ratio32"].append(S[2] / S[1])
        else:
            res = transmission_by_last_layer(sample, W, Hb, n_particles=n_particles,
                                             angles_deg=angles_deg, rng=seed + b)
            single = res.sum_attenuation[0].sum(axis=0)      # all layers
            multiple = res.sum_attenuation[1].sum(axis=0)
            out["delta"].append(multiple / single)
    result = {}
    for key, runs in out.items():
        if runs:
            runs = np.array(runs)
            result[key] = (runs.mean(axis=0), runs.std(axis=0, ddof=1) / math.sqrt(n_batches))
    return result


# ==========================================================================
#  One call: the MCAMS version of the pycorrect correction factors
# ==========================================================================
def mcams_factors_like_pycorrect(state, theta_deg, n_particles=200_000, n_batches=5,
                                 coarse_step_deg=10.0, sample_height_cm=None,
                                 ms_particles=50_000, seed=1):
    """Compute with MCAMS the attenuation and multiple-scattering factors that
    pycorrect.correct_2axis computes for the same set-up (a pycorrect
    CorrectState, e.g. `Correct().st` after reading a .com file).

    The factors are computed at coarse angles (every `coarse_step_deg`) and
    interpolated linearly (np.interp) onto `theta_deg`, the 2theta grid of the
    data, like pycorrect evaluates its factors at 10 angles and fits a
    polynomial.

    Returned keys (the same names as in the results of correct_2axis):
        'assch','acsch','acch','ahsch','ahch','ahh'  sample set-up factors
        'avvch'                                      vanadium rod factor
        'delta_sa'  array vs 2theta (MCAMS gives the angle dependence)
        'delta_va'  array vs 2theta
    Each also comes with '<key>_err' (statistical error).
    Only CYLINDRICAL geometry is handled (MCAMS also has slabs, but the
    pycorrect plate model has no multiple-scattering correction to compare).
    """
    from pycorrect.correct_2axis import _sample_params
    st = state
    if st.geometry != "CYLINDRICAL":
        raise ValueError("only the cylindrical geometry is handled here")
    p = _sample_params(st)                       # mu's exactly as pycorrect
    theta = np.asarray(theta_deg, float)
    coarse = np.arange(np.floor(theta.min() / coarse_step_deg) * coarse_step_deg,
                       theta.max() + coarse_step_deg, coarse_step_deg)
    r1, r2 = st.ca_r1, max(st.ca_r2, st.ca_r1)
    r3, r4 = st.fu_r1, st.fu_r2
    has_can = st.has("container") and r2 > r1
    has_fur = st.has("furnace") and r4 > r3
    if not has_can:
        r2 = 0.0
    if not has_fur:
        r3 = r4 = 0.0
    hs = sample_height_cm or max(2 * st.beam_height, 1.0)
    out = {}

    def run(smu, cmu, hmu, names_wanted):
        layers, names = cylinder_layers_from_pycorrect(r1, r2, r3, r4, smu, cmu, hmu)
        m, e = mc_attenuation_factors(layers, st.beam_width, coarse, n_particles, n_batches,
                                      sample_height_cm=hs, beam_height_cm=st.beam_height,
                                      seed=seed)
        for key, lname in names_wanted:
            k = names.index(lname)
            out[key] = np.interp(theta, coarse, m[k])
            out[key + "_err"] = np.interp(theta, coarse, e[k])

    ones = np.ones_like(theta)
    zeros = np.zeros_like(theta)
    # sample set-up: everything present
    want = [("assch", "sample")]
    if has_can:
        want.append(("acsch", "can"))
    if has_fur:
        want.append(("ahsch", "furnace"))
    run(p["sa_tmu"], p["ca_tmu"], p["fu_tmu"], want)
    if has_can:
        run(0.0, p["ca_tmu"], p["fu_tmu"],
            [("acch", "can")] + ([("ahch", "furnace")] if has_fur else []))
    if has_fur:
        run(0.0, 0.0, p["fu_tmu"], [("ahh", "furnace")])
    for key in ("acsch", "acch", "ahsch", "ahch", "ahh"):
        out.setdefault(key, ones.copy())
        out.setdefault(key + "_err", zeros.copy())

    # vanadium rod alone (pycorrect: va_furnace/va_container options not handled)
    if st.has("vanadium"):
        lay_v, _ = cylinder_layers_from_pycorrect(st.va_r, smu=p["va_tmu"])
        m, e = mc_attenuation_factors(lay_v, st.beam_width, coarse, n_particles, n_batches,
                                      sample_height_cm=hs, beam_height_cm=st.beam_height,
                                      seed=seed)
        out["avvch"], out["avvch_err"] = np.interp(theta, coarse, m[0]), np.interp(theta, coarse, e[0])

    # multiple scattering: sample height = beam height (Blech & Averbach / pycorrect)
    H = st.beam_height
    fs = p["sa_smu"] / p["sa_tmu"]
    fc = p["ca_smu"] / p["ca_tmu"]
    lay, _ = cylinder_layers_from_pycorrect(r1, r2 if has_can else 0.0, smu=p["sa_tmu"],
                                            cmu=p["ca_tmu"], s_frac=fs, c_frac=fc)
    d = mc_delta(lay, 2 * max(r1, r2) + 0.2, H, H, coarse, ms_particles, n_batches,
                 seed=seed, by_order=False)["delta"]
    out["delta_sa"], out["delta_sa_err"] = np.interp(theta, coarse, d[0]), np.interp(theta, coarse, d[1])
    if st.has("vanadium"):
        fv = p["va_smu"] / p["va_tmu"]
        lay_v, _ = cylinder_layers_from_pycorrect(st.va_r, smu=p["va_tmu"], s_frac=fv)
        d = mc_delta(lay_v, 2 * st.va_r + 0.2, H, H, coarse, ms_particles, n_batches,
                     seed=seed, by_order=False)["delta"]
        out["delta_va"], out["delta_va_err"] = np.interp(theta, coarse, d[0]), np.interp(theta, coarse, d[1])
    return out
