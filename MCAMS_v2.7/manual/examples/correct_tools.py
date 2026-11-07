"""
correct_tools.py - Helpers for the worked examples of the manual: the
attenuation factors in the notation of CORRECT (Paalman & Pings), and the
single, multiple and total scattering of the three measurements
I_sch, I_ch and I_h, computed with MCAMS.

This file is NOT part of the MCAMS package: it only uses its public
functions. Copy it next to your own scripts (or keep it in this folder) and
import it:   from correct_tools import ...

The notation of CORRECT
-----------------------
A diffraction experiment with a sample in a container inside a furnace (or
any sample environment: cryostat, pressure cell, ...) measures three things:

    I_sch : sample + container + heater
    I_ch  : empty container + heater (the same container, the same place)
    I_h   : heater alone (empty sample position)

Each of them is the sum of the scattering of its parts, each attenuated by
everything that is present in that measurement:

    I_sch = A_s,sch I_s + A_c,sch I_c + A_h,sch I_h
    I_ch  =               A_c,ch  I_c + A_h,ch  I_h
    I_h   =                             A_h,h   I_h

where I_s, I_c, I_h are the UNATTENUATED single scattering of sample,
container and heater, and A_x,y is the attenuation factor of the particles
scattered in x when the parts y are present: A_s,sch for the sample
attenuated by sample, container and heater; A_c,ch for the container
attenuated by container and heater; and so on. The three equations are
solved from the bottom up (Paalman & Pings):

    I_h = I_h(meas) / A_h,h
    I_c = (I_ch - A_h,ch I_h) / A_c,ch
    I_s = (I_sch - A_c,sch I_c - A_h,sch I_h) / A_s,sch

How MCAMS gives these factors
-----------------------------
One geometry, three samples. The layers of the real assembly are written
once, each with a ROLE:

    "s" : sample          "c" : container (one or several layers)
    "h" : heater / sample environment (one or several layers)
    "-" : empty space (a gap, an empty inner cylinder, ...)

and three samples are built from them: in the "ch" sample the layers of the
sample are made empty (mu = 0), in the "h" sample also the container. The
geometry (radii, thicknesses, heights) is the same in all three, exactly as
in the experiment.

The direct method of MCAMS (direct_attenuation.attenuation_factors_direct)
gives, for each sample, the attenuation factor A_k(2theta) of the particles
scattered once in layer k and attenuated by all the layers of that sample.
So, for example, the factor of the sample layer in the "sch" sample is
A_s,sch, and the factor of the container layers in the "ch" sample is A_c,ch.

When a part has SEVERAL layers (the inner and outer walls of an annular can,
the front and back windows of a flat cell), its factor is the average of the
factors of its layers weighted by how much each one scatters,

    A_part = sum_k w_k A_k / sum_k w_k ,   w_k = f_k mu_k <L_k>,

(f_k scatter fraction, mu_k attenuation coefficient, <L_k> mean path of the
beam in the layer), because the measured I_c is the sum of the scattering of
all the container layers.

Multiple scattering
-------------------
transmission_by_layer.transmission_by_last_layer follows every particle
through all its scatterings and gives, for each sample and each angle, the
single scattering S(2theta) and the multiple scattering M(2theta) towards
the detector, per incident particle and per steradian (isotropic
scattering). Their sum is what the detector measures. The ratio

    delta_y(2theta) = M_y / S_y      (y = sch, ch, h)

is the multiple scattering of the whole measurement y, all the parts
together (sample, container, heater, and all the cross terms such as
"scattered in the sample, then in the heater").

The correction "with the philosophy of CORRECT", but with the Monte Carlo:
    1. remove the multiple scattering from each measurement:
           I_y(single) = I_y / (1 + delta_y)
    2. apply the Paalman & Pings subtraction above with the A factors.
CORRECT does step 2 and then subtracts an analytical delta from the sample
only; here each measurement is corrected with its own simulated delta.

All the factors A_x,y (version 2.6 of the manual)
-------------------------------------------------
A part x (s, c or h) and a set y of parts present (any of the 7 runs s, c,
h, sc, sh, ch, sch) give 21 factors A_x,y:
  * x IN y (12 factors): real correction factors. Every set of runs needs
    some of them: sch/ch/h needs A_s,sch A_c,sch A_h,sch A_c,ch A_h,ch A_h,h;
    a run without furnace (sc, c) needs A_s,sc A_c,sc A_c,c; a sample
    without container in a furnace (sh, h) needs A_s,sh A_h,sh A_h,h; ...
  * x NOT in y (9 factors): the attenuation of the scattering of x by the
    OTHER parts only (e.g. A_s,ch: particles scattered in the sample,
    attenuated by container and heater, not by the sample itself). They
    never enter a correction, but they show how the attenuation is shared
    between the parts and whether products such as A_s,s * A_s,ch are a
    good approximation of A_s,sch.
To compute a factor with x not in y, the layers of x are made "ghosts":
they keep their geometry and get a vanishing attenuation coefficient
(GHOST_MU = 1e-12 per mm), so that points can be drawn in them while they
attenuate nothing. One direct-method run per set y then gives the factors
of the three parts (all_attenuation_factors).

With any set of runs, the unknowns I_s, I_c, I_h follow from the equations
    I_y(single) = sum_{x in y} A_x,y I_x
solved by weighted least squares (general_correction). For the nested runs
sch, ch, h this is exactly the Paalman & Pings chain; with more runs than
unknowns the residual (chi^2) tests the consistency of the measurements.

NumPy functions used in this file
---------------------------------
np.asarray, np.array : convert to an array.     np.where : choose element by element.
np.nansum            : sum ignoring NaN.        np.sqrt  : square root.
np.isfinite          : True where a number is not NaN or infinite.
np.linalg.lstsq      : least-squares solution of an overdetermined linear system.
"""

import math

import numpy as np

from sample_base import make_layer
from direct_attenuation import attenuation_factors_direct, mean_material_paths
from transmission_by_layer import transmission_by_last_layer

MEASUREMENTS = ("sch", "ch", "h")          # the three measurements of CORRECT
PARTS = ("s", "c", "h")                    # sample, container, heater
RUNS = ("s", "c", "h", "sc", "sh", "ch", "sch")   # every possible set of parts
GHOST_MU = 1e-12                           # 1/mm: a "ghost" layer scatters but does not attenuate


# --------------------------------------------------------------------------
# 1. The three samples
# --------------------------------------------------------------------------
def configuration(layers, roles, measurement, ghost=False):
    """Layers of one measurement: the layers whose role is not part of the
    `measurement` ("sch", "ch", "h", or any of RUNS) are made empty
    (mu = 0); "-" layers are always empty.

    ghost=True: the layers of the parts that are NOT in the measurement get
    mu = GHOST_MU instead of 0, so that the direct method can compute the
    attenuation of their scattering by the parts that ARE present (the
    factors A_x,y with x not in y).

    layers : list of layers (tuples, dicts or Layer objects), the real assembly
    roles  : one letter per layer: "s", "c", "h" or "-"
    Returns a new list of Layer objects (same thicknesses: same geometry).
    """
    if len(layers) != len(roles):
        raise ValueError("Give one role per layer.")
    out = []
    for layer, role in zip(layers, roles):
        lay = make_layer(layer)
        if role not in measurement:          # e.g. the sample in "ch"
            mu = GHOST_MU if (ghost and role in PARTS) else 0.0
            # dataclasses.replace would also do; we rebuild the record with the new mu
            lay = type(lay)(lay.name, lay.thickness, mu, lay.scatter_fraction,
                            lay.fill_height, lay.fullness, lay.bottom, lay.height)
        out.append(lay)
    return out


def correct_samples(make, layers, roles, runs=MEASUREMENTS):
    """The samples of the chosen runs (default the three of CORRECT: sch, ch
    and h), all with the same geometry.

    make  : a function that turns a list of layers into a sample, for example
            lambda L: create_cylindrical_sample(40.0, L)
    runs  : the runs wanted, e.g. RUNS for all seven.
    Returns a dict {"sch": sample, "ch": sample, "h": sample} (or the runs given).
    """
    return {m: make(configuration(layers, roles, m)) for m in runs}


def _indices(roles, part):
    """Indices of the layers that belong to `part` ("s", "c" or "h")."""
    return [k for k, r in enumerate(roles) if r == part]


# --------------------------------------------------------------------------
# 2. The attenuation factors A_x,y
# --------------------------------------------------------------------------
def attenuation_factors(samples, roles, beam_width, beam_height, angles_deg,
                        n_points=100_000, rng=1):
    """All the attenuation factors of CORRECT, with the direct method.

    Returns (A, err, mean_path):
      A, err    : dicts with keys "s,sch", "c,sch", "h,sch", "c,ch", "h,ch",
                  "h,h"; each value is an array over the angles.
      mean_path : <L_k> of every layer (mm), from the "sch" sample (the
                  geometry is the same in the three samples).
    """
    rng = np.random.default_rng(rng)
    A, err, mean_path = {}, {}, None
    for m in MEASUREMENTS:
        r = attenuation_factors_direct(samples[m], beam_width, beam_height,
                                       angles_deg=angles_deg, n_points=n_points, rng=rng)
        if m == "sch":
            mean_path = r.mean_path
        for part in m:                                   # the parts present in m
            k = _indices(roles, part)
            # weights w_k = f_k mu_k <L_k>: how much each layer of the part scatters
            w = (r.scatter_fractions * r.mus * r.mean_path)[k]
            A[f"{part},{m}"] = (w[:, None] * r.factors[k]).sum(axis=0) / w.sum()
            err[f"{part},{m}"] = np.sqrt(((w[:, None] * r.errors[k]) ** 2).sum(axis=0)) / w.sum()
    return A, err, mean_path


def unattenuated(sample, roles, beam_width, beam_height, n_runs=10, rng=3):
    """Unattenuated single scattering per incident particle and per steradian
    (isotropic) of each part: I_x = sum_k f_k mu_k <L_k> / (4 pi), k in part x.
    These are the I_s, I_c, I_h of the equations of CORRECT. Use the "sch"
    sample (all the parts present).

    The correction itself does NOT need them (it uses only the A factors and
    delta); they are the reference for the closure test, and the absolute
    level of the scattering of each part.

    <L_k> is the mean path of the beam in layer k. It is estimated by
    sampling (direct_attenuation.mean_material_paths); when most of the
    volume around the sample is empty (a furnace of large radius) its error
    can be a few 0.1 %, so it is averaged here over `n_runs` independent
    runs of 2e6 points.
    Returns (I, rel_err): two dicts {"s": ..., "c": ..., "h": ...}.
    """
    rng = np.random.default_rng(rng)
    runs = [mean_material_paths(sample, beam_width, beam_height, n_draws=2_000_000,
                                n_lines=200_000, rng=rng) for _ in range(n_runs)]
    L = np.mean([r[0] for r in runs], axis=0)                       # mean path of each layer
    L_err = np.sqrt(np.sum([r[1] ** 2 for r in runs], axis=0)) / n_runs
    w = sample.scatter_fractions * sample.mus
    I, rel = {}, {}
    for p in PARTS:
        k = _indices(roles, p)
        I[p] = (w[k] * L[k]).sum() / (4 * math.pi)
        rel[p] = np.sqrt(((w[k] * L_err[k]) ** 2).sum()) / (w[k] * L[k]).sum()
    return I, rel


def all_attenuation_factors(make, layers, roles, beam_width, beam_height, angles_deg,
                            n_points=100_000, rng=1):
    """All the factors A_x,y: x in s, c, h and y in the 7 runs of RUNS.

    For each run y one sample is built in which the parts in y are real, the
    other parts are ghosts (they scatter but do not attenuate, see
    `configuration`), and one direct-method run gives the factors of the
    three parts. For x in y these are the usual factors; for x not in y they
    are the attenuation of the scattering of x by the parts of y only.
    Parts with several layers are combined with the weights f_k mu_k <L_k>
    of the REAL layers.
    Returns (A, err): dicts with keys "s,sch", "s,s", "c,sh", ... (21 keys
    if the assembly has the three parts).
    """
    rng = np.random.default_rng(rng)
    real = [make_layer(l) for l in layers]
    w_real = np.array([l.scatter_fraction * l.mu for l in real])        # f_k mu_k
    present = [p for p in PARTS if p in roles]
    A, err = {}, {}
    for y in RUNS:
        if not all(p in present for p in y):
            continue                                                     # e.g. no heater
        smp = make(configuration(layers, roles, y, ghost=True))
        r = attenuation_factors_direct(smp, beam_width, beam_height,
                                       angles_deg=angles_deg, n_points=n_points, rng=rng)
        for x in present:
            k = _indices(roles, x)
            w = (w_real * r.mean_path)[k]
            A[f"{x},{y}"] = (w[:, None] * r.factors[k]).sum(axis=0) / w.sum()
            err[f"{x},{y}"] = np.sqrt(((w[:, None] * r.errors[k]) ** 2).sum(axis=0)) / w.sum()
    return A, err


def general_correction(I, A, delta=None, rel_err=None):
    """Unattenuated I_s, I_c, I_h from ANY set of runs, by least squares.

    I       : dict run -> measured intensity versus angle, e.g.
              {"sch": ..., "ch": ..., "h": ...} or {"sc": ..., "c": ...}
    A       : dict of factors (attenuation_factors or all_attenuation_factors)
    delta   : dict run -> multiple/single ratio (None: no correction)
    rel_err : dict run -> relative error of the measured intensity (array or
              number); default 1 % for every run. Used as weights.

    For every angle the equations  I_y / (1 + delta_y) = sum_{x in y} A_x,y I_x
    (one per run) are solved for the parts that appear in the runs, with
    np.linalg.lstsq after dividing each equation by its error. With as many
    runs as unknowns (sch, ch, h) the solution is exactly the Paalman &
    Pings chain; with more, chi2 = sum of squared normalised residuals
    measures their consistency (about n_runs - n_parts if all is well).
    Returns (parts, errors, chi2): dicts x -> I_x and x -> its error (from
    the covariance of the least-squares solution, (M^T W M)^-1), and chi2,
    all versus angle.
    """
    runs = list(I)
    unknowns = [p for p in PARTS if any(p in y for y in runs)]
    n_ang = len(np.atleast_1d(I[runs[0]]))
    b = np.array([np.asarray(I[y], float) / (1 + (delta[y] if delta is not None else 0.0))
                  for y in runs])                                        # (runs, angles)
    sig = np.array([np.broadcast_to(0.01 if rel_err is None else rel_err[y], (n_ang,))
                    for y in runs]) * np.abs(b)
    out = np.zeros((len(unknowns), n_ang))
    out_err = np.zeros((len(unknowns), n_ang))
    chi2 = np.zeros(n_ang)
    for a in range(n_ang):
        # matrix of the system: one row per run, one column per unknown part
        M = np.array([[A[f"{x},{y}"][a] if x in y else 0.0 for x in unknowns] for y in runs])
        Mw = M / sig[:, a:a + 1]                       # each equation divided by its error
        sol, *_ = np.linalg.lstsq(Mw, b[:, a] / sig[:, a], rcond=None)
        out[:, a] = sol
        out_err[:, a] = np.sqrt(np.diag(np.linalg.inv(Mw.T @ Mw)))   # errors of the solution
        chi2[a] = np.sum(((M @ sol - b[:, a]) / sig[:, a]) ** 2)
    return ({x: out[i] for i, x in enumerate(unknowns)},
            {x: out_err[i] for i, x in enumerate(unknowns)}, chi2)


def print_all_factors(angles, A, every=1):
    """The 21 factors as a table per angle: one row per scattering part x,
    one column per set of parts y. Factors with x not in y are marked *."""
    present = [p for p in PARTS if any(k.startswith(p + ",") for k in A)]
    for i in range(0, len(angles), every):
        print(f"2theta = {angles[i]:g} deg       " + "".join(f"{'y = ' + y:>10}" for y in RUNS))
        for x in present:
            row = f"   A_{x},y              "
            for y in RUNS:
                key = f"{x},{y}"
                row += f"{A[key][i]:9.4f}{'*' if x not in y else ' '}" if key in A else f"{'-':>10}"
            print(row)
    print("(* = x not in y: attenuation of the scattering of x by the other parts only)")


# --------------------------------------------------------------------------
# 3. Single, multiple and total scattering of the three measurements
# --------------------------------------------------------------------------
def scattering(samples, roles, beam_width, beam_height, angles_deg,
               n_particles=200_000, n_batches=10, rng=2):
    """Single and multiple scattering of the runs in `samples` (the three of
    CORRECT, or any of RUNS).

    For each run y, transmission_by_last_layer is run `n_batches` times. Its
    sums divided by 4 pi and by the number of incident particles are
    intensities per incident particle and per steradian (isotropic
    scattering). The errors are the standard errors of the mean of the
    batches. Use at least 10 batches: with fewer, the error estimate itself
    is very uncertain (with 4 batches it can easily be wrong by a factor of
    2). Returns a dict: result[y] = {
        "single", "multiple", "total" : (mean, error) over the angles,
        "delta"                       : (mean, error) of M/S,
        "single_by_part", "multiple_by_part" : dict part -> array, split by
                                        the part of the LAST scattering,
        "batches"                     : (S, M), the single and multiple
                                        scattering of every batch, shape
                                        (n_batches, A) each }.
    """
    rng = np.random.default_rng(rng)
    out = {}
    for m in samples:
        S, M, Sp, Mp = [], [], [], []
        for _ in range(n_batches):
            r = transmission_by_last_layer(samples[m], beam_width, beam_height,
                                           n_particles=n_particles,
                                           angles_deg=angles_deg, rng=rng)
            per_sr = r.sum_attenuation / (4 * math.pi * r.n_incident)   # (2, L, A)
            S.append(per_sr[0].sum(axis=0))
            M.append(per_sr[1].sum(axis=0))
            Sp.append({p: per_sr[0, _indices(roles, p)].sum(axis=0) for p in m})
            Mp.append({p: per_sr[1, _indices(roles, p)].sum(axis=0) for p in m})
        S, M = np.array(S), np.array(M)
        stat = lambda x: (x.mean(axis=0), x.std(axis=0, ddof=1) / math.sqrt(len(x)))
        out[m] = {"single": stat(S), "multiple": stat(M), "total": stat(S + M),
                  "delta": stat(M / S),
                  "single_by_part": {p: np.mean([d[p] for d in Sp], axis=0) for p in m},
                  "multiple_by_part": {p: np.mean([d[p] for d in Mp], axis=0) for p in m},
                  "batches": (S, M)}
    return out


# --------------------------------------------------------------------------
# 4. The correction chain of CORRECT, with the factors of MCAMS
# --------------------------------------------------------------------------
def correct_chain(I_sch, I_ch, I_h, A, delta=None):
    """Paalman & Pings subtraction, after removing the multiple scattering.

    I_sch, I_ch, I_h : measured (here: simulated total) intensities vs angle
    A                : dict of attenuation factors (attenuation_factors)
    delta            : dict {"sch": d, "ch": d, "h": d} of multiple/single
                       ratios; None = no multiple-scattering correction.
    Returns (I_s, I_c, I_h): the unattenuated single scattering of the three
    parts.
    """
    if delta is not None:                       # step 1: remove the multiple scattering
        I_sch = I_sch / (1 + delta["sch"])
        I_ch = I_ch / (1 + delta["ch"])
        I_h = I_h / (1 + delta["h"])
    h = I_h / A["h,h"]                          # step 2: Paalman & Pings, bottom up
    c = (I_ch - A["h,ch"] * h) / A["c,ch"]
    s = (I_sch - A["c,sch"] * c - A["h,sch"] * h) / A["s,sch"]
    return s, c, h


def at_angles(values, angles, new_angles):
    """Interpolate a dict of curves (factors A, or deltas) from `angles` onto
    `new_angles` (linear interpolation, np.interp). Use it to bring the
    factors (fine grid) onto the angles of the simulated runs, or both onto
    the angles of the measured data."""
    return {k: np.interp(new_angles, angles, v) for k, v in values.items()}


# --------------------------------------------------------------------------
# 5. Printing
# --------------------------------------------------------------------------
def print_factors(angles, A, err, every=1):
    """Table of the six factors versus angle."""
    keys = ["s,sch", "c,sch", "h,sch", "c,ch", "h,ch", "h,h"]
    print(" 2theta " + "".join(f"{'A_' + k:>11}" for k in keys))
    for i in range(0, len(angles), every):
        print(f"{angles[i]:7.1f} " + "".join(f"{A[k][i]:11.4f}" for k in keys))
    rel = max(np.nanmax(err[k] / A[k]) for k in keys)
    print(f"(largest relative error {100 * rel:.2f} %)")


def print_scattering(angles, sc, every=1):
    """Table of single, multiple and delta of the three measurements."""
    print(" 2theta" + "".join(f"{'S_' + m:>10}{'M_' + m:>10}{'d_' + m:>8}" for m in MEASUREMENTS))
    for i in range(0, len(angles), every):
        row = f"{angles[i]:7.1f}"
        for m in MEASUREMENTS:
            row += (f"{1e3 * sc[m]['single'][0][i]:10.4f}{1e3 * sc[m]['multiple'][0][i]:10.4f}"
                    f"{sc[m]['delta'][0][i]:8.3f}")
        print(row)
    print("(S, M in 1e-3 per incident particle per steradian; d = M/S)")


def check_closure(angles, sc, A, true, true_rel_err=None, every=1):
    """Feed the simulated totals to the correction chain and compare the
    result with the true unattenuated scattering of each part.

    The chain is applied batch by batch (each batch is an independent
    simulation), so the spread of the results gives their statistical
    error. Since delta comes from the same simulation, step 1 gives back
    exactly the simulated single scattering: the test checks the attenuation
    part (Monte Carlo single scattering against the factors of the direct
    method) and the bookkeeping of the subtraction. The last column shows
    what is left if the multiple scattering is NOT removed.
    Returns the relative errors (mean, error) of I_s, I_c, I_h.
    """
    res = {"s": [], "c": [], "h": [], "s_noms": []}
    nb = len(sc["sch"]["batches"][0])
    for b in range(nb):
        tot = {m: sc[m]["batches"][0][b] + sc[m]["batches"][1][b] for m in MEASUREMENTS}
        delta = {m: sc[m]["batches"][1][b] / sc[m]["batches"][0][b] for m in MEASUREMENTS}
        s, c, h = correct_chain(tot["sch"], tot["ch"], tot["h"], A, delta)
        s0, _, _ = correct_chain(tot["sch"], tot["ch"], tot["h"], A, None)
        res["s"].append(s / true["s"] - 1); res["c"].append(c / true["c"] - 1)
        res["h"].append(h / true["h"] - 1); res["s_noms"].append(s0 / true["s"] - 1)
    st = {k: (np.mean(v, axis=0), np.std(v, axis=0, ddof=1) / math.sqrt(nb)) for k, v in res.items()}
    if true_rel_err is not None:      # add the error of the reference (mean paths)
        for k in ("s", "c", "h"):
            st[k] = (st[k][0], np.hypot(st[k][1], true_rel_err[k]))
    print(" 2theta     I_s/true - 1       I_c/true - 1       I_h/true - 1    no MS corr.: I_s/true - 1")
    for i in range(0, len(angles), every):
        row = f"{angles[i]:7.1f}"
        for k in ("s", "c", "h"):
            row += f"  {st[k][0][i]:+8.4f} {st[k][1][i]:.4f}"
        print(row + f"   {st['s_noms'][0][i]:+8.4f}")
    print("(mean and statistical error: batches of the simulation and reference <L>)")
    return st


def print_contributions(angles, sc, measurement="sch", every=1):
    """Share of each part (by the layer of the LAST scattering) in the single
    and in the multiple scattering of one measurement."""
    d = sc[measurement]
    parts = list(d["single_by_part"])
    S, M = d["single"][0], d["multiple"][0]
    print(f"Contributions to I_{measurement} (part of the last scattering), % of the single / % of the multiple")
    print(" 2theta" + "".join(f"{'single ' + p:>11}" for p in parts) + "".join(f"{'multiple ' + p:>13}" for p in parts))
    for i in range(0, len(angles), every):
        print(f"{angles[i]:7.1f}" + "".join(f"{100 * d['single_by_part'][p][i] / S[i]:11.1f}" for p in parts)
              + "".join(f"{100 * d['multiple_by_part'][p][i] / M[i]:13.1f}" for p in parts))


def plot_correct(angles, A, sc, title, save_as, angles_mc=None):
    """Two panels: the six attenuation factors (on `angles`), and delta = M/S
    of the three measurements (on `angles_mc`, the angles of the simulated
    runs; default the same), versus 2theta. Saved to `save_as` (PDF or PNG)."""
    angles_mc = angles if angles_mc is None else angles_mc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.8))
    styles = {"s,sch": ("#4C78A8", "-"), "c,sch": ("#F58518", "-"), "h,sch": ("#54A24B", "-"),
              "c,ch": ("#F58518", "--"), "h,ch": ("#54A24B", "--"), "h,h": ("#54A24B", ":")}
    for k, (col, ls) in styles.items():
        a1.plot(angles, A[k], ls, color=col, lw=1.1, label=f"$A_{{{k}}}$")
    a1.set_xlabel(r"$2\theta$ (deg)"); a1.set_ylabel("attenuation factor")
    a1.set_xlim(0, 180); a1.set_xticks(range(0, 181, 30)); a1.grid(alpha=0.3)
    a1.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    a1.set_title("(a) attenuation factors", fontsize=9)
    for m, col in (("sch", "#4C78A8"), ("ch", "#F58518"), ("h", "#54A24B")):
        mean, err = sc[m]["delta"]
        # the line is the simulated delta, the shaded band its statistical error
        a2.fill_between(angles_mc, 100 * (mean - err), 100 * (mean + err), color=col, alpha=0.3, lw=0)
        a2.plot(angles_mc, 100 * mean, "-", lw=1.0, color=col,
                    label=f"$\\delta_{{{m}}}$ = M/S of $I_{{{m}}}$")
    a2.set_yscale("log")             # delta_sch is much larger than delta_ch and delta_h
    from matplotlib.ticker import NullFormatter
    a2.yaxis.set_minor_formatter(NullFormatter())   # label only the decades
    a2.set_xlabel(r"$2\theta$ (deg)"); a2.set_ylabel("multiple / single (%)")
    a2.set_xlim(0, 180); a2.set_xticks(range(0, 181, 30)); a2.grid(alpha=0.3)
    a2.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    a2.grid(alpha=0.3, which="both")
    a2.set_title("(b) multiple scattering of each measurement", fontsize=9)
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(save_as, bbox_inches="tight")
    plt.close(fig)


def plot_all_factors(angles, A, title, save_as):
    """Three panels (scattering part s, c, h): A_x,y versus 2theta for every
    set y; solid lines for x in y (correction factors), dashed for x not in
    y (attenuation by the other parts only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"s": "#4C78A8", "c": "#F58518", "h": "#54A24B", "sc": "#B279A2",
              "sh": "#9D755D", "ch": "#E45756", "sch": "#000000"}
    present = [p for p in PARTS if any(k.startswith(p + ",") for k in A)]
    fig, axes = plt.subplots(1, len(present), figsize=(7.8, 3.4), sharey=True)
    names = {"s": "sample", "c": "container", "h": "heater"}
    for ax, x in zip(np.atleast_1d(axes), present):
        for y in RUNS:
            key = f"{x},{y}"
            if key in A:
                ax.plot(angles, A[key], "-" if x in y else "--", color=colors[y], lw=1.2,
                        label=f"y = {y}")
        ax.set_title(f"scattered in the {names[x]}: $A_{{{x},y}}$", fontsize=9)
        ax.set_xlabel(r"$2\theta$ (deg)"); ax.set_xlim(0, 180); ax.set_xticks(range(0, 181, 60))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("attenuation factor")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(save_as, bbox_inches="tight")
    plt.close(fig)

