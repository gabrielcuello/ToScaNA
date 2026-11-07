"""
compare_vertical_extent.py - Partially filled containers and samples that do
not start at the bottom of the beam: the "fullness" treatment of CORRECT
(pycorrect) tested with MCAMS 2.2 (layers with their own bottom and height).

What CORRECT does (correct_2axis, steps 2, 6 and 9)
---------------------------------------------------
With `sample ... /fullness=f` CORRECT assumes that a fraction f of the beam
height sees the full container and the rest sees the EMPTY container:

    I_sc,full = ( I_sc - (1 - f) I_c ) / f                       (fullness step)
    I_s       = ( I_sc,full - A_c,sc I_c / A_c,c ) / A_s,sc       (Paalman & Pings)
    dsigma/dOmega  ∝  I_s / I_true  - delta_C                     (multiple scattering)

with A's of the FULL can, I_true the unattenuated single scattering of a
sample filling the whole beam height, and delta_C from Blech & Averbach with
the FULL beam height H.

What is simulated
-----------------
For each geometry MCAMS gives, at exact detector angles, the intensity per
incident particle and per steradian of the sample in its can (I_sc) and of
the empty can (I_c), split into single and multiple scattering (point
estimator of transmission_by_layer.py, all orders, isotropic scattering).
These "measurements" go through the steps above exactly as CORRECT would
process them, and the result is compared with the truth. The relative error

    e(2theta) = I_s / I_true - 1 - delta_C

is what remains in the corrected cross section (in units of the single
scattering level). It is split into
    e_single : the same steps applied to the single-scattering part only
               (tests the fullness step and the attenuation factors);
    delta_eff: the multiple scattering left after the steps (what CORRECT
               should subtract), to compare with delta_C.

Cases: cylinder (Ni powder in a V can) with the powder level at several
heights, a powder band resting on a support, a can whose bottom is inside the
beam, a non-uniform vertical beam profile; and a flat plate cell.

Usage:  python compare_vertical_extent.py [--quick]
Output: results/V_*.txt, figures/V_*.pdf  (used by the report, chapter 9)
"""
import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyc_mcams_bridge as br                                   # puts MCAMS on sys.path
from cylinder import create_cylindrical_sample                  # MCAMS 2.2
from slab import create_slab_sample                             # MCAMS 2.2
from sample_base import Layer                                   # MCAMS 2.2
from transmission_by_layer import transmission_by_last_layer    # MCAMS
from pycorrect import atcors, mscat, plate

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- materials (mm, 1/mm)
MU_NI, F_NI = 0.18027, 0.9370          # Ni at 0.4985 A: total mu, mu_s/mu_t
MU_V, F_V = 0.04591, 0.7785            # V
R1, R2 = 3.0, 3.2                       # powder radius, can outer radius (mm)
W, HB = 15.0, 20.0                      # beam width and height (mm): y from -10 to +10
CAN_BOTTOM, CAN_HEIGHT = -30.0, 60.0    # standard can: from y = -30 to +30 mm


# ==========================================================================
#  The "measurement": intensity per incident particle per steradian
# ==========================================================================
def measure(sample, angles, n, nb, seed, V=None):
    """Single and multiple scattering per incident particle per steradian
    (isotropic scattering), for nb independent runs: arrays (nb, A) each.
    The point estimator adds, for every scattering event, the attenuation
    towards the detector; dividing by 4 pi and by the number of incident
    particles gives the intensity per steradian."""
    s, m = [], []
    for b in range(nb):
        r = transmission_by_last_layer(sample, W, HB, V=V, n_particles=n,
                                       angles_deg=angles, rng=seed + b)
        s.append(r.sum_attenuation[0].sum(axis=0) / (4 * math.pi * r.n_incident))
        m.append(r.sum_attenuation[1].sum(axis=0) / (4 * math.pi * r.n_incident))
    return np.array(s), np.array(m)


def can_sample(powder_bottom=None, powder_height=None, can_bottom=CAN_BOTTOM,
               can_height=CAN_HEIGHT, powder_mu=MU_NI):
    """Ni powder (band from powder_bottom, height powder_height) in a V can
    (from can_bottom, height can_height). Heights in mm from the beam centre.
    powder_mu = 0 gives the empty can (same geometry)."""
    pb = can_bottom if powder_bottom is None else powder_bottom
    ph = can_height if powder_height is None else powder_height
    return create_cylindrical_sample(
        None, [Layer("Ni", 2 * R1, powder_mu, F_NI, bottom=pb, height=ph),
               Layer("V", R2 - R1, MU_V, F_V, bottom=can_bottom, height=can_height)],
        beam_height=HB)


# ==========================================================================
#  CORRECT's processing (pycorrect functions, cm units)
# ==========================================================================
def pp_factors_cylinder(angles):
    """A_s,sc, A_c,sc, A_c,c of the FULL can from pycorrect.atcors (cm)."""
    a = np.array([atcors(math.radians(t), R1 / 10, R2 / 10, 0, 0, W / 10,
                         MU_NI * 10, MU_V * 10, 0) for t in angles])
    return a[:, 0], a[:, 2], a[:, 1]


def delta_correct_cylinder(height_mm=HB):
    """CORRECT's delta for the sample in its can: effective cylinder of radius
    R2 with volume-averaged mu (correct_2axis step 9), height = beam height."""
    fs = (R1 / R2) ** 2
    tmu = ((1 - fs) * MU_V + fs * MU_NI) * 10
    smu = ((1 - fs) * F_V * MU_V + fs * F_NI * MU_NI) * 10
    return mscat(R2 / 10, height_mm / 10, tmu, smu, verbose=False)


def correct_chain(Isc, Ic, f, Assc, Acsc, Acc, f_can=1.0):
    """Fullness step + Paalman & Pings.

    f_can = 1 (default) is exactly correct_2axis. f_can < 1 is the fixed
    recipe for a can that covers only the fraction f_can of the beam height
    (its bottom inside the beam): the part of the beam that sees the empty
    can is (f_can - f), and the full-can intensity is I_c / f_can,
        I_sc,full = ( I_sc - (f_can - f)/f_can I_c ) / f ,   I_c,full = I_c / f_can
    """
    Isc_full = (Isc - (f_can - f) / f_can * Ic) / f
    Ic_full = Ic / f_can
    return (Isc_full - Acsc * Ic_full / Acc) / Assc


I_TRUE_CYL = F_NI * MU_NI * (math.pi * R1 ** 2 / W) / (4 * math.pi)   # full-height Ni, no attenuation


def run_case(name, sc, can, f_correct, angles, n, nb, V=None, f_alt=None, f_can_alt=1.0):
    """Simulate one cylinder case and push it through CORRECT. Returns a dict
    with e (total residual), e_single, delta_eff (mean and error over runs)."""
    t0 = time.time()
    s_sc, m_sc = measure(sc, angles, n, nb, 100, V)
    s_c, m_c = measure(can, angles, n, nb, 200, V)
    Assc, Acsc, Acc = pp_factors_cylinder(angles)
    dC = delta_correct_cylinder()
    out = {"angles": angles, "f": f_correct, "delta_C": dC,
           "fullness_mcams": float(sc.beam_fullness()[0])}

    def stats(x):
        return x.mean(axis=0), x.std(axis=0, ddof=1) / math.sqrt(len(x))

    variants = [("", f_correct, 1.0)]
    if f_alt is not None or f_can_alt != 1.0:      # the "fixed recipe" variant
        variants.append(("_alt", f_correct if f_alt is None else f_alt, f_can_alt))
    for tag, f, fc in variants:
        Is_single = correct_chain(s_sc, s_c, f, Assc, Acsc, Acc, fc) / I_TRUE_CYL
        Is_all = correct_chain(s_sc + m_sc, s_c + m_c, f, Assc, Acsc, Acc, fc) / I_TRUE_CYL
        out["e_single" + tag] = stats(Is_single - 1)
        out["delta_eff" + tag] = stats(Is_all - Is_single)
        out["e" + tag] = stats(Is_all - 1 - dC)
    print(f"  {name}: {time.time() - t0:.0f} s")
    return out


# ==========================================================================
#  Cases
# ==========================================================================
def cylinder_cases(q):
    angles = np.arange(10.0, 171.0, 20.0)
    n, nb = (20_000, 4) if q else (100_000, 8)
    can = can_sample(powder_mu=0.0)
    cases = {}
    # V1: powder from the bottom of the can (y=-30) up to a level L
    for L in (10.0, 5.0, 0.0, -5.0):
        f = (min(L, HB / 2) + HB / 2) / HB
        sc = can_sample(CAN_BOTTOM, L - CAN_BOTTOM)
        cases[f"level {L:+.0f} mm"] = run_case(f"V1 level {L:+.0f}", sc, can, f, angles, n, nb)
    # V2: powder band resting on a support: from y = +2 to +30 (lower part of
    # the beam sees the empty can), f = 8/20 = 0.4
    sc = can_sample(2.0, 28.0)
    cases["band +2..+30 mm"] = run_case("V2 band", sc, can, 0.4, angles, n, nb)
    # V3: can whose bottom is INSIDE the beam (y = -5): the beam below it
    # misses everything. Powder from -5 to +3 (f = 0.4), can covers 0.75.
    can3 = can_sample(powder_mu=0.0, can_bottom=-5.0, can_height=60.0)
    sc3 = can_sample(-5.0, 8.0, can_bottom=-5.0, can_height=60.0)
    # alt: fixed recipe with the fraction of beam covered by the can, f_can = 0.75
    cases["can bottom at -5 mm"] = run_case("V3 can bottom in beam", sc3, can3, 0.4, angles, n, nb,
                                            f_can_alt=0.75)
    cases["can bottom at -5 mm"]["alt_label"] = "e_single(f_can=0.75)  e_total(f_can=0.75)"
    # V4: Gaussian vertical beam profile (sigma 5 mm), level at -4 mm:
    # geometric f = 6/20 = 0.30; intensity-weighted f = Phi(-4/5) in the beam
    V = lambda y: np.exp(-y ** 2 / (2 * 5.0 ** 2))
    yy = np.linspace(-10, 10, 200001)
    w = V(yy)
    f_int = w[yy <= -4].sum() / w.sum()
    sc4 = can_sample(CAN_BOTTOM, -4.0 - CAN_BOTTOM)
    cases["Gaussian V(y), level -4 mm"] = run_case("V4 Gaussian profile", sc4, can, 0.30,
                                                   angles, n, nb, V=V, f_alt=f_int)
    cases["Gaussian V(y), level -4 mm"]["f_intensity"] = f_int
    return cases


def slab_case(q):
    """Flat cell: V window 0.2 mm, Ni 2.0 mm, V window 0.2 mm, 40 mm wide,
    normal incidence; Ni from y = -30 up to the level L. CORRECT: plate
    geometry (pycorrect.plate), no multiple-scattering correction."""
    angles = np.array([10, 30, 50, 70, 110, 130, 150, 170.0])
    n, nb = (20_000, 4) if q else (100_000, 8)
    t_s, t_w = 2.0, 0.2

    def cell(level, mu_ni=MU_NI):
        return create_slab_sample(40.0, 60.0, [
            Layer("V1", t_w, MU_V, F_V),
            Layer("Ni", t_s, mu_ni, F_NI, bottom=-30.0, height=level + 30.0),
            Layer("V2", t_w, MU_V, F_V)], y_bottom=-30.0, beam_height=HB)
    empty = cell(30.0, 0.0)
    pl = np.array([plate(math.radians(a), 0.0, t_s / 10, (t_s + 2 * t_w) / 10,
                         MU_NI * 10, MU_V * 10) for a in angles])
    Assc, Acc, Acsc = pl[:, 0], pl[:, 1], pl[:, 2]
    # the same with the original CORRECT 2.24 formulas (faithful=True), which
    # are wrong for A_c,sc in reflection geometry (2theta > 90 deg)
    pf = np.array([plate(math.radians(a), 0.0, t_s / 10, (t_s + 2 * t_w) / 10,
                         MU_NI * 10, MU_V * 10, faithful=True) for a in angles])
    I_true = F_NI * MU_NI * t_s / (4 * math.pi)
    out = {}
    for L in (10.0, 0.0):
        f = (min(L, HB / 2) + HB / 2) / HB
        t0 = time.time()
        s_sc, m_sc = measure(cell(L), angles, n, nb, 300)
        s_c, m_c = measure(empty, angles, n, nb, 400)
        Is_single = correct_chain(s_sc, s_c, f, Assc, Acsc, Acc) / I_true
        Is_all = correct_chain(s_sc + m_sc, s_c + m_c, f, Assc, Acsc, Acc) / I_true
        Is_single_f = correct_chain(s_sc, s_c, f, pf[:, 0], pf[:, 2], pf[:, 1]) / I_true
        Is_all_f = correct_chain(s_sc + m_sc, s_c + m_c, f, pf[:, 0], pf[:, 2], pf[:, 1]) / I_true
        st = lambda x: (x.mean(axis=0), x.std(axis=0, ddof=1) / math.sqrt(len(x)))
        out[f"level {L:+.0f} mm"] = {"angles": angles, "f": f, "delta_C": 0.0,
                                     "e_single": st(Is_single - 1),
                                     "delta_eff": st(Is_all - Is_single),
                                     "e": st(Is_all - 1),
                                     "e_single_alt": st(Is_single_f - 1),
                                     "e_alt": st(Is_all_f - 1),
                                     "alt_label": "e_single(CORRECT 2.24 plate)  e_total(CORRECT 2.24 plate)"}
        print(f"  slab level {L:+.0f}: {time.time() - t0:.0f} s")
    return out


# ==========================================================================
#  Output
# ==========================================================================
def write_table(cases, fname, title):
    with open(os.path.join(RES, fname), "w") as fh:
        fh.write(f"# {title}\n")
        for name, c in cases.items():
            fh.write(f"\n## {name}   fullness given to CORRECT f = {c['f']:.3f}"
                     + (f"   (MCAMS beam_fullness = {c['fullness_mcams']:.3f})" if "fullness_mcams" in c else "")
                     + (f"   (intensity-weighted f = {c['f_intensity']:.3f})" if "f_intensity" in c else "")
                     + f"   delta_C = {c['delta_C']:.4f}\n")
            alt = c.get("alt_label", "e_single(f_alt)  e_total(f_alt)")
            fh.write("# 2theta   e_single   err    delta_eff   err    e_total   err"
                     + (f"   {alt}" if "e_alt" in c else "") + "\n")
            for i, a in enumerate(c["angles"]):
                row = (f"{a:7.1f} {c['e_single'][0][i]:9.4f} {c['e_single'][1][i]:7.4f}"
                       f" {c['delta_eff'][0][i]:9.4f} {c['delta_eff'][1][i]:7.4f}"
                       f" {c['e'][0][i]:9.4f} {c['e'][1][i]:7.4f}")
                if "e_alt" in c:
                    row += (f" {c['e_single_alt'][0][i]:12.4f} {c['e_single_alt'][1][i]:7.4f}"
                            f" {c['e_alt'][0][i]:14.4f}")
                fh.write(row + "\n")


def plots(cyl, slab):
    # (a) residual of the single-scattering part, all cylinder cases
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, (name, c) in enumerate(list(cyl.items())[:4]):
        ax.errorbar(c["angles"], 100 * c["e_single"][0], yerr=100 * c["e_single"][1], fmt="o-",
                    ms=3, lw=1, color=C[i % 4], mfc="white", label=f"{name} (f = {c['f']:.2f})")
    for name, mk, col in (("band +2..+30 mm", "s", INK2), ("can bottom at -5 mm", "D", C[1])):
        c = cyl[name]
        ax.errorbar(c["angles"], 100 * c["e_single"][0], yerr=100 * c["e_single"][1], fmt=mk + "--",
                    ms=3, lw=1, color=col, mfc="white", label=f"{name} (f = {c['f']:.2f})")
    c = cyl["can bottom at -5 mm"]                  # the same with the fixed recipe
    ax.errorbar(c["angles"] + 2, 100 * c["e_single_alt"][0], yerr=100 * c["e_single_alt"][1],
                fmt="D:", ms=3, lw=1, color=C[2], label="can bottom at -5 mm, fixed (f_c = 0.75)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("2θ (deg)"); ax.set_ylabel("error of single part (%)")
    ax.set_xticks(range(0, 181, 30)); ax.set_xlim(0, 180); ax.set_ylim(-2.6, 0.8)
    ax.legend(fontsize=6.5, ncol=2, loc="lower center")
    fig.savefig(os.path.join(FIG, "V_single_residual.pdf")); plt.close(fig)

    # (b) multiple scattering: delta_eff versus angle for the levels, and delta_C
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, (name, c) in enumerate(list(cyl.items())[:4]):
        ax.errorbar(c["angles"], c["delta_eff"][0], yerr=c["delta_eff"][1], fmt="o-", ms=3, lw=1,
                    color=C[i % 4], mfc="white", label=f"MCAMS, {name} (f = {c['f']:.2f})")
    dC = list(cyl.values())[0]["delta_C"]
    ax.axhline(dC, color=INK, lw=1.4, ls="--", label=f"CORRECT δ (beam height), {dC:.3f}")
    ax.set_xlabel("2θ (deg)"); ax.set_ylabel("multiple / single after the steps")
    ax.set_xticks(range(0, 181, 30)); ax.set_xlim(0, 180)
    ax.set_ylim(0.25, 0.72)
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    fig.savefig(os.path.join(FIG, "V_multiple.pdf")); plt.close(fig)

    # (c) Gaussian profile: geometric vs intensity-weighted fullness
    c = cyl["Gaussian V(y), level -4 mm"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.errorbar(c["angles"], 100 * c["e_single"][0], yerr=100 * c["e_single"][1], fmt="o-", ms=3,
                color=C[1], mfc="white", label=f"geometric fullness f = {c['f']:.3f}")
    ax.errorbar(c["angles"], 100 * c["e_single_alt"][0], yerr=100 * c["e_single_alt"][1], fmt="s-",
                ms=3, color=C[0], mfc="white", label=f"intensity-weighted f = {c['f_intensity']:.3f}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("2θ (deg)"); ax.set_ylabel("error of single part (%)")
    ax.set_xticks(range(0, 181, 30)); ax.set_xlim(0, 180)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "V_profile.pdf")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    print("Cylinder cases")
    cyl = cylinder_cases(q)
    print("Slab cases")
    sl = slab_case(q)
    write_table(cyl, "V_cylinder.txt", "Ni powder in V can, beam 15 x 20 mm, CORRECT fullness step + P&P + delta")
    write_table(sl, "V_slab.txt", "Flat cell V/Ni/V, beam 15 x 20 mm, CORRECT plate geometry (no delta)")
    plots(cyl, sl)
    np.save(os.path.join(RES, "V_cases.npy"), {"cyl": cyl, "slab": sl}, allow_pickle=True)
    print("done")
