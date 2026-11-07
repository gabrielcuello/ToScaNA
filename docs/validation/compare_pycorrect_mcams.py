"""
compare_pycorrect_mcams.py - Compare the analytical corrections of pycorrect
(CORRECT 2.24) with the Monte Carlo simulation MCAMS.

Only the two effects that MCAMS simulates are compared:
  * ATTENUATION      : Paalman & Pings factors A_s,sch, A_c,sch, A_c,ch,
                       A_h,sch, A_h,ch, A_h,h (pycorrect.atcors)
  * MULTIPLE SCATTERING: Blech & Averbach double-scattering table delta_2 and
                       the total correction delta (pycorrect.mscat)
No Placzek correction and no vanadium normalisation.

Usage
-----
    python compare_pycorrect_mcams.py            # full run (about 10 minutes)
    python compare_pycorrect_mcams.py --quick    # fewer particles (about 1 minute)

Output
------
    results/*.txt   : tables (2theta, pycorrect, MCAMS, error) of every case
    results/*.npz   : the same numbers in NumPy format (np.load)
    figures/*.pdf   : the figures of the report (report/main.tex)

MCAMS, pycorrect and the bridge are imported from the toscana package
(toscana.corrections.mcams, .pycorrect, .bridge); toscana must be importable.

All geometrical inputs are written in cm and cm^-1 (pycorrect units); the
bridge converts them to mm for MCAMS.

NumPy/Matplotlib functions used
-------------------------------
np.arange(a, b, step) : angles; np.savez(file, name=array, ...) : save arrays
np.savetxt(file, table, header=...) : text tables; np.column_stack : columns
plt.subplots, ax.errorbar (points with error bars), ax.plot, fig.savefig
"""

import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # no window: only files
import matplotlib.pyplot as plt

from toscana.corrections import bridge as br
from toscana.corrections.pycorrect import atcors, mscat
from toscana.corrections.pycorrect.ndplib import load_double_scattering_table

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- plot style
# Validated categorical palette (first three slots are colour-blind safe).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.5, "legend.frameon": False,
    "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


# ==========================================================================
#  Physical input: Ni sample in a V can (same as the pycorrect manual example)
# ==========================================================================
LAM = 0.4985                                     # wavelength (A)
NI = dict(rho=0.0913, s=18.5, a=4.49)            # number density, sigma_s, sigma_a (barn)
V = dict(rho=0.0722, s=4.95, a=5.08)
MU_NI = br.mu_total(NI["rho"], NI["s"], NI["a"], LAM)      # 1/cm
MU_V = br.mu_total(V["rho"], V["s"], V["a"], LAM)
F_NI = NI["rho"] * NI["s"] / MU_NI                          # mu_s / mu_t
F_V = V["rho"] * V["s"] / MU_V
R1, R2 = 0.30, 0.32                              # sample and can radii (cm)
R3, R4 = 1.00, 1.04                              # furnace radii (cm)


# ==========================================================================
#  Part A - attenuation factors
# ==========================================================================
def pp_case(name, title, r1, r2, r3, r4, smu, cmu, hmu, beam_w, angles, npart, nb,
            wanted):
    """Run one attenuation case with both programs and save the table.

    wanted : list of (pycorrect index, MCAMS layer name, label), where the
             pycorrect index refers to the tuple returned by atcors:
             0 assch, 1 acch, 2 acsch, 3 ahsch, 4 ahch, 5 ahh.
    For acch (index 1), ahch (4) and ahh (5) MCAMS is run again with the
    inner bodies emptied (mu = 0), because these factors describe the empty
    container / the furnace alone.
    """
    t0 = time.time()
    pyc = np.array([atcors(math.radians(a), r1, r2, r3, r4, beam_w, smu, cmu, hmu)
                    for a in angles])                       # (A, 6)
    out = {"angles": angles}
    rows = [angles]
    header = "2theta"
    for idx, lname, label in wanted:
        # which bodies are present for this factor
        s_mu = smu if idx in (0, 2, 3) else 0.0
        c_mu = cmu if idx in (0, 1, 2, 3, 4) else 0.0
        layers, names = br.cylinder_layers_from_pycorrect(r1, r2, r3, r4, s_mu, c_mu, hmu)
        mean, err = br.mc_attenuation_factors(layers, beam_w, angles, npart, nb)
        k = names.index(lname)
        out[label + "_pyc"], out[label + "_mc"], out[label + "_err"] = pyc[:, idx], mean[k], err[k]
        rows += [pyc[:, idx], mean[k], err[k]]
        header += f"  {label}_pycorrect  {label}_MCAMS  {label}_err"
    np.savez(os.path.join(RES, name + ".npz"), **out)
    np.savetxt(os.path.join(RES, name + ".txt"), np.column_stack(rows), fmt="%10.5f",
               header=title + "\n" + header)
    print(f"  {name}: {time.time() - t0:.0f} s")
    return out


def plot_pp(out, labels, name, title_latex=""):
    """Upper panel: factors (lines pycorrect, open points MCAMS with error
    bars); lower panel: relative difference (MCAMS - pycorrect)/pycorrect in
    percent.  One colour per factor, named in the legend."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1.3]})
    ang = out["angles"]
    for i, (lab, tex) in enumerate(labels):
        col = C[i % len(C)]
        a1.plot(ang, out[lab + "_pyc"], color=col, lw=1.6, label=tex)
        a1.errorbar(ang, out[lab + "_mc"], yerr=out[lab + "_err"], fmt="o", ms=3,
                    color=col, mfc="white", mew=1.0, elinewidth=0.8)
        rel = 100 * (out[lab + "_mc"] / out[lab + "_pyc"] - 1)
        rel_e = 100 * out[lab + "_err"] / out[lab + "_pyc"]
        a2.errorbar(ang, rel, yerr=rel_e, fmt="o-", ms=2.5, lw=0.8, color=col,
                    elinewidth=0.6)
    a1.set_ylabel("attenuation factor")
    a1.set_xlim(-3, 183)
    a1.legend(loc="best", fontsize=8, ncol=len(labels),
              title="lines: pycorrect   points: MCAMS", title_fontsize=7.5)
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_ylabel("MCAMS/pycorrect − 1 (%)")
    a2.set_xlabel("2θ (deg)")
    a2.set_xticks(range(0, 181, 30))
    save(fig, name + ".pdf")


PP_PLOTS = [
    ("A1", "A1_sample_alone", [("Ass", "$A_{s,s}$")]),
    ("A2", "A2_sample_in_can", [("Assc", "$A_{s,sc}$"), ("Acsc", "$A_{c,sc}$"), ("Acc", "$A_{c,c}$")]),
    ("A3", "A3_narrow_beam", [("Assc", "$A_{s,sc}$"), ("Acsc", "$A_{c,sc}$"), ("Acc", "$A_{c,c}$")]),
    ("A4", "A4_furnace_sample_can", [("Assch", "$A_{s,sch}$"), ("Acsch", "$A_{c,sch}$")]),
    ("A4", "A4_furnace_factors", [("Ahsch", "$A_{h,sch}$"), ("Ahch", "$A_{h,ch}$"), ("Ahh", "$A_{h,h}$")]),
]
NPZ = {"A1": "A1_sample_alone", "A2": "A2_sample_in_can", "A3": "A3_narrow_beam",
       "A4": "A4_furnace"}


def replot_a():
    """Redraw the attenuation figures from the saved results/*.npz."""
    for key, fname, labels in PP_PLOTS:
        plot_pp(dict(np.load(os.path.join(RES, NPZ[key] + ".npz"))), labels, fname)


def part_a(q):
    print("Part A: attenuation factors")
    ang = np.arange(0.0, 181.0, 10.0)
    npart, nb = (40_000, 5) if q else (400_000, 10)
    res = {}
    # A1 sample alone, beam wider than the sample
    res["A1"] = pp_case("A1_sample_alone", "Ni R=0.30 cm alone, beam 1.5 cm",
                        R1, 0, 0, 0, MU_NI, 0, 0, 1.5, ang, npart, nb,
                        [(0, "sample", "Ass")])
    # A2 sample in can
    res["A2"] = pp_case("A2_sample_in_can", "Ni R=0.30 in V can R2=0.32 cm, beam 1.5 cm",
                        R1, R2, 0, 0, MU_NI, MU_V, 0, 1.5, ang, npart, nb,
                        [(0, "sample", "Assc"), (2, "can", "Acsc"), (1, "can", "Acc")])
    # A3 narrow beam
    res["A3"] = pp_case("A3_narrow_beam", "Ni in V can, beam 0.2 cm (narrower than sample)",
                        R1, R2, 0, 0, MU_NI, MU_V, 0, 0.2, ang, npart, nb,
                        [(0, "sample", "Assc"), (2, "can", "Acsc"), (1, "can", "Acc")])
    # A4 furnace
    res["A4"] = pp_case("A4_furnace", "Ni in V can inside V furnace R3=1.00 R4=1.04 cm, beam 1.5 cm",
                        R1, R2, R3, R4, MU_NI, MU_V, MU_V, 1.5, ang, npart, nb,
                        [(0, "sample", "Assch"), (2, "can", "Acsch"), (3, "furnace", "Ahsch"),
                         (4, "furnace", "Ahch"), (5, "furnace", "Ahh")])
    replot_a()
    return res


# ==========================================================================
#  Part B - multiple scattering
# ==========================================================================
def ba_table_value(r, h, mu):
    """Blech & Averbach delta_2 interpolated in doublescattering.dat exactly
    as pycorrect does (bilinear, faithful=False): returned by mscat as the
    second value with return_details=True."""
    _, d2 = mscat(r, h, mu, mu, verbose=False, return_details=True)
    return d2


def part_b(q):
    print("Part B: multiple scattering")
    out = {}
    angles = np.arange(10.0, 171.0, 20.0)            # 9 angles for the averages
    npart, nb = (15_000, 4) if q else (60_000, 8)
    R = 0.30                                          # cm
    muR_grid = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    rh_grid = [0.1, 0.25, 0.5, 1.0]

    # ---------- B1: delta_2 (pure scatterer) vs muR for several R/H -------
    t0 = time.time()
    B1 = {"muR": muR_grid}
    for rh in rh_grid:
        H = R / rh
        mc_d2, mc_d2e, mc_d, mc_de, mc_r32, tab, pyd = [], [], [], [], [], [], []
        for muR in muR_grid:
            mu = muR / R
            layers, _ = br.cylinder_layers_from_pycorrect(R, smu=mu, s_frac=1.0)
            r = br.mc_delta(layers, 2 * R + 0.2, H, H, angles, npart, nb)
            # angle average of each batch-mean (the error of the average is
            # taken as the mean error / sqrt(n_angles): angles share particles,
            # so this is only indicative; the angle spread is shown in B3)
            mc_d2.append(r["delta2"][0].mean()); mc_d2e.append(r["delta2"][1].mean())
            mc_d.append(r["delta"][0].mean()); mc_de.append(r["delta"][1].mean())
            mc_r32.append(r["ratio32"][0].mean())
            tab.append(ba_table_value(R, H, mu))
            pyd.append(mscat(R, H, mu, mu, verbose=False))
        B1[f"rh{rh}"] = dict(mc_d2=np.array(mc_d2), mc_d2e=np.array(mc_d2e),
                             mc_d=np.array(mc_d), mc_de=np.array(mc_de),
                             mc_r32=np.array(mc_r32), table=np.array(tab),
                             pyc=np.array(pyd))
        np.savetxt(os.path.join(RES, f"B1_delta2_RH{rh}.txt"),
                   np.column_stack([muR_grid, tab, mc_d2, mc_d2e, pyd, mc_d, mc_de, mc_r32]),
                   fmt="%10.5f",
                   header=f"pure scatterer, R={R} cm, R/H={rh} (sample height = beam height = H)\n"
                          "muR  delta2_table  delta2_MCAMS  err  delta_pycorrect  delta_MCAMS  err  ratio32_MCAMS")
    out["B1"] = B1
    print(f"  B1: {time.time() - t0:.0f} s")

    # ---------- B2: absorbing sample (mu_s/mu_t = 0.5), R/H = 0.1 ---------
    t0 = time.time()
    H = R / 0.1
    B2 = {"muR": muR_grid, "mc_d": [], "mc_de": [], "pyc": [], "mc_d2": [], "tab": []}
    for muR in muR_grid:
        mu = muR / R
        layers, _ = br.cylinder_layers_from_pycorrect(R, smu=mu, s_frac=0.5)
        r = br.mc_delta(layers, 2 * R + 0.2, H, H, angles, npart, nb)
        B2["mc_d"].append(r["delta"][0].mean()); B2["mc_de"].append(r["delta"][1].mean())
        B2["mc_d2"].append(r["delta2"][0].mean())
        B2["tab"].append(0.5 * ba_table_value(R, H, mu))
        B2["pyc"].append(mscat(R, H, mu, 0.5 * mu, verbose=False))
    B2 = {k: np.asarray(v) for k, v in B2.items()}
    np.savetxt(os.path.join(RES, "B2_absorbing.txt"),
               np.column_stack([muR_grid, B2["tab"], B2["mc_d2"], B2["pyc"], B2["mc_d"], B2["mc_de"]]),
               fmt="%10.5f", header="mu_s/mu_t = 0.5, R=0.3 cm, R/H=0.1\n"
               "muR  f*delta2_table  delta2_MCAMS  delta_pycorrect  delta_MCAMS  err")
    out["B2"] = B2
    print(f"  B2: {time.time() - t0:.0f} s")

    # ---------- B3: angle dependence, Ni sample (and Ni in V can) ----------
    t0 = time.time()
    ang3 = np.arange(5.0, 176.0, 10.0)
    Hb = 3.0                                           # beam height = sample height (cm)
    npart3, nb3 = (20_000, 4) if q else (100_000, 8)
    lay_s, _ = br.cylinder_layers_from_pycorrect(R1, smu=MU_NI, s_frac=F_NI)
    rs = br.mc_delta(lay_s, 2 * R2 + 0.2, Hb, Hb, ang3, npart3, nb3)
    lay_sc, _ = br.cylinder_layers_from_pycorrect(R1, R2, smu=MU_NI, cmu=MU_V,
                                                  s_frac=F_NI, c_frac=F_V)
    rsc = br.mc_delta(lay_sc, 2 * R2 + 0.2, Hb, Hb, ang3, npart3, nb3)
    pyc_s = mscat(R1, Hb, MU_NI, F_NI * MU_NI, verbose=False)
    fs = (R1 / R2) ** 2                                # pycorrect effective cylinder
    pyc_sc = mscat(R2, Hb, (1 - fs) * MU_V + fs * MU_NI,
                   (1 - fs) * F_V * MU_V + fs * F_NI * MU_NI, verbose=False)
    pyc_sc_faith = mscat(R2, Hb, (1 - fs) * MU_V + fs * MU_NI,
                         (1 - fs) * F_V * MU_V + fs * F_NI * MU_NI, verbose=False, faithful=True)
    B3 = dict(angles=ang3, mc_s=rs["delta"][0], mc_se=rs["delta"][1], pyc_s=pyc_s,
              mc_sc=rsc["delta"][0], mc_sce=rsc["delta"][1], pyc_sc=pyc_sc,
              pyc_sc_faithful=pyc_sc_faith,
              mc_s_d2=rs["delta2"][0], mc_sc_d2=rsc["delta2"][0])
    np.savetxt(os.path.join(RES, "B3_angle_dependence.txt"),
               np.column_stack([ang3, rs["delta"][0], rs["delta"][1], np.full(len(ang3), pyc_s),
                                rsc["delta"][0], rsc["delta"][1], np.full(len(ang3), pyc_sc)]),
               fmt="%10.5f", header="Ni R=0.30 cm (and in V can R2=0.32 cm), H = beam height = 3 cm\n"
               "2theta  delta_MCAMS_Ni  err  delta_pycorrect_Ni  delta_MCAMS_Ni+can  err  delta_pycorrect_Ni+can")
    out["B3"] = B3
    print(f"  B3: {time.time() - t0:.0f} s")

    # ---------- B4: sample taller than the beam --------------------------------
    t0 = time.time()
    B4 = {"Hs": np.array([3.0, 4.0, 6.0, 10.0])}
    vals, errs = [], []
    for Hs in B4["Hs"]:
        r = br.mc_delta(lay_s, 2 * R1 + 0.2, Hb, Hs, angles, npart3, nb3, by_order=False)
        vals.append(r["delta"][0].mean()); errs.append(r["delta"][1].mean())
    B4.update(mc=np.array(vals), mce=np.array(errs), pyc=pyc_s)
    np.savetxt(os.path.join(RES, "B4_tall_sample.txt"),
               np.column_stack([B4["Hs"], vals, errs, np.full(len(vals), pyc_s)]), fmt="%10.5f",
               header="Ni R=0.30 cm, beam height 3 cm, sample height Hs\nHs(cm) delta_MCAMS err delta_pycorrect(H=beam height)")
    out["B4"] = B4
    print(f"  B4: {time.time() - t0:.0f} s")
    np.savez(os.path.join(RES, "B_multiple_scattering.npz"),
             **{f"{k}_{kk}": np.asarray(vv) for k, v in out.items() for kk, vv in
                (v.items() if k != "B1" else [])})
    return out


def plot_b(out):
    B1, B2, B3, B4 = out["B1"], out["B2"], out["B3"], out["B4"]
    muR = B1["muR"]
    # --- delta_2: table vs MC
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for i, rh in enumerate([0.1, 0.25, 0.5, 1.0]):
        d = B1[f"rh{rh}"]
        ax.plot(muR, d["table"], color=C[i], lw=1.5)
        ax.errorbar(muR, d["mc_d2"], yerr=d["mc_d2e"], fmt="o", ms=3.5, color=C[i],
                    mfc="white", label=f"R/H = {rh}")
    ax.set_xlabel("μR")
    ax.set_ylabel("δ₂ = double / single")
    ax.plot([], [], color=INK2, label="Blech & Averbach table")
    ax.errorbar([], [], yerr=[], fmt="o", color=INK2, mfc="white", label="MCAMS")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    save(fig, "B1_delta2.pdf")

    # --- total delta vs muR (pure scatterer and absorbing)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    d = B1["rh0.1"]
    ax.plot(muR, d["pyc"], color=C[0], label="pycorrect, σs/σt = 1")
    ax.errorbar(muR, d["mc_d"], yerr=d["mc_de"], fmt="o", ms=3.5, color=C[0], mfc="white",
                label="MCAMS, σs/σt = 1")
    geo = d["table"] / (1 - d["table"])
    ax.plot(muR, geo, color=C[0], ls="--", lw=1.0, label="d/(1−d), σs/σt = 1")
    ax.plot(muR, B2["pyc"], color=C[1], label="pycorrect, σs/σt = 0.5")
    ax.errorbar(muR, B2["mc_d"], yerr=B2["mc_de"], fmt="s", ms=3.5, color=C[1], mfc="white",
                label="MCAMS, σs/σt = 0.5")
    ax.plot(muR, B2["tab"] / (1 - B2["tab"]), color=C[1], ls="--", lw=1.0,
            label="d/(1−d), σs/σt = 0.5")
    ax.set_xlabel("μR")
    ax.set_ylabel("δ = multiple / single")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "B2_delta_total.pdf")

    # --- ratio of successive orders
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for i, rh in enumerate([0.1, 0.25, 0.5, 1.0]):
        dd = B1[f"rh{rh}"]
        ax.plot(muR, dd["mc_r32"] / dd["mc_d2"], "o-", ms=3, color=C[i], label=f"R/H = {rh}")
    ax.axhline(0.694, color=INK2, ls="--", lw=1)
    ax.annotate("CORRECT series: 0.694", (muR[-1], 0.694), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=INK2, fontsize=8)
    ax.set_xlabel("μR")
    ax.set_ylabel("(order 3 / order 2) / δ₂")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "B2_order_ratio.pdf")

    # --- angle dependence
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ang = B3["angles"]
    ax.errorbar(ang, B3["mc_s"], yerr=B3["mc_se"], fmt="o", ms=3, color=C[0], mfc="white",
                label="MCAMS, Ni")
    ax.axhline(B3["pyc_s"], color=C[0], lw=1.5, label="pycorrect, Ni")
    ax.errorbar(ang, B3["mc_sc"], yerr=B3["mc_sce"], fmt="s", ms=3, color=C[1], mfc="white",
                label="MCAMS, Ni + V can")
    ax.axhline(B3["pyc_sc"], color=C[1], lw=1.5, label="pycorrect, Ni + V can (effective cylinder)")
    ax.set_xlabel("2θ (deg)")
    ax.set_ylabel("δ = multiple / single")
    ax.set_xticks(range(0, 181, 30))
    ax.set_ylim(0.44, 0.66)
    ax.legend(fontsize=7.5, loc="center right")
    save(fig, "B3_angle.pdf")

    # --- tall sample
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    ax.errorbar(B4["Hs"], B4["mc"], yerr=B4["mce"], fmt="o-", ms=3.5, color=C[0], mfc="white",
                label="MCAMS")
    ax.axhline(B4["pyc"], color=C[1], lw=1.5, label="pycorrect (H = beam height)")
    ax.set_xlabel("sample height (cm), beam height 3 cm")
    ax.set_ylabel("δ")
    ax.legend(fontsize=8)
    save(fig, "B4_tall_sample.pdf")


# ==========================================================================
#  Part C - independent check where the two programs disagree
# ==========================================================================
def chord_inside(X, Y, ux, uy, R):
    """Length of the half-line (X,Y) + t (ux,uy), t >= 0, inside the circle of
    radius R (2D, horizontal plane).  Roots t1 < t2 of |p + t u| = R."""
    b = X * ux + Y * uy
    disc = b * b - (X * X + Y * Y - R * R)
    s = np.sqrt(np.maximum(disc, 0.0))
    return np.where(disc > 0, np.maximum(0.0, (-b + s) - np.maximum(-b - s, 0.0)), 0.0)


def brute_factor(body, two_theta_deg, r1, r2, r3, r4, smu, cmu, hmu, beam_w,
                 n=4_000_000, seed=0):
    """Paalman & Pings factor by direct random sampling of the scattering
    point over the part of `body` ('sample', 'can', 'furnace') inside the beam
    band, with exact chord lengths in each circle.  Beam along +x here (the
    choice of axes does not matter in 2D).  Independent of both programs."""
    rng = np.random.default_rng(seed)
    lo, hi = {"sample": (0.0, r1), "can": (r1, r2), "furnace": (r3, r4)}[body]
    rr = np.sqrt(rng.uniform(lo ** 2, hi ** 2, n))          # uniform in area
    ph = rng.uniform(0.0, 2 * np.pi, n)
    X, Y = rr * np.cos(ph), rr * np.sin(ph)
    keep = np.abs(Y) <= beam_w / 2                           # inside the beam band
    X, Y = X[keep], Y[keep]

    def lengths(ux, uy):
        L = [chord_inside(X, Y, ux, uy, R) for R in (r1, r2, r3, r4)]
        ls = L[0]
        lc = (L[1] - L[0]) if r2 > r1 else 0.0
        lh = (L[3] - L[2]) if r4 > r3 else 0.0
        return ls, lc, lh

    si, ci, hi_ = lengths(-1.0, 0.0)                         # back along the beam
    t = math.radians(two_theta_deg)
    so, co, ho = lengths(math.cos(t), math.sin(t))           # out to the detector
    return float(np.exp(-smu * (si + so) - cmu * (ci + co) - hmu * (hi_ + ho)).mean())


def part_c():
    print("Part C: independent check (direct sampling)")
    rows = []
    a3 = np.load(os.path.join(RES, "A3_narrow_beam.npz"))
    for a in (0.0, 90.0, 180.0):
        i = int(np.argmin(abs(a3["angles"] - a)))
        b = brute_factor("sample", a, R1, R2, 0, 0, MU_NI, MU_V, 0, 0.2)
        rows.append(("A3 narrow beam", "A_s,sc", a, a3["Assc_pyc"][i], a3["Assc_mc"][i], a3["Assc_err"][i], b))
    a4 = np.load(os.path.join(RES, "A4_furnace.npz"))
    for a in (0.0, 30.0, 90.0, 150.0, 180.0):
        i = int(np.argmin(abs(a4["angles"] - a)))
        b = brute_factor("furnace", a, R1, R2, R3, R4, MU_NI, MU_V, MU_V, 1.5)
        rows.append(("A4 furnace", "A_h,sch", a, a4["Ahsch_pyc"][i], a4["Ahsch_mc"][i], a4["Ahsch_err"][i], b))
        b = brute_factor("furnace", a, R1, R2, R3, R4, 0, 0, MU_V, 1.5)
        rows.append(("A4 furnace", "A_h,h", a, a4["Ahh_pyc"][i], a4["Ahh_mc"][i], a4["Ahh_err"][i], b))
    with open(os.path.join(RES, "C_independent_check.txt"), "w") as fh:
        fh.write("# case  factor  2theta  pycorrect  MCAMS  MCAMS_err  direct_sampling\n")
        for r in rows:
            fh.write(f"{r[0]:<16s} {r[1]:<8s} {r[2]:6.1f} {r[3]:9.5f} {r[4]:9.5f} {r[5]:8.5f} {r[6]:9.5f}\n")
    print(open(os.path.join(RES, "C_independent_check.txt")).read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="fewer particles")
    ap.add_argument("--replot", action="store_true", help="only redraw part A figures")
    ap.add_argument("--part", choices=["a", "b", "c", "all"], default="all")
    q = ap.parse_args().quick
    part = ap.parse_args().part
    print(f"mu_t(Ni) = {MU_NI:.4f} 1/cm, mu_s/mu_t = {F_NI:.4f};  "
          f"mu_t(V) = {MU_V:.4f} 1/cm, mu_s/mu_t = {F_V:.4f}")
    if ap.parse_args().replot:
        replot_a()
    elif part in ("a", "all"):
        part_a(q)
    if part in ("b", "all"):
        plot_b(part_b(q))
    if part in ("c", "all"):
        part_c()
    print("done: tables in results/, figures in figures/")
