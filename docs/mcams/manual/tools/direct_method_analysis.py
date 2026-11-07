"""
direct_method_analysis.py - Validation of the direct method (direct_attenuation.py)
and comparison with the Monte Carlo of MCAMS. Produces the numbers and the
figures of the manual chapter "Direct method".

    python direct_method_analysis.py            (modules two levels above, about 6 min)
    MC_CODE_DIR=/path/to/code python direct_method_analysis.py

Parts
-----
1. EXACT CHECKS: the direct method against analytic or deterministic results
   (slab in transmission and reflection, cylinder at 0 and 180 degrees with a
   narrow beam, furnace factors at 30 and 150 degrees on a fine grid,
   partially filled slab).
2. DIRECT vs MONTE CARLO for four samples (statistical agreement, run time).
3. EFFICIENCY versus mu*R: figure of merit 1 / (relative error^2 * time).
4. REFERENCE for simulate_scattering_orders: single scattering per steradian.

Output: tools/output/direct_part1.txt ... direct_part4.txt (one per part, they
are included in the manual) and figures/fig_direct_*.pdf.
Run only some parts with:  python direct_method_analysis.py 2 4

NumPy/SciPy functions used
--------------------------
scipy.integrate.quad(f, a, b) : adaptive numerical integral of f from a to b
np.meshgrid(a, b)             : 2D grids of coordinates from two 1D arrays
np.hypot(a, b)                : sqrt(a^2 + b^2)
"""
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("MC_CODE_DIR", os.path.abspath(os.path.join(HERE, "..", "..")))
# toscana must be importable (ToScaNA manual, chapter Installation).
from toscana.corrections.mcams.cylinder import create_cylindrical_sample                    # noqa: E402
from toscana.corrections.mcams.slab import create_slab_sample                               # noqa: E402
from toscana.corrections.mcams.sample_base import Layer                                     # noqa: E402
from toscana.corrections.mcams.multiple_scattering import simulate_scattering_orders        # noqa: E402
from toscana.corrections.mcams.direct_attenuation import (attenuation_factors_direct,       # noqa: E402
                                attenuation_factors_mc, plot_attenuation_factors)

FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
OUTD = os.path.join(HERE, "output")
os.makedirs(OUTD, exist_ok=True)
LOG = None          # the log file of the part being run: output/direct_partN.txt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


def out(text=""):
    print(text)
    LOG.write(text + "\n")
    LOG.flush()


# Ni sample and V container at lambda = 0.5 A (mm^-1); used in several parts
MU_NI, F_NI = 0.18027, 0.937
MU_V, F_V = 0.04591, 0.779


# ==========================================================================
# 1. Exact checks
# ==========================================================================
def part1():
    out("=" * 78)
    out("1. EXACT CHECKS (direct method, 400 000 points per layer)")
    out("=" * 78)
    out(f"{'case':<34}{'2theta':>7}{'exact':>10}{'direct':>10}{'error':>9}{'z':>7}")

    def line(case, a, ex, d, e):
        # z = difference in units of the statistical error; when every point
        # has the same weight (slab at 0 deg) the error is 0 and the result exact
        z = f"{(d - ex) / e:7.2f}" if e > 1e-12 else "  exact"
        out(f"{case:<34}{a:7.0f}{ex:10.5f}{d:10.5f}{e:9.5f}{z}")

    # --- slab, normal incidence, transmission (0-60) and reflection (120-180)
    mu, t = 0.25, 2.0
    s = create_slab_sample(2000, 2000, [("A", t, mu)])
    angs = np.array([0, 30, 60, 120, 150, 180.0])
    r = attenuation_factors_direct(s, 10, 10, angles_deg=angs, n_points=400_000, rng=1)
    exact = []
    for i, a in enumerate(angs):
        c = math.cos(math.radians(a))
        if c > 0:            # leaves through the back face: path (t-z)/c
            b = 1 / c
            ex = math.exp(-mu * t) if b == 1 else \
                math.exp(-mu * t * b) * (math.exp(mu * t * (b - 1)) - 1) / (mu * t * (b - 1))
        else:                # leaves through the front face: path z/|c|
            k = 1 + 1 / abs(c)
            ex = (1 - math.exp(-mu * t * k)) / (mu * t * k)
        exact.append(ex)
        line("slab, normal incidence", a, ex, r.factors[0, i], r.errors[0, i])

    # --- partially filled slab: same A (in-plane detector), half the path
    sp = create_slab_sample(2000, 2000, [Layer("A", t, mu, fullness=0.5)],
                            y_bottom=-5, beam_height=10)
    rp = attenuation_factors_direct(sp, 10, 10, angles_deg=angs, n_points=400_000, rng=2)
    for i in (1, 4):
        line("slab, fullness 0.5", angs[i], exact[i], rp.factors[0, i], rp.errors[0, i])
    out(f"   mean path in the layer: full {r.mean_path[0]:.4f} mm, "
        f"fullness 0.5 {rp.mean_path[0]:.4f} +- {rp.mean_path_error[0]:.4f} mm (exact {t / 2:.4f})")

    # --- cylinder Ni + V can, narrow beam 2 mm: exact 1D integrals at 0, 180
    R1, R2, d = 3.0, 3.2, 1.0
    s1 = lambda y: math.sqrt(R1 * R1 - y * y)
    s2 = lambda y: math.sqrt(R2 * R2 - y * y)
    area = 2 * (R1 * R1 * math.asin(d / R1) + d * math.sqrt(R1 * R1 - d * d))
    A0 = quad(lambda y: 2 * s1(y) * math.exp(-2 * MU_NI * s1(y) - 2 * MU_V * (s2(y) - s1(y))),
              -d, d, epsabs=1e-13)[0] / area
    A180 = quad(lambda y: math.exp(-2 * MU_V * (s2(y) - s1(y)))
                * (1 - math.exp(-4 * MU_NI * s1(y))) / (2 * MU_NI), -d, d, epsabs=1e-13)[0] / area
    cyl = create_cylindrical_sample(50, [("Ni", 6.0, MU_NI), ("V", 0.2, MU_V)])
    rc = attenuation_factors_direct(cyl, 2 * d, 20, angles_deg=[0, 180], n_points=400_000, rng=3)
    line("cylinder Ni+V, beam 2 mm, A_s", 0, A0, rc.factors[0, 0], rc.errors[0, 0])
    line("cylinder Ni+V, beam 2 mm, A_s", 180, A180, rc.factors[0, 1], rc.errors[0, 1])

    # --- furnace (V, 10.0-10.4 mm) around Ni + V can: fine deterministic grid
    fur = create_cylindrical_sample(50, [("Ni", 6.0, MU_NI), ("V", 0.2, MU_V),
                                         ("gap", 6.8, 0.0), ("furnace", 0.4, MU_V)])
    rf = attenuation_factors_direct(fur, 15, 20, angles_deg=[30, 150], n_points=400_000, rng=4)

    def grid(tt, nr=40, nphi=80000):
        r_ = 10.0 + (np.arange(nr) + 0.5) * 0.4 / nr
        ph = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
        Rr, Pp = np.meshgrid(r_, ph)
        X, Y, w = (Rr * np.cos(Pp)).ravel(), (Rr * np.sin(Pp)).ravel(), Rr.ravel()
        k = np.abs(Y) <= 7.5
        X, Y, w = X[k], Y[k], w[k]

        def chord(ux, uy, R):            # path from (X, Y) along u inside circle R
            b = X * ux + Y * uy
            disc = b * b - (X * X + Y * Y - R * R)
            s = np.sqrt(np.maximum(disc, 0))
            return np.where(disc > 0, np.maximum(0, (-b + s) - np.maximum(-b - s, 0)), 0)

        def tau(ux, uy):
            c = [chord(ux, uy, R) for R in (3.0, 3.2, 10.0, 10.4)]
            return MU_NI * c[0] + MU_V * (c[1] - c[0]) + MU_V * (c[3] - c[2])
        a = math.radians(tt)
        f = np.exp(-tau(-1, 0) - tau(math.cos(a), math.sin(a)))
        return (f * w).sum() / w.sum()
    for i, a in enumerate((30, 150)):
        line("furnace A_h,sch (fine grid)", a, grid(a), rf.factors[3, i], rf.errors[3, i])
    out()


# ==========================================================================
# 2. Direct vs Monte Carlo
# ==========================================================================
def part2():
    out("=" * 78)
    out("2. DIRECT vs MONTE CARLO (direct: 200 000 points/layer; MC: 20 x 25 000 particles)")
    out("=" * 78)
    gauss = lambda x: np.exp(-x ** 2 / (2 * 3.0 ** 2))
    cases = [
        ("Ni in V can, beam 15 x 20 mm",
         create_cylindrical_sample(50, [("Ni", 6.0, MU_NI, F_NI), ("V", 0.2, MU_V, F_V)]),
         dict(beam_width=15, beam_height=20)),
        ("same, narrow beam 2 mm",
         create_cylindrical_sample(50, [("Ni", 6.0, MU_NI, F_NI), ("V", 0.2, MU_V, F_V)]),
         dict(beam_width=2, beam_height=20)),
        ("slab Al/Fe/Cu, plane angle 60 deg",
         create_slab_sample(20, 20, [("Al", 2.0, 0.020), ("Fe", 1.0, 0.250), ("Cu", 0.5, 0.300)],
                            plane_angle_deg=60),
         dict(beam_width=8, beam_height=8)),
        ("powder fullness 0.6 in V can, Gaussian H",
         create_cylindrical_sample(20, [Layer("powder", 6.0, 0.25, 0.8, fullness=0.6),
                                        ("V", 0.2, MU_V, F_V)], y_bottom=-4, beam_height=8),
         dict(beam_width=8, beam_height=8, H=gauss)),
    ]
    angles = np.arange(0, 181, 10.0)
    keep = None
    out(f"{'sample':<42}{'layer':<8}{'max|z|':>7}{'mean diff%':>11}"
        f"{'err d %':>9}{'err MC %':>9}{'t d (s)':>8}{'t MC (s)':>9}{'FOM d/MC':>9}")
    for name, s, kw in cases:
        d = attenuation_factors_direct(s, angles_deg=angles, n_points=200_000, rng=10, **kw)
        m = attenuation_factors_mc(s, angles_deg=angles, n_particles=25_000, n_batches=20,
                                   mean_path=d.mean_path, rng=11, **kw)
        for k, lname in enumerate(d.layer_names):
            if not np.all(np.isfinite(d.factors[k])):
                continue
            z = (m.factors[k] - d.factors[k]) / np.hypot(m.errors[k], d.errors[k])
            rd = np.mean(d.errors[k] / d.factors[k])
            rm = np.mean(m.errors[k] / m.factors[k])
            fom = (rm ** 2 * m.seconds) / (rd ** 2 * d.seconds)
            out(f"{name:<42}{lname:<8}{np.abs(z).max():7.2f}"
                f"{100 * np.mean(m.factors[k] / d.factors[k] - 1):11.3f}"
                f"{100 * rd:9.3f}{100 * rm:9.3f}{d.seconds:8.1f}{m.seconds:9.1f}{fom:9.1f}")
            name = ""
        if keep is None:
            keep = (d, m)
    plot_attenuation_factors(list(keep), title="Ni in a V can: direct method (lines) and Monte Carlo (points)",
                             save_as=os.path.join(FIG, "fig_direct_vs_mc.pdf"), show=False)
    plt.close("all")
    out()


# ==========================================================================
# 3. Efficiency versus mu R
# ==========================================================================
def part3():
    out("=" * 78)
    out("3. EFFICIENCY vs mu*R (single cylinder, R = 3 mm, beam 8 x 20 mm, 5 angles)")
    out("   FOM = 1/(relative error^2 x time); ratio > 1: direct method more efficient")
    out("=" * 78)
    angles = np.array([0, 45, 90, 135, 180.0])
    muRs = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    out(f"{'mu R':>6}{'A(90)':>9}{'P(int)':>8}{'err d %':>9}{'err MC %':>9}{'t d':>6}{'t MC':>6}{'FOM d/MC':>10}{'max|z|':>8}")
    for muR in muRs:
        s = create_cylindrical_sample(50, [("X", 6.0, muR / 3.0, 1.0)])
        d = attenuation_factors_direct(s, 8, 20, angles_deg=angles, n_points=100_000, rng=1)
        m = attenuation_factors_mc(s, 8, 20, angles_deg=angles, n_particles=25_000, n_batches=20,
                                   mean_path=d.mean_path, rng=2)
        rd = np.mean(d.errors[0] / d.factors[0])
        rm = np.mean(m.errors[0] / m.factors[0])
        ratio = (rm ** 2 * m.seconds) / (rd ** 2 * d.seconds)
        # fraction of the beam that interacts (MC "efficiency")
        p_int = 1 - np.mean(np.exp(-muR / 3.0 * 2 * np.sqrt(np.maximum(9 - np.linspace(-4, 4, 20001) ** 2, 0))))
        z = (m.factors[0] - d.factors[0]) / np.hypot(m.errors[0], d.errors[0])
        rows.append((muR, ratio))
        out(f"{muR:6.2f}{d.factors[0, 2]:9.4f}{p_int:8.3f}{100 * rd:9.3f}{100 * rm:9.3f}"
            f"{d.seconds:6.1f}{m.seconds:6.1f}{ratio:10.2f}{np.abs(z).max():8.2f}")
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.loglog(rows[:, 0], rows[:, 1], "o-", color=C[0], ms=4)
    ax.axhline(1, color="k", lw=0.8)
    ax.fill_between([0.015, 6], 1, 1e4, color=C[0], alpha=0.06, lw=0)
    ax.fill_between([0.015, 6], 1e-2, 1, color=C[1], alpha=0.06, lw=0)
    ax.text(0.03, 2.5, "direct method more efficient", color=C[0], fontsize=8)
    ax.text(1.0, 0.03, "Monte Carlo\nmore efficient", color=C[1], fontsize=8)
    ax.set_xlim(0.015, 6)
    ax.set_ylim(1e-2, 1e4)
    ax.set_xlabel(r"$\mu R$")
    ax.set_ylabel("figure of merit, direct / MC")
    fig.savefig(os.path.join(FIG, "fig_direct_fom.pdf"))
    plt.close(fig)
    out()


# ==========================================================================
# 4. Reference for the single-scattering counts
# ==========================================================================
def part4():
    out("=" * 78)
    out("4. SINGLE-SCATTERING REFERENCE for simulate_scattering_orders")
    out("   Ni in V can, beam 15 x 20 mm, azimuth window +-5 deg, 5-degree bins, 1.6e7 particles")
    out("=" * 78)
    s = create_cylindrical_sample(50, [("Ni", 6.0, MU_NI, F_NI), ("V", 0.2, MU_V, F_V)])
    edges = np.arange(0, 181, 5.0)
    t0 = time.perf_counter()
    res = simulate_scattering_orders(s, 15, 20, n_particles=16_000_000, angle_edges_deg=edges,
                                     phi_deg=0.0, phi_halfwidth_deg=5.0, rng=7)
    t_mc = time.perf_counter() - t0
    d = attenuation_factors_direct(s, 15, 20, angles_deg=res.angles_deg, n_points=200_000, rng=8)
    ref = d.single_per_steradian()
    single = res.per_steradian(res.single)
    err = np.sqrt(res.single) / (res.n_incident * res.solid_angle)
    multiple = res.per_steradian(res.multiple)
    z = (single - ref) / err
    sel = res.single > 100
    out(f"   Monte Carlo {t_mc:.0f} s, direct {d.seconds:.0f} s")
    out(f"   bins with > 100 counts: {sel.sum()} ;  chi2/bin of MC single vs direct = {np.mean(z[sel] ** 2):.2f}")
    out(f"   mean of MC/direct - 1 over those bins = {100 * np.mean(single[sel] / ref[sel] - 1):+.2f} %")
    out(f"   {'2theta':>7}{'direct':>11}{'MC single':>11}{'error':>10}{'MC multiple':>13}")
    for i in range(1, len(ref), 6):
        out(f"   {res.angles_deg[i]:7.1f}{ref[i]:11.5f}{single[i]:11.5f}{err[i]:10.5f}{multiple[i]:13.5f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(res.angles_deg, ref, color=C[0], lw=1.8, label="single, direct method (reference)")
    ax.errorbar(res.angles_deg[sel], single[sel], yerr=err[sel], fmt="o", ms=3, mfc="white",
                color=C[0], label="single, Monte Carlo")
    ax.errorbar(res.angles_deg[sel], multiple[sel],
                yerr=np.sqrt(res.multiple[sel]) / (res.n_incident * res.solid_angle[sel]),
                fmt="s", ms=3, mfc="white", color=C[1], label="multiple, Monte Carlo only")
    ax.set_xlabel(r"scattering angle $2\theta$ (deg)")
    ax.set_ylabel("particles / incident / sr")
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    fig.savefig(os.path.join(FIG, "fig_direct_orders.pdf"))
    plt.close(fig)
    out()


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for p in which:
        LOG = open(os.path.join(OUTD, f"direct_part{p}.txt"), "w")
        {"1": part1, "2": part2, "3": part3, "4": part4}[p]()
        LOG.close()
