# Example 18 - All the attenuation factors A_x,y, and corrections with any set of runs
import os
import numpy as np
from cylinder import create_cylindrical_sample
from correct_tools import (RUNS, correct_samples, all_attenuation_factors, print_all_factors,
                           scattering, unattenuated, general_correction, correct_chain,
                           plot_all_factors)
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

BEAM_W, BEAM_H = 10.0, 20.0
ANGLES = np.arange(0.0, 181.0, 1.0)           # for the factors (direct method)
ANG_MC = np.arange(0.0, 181.0, 2.0)           # for the simulated runs (Monte Carlo)

# The assembly of Example 13: Ni in a V can inside a V furnace.
layers = [("Ni powder", 6.0,  0.180,  0.94),
          ("V can",     0.2,  0.0459, 0.78),
          ("vacuum",   11.8,  0.0,    1.0),
          ("V heater",  0.1,  0.0459, 0.78)]
roles = ["s", "c", "-", "h"]
MAKE = lambda L: create_cylindrical_sample(40.0, L)

# 1) The 21 factors A_x,y (x = s, c, h; y = the 7 sets of parts)
A, err = all_attenuation_factors(MAKE, layers, roles, BEAM_W, BEAM_H, ANGLES)
print_all_factors(ANGLES, A, every=90)
print(f"(largest relative error {100 * max(np.max(err[k] / A[k]) for k in A):.2f} %)")
plot_all_factors(ANGLES, A, "Cylinder of Example 13: all the factors $A_{x,y}$",
                 os.path.join(FIGURES, "fig_all_factors_cylinder.pdf"))

# 2) Are the factors products of the attenuation by each part? (ratio - 1, %)
print("\nproduct / exact - 1 (%)      2theta: " + "".join(f"{a:8.0f}" for a in ANGLES[::30]))
tests = [("A_s,s A_s,c A_s,h  vs A_s,sch", A["s,s"] * A["s,c"] * A["s,h"], A["s,sch"]),
         ("A_s,s A_s,ch       vs A_s,sch", A["s,s"] * A["s,ch"], A["s,sch"]),
         ("A_c,c A_c,s        vs A_c,sc ", A["c,c"] * A["c,s"], A["c,sc"]),
         ("A_c,c A_c,h        vs A_c,ch ", A["c,c"] * A["c,h"], A["c,ch"]),
         ("A_c,ch A_c,s       vs A_c,sch", A["c,ch"] * A["c,s"], A["c,sch"]),
         ("A_h,h A_h,s A_h,c  vs A_h,sch", A["h,h"] * A["h,s"] * A["h,c"], A["h,sch"])]
for name, prod, exact in tests:
    print(f"  {name}       " + "".join(f"{100 * (p / e - 1):8.2f}" for p, e in zip(prod[::30], exact[::30])))

# 3) The seven runs, simulated (single + multiple scattering): our "measurements"
samples = correct_samples(MAKE, layers, roles, runs=RUNS)
sc = scattering(samples, roles, BEAM_W, BEAM_H, ANG_MC, n_particles=200_000, n_batches=10)
I = {y: sc[y]["total"][0] for y in RUNS}                     # measured intensities
d = {y: sc[y]["delta"][0] for y in RUNS}                     # multiple / single
e = {y: sc[y]["total"][1] / sc[y]["total"][0] for y in RUNS} # relative errors
IDX = (0, 45, 90)                                            # 0, 90, 180 deg on the 2-degree grid
print("\nmultiple/single of each run at 2theta = 0, 90, 180: ")
for y in RUNS:
    print(f"  delta_{y:<4} = " + "  ".join(f"{d[y][i]:.4f}" for i in IDX))
true, true_err = unattenuated(samples["sch"], roles, BEAM_W, BEAM_H)

# 4) Corrections with different sets of runs (least squares; factors at ANG_MC)
Am = {k: np.interp(ANG_MC, ANGLES, v) for k, v in A.items()}   # factors at the angles of the runs
experiments = {"sch, ch, h  (CORRECT)": ("sch", "ch", "h"),
               "sc, c       (no furnace)": ("sc", "c"),
               "sh, h       (no container)": ("sh", "h"),
               "s           (sample alone)": ("s",),
               "all 7 runs together": RUNS}
for name, runs in experiments.items():
    parts, perr, chi2 = general_correction({y: I[y] for y in runs}, Am,
                                           {y: d[y] for y in runs}, {y: e[y] for y in runs})
    print(f"\n{name}: corrected / true - 1 (+- error), at 2theta = 0, 90, 180 deg")
    for x, v in parts.items():
        # error: least-squares solution and reference <L> (true_err) in quadrature
        tot = np.hypot(perr[x] / true[x], true_err[x])
        print(f"   I_{x}: " + "   ".join(f"{v[i] / true[x] - 1:+.4f} +- {tot[i]:.4f}" for i in IDX))
    if len(runs) > len(parts):
        print(f"   chi2 at 0, 30, ..., 180 deg = " + " ".join(f"{c:.2f}" for c in chi2[::15])
              + f"   (degrees of freedom {len(runs) - len(parts)});"
              + f" mean over all angles {chi2.mean():.2f}")

# 5) With sch, ch, h the least squares is exactly the Paalman & Pings chain
p_ls, _, _ = general_correction({y: I[y] for y in ("sch", "ch", "h")}, Am, d)
s_pp, c_pp, h_pp = correct_chain(I["sch"], I["ch"], I["h"], Am, d)
print("\nleast squares - Paalman & Pings chain, largest |difference| / I_s:",
      f"{np.max(np.abs(p_ls['s'] - s_pp)) / true['s']:.1e}")
