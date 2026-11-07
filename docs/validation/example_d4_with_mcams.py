"""
example_d4_with_mcams.py - Use MCAMS to compute the corrections of a real
pycorrect run, on the same 2theta grid, and compare them.

Steps
-----
1. Build the synthetic D4 data set of the pycorrect package
   (examples/make_d4_example.py: Ni in a V can, V rod, lambda = 0.4985 A).
2. Read its command file with pycorrect, run the corrections, and keep the
   intermediate arrays (results[0]: assch, acsch, acch, delta_sa, ...).
3. Give the SAME pycorrect state to
   pyc_mcams_bridge.mcams_factors_like_pycorrect(), which builds the MCAMS
   samples, runs the Monte Carlo and returns the factors with the same names.
4. Save a table and a figure (figures/D4_example.pdf).

Usage:   python example_d4_with_mcams.py  [--quick]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toscana.corrections import bridge as br
from toscana.corrections.pycorrect import Correct

HERE = os.path.dirname(os.path.abspath(__file__))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight"})

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
quick = ap.parse_args().quick

# --- 1. synthetic D4 data from the pycorrect package --------------------------
pyc_root = os.path.abspath(os.path.join(HERE, "..", "pycorrect"))   # docs/pycorrect
subprocess.run([sys.executable, os.path.join(pyc_root, "examples", "make_d4_example.py")], check=True)
data = os.path.join(pyc_root, "examples", "d4_data")
cwd = os.getcwd()
os.chdir(data)

# --- 2. pycorrect -------------------------------------------------------------------
c = Correct(verbose=False, ask=False)
c.st.overwrite = True
c.run_file("d4_example.com")
r = c.results[0]
theta = r["theta"]
os.chdir(cwd)

# --- 3. MCAMS on the same set-up ------------------------------------------------
mc = br.mcams_factors_like_pycorrect(c.st, theta,
                                     n_particles=40_000 if quick else 300_000,
                                     n_batches=4 if quick else 8,
                                     ms_particles=15_000 if quick else 60_000)

# --- 4. table and figure ----------------------------------------------------------
keys = ["assch", "acsch", "acch", "avvch"]
cols = [theta]
head = "2theta"
for k in keys:
    cols += [r[k], mc[k], mc[k + "_err"]]
    head += f"  {k}_pyc {k}_mc err"
cols += [np.full(len(theta), r["delta_sa"]), mc["delta_sa"], mc["delta_sa_err"],
         np.full(len(theta), r["delta_va"]), mc["delta_va"], mc["delta_va_err"]]
head += "  delta_sa_pyc delta_sa_mc err  delta_va_pyc delta_va_mc err"
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
np.savetxt(os.path.join(HERE, "results", "D4_example.txt"), np.column_stack(cols)[::40],
           fmt="%9.5f", header="pycorrect D4 example (every 40th point)\n" + head)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
for i, (k, tex) in enumerate(zip(keys, ["$A_{s,sc}$", "$A_{c,sc}$", "$A_{c,c}$", "$A_{v,v}$"])):
    a1.plot(theta, r[k], color=C[i], lw=1.6)
    a1.plot(theta[::60], mc[k][::60], "o", ms=3, color=C[i], mfc="white")
    a1.annotate(tex, (theta[-1], r[k][-1]), xytext=(4, 0), textcoords="offset points",
                va="center", color=INK, fontsize=8)
a1.set_xlabel("2θ (deg)"); a1.set_ylabel("attenuation factor"); a1.set_xlim(0, 160)
a1.plot([], [], color=INK2, label="pycorrect"); a1.plot([], [], "o", color=INK2, mfc="white", label="MCAMS")
a1.legend(fontsize=8, loc="center right")
for i, (k, tex) in enumerate((("delta_sa", "sample"), ("delta_va", "vanadium"))):
    a2.axhline(r[k], color=C[i], lw=1.6)
    a2.fill_between(theta, mc[k] - mc[k + "_err"], mc[k] + mc[k + "_err"], color=C[i], alpha=0.25, lw=0)
    a2.plot(theta, mc[k], color=C[i], lw=1.0, ls="--", label=f"{tex}")
a2.set_xlabel("2θ (deg)"); a2.set_ylabel("δ = multiple / single")
a2.set_ylim(0, 0.75)
a2.legend(fontsize=7, loc="center right", title="solid: pycorrect   dashed: MCAMS", title_fontsize=7)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "D4_example.pdf"))
print(f"delta_sa: pycorrect {r['delta_sa']:.4f}, MCAMS {mc['delta_sa'].min():.4f}-{mc['delta_sa'].max():.4f}")
print(f"delta_va: pycorrect {r['delta_va']:.4f}, MCAMS {mc['delta_va'].min():.4f}-{mc['delta_va'].max():.4f}")
for k in keys:
    print(f"{k}: max |MCAMS/pycorrect - 1| = {100 * np.max(np.abs(mc[k] / r[k] - 1)):.2f} %")
