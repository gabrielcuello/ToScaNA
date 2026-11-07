"""
Regenerate the figures and the console excerpts used in pycorrect_manual.tex.

    cd manual
    python make_figures.py

It (1) builds the synthetic D4 example with examples/make_d4_example.py,
(2) runs pycorrect on it, and (3) writes PDF figures to manual/figures/ and
text excerpts to manual/listings/.
"""
import contextlib
import io
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# toscana must be importable (ToScaNA manual, chapter Installation).
from toscana.corrections.pycorrect import Correct, atcors, mscat                       # noqa: E402
from toscana.corrections.pycorrect.ndplib import load_double_scattering_table          # noqa: E402

FIG = os.path.join(HERE, "figures")
LST = os.path.join(HERE, "listings")
os.makedirs(FIG, exist_ok=True)
os.makedirs(LST, exist_ok=True)

# Reference categorical palette (validated: slots 1-3 pass all-pairs CVD)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


# ---------------------------------------------------------------- example
subprocess.run([sys.executable, os.path.join(ROOT, "examples", "make_d4_example.py")], check=True)
data = os.path.join(ROOT, "examples", "d4_data")
os.chdir(data)
if os.path.exists("corrected.dat"):
    os.remove("corrected.dat")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    c = Correct(verbose=True, ask=False)
    c.run_file("d4_example.com")
log = buf.getvalue().splitlines()
with open(os.path.join(LST, "d4_console.txt"), "w") as fh:
    # keep the head of the sample table only
    out, n = [], 0
    for line in log:
        out.append(line)
    sel = [l for l in out if l.strip()]
    i = next(k for k, l in enumerate(out) if "Sample absorption" in l)
    fh.write("\n".join(out[i:i + 14]) + "\n")
with open("corrected.dat") as fh:
    lines = fh.read().splitlines()
with open(os.path.join(LST, "d4_output_head.txt"), "w") as fh:
    fh.write("\n".join(lines[:19]) + "\n   ...\n")
shutil.copy("d4_example.com", os.path.join(LST, "d4_example.com"))

r = c.results[0]
true = np.loadtxt("true_dcs.txt")

# Fig: corrected vs true
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2]})
a1.plot(r["q"], r["dcs"], color=C1, lw=0.8, label="pycorrect output")
a1.plot(true[:, 0], true[:, 1], color=C2, lw=1.6, label="input (true) $d\\sigma/d\\Omega$")
a1.set_ylabel("$d\\sigma/d\\Omega$  (barn sr$^{-1}$ atom$^{-1}$)")
a1.legend(loc="upper right")
diff = r["dcs"] - np.interp(r["q"], *true.T)
a2.plot(r["q"], diff, color=INK2, lw=0.6)
a2.axhline(0, color=INK, lw=0.8)
a2.set_xlabel("$Q$ (Å$^{-1}$)")
a2.set_ylabel("difference")
save(fig, "d4_result.pdf")

# Fig: vanadium raw vs smoothed
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.plot(r["theta"], r["calib_raw"], color=INK2, lw=0.5, label="vanadium, corrected (raw)")
ax.plot(r["theta"], r["calib_smooth"], color=C1, lw=1.6, label="Savitzky–Golay, /smoothing=-5")
ax.set_xlabel("$2\\theta$ (deg)")
ax.set_ylabel("counts")
ax.legend(loc="lower left")
save(fig, "vanadium_smoothing.pdf")

# Fig: P&P factors vs 2theta
os.chdir(HERE)
lam = 0.4985
smu = 0.0913 * (18.5 + 4.49 * lam / 1.798)
cmu = 0.0722 * (4.95 + 5.08 * lam / 1.798)
tt = np.linspace(0, 180, 37)
A = np.array([atcors(math.radians(t), 0.30, 0.32, 0, 0, 1.5, smu, cmu, 0) for t in tt])
fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.plot(tt, A[:, 0], color=C1)
ax.plot(tt, A[:, 2], color=C2)
ax.plot(tt, A[:, 1], color=C3)
for y, lab, col in ((A[-1, 0], "$A_{s,sc}$", C1), (A[-1, 2], "$A_{c,sc}$", C2), (A[-1, 1], "$A_{c,c}$", C3)):
    ax.annotate(lab, (180, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
ax.set_xlabel("$2\\theta$ (deg)")
ax.set_ylabel("attenuation factor")
ax.set_xlim(0, 196)
ax.set_xticks(range(0, 181, 30))
save(fig, "pp_factors.pdf")

# Fig: Blech & Averbach table
rmu, ratio, d = load_double_scattering_table()
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for j, col in ((1, C1), (9, C2), (19, C3)):
    ax.plot(rmu, d[:, j], color=col, marker="o", ms=3, label=f"$R/H$ = {ratio[j]:.2f}")
ax.set_xlabel("$\\mu R$")
ax.set_ylabel("$\\delta_2$ = double / single")
ax.legend(loc="lower right")
save(fig, "blech_averbach.pdf")

with open(os.path.join(LST, "pp_table_head.txt"), "w") as fh:
    out = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "pp_factors_table.py")],
                         capture_output=True, text=True).stdout.splitlines()
    fh.write("\n".join(out[:9]) + "\n   ...\n")
print("figures in", FIG)
