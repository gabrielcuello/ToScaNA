"""
Build a realistic synthetic D4-like data set (vanadium with Bragg peaks,
empty V can, sample in can, empty instrument background, Poisson noise) and
write the command file  d4_example.com  that corrects it.

    python examples/make_d4_example.py        # writes files in examples/d4_data
    python -m pycorrect examples/d4_data/d4_example.com

All quantities: lengths in cm, densities in atoms/A^3, cross sections in barn.
"""
import math
import os
import sys

import numpy as np

# toscana must be importable (ToScaNA manual, chapter Installation).
from toscana.corrections.pycorrect import atcors, mscat                       # noqa: E402
from toscana.corrections.pycorrect.fileio import DataWriter                   # noqa: E402

rng = np.random.default_rng(1)          # numpy random generator (reproducible)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d4_data")
os.makedirs(out, exist_ok=True)

lam, H, W = 0.4985, 5.0, 1.5            # wavelength, beam height, beam width
R1, R2, Rv = 0.30, 0.32, 0.30           # sample, can outer, vanadium radii
theta = np.arange(1.5, 135.0, 0.125)
q = 4 * math.pi / lam * np.sin(np.radians(theta) / 2)
flux = 2e5 * (1 - 0.15 * (theta / 135) ** 2)

# --- sample: liquid-like structure factor, made of Ni (13.3 b scatt)
sa_rho, sa_s, sa_a = 0.0913, 18.5, 4.49
dcs = sa_s / 4 / math.pi * (1 + 0.6 * np.sin(2.6 * q) / (2.6 * q) * np.exp(-0.02 * q * q) * 3)
va = (0.0722, 4.95, 5.08)
tmu = lambda rho, s, a: rho * (s + a * lam / 1.798)
sa_tmu, va_tmu = tmu(sa_rho, sa_s, sa_a), tmu(*va)

A = np.array([atcors(math.radians(t), R1, R2, 0, 0, W, sa_tmu, va_tmu, 0) for t in theta[::40]])
A = np.array([np.interp(theta, theta[::40], A[:, k]) for k in range(6)])
Av = np.interp(theta, theta[::40],
               [atcors(math.radians(t), Rv, 0, 0, 0, W, va_tmu, 0, 0)[0] for t in theta[::40]])
fs = (R1 / R2) ** 2
d_sa = mscat(R2, H, (1 - fs) * va_tmu + fs * sa_tmu, (1 - fs) * va[0] * va[1] + fs * sa_rho * sa_s, verbose=False)
d_va = mscat(Rv, H, va_tmu, va[0] * va[1], verbose=False)

# V Bragg peaks (bcc a=3.024 A), Caglioti FWHM
peaks = np.zeros_like(theta)
for n in range(1, 10):
    s = lam / 2 / 3.024 * math.sqrt(2 * n)
    if s < 1:
        t2 = 2 * math.degrees(math.asin(s))
        tn = math.tan(math.asin(s))
        fw = math.sqrt(2.595388 * tn * tn - 2.206898 * tn + 0.709038)
        peaks += 0.08 / n * np.exp(-4 * math.log(2) * ((theta - t2) / fw) ** 2)

bkg = 50 + 2000 * np.exp(-theta / 1.5)                     # incl. direct-beam tail
I_v = flux * va[0] * Rv ** 2 * va[1] / 4 / math.pi * Av * (1 + d_va) * (1 + peaks)
I_c = flux * va[0] * (R2 ** 2 - R1 ** 2) * va[1] / 4 / math.pi * (1 + 0.5 * peaks)
I_s = flux * sa_rho * R1 ** 2 * (dcs + d_sa * sa_s / 4 / math.pi)
raw = {"vanadium": I_v + bkg, "can": A[1] * I_c + bkg,
       "sample": A[0] * I_s + A[2] * I_c + bkg, "background": bkg}
for name, y in raw.items():
    y = rng.poisson(y).astype(float)
    with DataWriter(os.path.join(out, name + ".dat"), overwrite=True) as w:
        w.append(theta, y, np.sqrt(np.maximum(y, 1)), instrument="D4", title=name,
                 fnx="Scattering angle (degrees)", fny="Counts")
np.savetxt(os.path.join(out, "true_dcs.txt"), np.c_[q, dcs], header="Q  dsigma/dOmega(true)")

with open(os.path.join(out, "d4_example.com"), "w") as fh:
    fh.write(f"""! Synthetic D4 run: Ni sample in a vanadium can
instrument d4
wavelength {lam}
beam {H} {W}
sample sample.dat {R1} /density={sa_rho} /temperature=300
component 1 ni
container can.dat {R2} /type=v
vanadium vanadium.dat {Rv} /smoothing=-5
background background.dat
placzek series
spectrum 1
output corrected.dat
execute /nolist
""")
print("written in", out)
