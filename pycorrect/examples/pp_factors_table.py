"""
Tabulate the Paalman & Pings factors and the Blech & Averbach multiple-
scattering ratio for a given geometry, ready to compare with Monte Carlo
trajectory simulations.

    python examples/pp_factors_table.py  > pp_table.txt

Edit the GEOMETRY section below.  Units: cm, atoms/A^3, barn, A.
Output columns: 2theta(deg)  Q(1/A)  A_s,sch  A_c,ch  A_c,sch  A_h,sch  A_h,ch  A_h,h
plus, in the header, delta (multiple/single scattering, all orders) and the
tabulated Blech-Averbach double-scattering ratio delta_2.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pycorrect import atcors, mscat          # noqa: E402

# ---------------- GEOMETRY ----------------------------------------------
lam = 0.5                    # wavelength [A]
beam_width, beam_height = 1.5, 5.0
r_sample, r_can = 0.30, 0.32                 # r_can <= r_sample -> no can
r_fur_in, r_fur_out = 0.0, 0.0               # r_fur_out <= r_fur_in -> no furnace
# (number density, sigma_scatt, sigma_abs at 1.798 A)
sample = (0.0913, 18.5, 4.49)                # Ni
can = (0.0722, 4.95, 5.08)                   # V
furnace = (0.0722, 4.95, 5.08)
faithful = False                             # True: CORRECT 2.24 numbers
# -------------------------------------------------------------------------

mu_t = lambda m: m[0] * (m[1] + m[2] * lam / 1.798)    # total mu [1/cm]
mu_s = lambda m: m[0] * m[1]                           # scattering mu
smu, cmu, hmu = mu_t(sample), mu_t(can), mu_t(furnace)

fs = (r_sample / r_can) ** 2 if r_can > r_sample else 1.0
rr = max(r_can, r_sample)
delta, delta2 = mscat(rr, beam_height, (1 - fs) * cmu + fs * smu,
                      (1 - fs) * mu_s(can) + fs * mu_s(sample),
                      faithful=faithful, verbose=False, return_details=True)
print(f"# lambda={lam} A  mu_sample={smu:.4f} 1/cm  mu*R={smu * r_sample:.4f}")
print(f"# Blech-Averbach: delta_2(table)={delta2:.5f}  delta(all orders, x sigma_s/sigma_t)={delta:.5f}")
print("# 2theta    Q       Assch    Acch     Acsch    Ahsch    Ahch     Ahh")
for tt in np.arange(0.0, 181.0, 5.0):
    a = atcors(math.radians(tt), r_sample, r_can, r_fur_in, r_fur_out, beam_width, smu, cmu, hmu)
    q = 4 * math.pi / lam * math.sin(math.radians(tt) / 2)
    print(f"{tt:7.2f} {q:7.3f} " + " ".join(f"{x:8.5f}" for x in a))
