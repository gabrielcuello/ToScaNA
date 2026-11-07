# Example 17 - CORRECT factors with MCAMS: spherical sample, container and heater (sphere)
import os
import numpy as np
from correct_tools import (correct_samples, attenuation_factors, unattenuated, scattering,
                           print_factors, print_scattering, print_contributions,
                           check_closure, plot_correct, at_angles)
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

BEAM_W, BEAM_H = 10.0, 20.0                   # beam (mm)
ANGLES = np.arange(0.0, 181.0, 1.0)           # 2theta for the factors (direct method), deg
ANG_MC = np.arange(0.0, 181.0, 2.0)           # 2theta for the simulated runs (Monte Carlo), deg
from sphere import create_spherical_sample

# From the centre outwards (first thickness = DIAMETER): Ni powder sphere
# (6 mm), V shell (0.2 mm), vacuum, V heater shell (0.1 mm, radius 15 mm).
layers = [("Ni powder", 6.0,  0.180,  0.94),
          ("V shell",   0.2,  0.0459, 0.78),
          ("vacuum",   11.8,  0.0,    1.0),
          ("V heater",  0.1,  0.0459, 0.78)]
roles = ["s", "c", "-", "h"]
MAKE = lambda L: create_spherical_sample(L)
TITLE, FIGNAME = "Sphere: Ni sphere in a V shell inside a spherical V heater", "fig_correct_sphere.pdf"

# The three samples I_sch, I_ch, I_h: the same geometry, with the parts that
# are not in the measurement made empty (mu = 0).
samples = correct_samples(MAKE, layers, roles)
print(samples["sch"].summary())
for m in ("sch", "ch", "h"):
    print(f"{m:>4}: mu of the layers = {samples[m].mus}")

# 1) The attenuation factors A_x,y (direct method)
A, err, mean_path = attenuation_factors(samples, roles, BEAM_W, BEAM_H, ANGLES)
print("\nAttenuation factors")
print_factors(ANGLES, A, err, every=30)

# 2) Single, multiple and total scattering of the three measurements
# (delta varies slowly with the angle: a 2 degree grid is enough, and it is
# interpolated onto the 1 degree grid of the factors when needed)
sc = scattering(samples, roles, BEAM_W, BEAM_H, ANG_MC, n_particles=200_000, n_batches=10)
print("\nScattering of the three measurements")
print_scattering(ANG_MC, sc, every=15)
print()
print_contributions(ANG_MC, sc, "sch", every=30)

# 3) Closure: the simulated totals, corrected as CORRECT does with these
#    factors, must give back the true unattenuated scattering of each part.
true, true_err = unattenuated(samples["sch"], roles, BEAM_W, BEAM_H)
print("\nCorrected / true - 1")
A_mc = at_angles(A, ANGLES, ANG_MC)          # the factors at the angles of the runs
check_closure(ANG_MC, sc, A_mc, true, true_err, every=15)

# 4) Figure: factors and multiple scattering versus angle
plot_correct(ANGLES, A, sc, TITLE, os.path.join(FIGURES, FIGNAME), angles_mc=ANG_MC)
