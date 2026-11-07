# Example 16 - CORRECT factors with MCAMS: round flat cell between round heater windows (disc)
import os
import numpy as np
from correct_tools import (correct_samples, attenuation_factors, unattenuated, scattering,
                           print_factors, print_scattering, print_contributions,
                           check_closure, plot_correct, at_angles)
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

BEAM_W, BEAM_H = 10.0, 20.0                   # beam (mm)
ANGLES = np.arange(0.0, 181.0, 1.0)           # 2theta for the factors (direct method), deg
ANG_MC = np.arange(0.0, 181.0, 2.0)           # 2theta for the simulated runs (Monte Carlo), deg
from disc import create_disc_sample

# The same stack as Example 15, with a circular section of 30 mm, rotated so
# that the plane of the disc makes 60 degrees with the beam.
layers = [("V heater 1", 0.1,  0.0459, 0.78),
          ("vacuum 1",  10.0,  0.0,    1.0),
          ("V window 1", 0.2,  0.0459, 0.78),
          ("Ni powder",  2.0,  0.180,  0.94),
          ("V window 2", 0.2,  0.0459, 0.78),
          ("vacuum 2",  10.0,  0.0,    1.0),
          ("V heater 2", 0.1,  0.0459, 0.78)]
roles = ["h", "-", "c", "s", "c", "-", "h"]
MAKE = lambda L: create_disc_sample(30.0, L, z_front=-11.25, plane_angle_deg=60.0)
TITLE, FIGNAME = "Disc (30 mm, plane at 60 deg): round V cell with Ni, round V heater windows", "fig_correct_disc.pdf"

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
