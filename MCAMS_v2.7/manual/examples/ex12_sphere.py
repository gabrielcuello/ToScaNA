# Example 12 - A spherical sample (version 2.4)
import numpy as np
from sample_base import Layer, scattered_direction
from sphere import create_spherical_sample
from beam import sample_beam_point
from multiple_scattering import simulate_scattering_orders
from direct_attenuation import attenuation_factors_direct

BEAM_W, BEAM_H = 8.0, 8.0          # beam size (mm), centred on the sphere

# A Ni powder sphere of diameter 6 mm in a vanadium sphere with a 0.2 mm wall.
# Layers from the inner one outwards: first thickness = DIAMETER.
ball = create_spherical_sample([("Ni", 6.0, 0.180, 0.94),
                                ("V",  0.2, 0.046, 0.78)], beam_height=BEAM_H)
print(ball.summary())
print("radii (mm):", ball.radii)

# The beam (8 x 8 mm) is larger than the sphere (6.4 mm): part of it misses.
x, y = sample_beam_point(BEAM_W, BEAM_H, n=200000, rng=1)
points, hit = ball.sample_interaction_points(x, y, rng=2)
print("\nbeam fraction meeting the sphere:",
      round(ball.section_area_in_beam(BEAM_W, BEAM_H) / (BEAM_W * BEAM_H), 4))
print("interacting per layer:", np.round(np.bincount(ball.layer_of_points(points),
                                                     minlength=2) / len(x), 4))

# Transmission towards the detector. A sphere looks the same from every
# direction, so the detector may be in or out of the horizontal plane.
for tt in (0.0, 90.0, 180.0):
    T_in = ball.transmission(ball.outgoing_path_lengths(points, scattered_direction(tt))).mean()
    T_up = ball.transmission(ball.outgoing_path_lengths(points, scattered_direction(tt, 90.0))).mean()
    print(f"2theta = {tt:5.1f}: transmission {T_in:.4f} (horizontal plane), {T_up:.4f} (vertical plane)")

# A spherical flask (glass wall 1 mm) half filled with water: the water goes
# from the bottom of the inner sphere (y = -10) up to the centre of the beam.
flask = create_spherical_sample([Layer("water", 20.0, 0.56, 1.0, bottom=-10.0, height=10.0),
                                 ("glass", 1.0, 0.03, 1.0)], beam_height=BEAM_H)
print()
print(flask.summary())

# Single and multiple scattering, and the exact single-scattering factors
res = simulate_scattering_orders(ball, BEAM_W, BEAM_H, n_particles=300000, rng=3)
print(f"\nmultiple / single = {res.multiple.sum() / res.single.sum():.3f}")
r = attenuation_factors_direct(ball, BEAM_W, BEAM_H, angles_deg=[0, 90, 180],
                               n_points=50000, rng=4)
print("Paalman-Pings factors of the Ni (0, 90, 180 deg):", np.round(r.factors[0], 4))
