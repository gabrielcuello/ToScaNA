# Example 11 - A disc (circular plate) sample (version 2.3)
import numpy as np
from toscana.corrections.mcams.sample_base import Layer, scattered_direction
from toscana.corrections.mcams.disc import create_disc_sample
from toscana.corrections.mcams.beam import sample_beam_point
from toscana.corrections.mcams.multiple_scattering import simulate_scattering_orders
from toscana.corrections.mcams.direct_attenuation import attenuation_factors_direct

BEAM_W, BEAM_H = 8.0, 8.0          # beam size (mm), centred on the disc

# A circular flat cell, 20 mm in diameter: two vanadium windows of 0.2 mm
# and 2 mm of Ni powder between them. Layers from the front face to the back.
cell = create_disc_sample(20.0, [("V",  0.2, 0.046, 0.78),
                                 ("Ni", 2.0, 0.180, 0.94),
                                 ("V",  0.2, 0.046, 0.78)], beam_height=BEAM_H)
print(cell.summary())

# Where do the particles interact? (fractions of the incident particles)
x, y = sample_beam_point(BEAM_W, BEAM_H, n=200000, rng=1)
points, hit = cell.sample_interaction_points(x, y, rng=2)
print("\ninteracting per layer:", np.round(np.bincount(cell.layer_of_points(points),
                                                     minlength=3) / len(x), 4))
print("Beer-Lambert for the Ni :", round(np.exp(-0.046 * 0.2) * (1 - np.exp(-0.18 * 2.0)), 4))

# Transmission towards the detector at three angles (exact directions)
for tt in (30.0, 90.0, 150.0):
    d = cell.to_sample_frame(scattered_direction(tt))
    T = cell.transmission(cell.outgoing_path_lengths(points, d)).mean()
    print(f"transmission at 2theta = {tt:5.1f} deg: {T:.4f}")

# The same disc rotated: plane at 45 degrees to the beam
tilted = cell.with_plane_angle(45.0)
pts_t, hit_t = tilted.sample_interaction_points(x, y, rng=2)
print("\nfraction interacting, plane angle 90 / 45 deg:", round(hit.mean(), 4), round(hit_t.mean(), 4))

# A beam larger than the disc: only the particles that meet the disc can
# interact. face_area_in_beam gives the part of the beam that hits the face.
big = 30.0
xb, yb = sample_beam_point(big, big, n=200000, rng=3)
for smp in (cell, tilted):
    _, h = smp.sample_interaction_points(xb, yb, rng=4)
    print(f"plane angle {smp.plane_angle_deg:4.0f}: beam area hitting the face "
          f"{smp.face_area_in_beam(big, big) / big**2:.4f}, interacting {h.mean():.4f}")

# Partially filled disc: Ni only up to y = +1 mm (heights from the beam centre)
half = create_disc_sample(20.0, [("V", 0.2, 0.046, 0.78),
                                 Layer("Ni", 2.0, 0.180, 0.94, bottom=-10.0, height=11.0),
                                 ("V", 0.2, 0.046, 0.78)], beam_height=BEAM_H)
print("\nfullness of the layers:", half.beam_fullness())

# Single and multiple scattering, and the exact single-scattering factors
res = simulate_scattering_orders(cell, BEAM_W, BEAM_H, n_particles=300000, rng=5)
print(f"multiple / single = {res.multiple.sum() / res.single.sum():.3f}")
r = attenuation_factors_direct(cell, BEAM_W, BEAM_H, angles_deg=[0, 45, 135],
                               n_points=50000, rng=6)
print("Paalman-Pings factors of the Ni (0, 45, 135 deg):", np.round(r.factors[1], 4))
