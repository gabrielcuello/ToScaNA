# Example 4 - Transmission versus scattering angle
from sample_factory import make_sample
from transmission_vs_angle import transmission_vs_angle

slab = make_sample("slab", [("Al", 2.0, 0.020), ("Fe", 1.0, 0.250), ("Cu", 0.5, 0.300)],
                   height=20, width=20)

for plane_angle in (90, 60, 30):
    rotated = slab.with_plane_angle(plane_angle)       # same sample, other orientation
    angles, T, f_int = transmission_vs_angle(rotated, beam_width=8, beam_height=8,
                                             n_particles=50000, rng=1)
    print(f"plane angle {plane_angle:2d} deg: fraction interacting = {f_int:.3f}, "
          f"T(30) = {T[30]:.3f}, T(90) = {T[90]:.3f}, T(150) = {T[150]:.3f}")
# To draw the curve:  plot_transmission(angles, T)
