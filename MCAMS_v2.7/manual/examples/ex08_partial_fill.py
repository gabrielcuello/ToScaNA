# Example 8 - A partially filled container
import numpy as np
from sample_base import Layer
from cylinder import create_cylindrical_sample
from slab import create_slab_sample
from beam import sample_beam_point
from multiple_scattering import simulate_scattering_orders

BEAM_W, BEAM_H = 8.0, 8.0          # beam size (mm), centred on y = 0

# A vanadium can (inner diameter 6 mm, wall 0.2 mm, 30 mm high) whose bottom
# is at the bottom of the beam (y_bottom = -4 mm). The powder fills it up to
# half the beam height: fullness 0.5, i.e. 4 mm of powder.
can = create_cylindrical_sample(
    30, [Layer("powder", 6.0, 0.250, 0.8, fullness=0.5),
         ("V", 0.2, 0.073, 0.5)],              # the wall is full
    y_bottom=-BEAM_H / 2, beam_height=BEAM_H)
print(can.summary())
print("fill heights (mm):", can.fill_heights, " fullness:", can.beam_fullness())

# The same filling given as a height: 5th element of the tuple (mm above the
# bottom of the sample). Both descriptions give the same sample.
same = create_cylindrical_sample(30, [("powder", 6.0, 0.250, 0.8, 4.0),
                                      ("V", 0.2, 0.073, 0.5)], y_bottom=-4.0)
print("same level of powder:", np.allclose(same.fill_tops, can.fill_tops))

# Where do the particles of the beam interact, for several fullness values?
x, y = sample_beam_point(BEAM_W, BEAM_H, n=200000, rng=1)
print("\nfullness  powder   wall   (fraction of the incident particles)")
for f in (1.0, 0.75, 0.5, 0.25):
    s = create_cylindrical_sample(
        30, [Layer("powder", 6.0, 0.250, 0.8, fullness=f), ("V", 0.2, 0.073, 0.5)],
        y_bottom=-BEAM_H / 2, beam_height=BEAM_H)
    points, hit = s.sample_interaction_points(x, y, rng=2)
    frac = np.bincount(s.layer_of_points(points), minlength=2) / len(x)
    print(f"  {f:4.2f}   {frac[0]:.4f}  {frac[1]:.4f}")

# Single and multiple scattering of the half-full can
res = simulate_scattering_orders(can, BEAM_W, BEAM_H, n_particles=300000, rng=3)
single, multiple = res.single.sum(), res.multiple.sum()
print(f"\nhalf full: single {single:.0f}, multiple {multiple:.0f} "
      f"({100 * multiple / (single + multiple):.1f} % multiple)")

# A flat cell: Al windows (full) and water up to 6 mm above the bottom of a
# 20 mm high cell centred on the beam (bottom at -10 mm, level at -4 mm):
# the water does not reach the beam at all, whose bottom is at -4 mm.
cell = create_slab_sample(20, 20, [("Al", 1.0, 0.010),
                                   ("water", 2.0, 0.56, 1.0, 6.0),
                                   ("Al", 1.0, 0.010)], beam_height=BEAM_H)
print()
print(cell.summary())
