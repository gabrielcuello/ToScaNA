# Example 10 - Layers with their own bottom and height (version 2.2)
import numpy as np
from toscana.corrections.mcams.sample_base import Layer
from toscana.corrections.mcams.cylinder import create_cylindrical_sample
from toscana.corrections.mcams.slab import create_slab_sample
from toscana.corrections.mcams.beam import sample_beam_point

BEAM_W, BEAM_H = 8.0, 8.0          # beam size (mm): from y = -4 to y = +4

# A vanadium can from y = -12 to +18 mm (30 mm high). The powder rests on a
# support: it starts at y = -2, i.e. 2 mm above the bottom of the beam
# (y = -4), and is 9 mm high. Heights are measured from the CENTRE of the beam.
# height=None: the sample is the envelope of the layers (-12 to +18).
can = create_cylindrical_sample(
    None, [Layer("powder", 6.0, 0.250, 0.8, bottom=-2.0, height=9.0),
           Layer("V",      0.2, 0.073, 0.5, bottom=-12.0, height=30.0)],
    beam_height=BEAM_H)
print(can.summary())
print("material from", can.fill_bottoms, "to", can.fill_tops, "mm")
print("fullness (output):", can.beam_fullness())     # powder: 6 of the 8 mm

# Where do the particles interact? Below y = -2 the beam crosses only the walls.
x, y = sample_beam_point(BEAM_W, BEAM_H, n=200000, rng=1)
pts, hit = can.sample_interaction_points(x, y, rng=2)
layer = can.layer_of_points(pts)
print("\ninteractions in the powder: y from %.2f to %.2f mm"
      % (pts[layer == 0, 1].min(), pts[layer == 0, 1].max()))

# A band of powder 4 mm high moved through the beam: the number of
# interactions in the powder only depends on the overlap with the beam.
print("\nband bottom  fullness  powder   wall   (fraction of the incident particles)")
for b in (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0):
    s = create_cylindrical_sample(
        None, [Layer("powder", 6.0, 0.250, 0.8, bottom=b, height=4.0),
               Layer("V", 0.2, 0.073, 0.5, bottom=-15.0, height=30.0)], beam_height=BEAM_H)
    pts, hit = s.sample_interaction_points(x, y, rng=2)
    frac = np.bincount(s.layer_of_points(pts), minlength=2) / len(x)
    print(f"   {b:+5.1f}      {s.beam_fullness()[0]:4.2f}    {frac[0]:.4f}  {frac[1]:.4f}")

# The forms of version 2.0 are still accepted, with the same meaning:
# fill_height = bottom of the sample + height.
old = create_cylindrical_sample(30, [("powder", 6.0, 0.250, 0.8, 9.0), ("V", 0.2, 0.073, 0.5)],
                                y_bottom=-12.0)
new = create_cylindrical_sample(30, [Layer("powder", 6.0, 0.250, 0.8, bottom=-12.0, height=9.0),
                                     ("V", 0.2, 0.073, 0.5)], y_bottom=-12.0)
print("\nfill_height 9 == bottom -12, height 9:",
      np.allclose(old.fill_tops, new.fill_tops) and np.allclose(old.fill_bottoms, new.fill_bottoms))

# A flat cell whose windows start INSIDE the beam (at y = -2): the lowest
# 2 mm of the beam pass below the cell. The liquid fills it up to y = +1.
cell = create_slab_sample(20, None, [
    Layer("Al",    1.0, 0.010, 1.0, bottom=-2.0, height=20.0),
    Layer("water", 3.0, 0.560, 1.0, bottom=-2.0, height=3.0),
    Layer("Al",    1.0, 0.010, 1.0, bottom=-2.0, height=20.0)], beam_height=BEAM_H)
print()
print(cell.summary())
