# Example 2 - Building the two kinds of sample
from toscana.corrections.mcams.slab import create_slab_sample
from toscana.corrections.mcams.cylinder import create_cylindrical_sample
from toscana.corrections.mcams.sample_factory import make_sample

# Layers are tuples (name, thickness in mm, mu in 1/mm, scatter fraction).
# The last number is optional (default 1.0).

# A slab, 20 x 20 mm, layers from the front face to the back, rotated so
# that the angle between the beam and the plane of the sample is 60 degrees.
slab = create_slab_sample(20, 20, [("Al", 2.0, 0.020, 1.0),
                                   ("Fe", 1.0, 0.250, 0.6),
                                   ("Cu", 0.5, 0.300, 0.5)],
                          plane_angle_deg=60)
print(slab.summary())

# Concentric cylinders, 20 mm high, from the axis outwards. The thickness of
# the first layer is its DIAMETER (6 mm: radius 3 mm); the others are radial.
cylinder = create_cylindrical_sample(20, [("Fe", 6.0, 0.250, 0.6),
                                          ("Al", 1.0, 0.020, 1.0),
                                          ("Cu", 0.5, 0.300, 0.5)])
print()
print(cylinder.summary())

# The same thing through the "factory": only one word changes.
same_cylinder = make_sample("cylinder", [("Fe", 6.0, 0.250, 0.6),
                                         ("Al", 1.0, 0.020, 1.0),
                                         ("Cu", 0.5, 0.300, 0.5)], height=20)
print()
print("outer radius of the cylinder:", same_cylinder.outer_radius, "mm")
