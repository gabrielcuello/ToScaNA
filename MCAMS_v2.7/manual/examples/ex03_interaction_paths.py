# Example 3 - Interaction points and paths to the detector
import numpy as np
from beam import sample_beam_point
from sample_base import scattered_direction
from sample_factory import make_sample

layers = [("Fe", 6.0, 0.250, 0.6), ("Al", 1.0, 0.020, 1.0), ("Cu", 0.5, 0.300, 0.5)]
sample = make_sample("cylinder", layers, height=20)

# 1) where 100000 particles start in an 8 x 8 mm beam
x, y = sample_beam_point(8, 8, n=100000, rng=1)

# 2) where they interact (rows of `points`), and which ones interact at all
points, interacted = sample.sample_interaction_points(x, y, rng=2)
print("fraction of the particles that interact:", round(interacted.mean(), 4))
layer = sample.layer_of_points(points)
print("interactions per layer:", np.bincount(layer, minlength=sample.n_layers))

# 3) direction of the detector at 2 theta = 60 degrees (lab frame), converted
#    to the frame of the sample, and path in each layer from every
#    interaction point to the exit of the sample
direction = sample.to_sample_frame(scattered_direction(60.0))
lengths = sample.outgoing_path_lengths(points, direction)      # (N, n_layers), in mm
print("mean path in each layer (mm):", np.round(lengths.mean(axis=0), 3))

# 4) Beer-Lambert attenuation exp(-sum(mu * L)) of every particle
print("mean transmission at 2theta = 60 deg:", round(sample.transmission(lengths).mean(), 4))
