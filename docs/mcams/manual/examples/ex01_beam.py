# Example 1 - Start points of the particles in the beam (beam.py)
import numpy as np
from toscana.corrections.mcams.beam import sample_beam_point

# One particle of a uniform beam 10 mm wide and 6 mm high: two numbers, in mm.
x, y = sample_beam_point(10, 6, rng=1)
print(f"one start point: x = {x:.3f} mm, y = {y:.3f} mm")

# Many particles: two arrays.
xs, ys = sample_beam_point(10, 6, n=100000, rng=1)
print(f"uniform beam  : std(x) = {xs.std():.3f} mm  (expected {10 / np.sqrt(12):.3f})")

# Gaussian horizontal profile (sigma = 1.5 mm), uniform vertical profile.
gaussian = lambda x: np.exp(-x**2 / (2 * 1.5**2))
xs, ys = sample_beam_point(10, 6, H=gaussian, n=100000, rng=1)
print(f"gaussian H(x) : std(x) = {xs.std():.3f} mm  (sigma = 1.5)")
print(f"                std(y) = {ys.std():.3f} mm  (uniform: {6 / np.sqrt(12):.3f})")
