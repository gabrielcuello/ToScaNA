# Example 9 - Direct method: single-scattering attenuation factors as a reference
import numpy as np
from toscana.corrections.mcams.sample_factory import make_sample
from toscana.corrections.mcams.direct_attenuation import (attenuation_factors_direct, attenuation_factors_mc,
                                compare_results)

# Ni rod (6 mm) in a V can (0.2 mm): (name, thickness mm, mu 1/mm, scatter fraction)
layers = [("Ni", 6.0, 0.1803, 0.937), ("V", 0.2, 0.0459, 0.779)]
sample = make_sample("cylinder", layers, height=50)
angles = np.arange(0, 181, 10.0)

# 1) The reference: direct method, 100 000 points in each layer
ref = attenuation_factors_direct(sample, 15, 20, angles_deg=angles,
                                 n_points=100_000, rng=1)
print(f"direct method: {ref.seconds:.1f} s, points per layer {ref.n_samples}")
print("mean path of the beam in each layer (mm):", np.round(ref.mean_path, 4))
for k, name in enumerate(ref.layer_names):
    print(f"  A_{name}(0, 90, 180 deg) = " + ", ".join(
        f"{ref.factors[k, i]:.4f}+-{ref.errors[k, i]:.4f}" for i in (0, 9, 18)))

# 2) The same quantity from the Monte Carlo, normalised with the same mean paths
mc = attenuation_factors_mc(sample, 15, 20, angles_deg=angles, n_particles=25_000,
                            n_batches=8, mean_path=ref.mean_path, rng=2)
print(f"\nMonte Carlo: {mc.seconds:.1f} s")
print(compare_results(ref, mc, every=6))

# 3) Single scattering per incident particle and per steradian (isotropic)
print("\nsingle scattering / incident / sr at 0, 90, 180 deg:",
      np.round(ref.single_per_steradian()[[0, 9, 18]], 5))
# To draw:  plot_attenuation_factors([ref, mc])
