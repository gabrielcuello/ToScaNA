# Example 6 - Transmission by layer of last scattering
from sample_factory import make_sample
from transmission_by_layer import transmission_by_last_layer

layers = [("Fe", 6.0, 0.250, 0.6), ("Al", 1.0, 0.020, 1.0), ("Cu", 0.5, 0.300, 0.5)]
sample = make_sample("cylinder", layers, height=20)

res = transmission_by_last_layer(sample, 8, 8, n_particles=50000, rng=3)

print("events per incident particle, last scattering in ...")
for k, name in enumerate(res.layer_names):
    single = res.events("single")[k] / res.n_incident
    multiple = res.events("multiple")[k] / res.n_incident
    print(f"  {name}: single {single:.4f}, multiple {multiple:.4f}")

print("transmission (all orders) for each layer of last scattering:")
print("  angle   " + "   ".join(f"{n:>6}" for n in res.layer_names))
for i in (0, 60, 120, 180):                       # index = angle in degrees (step 1)
    values = "   ".join(f"{res.transmission[k, i]:6.3f}" for k in range(sample.n_layers))
    print(f"  {res.angles_deg[i]:5.0f}   {values}")
# To draw the curves:  plot_transmission_by_layer(res)
