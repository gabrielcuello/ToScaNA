# Example 5 - Particles scattered once (single) or more (multiple)
from toscana.corrections.mcams.sample_factory import make_sample
from toscana.corrections.mcams.multiple_scattering import simulate_scattering_orders, henyey_greenstein

layers = [("Fe", 6.0, 0.250, 0.6), ("Al", 1.0, 0.020, 1.0), ("Cu", 0.5, 0.300, 0.5)]
sample = make_sample("cylinder", layers, height=20)

res = simulate_scattering_orders(sample, 8, 8, n_particles=300000, rng=2)

print("incident particles        :", res.n_incident)
print("no interaction (cross/miss):", res.n_no_interaction)
print("absorbed                  :", res.n_absorbed)
print("scattered and exited      :", res.n_exited)
single, multiple = res.single.sum(), res.multiple.sum()
print(f"single / multiple         : {single:.0f} / {multiple:.0f}  "
      f"({100 * multiple / (single + multiple):.1f} % multiple)")
print("bin centres of 2theta (first 4):", res.angles_deg[:4])
print("single counts in those bins   :", res.single[:4])

# Same simulation with a forward-peaked angular distribution of each scattering
fwd = simulate_scattering_orders(sample, 8, 8, n_particles=300000, rng=2,
                                 deflection_sampler=henyey_greenstein(0.8))
print(f"forward peaked (g = 0.8): multiple = {fwd.multiple.sum():.0f}")
# To draw the curves:  plot_scattering_orders(res, logy=True)
