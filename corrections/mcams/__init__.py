"""
MCAMS 2.7 -- Monte Carlo Attenuation and Multiple Scattering.

Integrated into ToScaNA as ``toscana.corrections.mcams`` (the modules were
stand-alone scripts importing each other as ``import beam``; they now use
package-relative imports, with no other change of behaviour). Units: mm,
mm^-1. Manual: ``docs/mcams/manual``; checks: ``tests/mcams_checks.py``.

Modules (import them explicitly, e.g.
``from toscana.corrections.mcams.cylinder import create_cylindrical_sample``)
--------------------------------------------------------------------------
beam
    Start points of the particles of a rectangular incident beam.
sample_base
    Layer and LayeredSample: what all sample geometries share.
slab, cylinder, disc, sphere
    The geometries: flat layers; concentric cylinders (annuli); flat discs;
    concentric spheres.
sample_factory
    make_sample: build a sample of the chosen geometry.
multiple_scattering
    Simulation of scattering orders (particles leaving after 1, 2, ... events).
transmission_vs_angle, transmission_by_layer
    Transmission of scattered particles versus angle, by last layer.
direct_attenuation
    Single-scattering attenuation factors by direct integration or Monte Carlo.
formula, nuclear_data, layer_builder
    Chemical formulas; atomic masses and cross sections; Layer input from a
    formula, a density and a geometry.
run_simulation
    Main script: a complete simulation from a parameter set.
"""
