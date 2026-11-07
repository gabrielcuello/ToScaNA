"""
Corrections for attenuation, multiple scattering, inelasticity and
normalisation of neutron diffraction data.

Subpackages and modules (not imported automatically; import what you need)
-----------------------------------------------------------------------
setup
    Description of a set-up as concentric cylinders (``CylinderSetup``,
    ``setups_from_measurement``, materials), exact transmissions and atoms
    per beam area.
factors
    ``compute_factors``: Paalman & Pings factors A_b,C, multiple scattering
    δ_C and transmissions T_C of a set-up with MCAMS (or pycorrect);
    ``FactorSet`` (save on the angle or Q scale), ``load_factors``.
pycorrect
    Python translation of CORRECT 2.24 (M.A. Howe, R.L. McGreevy,
    P. Zetterstrom): Paalman & Pings attenuation, Blech & Averbach multiple
    scattering, Placzek, vanadium normalisation; reads CORRECT ``.com``
    files. Command line: ``python -m toscana.corrections.pycorrect run.com``.
    Units: cm, cm^-1 (CORRECT convention).
mcams
    MCAMS 2.7, Monte Carlo attenuation and multiple scattering for cylinder,
    annulus, slab, disc and sphere geometries, partial filling, by-layer
    analysis. Units: mm.
bridge
    Computes with MCAMS the quantities pycorrect uses (Paalman & Pings
    factors, delta, delta_2) from a pycorrect state; converts cm <-> mm.

Manuals: ``docs/pycorrect``, ``docs/mcams``; validation of pycorrect against
MCAMS: ``docs/validation``.
"""
