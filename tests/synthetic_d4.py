"""
Synthetic D4 experiment with a known S(Q), for testing the correction chain.

`make_experiment(folder)` writes, in ``folder``, the regrouped data of a Ni
powder in a vanadium can, a vanadium rod and a background, in the format of
ToScaNA's ``.adat`` files (``#``-prefixed CORRECT header), plus a
sample-description file, and returns a dict with the true S(Q).

The raw intensities follow exactly the model that CORRECT inverts:

    I_v  = Φ n_v R_v² σ_v/4π · A_v (1 + δ_v)                      + bkg
    I_c  = Φ n_c (R2² - R1²) σ_c/4π                     (can, unattenuated)
    I_s  = Φ n_s R1² [dσ/dΩ + δ_s σ_s/4π]                (sample)
    raw_can = A_c,c I_c + bkg ;  raw_sample = A_s,sc I_s + A_c,sc I_c + bkg

with Paalman & Pings factors from pycorrect's `atcors` at every angle,
Blech & Averbach δ from `mscat`, no inelasticity, and
dσ/dΩ = σ_s/4π + <b>² [S(Q) - 1], cross sections from ToScaNA's table.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

LAMBDA = 0.4986
THETA = np.arange(1.5, 135.0, 0.125)


def s_true(q: np.ndarray) -> np.ndarray:
    """Liquid-like structure factor: S -> 1 at high Q, peak near 3 1/A."""
    return 1 + 0.6 * np.sin(2.5 * q) / (2.5 * q) * 3 * np.exp(-0.015 * q * q)


def _write_adat(path: pathlib.Path, theta: np.ndarray, y: np.ndarray, e: np.ndarray) -> None:
    """Same header layout as toscana.dataio.d4creg_outputs.saveDiffAngle."""
    with open(path, "w") as fh:
        fh.write(f"# {path.name}\n#Block  1\n#========\n#\n#Instrument:d4\n")
        fh.write("#User      :test    exp_0-00-000/processed\n#Run number: ['000001-000002']\n")
        fh.write("#Spectrum  :            1\n")
        fh.write(f"#Title     :{path.name}\n#Run date  : synthetic\n")
        fh.write("#X caption : 2Theta (degrees)\n#Y caption : Counts/monitor \n")
        fh.write("#Histogram :            F \n")
        fh.write(f"#Points    :         {len(theta)}   \n")
        for t, a, b in zip(theta, y, e):
            fh.write(f"{t:8.4f}     {a:14.8f}     {b:14.8f} \n")


def make_experiment(folder: pathlib.Path, noise: bool = False, seed: int = 3) -> dict:
    from toscana.corrections.pycorrect import atcors, mscat
    from toscana.isotopes import Element
    from toscana.physics.properties import getAtomicDensity
    from toscana.workflows.correction import material_properties

    folder = pathlib.Path(folder)
    # geometry (mm), as in the sample-description file
    sample_od, can_od, van_od, height = 6.0, 6.4, 6.0, 50.0
    beam_w, beam_h = 15.0, 50.0
    mass, density = 7.5, 8.908                        # g, g/cm3 (Ni)
    volume = math.pi * height * (sample_od / 2) ** 2 / 1000
    ni = Element("Ni")
    sa_rho = getAtomicDensity(mass / volume, ni.weight)
    sa_s, sa_a = ni.sig_sca, ni.sig_abs
    v_rho, v_s, v_a = material_properties("V")
    R1, R2, Rv, H, W = sample_od / 20, can_od / 20, van_od / 20, beam_h / 10, beam_w / 10
    tmu = lambda rho, s, a: rho * (s + a * LAMBDA / 1.798)  # noqa: E731
    sa_tmu, v_tmu = tmu(sa_rho, sa_s, sa_a), tmu(v_rho, v_s, v_a)

    theta = THETA
    q = 4 * math.pi / LAMBDA * np.sin(np.radians(theta) / 2)
    A = np.array([atcors(math.radians(t), R1, R2, 0, 0, W, sa_tmu, v_tmu, 0) for t in theta])
    Av = np.array([atcors(math.radians(t), Rv, 0, 0, 0, W, v_tmu, 0, 0)[0] for t in theta])
    fs = (R1 / R2) ** 2
    d_sa = mscat(R2, H, (1 - fs) * v_tmu + fs * sa_tmu, (1 - fs) * v_rho * v_s + fs * sa_rho * sa_s,
                 verbose=False)
    d_va = mscat(Rv, H, v_tmu, v_rho * v_s, verbose=False)
    b2 = abs(ni.bcoh) ** 2 / 100
    dcs = sa_s / 4 / math.pi + b2 * (s_true(q) - 1)
    flux = 1e5 * (1 - 0.15 * (theta / 135) ** 2)
    bkg = 5 + 200 * np.exp(-theta / 1.5)
    I_v = flux * v_rho * Rv ** 2 * v_s / 4 / math.pi * Av * (1 + d_va)
    I_c = flux * v_rho * (R2 ** 2 - R1 ** 2) * v_s / 4 / math.pi
    I_s = flux * sa_rho * R1 ** 2 * (dcs + d_sa * sa_s / 4 / math.pi)
    raw = {"vanadium": I_v + bkg, "can": A[:, 1] * I_c + bkg,
           "sample": A[:, 0] * I_s + A[:, 2] * I_c + bkg, "background": bkg}
    rng = np.random.default_rng(seed)
    for name, y in raw.items():
        e = np.sqrt(np.maximum(y, 1.0)) * 1e-3
        if noise:
            y = y + rng.standard_normal(len(y)) * e
        _write_adat(folder / f"{name}.adat", theta, y, e)

    (folder / "sample.par").write_text(f"""# synthetic Ni powder in a V can
<ins> D4 # instrument
<pro> 0-00-000 # proposal
<mpr> test # main proposer
<lco> test # local contact
<dte> 01/01/2026-02/01/2026 # dates
<cyc> 261 # cycle
<log> 1-1 # notebook-page
<env> none # environment
<zac> 0.0 # zero angle
<wle> {LAMBDA} # wavelength
<sli> {beam_w / 2} {-beam_w / 2} # slits
<fla> {beam_h / 2} {-beam_h / 2} # flags
<apr> 0.125 # angular precision
<asc> 1.5 135.0 0.125 # angular grid
<qsc> 0.3 23.5 0.02 # Q grid
<rsc> 0.0 20.0 0.01 # r grid
<Cmater> V # container
<Cshape> cylinder {sample_od} {can_od} 60.0 # can
<Nmater> V # normaliser
<Nshape> cylinder 0.0 {van_od} 50.0 # vanadium rod
<Amater> None # absorber
<Ashape> cylinder 0.0 6.0 60.0 # absorber
<Emater> None # environment
<Eshape> cylinder 40.0 40.2 100.0 # environment
<Sdescr> synthetic Ni powder # description
<StempK> 300.0 # temperature
<Sato01> Ni 1 # element
<Sdegcc> {density} # density
<Smassc> {mass} # mass
<Sshape> cylinder 0.0 {sample_od} {height} # sample
<Sfulln> 1.0 # fullness
<Stitle> synthetic_Ni # title
<CoFile> can.adat # container data
<AbFile> background.adat # absorber data
<NoFile> vanadium.adat # vanadium data
<EnFile> background.adat # environment data
<SaFile> sample.adat # sample data
""")
    return {"q": q, "s": s_true(q), "delta_sa": d_sa, "delta_va": d_va, "b2": b2,
            "self": sa_s / 4 / math.pi}
