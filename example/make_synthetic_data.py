"""
Synthetic D4 experiment for the correction example.

Writes, in ``data/``, the five regrouped measurements as d4creg would
(``.adat``: 2θ, I, σ; ``.qdat``: Q, I, σ):

    I_SCE   Ni powder in the vanadium container, in the furnace
    I_CE    empty container in the furnace
    I_E     empty furnace
    I_VE    vanadium rod in the furnace
    I_ACE   10B powder (black absorber) in the container, in the furnace

plus the sample-description file ``Ni_powder.par`` (in this folder), the
CORRECT command file ``data/Ni_powder.com`` and the true values
``data/truth_angle.dat``, ``data/truth_q.dat``.

Physics of the synthetic data (what the correction must invert)
---------------------------------------------------------------
* Set-up: Ni powder (Ø 5 mm, 60 mm, packing fraction ≈ 0.5) in a vanadium
  can (Ø 5.0/5.2 mm), inside a furnace with a vanadium shield (Ø 40.0/40.2
  mm); vanadium rod Ø 6.08 mm in the same furnace; absorber: 10B powder
  (1.25 g/cm³) in the same can; beam 8 × 50 mm², λ = 0.5 Å.
* Bodies: s sample, c container, h furnace (environment), v vanadium,
  a absorber. Configurations measured: h (I_E), ch (I_CE), sch (I_SCE),
  vh (I_VE), ach (I_ACE). Paalman & Pings factors A_b,C, multiple
  scattering δ_C(2θ) and P_b = Σρσ_s<L> from MCAMS (toscana.corrections:
  default statistics, seed 1, the same calculation as in the notebook).
* Every measurement contains the room background B(2θ) (not attenuated by
  the set-up), the single scattering of each body b attenuated by A_b,C and
  the multiple scattering δ_C K Σ_b A_b,C P_b/4π:

      I_E   = B + A_h,h I_h                                   + MS_h
      I_CE  = B + A_h,ch I_h + A_c,ch I_c                     + MS_ch
      I_SCE = B + A_h,sch I_h + A_c,sch I_c + A_s,sch I_s     + MS_sch
      I_VE  = B + A_h,vh I_h + A_v,vh K (1 - P_v) P_v/4π      + MS_vh
      I_ACE = B + A_h,ach I_h + A_c,ach I_c + A_a,ach K P_a/4π + MS_ach

  K(2θ): count rate per unit isotropic cross section (flux × efficiency);
  I_s = K n<L>_s (dσ/dΩ - Σσ/4π P_s), P_v, P_s: Placzek corrections (series);
  I_c = K P_c/4π; I_h = K P_h/4π (1 + 0.1 cos 2θ).
* Sample: crystalline Ni (fcc, a = 3.52 Å), Gaussian Bragg peaks, Debye-
  Waller factor; S(Q) = e^{-u²Q²} S_Bragg(Q) + 1 - e^{-u²Q²}.
* Without noise the correction recovers the input to rounding error; with
  counting noise (default) within the error bars.

Usage: ``python make_synthetic_data.py [--no-noise]`` (about 3 minutes).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import math
import pathlib

import numpy as np

import toscana_path  # noqa: F401  (makes `toscana` importable)

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
LAMBDA = 0.5
THETA = np.round(np.arange(1.5, 135.0 + 1e-9, 0.125), 6)          # <asc>
Q_GRID = np.round(np.arange(0.3, 23.0 + 1e-9, 0.02), 6)           # <qsc>
NAMES = {"h": "I_E", "ch": "I_CE", "sch": "I_SCE", "vh": "I_VE", "ach": "I_ACE"}

PAR = f"""# ToScaNA sample-description file: synthetic Ni powder (correction example)
<ins> D4 # instrument
<pro> 0-00-0000 # proposal
<mpr> Example # main proposer
<lco> Example # local contact
<dte> 01/01/2026-02/01/2026 # dates
<cyc> 261 # cycle
<log> 1-1 # notebook-page
<env> Furnace # environment
<zac> 0.0 # zero angle (deg)
<wle> {LAMBDA} # wavelength (A)
<sli> 4.0 -4.0 # left right slits (mm): beam 8 mm wide
<fla> 25.0 -25.0 # top bottom flags (mm): beam 50 mm high
<apr> 0.125 # angular precision (deg)
<asc> 1.5 135.0 0.125 # angular grid (deg)
<qsc> 0.3 23.0 0.02 # Q grid (1/A)
<rsc> 0.0 20.0 0.01 # r grid (A)
<Cmater> V # container material
<Cshape> cylinder 5.0 5.2 60.0 # container: type id od height (mm)
<Nmater> V # normaliser
<Nshape> cylinder 0.0 6.08 60.0 # vanadium rod
<Amater> [10B]@1.25 # absorber: 10B powder (g/cm3) in the container
<Ashape> cylinder 0.0 5.0 60.0 # absorber
<Emater> V # environment
<Eshape> cylinder 40.0 40.2 100.0 # furnace shield
<Sdescr> Ni powder at room temperature # description
<StempK> 300.0 # temperature (K)
<Sato01> Ni 1 # element
<Sdegcc> 8.908 # bulk density (g/cm3)
<Smassc> 5.25 # mass in the container (g)
<Sshape> cylinder 0.0 5.0 60.0 # sample: type id od height (mm)
<Sfulln> 1.0 # fullness
<Stitle> Ni_powder # title
<CoFile> data/I_CE.adat # container in furnace
<AbFile> data/I_ACE.adat # absorber in container in furnace
<NoFile> data/I_VE.adat # vanadium in furnace
<EnFile> data/I_E.adat # empty furnace
<SaFile> data/I_SCE.adat # sample in container in furnace
"""


def s_true(q: np.ndarray, a: float = 3.52, u2: float = 0.0047) -> np.ndarray:
    """Powder S(Q) of fcc Ni: Bragg peaks + Debye-Waller + diffuse (TDS ~ 1)."""
    rho = 4 / a ** 3
    g = {}
    for h, k, l in itertools.product(range(-12, 13), repeat=3):
        if (h, k, l) == (0, 0, 0) or len({h % 2, k % 2, l % 2}) != 1:
            continue                                    # fcc: h, k, l unmixed
        gg = round(2 * math.pi / a * math.sqrt(h * h + k * k + l * l), 8)
        if gg < 25:
            g[gg] = g.get(gg, 0) + 1
    s_bragg = np.zeros_like(q)
    for gg, mult in g.items():
        sig = 0.035 + 0.0015 * gg                       # resolution (1/A)
        s_bragg += mult * np.exp(-0.5 * ((q - gg) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    s_bragg *= (2 * math.pi) ** 3 * rho / (4 * math.pi * q ** 2)
    dw = np.exp(-u2 * q ** 2)
    return dw * s_bragg + 1 - dw


def write_regrouped(path: pathlib.Path, x, y, e, scale: str) -> None:
    """Header as toscana.dataio.d4creg_outputs (saveDiffAngle / saveDiffQ)."""
    if scale == "angle":
        cap = f"2Theta (degrees) binned from {x[0]} to {x[-1]} by 0.125 ({len(x)} bins)"
    else:
        cap = f"Q (1/Å) binned from {x[0]:4.3f} to {x[-1]:4.3f} by 0.020 ({len(x)} bins)"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {path.name}\n#Block  1\n#========\n#\n#Instrument:D4\n")
        fh.write("#User      :Example    exp_0-00-0000/processed\n")
        fh.write("#Run number: ['000001-000010']\n#Spectrum  :            1\n")
        fh.write(f"#Title     :{path.name} (synthetic)\n#Run date  : 01-Jan-26\n")
        fh.write(f"#X caption : {cap}\n#Y caption : Counts/monitor \n")
        fh.write("#Histogram :            F \n")
        fh.write(f"#Points    :         {len(x)}   \n")
        for a, b, c in zip(x, y, e):
            fh.write(f"{a:9.4f}     {b:14.6f}     {c:12.6f} \n")


def forward(m, theta, factors, setups):
    """Noise-free measurements and true values on the angles `theta`.

    factors, setups: dicts with keys "sample", "vanadium", "absorber"."""
    from toscana.isotopes import Element
    from toscana.workflows.correction import _placzek

    four_pi = 4 * math.pi
    q = 4 * math.pi / LAMBDA * np.sin(np.radians(theta) / 2)
    ni = Element("Ni")
    s = s_true(q)
    p_s = _placzek(theta, LAMBDA, ni.weight, float(m.TempK), "series")
    p_v = _placzek(theta, LAMBDA, Element("V").weight, 300.0, "series")
    dcs = ni.sig_sca / four_pi + abs(ni.bcoh) ** 2 / 100 * (s - 1)
    k = 6.0e5 * (1 - 0.15 * (theta / 135) ** 2) * (1 + 0.02 * np.sin(theta / 9))
    room = 150 + 4000 * np.exp(-theta / 2.5)
    ss, vs, as_ = setups["sample"], setups["vanadium"], setups["absorber"]
    single = {"s": k * factors["sample"].atoms_per_area["s"] * (dcs - ni.sig_sca / four_pi * p_s),
              "c": k * ss.scattering_power("c") / four_pi,
              "h": k * ss.scattering_power("h") / four_pi * (1 + 0.1 * np.cos(np.radians(theta))),
              "v": k * vs.scattering_power("v") / four_pi * (1 - p_v),
              "a": k * as_.scattering_power("a") / four_pi}
    meas = {}
    for key, setup in (("sample", ss), ("vanadium", vs), ("absorber", as_)):
        fset = factors[key]
        for cfg in fset.configurations:
            iso = sum(fset.A[f"A_{b},{cfg}"] * setup.scattering_power(b) for b in cfg) / four_pi
            meas[NAMES[cfg]] = (room + sum(fset.A[f"A_{b},{cfg}"] * single[b] for b in cfg)
                                + fset.delta[f"delta_{cfg}"] * k * iso)
    truth = {"S": s, "dcs": dcs, "I_s": single["s"], "I_c": single["c"], "I_h": single["h"],
             "I_v": k * vs.scattering_power("v") / four_pi, "K": k, "B": room}
    return q, meas, truth


def main() -> None:
    from toscana.corrections.factors import compute_factors
    from toscana.corrections.setup import (absorber_setup_from_measurement,
                                           setups_from_measurement)
    from toscana.dataio.running_params import getRunningParams
    from toscana.experiment.measurement import Measurement
    from toscana.workflows.correction import build_state

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    (HERE / "Ni_powder.par").write_text(PAR, encoding="utf-8")
    # Measurement reads the five files: write placeholders first
    for name in ("I_SCE", "I_CE", "I_E", "I_VE", "I_ACE"):
        write_regrouped(DATA / f"{name}.adat", THETA, np.ones_like(THETA),
                        np.ones_like(THETA), "angle")
    import os

    os.chdir(HERE)
    with contextlib.redirect_stdout(io.StringIO()):
        m = Measurement(getRunningParams("Ni_powder.par"))
    ss, vs = setups_from_measurement(m, environment="layer", vanadium_in_environment=True)
    setups = {"sample": ss, "vanadium": vs,
              "absorber": absorber_setup_from_measurement(m, environment="layer")}
    rng = np.random.default_rng(2026)
    theta_q = 2 * np.degrees(np.arcsin(Q_GRID * LAMBDA / (4 * math.pi)))
    for scale, theta in (("angle", THETA), ("q", theta_q)):
        print(f"MCAMS factors on the {scale} grid ...", flush=True)
        # the same configurations as run_correction(environment="layer", ...)
        factors = {"sample": compute_factors(ss, theta),
                   "vanadium": compute_factors(vs, theta, configurations=["vh"]),
                   "absorber": compute_factors(setups["absorber"], theta,
                                               configurations=["ach"])}
        q, meas, truth = forward(m, theta, factors, setups)
        x = theta if scale == "angle" else Q_GRID
        ext = "adat" if scale == "angle" else "qdat"
        for name, y in meas.items():
            e = np.sqrt(y)                         # counting statistics
            if not args.no_noise:
                y = y + rng.standard_normal(len(y)) * e
            write_regrouped(DATA / f"{name}.{ext}", x, y, e, scale)
        cols = ["S", "dcs", "I_s", "I_c", "I_h", "I_v", "K", "B"]
        header = ("true values of the synthetic experiment (noise free)\n"
                  f"columns: {'2theta(deg)' if scale == 'angle' else 'Q(1/A)'} "
                  "S(Q) dsigma/dOmega(barn/sr/atom) I_s I_c I_h I_v K B\n"
                  "I_s: sample single scattering before the Placzek correction; I_v: vanadium "
                  "after it, = K P_v/4pi; B: room background")
        np.savetxt(DATA / f"truth_{scale}.dat",
                   np.column_stack([x] + [truth[c] for c in cols]), fmt="%14.8g", header=header)

    # CORRECT command file with the same input as run_correction(method="pycorrect")
    st = build_state(m, files={"sample": "I_SCE.adat", "container": "I_CE.adat",
                               "vanadium": "I_VE.adat", "environment": "I_E.adat",
                               "background": "I_ACE.adat"},
                     environment="furnace", vanadium_in_environment=True)
    sp = st.species[0]
    com = f"""! Synthetic D4 example: Ni powder in a V can, in a V furnace; vanadium in the furnace.
! CORRECT cannot treat the absorber measurement as a configuration: it is used here
! as the background of every measurement (the closest CORRECT can do).
! Cross sections and densities from ToScaNA's tables (same as build_state).
! Run from this folder:  python -m toscana.corrections.pycorrect Ni_powder.com
instrument d4
wavelength {st.wavelength}
beam {st.beam_height:.4f} {st.beam_width:.4f}
sample I_SCE.adat {st.ca_r1:.4f} /density={st.sa_rho:.6f} /temperature={st.sa_temp} /fullness={st.sa_fullness}
component 1.0 {sp.weight:.4f} {sp.scatt_cs:.4f} {sp.abs_cs:.4f}
container I_CE.adat {st.ca_r2:.4f} /density={st.ca_rho:.6f} /scattering={st.ca_scat_cs:.4f} /absorption={st.ca_abs_cs:.4f}
furnace I_E.adat {st.fu_r1:.4f} {st.fu_r2:.4f} /density={st.fu_rho:.6f} /scattering={st.fu_scat_cs:.4f} /absorption={st.fu_abs_cs:.4f}
vanadium I_VE.adat {st.va_r:.4f} /furnace /density={st.va_rho:.6f} /scattering={st.va_scat_cs:.4f} /absorption={st.va_abs_cs:.4f}
background I_ACE.adat
placzek series
spectrum 1
output Ni_powder_pycorrect.dat
execute /nolist
quit
"""
    (DATA / "Ni_powder.com").write_text(com, encoding="utf-8")
    print("written:", ", ".join(sorted(p.name for p in DATA.iterdir())))


if __name__ == "__main__":
    main()
