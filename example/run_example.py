"""
Scriptable version of D4_corrections_example.ipynb.

    python run_example.py                       # MCAMS, full Paalman & Pings, .adat input
    python run_example.py --input qdat          # the same from the .qdat files
    python run_example.py --compare             # also: environment as background, pycorrect

The furnace is a body of the set-up (configurations E, CE, SCE, VE, ACE) and the
absorber measurement is one more configuration: factors A_E,E; A_E,CE, A_C,CE;
A_E,SCE, A_C,SCE, A_S,SCE; A_E,VE, A_V,VE; A_E,ACE, A_C,ACE, A_A,ACE.

Outputs in ``results/script`` (or ``--outdir``), on the Q and angle scales:
    Ni_powder_mcams_{sample,vanadium,absorber}_factors_{q,angle}.dat
    Ni_powder_mcams_intensities_{q,angle}.dat     I_s I_h I_c I_v K (+ errors), MS_C, B
    Ni_powder_mcams_dcs.dat, Ni_powder_mcams.soq  dσ/dΩ and S(Q)
    Ni_powder_{background,pycorrect}_*            the comparisons (--compare)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import time

import numpy as np

import toscana_path  # noqa: F401  (this ToScaNA as `toscana`)
from toscana.dataio.running_params import getRunningParams
from toscana.experiment.measurement import Measurement
from toscana.workflows.correction import run_correction, save_structure_factor, structure_factor

NAMES = {"sample": "I_SCE", "container": "I_CE", "environment": "I_E",
         "vanadium": "I_VE", "absorber": "I_ACE"}


def save_all(res, m, stem: str) -> None:
    """Factors and intensities on both scales, dσ/dΩ, S(Q)."""
    for scale in ("q", "angle"):
        for fs, tag in ((res.factors, "sample"), (res.vanadium_factors, "vanadium"),
                        (res.absorber_factors, "absorber")):
            if fs is not None:
                fs.save(f"{stem}_{tag}_factors_{scale}.dat", scale=scale, heading=[f"{tag} set-up"])
        if res.intensities:
            res.save_intensities(f"{stem}_intensities_{scale}.dat", scale=scale)
    np.savetxt(f"{stem}_dcs.dat", np.column_stack([res.theta, res.q, res.dcs, res.err]),
               fmt="%14.7g", header="2theta(deg) Q(1/A) dsigma/dOmega(barn/sr/atom) error")
    save_structure_factor(f"{stem}.soq", structure_factor(res, m, q_grid=m.qScale),
                          [f"{m.Title}, {res.method}"])


def main() -> None:
    ap = argparse.ArgumentParser(description="D4 corrections example (MCAMS, optional comparisons)")
    ap.add_argument("--input", choices=["adat", "qdat"], default="adat")
    ap.add_argument("--outdir", default=os.path.join("results", "script"))
    ap.add_argument("--compare", action="store_true",
                    help="also environment as background and pycorrect")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.outdir, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        m = Measurement(getRunningParams("Ni_powder.par"))
    files = {k: f"data/{n}.{args.input}" for k, n in NAMES.items()}
    truth = np.loadtxt(f"data/truth_{'angle' if args.input == 'adat' else 'q'}.dat")

    t0 = time.time()
    res = run_correction(m, files=files, environment="layer", vanadium_in_environment=True)
    print(f"MCAMS, full Paalman & Pings ({time.time() - t0:.0f} s)")
    for f in (res.factors, res.vanadium_factors, res.absorber_factors):
        print("   ", ", ".join(sorted(f.A)))
    z = (res.dcs - truth[:, 2]) / res.err
    print(f"    vs truth: mean {z.mean():+.3f} sigma, rms {z.std():.3f} sigma")
    save_all(res, m, os.path.join(args.outdir, f"{m.Title}_mcams"))

    if args.compare:
        bkg = run_correction(m, files=dict(files, empty=files["environment"]),
                             environment="background")
        pyc = run_correction(m, method="pycorrect", environment="furnace",
                             vanadium_in_environment=True,
                             files=dict(files, background=files["absorber"]),
                             workdir=os.path.join(args.outdir, "pycorrect_work"))
        for name, r in (("background", bkg), ("pycorrect", pyc)):
            save_all(r, m, os.path.join(args.outdir, f"{m.Title}_{name}"))
            ratio = r.dcs / truth[:, 2]
            print(f"{name:10s}: dσ/dΩ / true {ratio.min():.3f} to {ratio.max():.3f}")
    print("results in", args.outdir)


if __name__ == "__main__":
    main()
