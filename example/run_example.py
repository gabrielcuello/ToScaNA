"""
Scriptable version of D4_corrections_example.ipynb.

    python run_example.py                       # MCAMS, .adat input, outputs on Q and angle
    python run_example.py --input qdat          # the same from the .qdat files
    python run_example.py --pycorrect           # also pycorrect (CORRECT 2.24) for comparison

Outputs in ``results/script`` (or ``--outdir``):
    Ni_powder_mcams_sample_factors_{q,angle}.dat     A_s,sc  A_c,sc  A_c,c  delta_C (+ errors), T_C
    Ni_powder_mcams_vanadium_factors_{q,angle}.dat   A_v,v  delta_v
    Ni_powder_mcams_intensities_{q,angle}.dat        I_s I_c I_v (+ errors), MS_C, B_C
    Ni_powder_mcams_dcs.dat, Ni_powder_mcams.soq     dσ/dΩ and S(Q)
    Ni_powder_pycorrect_*                            the same with pycorrect (--pycorrect)
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
        for fs, tag in ((res.factors, "sample"), (res.vanadium_factors, "vanadium")):
            if fs is not None:
                fs.save(f"{stem}_{tag}_factors_{scale}.dat", scale=scale, heading=[f"{tag} set-up"])
        if res.intensities:
            res.save_intensities(f"{stem}_intensities_{scale}.dat", scale=scale)
    np.savetxt(f"{stem}_dcs.dat", np.column_stack([res.theta, res.q, res.dcs, res.err]),
               fmt="%14.7g", header="2theta(deg) Q(1/A) dsigma/dOmega(barn/sr/atom) error")
    save_structure_factor(f"{stem}.soq", structure_factor(res, m, q_grid=m.qScale),
                          [f"{m.Title}, {res.method}"])


def main() -> None:
    ap = argparse.ArgumentParser(description="D4 corrections example (MCAMS, optional pycorrect)")
    ap.add_argument("--input", choices=["adat", "qdat"], default="adat")
    ap.add_argument("--outdir", default=os.path.join("results", "script"))
    ap.add_argument("--pycorrect", action="store_true", help="also run pycorrect")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.outdir, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        m = Measurement(getRunningParams("Ni_powder.par"))
    files = {k: f"data/{n}.{args.input}" for k, n in NAMES.items()}
    files["empty"] = files["environment"]          # background: the empty environment

    t0 = time.time()
    res = run_correction(m, files=files)            # method="mcams" (default)
    print(f"MCAMS ({time.time() - t0:.0f} s): T = {res.factors.T} {res.vanadium_factors.T}")
    save_all(res, m, os.path.join(args.outdir, f"{m.Title}_mcams"))

    truth = np.loadtxt(f"data/truth_{'angle' if args.input == 'adat' else 'q'}.dat")
    z = (res.dcs - truth[:, 2]) / res.err
    print(f"  vs truth: mean {z.mean():+.3f} sigma, rms {z.std():.3f} sigma")

    if args.pycorrect:
        pyc = run_correction(m, method="pycorrect", files=dict(files, absorber=""),
                             workdir=os.path.join(args.outdir, "pycorrect_work"))
        save_all(pyc, m, os.path.join(args.outdir, f"{m.Title}_pycorrect"))
        ratio = pyc.dcs / res.dcs
        print(f"pycorrect / MCAMS dσ/dΩ: {ratio.min():.3f} to {ratio.max():.3f}")
    print("results in", args.outdir)


if __name__ == "__main__":
    main()
