"""
Command-line front end of `toscana.workflows.correction`.

Usage
-----
::

    python -m toscana.workflows.correct_cli SAMPLE.par [options]

    --method {mcams,pycorrect}     attenuation and multiple scattering (default mcams)
    --input {par,adat,qdat}        data files of the .par (default), or the same
                                   names with extension .adat / .qdat
    --scale {q,angle,both}         scale of the output tables (default both)
    --outdir DIR                   output folder (default: results)
    --environment {background,layer,none}
    --vanadium-in-environment      the vanadium was measured inside the environment
    --no-absorber                  ignore the absorber measurement
    --absorber-model {auto,configuration,transmission}
                                   absorber as a configuration of the P&P system
                                   (environment layer) or B = T I_empty + (1-T) I_abs
    --background FILE              pycorrect: subtracted from every measurement;
                                   mcams: empty-instrument measurement
    --placzek {series,diatomic,none}
    --vanadium-smoothing N
    --n-points N --ms-particles N --seed N   MCAMS statistics
    --mcams {delta,all}            method=pycorrect only: MCAMS factors in CORRECT
    --qgrid QMIN QMAX DQ           S(Q) on a regular grid (default: from <qsc>)

Outputs (``<title>`` from ``<Stitle>``, ``<m>`` the method), in ``--outdir``,
for each requested scale ``<s>`` (``q`` or ``angle``):

* ``<title>_<m>_sample_factors_<s>.dat``, ``..._vanadium_factors_<s>.dat``
  (and ``..._absorber_factors_<s>.dat`` with the absorber as a configuration):
  Paalman & Pings factors A_b,C, multiple scattering δ_C, transmissions
  T_C, n<L> and scattering powers (header);
* ``<title>_<m>_intensities_<s>.dat``: I_s, I_c, I_h, I_v (with errors),
  smoothed vanadium, multiple scattering MS_C and backgrounds B_C;
* ``<title>_<m>_dcs.dat``: 2θ, Q, dσ/dΩ (barn/sr/atom), error;
* ``<title>_<m>.soq``: Faber-Ziman S(Q) with errors.

From Python the same is ``main(["SAMPLE.par", "--scale", "q"])``, which
returns the `CorrectionResult`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from typing import Optional, Sequence

import numpy as np

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m toscana.workflows.correct_cli",
        description="Correct regrouped D4 data (attenuation, multiple scattering, "
                    "normalisation) and compute S(Q).")
    p.add_argument("par", help="sample-description file (.par)")
    p.add_argument("--method", choices=["mcams", "pycorrect"], default="mcams")
    p.add_argument("--input", choices=["par", "adat", "qdat"], default="par")
    p.add_argument("--scale", choices=["q", "angle", "both"], default="both")
    p.add_argument("--outdir", default="results")
    p.add_argument("--environment", choices=["background", "layer", "none"],
                   default="background")
    p.add_argument("--vanadium-in-environment", action="store_true")
    p.add_argument("--no-absorber", action="store_true")
    p.add_argument("--background", metavar="FILE",
                   help="pycorrect: file subtracted from every measurement (e.g. the absorber); "
                        "mcams: empty-instrument measurement")
    p.add_argument("--absorber-model", choices=["auto", "configuration", "transmission"],
                   default="auto")
    p.add_argument("--placzek", choices=["series", "diatomic", "none"], default="series")
    p.add_argument("--vanadium-smoothing", type=int, default=0)
    p.add_argument("--n-points", type=int, default=100_000)
    p.add_argument("--ms-particles", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mcams", choices=["delta", "all"], default=None)
    p.add_argument("--qgrid", type=float, nargs=3, metavar=("QMIN", "QMAX", "DQ"))
    p.add_argument("--quiet", action="store_true")
    return p


def _with_extension(path: str, ext: str) -> str:
    return os.path.splitext(path)[0] + "." + ext


def main(argv: Optional[Sequence[str]] = None):
    """Run the correction described by the command-line arguments `argv`.

    Returns the `CorrectionResult`; writes the files listed in the module
    docstring."""
    from toscana.dataio.running_params import getRunningParams
    from toscana.experiment.measurement import Measurement
    from toscana.workflows.correction import (run_correction, save_structure_factor,
                                              structure_factor)

    a = _parser().parse_args(argv)
    say = (lambda *x: None) if a.quiet else print
    with contextlib.redirect_stdout(io.StringIO()):
        m = Measurement(getRunningParams(a.par))
    files = {}
    if a.input != "par":
        for key, attr in (("sample", "File"), ("container", "conFile"), ("vanadium", "norFile"),
                          ("environment", "envFile"), ("absorber", "absFile")):
            name = getattr(m, attr)
            if str(name).strip():
                files[key] = _with_extension(name, a.input)
        if a.environment == "background":
            files["empty"] = files.get("environment", "")
    if a.no_absorber:
        files["absorber"] = ""
    if a.background:
        files["background" if a.method == "pycorrect" else "empty"] = a.background
    options = dict(method=a.method, files=files, environment=a.environment,
                   placzek=a.placzek, vanadium_smoothing=a.vanadium_smoothing)
    if a.method == "mcams":
        options.update(vanadium_in_environment=a.vanadium_in_environment,
                       absorber_model=a.absorber_model,
                       factor_options=dict(n_points=a.n_points, ms_particles=a.ms_particles,
                                           seed=a.seed))
    else:
        options.update(vanadium_in_environment=a.vanadium_in_environment,
                       mcams=a.mcams, output=f"{m.Title}_pycorrect_raw.dat",
                       workdir=os.path.join(a.outdir, "pycorrect_work"))
    say(f"{m.Title}: method={a.method}, input={a.input}, environment={a.environment}")
    res = run_correction(m, **options)
    for note in res.notes:
        say("  note:", note)

    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, f"{m.Title}_{res.method.replace('+', '_')}")
    scales = ["q", "angle"] if a.scale == "both" else [a.scale]
    written = []
    for s in scales:
        for fs, tag in ((res.factors, "sample"), (res.vanadium_factors, "vanadium"),
                        (res.absorber_factors, "absorber")):
            if fs is not None:
                name = f"{stem}_{tag}_factors_{s}.dat"
                fs.save(name, scale=s, heading=[f"{m.Title}: {tag} set-up"])
                written.append(name)
        if res.intensities:
            res.save_intensities(f"{stem}_intensities_{s}.dat", scale=s)
            written.append(f"{stem}_intensities_{s}.dat")
    with open(f"{stem}_dcs.dat", "w", encoding="utf-8") as fh:
        fh.write(f"# {m.Title}, method {res.method}, wavelength {res.wavelength} A\n")
        fh.write("# columns: 2theta(deg) Q(1/A) dsigma/dOmega(barn/sr/atom) error\n")
        np.savetxt(fh, np.column_stack([res.theta, res.q, res.dcs, res.err]), fmt="%14.7g")
    written.append(f"{stem}_dcs.dat")
    grid = a.qgrid
    if grid is None:
        grid = tuple(m.qScale)
    sq = structure_factor(res, m, q_grid=grid)
    save_structure_factor(f"{stem}.soq", sq, [f"{m.Title}, method {res.method}"])
    written.append(f"{stem}.soq")
    for name in written:
        say("  wrote", name)
    return res


if __name__ == "__main__":
    main()
