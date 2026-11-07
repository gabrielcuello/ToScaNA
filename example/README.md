# D4 corrections example: MCAMS (default) and pycorrect

A complete, synthetic D4 experiment: **Ni powder in a vanadium container, inside a furnace**,
λ = 0.5 Å, beam 8 × 50 mm². Starting from the regrouped files (as written by `d4creg`), it
computes the attenuation factors and the multiple scattering with MCAMS, makes the Paalman &
Pings subtraction to get I_S (sample) and I_V (vanadium) on the Q and angle scales, and goes on
to dσ/dΩ and S(Q). pycorrect (CORRECT 2.24) does the same for comparison.

## Contents

| file | what it is |
|---|---|
| `data/I_SCE.adat`, `.qdat` | sample in container in environment (2θ or Q, I, σ) |
| `data/I_CE.adat`, `.qdat` | empty container in environment |
| `data/I_E.adat`, `.qdat` | empty environment |
| `data/I_VE.adat`, `.qdat` | vanadium rod in environment |
| `data/I_ACE.adat`, `.qdat` | ¹⁰B powder (absorber) in container in environment |
| `data/truth_angle.dat`, `truth_q.dat` | true S(Q), dσ/dΩ, I_S, I_C, I_V, K (noise free) |
| `data/Ni_powder.com` | the same correction in CORRECT's command language (pycorrect) |
| `Ni_powder.par` | sample-description file (geometry, materials, composition, files) |
| `D4_corrections_example.ipynb` | **the notebook**: step by step, MCAMS, optional pycorrect |
| `run_example.py` | the notebook as a script |
| `correct.py` | ToScaNA's correction command line (wrapper of `toscana.workflows.correct_cli`) |
| `run_pycorrect.py` | pycorrect's own command line (wrapper of `toscana.corrections.pycorrect`) |
| `make_synthetic_data.py` | regenerates `data/` and `Ni_powder.par` (about 20 s) |
| `toscana_path.py` | makes *this* ToScaNA importable as `toscana` (see below) |

Outputs go to `results/` (not kept in git).

## Physics in one page

Bodies: `s` sample, `c` container, `v` vanadium. Configurations (what is in the beam): `c`
(empty can), `sc` (sample in can), `v` (vanadium). The furnace is treated as **background**
(ToScaNA's default, `environment="background"`), and so is the instrument.

1. **Background, with the absorber**: B_C = T_C · I_E + (1 − T_C) · I_ACE.
   The black absorber removes the furnace scattering produced *after* the sample position; with
   the sample in the beam only the transmitted fraction T_C reaches it.
2. **Vanadium**: I_V = (I_VE − B_v) / [(1 + δ_v) A_v,v (1 − P_v)], with P_v the Placzek
   correction; K = I_V / (P_v^scat / 4π) is the count rate per unit isotropic cross section.
3. **Multiple scattering** (δ is defined for isotropic scatterers, as in CORRECT):
   MS_C = δ_C K Σ_b A_b,C P_b / 4π, with P_b = Σ ρ σ_s ⟨L⟩ (scattering power).
4. **Paalman & Pings**: I_C = (I_CE − B_c − MS_c) / A_c,c ;
   I_S = (I_SCE − B_sc − MS_sc − A_c,sc I_C) / A_s,sc.
5. **Normalisation**: dσ/dΩ = I_S / (K n⟨L⟩_s) + Σ c_i σ_i P_i / 4π; then the Faber-Ziman S(Q).

A_b,C (attenuation of the scattering from body b in configuration C), δ_C(2θ) and T_C come from
MCAMS (`toscana.corrections.factors.compute_factors`). For other set-ups (hollow sample, annular
can, furnace as an attenuating body with several shields), see the manual: `<LayNN>` lines and
`environment="layer"`.

The synthetic data were generated with exactly this model (MCAMS factors, default statistics,
seed 1) plus counting noise: the MCAMS correction recovers the truth within the error bars
(rms deviation ≈ 1 σ). Without noise (`python make_synthetic_data.py --no-noise`) it recovers it
to 10⁻⁷.

## 1. Notebook

```bash
cd example
jupyter lab D4_corrections_example.ipynb
```

Set `INPUT = "qdat"` in the first cell to start from the Q-scale files. Section 3 does the
subtraction explicitly with the factors, so any step can be changed; section 4 checks that
`run_correction` gives the same; section 6 (`RUN_PYCORRECT = True`) adds pycorrect.

## 2. Command line

From the `example` folder:

```bash
# MCAMS (default); input: the .adat of the .par, or --input qdat; outputs on both scales
python correct.py Ni_powder.par
python correct.py Ni_powder.par --input qdat --scale q --outdir results/cli
# statistics of MCAMS, absorber, environment ...
python correct.py Ni_powder.par --n-points 200000 --ms-particles 5000000 --seed 3
python correct.py Ni_powder.par --no-absorber
python correct.py --help

# pycorrect through the same interface (same outputs, CORRECT's factors)
python correct.py Ni_powder.par --method pycorrect --outdir results/cli_pycorrect
python correct.py Ni_powder.par --method pycorrect --mcams delta   # CORRECT + MCAMS multiple scattering

# pycorrect with CORRECT's own command language (file names relative to the data folder)
cd data
python ../run_pycorrect.py Ni_powder.com          # writes Ni_powder_pycorrect.dat
```

When the `toscana` on your Python path *is* this ToScaNA (symbolic link and `PYTHONPATH`, see
the manual, "Making the package importable"), the wrappers are equivalent to

```bash
python -m toscana.workflows.correct_cli Ni_powder.par --input qdat
cd data && python -m toscana.corrections.pycorrect Ni_powder.com
```

Output files of `correct.py` (`<m>` = `mcams` or `pycorrect`, `<s>` = `q` or `angle`):

| file | columns |
|---|---|
| `Ni_powder_<m>_sample_factors_<s>.dat` | x, A_s,sc, A_c,sc, A_c,c, δ_sc, δ_c and their errors; header: T_C, n⟨L⟩, P_b |
| `Ni_powder_<m>_vanadium_factors_<s>.dat` | x, A_v,v, δ_v (+ errors); header: T_v, n⟨L⟩_v, P_v |
| `Ni_powder_<m>_intensities_<s>.dat` | x, I_c, I_c_err, I_s, I_s_err, I_v, I_v_err, I_v_smooth, MS_c, MS_sc, B_c, B_sc, B_v (MCAMS) |
| `Ni_powder_<m>_dcs.dat` | 2θ, Q, dσ/dΩ (barn/sr/atom), error |
| `Ni_powder_<m>.soq` | Q, S(Q), error on the `<qsc>` grid |

Factor files are read back with `toscana.corrections.factors.load_factors(file)`.

## 3. Scripts (Python)

```bash
python run_example.py                 # MCAMS from the .adat files -> results/script
python run_example.py --input qdat --pycorrect
```

The essential lines, for your own scripts or notebooks:

```python
import toscana_path                      # only needed if `toscana` is another version
from toscana.dataio.running_params import getRunningParams
from toscana.experiment.measurement import Measurement
from toscana.workflows.correction import run_correction, structure_factor

m = Measurement(getRunningParams("Ni_powder.par"))
files = {"sample": "data/I_SCE.qdat", "container": "data/I_CE.qdat", "vanadium": "data/I_VE.qdat",
         "environment": "data/I_E.qdat", "empty": "data/I_E.qdat", "absorber": "data/I_ACE.qdat"}

res = run_correction(m, files=files)                         # MCAMS
res.factors.A["A_s,sc"], res.factors.delta["delta_sc"], res.factors.T["T_sc"]
res.intensities["I_s"], res.intensities["I_v"]               # on res.theta / res.q
res.save_factors("results/Ni", scale="q")                    # or scale="angle"
res.save_intensities("results/Ni_I_q.dat", scale="q")
sq = structure_factor(res, m, q_grid=m.qScale)

pyc = run_correction(m, method="pycorrect", files=dict(files, absorber=""))   # comparison
```

Lower level, the factors alone (any set-up of concentric cylinders):

```python
from toscana.corrections.setup import setups_from_measurement
from toscana.corrections.factors import compute_factors
sample_setup, vanadium_setup = setups_from_measurement(m)
fs = compute_factors(sample_setup, res.theta)                 # method="mcams"
fs_pc = compute_factors(sample_setup, res.theta, method="pycorrect")
```

pycorrect with its command language from Python:

```python
import os
from toscana.corrections.pycorrect import Correct
os.chdir("data")
c = Correct(verbose=False, ask=False)
c.run_file("Ni_powder.com")
r = c.results[0]          # r["theta"], r["dcs"], r["assch"], r["delta_sa"], ...
```

MCAMS's own program (`toscana/corrections/mcams/run_simulation.py`) simulates one geometry whose
layers are edited in the script; it gives transmissions and scattering orders but does not do
the subtraction. Use it to explore a geometry; use `compute_factors`/`run_correction` for data.

## Notes

- **pycorrect cannot use the absorber**: it subtracts the empty environment in full. It also
  needs a solid sample in contact with the container wall and a solid vanadium; for other
  set-ups it stops with a message suggesting MCAMS.
- **Comparison**: the attenuation factors of MCAMS and pycorrect agree within 0.1 %; dσ/dΩ
  differs by up to ~20 % at low Q, from the multiple scattering (CORRECT: constant δ of the sample
  only, with an empirical series for the higher orders; MCAMS: δ(2θ) of the whole
  configuration). The synthetic truth follows MCAMS by construction; see `docs/validation` for
  the comparison of both with brute-force Monte Carlo.
- **Errors** include all the measurements; without vanadium smoothing (default) also the
  vanadium, through K.
- **Another `toscana` in your environment**: `toscana_path.py` makes sure the example uses the
  ToScaNA it belongs to (the parent folder), whatever else is installed.
