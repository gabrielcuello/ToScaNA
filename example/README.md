# D4 corrections example: MCAMS (default) and pycorrect

A complete, synthetic D4 experiment: **Ni powder in a vanadium container, inside a furnace**,
λ = 0.5 Å, beam 8 × 50 mm². Starting from the regrouped files (as written by `d4creg`), it
computes all the Paalman & Pings attenuation factors and the multiple scattering with MCAMS,
solves the Paalman & Pings equations for the sample (I_S), vanadium (I_V), container, furnace and
room background, on the Q and angle scales, and goes on to dσ/dΩ and S(Q). pycorrect (CORRECT
2.24) and the simpler "furnace as background" treatment are shown for comparison.

## Contents

| file | what it is |
|---|---|
| `data/I_E.adat`, `.qdat` | empty environment (furnace) — configuration `h` |
| `data/I_CE.adat`, `.qdat` | empty container in environment — `ch` |
| `data/I_SCE.adat`, `.qdat` | sample in container in environment — `sch` |
| `data/I_VE.adat`, `.qdat` | vanadium rod in environment — `vh` |
| `data/I_ACE.adat`, `.qdat` | ¹⁰B powder (absorber) in container in environment — `ach` |
| `data/truth_angle.dat`, `truth_q.dat` | true S(Q), dσ/dΩ, I_S, I_C, I_E, I_V, K, B (noise free) |
| `data/Ni_powder.com` | the closest CORRECT command file (pycorrect) |
| `Ni_powder.par` | sample-description file (geometry, materials, composition, files) |
| `D4_corrections_example.ipynb` | **the notebook**: the equations step by step, MCAMS, comparisons |
| `run_example.py` | the notebook as a script |
| `correct.py` | ToScaNA's correction command line (wrapper of `toscana.workflows.correct_cli`) |
| `run_pycorrect.py` | pycorrect's own command line (wrapper of `toscana.corrections.pycorrect`) |
| `make_synthetic_data.py` | regenerates `data/` and `Ni_powder.par` (about 2.5 min) |
| `toscana_path.py` | makes *this* ToScaNA importable as `toscana` (see below) |

Outputs go to `results/` (not kept in git).

## Physics in one page

Bodies: `s` sample, `c` container, `h` environment (furnace, the **E** of the file names),
`v` vanadium, `a` absorber. A configuration is the set of bodies in the beam. In the code
E is written `h`: A_S,SCE is `A_s,sch`, A_E,CE is `A_h,ch`, etc.

**Which attenuation factors.** In the Paalman & Pings formalism a factor A_b,C is needed for
every body b of every configuration C that was *measured*:

| measurement | factors |
|---|---|
| I_E | A_E,E |
| I_CE | A_E,CE, A_C,CE |
| I_SCE | A_E,SCE, A_C,SCE, A_S,SCE |
| I_VE | A_E,VE, A_V,VE |
| I_ACE | A_E,ACE, A_C,ACE, A_A,ACE |

Factors of configurations that were not measured (A_S,S: sample alone; A_S,SC, A_C,SC, A_C,C:
without the furnace) do not enter. A_S,SC, A_C,SC, A_C,C are the ones to use only when the
furnace is treated as a background (`environment="background"`), an approximation.

**Equations.** Each measurement contains the room background B (not attenuated by the set-up),
the single scattering I_b of each body attenuated by A_b,C, and the multiple scattering
MS_C = δ_C K Σ_b A_b,C P_b/4π (δ for isotropic scatterers, P_b = Σρσ_s⟨L⟩):

    I_E   = B + A_E,E I_E¹                                         + MS_E
    I_CE  = B + A_E,CE I_E¹ + A_C,CE I_C¹                          + MS_CE
    I_ACE = B + A_E,ACE I_E¹ + A_C,ACE I_C¹ + A_A,ACE K P_A/4π     + MS_ACE
    I_VE  = B + A_E,VE I_E¹ + A_V,VE K (1 - P_V^Placzek) P_V/4π    + MS_VE
    I_SCE = B + A_E,SCE I_E¹ + A_C,SCE I_C¹ + A_S,SCE I_S¹         + MS_SCE

K(2θ) is the count rate per unit isotropic cross section (flux × efficiency), fixed by the
vanadium. The first four equations are linear in **B, I_E¹, I_C¹, K** at each angle: the
absorber measurement is what determines the room background B. ToScaNA solves them (weighted
least squares if there are more measurements, e.g. an extra empty-instrument run), then

    I_S¹ = (I_SCE - B - A_E,SCE I_E¹ - A_C,SCE I_C¹ - MS_SCE) / A_S,SCE
    dσ/dΩ = I_S¹ / (K n⟨L⟩_S) + Σ c_i σ_i P_i^Placzek / 4π

Once B and K are known this is the familiar nested Paalman & Pings subtraction. Errors are
propagated to first order from all the measurements, with their correlations.

The synthetic data follow exactly this model (MCAMS factors, default statistics, seed 1) plus
counting noise: the correction recovers the truth within the error bars (rms ≈ 1σ, for dσ/dΩ
and for B). Without noise (`python make_synthetic_data.py --no-noise`) it recovers every unknown
to 10⁻⁷.

## 1. Notebook

```bash
cd example
jupyter lab D4_corrections_example.ipynb
```

Set `INPUT = "qdat"` in the first code cell to start from the Q-scale files. Section 3 builds and
solves the equations explicitly with the factors, so any step can be changed; section 4 checks
that `run_correction` gives the same; section 6 compares with the furnace treated as a
background and with pycorrect.

## 2. Command line

From the `example` folder:

```bash
# MCAMS, full Paalman & Pings; input: the .adat of the .par, or --input qdat; outputs on Q and angle
python correct.py Ni_powder.par --environment layer --vanadium-in-environment
python correct.py Ni_powder.par --environment layer --vanadium-in-environment --input qdat --scale q --outdir results/cli
# Monte Carlo statistics
python correct.py Ni_powder.par --environment layer --vanadium-in-environment --n-points 200000 --ms-particles 5000000 --seed 3
# the simpler treatment: furnace as background, B = T I_E + (1 - T) I_ACE
python correct.py Ni_powder.par --outdir results/cli_background
python correct.py --help

# pycorrect through the same interface: furnace as a body, absorber subtracted as background
python correct.py Ni_powder.par --method pycorrect --environment layer --vanadium-in-environment \
                  --background data/I_ACE.adat --outdir results/cli_pycorrect

# pycorrect with CORRECT's own command language (file names relative to the data folder)
cd data
python ../run_pycorrect.py Ni_powder.com          # writes Ni_powder_pycorrect.dat
```

When the `toscana` on your Python path *is* this ToScaNA (symbolic link and `PYTHONPATH`, see
the manual, "Making the package importable"), the wrappers are equivalent to
`python -m toscana.workflows.correct_cli ...` and `python -m toscana.corrections.pycorrect ...`.

Output files of `correct.py` (`<m>` = `mcams` or `pycorrect`, `<s>` = `q` or `angle`):

| file | columns |
|---|---|
| `Ni_powder_<m>_sample_factors_<s>.dat` | x, A_h,h, A_h,ch, A_c,ch, A_h,sch, A_c,sch, A_s,sch, δ_h, δ_ch, δ_sch (+ errors); header: T_C, n⟨L⟩, P_b |
| `Ni_powder_<m>_vanadium_factors_<s>.dat` | x, A_h,vh, A_v,vh, δ_vh (+ errors) |
| `Ni_powder_<m>_absorber_factors_<s>.dat` | x, A_h,ach, A_c,ach, A_a,ach, δ_ach (+ errors) |
| `Ni_powder_<m>_intensities_<s>.dat` | x, I_s, I_h, I_c (+ errors), I_v, I_v_err, I_v_smooth, K, K_err, MS_C, B, B_err |
| `Ni_powder_<m>_dcs.dat` | 2θ, Q, dσ/dΩ (barn/sr/atom), error |
| `Ni_powder_<m>.soq` | Q, S(Q), error on the `<qsc>` grid |

Factor files are read back with `toscana.corrections.factors.load_factors(file)`.

## 3. Scripts (Python)

```bash
python run_example.py                 # MCAMS from the .adat files -> results/script
python run_example.py --input qdat --compare
```

The essential lines, for your own scripts or notebooks:

```python
import toscana_path                      # only needed if `toscana` is another version
from toscana.dataio.running_params import getRunningParams
from toscana.experiment.measurement import Measurement
from toscana.workflows.correction import run_correction, structure_factor

m = Measurement(getRunningParams("Ni_powder.par"))
files = {"sample": "data/I_SCE.qdat", "container": "data/I_CE.qdat", "environment": "data/I_E.qdat",
         "vanadium": "data/I_VE.qdat", "absorber": "data/I_ACE.qdat"}

res = run_correction(m, files=files, environment="layer", vanadium_in_environment=True)
res.factors.A["A_s,sch"], res.factors.A["A_h,sch"], res.absorber_factors.A["A_h,ach"]
res.intensities["I_s"], res.intensities["I_v"], res.intensities["I_h"], res.backgrounds["B"]
res.save_factors("results/Ni", scale="q")                   # or scale="angle"
res.save_intensities("results/Ni_I_q.dat", scale="q")
sq = structure_factor(res, m, q_grid=m.qScale)
```

Lower level, the factors alone (any set-up of concentric cylinders):

```python
from toscana.corrections.setup import setups_from_measurement, absorber_setup_from_measurement
from toscana.corrections.factors import compute_factors
sample_setup, vanadium_setup = setups_from_measurement(m, environment="layer", vanadium_in_environment=True)
fs = compute_factors(sample_setup, res.theta)                 # h, ch, sch
fa = compute_factors(absorber_setup_from_measurement(m), res.theta, configurations=["ach"])
```

pycorrect with its command language from Python:

```python
import os
from toscana.corrections.pycorrect import Correct
os.chdir("data")
c = Correct(verbose=False, ask=False)
c.run_file("Ni_powder.com")
r = c.results[0]          # r["theta"], r["dcs"], r["assch"], r["ahsch"], r["delta_sa"], ...
```

MCAMS's own program (`toscana/corrections/mcams/run_simulation.py`) simulates one geometry whose
layers are edited in the script; it gives transmissions and scattering orders but does not do
the subtraction. Use it to explore a geometry; use `compute_factors`/`run_correction` for data.

## Notes

- **pycorrect** computes the furnace factors itself, but it cannot treat the absorber measurement
  as a configuration (the `.com` file subtracts it as a background) and its multiple scattering is
  one constant δ for the sample. Its furnace factors are 2–8 % lower than MCAMS's here; a
  brute-force integration for the thin furnace shell confirms the MCAMS values (A_E,E = 0.990,
  0.985, 0.986 at 20°, 60°, 120°). Sample, container and vanadium factors agree within 0.1 %.
- **Furnace as background** (`environment="background"`): dσ/dΩ is 0.94–1.03 of the truth here,
  mostly low at low Q.
- **Errors** include all the measurements and their correlations; without vanadium smoothing
  (default) also the vanadium, through K.
- **Another `toscana` in your environment**: `toscana_path.py` makes sure the example uses the
  ToScaNA it belongs to (the parent folder), whatever else is installed.
