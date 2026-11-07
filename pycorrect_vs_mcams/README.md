# pycorrect vs MCAMS — attenuation and multiple scattering

This folder compares the analytical corrections of **pycorrect** (CORRECT 2.24)
with the Monte Carlo code **MCAMS**. The comparison covers the Paalman & Pings
attenuation factors and the Blech & Averbach multiple-scattering correction.
Placzek and vanadium normalisation are left out, because MCAMS does not
simulate them.

## Layout

```
parent/
  MCAMS/        your MCAMS folder (beam.py, cylinder.py, ...)
  pycorrect/    the pycorrect folder (contains the pycorrect/ package)
  comparison/   this folder
```

If the two packages live elsewhere, set the environment variables
`MCAMS_DIR` and `PYCORRECT_DIR`.

## Files

| File | Purpose |
|---|---|
| `pyc_mcams_bridge.py` | Functions that compute with MCAMS the quantities pycorrect uses: Paalman & Pings factors, δ, δ₂, and all factors of a pycorrect run (`mcams_factors_like_pycorrect`). Unit conversion is included. |
| `compare_pycorrect_mcams.py` | The comparison. Part A: attenuation. Part B: multiple scattering. Part C: independent direct-sampling check. Writes `results/` and `figures/`. |
| `example_d4_with_mcams.py` | Runs the D4 example of the pycorrect manual with both programs. |
| `compare_vertical_extent.py` | Partially filled containers (MCAMS 2.2 layers with their own bottom and height): tests the fullness step of CORRECT for cylinders and flat cells. Writes `results/V_*.txt` and `figures/V_*.pdf`. |
| `report/main.tex` | LaTeX report (same style as the MCAMS and pycorrect manuals); `report/main.pdf` is the compiled version. |
| `results/`, `figures/` | Tables and figures of the last run. |

## Running

```
python compare_pycorrect_mcams.py           # about 4 minutes
python compare_pycorrect_mcams.py --quick   # about 20 s
python example_d4_with_mcams.py             # about 15 s
python compare_vertical_extent.py           # about 30 s (needs MCAMS 2.2)
cd report && latexmk -pdf main.tex
```

## Main conclusions

- **Wide beam:** the attenuation factors agree within the MC error
  (0.1–0.5 %).
- **Narrow beam:** with a beam narrower than the sample, CORRECT is about
  1 % too high. An independent calculation confirms the MCAMS value.
- **Furnace:** CORRECT interpolates the furnace factors with a parabola
  through 0°, 90° and 180°. It misses dips of A_h,sch by up to 9 % near 30°
  and 150°.
- **Double scattering:** the Blech & Averbach table of δ₂ is confirmed
  (≤ 1 % up to μR = 1, ≤ 2.5 % at μR = 2).
- **Higher orders:** the empirical series of CORRECT for them is far too low.
  - For the Ni example δ = 0.486 in CORRECT against 0.58–0.63 in MCAMS.
  - For a pure scatterer at μR = 1, δ = 0.81 against 1.31.
  - `d/(1−d)` is a much better approximation for μR ≲ 0.75.
- **What CORRECT also ignores:** δ depends on angle (±3.5 %) and on the sample
  height above and below the beam.
- **Partial fill (MCAMS 2.2):** the fullness step of CORRECT,
  `I_sc,full = (I_sc − (1−f) I_c)/f`, is exact for single scattering
  (cylinder and flat cell), wherever the powder sits in the beam, if the can
  covers the whole beam and the beam is uniform.
  - Can bottom inside the beam: −1.1 %; fixed by
    `I_sc,full = (I_sc − (f_c−f)/f_c · I_c)/f`, `I_c,full = I_c/f_c`.
  - Gaussian beam profile: −33 % with the geometric f; exact with the flux
    fraction.
  - CORRECT's δ does not depend on the fullness; the real one does.
- **PLATE bug:** the Fortran `PLATE` gives a wrong A_c,sc in reflection
  geometry (0.82 instead of 0.61 in the test cell) and doubles A_c,c at
  2θ = 0. Both are fixed in pycorrect (`faithful=False`, the default).
