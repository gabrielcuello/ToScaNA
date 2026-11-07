# pycorrect — Python translation of CORRECT 2.24

CORRECT (M.A. Howe, R.L. McGreevy, P. Zetterström; version 2.24, Jan 2000)
corrects neutron diffraction data for background, container and furnace
scattering. It applies **Paalman & Pings** attenuation factors, **Blech &
Averbach** multiple scattering and the **Placzek** inelasticity correction,
and normalises to vanadium. `pycorrect` reproduces it in plain Python (NumPy
and SciPy), keeping the same command language so that old `.com` files still
work.

```
pip install numpy scipy          # only dependencies
cd my_data_folder                # file names in .com files are relative to it
python -m pycorrect run.com      # or: python -m pycorrect  (interactive prompt)
python tests/test_pycorrect.py   # self-tests
```

Put the `pycorrect` folder on your `PYTHONPATH`, or run `pip install -e .`
from the top folder. `doublescattering.dat` and `xsect.dat` ship inside the
package. As in the Fortran, a copy in the current directory takes precedence.

## 1. Physics

| Correction | Reference | Fortran | Python |
|---|---|---|---|
| Attenuation, cylinder + can + furnace (+ narrow beam, + radial collimator) | Paalman & Pings, J. Appl. Phys. 33 (1962) 2635 | `ATCORS` (ndplib.f) | `ndplib.atcors` |
| Attenuation, flat plate | idem | `PLATE` (routines.f) | `routines.plate` |
| Multiple scattering, cylinder | Blech & Averbach, Phys. Rev. 137 (1965) A1113 | `MSCAT`, `MULSCS`, `H2EL` (ndplib.f) | `ndplib.mscat`, `mulscs`, `h2el` |
| Placzek, reactor | Yarnell et al. (series); Powles (diatomic) | `PLACZEK` | `correct_2axis.placzek` |
| Placzek, TOF | Howells / Powles | `PLAATOM`, `PLAMOL`, `LITLF`, `BIGF` | `correct_tof.*` |

Subtraction scheme (s = sample, c = container, h = heater/furnace):

    I_h = I_h,raw / A_h,h
    I_c = (I_ch − A_h,ch · I_h) / A_c,ch
    I_s = (I_sch − A_c,sch · I_c − A_h,sch · I_h) / A_s,sch
    I_v = I_v,raw / [A_v · (1 − P_v + δ_v)]                       (vanadium)
    dσ/dΩ = I_s / I_v · (n_v R_v² σ_v/4π)/(n_s R_s²) − δ_s σ_s/4π + P_s

- **P&P factors.** `atcors` integrates exp(−Σ μᵢ Lᵢ) over the cross-section
  in polar coordinates, with a 9 × 9 Gauss–Legendre grid. The furnace factors
  are computed at 0°, 90° and 180° and interpolated with a parabola. In each
  correction run, `absor` evaluates the factors at 10 angles and fits a
  4th-degree polynomial. All of this is exactly as in the Fortran.
- **Multiple scattering.** `mscat` interpolates the Blech–Averbach
  double/single ratio δ₂(μR, R/H) from `doublescattering.dat` and scales it by
  σ_s/σ_t. Higher orders are added as
  δ = d + 0.694 d² + 0.370 d³ + 0.161 d⁴ + 0.059 d⁵.
  Outside the table it computes δ₂ by direct 6-D Gauss integration (`mulscs`).

## 2. File-by-file correspondence

| Fortran file | Routine | Python module / object |
|---|---|---|
| correct_224.f | `PROGRAM CORRECT` (command loop) | `correct.main`, `Correct.run_file`, `Correct.interactive` |
| | `BLOCK DATA` (defaults) | `state.CorrectState` field defaults |
| | `OBEY_LINE` | `Correct.obey_line` |
| | `CORRECT_LIST` | `Correct.correct_list` |
| | `READ_ATOM_DATA` | `correct.read_atom_data` |
| | `FULLNAME`, `LENUSED` | `os.path.abspath`, `str.rstrip` |
| correct_224.inc | COMMON blocks | `state.CorrectState`, `Species`, `Spectrum`, `ndplib.Collimator` |
| parse_line.f | `PARSE_LINE` | `parse_line.parse_line` → `ParsedLine` |
| correct_2axis_224.f | `CORRECT_2AXIS` | `correct_2axis.correct_2axis` |
| | `ABSOR` | `correct_2axis.absor` |
| | `SMOOTH_VANADIUM` | `correct_2axis.smooth_vanadium` |
| | `SMOOTH_VANADIUM_SG`, `SMOOTH`, `QUADFIT2` | `smooth_vanadium_sg`, `_sg_smooth`, `_quadfit2` |
| | `PLACZEK` | `correct_2axis.placzek` |
| correct_tofdif_224.f | `CORRECT_TOF` | `correct_tof.correct_tof` |
| | `NORM`, `ABSORB`, `MULSCAT` | `norm_to_monitor`, `absorb`, `mulscat` |
| | `SPLINE_SMOOTH` | `correct_tof.spline_smooth` |
| | `PLAMOL`, `PLAATOM`, `LITLF`, `BIGF` | same names |
| ndplib.f | `ATCORS` + `ATCOR$GXY3`, `GXY3x`, `GXY3z` | `atcors`, `_AtcorsIntegrator.gxy3`, `.gxy3_beam` |
| | `ATCOR$F1`…`F17`, `ATCOR$G1`…`G19` | methods `F1`…`F17`, `G1`…`G19` (vectorised) |
| | `ATCOR$AA`, `BB`, `CC`, `ARCS`, `ARCC` | `_aa`, `_bb`, `_cc`, `_arcs`, `_arcc` |
| | `VIS`, `TRANS`, `ETA`, `AREA`, `BEAMWX`, `BEAMWZ` | `_vis`, `_trans`, (inline), `area`, `_beamw` |
| | `MSCAT`, `MULSCS`, `H2EL` | `mscat`, `mulscs`, `h2el` |
| routines.f | `PLATE` | `routines.plate` |
| | `POLYFIT`, `POLYFN` | `routines.polyfit`, `polyval_c` |
| fileio.f | `READ_FILE` | `fileio.read_file` → `Block` |
| | `OPEN_FILE`, `APPEND_FILE`, `CLOSE_FILE`, `WRITE_FILE` | `fileio.DataWriter`, `.append`, `.close`, `write_file` |
| | `GET_INPUT`, `PC_INQUIRE` | `fileio.get_input`, `pc_inquire` |
| harwell.f | `VB01S`, `MA10A`, `FM02AS`, `SA01A`, … | `numpy.linalg.lstsq` (inside `polyfit`) |
| | `VB06A` + `TG01B` | `harwell.lsq_cubic_spline` (`scipy.interpolate.LSQUnivariateSpline`) |
| | `VC03A`, `FA03A`, `*D` versions | unused by CORRECT, not translated |

`prep2.f` and its helpers (`rebin`, `polint`, `read_data`, `append_data`,
`i_c`, `wread`) belong to PREP2, a separate grouping and rebinning program.
They are not part of CORRECT, so they are not translated here.

## 3. Using it from Python (comparison with MC)

```python
from pycorrect import Correct, atcors, mscat
c = Correct(verbose=False, ask=False)
c.run_file("run.com")
r = c.results[0]          # one dict per spectrum
r["theta"], r["q"], r["dcs"], r["err"]                 # final result
r["assch"], r["acsch"], r["acch"], r["ahsch"], r["ahch"], r["ahh"]
r["delta_sa"], r["delta_va"], r["pc_sa"], r["calib_smooth"]
```

You can call `atcors` and `mscat` directly. `examples/pp_factors_table.py`
prints A_s,sch … A_h,h against 2θ, together with δ, for any geometry: the
numbers to set beside your MC trajectory results.
`examples/make_d4_example.py` builds a synthetic D4 data set (vanadium with
Bragg peaks, can, sample, background, Poisson noise) and a `.com` file to
correct it.

## 4. The `faithful` switch and the known bugs of CORRECT 2.24

The default is `faithful = False`, which gives the corrected physics. Use
`faithful on` in a `.com` file, or `state.faithful = True`, to reproduce the
original numbers.

| # | Where | CORRECT 2.24 behaviour (`faithful=True`) | Default (`faithful=False`) |
|---|---|---|---|
| 1 | `MSCAT` | `del=d1+(d1-d1)*…`: no interpolation in R/H (uses the grid value below) | bilinear interpolation in (μR, R/H) |
| 2 | 2-axis, container as calibrant | `MSCAT(r,h, smu, amu)` called with (σ_s, σ_a) in the place of (μ_t, μ_s) | correct (μ_t, μ_s) |
| 3 | 2-axis low-angle correction | `diff − c1 − c2·θ/nfit` (precedence); `/background` error taken from the furnace | `(diff − c1 − c2·θ)/nfit`; background errors |
| 4 | 2-axis vanadium with furnace/can | errors use ratio·e² instead of ratio²·e² | ratio²·e² |
| 5 | `SMOOTH_VANADIUM` | errors used as *weights* (large error → large weight) | weights 1/e² |
| 6 | `SMOOTH_VANADIUM_SG` | edge peaks: compares with the wrong side of the peak, so the start point never moves | skips edge peaks |
| 7 | 2-axis fullness | error not propagated | propagated |
| 8 | `placzek diatomic ω` | ω read from the 1st parameter → 0 | 2nd parameter |
| 9 | TOF wavelength cut | `errors = counts`, x not shifted | errors kept, x shifted |
| 10 | TOF vanadium in furnace | furnace error subtracted in quadrature | added |
| 11 | `PLATE` (flat plate) | reflection geometry: A_c,sc of the back window omits the sample on the way out (0.82 instead of 0.61 for 2 mm Ni in V windows at 110°); 2θ = 0 at normal incidence: A_c,c is the sum of the two windows, not the mean (×2) | `e^{(μ_c d + μ_s t)(b−a)}` for the back window; mean of the two windows |

Always fixed, because garbage cannot be reproduced:

- `write absorption` wrote the undefined array `av`; the vanadium factor is
  written instead.
- TOF vanadium-in-furnace used `ahsch` before it was computed; `ahvh` is used.
- `PLACZEK DIATOMIC` with ω = 0 gave 0/0; the limit β/tanh(β/2) → 2 is used.
- The PC version could not run the TOF branch: it asked for the GENIE format,
  which was not implemented. Here TEXT/DATA files are read.

## 5. Other small differences

- Numbers are in double precision (the Fortran used REAL\*4), so expect
  differences of about 1e-6 relative.
- The command parser keeps the case of file names and accepts Unix paths
  containing `/`. A `/word` is a qualifier only if it abbreviates one of the
  verb's qualifiers. Quote a path if one of its directories is named like a
  qualifier.
- `vanadium … /furnace /container` work without `=value`.
- `execute` does not end the program; a command file continues with its next
  line. The output file must not exist unless `state.overwrite = True`, as on
  the PC version.

## 6. Verification

- `atcors` against brute-force integration on a 1500² grid: A_s,sc within
  0.1%, A_c,sc and A_c,c within 0.3% (grid-limited) at 0–135°.
- With a beam narrower than the sample, the original single-panel quadrature
  is about 0.4% from the brute-force value. This is a property of the
  algorithm, reproduced as is, and worth knowing when comparing with MC.
- `plate` against the analytic flat-plate factors at normal incidence, and
  against the direct method of MCAMS 2.1 (agreement to 1e-4 at all angles
  with `faithful=False`).
- Round trip: raw data built from the forward model are corrected back to the
  input dσ/dΩ to better than 0.2% (2-axis) and 0.02% (TOF).

## NumPy/SciPy functions used (summary)

`numpy.polynomial.legendre.leggauss` (Gauss nodes and weights) ·
`numpy.linalg.lstsq` (least squares) · `numpy.vander` (polynomial design
matrix) · `numpy.polynomial.polynomial.polyval` (evaluate c₀ + c₁x + …) ·
`numpy.searchsorted` (index search) · `numpy.clip`, `maximum`, `where`,
`errstate` (safe element-wise maths) ·
`scipy.interpolate.LSQUnivariateSpline` (least-squares cubic spline,
replacing Harwell VB06A/TG01B). Each module's docstring explains the ones it
uses.
