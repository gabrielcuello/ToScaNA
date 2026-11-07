# d4numor — User manual

**D4 numor analysis without a graphical interface**
Version 1.0.0 — command-line and Jupyter companion of `D4NumorGUI.py` (v2.1-C)

---

## Contents

1. [What it is](#1-what-it-is)
2. [Installation](#2-installation)
3. [Selecting numors](#3-selecting-numors)
4. [Command-line usage](#4-command-line-usage)
5. [Python / Jupyter usage](#5-python--jupyter-usage)
6. [Variables](#6-variables)
7. [Weighted averages](#7-weighted-averages)
8. [Detector movement](#8-detector-movement)
9. [Preset calculator](#9-preset-calculator)
10. [Differences with the GUI](#10-differences-with-the-gui)
11. [Troubleshooting](#11-troubleshooting)
12. [The reader `nx_reader_D4.py` (version 2)](#12-the-reader-nx_reader_d4py-version-2)
13. [Quick reference](#13-quick-reference)

---

## 1. What it is

`d4numor.py` is a single Python file that does what *D4NumorGUI* does, without Qt or a display:

| GUI | Command line | Python / Jupyter |
|---|---|---|
| *Browse* + *Open Numors* | `-d DIR -n NUMORS` | `ns = d4.NumorSet.from_directory(DIR, NUMORS)` |
| Variable tree | `list` | `ns.list_variables()` |
| Weighted averages + table | `table VAR ...` | `ns.print_table([...])` |
| Plot | `plot VAR ...` | `ns.plot([...])` |
| *Normalize* | `--normalize` | `normalize=True` |
| *Binary* (first = Y, second = X) | `binary -y Y -x X` | `ns.plot_binary(y, x)` |
| Detector Movement Information | `summary`, `motion` | `ns.motion_summary()`, `ns.motion_table()` |
| *Export Plot* | `-o file.png` | `filename="file.png"` |
| *Export Table* | `export VAR ... -o file.csv` | `ns.export_csv([...], "file.csv")` |
| Scan angles + Preset + *Calculate* | `preset VALUE --angles ...` | `ns.preset(value, angles)` |
| *(new)* overview of a run | `summary` | `ns.summary()` |
| *(new)* single-numor details | `info NUMOR` | `d4.numor_info(path)` |
| *(new)* pandas table | – | `ns.to_dataframe()` |

The same file works in two ways:

* **as a program**: `python d4numor.py COMMAND ...` (figures are written to files, so it works over `ssh` without X11);
* **as a module**: `import d4numor as d4` in a script or a Jupyter notebook.

The data are read by your existing **`nx_reader_D4.py`**, exactly as the GUI does. The example notebook `D4numor_examples.ipynb` shows every feature.

---

## 2. Installation

### Files

Put these files in the same folder:

```
d4numor.py               the program / module
nx_reader_D4.py          the NeXus reader, version 2 (October 2026) — see §12
D4numor_examples.ipynb   example notebook (optional)
MANUAL.md                this manual (optional)
```

> **Use `nx_reader_D4.py` version 2.** The previous version cannot read the D4 files written with the current NeXus format (every numor fails with `index 10 is out of bounds for axis 0 with size 6`). Version 2 reads both the current and the old format, and `D4NumorGUI.py` works with it unchanged.

### Python packages

| Package | Needed for |
|---|---|
| `numpy` | everything |
| `matplotlib` | plots |
| `h5py` | reading the `.nxs` files (used by `nx_reader_D4.py`) |
| `pandas` | optional, only `NumorSet.to_dataframe()` |
| `jupyter` | optional, only for the notebook |

PySide6 / Qt is **not** needed. Python ≥ 3.8.

```bash
pip install numpy matplotlib h5py pandas     # if something is missing
```

### Where the reader is looked for

`nx_reader_D4.py` is searched, in order, in: the folder given with `--reader-dir` (CLI) or `reader_dir=` (Python), the folder of `d4numor.py`, the current folder, then the normal `PYTHONPATH`.

### Optional: a short command

```bash
chmod +x d4numor.py
alias d4n="python /path/to/d4numor.py"     # add to ~/.bashrc
d4n summary -d ../rawdata -n 499980-500029
```

---

## 3. Selecting numors

### Directory + numors (usual)

`-d DIR -n SPEC` on the command line, `NumorSet.from_directory(DIR, SPEC)` in Python.

| `SPEC` | Meaning |
|---|---|
| `499980` | one numor |
| `499980-500029` | inclusive range |
| `499980-500029:2` | every second numor of the range |
| `499980-499985,499990,500020-500022` | combination (commas, no spaces needed) |

In Python, `SPEC` can also be a list of integers or a `range`.

File names are built with the pattern `{:06d}.nxs` (`499980` → `499980.nxs`). Use `--pattern` / `pattern=` for other names, e.g. `--pattern "D4_{:06d}.nxs"`.

### Explicit files or wildcards

```bash
python d4numor.py summary -f '../rawdata/4999*.nxs'              # quote wildcards
python d4numor.py summary -f ../rawdata/499980.nxs ../rawdata/499987.nxs
```
```python
ns = d4.NumorSet.from_files("../rawdata/4999*.nxs")
```

### All files of a directory

`-d DIR` without `-n`, or `NumorSet.from_directory(DIR)`.

### What happens with problems

* Numors whose file does not exist: a warning lists them; they are skipped.
* Numors that exist but cannot be read: the reader prints `[ERROR] Failed reading ...`, a warning lists them and they are removed from the set (so they cannot break averages or the motion analysis).
* Files are always sorted by numor.

---

## 4. Command-line usage

General form:

```
python d4numor.py COMMAND [arguments] -d DIR -n NUMORS [options]
```

`python d4numor.py --help` lists the commands; `python d4numor.py COMMAND --help` gives the details of one command.

### Options common to all commands (except `info`)

| Option | Meaning |
|---|---|
| `-d DIR`, `--dir DIR` | raw-data directory |
| `-n SPEC`, `--numors SPEC` | numors (see §3) |
| `-f FILE ...`, `--files FILE ...` | explicit files / quoted wildcards |
| `--pattern P` | file-name pattern, default `{:06d}.nxs` |
| `--reader-dir DIR` | folder containing `nx_reader_D4.py` |
| `--extra-vars VAR ...` | read more variables than the GUI (§6.3) |
| `-q`, `--quiet` | do not print the loading messages |

Variable names are the internal names (`SampleTemp`, `Monitor`, `ttheta_value`, …), case-insensitive. A typing error gives a suggestion (`Unknown variable 'Monitr'. Did you mean: Monitor?`).

### `list` — variables available

```bash
python d4numor.py list                                  # all variables known to d4numor
python d4numor.py list -d ../rawdata -n 499980-500029   # variables actually read, by category
python d4numor.py list --all -d ../rawdata -n 499980    # also the missing ones, marked [ ]
```
Output (excerpt): `[x] SampleTemp  num  Sample temperature [K]  — Measured sample temperature`.
`num` = numeric (can be plotted), `text` = text, `arr` = one array per numor.

### `summary` — overview of a run

```bash
python d4numor.py summary -d /data/rawdata -n 499980-500029
```
```
================================================================
D4 numor summary  —  50 numors (499980 -> 500029)
================================================================
Directory   : /data/rawdata
Proposal ID : 6-02-685
Username    : magd2244
LC name     : CUELLO
Subtitle    : s2_CaND3_14p5MPM_220K_homog, s2_CaND3_14p5MPM_230K_homog, s2_CaND3_14p5MPM_240K_homog
Cycle       : 263
D4 Notebook : 0
First start : 25-Oct-26 13:11:38
Last end    : 25-Oct-26 15:00:33
----------------------------------------------------------------
Total counting time       :       3326.1 s  (0.92 h)
Total monitor             :     19000000
Total detector            :    136374354
Monitor rate [cts/s]      : mean    5712.33  [5688.15, 5743.78]
Detector rate [cts/s]     : mean   41000.47  [39604.68, 42706.48]
Sample temperature [K]    : mean      1.450  std 0.000  [1.450, 1.450]
Regulation temperature [K]: mean    232.000  std 7.559  [219.998, 240.002]
Set point temperature [K] : mean    232.000  std 7.559  [220.000, 240.000]
2θ range [deg]            : -1.003 -> 10.754
----------------------------------------------------------------
Detector Movement Information
  Angular range :     58.785 deg
  Movement time :   3199.000 s
  Mean speed    :    0.01838 deg/s
================================================================
```
When a text value differs between numors (e.g. the three subtitles), the distinct values are listed. Here the movement time includes the pauses for the temperature changes; use `motion` on one scan for the real detector speed (§8).

### `table` — weighted averages and table

```bash
python d4numor.py table Monitor TotalCount SampleTemp subtitle -d ../rawdata -n 499980-500029
python d4numor.py table SampleTemp -d ../rawdata -n 499980-500029 -o temp.csv   # also save CSV
python d4numor.py table start_time end_time --no-averages -d ../rawdata -n 499980-500029
```
| Option | Meaning |
|---|---|
| `-o FILE` | also write the table to a CSV file |
| `--no-averages` | do not print the weighted averages |

### `plot` — variables versus numor

```bash
python d4numor.py plot SampleTemp -d ../rawdata -n 499980-500029 -o temp.png
python d4numor.py plot Monitor TotalCount --normalize -d ../rawdata -n 499980-500029 -o counts.pdf
python d4numor.py plot SampleTemp Pcloche detrate --separate -d ../rawdata -n 499980-500029
```
| Option | Meaning |
|---|---|
| `-o FILE` | output image; the extension sets the format (`png`, `pdf`, `svg`, …). Default `d4_plot.png` |
| `--normalize` | divide each curve by its mean |
| `--separate` | one panel per variable instead of all on one |
| `--title TEXT` | figure title |
| `--print-table` | also print the table |
| `--show` | open a window (only if a display is available; otherwise ignored with a warning) |

Text variables are skipped with a warning.

### `binary` — Y versus X

```bash
python d4numor.py binary -y detrate -x SampleTemp -d ../rawdata -n 499980-500029 -o rate_vs_T.png
python d4numor.py binary -y detrate -x ttheta_value --annotate --csv rate_vs_2th.csv -d ../rawdata -n 499980-499989
```
Prints the table `Numor | X | Y`. Options: `-o FILE` (default `d4_binary.png`), `--csv FILE`, `--annotate` (numor next to each point), `--show`.

### `export` — CSV file

```bash
python d4numor.py export Monitor TotalCount SampleTemp -d ../rawdata -n 499980-500029 -o run.csv
python d4numor.py export --all -d ../rawdata -n 499980-500029 -o run_all.csv
python d4numor.py export --all --internal-names -d ../rawdata -n 499980-500029 -o run_all.csv
```
One line per numor, first column `Numor`. Column names are the labels (as in the GUI) or, with `--internal-names`, the internal names (better for scripts). Values keep full precision.

### `motion` — detector movement numor by numor

```bash
python d4numor.py motion -d ../rawdata -n 499980-499989
```
```
  N |  Numor |       2θ |      Δ2θ | Time [s] |     Speed
---------------------------------------------------------
  0 | 499980 |   -1.003 |    0.000 |      0.0 |    0.0000
  1 | 499981 |   -1.003 |    0.000 |      1.0 |    0.0000
  2 | 499982 |   -0.494 |    0.509 |     13.0 |    0.0392
  3 | 499983 |    0.254 |    0.748 |     16.0 |    0.0467
...
```
See §8.

### `preset` — preset calculator

```bash
python d4numor.py preset 30 -d ../rawdata -n 499980-500029
python d4numor.py preset 100000 --angles -0.5 0 4.5 5 6 7.5 9 10 10.75 -d ../rawdata -n 499980-500029
python d4numor.py preset 30 60 120 10000 100000 -d ../rawdata -n 499980-500029     # table
python d4numor.py preset 30 --speed motion -d ../rawdata -n 499980-500029
python d4numor.py preset 30 --mon-rate 5700 --speed 0.05 -d ../rawdata -n 499980-500029
```
```
Scan angles   : -0.5, -0.5, 0, 4.5, 5, 6, 7.5, 9, 10, 10.75 (10 points)
Monitor rate  : 5715.56 cts/s
Angular speed : 0.03869 deg/s
Input         : 30 (total time [min])
Preset result : 862617 monitor counts per point
Moving time   : 4.85 min
Moving/Total  : 16.15 %
(Counting time of the loaded numors: 11.1 min)
```
| Option | Meaning |
|---|---|
| `VALUE ...` | < 1000: total time in minutes; ≥ 1000: monitor counts per point. Several values → a table |
| `--angles A ...` | scan angles in degrees. Default: the GUI ones `-0.5 -0.5 0 4.5 5 6 7.5 9 10 10.75` |
| `--mon-rate R` | monitor rate (counts/s). Default: median of the loaded numors |
| `--speed S` | `gui` (default), `motion`, or a number in deg/s (§9) |

The loaded numors are only used to measure the monitor rate and the detector speed.

### `info` — one numor in detail

```bash
python d4numor.py info 499983 -d ../rawdata
python d4numor.py info ../rawdata/499983.nxs
```
Prints the output of `nx_reader_D4.show_single_numor`: proposal, title, user, local contact, sample, start/end, notebook, reactor cycle and power, incident energy and wavelength, requested and read 2θ, counting time, counts and rates, temperature, belljar pressure, beam stops, the nine detector sums.

### Exit codes

`0` success, `1` no command given, `2` error (message on standard error). Scripts can test it:
```bash
python d4numor.py summary -d ../rawdata -n 499980-500029 -q > run.txt || echo "failed"
```

---

## 5. Python / Jupyter usage

```python
import sys; sys.path.insert(0, "/path/to/folder/with/d4numor")   # if needed
import d4numor as d4

ns = d4.NumorSet.from_directory("../rawdata", "499980-500029")
ns.summary()
ns.print_table(["Monitor", "SampleTemp"])
ns.plot(["Monitor", "TotalCount"], normalize=True)
```

In Jupyter, use `%matplotlib inline` (or `%matplotlib widget` for zoomable figures). Plots appear automatically. In a plain script, use `filename=` to save, or `show=True` if you have a display. Add `;` at the end of a notebook line to avoid printing the returned value twice.

### 5.1 Creating a `NumorSet`

```python
d4.NumorSet.from_directory(directory, numors=None, pattern="{:06d}.nxs", **options)
d4.NumorSet.from_files(files_or_globs, **options)
d4.NumorSet(list_of_paths, **options)
```
Options:

| Option | Default | Meaning |
|---|---|---|
| `variables` | the GUI list (`d4.DEFAULT_VARIABLES`) | variables requested from the reader |
| `extra_variables` | `None` | more variables, added to the list |
| `reader` | auto | an already imported `nx_reader_D4` module |
| `reader_dir` | auto | folder of `nx_reader_D4.py` |
| `verbose` | `True` | print the loading message |

### 5.2 Attributes

| Attribute | Content |
|---|---|
| `ns.numors` | numor numbers (numpy array) |
| `ns.filepaths` | files read |
| `ns.data` | dict `name → numpy array` (one value per numor) |
| `ns.variables` | names of all variables, sorted by category |
| `ns.motion` | raw result of `analyse_angular_motion` |
| `ns.unreadable` | numors that could not be read |
| `ns.reader` | the `nx_reader_D4` module |
| `len(ns)` | number of numors |

### 5.3 Methods

**Variables**

| Method | Returns / does |
|---|---|
| `list_variables(show_unavailable=False, print_out=True)` | prints the tree; returns the text |
| `get(var)` | numpy array of a variable |
| `aligned(var)` | `(numors, values)` with matching lengths |
| `resolve(name)` | internal name from any case / from a label |
| `is_numeric(var)` | `True` if it can be plotted |

**Statistics and tables**

| Method | Returns / does |
|---|---|
| `summary(print_out=True)` | overview (§4 `summary`); returns the text |
| `table(vars, averages=True)` | text table (string) |
| `print_table(vars, averages=True)` | prints it |
| `weighted_average(var, weight="Monitor")` | one number |
| `weighted_averages(vars)` | `{var: {"monitor", "detector", "mean"}}` |
| `stats(var)` | `{"n", "mean", "std", "min", "max", "monitor_weighted", "detector_weighted"}` |
| `to_dataframe(vars=None, labels=False)` | pandas DataFrame indexed by numor |

**Plots** (all return the matplotlib figure and axes, so the figure can be modified)

| Method | Main options |
|---|---|
| `plot(vars, ...)` | `normalize`, `separate`, `ax`, `filename`, `show`, `title`, `style="o-"`, `figsize` |
| `plot_binary(y, x, ...)` | `annotate`, `ax`, `filename`, `show`, `style`, `figsize` |
| `binary_data(y, x)` | `(numors, x_values, y_values)` matched numor by numor |
| `binary_table(y, x)` | text table |

Use `ax=` to draw into your own axes (several sets on the same graph, subplot layouts…).

**Exports**

| Method | Does |
|---|---|
| `export_csv(vars, path, labels=True)` | CSV, one line per numor (`vars=None` → all) |
| `export_binary_csv(y, x, path)` | CSV `Numor, X, Y` |

**Detector movement and preset**

| Method | Returns / does |
|---|---|
| `motion_summary()` | text: range, time, mean speed |
| `motion_table()` | text: line per numor (§8) |
| `rates()` | the monitor rates and speeds available (§9) |
| `preset(value, scan_angles=None, mon_rate=None, angular_speed=None, print_out=True)` | dict (§9) |
| `preset_table(values, scan_angles=None, **same options)` | text table |
| `info(numor)` | details of one numor of the set |

### 5.4 Module functions and constants

| Name | Use |
|---|---|
| `d4.numor_info(path)` | details of any `.nxs` file |
| `d4.parse_numor_list("499980-499985,499990")` | list of integers |
| `d4.build_file_list(directory, numors, files, pattern)` | list of paths |
| `d4.get_reader(reader_dir=None)` | imports `nx_reader_D4` |
| `d4.label(var)`, `d4.category(var)` | display name, category |
| `d4.VARIABLE_MAP` | labels, categories and tooltips (same as the GUI) |
| `d4.DEFAULT_VARIABLES` | variables read by default |
| `d4.DEFAULT_SCAN_ANGLES` | default angles of the preset calculator |
| `d4.main(["summary", "-d", ...])` | run the command line from Python |
| `d4.D4Error` | exception raised for user errors |

### 5.5 Short recipes

```python
# Compare two scans on one graph
a = d4.NumorSet.from_directory(RAW, "499980-499989", verbose=False)
b = d4.NumorSet.from_directory(RAW, "499990-499999", verbose=False)
fig, ax = plt.subplots()
a.plot_binary("detrate", "ttheta_value", ax=ax)
b.plot_binary("detrate", "ttheta_value", ax=ax, style="s--")
ax.legend(["scan 1", "scan 2"])

# Statistics per sample
df = ns.to_dataframe(["SampleTemp", "Monitor", "subtitle"])
df.groupby("subtitle").agg(["mean", "std", "count"])

# Loop over many runs and save one figure each
for first in range(499980, 500030, 10):
    s = d4.NumorSet.from_directory(RAW, f"{first}-{first+9}", verbose=False)
    s.plot("SampleTemp", filename=f"T_{first}.png")
    plt.close("all")
```

---

## 6. Variables

### 6.1 Read by default (same list as the GUI)

| Category | Internal name | Label |
|---|---|---|
| Axes | `ttheta_value` | 2θ [deg] |
| | `omega_value` | ω (sample) [deg] |
| | `A1_value` | A1 (mono) [deg] |
| | `BSC_value`, `BSD_value` | beam stops belljar / detector [mm] |
| Counting | `Monitor` | monitor counts |
| | `TotalCount` | total detector counts |
| | `Time` | acquisition time [s] |
| | `SumDet1` … `SumDet9` | counts of each detector block |
| Environment | `SampleTemp` | sample temperature [K] |
| | `Pcloche` | belljar pressure (not in the current files: NaN) |
| | `RegTemp`, `SetPTemp` | regulation and set-point temperature [K] (*) |
| | `pressure` | sample pressure (*) |
| Metadata | `username`, `LCname`, `proposal`, `subtitle` | text |
| | `start_time`, `end_time` | text, `dd-Mmm-yy hh:mm:ss` |
| | `cycle`, `nbD4` | data cycle, D4 notebook |

(*) Read only with `nx_reader_D4` version 2 (list `d4.V2_VARIABLES`).

**Which temperature?** In the current files `SampleTemp` comes from the scanned variable *SampleTemp* (= `entry0/sample/temperature`). In the numors 499980–500029 it is 1.45 K everywhere, while `RegTemp` follows the 220/230/240 K steps: the sample-temperature channel was apparently not connected. Check with `table SetPTemp RegTemp SampleTemp` before using one of them.

**Detector sums.** In the old format `SumDet1`…`SumDet9` were in the scanned variables. In the current format they are not stored; the reader computes them from the 576 detector cells, 64 cells per detector (cells 0–63 → detector 1, …). Their sum equals the total detector count.

### 6.2 Computed

| Name | Definition |
|---|---|
| `monrate` | `Monitor / Time` [counts/s] |
| `detrate` | `TotalCount / Time` [counts/s] |
| `delta_theta` | 2θ(i) − 2θ(i−1) [deg] |
| `moving_time` | start(i) − end(i−1) [s] |
| `angular_speed` | \|Δ2θ\| / moving time [deg/s] (0 when the time is 0) |

The last three come from `nx_reader_D4.analyse_angular_motion` (§8).

### 6.3 More variables

`read_nexus` creates many more attributes than the GUI shows. Any of them can be read with `--extra-vars` / `extra_variables=`:

* beam and monochromator: `EI_value`, `EI_target`, `wavelength`, `DM_value`, `A2_value`, `GM_value`, `R1_value`, `R2_value`, `T1_value`, `T2_value`, … (in the current files Ei = 0, so `wavelength` is NaN)
* targets and offsets: `ttheta_target`, `ttheta_offset`, `omega_target`, `z_value`, …
* sample environment (v2): `RegTemp2`, `SetPTemp2`, `SetPPressure`, `use_temp`
* sample changer and masks (v2): `SCload_value`, `SCrot_value`, `masked2theta_value`
* slits: `VolGams_value`, `VolLohen_value`, …
* reactor: `reactor`, `power`, `days`
* counters: `time`, `duration`, `detsum`, `monsum`, `monrate_file`, `detrate_file`, `preset`, `mode`, …
* scan: `ttheta` (2θ from the scan table), `nb_step`, `actual_step`, `controller_list`, …
* `detdata`: the counts of all detection cells, one array per numor (`np.vstack(ns.get("detdata"))` gives a numors × cells matrix).

Any dataset can also be read by its path, e.g. `--extra-vars entry0/sample/regulation_temperature2`.

```bash
python d4numor.py table ttheta_target ttheta_value --extra-vars ttheta_target -d ../rawdata -n 499980-500029
```

To give them a label and a category, add entries to `VARIABLE_MAP` (in `d4numor.py`, or at run time):
```python
d4.VARIABLE_MAP["EI_value"] = {"label": "Incident energy [meV]", "category": "Beam", "tooltip": "Ei"}
```
Keep the `VARIABLE_MAP` of the GUI and of `d4numor.py` identical if you want both to show the same names.

---

## 7. Weighted averages

For a numeric variable *v* and the numors *i*:

* **Monitor-weighted**: Σ vᵢ·Monitorᵢ / Σ Monitorᵢ
* **Detector-weighted**: Σ vᵢ·TotalCountᵢ / Σ TotalCountᵢ

Numors where the value or the weight is NaN, or the weight is negative, are ignored. `ns.weighted_average(var, weight="Time")` uses any other variable as weight.

For a temperature or an angle these are the effective values of the measurement (the numors that counted more weigh more). For counts they have little meaning, but are shown for completeness, as in the GUI.

---

## 8. Detector movement

`nx_reader_D4.analyse_angular_motion` is called with the 2θ values and the start/end times of the numors (in the loaded order). For each numor *i* > 0:

* Δ2θᵢ = 2θᵢ − 2θᵢ₋₁
* moving timeᵢ = startᵢ − endᵢ₋₁ (time between the end of the previous numor and the start of this one)
* speedᵢ = |Δ2θᵢ| / moving timeᵢ

Totals: angular range = Σ|Δ2θ|, movement time = Σ moving time, mean speed = range / time.

The dead time between numors is not only movement (file writing, temperature changes…), so the speed is a lower bound. Load **consecutive numors of one scan** to get a meaningful speed: if the set contains a pause (e.g. a temperature change), the pause counts as moving time.

---

## 9. Preset calculator

It answers two questions for a planned 2θ scan, using `nx_reader_D4.get_preset`:

* **I have N minutes; what monitor preset per point?** → give the time (`value < 1000`).
* **I want M monitor counts per point; how long will it take?** → give the counts (`value ≥ 1000`).

With *n* scan points, monitor rate *R* (counts/s), detector speed *s* (deg/s):

* moving time *t*ₘ = (last angle − first angle) / *s* (`get_preset` uses the sum of the angle differences)
* time input *T* (min): monitor per point = (60·*T* − *t*ₘ) · *R* / *n*
* counts input *M*: total time (s) = *n* · *M* / *R* + *t*ₘ

**Moving/Total** is the fraction of the total time spent moving the detector (the "loss").

### Which rate and speed?

The loaded numors provide them. `ns.rates()` shows the options:

| Key | Definition | Used |
|---|---|---|
| `mon_rate_median` | median of `monrate` | default (as the GUI) |
| `mon_rate_global` | Σ Monitor / Σ Time | `mon_rate=ns.rates()["mon_rate_global"]` |
| `speed_gui` | mean of `angular_speed`, zeros included | default (as the GUI) |
| `speed_motion` | angular range / movement time | `angular_speed="motion"` / `--speed motion` |

`speed_gui` includes the 0 of the first numor and of repeated angles, so it is lower than `speed_motion` and the moving time is overestimated. The default keeps the GUI behaviour so both give the same numbers; use `motion`, or your own value, for planning.

### Result dictionary (Python)

| Key | Content |
|---|---|
| `preset` | monitor per point (time input) or total time in **seconds** (counts input) |
| `preset_units` | text describing the unit |
| `moving_time_min` | moving time [min] |
| `moving_ratio` | fraction of the total time spent moving (0–1) |
| `mon_rate`, `angular_speed` | values used |
| `scan_angles`, `input`, `input_kind` | what was asked |
| `total_scan_time_min` | counting time of the loaded numors |
| `raw` | what `get_preset` returned |

---

## 10. Differences with the GUI

Same reader, same variables, same formulas. Differences:

* **Export Table** — with several variables the GUI wrote one line per variable and numor; here it is one line per numor.
* **Weighted averages** ignore NaN values (the GUI returned NaN when one numor had no value).
* **Binary mode** matches X and Y numor by numor (the GUI truncated both to the shortest length).
* **Unreadable numors** are removed with a warning. In the reader, a file that fails is dropped from the numor list but NaN is still added to every variable, so in the GUI the values after it were attached to the wrong numors and the motion analysis failed.
* **Preset with a counts input**: the result is shown as a total time in s / min / h (the GUI printed "(units)"), and the moving fraction is also computed.
* **Mixed variable lengths**: a variable with one value fewer than the numors is attributed to numors 2…N instead of failing.
* Numors are taken from the file names (digits before `.nxs`), not from the list returned by the reader (see §12.5, points 2 and 5).
* `analyse_angular_motion` is silenced (it prints its totals even with `verbose=False`); the same numbers are in `summary`.
* **New**: `summary`, `motion`, `info`, preset tables, `--speed`/`--mon-rate`, extra variables, DataFrame export, case-insensitive names with suggestions.

---

## 11. Troubleshooting

| Message | Cause and fix |
|---|---|
| `Failed reading ...: index 10 is out of bounds for axis 0 with size 6` for every numor | old `nx_reader_D4.py`: replace it by version 2 (§12) |
| `[WARN] Variable 'Pcloche' not found` | normal with the current files (no belljar pressure stored) |
| `Cannot import 'nx_reader_D4'` | put `nx_reader_D4.py` next to `d4numor.py` or use `--reader-dir`. If the message mentions `h5py`: `pip install h5py` |
| `No numor file selected` | wrong directory, numors or file pattern. Check with `ls DIR` and `--pattern` |
| `Warning: N numor(s) not found` | these files do not exist (yet); the others are used |
| `[ERROR] Failed reading ...` + `could not be read and are ignored` | corrupt or incomplete file (e.g. numor still running); the rest is used |
| `[WARN] Variable 'X' not found` | printed by the reader when an attribute does not exist; the variable is NaN |
| `Unknown variable 'X'. Did you mean ...` | typing error; `list` shows the names |
| `Variable 'X' was not returned by the reader` | the variable exists in `VARIABLE_MAP` but not in the files; add it with `--extra-vars` if `read_nexus` provides it |
| `Angular speed not available` | the detector did not move in the loaded numors; load a 2θ scan or pass `--speed VALUE` |
| `Detector movement information not available` | start/end times or 2θ missing, or the times cannot be parsed |
| Figures do not appear (script) | normal on a machine without display: use `-o file.png`. In Python: `filename=` or `show=True` |
| Figures do not appear (Jupyter) | add `%matplotlib inline` at the top |

---

## 12. The reader `nx_reader_D4.py` (version 2)

### 12.1 Why a new version

The D4 NeXus files changed. Checked on numors 499980–500029 (25 Oct 2026):

| Item | Old format | Current format |
|---|---|---|
| `data_scan/scanned_variables/data` | 14 fixed rows: TotalCount, SumDet1–9, Time, Monitor, SampleTemp, 2θ | only the scanned quantities, here 6 rows: omega, TotalCount, Time, Monitor, SampleTemp, 2θ; described by `variables_names/{label,name,property,unit}` |
| `monitor/data` | present | **absent** (use `monitor/monsum`) |
| `sample/Pcloche` | present | **absent** |
| `sample/` | – | `regulation_temperature(2)`, `setpoint_temperature(2)`, `pressure`, `setpoint_pressure`, `use_temp` |
| `detector_data/data` | shape (1, 576, 1) | shape (1, 1, 576) |
| new axes | – | `SCloadaxis`, `SCrotaxis`, `masked2theta_s` |

The previous reader used fixed row numbers: it failed on every current file (`index 10 is out of bounds`), so the GUI and `d4numor` got only NaN. Even the rows that exist (1–5) would have been read as SumDet1–5 although they now hold other quantities.

### 12.2 How version 2 reads a file

* **Scanned variables by name.** The rows of the scan table are matched by their label, name or property (case-insensitive), whatever their order or number. Old files without names use the old row numbers.
* **Several sources per quantity**, first one present wins. Examples: Monitor ← scanned *Monitor* → `monitor/monsum` → `monitor/data`; TotalCount ← scanned *TotalCount* → `Detector/detsum`; SampleTemp ← scanned *SampleTemp* → `sample/temperature`; Time ← scanned *Time* → `entry0/time`.
* **Missing data never stop the reading**: the attribute is NaN (or '') and is listed in `nx.missing`. Only a file that cannot be opened raises an error.
* **Detector sums** `SumDet1…9`: from the scan table if present, otherwise computed from the cells (9 × 64). On the current files the block boundaries are visible every 64 cells and the nine sums add up exactly to the total count.
* `nx.sources` tells where each attribute came from (`scan:Monitor`, a dataset path, or `computed`); `nx.format` is `scan-table` or `legacy`.

The functions, their names, arguments and results are those of the previous version (`read_nexus`, `extract_variable`, `extract_multiple_variables`, `analyse_angular_motion`/`analyze_angular_motion`, `get_preset`, `elapsed_time`, `mean_nonzero`, `print_scan_summary`, `compute_angular_speed`, `analyze_scan`/`analyse_scan`, `show_single_numor`, `print_numor_summary`, `getReactorCycle`, `NeXusData`), so `D4NumorGUI.py` and `d4numor.py` use it without change.

### 12.3 Checking files (when the format changes again)

```bash
python nx_reader_D4.py ../rawdata/499980.nxs             # where every quantity comes from
python nx_reader_D4.py ../rawdata/499980.nxs --datasets  # + list of all datasets
python nx_reader_D4.py ../rawdata/49998*.nxs --show      # + show_single_numor for each file
```
```python
import nx_reader_D4 as R
R.check_file("../rawdata/499980.nxs")      # sources and missing quantities
R.list_datasets("../rawdata/499980.nxs")   # every dataset: type, shape, first values
nx = R.read_nexus("../rawdata/499980.nxs"); nx.scanned, nx.sources, nx.missing
```
Example of `check_file` output (excerpt):
```
Format : scan-table
Scanned variables: omega=142.59, totalcount=2.68635e+06, time=66.4758, monitor=380000, sampletemp=1.45, ..., 2theta=-1.003
Variable                        Value  Source
Time                          66.4758  scan:Time
Monitor                        380000  scan:Monitor
RegTemp                       219.999  entry0/sample/regulation_temperature
Pcloche                           nan  MISSING
SumDet1                        284341  computed
```
If a quantity moves to another place in a future format, add the new path to its list in the `FIELDS` dictionary at the top of `nx_reader_D4.py` (and new motors to `MOTORS`).

### 12.4 Things to know about the current files

* **SampleTemp is constant (1.45 K)** in the 50 numors checked; `RegTemp`/`SetPTemp` hold the real temperatures (§6.1).
* **Ei = 0** in `Monochromator/ei`, so `wavelength` is NaN. If you need it, set it yourself.
* **Belljar pressure** (`Pcloche`) is no longer stored; `pressure` is `entry0/sample/pressure` (0 in these numors).
* **Reactor cycles**: 2026 cycles 199, 200, 201 were added from the ILL web page (ill.eu, reactor cycles). Their data-cycle numbers 261, 262, 263 follow the usual convention and should be checked. Edit `REACTOR_CYCLES` for later cycles.
* **D4 notebook**: the table `D4_NOTEBOOKS` stops at notebook 44 (29-Sep-25); later numors get `00`. Add the new notebooks there.
* Monitor and detector rates computed by the reader (counts / time) agree with `monitor/monrate` and `Detector/detrate` stored in the file (available as `monrate_file`, `detrate_file`).

### 12.5 Bugs of the previous version fixed in version 2

1. `getReactorCycle` searched the D4 notebook for a fixed date (20-Sep-23), so `nbD4` was always 42.
2. `extract_multiple_variables` dropped the numor of an unreadable file but appended NaN to the variables (numors and values out of step); arrays such as `detdata` mixed with NaN raised an error.
3. `analyse_angular_motion` printed its totals even with `verbose=False`.
4. `read_nexus`: `controller_list` read `actual_step`; `RCU331H_offset` missing (`RCU331V_offset` twice); infinite wavelength when Ei = 0; the HDF5 file stayed open.
5. The numor was built from all digits of the file name (`D4_499980.nxs` → 4499980); now the digits before the extension are used.
6. The module called `logging.basicConfig` at import, changing the logging of the program that imported it.

### 12.6 Validation

On the 50 current numors, for each file: run number, Time, Monitor, TotalCount, 2θ, omega, SampleTemp and RegTemp equal the raw datasets; the 576 cells and the nine SumDet add up to `detsum`; computed rates equal the stored rates; reactor cycle 201. On files in the old format, all attributes are identical to those of the previous reader except the corrected `controller_list`.

---

## 13. Quick reference

```bash
D4="python /path/to/d4numor.py"
RUN="-d ../rawdata -n 499980-500029"

$D4 list $RUN                                   # variables
$D4 summary $RUN                                # overview
$D4 table Monitor TotalCount SampleTemp $RUN    # averages + table
$D4 plot Monitor TotalCount --normalize $RUN -o counts.png
$D4 plot SampleTemp Pcloche --separate $RUN -o env.png
$D4 binary -y detrate -x SampleTemp $RUN -o rate_T.png --csv rate_T.csv
$D4 export --all $RUN -o run.csv
$D4 motion $RUN
$D4 preset 30 60 120 100000 $RUN --speed motion
$D4 info 499983 -d ../rawdata
```

```python
import d4numor as d4
ns = d4.NumorSet.from_directory("../rawdata", "499980-500029")
ns.list_variables(); ns.summary()
ns.print_table(["Monitor", "SampleTemp"])
ns.plot(["Monitor", "TotalCount"], normalize=True)
ns.plot_binary("detrate", "SampleTemp")
print(ns.motion_table())
ns.preset(30); print(ns.preset_table([30, 60, 100000]))
ns.export_csv(None, "run.csv"); df = ns.to_dataframe()
```

---
*d4numor.py and nx_reader_D4.py v2 are based on D4NumorGUI.py and nx_reader_D4.py by Gabriel Cuello (ILL).*
