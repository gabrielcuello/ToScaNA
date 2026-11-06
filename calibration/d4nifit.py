"""
Ni powder calibration of the D4 wavelength λ and zero angle 2θ0 (d4nifit).

Before a D4 experiment is reduced, the incident wavelength λ and the zero-angle
offset 2θ0 of the detector are calibrated with the diffractogram of a Ni powder
standard (fcc, lattice parameter ``NI_LATTICE_PARAMETER`` = 3.52 Å by
default). The regrouped Ni diffractogram (2θ in degrees, e.g. ``ni.adat``
produced by the d4creg regrouping) is fitted with
``toscana.models.ni.niPeaks10``: a quadratic background plus ten Gaussian
peaks at the first ten fcc reflections, whose positions depend only on
(λ/a, 2θ0). The resulting λ and 2θ0 are then used for all the other files of
the experiment (conversion to Q, ``toscana.physics.conversions.ang2q``).

Public functions
----------------
setting_model_d4nifit(nifile, zac_ini=0.0, lattice=NI_LATTICE_PARAMETER)
    Read and normalise the data, estimate λ from the (111) peak, build the
    35 initial parameters and the plotting/fitting windows, plot the start.
fitting_model_d4nifit(nickel, limits, initial, lattice=NI_LATTICE_PARAMETER)
    Weighted lmfit fit (weights 1/σ) of ``niPeaks10`` inside ``limits``.
showing_results_d4nifit(result, nickel, axes, limits, xy_res, xy_ref, xy_fit, planes)
    Print and plot the results; save the figure as ``<basename>.png``.

Typical notebook sequence (setting -> fitting -> showing results)
-----------------------------------------------------------------
Usage::

    from toscana.calibration.d4nifit import (
        setting_model_d4nifit, fitting_model_d4nifit, showing_results_d4nifit)

    # 1. Setting: read ni.adat, normalise, initial λ from (111), plot the start.
    (axes, limits, xy_res, xy_ref, xy_fit,
     planes, initial, nickel) = setting_model_d4nifit("ni.adat", zac_ini=0.0)

    # Optional: adjust before fitting, e.g. the fitting window or a start value
    # limits[1] = 45.0          # upper 2θ limit of the fit (degrees)
    # initial[4] = 0.05         # starting 2θ0 (degrees)

    # 2. Fitting: weighted least squares, lattice parameter held fixed.
    result = fitting_model_d4nifit(nickel, limits, initial)

    # 3. Showing results: printed table + figure saved as <basename>.png.
    showing_results_d4nifit(result, nickel, axes, limits,
                            xy_res, xy_ref, xy_fit, planes)

    # Another Ni lattice parameter (the fitted λ scales linearly with it):
    # setting_model_d4nifit("ni.adat", lattice=3.5238)
    # fitting_model_d4nifit(nickel, limits, initial, lattice=3.5238)

The 35-element ``initial`` vector
---------------------------------
Built by `setting_model_d4nifit` and consumed by `fitting_model_d4nifit`
(and by ``niPeaks10(x, *initial)``), in the positional order of
``niPeaks10``. fp = 0.5·(ymax - 1), ymax being the normalised maximum
intensity (an estimate of the area of the first peak).

====== ============ =========================== =========== =========
Index  Parameter    Meaning (unit)              Initial     In the fit
====== ============ =========================== =========== =========
0      I0           constant background         1           free
1      slope        linear bkg (per deg)        0.0007·fp   free
2      quad         quadratic bkg (per deg²)    0.0         fixed
3      wavelength   λ (Å)                       from (111)  free
4      twotheta0    2θ0 (deg)                   zac_ini     free
5-14   A0..A9       peak areas (int.·deg)       see below   free
15-24  G0..G9       asymmetries (1/deg)         0.0         fixed
25-34  S0..S9       Gaussian σ (deg)            see below   free
====== ============ =========================== =========== =========

Initial areas A0..A9 (reflections (111), (200), (220), (311), (222), (400),
(331), (420), (422), (333)/(511)): fp × 1.00, 0.56, 0.54, 0.76, 0.24, 0.12,
0.42, 0.40, 0.32, 0.37 (typical relative Ni intensities). Initial widths
S0..S9: 0.2 (×6), 0.3, 0.3, 0.4, 0.4 degrees. The lattice parameter is not part
of ``initial``: it is passed as the ``lattice`` argument and held fixed.

Plot geometry lists (all in data coordinates: x in degrees 2θ, y in normalised
intensity; v = flat_valley_x, the integer 2θ midway between peaks 2 and 3)
---------------------------------------------------------------------------
axes = [0.5·v, 93·λ_ini, -0.1·ymax, 1.1·ymax]
    Plot window (xmin, xmax, ymin, ymax) passed to ``plt.axis``.
limits = [0.5·v, 93·λ_ini, 0, 1.1·ymax]
    Fitting rectangle (xmin, xmax, ymin, ymax): only points with
    xmin <= 2θ <= xmax and ymin <= y <= ymax are fitted. 93·λ (46.5° for
    λ = 0.5 Å) includes the ten reflections ((333)/(511) is at 43.3° for
    λ = 0.5 Å).
xy_res = [0.81·v, 0.90·ymax]
    Position of the text box with λ and 2θ0.
xy_ref = [1.60·v, 0.40·ymax]
    Position of the text box listing the ten reflections and their 2θ.
xy_fit = [1.85·v, 0.75·ymax]
    Position of the text box with the fit statistics (final plot only).

Physics and fit choices (audit 2026-11)
---------------------------------------
- P7: the lattice parameter is ``NI_LATTICE_PARAMETER`` = 3.52 Å by default
  (was 3.52024 Å hard-coded) and user-configurable via ``lattice``. It is a
  fixed lmfit parameter: the peak positions determine only λ/a, so λ and a
  are fully correlated and the fitted λ scales linearly with a.
- P11: the fit is weighted by 1/σ (the legacy fit ignored the errors); the
  errors are normalised with the intensities. Without valid errors (2-column
  files store σ = -1) the fit is unweighted and a warning is printed. lmfit
  keeps ``scale_covar=True``: the reported uncertainties are multiplied by
  sqrt(reduced χ²), so they also account for model misfit.
- C10: A2, A3, A4 start from ``initial[7..9]`` (legacy used ``initial[6..8]``).

Side effects
------------
All three functions draw matplotlib figures; `setting_model_d4nifit` also
calls ``DataXYE.plot`` twice (each ends with ``plt.show()``);
`showing_results_d4nifit` writes ``<basename>.png``. All print to stdout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toscana.data.xy import DataXYE
from toscana.numerics.fitting import fittingRange
from toscana.models.ni import niPeaks10
from toscana.physics.crystallography import NI_LATTICE_PARAMETER, reflections_fcc
from toscana.utils.dates import getDate

if TYPE_CHECKING:
    import lmfit as lm  # type: ignore[import-untyped]


def setting_model_d4nifit(  # noqa: N802 (legacy API)
    nifile: str, zac_ini: float = 0.0, lattice: float = NI_LATTICE_PARAMETER
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[str],
    list[float],
    DataXYE,
]:
    """
    Prepare initial parameters and plotting limits for Ni powder calibration.

    `lattice` is the Ni lattice parameter in Å (default 3.52 Å).

    Steps, as implemented:

    1. Read `nifile` into a ``DataXYE`` (2 or 3 columns x, y[, e]; lines
       starting with "#" or "!" are header; 2-column files get e = -1). Its
       constructor finds the peaks with 5° <= 2θ <= 40° whose height is more
       than 3.5 times the minimum in that range (``peaks_x``). The raw data
       are plotted (``nickel.plot()``, which calls ``plt.show()``).
    2. Initial wavelength from the first peak, assumed to be (111)
       (d_111 = a/√3): λ_ini = 2 (a/√3) sin(2θ_peak0 / 2). It is printed as
       "Automatically calculated initial wavelength = x.xxx Å". The zero
       offset is not used in this estimate.
    3. Normalisation: v = flat_valley_x = int((peaks_x[1] + peaks_x[2]) / 2)
       (truncated to an integer number of degrees), i.e. the valley between
       the second ((200)) and third ((220)) peaks; the reference intensity is
       y at the first point with x > v. ``nickel.y`` and ``nickel.ymax`` are
       divided by it, and so are the positive errors ``nickel.e`` (negative
       values, the "no error" flag, are kept). The normalised data are
       plotted again. Other summary attributes of ``nickel`` (``ymin``,
       ``yave``, ``emax``, ``peaks_y``, ...) are not rescaled.
    4. Plot window ``axes``, fitting rectangle ``limits`` and text positions
       ``xy_res``, ``xy_ref``, ``xy_fit`` (see the module docstring).
    5. The 35 initial parameters ``initial`` (index by index in the module
       docstring): I0 = 1, slope = 0.0007·fp, quad = 0, λ_ini, `zac_ini`,
       areas fp × (1.00, 0.56, 0.54, 0.76, 0.24, 0.12, 0.42, 0.40, 0.32,
       0.37), asymmetries 0, widths (0.2 ×6, 0.3, 0.3, 0.4, 0.4) degrees,
       with fp = 0.5·(ymax - 1) and ymax the normalised maximum.
    6. New figure (9 × 6 in) with the initial model ``niPeaks10(x, *initial,
       lattice=lattice)`` (red), the data (blue) and the residuals
       data - model (green), a box with λ_ini and `zac_ini` at ``xy_res``, a
       box with the ten planes and their 2θ (1 decimal) at ``xy_ref``, title
       "Nickel powder diffractogram: Starting point". The figure is not shown
       or saved here (in a notebook it is displayed at the end of the cell).

    Parameters
    ----------
    nifile : str
        Path of the regrouped Ni diffractogram (2θ in degrees, intensity and
        optionally error), e.g. ``"ni.adat"``. Its basename (path without
        extension) is later used for the saved figure.
    zac_ini : float, optional
        Initial zero-angle offset 2θ0 in degrees, stored in ``initial[4]``.
        Default 0.0.
    lattice : float, optional
        Ni lattice parameter a in Å, used for λ_ini, the reflection positions
        and the initial model. Default ``NI_LATTICE_PARAMETER`` (3.52 Å). Pass
        the same value to `fitting_model_d4nifit`.

    Returns
    -------
    axes : list of float
        [xmin, xmax, ymin, ymax] of the plots: [0.5·v, 93·λ_ini, -0.1·ymax,
        1.1·ymax] (degrees, normalised intensity).
    limits : list of float
        Fitting rectangle [xmin, xmax, ymin, ymax]: [0.5·v, 93·λ_ini, 0,
        1.1·ymax].
    xy_res : list of float
        [x, y] of the λ / 2θ0 text box: [0.81·v, 0.90·ymax].
    xy_ref : list of float
        [x, y] of the reflection-list text box: [1.60·v, 0.40·ymax].
    xy_fit : list of float
        [x, y] of the fit-statistics text box: [1.85·v, 0.75·ymax].
    planes : list of str
        Labels of the ten reflections: "(111)", "(200)", "(220)", "(311)",
        "(222)", "(400)", "(331)", "(420)", "(422)", "(333)". The tenth
        position also contains (511) (same h²+k²+l² = 27).
    initial : list of float
        The 35 initial parameter values (order of ``niPeaks10``).
    nickel : DataXYE
        The data object, with normalised ``y``, ``ymax`` and ``e``.

    Raises
    ------
    FileNotFoundError
        If `nifile` does not exist.
    ValueError
        If a data line of `nifile` does not have 2 or 3 columns (from
        ``DataXYE``).
    IndexError
        If fewer than 3 peaks are found between 5° and 40° (``peaks_x[2]``).
    UnboundLocalError
        If no data point lies above ``flat_valley_x`` (``index`` never set).

    Notes
    -----
    Bug fix vs legacy (audit 2026-11, P11): the errors are divided by the
    same valley intensity as the data, as needed by the weighted fit.

    See Also
    --------
    fitting_model_d4nifit, showing_results_d4nifit
    toscana.models.ni.niPeaks10
    """
    import matplotlib.pyplot as plt

    nickel = DataXYE(nifile)
    nickel.plot()

    # Initial λ from the first peak, assumed (111): λ = 2 d111 sin θ, d111 = a/√3
    wl_ini = 2 * lattice / np.sqrt(3.0) * np.sin(nickel.peaks_x[0] * np.pi / 360.0)
    print(f"Automatically calculated initial wavelength = {wl_ini:.3f} Å")

    # Normalisation point: first x above the (integer) midpoint of peaks 2 and 3
    flat_valley_x = int((nickel.peaks_x[1] + nickel.peaks_x[2]) / 2)
    for i in range(len(nickel.x)):
        if nickel.x[i] > flat_valley_x:
            index = i
            break
    flat_valley_y = nickel.y[index]

    nickel.y = nickel.y / flat_valley_y
    nickel.ymax = nickel.ymax / float(flat_valley_y)
    # Bug fix vs legacy: rescale the errors with the data (needed for the
    # weighted fit). Negative values are the "no error" flag of 2-column files.
    nickel.e = np.where(nickel.e > 0, nickel.e / flat_valley_y, nickel.e)

    nickel.plot()

    axes = [
        0.5 * flat_valley_x,
        93 * float(wl_ini),
        -0.1 * nickel.ymax,
        1.1 * nickel.ymax,
    ]
    limits = [0.5 * flat_valley_x, 93 * float(wl_ini), 0, 1.1 * nickel.ymax]

    xy_res = [0.81 * flat_valley_x, 0.90 * float(max(nickel.y))]
    xy_ref = [1.60 * flat_valley_x, 0.40 * float(max(nickel.y))]
    xy_fit = [1.85 * flat_valley_x, 0.75 * float(max(nickel.y))]

    # Estimate of the area of the first peak (normalised units)
    fp_area = 0.5 * (nickel.ymax - 1)
    # 35 initial values, order of niPeaks10: [0] I0, [1] slope, [2] quad,
    # [3] wavelength, [4] twotheta0, [5..14] A0..A9, [15..24] G0..G9,
    # [25..34] S0..S9
    initial = [
        1,
        0.0007 * fp_area,
        0.0,
        float(wl_ini),
        float(zac_ini),
        1.00 * fp_area,
        0.56 * fp_area,
        0.54 * fp_area,
        0.76 * fp_area,
        0.24 * fp_area,
        0.12 * fp_area,
        0.42 * fp_area,
        0.40 * fp_area,
        0.32 * fp_area,
        0.37 * fp_area,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.3,
        0.3,
        0.4,
        0.4,
    ]

    planes = [
        "(111)",
        "(200)",
        "(220)",
        "(311)",
        "(222)",
        "(400)",
        "(331)",
        "(420)",
        "(422)",
        "(333)",
    ]

    refl = reflections_fcc(float(wl_ini), float(zac_ini), lattice=lattice)[:10]
    model = niPeaks10(nickel.x, *initial, lattice=lattice)

    plt.figure(figsize=(9, 6))
    plt.plot(nickel.x, model, "r-", label="Initial model")
    plt.plot(nickel.x, nickel.y, "b-+", label="Ni powder data")
    plt.plot(nickel.x, nickel.y - model, "g-+", label="Residuals")

    plt.legend(loc="best")
    plt.title("Nickel powder diffractogram: Starting point")
    plt.xlabel(r"$2\theta$ (˚)")
    plt.ylabel("Intensity (arb. units)")

    wlength = r"$\lambda = {:.3f}$ Å".format(float(wl_ini))
    zac = r"$2\theta_0 = {:.3f}$˚".format(float(zac_ini))
    plt.text(
        xy_res[0],
        xy_res[1],
        s=wlength + "\n" + zac,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"),
    )

    planes_fit = ""
    for i in range(10):
        planes_fit += "{}: {:.1f}˚\n".format(planes[i], float(refl[i]))
    plt.text(
        xy_ref[0],
        xy_ref[1],
        s=planes_fit[:-1],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.axis((axes[0], axes[1], axes[2], axes[3]))
    plt.grid(True)
    plt.tight_layout()
    return axes, limits, xy_res, xy_ref, xy_fit, planes, initial, nickel


def fitting_model_d4nifit(  # noqa: N802 (legacy API)
    nickel: DataXYE,
    limits: list[float],
    initial: list[float],
    lattice: float = NI_LATTICE_PARAMETER,
) -> lm.model.ModelResult:
    """
    Build and fit the Ni powder model for wavelength calibration.

    `lattice` is the Ni lattice parameter in Å (default 3.52 Å). It is held
    fixed: the fitted wavelength scales linearly with it.

    Steps, as implemented:

    1. Select the points inside the rectangle `limits` with
       ``toscana.numerics.fitting.fittingRange``: xmin <= 2θ <= xmax and
       ymin <= y <= ymax (it prints a warning for fewer than 10 points and an
       error for none).
    2. Weights: if the selected errors are non-empty, all finite and all
       > 0, weights = 1/σ; otherwise (e.g. 2-column files, σ = -1, or a
       single invalid error) no weights are used and the line
       "WARNING: no valid errors in the Ni data; fitting without weights."
       is printed.
    3. ``lmfit.Model(niPeaks10)`` with independent variable ``x`` and 36
       parameters. Free (24): I0, slope, wavelength, twotheta0, A0..A9,
       S0..S9. Fixed (12): quad, G0..G9 and lattice (= `lattice`). No bounds.
    4. Starting values from `initial` by index: I0 = initial[0],
       slope = [1], quad = [2], wavelength = [3], twotheta0 = [4],
       A_i = [5 + i], G_i = [15 + i], S_i = [25 + i] (i = 0..9).
    5. ``gmodel.fit`` with lmfit defaults (method "leastsq",
       Levenberg-Marquardt; ``scale_covar=True``), minimising
       Σ [w (y - model)]².

    Parameters
    ----------
    nickel : DataXYE
        Normalised Ni data returned by `setting_model_d4nifit` (uses ``x``,
        ``y``, ``e``).
    limits : list of float
        Fitting rectangle [xmin, xmax, ymin, ymax] in degrees and normalised
        intensity (from `setting_model_d4nifit`; may be edited before).
    initial : list of float
        The 35 starting values (index table in the module docstring). Only
        the first 35 elements are used.
    lattice : float, optional
        Ni lattice parameter a in Å, a fixed parameter of the fit. Default
        ``NI_LATTICE_PARAMETER`` (3.52 Å). Use the same value as in
        `setting_model_d4nifit`.

    Returns
    -------
    lmfit.model.ModelResult
        The fit result. ``result.params`` contains all 36 parameters (values,
        ``stderr`` for the free ones, ``vary`` flags); ``result.params
        ["wavelength"]`` and ``["twotheta0"]`` are the calibration (Å,
        degrees). Pass it to `showing_results_d4nifit`.

    Raises
    ------
    IndexError
        If `initial` has fewer than 35 elements.
    ImportError
        If lmfit is not installed (imported inside the function).
    Exception
        Errors from lmfit/SciPy propagate, e.g. when fewer points than free
        parameters (24) are selected, or when the model returns NaN.

    Notes
    -----
    Divergence from legacy (audit 2026-11, P11): weighted least squares with
    weights 1/σ (legacy ignored the errors). With ``scale_covar=True`` the
    reported uncertainties are multiplied by sqrt(reduced χ²) and so also
    account for model misfit; χ² is the weighted sum Σ ((y - model)/σ)².

    Bug fix vs legacy (audit 2026-11, C10): A2, A3, A4 started from
    initial[6..8] instead of initial[7..9] (so A1 and A2 started equal and
    initial[9] was never used). Only starting values were affected, but a
    non-linear fit can converge differently.

    Divergence from legacy (audit 2026-11, P7): the lattice parameter is a
    fixed parameter (default 3.52 Å instead of the hard-coded 3.52024 Å).
    Because only λ/a is determined, λ_fit(a') = λ_fit(a)·a'/a.

    See Also
    --------
    setting_model_d4nifit, showing_results_d4nifit
    toscana.models.ni.niPeaks10
    """
    import lmfit as lm

    xfit, yfit, efit = fittingRange(
        limits[0], limits[1], limits[2], limits[3], nickel.x, nickel.y, nickel.e
    )
    # Weighted least squares, weights 1/σ (divergence from legacy, which ignored
    # the errors). lmfit keeps scale_covar=True, so the reported uncertainties
    # are also scaled by sqrt(reduced χ²) and account for model misfit.
    # Without valid errors (e.g. 2-column files, σ = -1) the fit is unweighted.
    efit = np.ravel(efit)
    if efit.size and np.all(np.isfinite(efit)) and np.all(efit > 0):
        weights = 1.0 / efit
    else:
        weights = None
        print("WARNING: no valid errors in the Ni data; fitting without weights.")

    gmodel = lm.Model(niPeaks10)
    gmodel.set_param_hint("I0", vary=True)
    gmodel.set_param_hint("slope", vary=True)
    gmodel.set_param_hint("quad", vary=False)
    gmodel.set_param_hint("wavelength", vary=True)
    gmodel.set_param_hint("twotheta0", vary=True)
    gmodel.set_param_hint("lattice", value=lattice, vary=False)
    for i in range(10):
        gmodel.set_param_hint(f"A{i}", vary=True)
    for i in range(10):
        gmodel.set_param_hint(f"G{i}", vary=False)
    for i in range(10):
        gmodel.set_param_hint(f"S{i}", vary=True)

    result = gmodel.fit(
        np.ravel(yfit),
        x=np.ravel(xfit),
        weights=weights,
        I0=initial[0],
        slope=initial[1],
        quad=initial[2],
        wavelength=initial[3],
        twotheta0=initial[4],
        A0=initial[5],
        A1=initial[6],
        # Bug fix vs legacy: A2..A4 were initial[6..8] (shifted by one, so
        # initial[9] was never used and A1/A2 started from the same value).
        A2=initial[7],
        A3=initial[8],
        A4=initial[9],
        A5=initial[10],
        A6=initial[11],
        A7=initial[12],
        A8=initial[13],
        A9=initial[14],
        G0=initial[15],
        G1=initial[16],
        G2=initial[17],
        G3=initial[18],
        G4=initial[19],
        G5=initial[20],
        G6=initial[21],
        G7=initial[22],
        G8=initial[23],
        G9=initial[24],
        S0=initial[25],
        S1=initial[26],
        S2=initial[27],
        S3=initial[28],
        S4=initial[29],
        S5=initial[30],
        S6=initial[31],
        S7=initial[32],
        S8=initial[33],
        S9=initial[34],
        lattice=lattice,
    )
    return result


def showing_results_d4nifit(  # noqa: N802 (legacy API)
    result: lm.model.ModelResult,
    nickel: DataXYE,
    axes: list[float],
    limits: list[float],
    xy_res: list[float],
    xy_ref: list[float],
    xy_fit: list[float],
    planes: list[str],
) -> None:
    """
    Print and plot the results of Ni powder calibration fitting (legacy behavior).

    Steps, as implemented:

    1. Print an empty line and "Results of fitting the nickel powder sample".
    2. For each parameter of ``result.params`` (in lmfit order, which is the
       positional order of ``niPeaks10``, lattice last) store value, stderr
       (0 if None) and relative error |stderr/value| (0 if value = 0 or
       stderr is None).
    3. Recompute the model over the whole data range,
       ``niPeaks10(nickel.x, *values)``, and the reflection positions
       ``reflections_fcc(λ_fit, 2θ0_fit, lattice)[:10]`` with the lattice
       read from ``result.params["lattice"]`` (default
       ``NI_LATTICE_PARAMETER`` if absent).
    4. Print I0, slope, quad, wavelength (Å) and twotheta0 (˚) as
       "value ± stderr  relative%", then a 103-character table with, for
       each of the ten planes: label, fitted 2θ position (deg), area A_i ±
       error and relative error, width S_i ± error and relative error.
       Fixed parameters (quad) are shown with error 0.
    5. Plot (new 9 × 6 in figure) the fitted model (red), data (blue "+")
       and residuals (green) with: title "Final fit for <filename>, <date>"
       (``getDate()``); a box at `xy_fit` with the number of function
       evaluations, χ², reduced χ² (parsed from ``result.fit_report()``) and
       the numors (characters 15-28 of header line 7 of the data file, if
       present); a box at `xy_res` with λ and 2θ0 ± errors (6 decimals); the
       reflection list at `xy_ref`; a small signature line below the lower
       left corner of `axes`. Axis window `axes`, grid.
    6. Save the figure as ``nickel.basename + ".png"`` and print
       "Results saved as <basename>.png".

    Parameters
    ----------
    result : lmfit.model.ModelResult
        Result of `fitting_model_d4nifit`.
    nickel : DataXYE
        The normalised data (from `setting_model_d4nifit`); uses ``x``,
        ``y``, ``filename``, ``basename`` and ``head``.
    axes : list of float
        Plot window [xmin, xmax, ymin, ymax] (degrees, normalised intensity).
    limits : list of float
        Fitting rectangle. Accepted for a uniform call signature but not used
        by this function.
    xy_res : list of float
        [x, y] (data coordinates) of the λ / 2θ0 text box.
    xy_ref : list of float
        [x, y] of the reflection-list text box.
    xy_fit : list of float
        [x, y] of the fit-statistics text box.
    planes : list of str
        Labels of the ten reflections (at least 10 elements).

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If ``result.params`` lacks I0, slope, quad, wavelength, twotheta0 or
        any A_i / S_i.
    UnboundLocalError
        If ``result.params`` has no "wavelength" or "twotheta0" parameter.
    IndexError
        If `planes` has fewer than 10 elements.
    OSError
        If the PNG file cannot be written.

    Notes
    -----
    Side effects: prints to stdout, draws a figure (not shown; displayed by
    the notebook) and writes ``<basename>.png``, where basename is the data
    file path without its extension (``os.path.splitext``). The file is thus
    written next to the data file when `nifile` contained a directory, or in
    the current working directory otherwise; an existing file is overwritten.

    The fit statistics are cut from ``result.fit_report()`` with fixed
    character offsets, so a change of the lmfit report layout garbles that
    text box (the fit itself is unaffected).

    Divergence from legacy (audit 2026-11, P7): the lattice parameter used for
    the reflection positions is read back from the fit result instead of
    the hard-coded 3.52024 Å.

    See Also
    --------
    setting_model_d4nifit, fitting_model_d4nifit
    """
    import matplotlib.pyplot as plt

    print()
    print("Results of fitting the nickel powder sample")

    nickel_table: dict[str, list[float]] = {}
    nickel_par: list[float] = []

    # nickel_table[name] = [value, stderr, relative error]; nickel_par keeps the
    # values in lmfit order (= positional order of niPeaks10, lattice last)
    for param in result.params.values():
        if param.value != 0:
            relative = (
                abs(float(param.stderr) / float(param.value))
                if param.stderr is not None
                else 0.0
            )
        else:
            relative = 0.0
        nickel_table[param.name] = [
            float(param.value),
            float(param.stderr) if param.stderr is not None else 0.0,
            float(relative),
        ]
        nickel_par.append(float(param.value))
        if param.name == "wavelength":
            wlength = r"$\lambda = ({:.6f} \pm {:.6f})$ Å".format(
                float(param.value), float(param.stderr or 0.0)
            )
        if param.name == "twotheta0":
            zac = r"$2\theta_0 = ({:.6f} \pm {:.6f})$˚".format(
                float(param.value), float(param.stderr or 0.0)
            )

    model = niPeaks10(nickel.x, *nickel_par)  # includes the (fixed) lattice
    lattice = (
        float(result.params["lattice"].value)
        if "lattice" in result.params
        else NI_LATTICE_PARAMETER
    )
    refl = reflections_fcc(nickel_par[3], nickel_par[4], lattice=lattice)[:10]

    def _stderr(name: str) -> float:
        v = nickel_table[name][1]
        return float(v) if v is not None else 0.0

    print(
        f"{result.params['I0'].name:>10} = "
        f"{float(nickel_table['I0'][0]):>9f} ± {_stderr('I0'):>9f}         {float(nickel_table['I0'][2]):>7.3%}"
    )
    print(
        f"{result.params['slope'].name:>10} = "
        f"{float(nickel_table['slope'][0]):>9f} ± {_stderr('slope'):>9f}         {float(nickel_table['slope'][2]):>7.3%}"
    )
    print(
        f"{result.params['quad'].name:>10} = "
        f"{float(nickel_table['quad'][0]):>9f} ± {_stderr('quad'):>9f}         {float(nickel_table['quad'][2]):>7.3%}"
    )
    print(
        f"{result.params['wavelength'].name:>10} = "
        f"({float(nickel_table['wavelength'][0]):>8f} ± {_stderr('wavelength'):>9f}) Å      {float(nickel_table['wavelength'][2]):>7.3%}"
    )
    print(
        f"{result.params['twotheta0'].name:>10} = "
        f"({float(nickel_table['twotheta0'][0]):>8.5f} ± {_stderr('twotheta0'):>9f})˚       {float(nickel_table['twotheta0'][2]):>7.3%}"
    )

    print(103 * "-")
    print(
        f"|{'Plane':>6s} | {'Center':>7s}  | {'Area        ':>20s}  |  {'%   ':>8s} ||"
        f"{'Sigma        ':>19s}  |   {'%   ':>8s} |"
    )
    print(103 * "-")
    for i in range(10):
        ai = f"A{i}"
        si = f"S{i}"
        print(
            f"|{planes[i]:>6s} | {float(refl[i]):>8.5f} | {float(nickel_table[ai][0]):>8.5f} ± {_stderr(ai):>9f}  |  {float(nickel_table[ai][2]):>8.3%} ||"
            f"{float(nickel_table[si][0]):>8.5f} ± {_stderr(si):>9f}  |  {float(nickel_table[si][2]):>9.3%} |"
        )
    print(103 * "-")

    plt.figure(figsize=(9, 6))
    plt.plot(nickel.x, model, "r-", label="Fit")
    plt.plot(nickel.x, nickel.y, "b+", label="Ni powder data")
    plt.plot(nickel.x, nickel.y - model, "g-+", label="Residuals")

    plt.legend(loc="best")
    plt.title("Final fit for " + nickel.filename + ", " + getDate())
    plt.xlabel(r"$2\theta$" + " (˚)")
    plt.ylabel("Intensity (arb. units)")

    # Fit statistics cut from the lmfit report at fixed offsets (layout dependent)
    fit_report = result.fit_report()
    ite = fit_report.find("function evals")
    chi = fit_report.find("chi-square")
    red = fit_report.find("reduced chi-square")
    aka = fit_report.find("Akaike")
    ite_s = fit_report[ite + 18 : chi - 62]
    chi_s = fit_report[chi + 20 : red - 5]
    red_s = fit_report[red + 20 : aka - 5]

    numors = ""
    if len(nickel.head) > 6 and len(nickel.head[6]) >= 28:
        numors = nickel.head[6][15:28]
    info_fit = "Iterations ={}\nchi-sq ={}\nreduced chi-sq ={}\nNumors: {}".format(
        ite_s, chi_s, red_s, numors
    )

    signature = "d4nifit.ipynb (Dec 2022), G.J. Cuello (ILL), cuello@ill.fr"
    plt.text(
        xy_fit[0],
        xy_fit[1],
        s=info_fit,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"),
    )
    plt.text(
        xy_res[0],
        xy_res[1],
        s=wlength + "\n" + zac,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"),
    )
    plt.text(
        axes[0] - 0.05 * (axes[1] - axes[0]),
        axes[2] - 0.1 * (axes[3] - axes[2]),
        s=signature,
        fontsize=6,
    )

    planes_fit = ""
    for i in range(10):
        planes_fit += "{}: {:.1f}˚\n".format(planes[i], float(refl[i]))
    plt.text(
        xy_ref[0],
        xy_ref[1],
        s=planes_fit[:-1],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.axis((axes[0], axes[1], axes[2], axes[3]))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(nickel.basename + ".png")
    print("Results saved as " + nickel.basename + ".png")
    return
