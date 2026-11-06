"""
Numerical methods of ToScaNA (package ``toscana.numerics``).

Generic, physics-free numerical tools used along the D4 reduction chain
(raw numors -> corrections -> regrouping onto 2θ and Q grids -> CORRECT ->
S(Q) -> sine Fourier transform to G(r), g(r), RDF, T(r), N(r)). This package
depends only on NumPy and SciPy; it does not import other ToScaNA packages
(the legacy alias ``toscana.math`` maps to it).

Submodules and public functions (all re-exported here)
------------------------------------------------------
fourier
    sineFT          S(Q) -> r, G, g, RDF, T, N (optionally with 1σ errors).
    getSineFT       Generic sine transform 2/π·c·∫ [y(v) - s] v sin(vu) W(v) dv.
    backFT          Cut g(r), transform to S(Q) and back.
    getSineIntegral Sine integral Si(x).
windows
    Lorch           Standard Lorch window at one point (M(0) = 1).
    LorchArray      Standard Lorch window over an array.
binning
    get_xlim        Bin edges and centres of a regular grid.
    get_bins        Bins covered by a rectangle and the fraction in each.
    rebin           Fractional rebinning of (x, y, σ) data onto a regular grid.
operations
    binary_sum      w1·y1 + w2·y2 with propagated errors.
    wsum2           Weighted sum / mixture / inverse-variance mean of two sets.
    ratio           y1/y2 with propagated errors (lists).
line_shapes
    sigmoidal       Logistic step from lowQ to highQ.
    siglin          sigmoidal + linear term.
    step            Box window: 1 for 0 <= q < qmax, else 0.
    Lorentzian, Lorentzian_error   Lorentzian (height A, FWHM γ) [and error].
    Gaussian, Gaussian_error       Area-normalised Gaussian [and error].
    GaussianA       Gaussian with exponential asymmetry.
    LorGau, LorGau_error           Unit-height Gaussian/Lorentzian mixture
                                   [and error].
fitting
    get_chi         RMS difference of two sets of values.
    fit_and_find_extremum  Parabola fit and its extremum.
    fittingRange0, fittingRange, fit_range  Points inside a rectangle.
polynomials
    polyQ1, polyQ2, polyQ4  Polynomials (Horner scheme).
peak_finding
    find_minimum_within_range  Minimum of y in an x range.
    find_peaks_in_range        Local maxima above 3.5 × the range minimum.
signal_processing
    smooth_curve    Centred moving average.

Conventions and units
---------------------
Q in Å⁻¹, r in Å, angles 2θ in degrees, wavelengths in Å, densities in
atoms/Å³. Data sets are NumPy arrays with columns (x, y, σ_y); errors are 1σ
and assumed uncorrelated. Several functions keep the legacy names (camelCase)
and the legacy behaviour of printing messages; the corrections made during
the audit of 2026-11 are documented in each function's Notes and in Appendix A
of docs/ToScaNA_appendices.tex.
"""

from .binning import get_bins, get_xlim, rebin
from .fitting import (
    fit_and_find_extremum,
    fit_range,
    fittingRange,
    fittingRange0,
    get_chi,
)
from .fourier import backFT, getSineFT, getSineIntegral, sineFT
from .line_shapes import (
    Gaussian,
    Gaussian_error,
    GaussianA,
    Lorentzian,
    Lorentzian_error,
    LorGau,
    LorGau_error,
    siglin,
    sigmoidal,
    step,
)
from .operations import binary_sum, ratio, wsum2
from .peak_finding import find_minimum_within_range, find_peaks_in_range
from .polynomials import polyQ1, polyQ2, polyQ4
from .signal_processing import smooth_curve
from .windows import Lorch, LorchArray

__all__ = [
    "backFT",
    "binary_sum",
    "fit_and_find_extremum",
    "get_bins",
    "get_chi",
    "getSineFT",
    "getSineIntegral",
    "get_xlim",
    "Lorch",
    "LorchArray",
    "find_minimum_within_range",
    "find_peaks_in_range",
    "Gaussian",
    "GaussianA",
    "Gaussian_error",
    "polyQ1",
    "polyQ2",
    "polyQ4",
    "fittingRange0",
    "fittingRange",
    "fit_range",
    "LorGau",
    "LorGau_error",
    "Lorentzian",
    "Lorentzian_error",
    "ratio",
    "rebin",
    "siglin",
    "sigmoidal",
    "sineFT",
    "step",
    "smooth_curve",
    "wsum2",
]
