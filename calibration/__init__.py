"""
Calibration of the D4 wavelength and zero angle with a Ni powder standard.

Public functions (from ``toscana.calibration.d4nifit``)
-------------------------------------------------------
setting_model_d4nifit(nifile, zac_ini=0.0, lattice=NI_LATTICE_PARAMETER)
    Read and normalise the Ni diffractogram, estimate λ from the (111) peak,
    build the 35 initial parameters, plot the starting model.
fitting_model_d4nifit(nickel, limits, initial, lattice=NI_LATTICE_PARAMETER)
    Weighted lmfit fit of ``toscana.models.ni.niPeaks10`` (λ, 2θ0, background,
    areas and widths free; lattice parameter fixed).
showing_results_d4nifit(result, nickel, axes, limits, xy_res, xy_ref, xy_fit, planes)
    Print λ, 2θ0, background and the reflection table; plot the fit and save
    it as ``<basename>.png`` in the current working directory.

Typical notebook sequence (d4nifit.ipynb)::

    from toscana.calibration import (
        setting_model_d4nifit, fitting_model_d4nifit, showing_results_d4nifit)
    (axes, limits, xy_res, xy_ref, xy_fit,
     planes, initial, nickel) = setting_model_d4nifit("ni.adat")
    result = fitting_model_d4nifit(nickel, limits, initial)
    showing_results_d4nifit(result, nickel, axes, limits,
                            xy_res, xy_ref, xy_fit, planes)

Units: 2θ and 2θ0 in degrees, λ and the lattice parameter in Å, intensities
normalised to the valley between the second and third Ni peaks.
"""

from .d4nifit import (
    fitting_model_d4nifit,
    setting_model_d4nifit,
    showing_results_d4nifit,
)

__all__ = [
    "fitting_model_d4nifit",
    "setting_model_d4nifit",
    "showing_results_d4nifit",
]
