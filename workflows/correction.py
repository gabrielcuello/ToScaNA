"""
From regrouped D4 data to S(Q): Paalman & Pings corrections with MCAMS
(default) or pycorrect.

Position in the processing chain
--------------------------------
d4creg (``.adat``/``.qdat``) -> **this module** -> S(Q) ->
``toscana.numerics.fourier.sineFT`` (G(r), g(r), RDF, T(r), N(r)).

Two methods
-----------
``method="mcams"`` (default)
    ToScaNA's own Paalman & Pings chain, with factors from MCAMS
    (`toscana.corrections.factors`): any number of concentric cylinders
    (hollow sample, annular container, several environment layers, gaps),
    partial filling, angle-dependent multiple scattering, exact transmission
    and normalisation by the number of atoms in the beam (n<L>):

    1. every measurement minus its background, B_C = T_C I_empty +
       (1 - T_C) I_absorber with an absorber measurement, I_empty otherwise;
    2. vanadium: (M_v - B_v) / (1 + δ_v) / A_v,v, Placzek correction
       1/(1 - P_v), smoothing: I_v; K = I_v / (n<L>_v σ_v / 4π) is the
       count rate per unit isotropic cross section per atom/beam area;
    3. multiple scattering of each configuration subtracted (δ is defined for
       isotropic scatterers, as in CORRECT):
       MS_C = δ_C K Σ_b A_b,C Σρσ<L>_b / 4π;
    4. nested subtraction from the outermost body inwards, e.g.
       I_h = single_h / A_h,h;
       I_c = (single_ch - A_h,ch I_h) / A_c,ch;
       I_s = (single_sch - A_c,sch I_c - A_h,sch I_h) / A_s,sch;
    5. dσ/dΩ = I_s / (K n<L>_s) + Σ c_i σ_i/4π P_i (Placzek of the sample).

    With CORRECT's own factors this chain gives exactly CORRECT's result
    (tests/test_correction_mcams.py).
``method="pycorrect"``
    CORRECT 2.24 itself (pycorrect), for the set-ups it can describe (solid
    sample, one container wall, one furnace layer, solid vanadium);
    optionally with MCAMS factors (``mcams="delta"`` or ``"all"``).

All cross sections come from ToScaNA's isotope table. Data files may be on
the angle (``.adat``) or the Q (``.qdat``) scale: Q is converted to 2θ with
the wavelength. Factors and intermediate intensities are returned and can be
saved on either scale for the subtraction in notebooks.

Public names
------------
read_regrouped, run_correction, CorrectionResult, structure_factor,
save_structure_factor, build_state (pycorrect input), MATERIALS,
material_properties (from toscana.corrections.setup).
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from toscana.corrections.setup import MATERIALS, material_properties  # noqa: F401
from toscana.isotopes import Element

__all__ = [
    "MATERIALS", "CorrectionResult", "build_state", "material_properties", "read_regrouped",
    "run_correction", "save_structure_factor", "structure_factor",
]


# ============================================================================
# Reading regrouped data
# ============================================================================
def read_regrouped(path: str, wavelength: float, scale: Optional[str] = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Read a regrouped diffractogram on the angle or the Q scale.

    Parameters
    ----------
    path : str
        ``.adat``/``.qdat`` of d4creg, a CORRECT/pycorrect text file, or any
        file with columns x, y[, error] (lines that do not start with three
        numbers are ignored).
    wavelength : float
        Å, to convert Q to 2θ.
    scale : {"angle", "q"}, optional
        Scale of the x column. Default: from the header (``X caption``
        containing "theta"/"angle" or "Q"/"momentum"), else from the extension
        (``.adat``, ``.reg``: angle; ``.qdat``, ``.soq``: Q).

    Returns
    -------
    theta, y, e : ndarray
        2θ (deg, increasing), intensity, error (0 if absent).
    scale : str
        The scale found in the file.
    """
    header, rows = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tokens = line.replace(",", " ").split()
            try:
                vals = [float(t) for t in tokens[:3]]
                if len(vals) >= 2 and not line.lstrip().startswith("#"):
                    rows.append(vals + [0.0] * (3 - len(vals)))
                    continue
            except ValueError:
                pass
            header.append(line.lower())
    if not rows:
        raise ValueError(f"No data in {path}.")
    if scale is None:
        caption = " ".join(h for h in header if "x caption" in h)
        if "theta" in caption or "angle" in caption:
            scale = "angle"
        elif "q" in caption.split(":")[-1] or "momentum" in caption:
            scale = "q"
        else:
            ext = os.path.splitext(path)[1].lower()
            scale = {".adat": "angle", ".reg": "angle", ".qdat": "q", ".soq": "q"}.get(ext)
        if scale is None:
            raise ValueError(f"Cannot tell whether {path} is on the angle or Q scale: give scale=.")
    data = np.array(rows, dtype=float)
    x, y, e = data[:, 0], data[:, 1], data[:, 2]
    if scale == "q":
        s = x * wavelength / (4 * math.pi)
        ok = (s > 0) & (s < 1)
        x, y, e = 2 * np.degrees(np.arcsin(s[ok])), y[ok], e[ok]
    elif scale != "angle":
        raise ValueError("scale must be 'angle' or 'q'.")
    order = np.argsort(x)
    return x[order], y[order], e[order], scale


def _on_grid(theta: np.ndarray, data: tuple, notes: list, name: str):
    t, y, e = data
    if len(t) == len(theta) and np.allclose(t, theta, atol=1e-6):
        return y, e
    notes.append(f"{name}: interpolated onto the sample's angle grid")
    return np.interp(theta, t, y), np.interp(theta, t, e)


# ============================================================================
# Result
# ============================================================================
@dataclass
class CorrectionResult:
    """
    Result of `run_correction`.

    Attributes
    ----------
    theta, q : ndarray
        2θ (deg) and Q (1/Å) of the data points.
    dcs, err : ndarray
        Corrected differential cross section per atom and its error, barn/sr
        (method "mcams": errors of all measurements; the vanadium's only when
        it is not smoothed).
    wavelength : float
        Å.
    method : str
        "mcams" or "pycorrect" (with "+mcams-delta"/"+mcams-all").
    factors, vanadium_factors : FactorSet or None
        Paalman & Pings factors, transmissions and multiple scattering of the
        sample set-up and of the vanadium set-up, on the data grid
        (``.table("q")``, ``.save(file, scale="q")``).
    intensities : dict
        Single-scattering intensities per body after the subtraction
        (``"I_s"``, ``"I_c"``, ``"I_h"``, ``"I_v"``, ``"I_v_smooth"``) and their
        errors (``"<key>_err"``), on the data grid (mcams method).
    backgrounds : dict
        Background subtracted from each measurement (``"B_<config>"``).
    pycorrect : dict or None
        Full pycorrect result (method "pycorrect"; with MCAMS factors, the
        run that used them).
    mcams : dict or None
        MCAMS factors used in the hybrid pycorrect run.
    state : CorrectState or None
        pycorrect input (method "pycorrect").
    notes : list of str
        What was done that the user may want to know (interpolations, ...).
    """

    theta: np.ndarray
    q: np.ndarray
    dcs: np.ndarray
    err: np.ndarray
    wavelength: float
    method: str
    factors: Any = None
    vanadium_factors: Any = None
    intensities: dict = field(default_factory=dict)
    backgrounds: dict = field(default_factory=dict)
    pycorrect: Optional[dict] = None
    mcams: Optional[dict] = None
    state: Any = None
    notes: list = field(default_factory=list)

    def save_factors(self, basename: str, scale: str = "q") -> list[str]:
        """Write ``<basename>_sample_factors.dat`` and
        ``<basename>_vanadium_factors.dat`` on the chosen scale; returns the
        file names."""
        out = []
        for fs, tag in ((self.factors, "sample"), (self.vanadium_factors, "vanadium")):
            if fs is not None:
                name = f"{basename}_{tag}_factors.dat"
                fs.save(name, scale=scale, heading=[f"{tag} set-up"])
                out.append(name)
        return out

    def save_intensities(self, filename: str, scale: str = "q") -> None:
        """Write the intermediate intensities (and backgrounds) as a table."""
        x = self.theta if scale == "angle" else self.q
        cols = dict(self.intensities)
        cols.update(self.backgrounds)
        names = list(cols)
        order = np.argsort(x)
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(f"# {filename}\n# single-scattering intensities per body after the "
                     "Paalman & Pings subtraction (I_b), backgrounds (B_C)\n")
            fh.write("# columns: " + " ".join(["2theta(deg)" if scale == "angle" else "Q(1/A)"]
                                              + names) + "\n")
            for i in order:
                fh.write(f"{x[i]:10.5f} " + " ".join(f"{cols[n][i]:14.7g}" for n in names) + "\n")


# ============================================================================
# The general chain (MCAMS factors)
# ============================================================================
def _placzek(theta, lam, mass, temp, method):
    from toscana.corrections.pycorrect.correct_2axis import placzek

    names = {"series": "SERIES_EXPANSION", "diatomic": "DIATOMIC", "none": "NONE"}
    return placzek(theta, lam, mass, temp, "D4", names[method], 0.0)


def _chain(fs, measured, background, multiple=None):
    """Nested Paalman & Pings subtraction for one set-up.

    measured[cfg] = (y, e) on the data grid; background[cfg] = (y, e).
    multiple[cfg]: multiple scattering (counts) to subtract; if None, the
    measurement is divided by 1 + δ_C (exact for an isotropic scatterer such
    as vanadium). Returns {body: (I, e)} single-scattering intensities."""
    out = {}
    for cfg in fs.configurations:
        new = cfg[0]
        y, e = measured[cfg]
        by, be = background.get(cfg, (0.0, 0.0))
        var = e ** 2 + np.asarray(be) ** 2
        if multiple is None:
            d = fs.delta.get(f"delta_{cfg}", 0.0)
            single = (y - by) / (1 + d)
            var = var / (1 + d) ** 2
        else:
            single = y - by - multiple.get(cfg, 0.0)
        for body in cfg[1:]:
            a = fs.A[f"A_{body},{cfg}"]
            single = single - a * out[body][0]
            var = var + (a * out[body][1]) ** 2
        a_new = fs.A[f"A_{new},{cfg}"]
        out[new] = (single / a_new, np.sqrt(var) / a_new)
    return out


def _files_of(m, files, environment):
    files = dict(files or {})
    absorber_given = str(m.absorber[0]).strip().lower() not in ("", "none")
    return {
        "sample": files.get("sample", m.File),
        "container": files.get("container", m.conFile),
        "vanadium": files.get("vanadium", m.norFile),
        "environment": files.get("environment", m.envFile),
        "empty": files.get("empty", m.envFile if environment == "background" else ""),
        "absorber": files.get("absorber", m.absFile if absorber_given else ""),
    }


def run_correction(
    m: Any,
    *,
    method: str = "mcams",
    files: Optional[Mapping[str, str]] = None,
    environment: str = "background",
    vanadium_in_environment: bool = False,
    placzek: str = "series",
    vanadium_smoothing: int = 0,
    materials: Optional[Mapping[str, tuple]] = None,
    factors: Any = None,
    vanadium_factors: Any = None,
    factor_options: Optional[Mapping[str, Any]] = None,
    mcams: Optional[str] = None,
    mcams_options: Optional[Mapping[str, Any]] = None,
    workdir: Optional[str] = None,
    output: str = "corrected.dat",
    verbose: bool = False,
) -> CorrectionResult:
    """
    Correct the sample data of a Measurement.

    Parameters
    ----------
    m : Measurement
        Description of the experiment (sample-description file, including
        ``<LayNN>`` layers).
    method : {"mcams", "pycorrect"}
        ToScaNA's Paalman & Pings chain with MCAMS factors (default), or
        CORRECT 2.24 (pycorrect) when it can describe the set-up.
    files : mapping, optional
        Data files replacing those of `m`: ``"sample"``, ``"container"``,
        ``"vanadium"``, ``"environment"``, ``"empty"`` (empty instrument or
        environment used as background), ``"absorber"``. ``.adat`` or
        ``.qdat`` (see `read_regrouped`).
    environment : {"background", "layer", "none"}
        ``"background"`` (default): the environment measurement is the
        background I_empty of all measurements (as in ToScaNA's CORRECT files
        so far). ``"layer"`` (alias ``"furnace"``): the environment and the
        extra environment layers are a body ``h`` of the set-up, corrected
        for attenuation; I_empty is then ``files["empty"]`` if given.
    vanadium_in_environment : bool
        The vanadium was measured inside the environment.
    placzek : {"series", "diatomic", "none"}
        Placzek inelasticity correction (pycorrect's routine).
    vanadium_smoothing : int
        Polynomial smoothing of the vanadium (number of coefficients, with
        rejection of Bragg peaks; 0 = none; negative: Savitzky-Golay as in
        CORRECT).
    materials : mapping, optional
        Extra container/environment materials.
    factors, vanadium_factors : FactorSet, optional
        Precomputed factors (e.g. from a previous run or a file) instead of
        computing them; must be on the data grid.
    factor_options : mapping, optional
        Passed to `compute_factors` (n_points, ms_particles, step_deg,
        ms_fit_degree, seed).
    mcams, mcams_options
        method="pycorrect" only: replace CORRECT's multiple scattering
        ("delta") or all factors ("all") by MCAMS ones (bridge).
    workdir, output, verbose
        pycorrect's output file and listing.

    Returns
    -------
    CorrectionResult

    Raises
    ------
    ValueError
        Unknown option; set-up not supported by pycorrect; missing
        measurement for a body of the set-up.
    """
    from toscana.corrections.factors import compute_factors
    from toscana.corrections.setup import setups_from_measurement

    if method == "pycorrect":
        return _run_pycorrect(m, files=files, environment=environment, placzek=placzek,
                              vanadium_smoothing=vanadium_smoothing, materials=materials,
                              mcams=mcams, mcams_options=mcams_options, workdir=workdir,
                              output=output, verbose=verbose)
    if method != "mcams":
        raise ValueError("method must be 'mcams' or 'pycorrect'.")
    if mcams is not None:
        raise ValueError("mcams= is for method='pycorrect'; method='mcams' always uses MCAMS.")
    if placzek not in ("series", "diatomic", "none"):
        raise ValueError("placzek must be 'series', 'diatomic' or 'none'.")
    env_mode = "layer" if environment == "furnace" else environment
    sample_setup, van_setup = setups_from_measurement(
        m, environment=env_mode, vanadium_in_environment=vanadium_in_environment,
        materials=materials)
    lam = float(m.wavelength)
    paths = _files_of(m, files, env_mode)
    notes: list[str] = []

    theta, ys, es, _scale = read_regrouped(paths["sample"], lam)
    def load(key):
        p = paths.get(key, "")
        if not str(p).strip():
            return None
        t, y, e, _ = read_regrouped(p, lam)
        return _on_grid(theta, (t, y, e), notes, key)

    data = {"sample": (ys, es)}
    for key in ("container", "vanadium", "environment", "empty", "absorber"):
        loaded = load(key)
        if loaded is not None:
            data[key] = loaded

    opts = dict(factor_options or {})
    fs = factors or compute_factors(sample_setup, theta, **opts)
    fv = vanadium_factors or compute_factors(van_setup, theta, **opts)

    def measurement_of(cfg):
        if "s" in cfg:
            return data["sample"]
        if "v" in cfg:
            return data["vanadium"]
        if "c" in cfg:
            key = "container"
        else:
            key = "environment"
        if key not in data:
            raise ValueError(f"No measurement for configuration {cfg!r} ({key} file).")
        return data[key]

    def background_of(cfg, fset):
        empty, absorber = data.get("empty"), data.get("absorber")
        if absorber is not None:
            t = fset.T[f"T_{cfg}"]
            ey, ee = empty if empty is not None else (0.0, 0.0)
            by = t * ey + (1 - t) * absorber[0]
            be = np.sqrt((t * np.asarray(ee)) ** 2 + ((1 - t) * absorber[1]) ** 2)
            return by, be
        return empty if empty is not None else (np.zeros_like(theta), np.zeros_like(theta))

    backgrounds = {}
    sample_meas, sample_bkg = {}, {}
    for cfg in fs.configurations:
        sample_meas[cfg] = measurement_of(cfg)
        sample_bkg[cfg] = background_of(cfg, fs)
        backgrounds[f"B_{cfg}"] = np.broadcast_to(sample_bkg[cfg][0], theta.shape).copy()
    van_meas, van_bkg = {}, {}
    for cfg in fv.configurations:
        van_meas[cfg] = measurement_of(cfg)
        van_bkg[cfg] = background_of(cfg, fv)
        backgrounds[f"B_{cfg}"] = np.broadcast_to(van_bkg[cfg][0], theta.shape).copy()

    # vanadium first: it gives the flux x efficiency K(2θ)
    Iv = _chain(fv, van_meas, van_bkg)
    iv, iv_err = Iv["v"]
    pv = _placzek(theta, lam, Element("V").weight, 300.0, placzek)
    iv, iv_err = iv / (1 - pv), iv_err / (1 - pv)
    from toscana.corrections.pycorrect.correct_2axis import smooth_vanadium, smooth_vanadium_sg
    from toscana.corrections.pycorrect.state import CorrectState

    if vanadium_smoothing >= 0:
        iv_smooth = smooth_vanadium(theta, iv, np.maximum(iv_err, 1e-30), vanadium_smoothing)
    else:
        st0 = CorrectState()
        iv_smooth = smooth_vanadium_sg(theta, iv, vanadium_smoothing, lam,
                                       st0.calib_ur, st0.calib_vr, st0.calib_wr)

    # K = I_v / (n<L>_v σ_v / 4π): counts per unit isotropic cross section.
    # Multiple scattering of configuration C (δ_C is defined for isotropic
    # scatterers, as in CORRECT): M_C = δ_C K Σ_b A_b,C n<L>_b σ_b / 4π.
    k = iv_smooth / (van_setup.scattering_power("v") / (4 * math.pi))
    multiple = {}
    for cfg in fs.configurations:
        iso = sum(fs.A[f"A_{b},{cfg}"] * sample_setup.scattering_power(b) for b in cfg)
        multiple[cfg] = fs.delta.get(f"delta_{cfg}", 0.0) * k * iso / (4 * math.pi)
    I = _chain(fs, sample_meas, sample_bkg, multiple)

    # normalisation by the number of atoms in the beam, and Placzek of the sample
    i_s, i_s_err = I["s"]
    dcs = i_s / k / fs.atoms_per_area["s"]
    err = i_s_err / k / fs.atoms_per_area["s"]
    if vanadium_smoothing == 0:
        # Unsmoothed vanadium: the error of K enters point by point. Since
        # dσ/dΩ = Y/(A_s K n<L>) - (MS terms independent of K), it scales the
        # intensity BEFORE the multiple-scattering subtraction, Y/A_s.
        y_s = _chain(fs, sample_meas, sample_bkg, {c: 0.0 for c in fs.configurations})["s"][0]
        err = np.hypot(err, y_s / k / fs.atoms_per_area["s"] * iv_err / np.abs(iv))
    comp = sample_setup.layers_of("s")[0].composition
    for symbol, c in comp.items():
        e = Element(symbol)
        dcs = dcs + c * e.sig_sca / (4 * math.pi) * _placzek(theta, lam, e.weight,
                                                             float(m.TempK), placzek)

    intensities = {}
    for body, (val, unc) in list(I.items()) + [("v", (iv, iv_err))]:
        intensities[f"I_{body}"] = val
        intensities[f"I_{body}_err"] = unc
    intensities["I_v_smooth"] = iv_smooth
    for cfg, val in multiple.items():
        intensities[f"MS_{cfg}"] = np.broadcast_to(val, theta.shape).copy()
    q =4 * math.pi / lam * np.sin(np.radians(theta) / 2)
    return CorrectionResult(theta=theta, q=q, dcs=dcs, err=err, wavelength=lam, method="mcams",
                            factors=fs, vanadium_factors=fv, intensities=intensities,
                            backgrounds=backgrounds, notes=notes)


# ============================================================================
# pycorrect (CORRECT 2.24) path
# ============================================================================
def _present(name: Any, path: Any) -> bool:
    return bool(str(name).strip()) and str(name).strip().lower() != "none" and bool(
        str(path).strip()
    )


def build_state(
    m: Any,
    *,
    files: Optional[Mapping[str, str]] = None,
    environment: str = "background",
    placzek: str = "series",
    vanadium_smoothing: int = 0,
    materials: Optional[Mapping[str, tuple]] = None,
    output: str = "corrected.dat",
):
    """
    pycorrect state (input of CORRECT) for a `Measurement`.

    Parameters
    ----------
    m : toscana.experiment.Measurement
        Description of the experiment (see ``getRunningParams``).
    files : mapping, optional
        Data files on the ANGLE scale, overriding those of `m`. Keys:
        ``"sample"``, ``"container"``, ``"vanadium"``, ``"environment"``,
        ``"background"`` (an empty-instrument measurement used as background
        of all bodies). (`run_correction` converts ``.qdat`` files first.)
    environment : {"background", "furnace", "layer", "none"}
        ``"background"`` (default): the environment measurement is the
        background of every measurement; ``"furnace"``/``"layer"``: a furnace
        with the dimensions and material of ``m.environ``; ``"none"``.
    placzek : {"series", "diatomic", "none"}
    vanadium_smoothing : int
        CORRECT's ``/smoothing`` of the vanadium (0: none).
    materials : mapping, optional
        Extra entries for `MATERIALS`.
    output : str
        File written by pycorrect (dσ/dΩ vs Q).

    Returns
    -------
    toscana.corrections.pycorrect.state.CorrectState

    Raises
    ------
    ValueError
        Set-up that CORRECT's cylindrical 2-axis model cannot describe
        (hollow sample or vanadium, extra ``<LayNN>`` layers), missing sample
        file, non-positive beam size or radius, unknown option.

    Notes
    -----
    Mapping (Measurement in mm -> pycorrect in cm): sample radius
    ``OuterDiam/20``, EFFECTIVE atomic density, temperature, fullness,
    species with ToScaNA cross sections; container outer radius
    ``container[3]/20`` (its inner radius is the sample radius); vanadium
    radius ``normaliser[3]/20``; furnace radii ``environ[2]/20``,
    ``environ[3]/20``; beam ``beamHeight/10`` × ``beamWidth/10``.
    """
    from toscana.corrections.pycorrect.state import CorrectState, Species, Spectrum

    if environment == "layer":
        environment = "furnace"
    if environment not in ("background", "furnace", "none"):
        raise ValueError("environment must be 'background', 'furnace' (or 'layer') or 'none'.")
    placzek_map = {"series": "SERIES_EXPANSION", "diatomic": "DIATOMIC", "none": "NONE"}
    if placzek not in placzek_map:
        raise ValueError("placzek must be 'series', 'diatomic' or 'none'.")
    if getattr(m, "extra_layers", []):
        raise ValueError("pycorrect cannot describe the extra <LayNN> layers: use method='mcams'.")
    files = dict(files or {})
    st = CorrectState()
    st.instrument, st.inst_type, st.geometry = "D4", "2-AXIS", "CYLINDRICAL"
    st.wavelength = float(m.wavelength)
    st.beam_height, st.beam_width = float(m.beamHeight) / 10, float(m.beamWidth) / 10
    if st.beam_height <= 0 or st.beam_width <= 0:
        raise ValueError(f"Beam size must be positive (mm): {m.beamHeight} x {m.beamWidth}.")
    st.placzek_correction = placzek_map[placzek]
    st.vanadium_smoothing = int(vanadium_smoothing)
    st.overwrite = True
    st.spectra = [Spectrum(1)]
    st.files["output"] = output

    if float(m.InnerDiam) > 0:
        raise ValueError("pycorrect cannot describe a hollow (annular) sample: use method='mcams'.")
    st.files["sample"] = files.get("sample", m.File)
    if not str(st.files["sample"]).strip():
        raise ValueError("No sample data file.")
    st.ca_r1 = float(m.OuterDiam) / 20
    if st.ca_r1 <= 0:
        raise ValueError("Sample diameter must be positive.")
    st.sa_rho = float(m.EffAtomicDensity)
    st.sa_temp = float(m.TempK)
    st.sa_fullness = float(m.Fullness)
    st.species = []
    for symbol, conc in m.conc.items():
        e = Element(symbol)
        st.species.append(Species(symbol, float(conc), float(e.weight), float(e.sig_sca),
                                  float(e.sig_abs)))

    can_file = files.get("container", m.conFile)
    if _present(m.container[0], can_file):
        if abs(float(m.container[2]) - float(m.OuterDiam)) > 1e-6:
            raise ValueError("pycorrect needs the container inner diameter equal to the sample "
                             "outer diameter (no gap): use method='mcams'.")
        st.files["container"] = can_file
        st.ca_r2 = float(m.container[3]) / 20
        st.ca_rho, st.ca_scat_cs, st.ca_abs_cs = material_properties(m.container[0], materials)
    else:
        st.ca_r2 = st.ca_r1

    van_file = files.get("vanadium", m.norFile)
    if _present("V", van_file):
        if float(m.normaliser[2]) > 0:
            raise ValueError("pycorrect cannot describe a hollow vanadium: use method='mcams'.")
        st.files["vanadium"] = van_file
        st.va_r = float(m.normaliser[3]) / 20
        st.va_rho, st.va_scat_cs, st.va_abs_cs = material_properties("V", materials)

    env_file = files.get("environment", m.envFile)
    if environment == "background" and str(env_file).strip():
        for key in ("bkg_furnace", "bkg_container", "bkg_sample", "bkg_vanadium"):
            st.files[key] = env_file
    elif environment == "furnace" and _present(m.environ[0], env_file):
        st.files["furnace"] = env_file
        st.fu_r1, st.fu_r2 = float(m.environ[2]) / 20, float(m.environ[3]) / 20
        st.fu_rho, st.fu_scat_cs, st.fu_abs_cs = material_properties(m.environ[0], materials)
    if files.get("background"):
        for key in ("bkg_furnace", "bkg_container", "bkg_sample", "bkg_vanadium"):
            st.files[key] = files["background"]
    return st


def _angle_files(m, files, lam, workdir):
    """Copies on the angle scale of the files that are on the Q scale."""
    out = dict(files or {})
    names = {"sample": m.File, "container": m.conFile, "vanadium": m.norFile,
             "environment": m.envFile}
    for key, default in names.items():
        path = out.get(key, default)
        if not str(path).strip() or not os.path.isfile(path):
            continue
        t, y, e, scale = read_regrouped(path, lam)
        if scale == "q":
            from toscana.corrections.pycorrect.fileio import DataWriter

            new = os.path.join(workdir, os.path.basename(path) + ".angle")
            with DataWriter(new, overwrite=True) as w:
                w.append(t, y, e, instrument="D4", title=os.path.basename(path),
                         fnx="Scattering angle (degrees)", fny="Counts")
            out[key] = new
    return out


def _run_pycorrect(m, *, files, environment, placzek, vanadium_smoothing, materials, mcams,
                   mcams_options, workdir, output, verbose) -> CorrectionResult:
    from toscana.corrections.factors import FactorSet
    from toscana.corrections.pycorrect.correct_2axis import correct_2axis

    if mcams not in (None, "delta", "all"):
        raise ValueError("mcams must be None, 'delta' or 'all'.")
    lam = float(m.wavelength)
    tmp = workdir or tempfile.mkdtemp(prefix="toscana_pycorrect_")
    os.makedirs(tmp, exist_ok=True)
    st = build_state(m, files=_angle_files(m, files, lam, tmp), environment=environment,
                     placzek=placzek, vanadium_smoothing=vanadium_smoothing, materials=materials,
                     output=os.path.join(tmp, os.path.basename(output)))
    first = correct_2axis(st, verbose=verbose)[0]
    final, mc, label = first, None, "pycorrect"
    if mcams:
        from toscana.corrections import bridge

        mc = bridge.mcams_factors_like_pycorrect(st, first["theta"], **dict(mcams_options or {}))
        keys = ["delta_sa"] + (["delta_va"] if st.has("vanadium") else [])
        if mcams == "all":
            keys += [k for k in ("assch", "acsch", "acch", "ahsch", "ahch", "ahh") if k in mc]
            if st.has("vanadium") and not (st.va_furnace or st.va_container):
                keys.append("avvch")
        final = correct_2axis(st, verbose=verbose, factors={k: mc[k] for k in keys})[0]
        label = f"pycorrect+mcams-{mcams}"
    theta = np.asarray(final["theta"], float)
    has_c, has_h = st.has("container"), st.has("furnace")
    cfg_s = "s" + ("c" if has_c else "") + ("h" if has_h else "")
    A = {f"A_s,{cfg_s}": final["assch"]}
    if has_c:
        A[f"A_c,{cfg_s}"] = final["acsch"]
        A["A_c,c" + ("h" if has_h else "")] = final["acch"]
    if has_h:
        A[f"A_h,{cfg_s}"] = final["ahsch"]
        if has_c:
            A["A_h,ch"] = final["ahch"]
        A["A_h,h"] = final["ahh"]
    configs = (["h"] if has_h else []) + ([("c" + ("h" if has_h else ""))] if has_c else []) + [cfg_s]
    delta = {f"delta_{cfg_s}": np.broadcast_to(final["delta_sa"], theta.shape).copy()}
    fs = FactorSet(theta, lam, configs, A, {k: np.zeros_like(v) for k, v in A.items()}, delta,
                   {k: np.zeros_like(v) for k, v in delta.items()}, {}, {}, label,
                   {"note": "CORRECT's definitions: delta_sa is subtracted as "
                            "delta*sigma_s/4pi after normalisation"})
    fv = None
    if st.has("vanadium"):
        fv = FactorSet(theta, lam, ["v"], {"A_v,v": final["avvch"]},
                       {"A_v,v": np.zeros_like(theta)},
                       {"delta_v": np.broadcast_to(final["delta_va"], theta.shape).copy()},
                       {"delta_v": np.zeros_like(theta)}, {}, {}, label)
    return CorrectionResult(theta=theta, q=np.asarray(final["q"], float),
                            dcs=np.asarray(final["dcs"], float), err=np.asarray(final["err"], float),
                            wavelength=lam, method=label, factors=fs, vanadium_factors=fv,
                            pycorrect=final, mcams=mc, state=st,
                            notes=[] if final is first else ["pycorrect: first run (CORRECT's own "
                                                             "factors) replaced by the MCAMS one"])


# ============================================================================
# S(Q)
# ============================================================================
def structure_factor(
    result: CorrectionResult,
    m: Any,
    *,
    q_grid: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Faber-Ziman S(Q) from the corrected differential cross section.

        dσ/dΩ = Σ c_i σ_i / 4π  +  <b>² [S(Q) - 1]

    so ``S = 1 + (dσ/dΩ - Σ c_i σ_i/4π) / <b>²`` and ``σ_S = σ_dσ/<b>²``,
    with σ_i the bound scattering cross sections (coherent + incoherent) and
    <b> = Σ c_i b_i (ToScaNA's table; <b>² in barn = |<b> in fm|²/100).

    Parameters
    ----------
    result : CorrectionResult
    m : Measurement
        For the composition ``m.conc``.
    q_grid : (qmin, qmax, dq), optional
        Rebin onto a regular Q grid (`toscana.numerics.binning.rebin`).
        Default: the points of the data, sorted by Q.

    Returns
    -------
    ndarray, shape (n, 3)
        Q (1/Å), S(Q), error: the input of ``sineFT(..., with_errors=True)``.

    Raises
    ------
    ValueError
        <b>² ≈ 0 (null-scattering composition).
    """
    from toscana.numerics.binning import rebin

    conc = {s: float(c) for s, c in m.conc.items()}
    total = sum(conc.values())
    elements = {s: Element(s) for s in conc}
    self_term = sum(c / total * elements[s].sig_sca for s, c in conc.items()) / (4 * math.pi)
    mean_b = sum(c / total * elements[s].bcoh for s, c in conc.items())
    b2 = abs(mean_b) ** 2 / 100.0
    if b2 < 1e-6:
        raise ValueError("<b>^2 is ~0 for this composition: S(Q) cannot be normalised.")
    order = np.argsort(result.q)
    q = result.q[order]
    s = 1.0 + (result.dcs[order] - self_term) / b2
    e = result.err[order] / b2
    sq = np.column_stack([q, s, e])
    if q_grid is None:
        return sq
    qmin, qmax, dq = (float(v) for v in q_grid)
    step = float(np.median(np.diff(np.sort(result.theta))))
    with contextlib.redirect_stdout(io.StringIO()):  # rebin prints a debug line
        return rebin(step, float(result.wavelength), sq, qmin, qmax, dq)


def save_structure_factor(filename: str, sq: np.ndarray, heading: Sequence[str] = ()) -> None:
    """Write S(Q) (columns Q, S, error) with a header describing the columns."""
    from toscana.dataio.saving import saveFile_3col

    lines = list(heading) + [
        "Columns: Q (1/A), S(Q) (Faber-Ziman, dimensionless), 1-sigma error of S(Q)",
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        saveFile_3col(filename, np.asarray(sq), lines)
