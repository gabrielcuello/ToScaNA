"""
From regrouped D4 data to S(Q): corrections with pycorrect, optionally MCAMS.

Position in the processing chain
--------------------------------
d4creg (``.adat`` files, angle scale) -> **this module** -> S(Q) ->
``toscana.numerics.fourier.sineFT`` (G(r), g(r), RDF, T(r), N(r)).

The experiment is described by a `toscana.experiment.Measurement` (sample
composition, density, geometry, container, vanadium, environment, data files,
beam). `build_state` translates it into the input of pycorrect (CORRECT 2.24
in Python) without writing a ``.com`` file; `run_correction` runs the
corrections (background, container and furnace subtraction, Paalman & Pings
attenuation, vanadium normalisation, multiple scattering, Placzek) and can
replace CORRECT's factors by Monte Carlo ones from MCAMS; `structure_factor`
turns the differential cross section into S(Q).

Public names
------------
MATERIALS
    Composition and density of container / furnace materials (V, TiZr,
    SiO2, Nb). Edit or pass ``materials=`` for others.
material_properties(material)
    Atomic density and cross sections of a material from ToScaNA's table.
build_state(m, ...)
    pycorrect state (`CorrectState`) for a Measurement.
run_correction(m, ..., mcams=None)
    Run the corrections; returns a `CorrectionResult`.
structure_factor(result, m, ...)
    S(Q) = 1 + [dσ/dΩ - Σ c_i σ_i/4π] / <b>², with errors.
save_structure_factor(filename, sq, heading)
    Write S(Q) as a 3-column file with a descriptive header.

Units
-----
Measurement: mm, g/cm³; pycorrect: cm, atoms/Å³, barn (converted here);
dσ/dΩ in barn/sr per atom; Q in 1/Å; S(Q) dimensionless.

Nuclear data
------------
All cross sections (sample species, container, furnace, vanadium) come from
ToScaNA's isotope table (`toscana.isotopes.Element`), not from CORRECT's
``xsect.dat`` or its /type presets, so that the whole package uses one set of
values (e.g. vanadium b_coh = -0.443 fm, σ_s = 5.10466 b, density 6.11 g/cm³).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from toscana.experiment.info import VANADIUM_DENSITY_GCC
from toscana.isotopes import Element
from toscana.physics.properties import getAtomicDensity

#: Container / furnace materials: (composition {symbol: atoms}, density g/cm³).
#: TiZr is the null-scattering alloy Ti0.676Zr0.324 (coherent length ~ 0);
#: its density depends on the batch (5.1-5.3 g/cm³): override if known.
MATERIALS: dict[str, tuple[dict[str, float], float]] = {
    "V": ({"V": 1.0}, VANADIUM_DENSITY_GCC),
    "TiZr": ({"Ti": 0.676, "Zr": 0.324}, 5.23),
    "SiO2": ({"Si": 1.0, "O": 2.0}, 2.20),
    "Nb": ({"Nb": 1.0}, 8.57),
}

MaterialSpec = Union[str, tuple[Mapping[str, float], float]]


def material_properties(
    material: MaterialSpec, materials: Optional[Mapping[str, tuple]] = None
) -> tuple[float, float, float]:
    """
    Atomic density and mean cross sections of a material.

    Parameters
    ----------
    material : str or (dict, float)
        A name of `MATERIALS` (or of `materials`), or a pair
        ``(composition, density)`` with composition ``{symbol: atoms}`` (any
        normalisation) and density in g/cm³.
    materials : mapping, optional
        Extra or replacement entries for `MATERIALS`.

    Returns
    -------
    rho : float
        Atomic number density, atoms/Å³.
    sigma_s, sigma_a : float
        Mean bound scattering cross section (coherent + incoherent) and
        absorption cross section at 2200 m/s, barn per atom, from
        `toscana.isotopes.Element`.

    Raises
    ------
    KeyError
        Unknown material name.
    ValueError
        Unknown element symbol (from `Element`).
    """
    table = dict(MATERIALS, **(materials or {}))
    if isinstance(material, str):
        if material not in table:
            raise KeyError(
                f"Unknown material {material!r}; known: {sorted(table)}. Pass "
                "materials={name: ({symbol: atoms}, density_g_cm3)}."
            )
        composition, density = table[material]
    else:
        composition, density = material
    total = float(sum(composition.values()))
    conc = {s: n / total for s, n in composition.items()}
    elements = {s: Element(s) for s in conc}
    molar = sum(c * elements[s].weight for s, c in conc.items())
    rho = getAtomicDensity(density=float(density), molarMass=molar)
    sigma_s = sum(c * elements[s].sig_sca for s, c in conc.items())
    sigma_a = sum(c * elements[s].sig_abs for s, c in conc.items())
    return rho, sigma_s, sigma_a


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
        Data files on the ANGLE scale (the ``.adat`` files of d4creg),
        overriding those of `m`. Keys: ``"sample"``, ``"container"``,
        ``"vanadium"``, ``"environment"``, ``"background"`` (an empty-
        instrument measurement used as background of all bodies).
    environment : {"background", "furnace", "none"}
        How the environment measurement (``m.envFile``) is used:
        ``"background"`` (default) as background of every measurement, as in
        the CORRECT files written by ToScaNA so far; ``"furnace"`` as a
        furnace (heat shields) with the dimensions and material of
        ``m.environ``, corrected for its own attenuation; ``"none"``.
    placzek : {"series", "diatomic", "none"}
        Placzek inelasticity correction of CORRECT. Default ``"series"``.
    vanadium_smoothing : int
        CORRECT's ``/smoothing`` of the vanadium (0: CORRECT's default).
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
        Hollow sample or hollow vanadium (not handled by CORRECT's
        cylindrical 2-axis model), missing sample file, non-positive beam
        size or radius, unknown `environment` or `placzek`.

    Notes
    -----
    Mapping (Measurement in mm -> pycorrect in cm):

    - sample radius ``OuterDiam/20``, atomic density ``EffAtomicDensity``
      (mass/volume, so the packing fraction is included), temperature
      ``TempK``, fullness ``Fullness``, species with fraction ``conc`` and
      ToScaNA cross sections;
    - container: material ``container[0]``, outer radius ``container[3]/20``
      (its inner radius is the sample radius, as in CORRECT);
    - vanadium: radius ``normaliser[3]/20``, 6.11 g/cm³, ToScaNA's V data;
    - furnace (``environment="furnace"``): radii ``environ[2]/20``,
      ``environ[3]/20``, material ``environ[0]``;
    - beam: height ``beamHeight/10``, width ``beamWidth/10``; wavelength.
    """
    from toscana.corrections.pycorrect.state import CorrectState, Species, Spectrum

    if environment not in ("background", "furnace", "none"):
        raise ValueError("environment must be 'background', 'furnace' or 'none'.")
    placzek_map = {"series": "SERIES_EXPANSION", "diatomic": "DIATOMIC", "none": "NONE"}
    if placzek not in placzek_map:
        raise ValueError("placzek must be 'series', 'diatomic' or 'none'.")
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

    # ---- sample
    if float(m.InnerDiam) > 0:
        raise ValueError("Hollow (annular) samples are not handled by CORRECT's 2-axis model.")
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

    # ---- container
    can_file = files.get("container", m.conFile)
    if _present(m.container[0], can_file):
        st.files["container"] = can_file
        st.ca_r2 = float(m.container[3]) / 20
        st.ca_rho, st.ca_scat_cs, st.ca_abs_cs = material_properties(m.container[0], materials)
    else:
        st.ca_r2 = st.ca_r1

    # ---- vanadium
    van_file = files.get("vanadium", m.norFile)
    if _present("V", van_file):
        if float(m.normaliser[2]) > 0:
            raise ValueError("A hollow vanadium normaliser is not handled by CORRECT.")
        st.files["vanadium"] = van_file
        st.va_r = float(m.normaliser[3]) / 20
        st.va_rho, st.va_scat_cs, st.va_abs_cs = material_properties("V", materials)

    # ---- environment and background
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


@dataclass
class CorrectionResult:
    """
    Result of `run_correction`.

    Attributes
    ----------
    theta, q : ndarray
        Scattering angle (deg) and momentum transfer (1/Å) of the data points.
    dcs, err : ndarray
        Corrected differential cross section per atom and its error, barn/sr.
    factors : dict
        Factors actually used: 'assch','acsch','acch','ahsch','ahch','ahh',
        'avvch' (if vanadium), 'delta_sa', 'delta_va', 'pc_sa', 'pc_va'.
    source : str
        "pycorrect", or "mcams-delta" / "mcams-all" when MCAMS factors were
        used (see `run_correction`).
    state : CorrectState
        The pycorrect input.
    pycorrect : dict
        Full pycorrect result with CORRECT's own factors.
    mcams : dict or None
        MCAMS factors with their statistical errors ('<key>_err'), if used.
    """

    theta: np.ndarray
    q: np.ndarray
    dcs: np.ndarray
    err: np.ndarray
    factors: dict
    source: str
    state: Any
    pycorrect: dict
    mcams: Optional[dict] = field(default=None)


_FACTOR_NAMES = ("assch", "acsch", "acch", "ahsch", "ahch", "ahh", "avvch",
                 "delta_sa", "delta_va", "pc_sa", "pc_va")


def run_correction(
    m: Any,
    *,
    mcams: Optional[str] = None,
    mcams_options: Optional[Mapping[str, Any]] = None,
    workdir: Optional[str] = None,
    verbose: bool = False,
    **state_options: Any,
) -> CorrectionResult:
    """
    Correct the sample data of a Measurement with pycorrect (and MCAMS).

    Parameters
    ----------
    m : Measurement
        Description of the experiment.
    mcams : {None, "delta", "all"}
        None (default): CORRECT's factors. ``"delta"``: replace the
        multiple-scattering ratios of sample and vanadium by MCAMS values
        (angle dependent; CORRECT's series for the higher orders is known to
        be too low, see docs/validation). ``"all"``: replace also the
        Paalman & Pings attenuation factors.
    mcams_options : mapping, optional
        Passed to `toscana.corrections.bridge.mcams_factors_like_pycorrect`
        (``n_particles``, ``n_batches``, ``ms_particles``,
        ``coarse_step_deg``, ``seed``, ``sample_height_cm``).
    workdir : str, optional
        Folder for pycorrect's output file (default: current folder).
    verbose : bool
        Print CORRECT's tables.
    **state_options
        Passed to `build_state` (files, environment, placzek,
        vanadium_smoothing, materials, output).

    Returns
    -------
    CorrectionResult

    Notes
    -----
    pycorrect is run once with its own factors. With `mcams`, MCAMS computes
    the factors on the same set-up (`bridge`), and pycorrect is run again
    with them through ``correct_2axis(..., factors=...)``, so the vanadium
    multiple scattering enters before the vanadium smoothing exactly as in
    CORRECT. The output file then holds the MCAMS-corrected result.
    """
    from toscana.corrections.pycorrect.correct_2axis import correct_2axis

    if mcams not in (None, "delta", "all"):
        raise ValueError("mcams must be None, 'delta' or 'all'.")
    st = build_state(m, **state_options)
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        st.files["output"] = os.path.join(workdir, os.path.basename(st.files["output"]))
    first = correct_2axis(st, verbose=verbose)[0]
    final, mc, source = first, None, "pycorrect"
    if mcams:
        from toscana.corrections import bridge

        mc = bridge.mcams_factors_like_pycorrect(st, first["theta"], **dict(mcams_options or {}))
        keys = ["delta_sa"] + (["delta_va"] if st.has("vanadium") else [])
        if mcams == "all":
            keys += [k for k in ("assch", "acsch", "acch", "ahsch", "ahch", "ahh") if k in mc]
            if st.has("vanadium") and not (st.va_furnace or st.va_container):
                keys.append("avvch")
        final = correct_2axis(st, verbose=verbose, factors={k: mc[k] for k in keys})[0]
        source = f"mcams-{mcams}"
    factors = {k: final[k] for k in _FACTOR_NAMES if k in final}
    return CorrectionResult(
        theta=np.asarray(final["theta"], float), q=np.asarray(final["q"], float),
        dcs=np.asarray(final["dcs"], float), err=np.asarray(final["err"], float),
        factors=factors, source=source, state=st, pycorrect=first, mcams=mc,
    )


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
        From `run_correction`.
    m : Measurement
        For the composition ``m.conc``.
    q_grid : (qmin, qmax, dq), optional
        Rebin onto a regular Q grid (`toscana.numerics.binning.rebin`, each
        point a rectangle of the angular step converted to ΔQ). Default: the
        points of the data, sorted by Q.

    Returns
    -------
    ndarray, shape (n, 3)
        Columns Q (1/Å), S(Q), error: the input of ``sineFT(...,
        with_errors=True)``.

    Notes
    -----
    The self scattering is the static (bound) value; with
    ``placzek="none"`` the inelasticity is not corrected and appears in S(Q).
    For a sample with <b> ≈ 0 (null alloys) S(Q) is not defined: an error is
    raised.
    """
    from toscana.numerics.binning import rebin

    conc = {s: float(c) for s, c in m.conc.items()}
    total = sum(conc.values())
    elements = {s: Element(s) for s in conc}
    self_term = sum(c / total * elements[s].sig_sca for s, c in conc.items()) / (4 * math.pi)
    mean_b = sum(c / total * elements[s].bcoh for s, c in conc.items())  # fm, complex
    b2 = abs(mean_b) ** 2 / 100.0  # barn
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
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):  # rebin prints a debug line
        return rebin(step, float(result.state.wavelength), sq, qmin, qmax, dq)


def save_structure_factor(filename: str, sq: np.ndarray, heading: Sequence[str] = ()) -> None:
    """Write S(Q) (columns Q, S, error) with a header describing the columns."""
    from toscana.dataio.saving import saveFile_3col

    lines = list(heading) + [
        "Columns: Q (1/A), S(Q) (Faber-Ziman, dimensionless), 1-sigma error of S(Q)",
    ]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        saveFile_3col(filename, np.asarray(sq), lines)
