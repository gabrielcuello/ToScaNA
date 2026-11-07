"""
Attenuation (Paalman & Pings), transmission and multiple-scattering factors.

Position in the processing chain
--------------------------------
Input: a `CylinderSetup` (toscana.corrections.setup) and the scattering
angles of the data. Output: a `FactorSet`, used by the Paalman & Pings
subtraction of `toscana.workflows.correction` and available to notebooks,
on the angle or the Q scale.

Definitions
-----------
A_{b,C}(2θ)
    Paalman & Pings factor: attenuation of the radiation singly scattered in
    body b when all the bodies of configuration C are in the beam (e.g.
    ``A_s,sch``, ``A_c,sch``, ``A_h,sch``, ``A_c,ch``, ``A_h,ch``, ``A_h,h``,
    ``A_v,v``). For a body of several layers it is the average over its
    layers weighted by their single-scattering power ρσ_s<L>.
delta_C(2θ)
    Multiple scattering of configuration C relative to its single scattering
    (all orders >= 2): measured = single × (1 + delta_C).
T_C
    Fraction of the beam crossing configuration C without interaction (for
    the absorber background T·I_empty + (1 - T)·I_absorber).

Methods
-------
"mcams" (default)
    Attenuation by MCAMS's direct method (deterministic sampling of the
    scattering points, statistical error ~0.05-0.2 %), at coarse angles,
    interpolated with a cubic spline onto the data grid; multiple scattering
    by MCAMS's Monte Carlo (multiple) divided by the direct single
    scattering; transmission exact. Any number of concentric layers, gaps,
    hollow samples, annular containers, partial filling.
"pycorrect"
    CORRECT 2.24 (Paalman & Pings numerical integration; Blech & Averbach δ,
    independent of angle). Only a solid sample, at most one container wall
    in contact with it and one environment layer; vanadium alone and solid.

Public names
------------
FactorSet, compute_factors, load_factors
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .setup import CylinderSetup

__all__ = ["FactorSet", "compute_factors", "load_factors"]


@dataclass
class FactorSet:
    """
    Factors of one set-up on a grid of scattering angles.

    Attributes
    ----------
    angles : ndarray
        2θ in degrees (the grid of the data).
    wavelength : float
        Å (for the Q scale).
    configurations : list of str
        E.g. ``["h", "ch", "sch"]``.
    A, A_err : dict
        ``"A_<body>,<config>"`` -> array over `angles` (and its statistical
        error; zero for the pycorrect method).
    delta, delta_err : dict
        ``"delta_<config>"`` -> array over `angles`.
    T : dict
        ``"T_<config>"`` -> transmission (float).
    atoms_per_area : dict
        ``body`` -> Σ ρ<L> (atoms/Å³ · mm), for the normalisation.
    method : str
        "mcams" or "pycorrect".
    info : dict
        Settings of the calculation (and binned Monte Carlo δ).
    scattering_power : dict
        ``body`` -> Σ ρ σ_s <L> (barn · atoms/Å³ · mm). The multiple
        scattering of configuration C, in the units of the vanadium-derived
        constant K = I_v / (Σρσ<L>_v / 4π), is
        δ_C K Σ_b A_b,C Σρσ<L>_b / 4π (δ is defined for isotropic
        scatterers: total multiple / total single scattering).
    """

    angles: np.ndarray
    wavelength: float
    configurations: list
    A: dict
    A_err: dict
    delta: dict
    delta_err: dict
    T: dict
    atoms_per_area: dict
    method: str
    info: dict = field(default_factory=dict)
    scattering_power: dict = field(default_factory=dict)

    @property
    def q(self) -> np.ndarray:
        """Q = 4π/λ sin(θ), 1/Å."""
        return 4 * math.pi / self.wavelength * np.sin(np.radians(self.angles) / 2)

    def columns(self) -> list[str]:
        """Names of all the angle-dependent quantities."""
        return list(self.A) + list(self.delta)

    def table(self, scale: str = "angle") -> tuple[np.ndarray, dict]:
        """(x, {name: values}) with x = 2θ (deg) or Q (1/Å), increasing."""
        if scale not in ("angle", "q"):
            raise ValueError("scale must be 'angle' or 'q'.")
        x = self.angles if scale == "angle" else self.q
        cols = dict(self.A)
        cols.update(self.delta)
        cols.update({f"{k}_err": v for k, v in self.A_err.items()})
        cols.update({f"{k}_err": v for k, v in self.delta_err.items()})
        order = np.argsort(x)
        return x[order], {k: np.asarray(v)[order] for k, v in cols.items()}

    def save(self, filename: str, scale: str = "angle", heading: Sequence[str] = ()) -> None:
        """Write the factors as a text table (one row per angle or Q).

        Header: user `heading`, method, wavelength, transmissions, atoms per
        area, then the column names. Reload with `load_factors`."""
        x, cols = self.table(scale)
        names = list(cols)
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(f"# {filename}\n")
            for line in heading:
                fh.write(f"# {line}\n")
            fh.write(f"# method: {self.method}; wavelength = {self.wavelength} A\n")
            fh.write("# A_b,C: Paalman & Pings factor of body b in configuration C "
                     "(s sample, c container, h environment, v vanadium)\n")
            fh.write("# delta_C: multiple / single scattering of configuration C, all bodies, "
                     "isotropic scatterers\n")
            fh.write("#   MS_C = delta_C * K * sum_b A_b,C * P_b / 4pi, "
                     "K = I_v / (P_v / 4pi)\n")
            for k, v in self.T.items():
                fh.write(f"# {k} = {v:.6f}  (fraction of the beam transmitted)\n")
            for k, v in self.atoms_per_area.items():
                fh.write(f"# n<L>_{k} = {v:.8g}  (atoms/A^3 * mm)\n")
            for k, v in self.scattering_power.items():
                fh.write(f"# P_{k} = {v:.8g}  (sum rho sigma_s <L>, barn * atoms/A^3 * mm)\n")
            xname = "2theta(deg)" if scale == "angle" else "Q(1/A)"
            fh.write("# columns: " + " ".join([xname] + names) + "\n")
            for i in range(len(x)):
                fh.write(f"{x[i]:12.7f} " + " ".join(f"{cols[n][i]:12.7f}" for n in names) + "\n")


def load_factors(filename: str) -> tuple[np.ndarray, dict, dict]:
    """Read a file written by `FactorSet.save`.

    Returns (x, columns {name: array}, scalars {"T_...", "n<L>_...": float})."""
    names, scalars = None, {}
    with open(filename, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if text.startswith("columns:"):
                names = text.split(":", 1)[1].split()
            elif text.startswith(("T_", "n<L>_")) and "=" in text:
                key, rest = text.split("=", 1)
                scalars[key.strip()] = float(rest.split()[0])
    data = np.loadtxt(filename, ndmin=2)
    return data[:, 0], {n: data[:, i + 1] for i, n in enumerate(names[1:])}, scalars


# ----------------------------------------------------------------------------
def _interp(coarse: np.ndarray, values: np.ndarray, fine: np.ndarray) -> np.ndarray:
    from scipy.interpolate import CubicSpline

    ok = np.isfinite(values)
    if ok.sum() < 4:
        return np.interp(fine, coarse[ok], values[ok])
    return CubicSpline(coarse[ok], values[ok])(fine)


def compute_factors(
    setup: CylinderSetup,
    angles_deg: Sequence[float],
    *,
    method: str = "mcams",
    configurations: Optional[Sequence[str]] = None,
    multiple_scattering: bool = True,
    n_points: int = 100_000,
    ms_particles: int = 2_000_000,
    step_deg: float = 5.0,
    ms_fit_degree: Optional[int] = 3,
    seed: int = 1,
) -> FactorSet:
    """
    Paalman & Pings factors, transmissions and multiple scattering.

    Parameters
    ----------
    setup : CylinderSetup
        Sample set-up or vanadium set-up.
    angles_deg : array
        2θ of the data (deg). The output is on this grid.
    method : {"mcams", "pycorrect"}
    configurations : list of str, optional
        Default: all measured configurations of the set-up
        (`CylinderSetup.configurations`).
    multiple_scattering : bool
        Compute delta_C (MCAMS Monte Carlo; the slowest part).
    n_points : int
        MCAMS direct method: points per layer (error ∝ 1/sqrt(n_points)).
    ms_particles : int
        MCAMS Monte Carlo: incident particles per configuration (2e6: about
        2 s, a few hundred multiply scattered particles per 5° bin).
    step_deg : float
        Coarse angular step of the MCAMS calculation: attenuation at every
        step/2 (cubic spline onto the data grid), multiple scattering in bins
        of `step_deg`.
    ms_fit_degree : int or None
        delta_C is fitted with a polynomial of this degree in 2θ (weighted by
        the Monte Carlo errors), since multiple scattering varies smoothly
        with angle; None keeps the binned values (linear interpolation). The
        binned values are kept in ``FactorSet.info["delta_binned"]``.
    seed : int
        Random seed (results are reproducible).

    Returns
    -------
    FactorSet

    Raises
    ------
    ValueError
        Unknown method; set-up not supported by pycorrect.
    """
    angles = np.asarray(angles_deg, dtype=float)
    configs = list(configurations or setup.configurations())
    if method == "pycorrect":
        return _pycorrect_factors(setup, angles, configs)
    if method != "mcams":
        raise ValueError("method must be 'mcams' or 'pycorrect'.")
    from .mcams.direct_attenuation import attenuation_factors_direct
    from .mcams.multiple_scattering import simulate_scattering_orders

    lo = max(0.0, step_deg * math.floor(angles.min() / step_deg) - step_deg)
    hi = min(180.0, step_deg * math.ceil(angles.max() / step_deg) + step_deg)
    edges = np.arange(lo, hi + step_deg / 2, step_deg)
    centres = (edges[:-1] + edges[1:]) / 2
    coarse = np.unique(np.concatenate([edges, centres]))
    A, A_err, delta, delta_err, T, binned = {}, {}, {}, {}, {}, {}
    for cfg in configs:
        sample, names = setup.mcams_sample(cfg)
        res = attenuation_factors_direct(sample, setup.beam_width, setup.beam_height,
                                         angles_deg=coarse, n_points=n_points, rng=seed)
        for body in cfg:
            idx = [i for i, n in enumerate(names) if n and n in {l.name for l in setup.layers_of(body)}]
            w = np.array([res.scatter_fractions[i] * res.mus[i] * res.mean_path[i] for i in idx])
            f = np.array([res.factors[i] for i in idx])
            e = np.array([res.errors[i] for i in idx])
            fb = (w[:, None] * f).sum(0) / w.sum()
            eb = np.sqrt(((w[:, None] * e) ** 2).sum(0)) / w.sum()
            A[f"A_{body},{cfg}"] = _interp(coarse, fb, angles)
            A_err[f"A_{body},{cfg}"] = np.interp(angles, coarse, eb)
        T[f"T_{cfg}"] = setup.transmission(cfg)
        if multiple_scattering:
            # delta = multiple (Monte Carlo) / single (exact, direct method,
            # same run as the factors: its values at the bin centres)
            ms = simulate_scattering_orders(sample, setup.beam_width, setup.beam_height,
                                            n_particles=ms_particles, angle_edges_deg=edges,
                                            rng=seed)
            single = res.single_per_steradian()[np.searchsorted(coarse, ms.angles_deg)]
            with np.errstate(divide="ignore", invalid="ignore"):
                d = ms.per_steradian(ms.multiple) / single
                de = d / np.sqrt(np.maximum(ms.multiple, 1.0))
            binned.setdefault(cfg, (ms.angles_deg, d, de))
            ok = np.isfinite(d) & (ms.multiple > 0)
            if ms_fit_degree is not None and ok.sum() > ms_fit_degree + 2:
                coef, cov = np.polyfit(ms.angles_deg[ok], d[ok], ms_fit_degree,
                                       w=1 / de[ok], cov="unscaled")
                delta[f"delta_{cfg}"] = np.polyval(coef, angles)
                basis = np.vander(angles, ms_fit_degree + 1)
                delta_err[f"delta_{cfg}"] = np.sqrt(np.einsum("ij,jk,ik->i", basis, cov, basis))
            else:
                delta[f"delta_{cfg}"] = np.interp(angles, ms.angles_deg[ok], d[ok])
                delta_err[f"delta_{cfg}"] = np.interp(angles, ms.angles_deg[ok], de[ok])
    atoms = {b: setup.atoms_per_area(b) for b in setup.bodies}
    power = {b: setup.scattering_power(b) for b in setup.bodies}
    return FactorSet(angles, setup.wavelength, configs, A, A_err, delta, delta_err, T, atoms,
                     "mcams", {"n_points": n_points, "ms_particles": ms_particles,
                               "step_deg": step_deg, "seed": seed, "ms_fit_degree": ms_fit_degree,
                               "delta_binned": binned}, power)


def pycorrect_supports(setup: CylinderSetup) -> tuple[bool, str]:
    """Whether CORRECT's cylindrical model can describe `setup` (and why not)."""
    bodies = setup.bodies
    if "v" in bodies:
        if bodies != ["v"] or len(setup.layers) != 1 or setup.layers[0].inner_diam > 0:
            return False, "CORRECT handles a solid vanadium rod measured alone."
        return True, ""
    s = setup.layers_of("s")
    if len(s) != 1 or s[0].inner_diam > 0:
        return False, "CORRECT needs one solid sample cylinder (no hollow sample)."
    if s[0].fullness < 1:
        return False, "Use MCAMS for partial filling (CORRECT's fullness step is approximate)."
    c = setup.layers_of("c")
    if len(c) > 1 or (c and abs(c[0].inner_diam - s[0].outer_diam) > 1e-6):
        return False, "CORRECT needs one container wall touching the sample (no annular can, no gap)."
    if len(setup.layers_of("h")) > 1:
        return False, "CORRECT handles one environment (furnace) layer."
    if set(bodies) - {"s", "c", "h"}:
        return False, "Unknown bodies for CORRECT."
    return True, ""


def _pycorrect_factors(setup: CylinderSetup, angles: np.ndarray, configs: list) -> FactorSet:
    from .pycorrect.correct_2axis import absor
    from .pycorrect.ndplib import Collimator, mscat

    ok, why = pycorrect_supports(setup)
    if not ok:
        raise ValueError(f"pycorrect cannot describe this set-up: {why} Use method='mcams'.")
    lam = setup.wavelength
    mu = {b: sum(l.mu(lam) for l in setup.layers_of(b)) * 10 for b in setup.bodies}  # 1/cm
    smu = {b: sum(l.rho * l.sigma_s for l in setup.layers_of(b)) for b in setup.bodies}
    width_cm, height_cm = setup.beam_width / 10, setup.beam_height / 10
    A, delta, T = {}, {}, {}
    if setup.bodies == ["v"]:
        r = setup.layers[0].outer_diam / 20
        f = absor("CYLINDRICAL", 0.0, r, 0.0, 0.0, 0.0, width_cm, mu["v"], 0.0, 0.0, angles,
                  Collimator(), False)
        A["A_v,v"] = f["assch"]
        delta["delta_v"] = np.full_like(angles, mscat(r, height_cm, mu["v"], smu["v"], verbose=False))
    else:
        r1 = setup.layers_of("s")[0].outer_diam / 20
        c = setup.layers_of("c")
        r2 = c[0].outer_diam / 20 if c else r1
        h = setup.layers_of("h")
        r3, r4 = (h[0].inner_diam / 20, h[0].outer_diam / 20) if h else (0.0, 0.0)
        # one CORRECT call gives the factors of all configurations
        f = absor("CYLINDRICAL", 0.0, r1, r2, r3, r4, width_cm, mu.get("s", 0.0),
                  mu.get("c", 0.0), mu.get("h", 0.0), angles, Collimator(), False)
        keys = {("s", 3): "assch", ("c", 3): "acsch", ("h", 3): "ahsch",
                ("c", 2): "acch", ("h", 2): "ahch", ("h", 1): "ahh"}
        for cfg in configs:
            for body in cfg:
                # CORRECT's names: which bodies are present (sample set-up = 3)
                level = 3 if "s" in cfg else (2 if "c" in cfg else 1)
                if (body, level) not in keys:
                    raise ValueError(f"pycorrect has no factor for body {body!r} in {cfg!r}.")
                A[f"A_{body},{cfg}"] = f[keys[(body, level)]]
        fs = (r1 / r2) ** 2
        for cfg in configs:
            if "s" in cfg:
                tmu = (1 - fs) * mu.get("c", 0.0) + fs * mu["s"] if "c" in cfg else mu["s"]
                sm = (1 - fs) * smu.get("c", 0.0) + fs * smu["s"] if "c" in cfg else smu["s"]
                delta[f"delta_{cfg}"] = np.full_like(angles, mscat(r2, height_cm, tmu, sm, verbose=False))
    for cfg in configs:
        T[f"T_{cfg}"] = setup.transmission(cfg)
    atoms = {b: setup.atoms_per_area(b) for b in setup.bodies}
    zeros = {k: np.zeros_like(v) for k, v in A.items()}
    dz = {k: np.zeros_like(v) for k, v in delta.items()}
    power = {b: setup.scattering_power(b) for b in setup.bodies}
    return FactorSet(angles, lam, configs, A, zeros, delta, dz, T, atoms, "pycorrect", {}, power)
