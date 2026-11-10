"""
Geometry of a D4 set-up: concentric cylindrical layers grouped in bodies.

Position in the processing chain
--------------------------------
Describes what is in the beam, for the computation of attenuation factors,
transmission and multiple scattering (`toscana.corrections.factors`) and for
the Paalman & Pings subtraction (`toscana.workflows.correction`).

Concepts
--------
Layer
    A cylinder or annulus of one material, centred on the beam axis
    (vertical), with inner and outer diameter, height (centred on the beam)
    and optionally a filling (fraction of the beam height covered).
Body
    The layers that are measured together, labelled by one letter as in
    CORRECT: ``"s"`` sample, ``"c"`` container (may have several walls, e.g.
    an annular can), ``"h"`` environment (heater, furnace shields, cryostat
    tails: any number of layers), ``"v"`` vanadium. Paalman & Pings factors
    are given per body.
Configuration
    The bodies present in one measurement, e.g. ``"sch"`` (sample in its
    container in the environment), ``"ch"`` (empty container in the
    environment), ``"h"`` (environment alone), ``"v"`` (vanadium).

Public names
------------
MATERIALS, material_properties
    Container/environment materials and their nuclear properties (ToScaNA's
    isotope table). A material is a name of MATERIALS, ``"formula@density"``
    (e.g. ``"Al@2.70"``, ``"Ti0.676Zr0.324@5.23"``) or a pair
    ``(composition, density)``.
CylLayer, CylinderSetup
    The set-up (layers, beam, wavelength) and its MCAMS sample.
setups_from_measurement(m, ...)
    Sample set-up and vanadium set-up of a `Measurement`, including the extra
    layers of ``<LayNN>`` tags.

Units
-----
Diameters, heights, beam size in mm; densities in g/cm³ (input) and
atoms/Å³; cross sections in barn; μ in 1/mm (MCAMS convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from toscana.experiment.info import VANADIUM_DENSITY_GCC
from toscana.isotopes import Element
from toscana.physics.properties import THERMAL_WAVELENGTH, getAtomicDensity

#: Container / environment materials: (composition {symbol: atoms}, density g/cm³).
#: TiZr is the null-scattering alloy Ti0.676Zr0.324; its density depends on the
#: batch (5.1-5.3 g/cm³). Al is used for cryostat and furnace tails.
MATERIALS: dict[str, tuple[dict[str, float], float]] = {
    "V": ({"V": 1.0}, VANADIUM_DENSITY_GCC),
    "TiZr": ({"Ti": 0.676, "Zr": 0.324}, 5.23),
    "SiO2": ({"Si": 1.0, "O": 2.0}, 2.20),
    "Nb": ({"Nb": 1.0}, 8.57),
    "Al": ({"Al": 1.0}, 2.70),
    "Cu": ({"Cu": 1.0}, 8.96),
}

MaterialSpec = Union[str, tuple]
#: Measurement order of the bodies, from the outside: environment first. The
#: innermost body is the sample, the absorber (``a``, e.g. 10B powder in the
#: container, measured instead of the sample) or the vanadium.
BODY_ORDER = ("h", "c", "s", "a", "v")
ROLE_TO_BODY = {"sample": "s", "container": "c", "environment": "h", "vanadium": "v",
                "absorber": "a"}


def atomic_weight(name: str) -> float:
    """Atomic weight (g/mol) of an element or isotope of ToScaNA's table.

    The table has no weight for some isotopes (e.g. B10, B11, Li7: 0); they
    are then taken from MCAMS's nuclear data (isotope masses, or the mass
    number, error < 0.1 %)."""
    weight = float(Element(name).weight)
    if weight > 0:
        return weight
    import re
    import warnings

    from .mcams.formula import _make_species
    from .mcams.nuclear_data import AtomicData

    match = re.fullmatch(r"([A-Z][a-z]?)(\d+)", name)
    if not match:
        raise ValueError(f"No atomic weight for {name!r} in ToScaNA's table.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(AtomicData(use_periodictable=False).mass(
            _make_species(match.group(1), int(match.group(2)))))


def _composition(material: MaterialSpec, materials: Optional[Mapping[str, tuple]] = None):
    """(composition {table name: atoms}, density g/cm³) of a material spec."""
    table = dict(MATERIALS, **(materials or {}))
    if isinstance(material, str):
        if "@" in material:
            from .mcams.formula import parse_formula

            formula, density = material.split("@", 1)
            comp = {}
            for sp, n in parse_formula(formula.strip()).items():
                name = sp.symbol if sp.mass_number is None else f"{sp.symbol}{sp.mass_number}"
                comp[name] = comp.get(name, 0.0) + float(n)
            return comp, float(density)
        if material not in table:
            raise KeyError(
                f"Unknown material {material!r}; known: {sorted(table)}. Use "
                "'formula@density' (e.g. 'Al@2.70') or materials={name: "
                "({symbol: atoms}, density_g_cm3)}."
            )
        return table[material]
    composition, density = material
    return dict(composition), float(density)


def material_properties(
    material: MaterialSpec, materials: Optional[Mapping[str, tuple]] = None
) -> tuple[float, float, float]:
    """
    Atomic density and mean cross sections of a material.

    Parameters
    ----------
    material : str or (dict, float)
        Name of `MATERIALS` (or of `materials`), ``"formula@density"``
        (formula as in MCAMS: ``"Al2O3@3.95"``, ``"D2O@1.107"``,
        ``"[56Fe]@7.87"``) or ``(composition, density)`` with composition
        ``{symbol: atoms}`` and density in g/cm³.
    materials : mapping, optional
        Extra or replacement entries for `MATERIALS`.

    Returns
    -------
    rho : float
        Atomic number density, atoms/Å³.
    sigma_s, sigma_a : float
        Mean bound scattering cross section (coherent + incoherent) and
        absorption cross section at 2200 m/s, barn per atom (ToScaNA's table).

    Raises
    ------
    KeyError
        Unknown material name.
    ValueError
        Unknown element symbol (from `Element`).
    """
    composition, density = _composition(material, materials)
    total = float(sum(composition.values()))
    conc = {s: n / total for s, n in composition.items()}
    elements = {s: Element(s) for s in conc}
    molar = sum(c * atomic_weight(s) for s, c in conc.items())
    rho = getAtomicDensity(density=float(density), molarMass=molar)
    sigma_s = sum(c * elements[s].sig_sca for s, c in conc.items())
    sigma_a = sum(c * elements[s].sig_abs for s, c in conc.items())
    return rho, sigma_s, sigma_a


@dataclass
class CylLayer:
    """
    One cylindrical layer of a set-up.

    Attributes
    ----------
    name : str
        Label (free text, unique in the set-up).
    body : str
        ``"s"``, ``"c"``, ``"h"``, ``"a"`` (absorber) or ``"v"``.
    rho, sigma_s, sigma_a : float
        Atomic density (atoms/Å³), bound scattering and absorption (2200 m/s)
        cross sections (barn per atom).
    inner_diam, outer_diam : float
        Diameters in mm (inner 0 for a full cylinder).
    height : float
        Height of the material in mm, centred on the beam.
    fullness : float
        Fraction of the BEAM height covered by the material, from the bottom
        of the layer up (1 = the whole beam, the usual case). The material
        then spans from ``-height/2`` to ``-beam_height/2 + fullness *
        beam_height`` (CORRECT's "fullness" of a partially filled can).
    composition : dict
        ``{element: concentration}`` (only needed for the sample: Placzek
        correction and S(Q) normalisation).
    """

    name: str
    body: str
    rho: float
    sigma_s: float
    sigma_a: float
    inner_diam: float
    outer_diam: float
    height: float
    fullness: float = 1.0
    composition: dict = field(default_factory=dict)

    def mu(self, wavelength: float) -> float:
        """Total linear attenuation coefficient at `wavelength`, 1/mm.

        μ = ρ (σ_s + σ_a λ/1.798 Å); ρσ in atoms/Å³·barn = 1/cm, /10 -> 1/mm."""
        return self.rho * (self.sigma_s + self.sigma_a * wavelength / THERMAL_WAVELENGTH) / 10.0

    def scatter_fraction(self, wavelength: float) -> float:
        """Fraction of interactions that are scattering events."""
        total = self.sigma_s + self.sigma_a * wavelength / THERMAL_WAVELENGTH
        return self.sigma_s / total if total > 0 else 1.0

    def vertical_range(self, beam_height: float) -> tuple[float, float]:
        """(bottom, top) of the material in mm (0 = beam centre)."""
        bottom = -self.height / 2
        top = self.height / 2
        if self.fullness < 1.0:
            top = min(top, -beam_height / 2 + self.fullness * beam_height)
        return bottom, max(bottom, top)


def make_layer(name: str, body: str, material: MaterialSpec, inner_diam: float,
               outer_diam: float, height: float, fullness: float = 1.0,
               materials: Optional[Mapping[str, tuple]] = None) -> CylLayer:
    """`CylLayer` from a material spec (see `material_properties`)."""
    rho, s, a = material_properties(material, materials)
    composition, _ = _composition(material, materials)
    total = sum(composition.values())
    return CylLayer(name, body, rho, s, a, float(inner_diam), float(outer_diam),
                    float(height), float(fullness), {k: v / total for k, v in composition.items()})


@dataclass
class CylinderSetup:
    """
    Concentric cylindrical layers in a rectangular beam.

    Attributes
    ----------
    layers : list of CylLayer
        Any order; sorted by diameter when used. Must not overlap.
    beam_width, beam_height : float
        Beam size, mm (uniform rectangular beam centred on the axis).
    wavelength : float
        Å.
    """

    layers: list
    beam_width: float
    beam_height: float
    wavelength: float

    def __post_init__(self) -> None:
        self.layers = sorted(self.layers, key=lambda l: (l.inner_diam, l.outer_diam))
        for a, b in zip(self.layers, self.layers[1:]):
            if b.inner_diam < a.outer_diam - 1e-9:
                raise ValueError(f"Layers {a.name!r} ({a.inner_diam}-{a.outer_diam} mm) and "
                                 f"{b.name!r} ({b.inner_diam}-{b.outer_diam} mm) overlap.")
        for l in self.layers:
            if l.outer_diam <= l.inner_diam or l.height <= 0:
                raise ValueError(f"Layer {l.name!r}: needs outer > inner diameter and height > 0.")
        if self.beam_width <= 0 or self.beam_height <= 0:
            raise ValueError("Beam width and height must be positive (mm).")

    @property
    def bodies(self) -> list[str]:
        """Bodies present, in measurement order (outside first)."""
        present = {l.body for l in self.layers}
        order = [b for b in BODY_ORDER if b in present]
        return order + sorted(present - set(order))

    def configurations(self) -> list[str]:
        """Measured configurations, from the outermost body inwards.

        E.g. bodies h, c, s -> ["h", "ch", "sch"]; vanadium alone -> ["v"]."""
        bodies = self.bodies
        out = []
        for k in range(1, len(bodies) + 1):
            out.append("".join(reversed(bodies[:k])))
        return out

    def layers_of(self, body: str) -> list:
        return [l for l in self.layers if l.body == body]

    def mean_path(self, body: str) -> float:
        """Exact mean path (mm) of the beam in the material of `body`.

        Averaged over the beam area (also where the beam misses the body):
        n<L> is then the number of atoms per unit beam area, used for the
        normalisation to vanadium."""
        return sum(_mean_chord(l, self.beam_width, self.beam_height) for l in self.layers_of(body))

    def atoms_per_area(self, body: str) -> float:
        """Σ ρ <L> over the layers of `body` (atoms/Å³ · mm)."""
        return sum(l.rho * _mean_chord(l, self.beam_width, self.beam_height)
                   for l in self.layers_of(body))

    def scattering_power(self, body: str) -> float:
        """Σ ρ σ_s <L> over the layers of `body` (barn · atoms/Å³ · mm).

        Divided by 4π, the single-scattering intensity of `body` if it were
        an isotropic scatterer, per unit flux and before attenuation: the
        reference of the multiple-scattering ratios δ."""
        return sum(l.rho * l.sigma_s * _mean_chord(l, self.beam_width, self.beam_height)
                   for l in self.layers_of(body))

    def mcams_sample(self, present: Optional[str] = None):
        """MCAMS `CylindricalSample` of the set-up.

        Layers of bodies not in `present` (a configuration string) keep their
        geometry with μ = 0. Gaps between layers become empty layers."""
        from .mcams.cylinder import create_cylindrical_sample
        from .mcams.sample_base import Layer

        present = present if present is not None else "".join(self.bodies)
        lam, hb = self.wavelength, self.beam_height
        items, names, r = [], [], 0.0
        for l in self.layers:
            if l.inner_diam > r * 2 + 1e-9:                      # gap -> empty layer
                thick = l.inner_diam if not items else (l.inner_diam - 2 * r) / 2
                items.append(Layer(f"gap{len(items)}", thick, 0.0, 1.0))
                names.append(None)
                r = l.inner_diam / 2
            thick = l.outer_diam if not items else (l.outer_diam - l.inner_diam) / 2
            bottom, top = l.vertical_range(hb)
            mu = l.mu(lam) if l.body in present else 0.0
            items.append(Layer(l.name, thick, mu, l.scatter_fraction(lam),
                               bottom=bottom, height=top - bottom))
            names.append(l.name)
            r = l.outer_diam / 2
        height = max(l.height for l in self.layers)
        sample = create_cylindrical_sample(height, items, y_bottom=-height / 2, beam_height=hb)
        return sample, names

    def transmission(self, present: Optional[str] = None, n_x: int = 20001) -> float:
        """Exact fraction of the beam crossing the set-up without interaction.

        Average over the uniform rectangular beam of exp(-Σ μ_k L_k(x, y)),
        L_k the chord of layer k at lateral offset x (where layer k has
        material at height y). Bodies not in `present` are transparent."""
        present = present if present is not None else "".join(self.bodies)
        lam, w, hb = self.wavelength, self.beam_width, self.beam_height
        active = [l for l in self.layers if l.body in present]
        x = np.linspace(-w / 2, w / 2, n_x)
        cuts = {-hb / 2, hb / 2}
        for l in active:
            for v in l.vertical_range(hb):
                if -hb / 2 < v < hb / 2:
                    cuts.add(v)
        ys = sorted(cuts)
        total = 0.0
        for y0, y1 in zip(ys, ys[1:]):
            yc = (y0 + y1) / 2
            tau = np.zeros_like(x)
            for l in active:
                b, t = l.vertical_range(hb)
                if b <= yc <= t:
                    tau += l.mu(lam) * _chord(l, x)
            total += (y1 - y0) * np.trapezoid(np.exp(-tau), x) / w
        return float(total / hb)


def _chord(l: CylLayer, x: np.ndarray) -> np.ndarray:
    """Path (mm) along the beam through annulus `l` at lateral offset x."""
    ro, ri = l.outer_diam / 2, l.inner_diam / 2
    return (2 * np.sqrt(np.clip(ro * ro - x * x, 0, None))
            - 2 * np.sqrt(np.clip(ri * ri - x * x, 0, None)))


def _mean_chord(l: CylLayer, beam_width: float, beam_height: float) -> float:
    """Mean path of the beam in layer `l` (mm), averaged over the beam area."""
    x = np.linspace(-beam_width / 2, beam_width / 2, 20001)
    b, t = l.vertical_range(beam_height)
    covered = max(0.0, min(t, beam_height / 2) - max(b, -beam_height / 2)) / beam_height
    return float(np.trapezoid(_chord(l, x), x) / beam_width * covered)


def _role_material(name: Any) -> bool:
    return bool(str(name).strip()) and str(name).strip().lower() != "none"


def setups_from_measurement(
    m: Any,
    *,
    environment: str = "background",
    vanadium_in_environment: bool = False,
    materials: Optional[Mapping[str, tuple]] = None,
) -> tuple[CylinderSetup, CylinderSetup]:
    """
    Sample set-up and vanadium set-up of a `Measurement`.

    Parameters
    ----------
    m : toscana.experiment.Measurement
    environment : {"background", "layer", "none"}
        ``"layer"`` (or ``"furnace"``): the environment ``<Eshape>``/``<Emater>``
        and the extra layers with role ``environment`` form body ``"h"``.
        Otherwise the environment is not part of the set-up (its measurement
        is then used as background, see `run_correction`).
    vanadium_in_environment : bool
        The vanadium was measured inside the environment (body ``"h"`` is
        then also part of the vanadium set-up).
    materials : mapping, optional
        Extra entries for `MATERIALS`.

    Returns
    -------
    (sample_setup, vanadium_setup)

    Notes
    -----
    Sample: ``<Sshape>`` diameters (a non-zero inner diameter gives a hollow
    sample), height, ``<Sfulln>``, composition and EFFECTIVE atomic density
    (mass/volume, so the packing fraction is included). Container:
    ``<Cshape>``/``<Cmater>`` (inner and outer diameter: annulus). Vanadium:
    ``<Nshape>``. Extra layers: ``<LayNN> name role material inner_d outer_d
    height`` with role sample, container or environment (e.g. the inner wall
    of an annular can, a second heat shield).
    """
    if environment == "furnace":
        environment = "layer"
    if environment not in ("background", "layer", "none"):
        raise ValueError("environment must be 'background', 'layer' (or 'furnace') or 'none'.")
    lam = float(m.wavelength)
    bw, bh = float(m.beamWidth), float(m.beamHeight)
    layers = []
    comp = {s: float(c) for s, c in m.conc.items()}
    sig_s = sum(c * Element(s).sig_sca for s, c in comp.items())
    sig_a = sum(c * Element(s).sig_abs for s, c in comp.items())
    layers.append(CylLayer("sample", "s", float(m.EffAtomicDensity), sig_s, sig_a,
                           float(m.InnerDiam), float(m.OuterDiam), float(m.Height),
                           float(m.Fullness), comp))
    if _role_material(m.container[0]) and str(getattr(m, "conFile", "")).strip():
        layers.append(make_layer("container", "c", m.container[0], m.container[2],
                                 m.container[3], m.container[4], materials=materials))
    if environment == "layer" and _role_material(m.environ[0]):
        layers.append(make_layer("environment", "h", m.environ[0], m.environ[2],
                                 m.environ[3], m.environ[4], materials=materials))
    for extra in getattr(m, "extra_layers", []):
        body = ROLE_TO_BODY.get(extra["role"].lower())
        if body is None or body == "v":
            raise ValueError(f"<Lay> {extra['name']!r}: role must be sample, container or environment.")
        if body == "h" and environment != "layer":
            continue
        layers.append(make_layer(extra["name"], body, extra["material"], extra["inner_diam"],
                                 extra["outer_diam"], extra["height"], materials=materials))
    sample_setup = CylinderSetup(layers, bw, bh, lam)

    vlayers = [make_layer("vanadium", "v", "V", m.normaliser[2], m.normaliser[3],
                          m.normaliser[4], materials=materials)]
    if vanadium_in_environment:
        vlayers += [replace(l) for l in layers if l.body == "h"]
    vanadium_setup = CylinderSetup(vlayers, bw, bh, lam)
    return sample_setup, vanadium_setup


def absorber_setup_from_measurement(
    m: Any,
    *,
    environment: str = "layer",
    materials: Optional[Mapping[str, tuple]] = None,
) -> CylinderSetup:
    """
    Set-up of the absorber measurement: the sample set-up with the sample
    replaced by the absorber (``<Ashape>``, ``<Amater>``; body ``"a"``).

    The absorber (e.g. 10B powder, ``<Amater> [10B]@1.25``) fills the
    container like the sample; its configuration is e.g. ``"ach"``. Used by
    `run_correction` when the absorber measurement is a configuration of the
    Paalman & Pings system (``absorber_model="configuration"``).

    Raises
    ------
    ValueError
        No absorber material in ``<Amater>``.
    """
    if not _role_material(m.absorber[0]):
        raise ValueError("No absorber material (<Amater>).")
    sample_setup, _ = setups_from_measurement(m, environment=environment, materials=materials)
    layers = [l for l in sample_setup.layers if l.body != "s"]
    layers.append(make_layer("absorber", "a", m.absorber[0], m.absorber[2], m.absorber[3],
                             m.absorber[4], materials=materials))
    return CylinderSetup(layers, sample_setup.beam_width, sample_setup.beam_height,
                         sample_setup.wavelength)
