"""
layer_builder.py - Prepare the input of a Layer from the chemical formula,
the density and the cross sections of the material.

A layer of the sample is described by four numbers/labels:
    (label, thickness in mm, mu in 1/mm, scatter fraction)
(see sample_base.Layer). This module computes `mu` and the scatter fraction
from what you usually know about the material.

The physics (it is short)
-------------------------
1. The formula gives the number of atoms of each species in one FORMULA UNIT
   (for instance one H2O molecule), and so its molar mass M (g/mol).
2. The density rho (g/cm^3) gives the number of formula units per cm^3:
        n = rho * N_A / M          (N_A = 6.02214076e23 per mol)
3. The cross section of a formula unit, sigma (barn = 1e-24 cm^2), is the sum
   of the cross sections of its atoms. The macroscopic cross section, the
   linear attenuation coefficient, is
        mu = n * sigma             (in 1/cm; divide by 10 for 1/mm)
   and it is computed separately for scattering and absorption:
        mu_scat = n * sigma_scat ,   mu_abs = n * sigma_abs ,
        mu = mu_scat + mu_abs      (total, the one used by the simulations)
4. The scatter fraction is the part of the interactions that are scattering:
        scatter_fraction = sigma_scat / (sigma_scat + sigma_abs)

Where the cross sections come from (your choice)
------------------------------------------------
 A. YOUR OWN values for the whole formula unit:
        sigma_scat=..., sigma_abs=...      (barn per formula unit)
 B. YOUR OWN values per species (barn per atom), for some or all of them:
        cross_sections={"Fe": (11.62, 2.56), "[56Fe]": (...), "D": (...)}
    Keys are written like the species of the formula without brackets
    ("Fe", "56Fe", "D" or "2H"); the value is (scattering, absorption).
 C. TABULATED values (by default ToScaNA's isotope table, ToscanaData in
    nuclear_data.py; periodictable with data=AtomicData()) for all
    the species that you did not provide in B.
Rule: if sigma_scat and sigma_abs are given, A is used for everything.
Otherwise every species takes B if present and C if not.
Tabulated absorption cross sections are given at 2200 m/s (1.798 Angstrom);
with `wavelength_A=...` they are scaled with the 1/v law (proportional to the
wavelength), which fails for strong resonances (Cd, Sm, Eu, Gd, ...). Your
own values (A and B) are never modified.

Atomic masses: from `masses={"Fe": 55.845}` if given, else from the tables.

Isotopes in the formula: see formula.py, e.g. "D2O" or "[2H]2O", "[10B]4C".

Main function
-------------
    layer = make_layer_tuple("water", 2.0, "H2O", 1.0,
                             sigma_scat=168.3, sigma_abs=0.665)
    # -> ("water", 2.0, 0.565..., 0.996...)  ready for create_slab_sample,
    #    create_cylindrical_sample or make_sample

For the details of the calculation (molar mass, number density, mu of
scattering and absorption) use `compute_layer_properties`.

Plain Python is enough for this module (no NumPy needed).
"""

from dataclasses import dataclass

from .formula import parse_formula, parse_species, format_composition
from .sample_base import Layer
from .nuclear_data import AtomicData, TableError, ToscanaData

# Physical constants and unit conversions
AVOGADRO = 6.02214076e23          # atoms (or molecules) per mole (exact)
BARN_TO_CM2 = 1.0e-24             # 1 barn in cm^2
CM_TO_MM = 0.1                    # a quantity per cm is 0.1 times that per mm
THERMAL_WAVELENGTH_A = 1.798      # wavelength of 2200 m/s neutrons (Angstrom)


# --------------------------------------------------------------------------
# The detailed result
# --------------------------------------------------------------------------
@dataclass
class LayerProperties:
    """Everything computed for one layer.

    label, thickness_mm, formula, density_g_cm3 : the inputs
    composition          : {Species: atoms per formula unit}
    molar_mass           : g/mol of one formula unit
    formula_units_per_cm3: number of formula units per cm^3
    sigma_scat_barn, sigma_abs_barn : cross sections per formula unit (barn)
    mu_scat, mu_abs, mu  : linear attenuation coefficients (1/mm):
                           scattering, absorption and total
    scatter_fraction     : mu_scat / mu
    sources              : text saying where the cross sections came from
    """
    label: str
    thickness_mm: float
    formula: str
    density_g_cm3: float
    composition: dict
    molar_mass: float
    formula_units_per_cm3: float
    sigma_scat_barn: float
    sigma_abs_barn: float
    mu_scat: float
    mu_abs: float
    mu: float
    scatter_fraction: float
    sources: str

    def as_tuple(self, fill_height=None):
        """(label, thickness in mm, mu in 1/mm, scatter fraction): the input
        of create_slab_sample / create_cylindrical_sample / make_sample.
        With `fill_height` (mm from the bottom of the sample) a 5th element
        is added: the layer is then filled only up to that height."""
        layer = (self.label, self.thickness_mm, self.mu, self.scatter_fraction)
        return layer if fill_height is None else layer + (fill_height,)

    def as_layer(self, fill_height=None, fullness=None, bottom=None, height=None):
        """The same data as a `Layer` object, optionally partially filled:
        give `bottom` and/or `height` (mm, from the centre of the beam;
        version 2.2, recommended), OR `fill_height` (mm from the bottom of
        the sample) OR `fullness` (fraction of the beam height covered by the
        material)."""
        return Layer(self.label, self.thickness_mm, self.mu,
                     self.scatter_fraction, fill_height, fullness, bottom, height)

    def summary(self):
        """Readable description of the calculation."""
        return "\n".join([
            f"Layer '{self.label}': {self.formula}  "
            f"({format_composition(self.composition)})",
            f"  density            {self.density_g_cm3:g} g/cm3, "
            f"molar mass {self.molar_mass:.4f} g/mol",
            f"  formula units/cm3  {self.formula_units_per_cm3:.5e}",
            f"  sigma_scat {self.sigma_scat_barn:.5g} b, "
            f"sigma_abs {self.sigma_abs_barn:.5g} b (per formula unit)",
            f"  mu_scat {self.mu_scat:.5g}/mm, mu_abs {self.mu_abs:.5g}/mm, "
            f"mu {self.mu:.5g}/mm",
            f"  scatter fraction   {self.scatter_fraction:.5f}",
            f"  cross sections: {self.sources}"])


# --------------------------------------------------------------------------
# Helper: user dictionaries with keys written as "Fe", "56Fe", "D", ...
# --------------------------------------------------------------------------
def _species_dict(user_dict):
    """Convert {"Fe": value, "56Fe": value, ...} into {Species: value}."""
    return {parse_species(key): value for key, value in (user_dict or {}).items()}


# --------------------------------------------------------------------------
# Main calculation
# --------------------------------------------------------------------------
def compute_layer_properties(label, thickness, formula, density,
                             sigma_scat=None, sigma_abs=None,
                             cross_sections=None, masses=None,
                             wavelength_A=None, data=None):
    """
    Compute the attenuation coefficient and the scatter fraction of a layer.

    Parameters
    ----------
    label : str
        Name of the layer (free text).
    thickness : float
        Thickness in mm (for the innermost layer of a cylindrical sample: the
        DIAMETER, as in the sample modules).
    formula : str
        Chemical formula, with the notation of formula.py (isotopes in
        brackets, e.g. "H2O", "Fe2O3Na2Li", "[2H]2O", "Ca(OH)2").
    density : float
        Density of the layer in g/cm^3 (the global density of the material,
        porosity included). 0 is allowed (empty layer).
    sigma_scat, sigma_abs : float, optional
        Cross sections in barn PER FORMULA UNIT (option A). Give both or none.
    cross_sections : dict, optional
        {species: (sigma_scat, sigma_abs)} in barn PER ATOM (option B).
    masses : dict, optional
        {species: g/mol} to use instead of the tabulated masses.
    wavelength_A : float, optional
        Neutron wavelength in Angstrom: scales the TABULATED absorption
        (1/v law). None: values at 2200 m/s are used as they are.
    data : object, optional
        Source of tabulated values (see nuclear_data). Default:
        ToscanaData(), ToScaNA's isotope table (integration into ToScaNA,
        2026-11; the stand-alone MCAMS used AtomicData(), i.e. the optional
        package periodictable). Pass data=AtomicData() to use periodictable.

    Returns
    -------
    LayerProperties   (use .as_tuple() for the Layer input)
    """
    # --- Step 0: check the inputs -----------------------------------------------
    if thickness <= 0:
        raise ValueError("The thickness must be positive (mm).")
    if density < 0:
        raise ValueError("The density cannot be negative.")
    if (sigma_scat is None) != (sigma_abs is None):
        raise ValueError("Give both sigma_scat and sigma_abs, or none of them.")
    if sigma_scat is not None and cross_sections:
        raise ValueError("Give either sigma_scat/sigma_abs (whole formula "
                         "unit) or cross_sections (per atom), not both.")
    if sigma_scat is not None and (sigma_scat < 0 or sigma_abs < 0):
        raise ValueError("Cross sections cannot be negative.")
    if data is None:
        data = ToscanaData()

    # --- Step 1: read the formula: {species: atoms per formula unit} ----------
    composition = parse_formula(formula)

    # --- Step 2: molar mass of the formula unit -------------------------------
    # Sum over the species of (atoms) x (mass of one mole of those atoms);
    # your own masses (if any) take precedence over the tables.
    user_masses = _species_dict(masses)

    def mass_of(species):
        """Mass in g/mol of one species: yours if given, else the tables'."""
        return user_masses[species] if species in user_masses else data.mass(species)

    molar_mass = sum(amount * mass_of(sp) for sp, amount in composition.items())

    # --- Step 3: formula units per cm^3  n = rho * N_A / M ---------------------
    units_per_cm3 = density * AVOGADRO / molar_mass

    # --- Step 4: cross sections per formula unit ------------------------------
    if sigma_scat is not None:
        # Option A: the user gives the values of the whole formula unit.
        scat_fu, abs_fu = float(sigma_scat), float(sigma_abs)
        sources = "given by the user for the whole formula unit"
    else:
        # Options B and C: add up the atoms, species by species.
        user_xs = _species_dict(cross_sections)
        scat_fu = abs_fu = 0.0
        from_user, from_table = [], []
        for sp, amount in composition.items():
            if sp in user_xs:                                  # option B
                s_scat, s_abs = user_xs[sp]
                from_user.append(str(sp))
            else:                                              # option C
                s_scat, s_abs = data.neutron_xs(sp)
                if wavelength_A is not None:
                    s_abs *= wavelength_A / THERMAL_WAVELENGTH_A   # 1/v law
                from_table.append(str(sp))
            scat_fu += amount * s_scat
            abs_fu += amount * s_abs
        sources = (f"user: {', '.join(from_user) or '-'}; "
                   f"tables: {', '.join(from_table) or '-'}")
        if wavelength_A is not None and from_table:
            sources += f" (absorption scaled to {wavelength_A:g} Angstrom)"

    # --- Step 5: linear attenuation coefficients (1/mm) ----------------------
    # n [1/cm^3] * sigma [barn] * 1e-24 [cm^2/barn] = coefficient in 1/cm;
    # multiplying by 0.1 gives 1/mm.
    mu_scat = units_per_cm3 * scat_fu * BARN_TO_CM2 * CM_TO_MM
    mu_abs = units_per_cm3 * abs_fu * BARN_TO_CM2 * CM_TO_MM
    mu = mu_scat + mu_abs

    # --- Step 6: scatter fraction ---------------------------------------------
    total_sigma = scat_fu + abs_fu
    # (If the material has no cross section at all nothing can interact; any
    # value is irrelevant, we use 1.)
    scatter_fraction = scat_fu / total_sigma if total_sigma > 0 else 1.0

    return LayerProperties(
        label=label, thickness_mm=float(thickness), formula=formula,
        density_g_cm3=float(density), composition=composition,
        molar_mass=molar_mass, formula_units_per_cm3=units_per_cm3,
        sigma_scat_barn=scat_fu, sigma_abs_barn=abs_fu,
        mu_scat=mu_scat, mu_abs=mu_abs, mu=mu,
        scatter_fraction=scatter_fraction, sources=sources)


def make_layer_tuple(label, thickness, formula, density, sigma_scat=None,
                     sigma_abs=None, cross_sections=None, masses=None,
                     wavelength_A=None, data=None, fill_height=None):
    """Same as `compute_layer_properties` (same parameters), but returns only
    the tuple (label, thickness_mm, mu_per_mm, scatter_fraction) that
    defines a Layer, ready to be put in the list of layers of
    create_slab_sample, create_cylindrical_sample or make_sample.
    With `fill_height` (mm from the bottom of the sample) the tuple has a 5th
    element and the layer is filled only up to that height."""
    return compute_layer_properties(
        label, thickness, formula, density, sigma_scat, sigma_abs,
        cross_sections, masses, wavelength_A, data).as_tuple(fill_height)


def build_layer(label, thickness, formula, density, sigma_scat=None,
                sigma_abs=None, cross_sections=None, masses=None,
                wavelength_A=None, data=None, fill_height=None, fullness=None,
                bottom=None, height=None):
    """Same as `make_layer_tuple`, but returns a `Layer` object, which can
    also have a limited vertical extent: `bottom` and `height` (mm, from the
    centre of the beam; version 2.2), or `fill_height`, or `fullness`
    (fraction of the beam height covered by the material; the sample then
    needs its beam_height)."""
    return compute_layer_properties(
        label, thickness, formula, density, sigma_scat, sigma_abs,
        cross_sections, masses, wavelength_A, data).as_layer(fill_height,
                                                             fullness, bottom,
                                                             height)


# --------------------------------------------------------------------------
# Demo:  python layer_builder.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from .cylinder import create_cylindrical_sample

    print("1) Water, with your own cross sections for the whole molecule")
    water = compute_layer_properties("water", 2.0, "H2O", 1.0,
                                     sigma_scat=168.3, sigma_abs=0.665)
    print(water.summary())

    print("\n2) Heavy water, cross sections per atom given by the user (D and O)")
    # The numbers are illustrative: use your own or tabulated values.
    heavy = compute_layer_properties(
        "D2O", 6.0, "D2O", 1.107,
        cross_sections={"D": (7.64, 0.000519), "O": (4.232, 0.00019)})
    print(heavy.summary())

    print("\n3) Aluminium container, per-atom values given by the user")
    can = compute_layer_properties("Al can", 1.0, "Al", 2.70,
                                   cross_sections={"Al": (1.503, 0.231)})
    print(can.summary())

    print("\n4) The tuples go straight into a sample (cylinder from the axis out):")
    sample = create_cylindrical_sample(20, [heavy.as_tuple(), can.as_tuple()])
    print(sample.summary())

    print("\n5) The same can, filled with heavy water only up to 70 % of an "
          "8 mm beam (bottom of the can at the bottom of the beam):")
    half_full = create_cylindrical_sample(
        20, [heavy.as_layer(fullness=0.7), can.as_tuple()],
        y_bottom=-4.0, beam_height=8.0)
    print(half_full.summary())

    print("\n6) Tabulated values")
    data = ToscanaData()
    if data.has_neutron_data:
        iron = compute_layer_properties("Fe plate", 1.0, "Fe", 7.87, data=data)
        print(iron.summary())
    else:
        print("The package 'periodictable' is not installed: tabulated cross "
              "sections are not available (pip install periodictable).")
        try:
            make_layer_tuple("Fe plate", 1.0, "Fe", 7.87)
        except TableError as error:
            print("Expected message:", error)
