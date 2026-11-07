# Example 7 - Layers from chemical formula, density and cross sections
from toscana.corrections.mcams.formula import parse_formula, format_composition
from toscana.corrections.mcams.layer_builder import compute_layer_properties, make_layer_tuple
from toscana.corrections.mcams.cylinder import create_cylindrical_sample

# 1) Reading formulas (isotopes between brackets, D = deuterium)
for text in ("H2O", "Fe2O3Na2Li", "D2O", "[56Fe]2O3", "Ca(OH)2"):
    print(f"{text:<12} -> {format_composition(parse_formula(text))}")

# 2) Water: cross sections of the whole molecule given by the user (barn)
water = compute_layer_properties("water", 2.0, "H2O", 1.0,
                                 sigma_scat=168.3, sigma_abs=0.665)
print()
print(water.summary())

# 3) Heavy water and aluminium: cross sections per atom given by the user.
#    The numbers are illustrative: use your own or tabulated values.
heavy = make_layer_tuple("D2O", 6.0, "D2O", 1.107,
                         cross_sections={"D": (7.64, 0.000519), "O": (4.232, 0.00019)})
can = make_layer_tuple("Al can", 1.0, "Al", 2.70, cross_sections={"Al": (1.503, 0.231)})
print()
print("layer tuples:", heavy, can, sep="\n  ")

# 4) The tuples go straight into a sample (cylinder: from the axis outwards)
sample = create_cylindrical_sample(20, [heavy, can])
print()
print(sample.summary())
