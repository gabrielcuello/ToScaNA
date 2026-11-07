"""
run_simulation.py - Main script of MCAMS (Monte Carlo Attenuation and
Multiple Scattering): choose the sample geometry (slab, cylinder, disc or
sphere), and
optionally how full the container is, and run the whole simulation chain.

What this script does, step by step
-----------------------------------
 1. Build the sample of the chosen geometry (sample_factory.make_sample).
 2. Transmission of the scattered particles vs scattering angle 2theta
    (transmission_vs_angle.py): single-scattering estimate at exact angles.
 3. Particle-by-particle simulation (multiple_scattering.py): number of
    particles scattered once (single), twice or more (multiple), and in total,
    as a function of 2theta.
 4. Transmission vs angle separated by the layer of the LAST scattering
    (transmission_by_layer.py): one curve per layer, for total / single /
    multiple scattering.
 5. Save and show the plots; the file names contain the geometry.

How to choose the geometry
--------------------------
  * From the command line:
        python run_simulation.py --geometry slab
        python run_simulation.py --geometry cylinder --particles 500000
        python run_simulation.py --geometry slab --plane-angle 60
        python run_simulation.py --geometry disc --plane-angle 45   (version 2.3)
        python run_simulation.py --geometry sphere                  (version 2.4)
        python run_simulation.py --geometry cylinder --no-show   (only save the plots)
        python run_simulation.py --geometry cylinder --fullness 0.5
        python run_simulation.py --geometry cylinder --bottom -6 --height 5
  * Or from Python (for example in a Jupyter notebook):
        from run_simulation import run_simulation
        results = run_simulation("cylinder")
        results = run_simulation("cylinder", fullness=0.5)
        results = run_simulation("cylinder", bottom=-6, height=5)
  * Or simply change DEFAULT_GEOMETRY below.

Where to put YOUR sample
------------------------
Edit SLAB_LAYERS, CYLINDER_LAYERS, DISC_LAYERS and SPHERE_LAYERS below. Each
layer is a tuple
    (name, thickness in mm, mu in 1/mm, scatter fraction)
  * slab     : layers from the FRONT face to the back.
  * disc     : the same; all the layers have the diameter DISC_DIAMETER and
               the disc is centred on the beam.
  * cylinder : layers from the INNER cylinder outwards; the thickness of the
               first (innermost) layer is its DIAMETER, the others are the
               radial thickness of each annulus.
  * sphere   : the same as the cylinder (inner sphere: DIAMETER; then the
               radial thickness of each shell); centred on the beam.
A 5th element, the fill height in mm from the bottom of the sample, makes a
layer partially filled. For the layers named in CONTENT_LAYERS:
  * --bottom and --height (or bottom=..., height=...; version 2.2) set the
    band of heights of their material, in mm from the centre of the beam:
    from `bottom` to `bottom + height`;
  * --fullness (or fullness=...; version 2.0) sets the fraction of the beam
    height covered by their material, starting at the bottom of the sample.
The fullness of every layer is printed in the summary of the sample.
The numbers below are only illustrative.
"""

import argparse          # reads the options typed on the command line

import numpy as np

from .sample_factory import make_sample
from .transmission_vs_angle import transmission_vs_angle, plot_transmission
from .multiple_scattering import simulate_scattering_orders, plot_scattering_orders
from .transmission_by_layer import (transmission_by_last_layer,
                                   plot_transmission_by_layer)

# --------------------------------------------------------------------------
# Settings you can edit
# --------------------------------------------------------------------------
DEFAULT_GEOMETRY = "cylinder"          # "slab", "cylinder", "disc" or "sphere"

# Slab: from the front face to the back (name, thickness, mu, scatter fraction).
SLAB_LAYERS = [("Al", 2.0, 0.020, 1.0),
               ("Fe", 1.0, 0.250, 0.6),
               ("Cu", 0.5, 0.300, 0.5)]
SLAB_WIDTH, SLAB_HEIGHT = 20.0, 20.0   # mm

# Cylinder: from the inner cylinder outwards. The first thickness is the
# DIAMETER (6 mm -> radius 3 mm); then annuli of 1.0 and 0.5 mm.
CYLINDER_LAYERS = [("Fe", 6.0, 0.250, 0.6),
                   ("Al", 1.0, 0.020, 1.0),
                   ("Cu", 0.5, 0.300, 0.5)]
CYLINDER_HEIGHT = 20.0                 # mm

# Disc (version 2.3): a circular plate centred on the beam, layers from the
# front face to the back, as for the slab; all have the same diameter.
DISC_LAYERS = [("Al", 2.0, 0.020, 1.0),
               ("Fe", 1.0, 0.250, 0.6),
               ("Cu", 0.5, 0.300, 0.5)]
DISC_DIAMETER = 20.0                   # mm

# Sphere (version 2.4): concentric spheres centred on the beam, from the
# inner one outwards, as for the cylinder (first thickness = DIAMETER).
SPHERE_LAYERS = [("Fe", 6.0, 0.250, 0.6),
                 ("Al", 1.0, 0.020, 1.0),
                 ("Cu", 0.5, 0.300, 0.5)]

# Incident beam (mm): a uniform rectangle. To use profiles, pass H=... and
# V=... to the functions below (see beam.py).
BEAM_WIDTH, BEAM_HEIGHT = 8.0, 8.0

# Partially filled container.
# Height (mm) of the bottom of the sample; None = sample centred on the beam
# (as in the full-sample examples). With -BEAM_HEIGHT / 2 the bottom of the
# sample is at the bottom of the beam, and the fullness is then simply
# (height of the material) / (beam height).
SAMPLE_Y_BOTTOM = None
# Names of the layers that are the "content" of the container: the option
# --fullness applies to them (the other layers, e.g. the walls, stay full).
CONTENT_LAYERS = {"slab": ["Fe"], "cylinder": ["Fe"], "disc": ["Fe"], "sphere": ["Fe"]}


def _with_extent(layers, names, fullness=None, bottom=None, height=None):
    """Copy of the list of layer tuples in which the layers called `names`
    get a limited vertical extent: `bottom`/`height` (version 2.2) or
    `fullness` (version 2.0). They are turned into dicts, the form that
    accepts these fields."""
    extra = {k: v for k, v in (("fullness", fullness), ("bottom", bottom),
                               ("height", height)) if v is not None}
    if not extra:
        return layers
    keys = ("name", "thickness", "mu", "scatter_fraction")
    out = []
    for layer in layers:
        if layer[0] in names:
            # zip pairs the field names with the values of the tuple.
            out.append(dict(zip(keys, layer[:4]), **extra))
        else:
            out.append(layer)
    return out


def build_sample(geometry, plane_angle_deg=90.0, fullness=None, bottom=None,
                 height=None):
    """Create the sample of the chosen geometry from the settings above.

    For the layers named in CONTENT_LAYERS:
    bottom, height : band of heights of their material (mm, from the centre
                     of the beam; version 2.2). None = not limited.
    fullness       : fraction of the beam height covered by their material
                     (version 2.0); do not combine with bottom/height.
    """
    name = geometry.strip().lower()
    if name in ("sphere", "spherical", "ball"):
        # The sphere is centred on the beam and has no orientation.
        layers = _with_extent(SPHERE_LAYERS, CONTENT_LAYERS["sphere"], fullness, bottom, height)
        return make_sample("sphere", layers, beam_height=BEAM_HEIGHT)
    if name in ("disc", "disk"):
        # The disc is centred on the beam: SAMPLE_Y_BOTTOM is not used.
        layers = _with_extent(DISC_LAYERS, CONTENT_LAYERS["disc"], fullness, bottom, height)
        return make_sample("disc", layers, diameter=DISC_DIAMETER,
                           plane_angle_deg=plane_angle_deg, beam_height=BEAM_HEIGHT)
    if name in ("slab", "plate", "planar", "flat"):
        layers = _with_extent(SLAB_LAYERS, CONTENT_LAYERS["slab"], fullness, bottom, height)
        return make_sample("slab", layers, height=SLAB_HEIGHT,
                           width=SLAB_WIDTH, plane_angle_deg=plane_angle_deg,
                           y_bottom=SAMPLE_Y_BOTTOM, beam_height=BEAM_HEIGHT)
    layers = _with_extent(CYLINDER_LAYERS, CONTENT_LAYERS["cylinder"], fullness, bottom, height)
    return make_sample("cylinder", layers, height=CYLINDER_HEIGHT,
                       y_bottom=SAMPLE_Y_BOTTOM, beam_height=BEAM_HEIGHT)


def run_simulation(geometry=DEFAULT_GEOMETRY, plane_angle_deg=90.0,
                   n_particles=500000, show=True, output_prefix="",
                   fullness=None, bottom=None, height=None):
    """Run the complete chain for the chosen geometry.

    Parameters
    ----------
    geometry : str
        "slab", "cylinder", "disc" or "sphere".
    plane_angle_deg : float
        Slab and disc: angle between the beam and the plane of the sample
        (90 = perpendicular).
    n_particles : int
        Particles for the single/multiple scattering simulation (the
        transmission curves, which are computed for every angle, use one tenth
        of this number: they need fewer).
    show : bool
        Open the plot windows (False: only save the PNG files).
    output_prefix : str
        Text put in front of the PNG file names (e.g. a folder "results/").
    fullness : float, optional
        Fraction of the beam height covered by the material of the layers
        named in CONTENT_LAYERS (partially filled container). None = full.
        The file names then end with "_full<value>".
    bottom, height : float, optional
        Band of heights (mm, from the centre of the beam) of the material of
        the layers named in CONTENT_LAYERS (version 2.2). The file names then
        end with "_b<bottom>_h<height>".

    Returns
    -------
    dict with the sample and the results of both calculations.
    """
    # --- Step 1: the sample ---------------------------------------------------
    sample = build_sample(geometry, plane_angle_deg, fullness, bottom, height)
    print(sample.summary())
    # Tag used in the file names: e.g. "cylinder", "cylinder_full0.5" or
    # "cylinder_b-6_h5".
    tag = geometry
    if fullness is not None:
        tag += f"_full{fullness:g}"
    if bottom is not None or height is not None:
        b_txt = f"{bottom:g}" if bottom is not None else "base"   # :g -> "-6", not "-6.0"
        h_txt = f"{height:g}" if height is not None else "top"
        tag += f"_b{b_txt}_h{h_txt}"

    # --- Step 2: transmission vs scattering angle -----------------------------
    angles, T, f_int = transmission_vs_angle(
        sample, BEAM_WIDTH, BEAM_HEIGHT,
        n_particles=max(n_particles // 10, 2), rng=1)
    print(f"\nFraction of the incident particles that interact: {f_int:.3f}")
    plot_transmission(
        angles, T, title=f"Transmission vs angle ({geometry})",
        save_as=f"{output_prefix}transmission_{tag}.png", show=show)

    # --- Step 3: single / multiple scattering ---------------------------------
    res = simulate_scattering_orders(
        sample, BEAM_WIDTH, BEAM_HEIGHT, n_particles=n_particles, rng=2)
    n_single, n_multi = res.single.sum(), res.multiple.sum()
    print(f"\nIncident particles          : {res.n_incident}")
    print(f"No interaction (cross/miss) : {res.n_no_interaction}")
    print(f"Absorbed                    : {res.n_absorbed}")
    print(f"Scattered and exited        : {res.n_exited}")
    print(f"Exited after 1 scattering   : {n_single:.0f}")
    print(f"Exited after 2 or more      : {n_multi:.0f} "
          f"({100 * n_multi / (n_single + n_multi):.1f} % of the scattered)")

    # --- Step 4: plot ---------------------------------------------------------
    plot_scattering_orders(
        res, logy=True, title=f"Single and multiple scattering vs angle ({geometry})",
        save_as=f"{output_prefix}scattering_orders_{tag}.png", show=show)

    # --- Step 5: transmission by layer of the last scattering -----------------
    # One vector (curve) per layer in which the particle scattered last, with
    # the attenuation of ALL the layers crossed on the way to the detector.
    by_layer = transmission_by_last_layer(
        sample, BEAM_WIDTH, BEAM_HEIGHT,
        n_particles=max(n_particles // 10, 2), rng=3)
    print("\nScattering events per incident particle, by layer of scattering:")
    for kind in ("single", "multiple"):
        per_incident = by_layer.events(kind) / by_layer.n_incident
        # float(v) turns NumPy numbers into plain Python numbers for printing.
        print(f"  {kind:<9}", {name: round(float(v), 4) for name, v in
                              zip(by_layer.layer_names, per_incident)})
    plot_transmission_by_layer(
        by_layer, title=f"Transmission by layer of last scattering ({geometry})",
        save_as=f"{output_prefix}transmission_by_layer_{tag}.png", show=show)

    return {"sample": sample, "angles": angles, "transmission": T,
            "interaction_fraction": f_int, "scattering": res,
            "by_layer": by_layer}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # argparse turns the words typed after the script name into Python values.
    parser = argparse.ArgumentParser(
        description="Simulate a slab, cylindrical, disc or spherical sample in the beam.")
    parser.add_argument("--geometry", choices=["slab", "cylinder", "disc", "sphere"],
                        default=DEFAULT_GEOMETRY, help="sample geometry")
    parser.add_argument("--plane-angle", type=float, default=90.0,
                        help="slab and disc: angle between beam and sample plane (deg)")
    parser.add_argument("--particles", type=int, default=500000,
                        help="number of particles of the scattering simulation")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open plot windows, only save the PNG files")
    parser.add_argument("--fullness", type=float, default=None,
                        help="fraction of the beam height covered by the "
                             "layers named in CONTENT_LAYERS (default: full)")
    parser.add_argument("--bottom", type=float, default=None,
                        help="bottom of the material of the CONTENT_LAYERS, mm "
                             "from the centre of the beam (version 2.2)")
    parser.add_argument("--height", type=float, default=None,
                        help="height of the material of the CONTENT_LAYERS, mm "
                             "(version 2.2)")
    args = parser.parse_args()

    run_simulation(args.geometry, args.plane_angle, args.particles,
                   show=not args.no_show, fullness=args.fullness,
                   bottom=args.bottom, height=args.height)
