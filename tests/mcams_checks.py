"""
test_mcams.py - Self-test of the geometry of MCAMS (Monte Carlo Attenuation
and Multiple Scattering), in particular of the partially filled layers.

Run it from the code folder:   python tests/test_mcams.py

What is checked
---------------
1. Paths in the material from random points in random directions (and along
   the axes) against a brute-force calculation that walks along the ray in
   tiny steps and asks which layer contains each step.
2. The next interaction: the optical depth between the start point and the
   new point must be the random optical depth drawn; a particle interacts if
   and only if that value is smaller than the optical depth to the exit; the
   new point must be in the material.
3. Interaction rates against exact formulas (Beer-Lambert law times the
   fraction of the beam that hits the material).
4. A layer with fill height 0 behaves as a layer with mu = 0.
5. Direct method (direct_attenuation.py): analytic slab factors, exact
   cylinder factor at 2theta = 0, mean paths, NaN for mu = 0 layers, and
   agreement with the Monte Carlo within the statistical errors.
6. Version 2.2, layers with their own bottom and height: interacting
   fractions against exact formulas for bands below, across, above and
   outside the beam (slab) and inside it (cylinder), beam_fullness(), the
   sample height taken from the layers, fill_height == bottom + height,
   refused combinations, mean path of a band. Sections 2 and 3 also run on
   samples with bands.
7. Version 2.3, disc geometry: sections 2 and 3 also run on discs (tilted,
   with partially filled layers and bands); a very large disc gives the same
   numbers as a very large slab; the fraction of a wide beam that hits a thin
   disc equals the area of its face (normal and oblique incidence); exact
   interaction rates; exact direct-method factors at 2theta = 0 and 90 deg;
   direct method against Monte Carlo for a tilted disc; factory and copies.
8. Version 2.4, sphere geometry: sections 2 and 3 also run on spheres (with
   a partially filled layer and with bands); paths do not change when points
   and directions are rotated together (a sphere has no special direction);
   exact interaction rates for a pencil beam, a wide beam, a band and a black
   sphere; exact direct-method factor at 2theta = 0 and mean path; direct
   method against Monte Carlo; factory.
"""
import os
import sys

import numpy as np

# MCAMS is toscana.corrections.mcams; _pkg (next to this file) makes the
# package importable as `toscana`. Run by tests/test_mcams.py.
import _pkg  # noqa: E402,F401
from toscana.corrections.mcams import beam  # noqa: E402
from toscana.corrections.mcams import slab  # noqa: E402
from toscana.corrections.mcams import cylinder  # noqa: E402
from toscana.corrections.mcams import disc  # noqa: E402
from toscana.corrections.mcams import sphere  # noqa: E402
from toscana.corrections.mcams.sample_base import Layer     # noqa: E402

new = {"beam": beam, "slab": slab, "cylinder": cylinder, "disc": disc, "sphere": sphere}
rng = np.random.default_rng(0)
ok = True


def check(name, cond, info=""):
    """Print PASS or FAIL for one check and remember any failure."""
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {info}")


def rand_dirs(n):
    """n random unit vectors, uniformly distributed on the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- 2. brute force
def brute_paths(sample, p, d, step=2e-4):
    """Material path per layer by stepping along the ray (midpoint rule)."""
    out = np.zeros((len(p), sample.n_layers))
    for j in range(len(p)):
        # distance to exit: step until outside the box
        t = np.arange(step / 2, 60.0, step)
        pts = p[j] + t[:, None] * d[j]
        inside = sample.contains(pts, tol=0)
        # stop at first exit (convex box)
        stop = np.argmin(inside) if not inside.all() else len(t)
        lay = sample.layer_of_points(pts[:stop])
        for i in range(sample.n_layers):
            out[j, i] = step * np.sum(lay == i)
    return out


def random_inside(sample, n):
    pts = []
    while len(pts) < n:
        if hasattr(sample, "width"):
            q = rng.uniform(sample._lo, sample._hi, size=(4 * n, 3))
        else:
            R = sample.outer_radius
            q = np.column_stack([rng.uniform(-R, R, 4 * n),
                                 rng.uniform(sample.y_bottom, sample.y_top, 4 * n),
                                 rng.uniform(-R, R, 4 * n)])
        q = q[sample.contains(q)]
        pts.extend(q)
    return np.array(pts[:n])


S = new["slab"]
C = new["cylinder"]
sl = S.create_slab_sample(10, 10, [("Al", 1.0, 0.05), ("w", 2.0, 0.5, 1.0, 3.0),
                                   ("Cu", 1.5, 0.3, 1.0, 6.5), ("Al", 1.0, 0.05)],
                          plane_angle_deg=70)
cy = C.create_cylindrical_sample(10, [("in", 4.0, 0.4, 1.0, 2.5), ("mid", 1.0, 0.2, 1.0, 7.0),
                                      ("wall", 0.5, 0.05)])
# Version 2.2: layers with their own bottom and height (bands of material that
# start above the bottom of the sample, end below its top, or both).
slb = S.create_slab_sample(10, None, [Layer("Al", 1.0, 0.05, bottom=-5.0, height=10.0),
                                      Layer("w", 2.0, 0.5, 1.0, bottom=-2.0, height=4.5),
                                      Layer("Cu", 1.5, 0.3, 1.0, bottom=1.0, height=3.0),
                                      ("Al", 1.0, 0.05)], plane_angle_deg=70)
cyb = C.create_cylindrical_sample(None, [Layer("in", 4.0, 0.4, 1.0, bottom=-3.0, height=4.0),
                                         Layer("mid", 1.0, 0.2, 1.0, bottom=-1.0, height=5.0),
                                         Layer("wall", 0.5, 0.05, 1.0, bottom=-5.0, height=10.0)])
# Version 2.3: discs (circular plates), tilted, with partial layers and bands.
D = new["disc"]
dsc = D.create_disc_sample(10, [("Al", 1.0, 0.05), ("w", 2.0, 0.5, 1.0, 6.0),
                                ("Cu", 1.5, 0.3), ("Al", 1.0, 0.05)], plane_angle_deg=70)
dsb = D.create_disc_sample(10, [("Al", 1.0, 0.05), Layer("w", 2.0, 0.5, 1.0, bottom=-2.0, height=4.5),
                                Layer("Cu", 1.5, 0.3, 1.0, bottom=1.0, height=3.0),
                                ("Al", 1.0, 0.05)], plane_angle_deg=125)
# Version 2.4: concentric spheres, with a partial layer and with bands.
SP = new["sphere"]
sph = SP.create_spherical_sample([("in", 4.0, 0.4, 1.0, 3.5), ("mid", 1.0, 0.2), ("wall", 0.5, 0.05)])
spb = SP.create_spherical_sample([Layer("in", 4.0, 0.4, 1.0, bottom=-1.5, height=2.5),
                                  Layer("mid", 1.0, 0.2, 1.0, bottom=-4.0, height=4.5),
                                  ("wall", 0.5, 0.05)])
SAMPLES = [("slab", sl), ("cylinder", cy), ("slab bands", slb), ("cylinder bands", cyb),
           ("disc", dsc), ("disc bands", dsb), ("sphere", sph), ("sphere bands", spb)]
for name, smp in SAMPLES:
    p = random_inside(smp, 300)
    d = rand_dirs(300)
    L = smp.outgoing_path_lengths(p, d)
    B = brute_paths(smp, p, d)
    err = np.abs(L - B).max()
    check(f"{name} partial: outgoing paths vs brute force", err < 2e-3, f"max diff {err:.1e} mm")
    # horizontal and vertical special directions
    for dd in ([1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]):
        L = smp.outgoing_path_lengths(p[:50], np.array(dd, float))
        B = brute_paths(smp, p[:50], np.tile(np.array(dd, float), (50, 1)))
        err = np.abs(L - B).max()
        check(f"{name} partial: direction {dd}", err < 2e-3, f"max diff {err:.1e}")

# ---------------------------------------------------------------- 3. next interaction
for name, smp in SAMPLES:
    n = 20000
    p = random_inside(smp, n)
    d = rand_dirs(n)
    newp, hit = smp.next_interaction(p, d, rng=11)
    tau = -np.log(1 - np.random.default_rng(11).uniform(size=n))
    # optical depth from p to newp: total minus that from newp onward
    tau_total = smp.transmission(smp.outgoing_path_lengths(p, d))
    tau_total = -np.log(tau_total)
    tau_after = -np.log(smp.transmission(smp.outgoing_path_lengths(newp[hit], d[hit])))
    err = np.abs((tau_total[hit] - tau_after) - tau[hit]).max()
    check(f"{name} partial: next interaction optical depth", err < 1e-8, f"max diff {err:.1e}")
    check(f"{name} partial: interacts iff tau < total",
          np.array_equal(hit, tau < tau_total))
    check(f"{name} partial: interaction points in material",
          np.all(smp.layer_of_points(newp[hit]) >= 0))

# ---------------------------------------------------------------- 4. analytic
mu, t = 0.4, 2.0
for f in (0.25, 0.5, 0.8, 1.0):
    smp = S.create_slab_sample(100, 8, [{"name": "c", "thickness": t, "mu": mu, "fullness": f}],
                               y_bottom=-4, beam_height=8)
    x, y = new["beam"].sample_beam_point(8, 8, n=400000, rng=2)
    _, hit = smp.sample_interaction_points(x, y, rng=3)
    expect = f * (1 - np.exp(-mu * t))
    sig = np.sqrt(expect * (1 - expect) / len(x))
    check(f"slab fullness {f}: interacting fraction", abs(hit.mean() - expect) < 4 * sig,
          f"{hit.mean():.4f} vs {expect:.4f}")
    # beam_fullness round trip
    check(f"slab fullness {f}: beam_fullness()", np.isclose(smp.beam_fullness()[0], f))

# cylinder, beam centred, sample centred, fill height h: fraction of beam below level
R = 3.0
smp = C.create_cylindrical_sample(20, [("c", 2 * R, mu, 1.0, 13.0)])   # level y = 3
x, y = new["beam"].sample_beam_point(0.001, 8, n=400000, rng=2)        # narrow beam through axis
_, hit = smp.sample_interaction_points(x, y, rng=3)
expect = (7.0 / 8.0) * (1 - np.exp(-2 * R * mu))
sig = np.sqrt(expect * (1 - expect) / len(x))
check("cylinder fill 13 mm: narrow beam interacting fraction", abs(hit.mean() - expect) < 4 * sig,
      f"{hit.mean():.4f} vs {expect:.4f}")

# empty layer behaves like mu = 0
a = S.create_slab_sample(10, 10, [("a", 1, 0.3), ("b", 2, 0.5, 1.0, 0.0), ("c", 1, 0.3)])
b = S.create_slab_sample(10, 10, [("a", 1, 0.3), ("b", 2, 0.0), ("c", 1, 0.3)])
x, y = new["beam"].sample_beam_point(8, 8, n=50000, rng=2)
pa, ha = a.sample_interaction_points(x, y, rng=3)
pb, hb = b.sample_interaction_points(x, y, rng=3)
check("empty layer == mu 0 layer", np.array_equal(ha, hb) and np.allclose(pa, pb))

# ---------------------------------------------------------------- 5. direct method
# (direct_attenuation.py) against exact results and against the Monte Carlo
import math                                                         # noqa: E402
from toscana.corrections.mcams.direct_attenuation import (attenuation_factors_direct,        # noqa: E402
                                attenuation_factors_mc, mean_material_paths)

mu, t = 0.25, 2.0
smp = S.create_slab_sample(2000, 2000, [("A", t, mu)])
angs = np.array([30.0, 150.0])
r = attenuation_factors_direct(smp, 10, 10, angles_deg=angs, n_points=100_000, rng=1)
b = 1 / math.cos(math.radians(30))
exact30 = math.exp(-mu * t * b) * (math.exp(mu * t * (b - 1)) - 1) / (mu * t * (b - 1))
k = 1 + 1 / abs(math.cos(math.radians(150)))
exact150 = (1 - math.exp(-mu * t * k)) / (mu * t * k)
for i, ex in enumerate((exact30, exact150)):
    check(f"direct: slab A({angs[i]:.0f}) analytic",
          abs(r.factors[0, i] - ex) < 4 * r.errors[0, i] + 1e-6,
          f"{r.factors[0, i]:.5f} vs {ex:.5f}")
check("direct: slab mean path = thickness", abs(r.mean_path[0] - t) < 1e-9)

# cylinder at 2theta = 0: exact 1D integral (the attenuation is constant on each beam line)
R, mu = 3.0, 0.2
cyl = C.create_cylindrical_sample(50, [("c", 2 * R, mu)])
r = attenuation_factors_direct(cyl, 8, 20, angles_deg=[0.0], n_points=100_000, rng=2)
ys = np.linspace(-R, R, 200001)
chord = 2 * np.sqrt(np.maximum(R * R - ys * ys, 0))
trapz = getattr(np, "trapezoid", None) or np.trapz       # name changed in NumPy 2.0
exact0 = trapz(chord * np.exp(-mu * chord), ys) / (np.pi * R * R)
check("direct: cylinder A(0) exact", abs(r.factors[0, 0] - exact0) < 4 * r.errors[0, 0] + 2e-6,
      f"{r.factors[0, 0]:.5f} vs {exact0:.5f}")
check("direct: cylinder mean path = area / beam width",
      abs(r.mean_path[0] - np.pi * R * R / 8) < 4 * r.mean_path_error[0])

# a gap (mu = 0) gets NaN; partial filling halves the mean path
gap = C.create_cylindrical_sample(50, [("c", 6.0, 0.2), ("gap", 1.0, 0.0), ("w", 0.5, 0.1)])
r = attenuation_factors_direct(gap, 12, 20, angles_deg=[90.0], n_points=20_000, rng=3)
check("direct: mu = 0 layer gives NaN", np.isnan(r.factors[1, 0]) and np.isfinite(r.factors[2, 0]))
half = S.create_slab_sample(100, 100, [{"name": "h", "thickness": 2.0, "mu": 0.3, "fullness": 0.5}],
                            y_bottom=-4, beam_height=8)
mp, mpe = mean_material_paths(half, 8, 8, rng=4)
check("direct: fullness 0.5 halves the mean path", abs(mp[0] - 1.0) < 4 * mpe[0],
      f"{mp[0]:.4f} +- {mpe[0]:.4f}")

# direct method and Monte Carlo agree within the statistics
smp = C.create_cylindrical_sample(50, [("Ni", 6.0, 0.18, 0.94), ("V", 0.2, 0.046, 0.78)])
angs = np.array([0.0, 90.0, 180.0])
d = attenuation_factors_direct(smp, 15, 20, angles_deg=angs, n_points=50_000, rng=5)
m = attenuation_factors_mc(smp, 15, 20, angles_deg=angs, n_particles=25_000, n_batches=8,
                           mean_path=d.mean_path, rng=6)
z = np.abs(m.factors - d.factors) / np.hypot(m.errors, d.errors)
check("direct vs Monte Carlo: |z| < 4 for all layers and angles", np.all(z < 4),
      f"max |z| = {z.max():.2f}")

# ---------------------------------------------------------------- 6. bottom + height (v2.2)
mu, t, hb = 0.4, 2.0, 8.0                    # slab layer; beam from -4 to +4 mm
x, y = new["beam"].sample_beam_point(8, hb, n=400000, rng=12)
for b, h in ((-6.0, 5.0), (-2.0, 5.0), (1.0, 6.0), (-6.0, 12.0), (5.0, 3.0)):
    smp = S.create_slab_sample(100, 20, [Layer("c", t, mu, bottom=b, height=h)],
                               y_bottom=-10, beam_height=hb)
    f = max(0.0, min(b + h, hb / 2) - max(b, -hb / 2)) / hb     # covered fraction of the beam
    _, hit = smp.sample_interaction_points(x, y, rng=13)
    expect = f * (1 - np.exp(-mu * t))
    sig = max(np.sqrt(expect * (1 - expect) / len(x)), 1e-9)
    check(f"slab band {b:+.0f}..{b + h:+.0f} mm: interacting fraction",
          abs(hit.mean() - expect) < 4 * sig + 1e-9, f"{hit.mean():.4f} vs {expect:.4f}")
    check(f"slab band {b:+.0f}..{b + h:+.0f} mm: beam_fullness() = {f:.3f}",
          np.isclose(smp.beam_fullness()[0], f))

# cylinder, narrow beam through the axis, band of material inside the beam
R = 3.0
xn, yn = new["beam"].sample_beam_point(0.001, hb, n=400000, rng=14)
smp = C.create_cylindrical_sample(20, [Layer("c", 2 * R, mu, bottom=-1.0, height=4.0)])
_, hit = smp.sample_interaction_points(xn, yn, rng=15)
expect = (4.0 / hb) * (1 - np.exp(-2 * R * mu))
sig = np.sqrt(expect * (1 - expect) / len(xn))
check("cylinder band -1..+3 mm: narrow beam interacting fraction", abs(hit.mean() - expect) < 4 * sig,
      f"{hit.mean():.4f} vs {expect:.4f}")

# sample height from the layers (height=None) and the layer levels
check("height=None: sample spans the layers", np.isclose(cyb.y_bottom, -5) and np.isclose(cyb.y_top, 5))
check("levels of the bands", np.allclose(cyb.fill_bottoms, [-3, -1, -5]) and np.allclose(cyb.fill_tops, [1, 4, 5]))

# a layer given with fill_height = a layer with bottom = sample bottom and the same height
a = C.create_cylindrical_sample(20, [("p", 6.0, 0.25, 0.8, 5.0), ("V", 0.2, 0.04, 0.5)])
b = C.create_cylindrical_sample(20, [Layer("p", 6.0, 0.25, 0.8, bottom=-10.0, height=5.0), ("V", 0.2, 0.04, 0.5)])
x2, y2 = new["beam"].sample_beam_point(8, 30, n=50000, rng=16)
pa, ha = a.sample_interaction_points(x2, y2, rng=17)
pb, hb_ = b.sample_interaction_points(x2, y2, rng=17)
check("fill_height h == bottom=sample bottom, height=h", np.array_equal(ha, hb_) and np.allclose(pa, pb))

# bad combinations are refused
try:
    Layer("x", 1.0, 0.1, bottom=0.0, fullness=0.5)
    check("bottom + fullness refused", False)
except ValueError:
    check("bottom + fullness refused", True)
try:
    Layer("x", 1.0, 0.1, height=-1.0)
    check("negative height refused", False)
except ValueError:
    check("negative height refused", True)

# direct method: mean path of a band = thickness x covered fraction of the beam
smp = S.create_slab_sample(100, 20, [Layer("c", 2.0, 0.3, bottom=-2.0, height=3.0)], beam_height=8)
mp, mpe = mean_material_paths(smp, 8, 8, rng=18)
check("direct: band -2..+1 mm, mean path = 2 mm x 3/8", abs(mp[0] - 0.75) < 4 * mpe[0],
      f"{mp[0]:.4f} +- {mpe[0]:.4f}")

# ---------------------------------------------------------------- 7. disc geometry (v2.3)
from toscana.corrections.mcams.sample_factory import make_sample                               # noqa: E402
from toscana.corrections.mcams.disc import DiscSample                                          # noqa: E402

# (a) a very large disc behaves exactly as a very large slab (same seeds)
lay = [("Al", 1.0, 0.05), ("w", 2.0, 0.5, 0.9), ("Cu", 1.5, 0.3, 0.5)]
big_d = D.create_disc_sample(4000, lay, plane_angle_deg=60)
big_s = S.create_slab_sample(4000, 4000, lay, plane_angle_deg=60)
x, y = new["beam"].sample_beam_point(8, 8, n=50000, rng=2)
pd_, hd = big_d.sample_interaction_points(x, y, rng=3)
ps_, hs = big_s.sample_interaction_points(x, y, rng=3)
check("disc: very large disc == very large slab (interaction points)",
      np.array_equal(hd, hs) and np.allclose(pd_, ps_, atol=1e-9))
dd = rand_dirs(len(pd_))
# keep directions that leave through the front or back face (|d_z'| > 0.05):
# nearly grazing ones travel far enough to reach the rim, where the two
# shapes differ (circle against square)
dd = dd[np.abs(dd[:, 2]) > 0.05]
pd_, ps_ = pd_[:len(dd)], ps_[:len(dd)]
check("disc: very large disc == very large slab (outgoing paths)",
      np.allclose(big_d.outgoing_path_lengths(pd_, dd), big_s.outgoing_path_lengths(ps_, dd), atol=1e-9))

# (b) fraction of a wide beam (30 x 30 mm) that hits a thin, black disc
# (mu so large that every particle reaching it interacts) = face area / beam area
for ang in (90.0, 50.0):
    thin = D.create_disc_sample(20, [("black", 0.01, 1e4)], plane_angle_deg=ang)
    x, y = new["beam"].sample_beam_point(30, 30, n=400000, rng=4)
    _, hit = thin.sample_interaction_points(x, y, rng=5)
    expect = thin.face_area_in_beam(30, 30) / 900.0
    sig = np.sqrt(expect * (1 - expect) / len(x))
    check(f"disc: hit fraction of a wide beam, plane angle {ang:g}",
          abs(hit.mean() - expect) < 4 * sig, f"{hit.mean():.4f} vs {expect:.4f}")
check("disc: face area at normal incidence = pi R^2",
      np.isclose(D.create_disc_sample(20, [("a", 1, 0.1)]).face_area_in_beam(30, 30), np.pi * 100, rtol=1e-6))

# (c) exact interaction rates: beam inside the disc (Beer-Lambert, oblique
# path 1/cos), wide beam at normal incidence (area x Beer-Lambert), band
mu, t = 0.3, 2.0
for ang in (90.0, 60.0):
    smp = D.create_disc_sample(40, [("c", t, mu)], plane_angle_deg=ang)
    x, y = new["beam"].sample_beam_point(8, 8, n=400000, rng=6)
    _, hit = smp.sample_interaction_points(x, y, rng=7)
    expect = 1 - np.exp(-mu * t / np.sin(np.radians(ang)))
    sig = np.sqrt(expect * (1 - expect) / len(x))
    check(f"disc: beam inside the disc, plane angle {ang:g}", abs(hit.mean() - expect) < 4 * sig,
          f"{hit.mean():.4f} vs {expect:.4f}")
smp = D.create_disc_sample(10, [("c", t, mu)])
x, y = new["beam"].sample_beam_point(30, 30, n=400000, rng=8)
_, hit = smp.sample_interaction_points(x, y, rng=9)
expect = np.pi * 25 / 900 * (1 - np.exp(-mu * t))
sig = np.sqrt(expect * (1 - expect) / len(x))
check("disc: wide beam, normal incidence", abs(hit.mean() - expect) < 4 * sig,
      f"{hit.mean():.4f} vs {expect:.4f}")
smp = D.create_disc_sample(20, [Layer("c", t, mu, 1.0, bottom=-6.0, height=6.0)], beam_height=8)
x, y = new["beam"].sample_beam_point(8, 8, n=400000, rng=10)
_, hit = smp.sample_interaction_points(x, y, rng=11)
expect = 0.5 * (1 - np.exp(-mu * t))
sig = np.sqrt(expect * (1 - expect) / len(x))
check("disc: band -6..0 mm covers half of the beam", abs(hit.mean() - expect) < 4 * sig
      and np.isclose(smp.beam_fullness()[0], 0.5), f"{hit.mean():.4f} vs {expect:.4f}")
check("disc: band cut at the rim", np.isclose(
    D.create_disc_sample(10, [Layer("c", 1, 0.1, bottom=-20, height=22)]).fill_bottoms[0], -5.0))

# (d) direct method, exact factors. Normal incidence, 2theta = 0: every path
# is the full thickness, A = exp(-mu t). 2theta = 90 (detector along x'):
# A = [mean of exp(-mu z) over the depth] x [mean over the lit face of
# exp(-mu (sqrt(R^2 - y^2) - x))], a separable integral computed on a grid.
R_, mu, t = 5.0, 0.3, 2.0
smp = D.create_disc_sample(2 * R_, [("c", t, mu)])
r = attenuation_factors_direct(smp, 6, 6, angles_deg=[0.0, 90.0], n_points=200_000, rng=12)
check("direct: disc A(0) = exp(-mu t)", abs(r.factors[0, 0] - math.exp(-mu * t)) < 1e-9,
      f"{r.factors[0, 0]:.6f} vs {math.exp(-mu * t):.6f}")
g = np.linspace(-3, 3, 1201)
X, Y = np.meshgrid(g, g)
face = np.exp(-mu * (np.sqrt(R_ ** 2 - Y ** 2) - X)).mean()          # beam 6 x 6 inside the disc
depth = (1 - math.exp(-mu * t)) / (mu * t)
check("direct: disc A(90) exact integral", abs(r.factors[0, 1] - face * depth) < 4 * r.errors[0, 1] + 1e-5,
      f"{r.factors[0, 1]:.5f} vs {face * depth:.5f}")

# (e) direct method and Monte Carlo agree for a tilted disc with a can-like layout
smp = D.create_disc_sample(12, [("V", 0.2, 0.046, 0.78), ("Ni", 2.0, 0.18, 0.94),
                                ("V", 0.2, 0.046, 0.78)], plane_angle_deg=60)
angs = np.array([20.0, 80.0, 140.0])
d = attenuation_factors_direct(smp, 15, 20, angles_deg=angs, n_points=50_000, rng=13)
m = attenuation_factors_mc(smp, 15, 20, angles_deg=angs, n_particles=25_000, n_batches=8,
                           mean_path=d.mean_path, rng=14)
z = np.abs(m.factors - d.factors) / np.hypot(m.errors, d.errors)
check("disc: direct vs Monte Carlo, |z| < 4", np.all(z[np.isfinite(z)] < 4), f"max |z| = {np.nanmax(z):.2f}")

# (f) factory and copies
fs = make_sample("disc", [("a", 1.0, 0.1)], diameter=12, plane_angle_deg=40)
check("disc: make_sample('disc') and with_plane_angle",
      isinstance(fs, DiscSample) and isinstance(fs.with_plane_angle(70), DiscSample)
      and fs.with_plane_angle(70).plane_angle_deg == 70 and fs.diameter == 12)
try:
    make_sample("disc", [("a", 1.0, 0.1)])
    check("disc: make_sample without diameter refused", False)
except ValueError:
    check("disc: make_sample without diameter refused", True)

# ---------------------------------------------------------------- 8. sphere geometry (v2.4)
from toscana.corrections.mcams.sphere import SphericalSample                                   # noqa: E402

# (a) a sphere has no special direction: rotating points and directions
# together does not change the paths (full layers)
full = SP.create_spherical_sample([("in", 4.0, 0.4), ("mid", 1.0, 0.2), ("wall", 0.5, 0.05)])
p = random_inside(full, 2000)
d = rand_dirs(2000)
q, _ = np.linalg.qr(rng.normal(size=(3, 3)))                         # random rotation matrix
err = np.abs(full.outgoing_path_lengths(p, d) - full.outgoing_path_lengths(p @ q.T, d @ q.T)).max()
check("sphere: paths invariant under rotation", err < 1e-9, f"max diff {err:.1e} mm")

# (b) exact interaction rates for one full sphere (R = 5 mm, mu = 0.2/mm):
#     pencil beam through the centre: 1 - exp(-2 mu R);
#     beam wider than the sphere: (2 pi / (w h)) * integral_0^R (1 - exp(-2 mu u)) u du
#     (u = half chord; the area element of the beam is 2 pi s ds = 2 pi u du)
Rs, mu = 5.0, 0.2
ball = SP.create_spherical_sample([("b", 2 * Rs, mu)])
x, y = new["beam"].sample_beam_point(1e-4, 1e-4, n=200000, rng=20)
_, hit = ball.sample_interaction_points(x, y, rng=21)
expect = 1 - np.exp(-2 * mu * Rs)
sig = np.sqrt(expect * (1 - expect) / len(x))
check("sphere: pencil beam through the centre", abs(hit.mean() - expect) < 4 * sig,
      f"{hit.mean():.4f} vs {expect:.4f}")
u = np.linspace(0, Rs, 200001)
trapz = getattr(np, "trapezoid", None) or np.trapz
expect = 2 * np.pi * trapz((1 - np.exp(-2 * mu * u)) * u, u) / (12 * 12)
x, y = new["beam"].sample_beam_point(12, 12, n=400000, rng=22)
_, hit = ball.sample_interaction_points(x, y, rng=23)
sig = np.sqrt(expect * (1 - expect) / len(x))
check("sphere: beam wider than the sphere", abs(hit.mean() - expect) < 4 * sig,
      f"{hit.mean():.4f} vs {expect:.4f}")
# black sphere: every particle that meets it interacts -> fraction = area / beam area
black = SP.create_spherical_sample([("k", 2 * Rs, 1e4)])
x, y = new["beam"].sample_beam_point(8, 8, n=400000, rng=24)
_, hit = black.sample_interaction_points(x, y, rng=25)
expect = black.section_area_in_beam(8, 8) / 64
sig = np.sqrt(expect * (1 - expect) / len(x))
check("sphere: black sphere, hit fraction = section area", abs(hit.mean() - expect) < 4 * sig,
      f"{hit.mean():.4f} vs {expect:.4f}")
check("sphere: section area of a wide beam = pi R^2",
      np.isclose(ball.section_area_in_beam(30, 30), np.pi * Rs * Rs, rtol=1e-6))
# half-filled sphere (material from the bottom up to y = 0), vertical pencil
# line of beam (x = 0, -4 < y < 4): (1/8) integral_{-4}^{0} (1 - exp(-2 mu sqrt(R^2 - y^2))) dy
halfball = SP.create_spherical_sample([Layer("b", 2 * Rs, mu, 1.0, bottom=-Rs, height=Rs)], beam_height=8)
yy = np.linspace(-4, 0, 200001)
expect = trapz(1 - np.exp(-2 * mu * np.sqrt(Rs ** 2 - yy ** 2)), yy) / 8
x, y = new["beam"].sample_beam_point(1e-4, 8, n=400000, rng=26)
_, hit = halfball.sample_interaction_points(x, y, rng=27)
sig = np.sqrt(expect * (1 - expect) / len(x))
check("sphere: half-filled, line beam", abs(hit.mean() - expect) < 4 * sig
      and np.isclose(halfball.beam_fullness()[0], 0.5), f"{hit.mean():.4f} vs {expect:.4f}")

# (c) direct method, exact: at 2theta = 0 the attenuation is constant on each
# beam line (chord L), so A(0) = int L exp(-mu L) dA / int L dA over the lit
# section; for a beam wider than the sphere int L dA = 4/3 pi R^3 and the mean
# path is (4/3 pi R^3) / (w h)
r = attenuation_factors_direct(ball, 12, 12, angles_deg=[0.0], n_points=100_000, rng=28)
num = trapz(2 * u * np.exp(-2 * mu * u) * 2 * np.pi * u, u)
exact0 = num / (4 / 3 * np.pi * Rs ** 3)
check("direct: sphere A(0) exact", abs(r.factors[0, 0] - exact0) < 4 * r.errors[0, 0] + 2e-6,
      f"{r.factors[0, 0]:.5f} vs {exact0:.5f}")
mp_exact = 4 / 3 * np.pi * Rs ** 3 / 144
check("direct: sphere mean path = volume / beam area",
      abs(r.mean_path[0] - mp_exact) < 4 * r.mean_path_error[0], f"{r.mean_path[0]:.4f} vs {mp_exact:.4f}")

# (d) direct method and Monte Carlo agree for a sphere in a spherical can
smp = SP.create_spherical_sample([("Ni", 6.0, 0.18, 0.94), ("V", 0.2, 0.046, 0.78)])
angs = np.array([0.0, 90.0, 180.0])
d = attenuation_factors_direct(smp, 8, 8, angles_deg=angs, n_points=50_000, rng=29)
m = attenuation_factors_mc(smp, 8, 8, angles_deg=angs, n_particles=25_000, n_batches=8,
                           mean_path=d.mean_path, rng=30)
z = np.abs(m.factors - d.factors) / np.hypot(m.errors, d.errors)
check("sphere: direct vs Monte Carlo, |z| < 4", np.all(z < 4), f"max |z| = {z.max():.2f}")

# (e) factory
fs = make_sample("sphere", [("a", 6.0, 0.1), ("b", 0.5, 0.05)])
check("sphere: make_sample('sphere'), radii, extent",
      isinstance(fs, SphericalSample) and np.allclose(fs.radii, [3.0, 3.5])
      and fs.y_bottom == -3.5 and fs.y_top == 3.5)

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
