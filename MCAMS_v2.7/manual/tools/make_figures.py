"""
make_figures.py - Generate ALL the figures of the user's manual (PDF files in
../figures/), using the simulation modules themselves.

Run it with:   python make_figures.py            (all the figures)
               python make_figures.py slab cylinder   (only some of them)
Available names: modules instrument beam slab cylinder trajectories results validation
                 partial_geometry partial_trajectories partial_results
                 band_geometry band_position   (version 2.2)
                 disc_geometry disc_results disc_validation   (version 2.3)
                 sphere_geometry sphere_results sphere_validation   (version 2.4)
                 correct_configs   (chapter on the factors of CORRECT, version 2.5)

The python modules (beam.py, slab.py, ...) are looked for in the folder given
by the environment variable MC_CODE_DIR or, by default, two levels above this
file: the folder manual/ is expected to be INSIDE the folder of the code.

Figures that show a GEOMETRY are drawn with matplotlib primitives (they are
schematic). Figures that show RESULTS are produced by the real functions of
the package, with the example samples defined in run_simulation.py.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # no window: only write files
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import (Rectangle, Circle, FancyArrowPatch,
                                FancyBboxPatch, Polygon, Arc)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.environ.get("MC_CODE_DIR",
                          os.path.abspath(os.path.join(HERE, "..", "..")))
OUT = os.path.abspath(os.path.join(HERE, "..", "figures"))
sys.path.insert(0, CODE_DIR)

from beam import sample_beam_point
from sample_base import scattered_direction
from transmission_vs_angle import transmission_vs_angle, plot_transmission
from multiple_scattering import simulate_scattering_orders, plot_scattering_orders
from transmission_by_layer import transmission_by_last_layer, plot_transmission_by_layer
from run_simulation import BEAM_WIDTH, BEAM_HEIGHT, build_sample

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D"]
RED, BLUE = "#C0392B", "#2E6DA4"


def save(fig, name):
    """Write the figure as PDF in the figures folder."""
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("written", path)


def arrow(ax, p0, p1, color="k", lw=1.4, style="-|>", ls="-", ms=10):
    """Arrow from p0 to p1 in data coordinates."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, ls=ls))


def axes_triad(ax, origin, length):
    """Triad of the top view: z to the right, x up, y out of the page."""
    ox, oy = origin
    arrow(ax, (ox, oy), (ox + length, oy), lw=1.0)
    arrow(ax, (ox, oy), (ox, oy + length), lw=1.0)
    ax.text(ox + length * 1.12, oy, "$z$", va="center")
    ax.text(ox, oy + length * 1.15, "$x$", ha="center")
    ax.add_patch(Circle((ox, oy), length * 0.14, fill=False, lw=1.0))
    ax.add_patch(Circle((ox, oy), length * 0.03, color="k"))
    ax.text(ox - length * 0.25, oy - length * 0.25, "$y$", ha="right", va="top")


# ---------------------------------------------------------------- modules
def fig_modules():
    """Diagram of the modules and of their dependencies."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(-0.8, 14.4); ax.set_ylim(-1.2, 6.1); ax.axis("off")
    boxes = {
        "run_simulation": (5.2, 5.4),
        "transmission_vs_angle": (1.7, 4.0), "multiple_scattering": (5.2, 4.0),
        "transmission_by_layer": (8.7, 4.0), "direct_attenuation": (12.6, 4.0),
        "sample_factory": (2.4, 2.6), "slab": (0.3, 1.3), "disc": (2.4, 1.3), "cylinder": (4.5, 1.3),
        "sphere": (6.9, 1.3),
        "sample_base": (2.4, 0.0), "beam": (6.0, 0.0),
        "layer_builder": (11.4, 2.6), "formula": (10.1, 1.3), "nuclear_data": (12.7, 1.3)}
    h = 0.7
    width = {n: 0.125 * len(n + ".py") + 0.5 for n in boxes}
    for name, (x, y) in boxes.items():
        if name in ("slab", "disc", "cylinder", "sphere", "sample_base", "sample_factory"):
            fc = "#DCE9F7"
        elif name in ("layer_builder", "formula", "nuclear_data"):
            fc = "#FDE9D0"
        elif name == "beam":
            fc = "#F3E1EE"
        else:
            fc = "#E3F2DC"
        w = width[name]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03",
                                    fc=fc, ec="k", lw=0.8))
        ax.text(x, y, name + ".py", ha="center", va="center", family="monospace", fontsize=7.2)

    def link(a, b, ls="-"):
        (xa, ya), (xb, yb) = boxes[a], boxes[b]
        ax.annotate("", xy=(xb, yb + h / 2 + 0.04), xytext=(xa, ya - h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9, ls=ls,
                                    shrinkA=0, shrinkB=0))
    for a in ("transmission_vs_angle", "multiple_scattering", "transmission_by_layer", "sample_factory"):
        link("run_simulation", a)
    link("sample_factory", "slab"); link("sample_factory", "cylinder"); link("sample_factory", "disc")
    link("slab", "sample_base"); link("cylinder", "sample_base")
    # disc.py is a slab with a circular section: DiscSample inherits from SlabSample
    xs_, ys_ = boxes["slab"]; xd_, yd_ = boxes["disc"]
    ax.annotate("", xy=(xs_ + width["slab"] / 2, ys_), xytext=(xd_ - width["disc"] / 2, yd_),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9, ls="--", shrinkA=0, shrinkB=0))
    # sphere.py: concentric spheres, SphericalSample inherits from CylindricalSample
    link("sample_factory", "sphere")
    xc_, yc_ = boxes["cylinder"]; xp_, yp_ = boxes["sphere"]
    ax.annotate("", xy=(xc_ + width["cylinder"] / 2, yc_), xytext=(xp_ - width["sphere"] / 2, yp_),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9, ls="--", shrinkA=0, shrinkB=0))
    for a in ("transmission_vs_angle", "multiple_scattering", "transmission_by_layer", "direct_attenuation"):
        ax.annotate("", xy=(2.4, 0.38), xytext=(boxes[a][0], boxes[a][1] - h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.7, ls=":", shrinkA=0, shrinkB=0))
    link("layer_builder", "formula"); link("layer_builder", "nuclear_data")
    # direct_attenuation.py uses transmission_by_layer.py for the Monte Carlo comparison (dashed)
    xd, yd = boxes["direct_attenuation"]
    xt, yt = boxes["transmission_by_layer"]
    ax.annotate("", xy=(xt + width["transmission_by_layer"] / 2, yt),
                xytext=(xd - width["direct_attenuation"] / 2, yd),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.9, ls="--", shrinkA=0, shrinkB=0))
    ax.text(xd, yd + 0.55, "reference for single scattering", ha="center", fontsize=7.2, color="0.35")
    ax.annotate("layer tuples", xy=(2.4 + width["sample_factory"] / 2 + 0.05, 2.6), xytext=(9.8, 2.6),
                va="center", ha="right", fontsize=7.5, color="#B35F00",
                arrowprops=dict(arrowstyle="-|>", color="#B35F00", lw=1.1, ls="--"))
    ax.text(6.0, -0.62, "start points of the particles", ha="center", fontsize=7.5, color="0.35")
    ax.text(-0.7, -1.05, "solid arrows: a module calls another;  dotted: the analysis modules work "
            "through the common sample interface;  dashed: inherits (disc from slab, sphere from cylinder)", fontsize=7.2, color="0.35")
    save(fig, "fig_modules.pdf")


# ------------------------------------------------------------- instrument
def fig_instrument():
    """Top view of the 2-axis instrument: beam, sample, detector circle, 2theta."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_aspect("equal"); ax.axis("off")
    R = 5.0
    tt = np.radians(55.0)
    ax.add_patch(Circle((0, 0), R, fill=False, ls="--", lw=0.9, color="0.4"))
    ax.text(R * np.cos(np.radians(-40)) + 0.15, R * np.sin(np.radians(-40)) - 0.2,
            "circle of the detector", color="0.4", rotation=-50, ha="left", fontsize=8)
    ax.add_patch(Rectangle((-0.35, -1.0), 0.7, 2.0, fc="#BBBBBB", ec="k"))
    ax.text(0, -1.4, "sample", ha="center", va="top")
    ax.plot([0], [0], "k+", ms=7)
    arrow(ax, (-R - 1.4, 0), (-0.45, 0), color=RED, lw=2.0)
    ax.text(-R - 1.3, 0.35, "incident beam, along $+z$", color=RED, ha="left")
    d = np.array([np.cos(tt), np.sin(tt)])
    arrow(ax, 0.1 * d, 0.93 * R * d, color=BLUE, lw=1.6)
    det = Rectangle((-0.25, -0.7), 0.5, 1.4, fc=BLUE, ec="k")
    det.set_transform(mtransforms.Affine2D().rotate(tt).translate(*(R * d)) + ax.transData)
    ax.add_patch(det)
    ax.text(R * d[0] + 0.5, R * d[1] + 0.25, "detector", color=BLUE)
    ax.add_patch(Arc((0, 0), 4.2, 4.2, theta1=0, theta2=55, lw=1.2))
    ax.text(2.35, 0.75, r"$2\theta$", fontsize=11)
    ax.plot([0, R * 1.05], [0, 0], ls=":", color="0.5", lw=0.9)
    for ang in (100, 150):
        a = np.radians(ang)
        r = Rectangle((-0.25, -0.7), 0.5, 1.4, fc="none", ec="0.55", ls="--")
        r.set_transform(mtransforms.Affine2D().rotate(a).translate(R * np.cos(a), R * np.sin(a)) + ax.transData)
        ax.add_patch(r)
    arrow(ax, (R * np.cos(np.radians(64)), R * np.sin(np.radians(64))),
          (R * np.cos(np.radians(92)), R * np.sin(np.radians(92))), color="0.5", lw=1.0)
    ax.text(0.0, R + 0.9, "the detector moves on the circle,\nin the horizontal plane", ha="center", color="0.4")
    axes_triad(ax, (-R - 1.2, -R - 0.2), 1.3)
    ax.text(-R - 0.7, -R - 0.9, "top view: $z$ = beam, $x$ horizontal,\n$y$ vertical (out of the page)",
            fontsize=8, ha="left", va="top")
    ax.set_xlim(-R - 2.0, R + 2.4); ax.set_ylim(-R - 2.4, R + 1.9)
    save(fig, "fig_instrument.pdf")


# ------------------------------------------------------------------- beam
def fig_beam():
    """Beam rectangle, profiles H(x), V(y) and sampled start points."""
    W, Hh = 10.0, 6.0
    g = lambda x: np.exp(-x ** 2 / (2 * 1.5 ** 2))
    xs, ys = sample_beam_point(W, Hh, H=g, n=6000, rng=3)
    fig = plt.figure(figsize=(6.0, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 3), wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.scatter(xs, ys, s=2, alpha=0.35, color=BLUE, linewidths=0)
    ax.add_patch(Rectangle((-W / 2, -Hh / 2), W, Hh, fill=False, ec="k", lw=1.2))
    ax.set_xlim(-W / 2 - 0.5, W / 2 + 0.5); ax.set_ylim(-Hh / 2 - 0.5, Hh / 2 + 0.5)
    ax.set_xlabel("$x$ (mm), horizontal"); ax.set_ylabel("$y$ (mm), vertical")
    xx = np.linspace(-W / 2, W / 2, 300)
    axt.plot(xx, g(xx), color=RED, lw=1.8); axt.fill_between(xx, g(xx), alpha=0.15, color=RED)
    axt.set_ylabel("$H(x)$"); axt.set_yticks([]); plt.setp(axt.get_xticklabels(), visible=False)
    yy = np.linspace(-Hh / 2, Hh / 2, 300)
    axr.plot(np.ones_like(yy), yy, color=RED, lw=1.8); axr.fill_betweenx(yy, 1, 0, alpha=0.15, color=RED)
    axr.set_xlim(0, 1.4); axr.set_xlabel("$V(y)$"); axr.set_xticks([])
    plt.setp(axr.get_yticklabels(), visible=False)
    for a in (axt, axr):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    save(fig, "fig_beam.pdf")


# --------------------------------------------------------- slab geometry
def _box3d(ax, z0, z1, hx, hy, color, alpha=0.55):
    """Box with x in [-hx,hx], y in [-hy,hy], z in [z0,z1] (plot axes X=z, Y=x, Z=y)."""
    X = [z0, z1]; Y = [-hx, hx]; Z = [-hy, hy]
    v = [(a, b, c) for a in X for b in Y for c in Z]
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    ax.add_collection3d(Poly3DCollection([[v[i] for i in f] for f in faces], facecolors=color,
                                         edgecolors="k", linewidths=0.6, alpha=alpha))


def fig_slab_geometry():
    """Slab: perspective view (perpendicular to the beam) and top view (rotated)."""
    fig = plt.figure(figsize=(7.4, 3.9))
    # ---- (a) perspective, sample perpendicular to the beam
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    t = np.array([2.0, 1.0, 0.5]); edges = np.concatenate(([0], np.cumsum(t)))
    for i in range(3):
        _box3d(ax, edges[i], edges[i + 1], 6.0, 6.0, COLORS[i])
    bz = -7.0                                   # plane of the beam
    ax.add_collection3d(Poly3DCollection([[(bz, -2, -2), (bz, 2, -2), (bz, 2, 2), (bz, -2, 2)]],
                                         facecolors=RED, alpha=0.35, edgecolors=RED))
    ax.quiver(bz, 0, 0, 5.0, 0, 0, color=RED, lw=2, arrow_length_ratio=0.2)
    ax.text(bz, 0, 3.2, "beam", color=RED, ha="center")
    for vec, lab, pos in (((4, 0, 0), "$z$", (-1.6, -8.5, -6.5)), ((0, 4, 0), "$x$", (-7, -4, -6.5)),
                          ((0, 0, 4), "$y$", (-7, -8.5, -2))):
        ax.quiver(-7, -8.5, -6.5, *vec, color="k", arrow_length_ratio=0.2)
        ax.text(*pos, lab)
    ax.set_xlim(-8, 5); ax.set_ylim(-9, 7); ax.set_zlim(-7, 7)
    ax.set_box_aspect((1.0, 1.4, 1.1)); ax.view_init(elev=18, azim=-58); ax.set_axis_off()
    ax.set_title("(a) slab sample, perpendicular to the beam", fontsize=9)

    # ---- (b) top view, rotated sample (plane angle alpha)
    ax2 = fig.add_subplot(1, 2, 2); ax2.set_aspect("equal"); ax2.axis("off")
    alpha = 60.0; om = np.radians(90.0 - alpha)       # tilt of the normal

    def top(zp, xp):          # sample frame (z', x') -> top view (horizontal z, vertical x)
        return (zp * np.cos(om) - xp * np.sin(om), zp * np.sin(om) + xp * np.cos(om))
    half = 6.0
    for i in range(3):
        pts = [top(edges[i], -half), top(edges[i + 1], -half), top(edges[i + 1], half), top(edges[i], half)]
        ax2.add_patch(Polygon(pts, closed=True, fc=COLORS[i], ec="k", lw=0.8, alpha=0.7))
    for xb in (-3.0, 0.0, 3.0):                       # beam arrows end on the front face
        arrow(ax2, (-9.5, xb), (-xb * np.tan(om), xb), color=RED, lw=1.5)
    ax2.text(-9.5, 4.2, "beam ($+z$)", color=RED, ha="left")
    ax2.plot([0], [0], "ko", ms=4, zorder=5)
    ax2.annotate("rotation axis (vertical)", xy=(0, 0), xytext=(-8.0, -5.2), fontsize=7.5,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3"))
    # angle alpha between the beam and the plane of the sample, on the upstream side
    ax2.plot([0, -9 * np.sin(om)], [0, 9 * np.cos(om)], color="0.2", lw=0.8, ls=":")   # front-face line
    ax2.add_patch(Arc((0, 0), 6.0, 6.0, theta1=180 - alpha, theta2=180, lw=1.2))
    ax2.text(3.9 * np.cos(np.radians(180 - alpha / 2)), 3.9 * np.sin(np.radians(180 - alpha / 2)) + 0.1,
             r"$\alpha$", fontsize=11)
    arrow(ax2, (0, 0), (5.0 * np.cos(om), 5.0 * np.sin(om)), color="0.15", lw=1.2)
    ax2.text(5.3 * np.cos(om), 5.3 * np.sin(om) + 0.2, "$z'$ (normal)", fontsize=8)
    axes_triad(ax2, (-9.5, -6.8), 1.6)
    ax2.set_xlim(-10.5, 10); ax2.set_ylim(-8.8, 8)
    ax2.set_title(r"(b) top view, plane angle $\alpha$ = 60$^\circ$", fontsize=9)
    save(fig, "fig_slab_geometry.pdf")


# ----------------------------------------------------- cylinder geometry
def fig_cylinder_geometry():
    """Cylinders: side view and top view."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": (1, 1.05)})
    radii = np.array([3.0, 4.0, 4.5]); Hh = 12.0
    # ---- (a) side view: z horizontal, y vertical
    a1.set_aspect("equal"); a1.axis("off")
    for i in (2, 1, 0):
        a1.add_patch(Rectangle((-radii[i], -Hh / 2), 2 * radii[i], Hh, fc=COLORS[i], ec="k", lw=0.8, alpha=0.75))
    arrow(a1, (-9, 0), (-radii[2] - 0.2, 0), color=RED, lw=1.8)
    a1.plot([-9, -9], [-2.5, 2.5], color=RED, lw=2)
    a1.text(-9, 3.1, "beam", color=RED, ha="center")
    arrow(a1, (radii[2] + 1.3, -Hh / 2), (radii[2] + 1.3, Hh / 2), style="<|-|>", lw=0.9)
    a1.text(radii[2] + 1.65, 0, "height", rotation=90, va="center", fontsize=8)
    a1.plot([0, 0], [-Hh / 2 - 1, Hh / 2 + 1], ls="-.", color="k", lw=0.8)
    a1.text(0, Hh / 2 + 1.4, "axis (vertical)", ha="center", fontsize=8)
    a1.text(0, -Hh / 2 - 1.4, "layers: 1 (inner), 2, 3 (outer)", ha="center", fontsize=8, va="top")
    arrow(a1, (-9, -6.5), (-7.4, -6.5), lw=1.0); a1.text(-7.1, -6.5, "$z$", va="center")
    arrow(a1, (-9, -6.5), (-9, -4.9), lw=1.0); a1.text(-9, -4.6, "$y$", ha="center")
    a1.set_xlim(-10, 7.5); a1.set_ylim(-9.3, 9.0)
    a1.set_title("(a) side view", fontsize=9)
    # ---- (b) top view
    a2.set_aspect("equal"); a2.axis("off")
    for i in (2, 1, 0):
        a2.add_patch(Circle((0, 0), radii[i], fc=COLORS[i], ec="k", lw=0.8, alpha=0.75))
    for xb in (-2.4, 0.0, 2.4):
        arrow(a2, (-8.5, xb), (-np.sqrt(radii[2] ** 2 - xb ** 2) - 0.05, xb), color=RED, lw=1.5)
    a2.text(-8.5, 3.5, "beam", color=RED)
    a2.plot([0], [0], "k+", ms=7)
    for i, ang in ((0, 100), (1, 55), (2, 20)):
        a = np.radians(ang)
        a2.plot([0, radii[i] * np.cos(a)], [0, radii[i] * np.sin(a)], color="k", lw=0.8)
        a2.text(0.62 * radii[i] * np.cos(a) - 0.55, 0.62 * radii[i] * np.sin(a) + 0.12, f"$r_{i + 1}$", fontsize=8)
    two = np.radians(-60); d = np.array([np.cos(two), np.sin(two)])
    arrow(a2, (0, 0), 7.0 * d, color=BLUE, lw=1.5)
    a2.text(7.2 * d[0] - 0.2, 7.2 * d[1] - 0.6, "detector", color=BLUE)
    a2.text(0, 6.4, "first layer given by its DIAMETER $= 2 r_1$;\nthe others by their radial thickness",
            ha="center", fontsize=8)
    axes_triad(a2, (-8.5, -5.2), 1.5)
    a2.set_xlim(-9.5, 9); a2.set_ylim(-7.4, 7.8)
    a2.set_title("(b) top view", fontsize=9)
    save(fig, "fig_cylinder_geometry.pdf")


# ---------------------------------------------------------- trajectories
def fig_trajectories():
    """Top view: some particles, their interaction points and the path to the detector."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    two_theta = 100.0
    rng = np.random.default_rng(5)
    for ax, geom in zip(axes, ("slab", "cylinder")):
        sample = build_sample(geom, plane_angle_deg=60.0)
        ax.set_aspect("equal"); ax.axis("off")
        R = 10.0
        if geom == "slab":
            om = np.arctan2(sample._sin, sample._cos)          # tilt of the sample
            bounds = list(sample.boundaries)
            tp = lambda zp, xp: (zp * np.cos(om) - xp * np.sin(om), zp * np.sin(om) + xp * np.cos(om))
            for i in range(sample.n_layers):
                pts = [tp(bounds[i], -10), tp(bounds[i + 1], -10), tp(bounds[i + 1], 10), tp(bounds[i], 10)]
                ax.add_patch(Polygon(pts, closed=True, fc=COLORS[i], ec="k", lw=0.6, alpha=0.6))
        else:
            for i in reversed(range(sample.n_layers)):
                ax.add_patch(Circle((0, 0), sample.radii[i], fc=COLORS[i], ec="k", lw=0.6, alpha=0.6))
        ax.add_patch(Circle((0, 0), R, fill=False, ls="--", lw=0.7, color="0.5"))
        xb = rng.uniform(-3.5, 3.5, 400); yb = rng.uniform(-3.5, 3.5, 400)
        points, hit = sample.sample_interaction_points(xb, yb, rng=rng)
        xb_hit = xb[hit]
        d_lab = scattered_direction(two_theta)
        length = sample.outgoing_path_lengths(points, sample.to_sample_frame(d_lab)).sum(axis=1)
        p_lab = sample.to_lab_frame(points)
        layer = sample.layer_of_points(points)
        for j in range(12):
            zz, xx = p_lab[j, 2], p_lab[j, 0]
            ax.plot([-R, zz], [xb_hit[j], xx], color=RED, lw=0.8)               # incident
            ez, ex = zz + length[j] * d_lab[2], xx + length[j] * d_lab[0]       # exit point
            ax.plot([zz, ez], [xx, ex], color=BLUE, lw=1.0)                     # inside the sample
            ax.plot([ez, ez + 3.0 * d_lab[2]], [ex, ex + 3.0 * d_lab[0]], color=BLUE, lw=0.8, ls=":")
            ax.plot(zz, xx, "o", color="k", ms=3.2, mfc=COLORS[layer[j]], mew=0.8)
        ax.text(-R, R * 0.97, "incident beam", color=RED, fontsize=8, ha="left")
        ax.text(2.0, -R - 0.7, r"towards the detector, $2\theta$ = 100$^\circ$", color=BLUE, fontsize=8, ha="center")
        ax.set_xlim(-R - 1, R + 1); ax.set_ylim(-R - 1.8, R + 1)
        ax.set_title("(a) slab, $\\alpha$ = 60$^\\circ$" if geom == "slab" else "(b) cylinder", fontsize=9)
    save(fig, "fig_trajectories.pdf")


# --------------------------------------------------------------- results
def fig_results():
    """Results plots produced by the package functions themselves."""
    gslab = build_sample("slab", 90.0)
    gcyl = build_sample("cylinder")

    curves = {}
    for a in (90, 60, 45, 30):
        ang, T, _ = transmission_vs_angle(gslab.with_plane_angle(a), BEAM_WIDTH, BEAM_HEIGHT,
                                          n_particles=50000, rng=1)
        curves[f"beam/plane angle = {a}°"] = T
    plot_transmission(ang, curves, title="Slab: transmission vs angle",
                      save_as=os.path.join(OUT, "fig_transmission_slab.pdf"), show=False)
    ang, T, f = transmission_vs_angle(gcyl, BEAM_WIDTH, BEAM_HEIGHT, n_particles=50000, rng=1)
    plot_transmission(ang, T, title="Cylinder: transmission vs angle",
                      save_as=os.path.join(OUT, "fig_transmission_cylinder.pdf"), show=False)
    print("cylinder interaction fraction", round(f, 3))

    for name, smp in (("slab", gslab), ("cylinder", gcyl)):
        res = simulate_scattering_orders(smp, BEAM_WIDTH, BEAM_HEIGHT, n_particles=1000000, rng=2)
        plot_scattering_orders(res, logy=True, title=f"{name.capitalize()}: single and multiple scattering",
                               save_as=os.path.join(OUT, f"fig_orders_{name}.pdf"), show=False)
        print(name, "incident", res.n_incident, "no-interaction", res.n_no_interaction, "absorbed",
              res.n_absorbed, "exited", res.n_exited, "single", int(res.single.sum()),
              "multiple", int(res.multiple.sum()))

    for name, smp in (("slab", gslab), ("cylinder", gcyl)):
        r = transmission_by_last_layer(smp, BEAM_WIDTH, BEAM_HEIGHT, n_particles=60000, rng=3)
        plot_transmission_by_layer(r, title=f"{name.capitalize()}: transmission by layer of last scattering",
                                   save_as=os.path.join(OUT, f"fig_bylayer_{name}.pdf"), show=False)
        print(name, "events/incident single", np.round(r.events("single") / r.n_incident, 4),
              "multiple", np.round(r.events("multiple") / r.n_incident, 4))


# ------------------------------------------------------------ validation
def fig_validation():
    """Two checks against independent calculations."""
    from slab import create_slab_sample
    from cylinder import create_cylindrical_sample
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    # (a) interaction fraction of a slab versus orientation (single layer, very wide sample)
    mu, t = 0.25, 3.0
    alphas = np.arange(15, 91, 15)
    sim = []
    for al in alphas:
        s = create_slab_sample(2000, 2000, [("X", t, mu)], plane_angle_deg=float(al))
        x, y = sample_beam_point(5, 5, n=300000, rng=1)
        _, hit = s.sample_interaction_points(x, y, rng=2)
        sim.append(hit.mean())
    aa = np.linspace(10, 90, 200)
    a1.plot(aa, 1 - np.exp(-mu * t / np.sin(np.radians(aa))), "k-", label="Beer-Lambert")
    a1.plot(alphas, sim, "o", color=RED, label="simulation")
    a1.set_xlabel("angle beam / plane of the slab (deg)"); a1.set_ylabel("fraction of particles that interact")
    a1.legend(); a1.grid(alpha=0.3); a1.set_title("(a) oblique incidence on a slab", fontsize=9)
    # (b) cylinder: particle-by-particle counts versus exact-direction estimator
    cy = create_cylindrical_sample(3000, [("A", 4.0, 0.30, 0.8), ("B", 1.0, 0.15, 0.5)])
    edges = np.arange(15, 166, 10.0)
    res = simulate_scattering_orders(cy, 5, 5, n_particles=3000000, angle_edges_deg=edges,
                                     phi_deg=0.0, phi_halfwidth_deg=5.0, rng=11)
    est = transmission_by_last_layer(cy, 5, 5, n_particles=200000, angles_deg=res.angles_deg, rng=12)
    for kind, counts, col in (("single", res.single, BLUE), ("multiple", res.multiple, RED)):
        expect = (est.events(kind)[:, None] / est.n_incident
                  * np.nan_to_num(est.transmission_of(kind))).sum(axis=0) / (4 * np.pi)
        val = res.per_steradian(counts)
        err = val / np.sqrt(np.maximum(counts, 1))
        a2.plot(res.angles_deg, expect, "-", color=col, label=f"{kind}: exact-direction estimate")
        a2.errorbar(res.angles_deg, val, yerr=err, fmt="o", ms=3, color=col, capsize=2,
                    label=f"{kind}: particle by particle")
    a2.set_xlabel("scattering angle $2\\theta$ (deg)"); a2.set_ylabel("per incident particle per sr")
    a2.set_yscale("log"); a2.grid(alpha=0.3); a2.legend(fontsize=7)
    a2.set_title("(b) cylinder, two independent methods", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_validation.pdf")


# ------------------------------------------------- partially filled samples
# The example container of the chapter "Partially filled containers": a
# vanadium can (inner diameter 6 mm, wall 0.2 mm, height 30 mm) whose bottom
# is at the bottom of an 8 x 8 mm beam, filled with a powder.
def _can(fullness):
    from sample_base import Layer
    from cylinder import create_cylindrical_sample
    return create_cylindrical_sample(
        30, [Layer("powder", 6.0, 0.250, 0.8, fullness=fullness),
             ("V", 0.2, 0.073, 0.5)],
        y_bottom=-BEAM_HEIGHT / 2, beam_height=BEAM_HEIGHT)


def _cell(fill_height):
    """Flat cell: two Al windows and water up to fill_height (mm), bottom of the
    cell at the bottom of the beam."""
    from slab import create_slab_sample
    return create_slab_sample(20, 20, [("Al", 1.0, 0.010), ("water", 3.0, 0.56, 1.0, fill_height),
                                       ("Al", 1.0, 0.010)],
                              y_bottom=-BEAM_HEIGHT / 2, beam_height=BEAM_HEIGHT)


def fig_partial_geometry():
    """Side views (z horizontal, y vertical) of a partially filled slab cell and
    of a partially filled cylindrical can, with the beam and the definitions."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.2))
    hb, yb = 8.0, -4.0                     # beam height, bottom of the beam
    for ax in (a1, a2):
        ax.set_aspect("equal"); ax.axis("off")
        # the beam, coming from the left
        ax.add_patch(Rectangle((-11.5, yb), 4.0, hb, fc=RED, alpha=0.18, ec=RED, lw=1.0))
        for yy in np.linspace(yb + 0.8, yb + hb - 0.8, 5):
            arrow(ax, (-11.2, yy), (-8.0, yy), color=RED, lw=1.0, ms=7)
        ax.text(-9.5, yb + hb + 0.6, "beam", color=RED, ha="center", fontsize=8)
        arrow(ax, (-12.4, yb), (-12.4, yb + hb), style="<|-|>", lw=0.8, ms=7)
        ax.text(-12.8, yb + hb / 2, "$h_b$", ha="right", va="center")
        ax.plot([-13, 9], [yb, yb], ls=":", color="0.5", lw=0.7)
        ax.plot([-13, 9], [yb + hb, yb + hb], ls=":", color="0.5", lw=0.7)
    # ---- (a) slab cell: Al window | water up to h | Al window, side view
    level = yb + 4.5
    z = [-1.0, 0.0, 3.0, 4.0]
    top = yb + 14.0
    a1.add_patch(Rectangle((z[0], yb), 1.0, top - yb, fc=COLORS[0], ec="k", lw=0.8, alpha=0.75))
    a1.add_patch(Rectangle((z[2], yb), 1.0, top - yb, fc=COLORS[0], ec="k", lw=0.8, alpha=0.75))
    a1.add_patch(Rectangle((z[1], yb), 3.0, level - yb, fc=COLORS[1], ec="k", lw=0.8, alpha=0.75))
    a1.add_patch(Rectangle((z[1], level), 3.0, top - level, fc="white", ec="k", lw=0.6, ls="--"))
    a1.text(1.5, (level + top) / 2, "empty", ha="center", va="center", fontsize=8, color="0.4", rotation=90)
    a1.text(1.5, (yb + level) / 2, "water", ha="center", va="center", fontsize=8, rotation=90)
    arrow(a1, (5.3, yb), (5.3, level), style="<|-|>", lw=0.8, ms=7)
    a1.text(5.6, (yb + level) / 2, "fill height $h$", va="center", fontsize=8)
    a1.text(5.6, yb - 0.9, r"$y_\mathrm{bottom}$", fontsize=8, va="center")
    a1.text(-0.5, top + 0.5, "Al", ha="center", fontsize=8); a1.text(3.5, top + 0.5, "Al", ha="center", fontsize=8)
    a1.text(-2.0, -8.3, "fullness $f = h / h_b$ = 0.56", fontsize=8.5, ha="left")
    a1.set_xlim(-13.5, 10); a1.set_ylim(-9.2, 11.5)
    a1.set_title("(a) slab cell, side view", fontsize=9)
    # ---- (b) cylindrical can: wall full, powder up to the level
    R, w = 3.0, 0.5
    level = yb + 0.5 * hb
    top = yb + 14.0
    a2.add_patch(Rectangle((-R - w, yb), w, top - yb, fc=COLORS[2], ec="k", lw=0.8, alpha=0.75))
    a2.add_patch(Rectangle((R, yb), w, top - yb, fc=COLORS[2], ec="k", lw=0.8, alpha=0.75))
    a2.add_patch(Rectangle((-R, yb), 2 * R, level - yb, fc=COLORS[1], ec="k", lw=0.8, alpha=0.75))
    a2.add_patch(Rectangle((-R, level), 2 * R, top - level, fc="white", ec="k", lw=0.6, ls="--"))
    a2.text(0, (level + top) / 2, "empty", ha="center", va="center", fontsize=8, color="0.4")
    a2.text(0, (yb + level) / 2, "powder", ha="center", va="center", fontsize=8)
    a2.plot([0, 0], [yb - 0.8, top + 0.8], ls="-.", color="k", lw=0.7)
    a2.text(R + w / 2, top + 0.5, "wall", ha="center", fontsize=8)
    arrow(a2, (R + w + 0.9, yb), (R + w + 0.9, level), style="<|-|>", lw=0.8, ms=7)
    a2.text(R + w + 1.2, (yb + level) / 2, "$h$", va="center", fontsize=9)
    a2.text(-2.0, -8.3, "fullness $f = h / h_b$ = 0.5", fontsize=8.5, ha="left")
    a2.set_xlim(-13.5, 10); a2.set_ylim(-9.2, 11.5)
    a2.set_title("(b) cylindrical can, side view (cut through the axis)", fontsize=9)
    save(fig, "fig_partial_geometry.pdf")


def fig_partial_trajectories():
    """Side view of a half-full can: incident particles, first interactions and
    a few multiply scattered paths, some of them through the empty space."""
    from multiple_scattering import rotate_directions, isotropic_cos
    smp = _can(0.5)
    rng = np.random.default_rng(8)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.set_aspect("equal")
    R, Ro = smp.radii
    level = smp.fill_tops[0]
    yb, yt = smp.y_bottom, 13.5           # the can continues above the plot
    ax.add_patch(Rectangle((-Ro, yb), Ro - R, yt - yb, fc=COLORS[2], alpha=0.6, ec="k", lw=0.6))
    ax.add_patch(Rectangle((R, yb), Ro - R, yt - yb, fc=COLORS[2], alpha=0.6, ec="k", lw=0.6))
    ax.add_patch(Rectangle((-R, yb), 2 * R, level - yb, fc=COLORS[1], alpha=0.55, ec="k", lw=0.6))
    ax.add_patch(Rectangle((-R, level), 2 * R, yt - level, fc="white", ec="k", lw=0.5, ls="--"))
    ax.add_patch(Rectangle((-9.5, -4), 1.0, 8, fc=RED, alpha=0.18, ec=RED))
    x, y = sample_beam_point(0.2, 8.0, n=60, rng=rng)
    pts, hit = smp.sample_interaction_points(x, y, rng=rng)
    yh = y[hit]
    for j, yy in enumerate(y):            # incident particles that do not interact
        if not hit[j]:
            ax.plot([-9, 9], [yy, yy], color=RED, lw=0.5, alpha=0.35)
    shown = 0
    for k in range(len(pts)):
        if shown >= 14:
            break
        p = pts[k:k + 1].copy()
        ax.plot([-9, p[0, 2]], [yh[k], p[0, 1]], color=RED, lw=0.8)
        d = np.array([[0.0, 0.0, 1.0]])
        path = [p[0]]
        for _ in range(4):                # up to four scatterings
            d = rotate_directions(d, isotropic_cos(1, rng), rng)
            newp, inter = smp.next_interaction(p, d, rng=rng)
            if not inter[0]:
                L = smp._trace(p, d)
                tout = (L[0] + L[1])[0].max()
                path.append(p[0] + d[0] * (tout + 2.0))
                break
            path.append(newp[0]); p = newp
        path = np.array(path)
        ax.plot(path[:, 2], path[:, 1], color=BLUE, lw=0.8)
        ax.plot(path[1:-1, 2], path[1:-1, 1], "o", ms=2.5, color=BLUE)
        ax.plot(path[0, 2], path[0, 1], "o", ms=3.5, mfc=COLORS[smp.layer_of_points(path[:1])[0] + 1], mec="k", mew=0.6)
        shown += 1
    ax.text(-9.0, 4.6, "beam", color=RED, fontsize=8)
    ax.text(0, 9.5, "empty space", ha="center", fontsize=8, color="0.4")
    ax.text(0, -4.8, "powder (half the beam height)", ha="center", va="top", fontsize=8)
    ax.set_xlim(-10, 10); ax.set_ylim(-6.5, 12.5)
    ax.set_xlabel("$z$ (mm), along the beam"); ax.set_ylabel("$y$ (mm), vertical")
    ax.set_title("Half-full can, side view: first interactions and scattered paths", fontsize=9)
    save(fig, "fig_partial_trajectories.pdf")


def fig_partial_results():
    """Results versus fullness for the example can, and transmission curves."""
    fulls = np.round(np.arange(0.1, 1.01, 0.1), 2)
    x, y = sample_beam_point(BEAM_WIDTH, BEAM_HEIGHT, n=400000, rng=1)
    f_pow, f_wall, single, multi = [], [], [], []
    for f in fulls:
        smp = _can(float(f))
        pts, hit = smp.sample_interaction_points(x, y, rng=2)
        frac = np.bincount(smp.layer_of_points(pts), minlength=2) / len(x)
        f_pow.append(frac[0]); f_wall.append(frac[1])
        res = simulate_scattering_orders(smp, BEAM_WIDTH, BEAM_HEIGHT, n_particles=400000, rng=3)
        single.append(res.single.sum() / res.n_incident); multi.append(res.multiple.sum() / res.n_incident)
    single, multi = np.array(single), np.array(multi)
    # The same can filled EXACTLY up to the top of the beam (h = h_b, i.e. a
    # fill height of 8 mm), to compare with fullness = 1 (completely full).
    from sample_base import Layer
    from cylinder import create_cylindrical_sample
    edge = create_cylindrical_sample(30, [Layer("powder", 6.0, 0.250, 0.8, fill_height=BEAM_HEIGHT),
                                          ("V", 0.2, 0.073, 0.5)],
                                     y_bottom=-BEAM_HEIGHT / 2, beam_height=BEAM_HEIGHT)
    r_edge = simulate_scattering_orders(edge, BEAM_WIDTH, BEAM_HEIGHT, n_particles=400000, rng=3)
    s_edge, m_edge = r_edge.single.sum() / r_edge.n_incident, r_edge.multiple.sum() / r_edge.n_incident
    print("h = h_b: single", round(s_edge, 4), "multiple", round(m_edge, 4), "ratio", round(m_edge / s_edge, 3))
    print("fullness", fulls); print("powder", np.round(f_pow, 4)); print("wall", np.round(f_wall, 4))
    print("single", np.round(single, 4)); print("multiple", np.round(multi, 4))
    print("ratio", np.round(multi / single, 3))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    a1.plot(fulls, f_pow, "o-", color=COLORS[1], label="interact in the powder")
    a1.plot(fulls, f_wall, "s-", color=COLORS[2], label="interact in the wall")
    a1.plot(fulls, single, "^-", color=BLUE, label="leave after 1 scattering")
    a1.plot(fulls, multi, "v-", color=RED, label=r"leave after $\geq$2 scatterings")
    a1.plot([1.0], [s_edge], "^", mfc="white", color=BLUE)
    a1.plot([1.0], [m_edge], "v", mfc="white", color=RED)
    a1.set_xlabel("fullness $f$"); a1.set_ylabel("per incident particle")
    a1.grid(alpha=0.3); a1.legend(fontsize=7); a1.set_title("(a) can: interactions and exits", fontsize=9)
    a2.plot(fulls[:-1], multi[:-1] / single[:-1], "o-", color="k", label="material up to $f\\,h_b$")
    a2.plot([0.9, 1.0], [multi[-2] / single[-2], m_edge / s_edge], "o--", color="k", mfc="white",
            label="filled exactly to the top of the beam")
    a2.plot([1.0], [multi[-1] / single[-1]], "s", color=RED, label="completely full ($f$ = 1)")
    a2.legend(fontsize=7)
    a2.set_xlabel("fullness $f$"); a2.set_ylabel("multiple / single")
    a2.grid(alpha=0.3); a2.set_title("(b) ratio of multiple to single scattering", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_partial_fullness.pdf")

    curves = {}
    for f in (1.0, 0.75, 0.5, 0.25):
        ang, T, _ = transmission_vs_angle(_can(f), BEAM_WIDTH, BEAM_HEIGHT, n_particles=60000, rng=1)
        curves[f"fullness {f:g}"] = T
    plot_transmission(ang, curves, title="Partially filled can: transmission vs angle",
                      save_as=os.path.join(OUT, "fig_partial_transmission_cylinder.pdf"), show=False)
    curves = {}
    for h in (8.0, 6.0, 4.0, 2.0):
        ang, T, _ = transmission_vs_angle(_cell(h), BEAM_WIDTH, BEAM_HEIGHT, n_particles=60000, rng=1)
        curves[f"water height {h:g} mm (fullness {h / BEAM_HEIGHT:g})"] = T
    plot_transmission(ang, curves, title="Partially filled flat cell: transmission vs angle",
                      save_as=os.path.join(OUT, "fig_partial_transmission_slab.pdf"), show=False)


# ------------------------------------------------- layers with their own band (2.2)
def fig_band_geometry():
    """Side views of samples whose layers have their own bottom and height
    (version 2.2): (a) powder resting on a support inside a can, (b) flat cell
    whose windows start inside the beam."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.4))
    hb = 8.0
    for ax in (a1, a2):
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((-12.5, -hb / 2), 4.0, hb, fc=RED, alpha=0.18, ec=RED, lw=1.0))
        for yy in np.linspace(-hb / 2 + 0.8, hb / 2 - 0.8, 5):
            arrow(ax, (-12.2, yy), (-9.0, yy), color=RED, lw=1.0, ms=7)
        ax.text(-10.5, hb / 2 + 0.6, "beam", color=RED, ha="center", fontsize=8)
        for yy in (-hb / 2, hb / 2):
            ax.plot([-13, 10], [yy, yy], ls=":", color="0.5", lw=0.7)
        ax.plot([-13, 10], [0, 0], ls="-.", color="0.3", lw=0.7)
        ax.text(-13.2, 0, "$y=0$", ha="right", va="center", fontsize=8)
        ax.text(-13.2, hb / 2, "$+h_b/2$", ha="right", va="center", fontsize=7.5)
        ax.text(-13.2, -hb / 2, "$-h_b/2$", ha="right", va="center", fontsize=7.5)
        ax.set_xlim(-16.5, 11); ax.set_ylim(-16, 14)
    # ---- (a) can from -12 up; powder band from b = -2 to t = 7 on a support
    R, w, cb, ct, b, t = 3.0, 0.5, -12.0, 12.5, -2.0, 7.0
    for xx in (-R - w, R):
        a1.add_patch(Rectangle((xx, cb), w, ct - cb, fc=COLORS[2], ec="k", lw=0.8, alpha=0.75))
    a1.add_patch(Rectangle((-R, b), 2 * R, t - b, fc=COLORS[1], ec="k", lw=0.8, alpha=0.75))
    a1.add_patch(Rectangle((-R, cb), 2 * R, b - cb, fc="white", ec="k", lw=0.6, ls="--"))
    a1.add_patch(Rectangle((-R, t), 2 * R, ct - t, fc="white", ec="k", lw=0.6, ls="--"))
    a1.add_patch(Rectangle((-R, b - 1.0), 2 * R, 1.0, fc="0.75", hatch="////", ec="0.4", lw=0.5))
    a1.text(0, (b + t) / 2 + 1, "powder", ha="center", va="center", fontsize=8)
    a1.text(0, b - 4.5, "empty\n(support)", ha="center", va="center", fontsize=7, color="0.4")
    a1.text(0, (t + ct) / 2, "empty", ha="center", va="center", fontsize=8, color="0.4")
    arrow(a1, (R + w + 1.0, b), (R + w + 1.0, t), style="<|-|>", lw=0.8, ms=7)
    a1.text(R + w + 1.3, (b + t) / 2, "height $h_i$", va="center", fontsize=8)
    a1.plot([R + w, R + w + 2.5], [b, b], color="k", lw=0.6)
    a1.text(R + w + 2.7, b - 0.9, "bottom $b_i$", va="center", fontsize=8)
    a1.text(R + w + 1.0, cb + 0.8, "wall: $b=-12$", fontsize=7.5)
    a1.text(-12.8, -15.2, "fullness (output) = 6/8 = 0.75", fontsize=8.5, ha="left")
    a1.set_title("(a) powder on a support in a can", fontsize=9)
    # ---- (b) flat cell: windows from -2 to 12.5, water from -2 to +1
    z = [-1.0, 0.0, 3.0, 4.0]; cb, ct, lv = -2.0, 12.5, 1.0
    for zz in (z[0], z[2]):
        a2.add_patch(Rectangle((zz, cb), 1.0, ct - cb, fc=COLORS[0], ec="k", lw=0.8, alpha=0.75))
    a2.add_patch(Rectangle((z[1], cb), 3.0, lv - cb, fc=COLORS[1], ec="k", lw=0.8, alpha=0.75))
    a2.add_patch(Rectangle((z[1], lv), 3.0, ct - lv, fc="white", ec="k", lw=0.6, ls="--"))
    a2.text(1.5, (lv + ct) / 2, "empty", ha="center", va="center", fontsize=8, color="0.4", rotation=90)
    a2.text(1.5, (cb + lv) / 2, "water", ha="center", va="center", fontsize=7)
    for yy in (-3.6, -2.8):                         # part of the beam passes below the cell
        arrow(a2, (-8.5, yy), (8.5, yy), color=RED, lw=0.7, ms=6)
    a2.text(5.0, -4.6, "passes below\nthe cell", color=RED, fontsize=7.5, va="top")
    a2.text(-0.5, ct + 0.5, "Al", ha="center", fontsize=8); a2.text(3.5, ct + 0.5, "Al", ha="center", fontsize=8)
    a2.text(5.0, cb - 0.1, "$b=-2$ (all layers)", fontsize=7.5, va="bottom")
    a2.text(5.0, lv + 0.2, "water: $t=+1$", fontsize=7.5, va="bottom")
    a2.text(-12.8, -15.2, "fullness: windows 0.75, water 0.375", fontsize=8.5, ha="left")
    a2.set_title("(b) flat cell starting inside the beam", fontsize=9)
    save(fig, "fig_band_geometry.pdf")


def fig_band_position():
    """A band of powder 4 mm high moved through an 8 mm beam, in a V can from
    -15 to +15 mm: interactions follow the overlap with the beam; the ratio of
    multiple to single scattering does not."""
    from sample_base import Layer
    from cylinder import create_cylindrical_sample
    bottoms = np.arange(-7.0, 3.01, 1.0)
    x, y = sample_beam_point(BEAM_WIDTH, BEAM_HEIGHT, n=300000, rng=1)
    rows = []
    for b in bottoms:
        s = create_cylindrical_sample(None, [Layer("powder", 6.0, 0.250, 0.8, bottom=float(b), height=4.0),
                                             Layer("V", 0.2, 0.073, 0.5, bottom=-15.0, height=30.0)],
                                      beam_height=BEAM_HEIGHT)
        pts, hit = s.sample_interaction_points(x, y, rng=2)
        fp = np.mean(s.layer_of_points(pts) == 0) * len(pts) / len(x)
        r = simulate_scattering_orders(s, BEAM_WIDTH, BEAM_HEIGHT, n_particles=300000, rng=3)
        rows.append((b, s.beam_fullness()[0], fp, r.single.sum() / r.n_incident,
                     r.multiple.sum() / r.n_incident))
    rows = np.array(rows)
    print("bottom fullness powder single multiple ratio")
    for row in rows:
        print(" ".join(f"{v:8.4f}" for v in row), f"{row[4] / row[3]:8.3f}")
    c = rows[:, 0] + 2.0                                  # centre of the band
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    a1.plot(c, rows[:, 2], "o-", color=COLORS[1], label="interact in the powder")
    a1.plot(c, rows[:, 1] * rows[:, 2].max() / rows[:, 1].max(), "--", color="k", lw=0.9,
            label="beam_fullness (scaled)")
    a1.plot(c, rows[:, 3], "^-", color=BLUE, label="leave after 1 scattering")
    a1.plot(c, rows[:, 4], "v-", color=RED, label=r"leave after $\geq$2 scatterings")
    a1.set_xlabel("centre of the powder band (mm)"); a1.set_ylabel("per incident particle")
    a1.axvspan(-4, 4, color=RED, alpha=0.06)
    a1.grid(alpha=0.3); a1.legend(fontsize=7); a1.set_title("(a) band 4 mm high, beam 8 mm", fontsize=9)
    ok = rows[:, 3] > 0.005
    a2.plot(c[ok], rows[ok, 4] / rows[ok, 3], "o-", color="k")
    a2.axvspan(-4, 4, color=RED, alpha=0.06)
    a2.set_xlabel("centre of the powder band (mm)"); a2.set_ylabel("multiple / single")
    a2.grid(alpha=0.3); a2.set_title("(b) ratio of multiple to single scattering", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_band_position.pdf")


# ------------------------------------------------------- disc geometry (2.3)
def _disc_sample(diameter=20.0, angle=90.0, layers=None):
    """The example disc: the layers of the example slab of run_simulation.py
    (DISC_LAYERS), with a circular section."""
    from disc import create_disc_sample
    from run_simulation import DISC_LAYERS
    return create_disc_sample(diameter, layers or DISC_LAYERS, plane_angle_deg=angle,
                              beam_height=BEAM_HEIGHT)


def fig_disc_geometry():
    """Disc: perspective view of a three-layer disc perpendicular to the beam,
    and the disc seen along the beam when it is rotated (an ellipse)."""
    fig = plt.figure(figsize=(7.6, 3.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    t = np.array([2.0, 1.0, 0.5]); edges = np.concatenate(([0], np.cumsum(t)))
    R = 6.0
    ph = np.linspace(0, 2 * np.pi, 73)
    for i in range(3):
        z0, z1 = edges[i], edges[i + 1]
        side = [[(z0, R * np.cos(a), R * np.sin(a)), (z1, R * np.cos(a), R * np.sin(a)),
                 (z1, R * np.cos(b), R * np.sin(b)), (z0, R * np.cos(b), R * np.sin(b))]
                for a, b in zip(ph[:-1], ph[1:])]
        ax.add_collection3d(Poly3DCollection(side, facecolors=COLORS[i], edgecolors="none", alpha=0.55))
        for z in (z0, z1):
            ax.plot([z] * len(ph), R * np.cos(ph), R * np.sin(ph), color="k", lw=0.6)
    front = [[(0.0, R * np.cos(a), R * np.sin(a)) for a in ph]]
    ax.add_collection3d(Poly3DCollection(front, facecolors=COLORS[0], edgecolors="k", lw=0.6, alpha=0.35))
    bz = -7.0
    ax.add_collection3d(Poly3DCollection([[(bz, -2, -2), (bz, 2, -2), (bz, 2, 2), (bz, -2, 2)]],
                                         facecolors=RED, alpha=0.35, edgecolors=RED))
    ax.quiver(bz, 0, 0, 5.0, 0, 0, color=RED, lw=2, arrow_length_ratio=0.2)
    ax.text(bz, 0, 3.2, "beam", color=RED, ha="center")
    ax.plot([-1, edges[-1] + 2], [0, 0], [0, 0], ls="-.", color="k", lw=0.7)
    ax.text(edges[-1] + 2.2, 0, 0.4, "$z'$", fontsize=9)
    for vec, lab, pos in (((4, 0, 0), "$z$", (-1.6, -8.5, -6.5)), ((0, 4, 0), "$x$", (-7, -4, -6.5)),
                          ((0, 0, 4), "$y$", (-7, -8.5, -2))):
        ax.quiver(-7, -8.5, -6.5, *vec, color="k", arrow_length_ratio=0.2)
        ax.text(*pos, lab)
    ax.set_xlim(-8, 5); ax.set_ylim(-9, 7); ax.set_zlim(-7, 7)
    ax.set_box_aspect((1.0, 1.4, 1.1)); ax.view_init(elev=18, azim=-58); ax.set_axis_off()
    ax.set_title("(a) disc of diameter $D$, perpendicular to the beam", fontsize=9)

    # (b) seen along the beam: tilted disc = ellipse; beam rectangle; filled band
    ax2 = fig.add_subplot(1, 2, 2); ax2.set_aspect("equal")
    om = np.radians(90.0 - 50.0)
    from matplotlib.patches import Ellipse
    ax2.add_patch(Ellipse((0, 0), 2 * R, 2 * R, fill=False, ls=":", ec="0.5", lw=0.8))
    ax2.add_patch(Ellipse((0, 0), 2 * R * np.cos(om), 2 * R, fc=COLORS[0], ec="k", lw=0.9, alpha=0.35))
    # band of material from y = -R to y = 1.5 in the middle layer (shaded)
    yy = np.linspace(-R, 1.5, 200)
    xx = R * np.cos(om) * np.sqrt(1 - (yy / R) ** 2)
    ax2.fill_betweenx(yy, -xx, xx, color=COLORS[1], alpha=0.55, lw=0)
    ax2.plot([-xx[-1], xx[-1]], [1.5, 1.5], color="k", lw=0.8)
    ax2.text(0, -4.6, "material (band)", ha="center", va="center", fontsize=7.5)
    ax2.add_patch(Rectangle((-2.5, -3.0), 5.0, 6.0, fill=False, ec=RED, lw=1.4))
    ax2.text(-2.5, 3.3, "beam", color=RED, fontsize=8)
    ax2.add_patch(Rectangle((-8.0, -8.0), 16.0, 16.0, fill=False, ec=RED, lw=0.9, ls="--"))
    ax2.text(-7.8, 8.3, "wide beam: only the ellipse is hit", color=RED, fontsize=7.5)
    arrow(ax2, (0, 0), (0, R), style="<|-|>", lw=0.7, ms=6)
    ax2.text(0.25, R * 0.55, "$R$", fontsize=9)
    arrow(ax2, (0, -R - 0.9), (R * np.cos(om), -R - 0.9), style="<|-|>", lw=0.7, ms=6)
    ax2.text(R * np.cos(om) / 2, -R - 1.9, r"$R\cos\omega$", fontsize=8.5, ha="center")
    ax2.text(R + 0.3, -R + 0.3, "disc face\n(if $\\alpha$ = 90°)", fontsize=7, color="0.45")
    ax2.set_xlim(-9.5, 9.5); ax2.set_ylim(-9.3, 9.3); ax2.axis("off")
    ax2.set_title(r"(b) seen along the beam, $\alpha$ = 50$^\circ$ ($\omega$ = 40$^\circ$)", fontsize=9)
    save(fig, "fig_disc_geometry.pdf")


def fig_disc_results():
    """Results for the example disc, made with the package functions:
    transmission for several orientations, single/multiple scattering,
    transmission by layer, and the comparison disc / square slab."""
    from slab import create_slab_sample
    from run_simulation import DISC_LAYERS
    gdisc = _disc_sample()
    curves = {}
    for a in (90, 60, 45, 30):
        ang, T, _ = transmission_vs_angle(gdisc.with_plane_angle(a), BEAM_WIDTH, BEAM_HEIGHT,
                                          n_particles=50000, rng=1)
        curves[f"beam/plane angle = {a}°"] = T
    plot_transmission(ang, curves, title="Disc (D = 20 mm): transmission vs angle",
                      save_as=os.path.join(OUT, "fig_transmission_disc.pdf"), show=False)
    res = simulate_scattering_orders(gdisc, BEAM_WIDTH, BEAM_HEIGHT, n_particles=1000000, rng=2)
    plot_scattering_orders(res, logy=True, title="Disc: single and multiple scattering",
                           save_as=os.path.join(OUT, "fig_orders_disc.pdf"), show=False)
    print("disc incident", res.n_incident, "no-interaction", res.n_no_interaction, "absorbed",
          res.n_absorbed, "exited", res.n_exited, "single", int(res.single.sum()),
          "multiple", int(res.multiple.sum()))
    r = transmission_by_last_layer(gdisc, BEAM_WIDTH, BEAM_HEIGHT, n_particles=60000, rng=3)
    plot_transmission_by_layer(r, title="Disc: transmission by layer of last scattering",
                               save_as=os.path.join(OUT, "fig_bylayer_disc.pdf"), show=False)

    # disc against the square slab of side D, normal incidence, beam 8 x 8 mm
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for k, (D, ls) in enumerate(((20.0, "-"), (10.0, "--"))):
        sq = create_slab_sample(D, D, DISC_LAYERS, beam_height=BEAM_HEIGHT)
        ang, Td, _ = transmission_vs_angle(_disc_sample(D), BEAM_WIDTH, BEAM_HEIGHT,
                                           n_particles=60000, rng=1)
        ang, Ts, _ = transmission_vs_angle(sq, BEAM_WIDTH, BEAM_HEIGHT, n_particles=60000, rng=1)
        a1.plot(ang, Td, ls, color=COLORS[0], label=f"disc, D = {D:g} mm")
        a1.plot(ang, Ts, ls, color=COLORS[1], label=f"square slab, side {D:g} mm")
        i90 = np.argmin(np.abs(ang - 90))
        print(f"D={D}: T(90) disc {Td[i90]:.4f} slab {Ts[i90]:.4f}; T(60) disc {Td[i90-30]:.4f} slab {Ts[i90-30]:.4f}")
    a1.set_xlabel(r"$2\theta$ (deg)"); a1.set_ylabel("transmission"); a1.set_xlim(0, 180)
    a1.grid(alpha=0.3); a1.legend(fontsize=7); a1.set_title("(a) transmission (single scattering)", fontsize=9)
    sizes = np.array([9.0, 12.0, 16.0, 20.0, 30.0, 40.0])
    rd, rs = [], []
    for D in sizes:
        for smp, store in ((_disc_sample(D), rd),
                           (create_slab_sample(D, D, DISC_LAYERS, beam_height=BEAM_HEIGHT), rs)):
            o = simulate_scattering_orders(smp, BEAM_WIDTH, BEAM_HEIGHT, n_particles=300000, rng=4)
            store.append(o.multiple.sum() / o.single.sum())
    print("sizes", sizes, "ratio disc", np.round(rd, 4), "slab", np.round(rs, 4))
    a2.plot(sizes, rd, "o-", color=COLORS[0], label="disc of diameter D")
    a2.plot(sizes, rs, "s--", color=COLORS[1], label="square slab of side D")
    a2.set_xlabel("D (mm)"); a2.set_ylabel("multiple / single"); a2.grid(alpha=0.3)
    a2.legend(fontsize=7); a2.set_title("(b) multiple scattering, beam 8 x 8 mm", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_disc_vs_slab.pdf")


def fig_disc_validation():
    """Fraction of a beam that hits a thin, black disc against the area of
    its face seen along the beam, versus beam size, for three orientations."""
    from disc import create_disc_sample
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    sizes = np.array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0])
    for k, ang in enumerate((90.0, 60.0, 30.0)):
        thin = create_disc_sample(20.0, [("black", 0.01, 1e4)], plane_angle_deg=ang)
        mc, ex = [], []
        for w in sizes:
            x, y = sample_beam_point(w, w, n=200000, rng=int(w) + k)
            _, hit = thin.sample_interaction_points(x, y, rng=7)
            mc.append(hit.mean()); ex.append(thin.face_area_in_beam(w, w) / (w * w))
        ax.plot(sizes, ex, "-", color=COLORS[k], lw=1.0)
        ax.plot(sizes, mc, "o", color=COLORS[k], mfc="white", label=f"plane angle {ang:g}°")
        print("angle", ang, "MC", np.round(mc, 4), "exact", np.round(ex, 4))
    ax.set_xlabel("side of a square beam (mm)"); ax.set_ylabel("fraction of the beam that hits the disc")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Disc D = 20 mm: simulation (points) and face area (lines)", fontsize=9)
    save(fig, "fig_disc_validation.pdf")


# ----------------------------------------------------- sphere geometry (2.4)
def _sphere_sample(layers=None):
    """The example sphere: the layers of run_simulation.py (SPHERE_LAYERS)."""
    from sphere import create_spherical_sample
    from run_simulation import SPHERE_LAYERS
    return create_spherical_sample(layers or SPHERE_LAYERS, beam_height=BEAM_HEIGHT)


def fig_sphere_geometry():
    """Sphere: cut through the centre with the radii and the beam, and a
    half-filled spherical flask in side view."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.0))
    radii = [3.0, 4.0, 4.5]
    for ax in (a1, a2):
        ax.set_aspect("equal"); ax.axis("off")
    # (a) cut through the centre, any plane containing the beam
    for i in reversed(range(3)):
        a1.add_patch(Circle((0, 0), radii[i], fc=COLORS[i], ec="k", lw=0.8, alpha=0.7))
    a1.add_patch(Rectangle((-10.5, -2.5), 3.0, 5.0, fc=RED, alpha=0.18, ec=RED))
    for yy in (-1.6, 0.0, 1.6):
        arrow(a1, (-10.3, yy), (-7.6, yy), color=RED, lw=1.0, ms=7)
    a1.text(-9.0, 3.0, "beam", color=RED, ha="center", fontsize=8)
    a1.plot([0], [0], "ko", ms=3)
    for i, ang in enumerate((60, 30, 0)):
        t = np.radians(ang)
        arrow(a1, (0, 0), (radii[i] * np.cos(t), radii[i] * np.sin(t)), lw=0.7, ms=6)
        a1.text(radii[i] * np.cos(t) + 0.25, radii[i] * np.sin(t) + 0.2, f"$r_{i + 1}$", fontsize=8.5)
    a1.text(0, -5.4, "inner sphere: thickness = diameter $2r_1$\nshells: $r_k = r_{k-1} + t_k$",
            ha="center", va="top", fontsize=7.5)
    a1.text(5.2, 3.6, "centre = beam axis\n= centre of the instrument", fontsize=7, color="0.35")
    a1.set_xlim(-11, 10); a1.set_ylim(-8.2, 6.5)
    a1.set_title("(a) concentric spheres, cut through the centre", fontsize=9)
    # (b) half-filled flask: water up to y = 0, glass wall full
    Rw, Rg = 4.0, 4.5
    a2.add_patch(Circle((0, 0), Rg, fc=COLORS[0], ec="k", lw=0.8, alpha=0.6))
    a2.add_patch(Circle((0, 0), Rw, fc="white", ec="k", lw=0.6, ls="--"))
    th = np.linspace(np.pi, 2 * np.pi, 100)
    a2.add_patch(Polygon(np.column_stack([Rw * np.cos(th), Rw * np.sin(th)]), closed=True,
                         fc=COLORS[1], ec="k", lw=0.6, alpha=0.8))
    a2.text(0, -1.8, "water", ha="center", fontsize=8)
    a2.text(0, 1.8, "empty", ha="center", fontsize=8, color="0.4")
    a2.text(Rg + 0.3, 2.8, "glass", fontsize=8)
    for yy in (-3.0, 3.0):
        a2.plot([-11, 10], [yy, yy], ls=":", color="0.5", lw=0.7)
    a2.add_patch(Rectangle((-10.5, -3.0), 3.0, 6.0, fc=RED, alpha=0.18, ec=RED))
    a2.text(-9.0, 3.5, "beam", color=RED, ha="center", fontsize=8)
    a2.plot([-Rw - 1.5, Rw + 1.5], [0, 0], color="k", lw=0.8)
    a2.text(Rw + 1.6, 0, "top $t = 0$", va="center", fontsize=7.5)
    a2.text(Rw + 1.6, -Rw, "bottom $b = -r_1$", va="center", fontsize=7.5)
    a2.text(-4.5, -6.6, "water: bottom = -$r_1$, height = $r_1$ (fullness 0.5)", fontsize=7.5)
    a2.set_xlim(-11, 10); a2.set_ylim(-8.2, 6.5)
    a2.set_title("(b) half-filled spherical flask, side view", fontsize=9)
    save(fig, "fig_sphere_geometry.pdf")


def fig_sphere_results():
    """Results for the example sphere, and the attenuation factor of a sphere
    compared with that of a cylinder (direct method)."""
    from cylinder import create_cylindrical_sample
    from sphere import create_spherical_sample
    from direct_attenuation import attenuation_factors_direct
    gs = _sphere_sample()
    ang, T, f = transmission_vs_angle(gs, BEAM_WIDTH, BEAM_HEIGHT, n_particles=50000, rng=1)
    plot_transmission(ang, T, title="Sphere: transmission vs angle",
                      save_as=os.path.join(OUT, "fig_transmission_sphere.pdf"), show=False)
    print("sphere interaction fraction", round(f, 3))
    res = simulate_scattering_orders(gs, BEAM_WIDTH, BEAM_HEIGHT, n_particles=1000000, rng=2)
    plot_scattering_orders(res, logy=True, title="Sphere: single and multiple scattering",
                           save_as=os.path.join(OUT, "fig_orders_sphere.pdf"), show=False)
    print("sphere incident", res.n_incident, "no-interaction", res.n_no_interaction, "absorbed",
          res.n_absorbed, "exited", res.n_exited, "single", int(res.single.sum()),
          "multiple", int(res.multiple.sum()))
    r = transmission_by_last_layer(gs, BEAM_WIDTH, BEAM_HEIGHT, n_particles=60000, rng=3)
    plot_transmission_by_layer(r, title="Sphere: transmission by layer of last scattering",
                               save_as=os.path.join(OUT, "fig_bylayer_sphere.pdf"), show=False)

    # A(2theta) of a homogeneous sphere and of a long cylinder of the same
    # radius (beam wider than both, 3 mm radius), for three values of mu R
    Rr = 3.0
    angs = np.arange(0.0, 181.0, 1.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    out = {}
    for k, muR in enumerate((0.5, 1.0, 2.0)):
        mu = muR / Rr
        sph = create_spherical_sample([("s", 2 * Rr, mu)])
        cyl = create_cylindrical_sample(60.0, [("c", 2 * Rr, mu)])
        As = attenuation_factors_direct(sph, 8, 8, angles_deg=angs, n_points=60000, rng=10 + k).factors[0]
        Ac = attenuation_factors_direct(cyl, 8, 8, angles_deg=angs, n_points=60000, rng=20 + k).factors[0]
        out[muR] = (As, Ac)
        a1.plot(angs, As, "-", color=COLORS[k], label=f"sphere, $\\mu R$ = {muR:g}")
        a1.plot(angs, Ac, "--", color=COLORS[k], label=f"cylinder, $\\mu R$ = {muR:g}")
        a2.plot(angs, As / As[0], "-", color=COLORS[k], label=f"sphere, $\\mu R$ = {muR:g}")
        a2.plot(angs, Ac / Ac[0], "--", color=COLORS[k], label=f"cylinder, $\\mu R$ = {muR:g}")
        print(f"muR={muR}: sphere A(0,90,180) = {As[0]:.4f} {As[90]:.4f} {As[180]:.4f}; "
              f"cylinder {Ac[0]:.4f} {Ac[90]:.4f} {Ac[180]:.4f}")
    for ax, ttl, yl in ((a1, "(a) attenuation factor A(2θ)", "A"),
                        (a2, "(b) normalised to 2θ = 0", "A(2θ) / A(0)")):
        ax.set_xlabel(r"$2\theta$ (deg)"); ax.set_ylabel(yl); ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 30)); ax.grid(alpha=0.3); ax.set_title(ttl, fontsize=9)
    a2.legend(fontsize=6.5, ncol=1, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_sphere_vs_cylinder.pdf")


def fig_sphere_validation():
    """Fraction of a wide beam that interacts in a homogeneous sphere, versus
    mu R: simulation against the exact integral."""
    from sphere import create_spherical_sample
    Rr, w = 3.0, 8.0
    trapz = getattr(np, "trapezoid", None) or np.trapz
    u = np.linspace(0, Rr, 20001)
    muRs = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    mc, ex = [], []
    x, y = sample_beam_point(w, w, n=200000, rng=5)
    for muR in muRs:
        mu = muR / Rr
        smp = create_spherical_sample([("s", 2 * Rr, mu)])
        _, hit = smp.sample_interaction_points(x, y, rng=6)
        mc.append(hit.mean())
        ex.append(2 * np.pi * trapz((1 - np.exp(-2 * mu * u)) * u, u) / (w * w))
    print("muR", muRs, "MC", np.round(mc, 4), "exact", np.round(ex, 4))
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.semilogx(muRs, ex, "-", color=COLORS[0], label="exact integral")
    ax.semilogx(muRs, mc, "o", color=COLORS[0], mfc="white", label="simulation, $2\\times10^5$ particles")
    ax.axhline(np.pi * Rr ** 2 / w ** 2, color="0.5", ls=":", lw=0.9, label=r"$\pi R^2 / (w_b h_b)$ (black sphere)")
    ax.set_xlabel(r"$\mu R$"); ax.set_ylabel("fraction of the beam that interacts")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    ax.set_title("Sphere R = 3 mm in an 8 x 8 mm beam", fontsize=9)
    save(fig, "fig_sphere_validation.pdf")


# ----------------------------------------------- CORRECT configurations (2.5)
def fig_correct_configs():
    """Top views of the three measurements sch, ch, h for the two cylinder
    cases of the chapter on CORRECT factors (radii not to scale)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    col = {"s": COLORS[1], "c": "#8DA0B8", "h": "#54A24B", "-": "white"}
    cases = [("cylinder", [("s", 1.6), ("c", 1.85), ("-", 4.0), ("h", 4.25)], ["s", "c", "-", "h"]),
             ("annulus", [("-", 0.9), ("c", 1.15), ("s", 2.0), ("c", 2.25), ("-", 4.0), ("h", 4.25)],
              ["-", "c", "s", "c", "-", "h"])]
    for row, (name, rings, roles) in enumerate(cases):
        for j, meas in enumerate(("sch", "ch", "h")):
            ax = axes[row, j]; ax.set_aspect("equal"); ax.axis("off")
            from matplotlib.patches import Wedge
            r_in = 0.0
            for role, r in rings:              # each layer is a ring from r_in to r
                present = role in meas and role != "-"
                ax.add_patch(Wedge((0, 0), r, 0, 360, width=r - r_in,
                                   fc=col[role] if present else "white",
                                   ec="k" if present else "0.6", lw=0.7 if present else 0.5,
                                   ls="-" if present else "--", alpha=0.85))
                r_in = r
            arrow(ax, (-5.3, 0), (-4.4, 0), color=RED, lw=1.2, ms=8)
            ax.set_xlim(-5.6, 4.6); ax.set_ylim(-4.6, 4.6)
            ax.set_title(f"{name}: $I_{{{meas}}}$", fontsize=9)
    from matplotlib.patches import Patch
    handles = [Patch(fc=col["s"], ec="k", label="sample (s)"), Patch(fc=col["c"], ec="k", label="container (c)"),
               Patch(fc=col["h"], ec="k", label="heater (h)"),
               Patch(fc="white", ec="0.6", ls="--", label="empty (mu = 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_correct_configs.pdf")


if __name__ == "__main__":
    table = {"modules": fig_modules, "instrument": fig_instrument, "beam": fig_beam,
             "slab": fig_slab_geometry, "cylinder": fig_cylinder_geometry,
             "trajectories": fig_trajectories, "results": fig_results, "validation": fig_validation,
             "partial_geometry": fig_partial_geometry,
             "partial_trajectories": fig_partial_trajectories,
             "partial_results": fig_partial_results,
             "band_geometry": fig_band_geometry, "band_position": fig_band_position,
             "disc_geometry": fig_disc_geometry, "disc_results": fig_disc_results,
             "disc_validation": fig_disc_validation,
             "sphere_geometry": fig_sphere_geometry, "sphere_results": fig_sphere_results,
             "sphere_validation": fig_sphere_validation,
             "correct_configs": fig_correct_configs}
    for key in (sys.argv[1:] or list(table)):
        table[key]()
