"""
Tests for pycorrect.   Run with:   python -m pytest -q tests   (or python tests/test_pycorrect.py)

1. Paalman & Pings factors from atcors() against an independent brute-force
   integration on a fine Cartesian grid (exact path lengths in a disc/annulus).
2. Round trip of the whole 2-axis chain: synthetic raw data are built with the
   forward model  I_sch = A_s,sch I_s + A_c,sch I_c,  I_ch = A_c,ch I_c,
   I_v = A_v (1 + delta_v) I_v0 ; CORRECT must give back the input dsigma/dOmega.
3. Command parser.
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pycorrect import Correct, atcors, mscat
from pycorrect.fileio import DataWriter, read_file
from pycorrect.parse_line import parse_line


# ---------------------------------------------------------------- helpers
def brute_force_pp(tt, R1, R2, smu, cmu, n=1500):
    """A_s,sc, A_c,sc, A_c,c by direct summation over a fine grid.
    Beam along +x, detector at angle tt in the xy plane."""
    xs = np.linspace(-R2, R2, n)
    X, Y = np.meshgrid(xs, xs)
    dA = (xs[1] - xs[0]) ** 2
    r = np.hypot(X, Y)

    def path(px, py, ux, uy):
        """(length in sample, length in can) from point to the outside along u"""
        b = px * ux + py * uy
        L2 = -b + np.sqrt(np.maximum(b * b - (px * px + py * py - R2 * R2), 0))
        c1 = px * px + py * py - R1 * R1
        disc = np.maximum(b * b - c1, 0)
        inside = c1 < 0
        ahead = (-b - np.sqrt(disc)) > 0
        ls = np.where(inside, -b + np.sqrt(disc), np.where((b * b - c1 > 0) & ahead, 2 * np.sqrt(disc), 0))
        return ls, L2 - ls

    si, ci = path(X, Y, -1.0, 0.0)                    # back along incoming beam
    so, co = path(X, Y, math.cos(tt), math.sin(tt))   # out to detector
    att = np.exp(-smu * (si + so) - cmu * (ci + co))
    ms, mc = r < R1, (r >= R1) & (r < R2)
    assc = att[ms].sum() * dA / (math.pi * R1 ** 2)
    acsc = att[mc].sum() * dA / (math.pi * (R2 ** 2 - R1 ** 2))
    acc = np.exp(-cmu * (ci + co))[mc].sum() * dA / (math.pi * (R2 ** 2 - R1 ** 2))
    return assc, acc, acsc


def test_paalman_pings_vs_brute_force():
    for deg in (0, 45, 90, 135):
        t = math.radians(deg)
        a = atcors(t, 0.30, 0.35, 0.0, 0.0, 4.0, 0.8, 0.5, 0.0)
        b = brute_force_pp(t, 0.30, 0.35, 0.8, 0.5)
        assert abs(a[0] - b[0]) / b[0] < 1e-3, (deg, a[0], b[0])   # A_s,sc
        assert abs(a[1] - b[1]) / b[1] < 3e-3, (deg, a[1], b[1])   # A_c,c
        assert abs(a[2] - b[2]) / b[2] < 3e-3, (deg, a[2], b[2])   # A_c,sc


def test_mscat_limits():
    assert mscat(0.3, 3.0, 1e-6, 1e-6, verbose=False) < 1e-4       # no ms at mu=0
    d_scat = mscat(0.3, 3.0, 0.8, 0.8, verbose=False)
    d_abs = mscat(0.3, 3.0, 0.8, 0.4, verbose=False)
    assert d_scat > d_abs > 0                                       # scales with smu/tmu


def test_parser():
    pl = parse_line('Sample MyFile.dat 0.3 /dens=0.07 /temp=20')
    assert pl.verb == "sample"
    assert pl.param == ["MyFile.dat", "0.3"] and pl.rparam[1] == 0.3
    assert pl.qual == ["/density", "/temperature"] and pl.rqval == [0.07, 20.0]
    assert parse_line("! comment").verb == ""


def test_roundtrip_2axis():
    tmp = tempfile.mkdtemp()
    lam, R1, R2, Rv, H = 0.5, 0.30, 0.32, 0.32, 3.0
    theta = np.arange(2.0, 130.0, 0.5)
    q = 4 * math.pi / lam * np.sin(np.radians(theta) / 2)
    flux = 1e4 * (1 + 0.2 * np.cos(np.radians(theta)))       # incident x efficiency

    # cross sections: sample = Si (2.171 b scatt, 0.171 b abs), V can and V rod
    sa_rho, sa_s, sa_a = 0.05, 2.171, 0.171
    va_rho, va_s, va_a = 0.0722, 4.95, 5.08
    tmu = lambda rho, s, a: rho * (s + a * lam / 1.798)
    sa_tmu, ca_tmu, va_tmu = tmu(sa_rho, sa_s, sa_a), tmu(va_rho, va_s, va_a), tmu(va_rho, va_s, va_a)

    dcs_true = sa_s / 4 / math.pi * (1 + 0.3 * np.sin(q * 2.5) * np.exp(-0.1 * q))
    # attenuation factors on the grid (computed exactly, no interpolation)
    A = np.array([atcors(math.radians(t), R1, R2, 0, 0, 4.0, sa_tmu, ca_tmu, 0) for t in theta])
    Av = np.array([atcors(math.radians(t), Rv, 0, 0, 0, 4.0, va_tmu, 0, 0)[0] for t in theta])
    fs = (R1 / R2) ** 2
    d_sa = mscat(R2, H, (1 - fs) * ca_tmu + fs * sa_tmu,
                 (1 - fs) * va_rho * va_s + fs * sa_rho * sa_s, verbose=False)
    d_va = mscat(Rv, H, va_tmu, va_rho * va_s, verbose=False)
    # forward model (I per unit flux and per unit length of beam)
    I_v = flux * va_rho * Rv ** 2 * va_s / 4 / math.pi * Av * (1 + d_va)
    I_s = flux * sa_rho * R1 ** 2 * (dcs_true + d_sa * sa_s / 4 / math.pi)
    I_c = flux * va_rho * (R2 ** 2 - R1 ** 2) * va_s / 4 / math.pi
    raw_can = A[:, 1] * I_c
    raw_sam = A[:, 0] * I_s + A[:, 2] * I_c
    for name, y in (("van", I_v), ("can", raw_can), ("sam", raw_sam)):
        with DataWriter(os.path.join(tmp, name + ".dat")) as w:
            w.append(theta, y, np.sqrt(np.abs(y)), title=name, fnx="Angle")
    com = os.path.join(tmp, "run.com")
    with open(com, "w") as fh:
        fh.write(f"""instrument d4
wavelength {lam}
beam {H} 4.0
sample {tmp}/sam.dat {R1} /density={sa_rho}
component 1 si
container {tmp}/can.dat {R2} /type=v
vanadium {tmp}/van.dat {Rv} /smoothing=0
spectrum 1
output {tmp}/out.dat
execute /nolist
""")
    c = Correct(verbose=False, ask=False)
    c.run_file(com)
    res = c.results[0]
    rel = np.abs(res["dcs"] - dcs_true) / dcs_true
    assert rel.max() < 2e-3, rel.max()
    back = read_file(os.path.join(tmp, "out.dat"), 1)
    assert np.allclose(back.y, res["dcs"], rtol=1e-6)


def test_plate_factors():
    """Flat plate: analytic factors (infinite slab, normal incidence).
    Reflection geometry: A_c,sc = average of the front wall (no sample on the
    path) and the back wall (sample and front wall, in and out)."""
    from pycorrect import plate
    t, d, ms, mc = 0.2, 0.02, 1.8, 0.46                     # cm, cm, 1/cm, 1/cm
    tt = math.radians(110.0)
    k = 1 + 1 / abs(math.cos(tt))                         # in + out path factor
    g = (1 - math.exp(-mc * d * k)) / (mc * d * k)        # wall average
    acsc = 0.5 * (g + math.exp(-(mc * d + ms * t) * k) * g)
    assc, acc, ac = plate(tt, 0.0, t, t + 2 * d, ms, mc)
    assert abs(ac - acsc) < 1e-12, (ac, acsc)
    assert abs(acc - 0.5 * (g + math.exp(-mc * d * k) * g)) < 1e-12
    # 2theta = 0: both walls see both walls
    assc0, acc0, _ = plate(0.0, 0.0, t, t + 2 * d, ms, mc)
    assert abs(acc0 - math.exp(-2 * mc * d)) < 1e-12
    # faithful=True reproduces CORRECT 2.24 (different numbers)
    assert plate(tt, 0.0, t, t + 2 * d, ms, mc, faithful=True)[2] > ac + 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
