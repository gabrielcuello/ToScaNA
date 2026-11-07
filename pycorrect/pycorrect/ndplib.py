"""
ndplib.py  --  Python translation of  ndplib.f  (CORRECT 2.24, "lib" folder)
==============================================================================

This module contains the two pieces of physics that CORRECT is really about:

1.  **Paalman & Pings attenuation factors** for a cylindrical sample, inside a
    cylindrical container, inside a cylindrical furnace / cryostat shield.

        H.H. Paalman and C.J. Pings, J. Appl. Phys. 33 (1962) 2635.

    Fortran:  SUBROUTINE ATCORS  (+ ATCOR$GXY3, ATCOR$GXY3x, ATCOR$GXY3z,
              ATCOR$F1..F17, ATCOR$G1..G19, ATCOR$AA, ATCOR$BB, ATCOR$CC,
              ATCOR$ARCS, ATCOR$ARCC, VIS, TRANS, ETA, AREA, BEAMWX, BEAMWZ)
    Python :  function `atcors()` and the class `_AtcorsIntegrator`.

    Notation (identical to Fortran and to Paalman & Pings):
        first subscript  = where the neutron is scattered
        letters after ','= which bodies attenuate it
        s = sample, c = container (can), h = heater (furnace)

        A_s,sch : scattered in sample,    attenuated by sample+can+furnace
        A_c,sch : scattered in container, attenuated by sample+can+furnace
        A_c,ch  : scattered in container, attenuated by can+furnace (empty can)
        A_h,sch : scattered in furnace,   attenuated by sample+can+furnace
        A_h,ch  : scattered in furnace,   attenuated by can+furnace
        A_h,h   : scattered in furnace,   attenuated by furnace only

    Each factor is   A = (1/Area) * Integral_Area  exp(-sum_i mu_i * L_i) dA
    where L_i is the total path (in + out) inside body i.  The integral is
    done in polar coordinates (r, omega) with 9-point Gauss-Legendre in both
    variables (that is the 9 "U" nodes and "H" weights of the Fortran).

    Poncet's extensions present in this version (and reproduced here):
      * no container  -> give r2 <= r1
      * no furnace    -> give r4 <= r3
      * beam narrower than the sample/can (weights set to zero outside beam:
        ATCOR$GXY3x/z + BEAMWX/BEAMWZ)
      * visibility function of an oscillating radial collimator (VIS)

    The furnace factors are computed at 2theta = 0, 90 and 180 deg only and
    interpolated with a parabola in 2theta (exactly as the Fortran does).

2.  **Blech & Averbach multiple scattering** for a cylinder.

        I.A. Blech and B.L. Averbach, Phys. Rev. 137 (1965) A1113.

    Fortran:  SUBROUTINE MSCAT  (+ MULSCS, H2EL)
    Python :  function `mscat()`  (+ `mulscs()`, `h2el()`)

    The file  doublescattering.dat  is the Blech-Averbach table of the ratio
    double/single scattering, delta_2, versus mu*R (37 values) and R/H
    (20 values).  MSCAT interpolates it, multiplies by sigma_s/sigma_t (the
    probability that the second event is a scattering), and adds the higher
    orders with an empirical power series in delta_2:

        delta = d + 0.69396 d^2 + 0.37011 d^3 + 0.16053 d^4 + 0.05871 d^5

    Outside the table MULSCS computes delta_2 directly by a 6-D Gauss
    integration (the Blech-Averbach integrals).

Angles inside this module are in RADIANS and represent the scattering angle
2theta (the Fortran variable is called THETA but it is the full 2theta).
Lengths are in cm and linear attenuation coefficients mu in cm^-1.

NumPy functions used (quick reference)
---------------------------------------
numpy.polynomial.legendre.leggauss(n) : nodes and weights of n-point Gauss-
                                         Legendre quadrature on [-1,1]. For n=9
                                         it returns exactly the U(9), H(9)
                                         constants hard-coded in ATCORS/MULSCS.
numpy.sqrt / sin / cos / exp / arcsin : element-wise maths on arrays.
numpy.maximum(a, b)                   : element-wise max (Fortran MAX on arrays).
numpy.clip(a, lo, hi)                 : limit values to [lo, hi].
numpy.errstate(...)                   : context manager that silences floating
                                         point warnings (e.g. 0/0) locally.
numpy.full_like(a, v)                 : array with the shape of `a` filled with v.
numpy.newaxis (None in indexing)      : adds an axis so that arrays broadcast.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

PI = math.pi

# --------------------------------------------------------------------------
# 9-point Gauss-Legendre nodes (U) and weights (H) on [-1, 1].
# Fortran:  U(1..9), H(1..9) in COMMON /CAP11/  (and LAMBDA/WEIGHT in MULSCS)
# The Fortran lists +0.968.., +0.836.., ... , 0, -0.324.., ...; the order does
# not matter for a quadrature sum.  leggauss gives the identical numbers.
# --------------------------------------------------------------------------
GL_U, GL_H = np.polynomial.legendre.leggauss(9)


# ==========================================================================
#  Radial oscillating collimator  (Fortran COMMON /coll1/../coll4/)
# ==========================================================================
@dataclass
class Collimator:
    """Parameters of the radial oscillating collimator.

    Fortran correspondence (correct_224.inc):
        enabled      <- correct_coll   (COMMON /coll3/)
        r1, r2       <- r1_coll, r2_coll  inner/outer radius of the blades [cm]
        rd           <- r_detector        sample-detector distance [cm]
        blade_space  <- blade_space       blade spacing at r1 [cm]
        blade_thick  <- blade_thick       blade thickness [cm]
        r_offset     <- r_offset          offset of collimator axis [cm]
        phi_offset   <- phi_offset        direction of that offset [rad]
        alpha2, beta1, beta2 <- alpha2_coll, beta1_coll, beta2_coll
                       (derived quantities computed in OBEY_LINE, verb
                        'collimator'; here computed by `update()`).
    """
    enabled: bool = False
    r1: float = 15.0
    r2: float = 25.0
    rd: float = 100.0
    blade_space: float = 0.5
    blade_thick: float = 0.05
    r_offset: float = 0.0
    phi_offset: float = 0.0
    alpha2: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.0

    def update(self) -> None:
        """Derived angles.  Fortran: obey_line, verb 'collimator'."""
        self.alpha2 = 2.0 * math.atan(0.5 * self.blade_space / self.r1)
        self.beta1 = 0.5 * (self.alpha2 - self.blade_thick / self.r1)
        self.beta2 = 0.5 * (self.alpha2 - self.blade_thick / self.r2)


NO_COLLIMATOR = Collimator(enabled=False)


def _trans(b, col: Collimator):
    """Transmission of the collimator for impact parameter b.

    Fortran: FUNCTION TRANS(B).  Vectorised over b.
    np.arcsin needs |x|<=1; Fortran would produce NaN beyond that, here the
    argument is clipped (the geometry never needs |b|>r1 in practice).
    """
    g1up = np.arcsin(np.clip(b / col.r1, -1, 1)) - col.beta1
    g2up = np.arcsin(np.clip(b / col.r2, -1, 1)) - col.beta2
    g1do = g1up + 2.0 * col.beta1
    g2do = g2up + 2.0 * col.beta2
    gmin = np.maximum(g1up, g2up)
    gmax = np.minimum(g1do, g2do)
    return np.maximum(0.0, gmax - gmin) / col.alpha2


def _vis(r, phi, phid, col: Collimator, d: float):
    """Visibility of a volume element at (r, phi) for a detector at 2theta=phid.

    Fortran: FUNCTION VIS(R,PHI,PHID) which calls TRANS and ETA.
    ETA(phip,rp) = 1 if |rp*sin(phip)| <= d (point inside the beam), else 0.
    Returns 1 everywhere when no collimator correction is requested.
    """
    if not col.enabled:
        return 1.0
    rp = np.asarray(r, dtype=float)
    phip = np.asarray(phi, dtype=float)
    if col.r_offset != 0.0:
        ro, phio = col.r_offset, col.phi_offset
        rp = np.maximum(ro, np.sqrt(r * r + ro * ro + 2 * r * ro * np.cos(phio - phi)))
        with np.errstate(divide="ignore", invalid="ignore"):
            phip = np.where(rp != 0,
                            phi + np.arcsin(np.clip(ro * np.sin(phio - phi) / rp, -1, 1)),
                            0.0)
    psi = phip - phid
    eps = np.arctan(rp * np.sin(psi) / (col.rd - rp * np.cos(psi)))
    v = _trans(col.rd * np.sin(eps), col)
    eta = (np.abs(rp * np.sin(phip)) <= d).astype(float)   # FUNCTION ETA
    return v * eta


# ==========================================================================
#  Elementary geometric functions
# ==========================================================================
def _arcs(x):
    """Fortran ATCOR$ARCS(x): arcsine written as atan(x/sqrt(1-x^2)), returning
    pi/2 when 1-x^2 <= 0.  For 0<=x<=1 this is exactly arcsin(x)."""
    return np.arcsin(np.clip(x, -1.0, 1.0))


def _arcc(x):
    """Fortran ATCOR$ARCC(x) = pi/2 - ARCS(x)  (= arccos for 0<=x<=1)."""
    return 0.5 * PI - _arcs(x)


def _aa(rr, r, omega):
    """Fortran ATCOR$AA: half-chord of a circle of radius rr along the
    *incident* direction through the point (r, omega):  sqrt(rr^2 - r^2 sin^2 w).
    Negative arguments are set to zero (point outside the circle)."""
    s = np.sin(omega)
    return np.sqrt(np.maximum(rr * rr - r * r * s * s, 0.0))


def _bb(rr, r, omega, theta):
    """Fortran ATCOR$BB: same as AA but along the *scattered* direction
    (rotated by the scattering angle theta = 2theta)."""
    s = np.sin(omega - theta)
    return np.sqrt(np.maximum(rr * rr - r * r * s * s, 0.0))


def _cc(r, omega, theta):
    """Fortran ATCOR$CC: projection correction r*(cos(w-theta) - cos(w)).
    Combined with AA and BB it gives the in+out path lengths."""
    return r * (np.cos(omega - theta) - np.cos(omega))


def area(r: float, d: float) -> float:
    """Area of a circle of radius r intercepted by a beam of half-width d.

    Fortran: FUNCTION AREA(R,D)
    """
    if d >= r:
        return PI * r * r
    return 2.0 * (r * r * math.asin(d / r) + d * math.sqrt(r * r - d * d))


def _beamw(r: float, theta: float, d: float, shifted: bool):
    """Angular integration limits (in omega) of the part of a ring of radius r
    that lies inside a beam of half-width d.

    Fortran: SUBROUTINE BEAMWX (shifted=False) and BEAMWZ (shifted=True;
    BEAMWZ is the same construction reflected about the symmetry line,
    i.e. all limits rotated by theta).

    Returns (bw, bw2, w1, w2, w3, w4):
        bw  : True if the ring is partly outside the beam (d/r <= 1)
        bw2 : True if there is a second omega interval [w3, w4]
    """
    uu = d / r
    if uu > 1.0:
        return False, False, 0.0, 0.0, 0.0, 0.0
    wa = math.asin(uu)
    wb = PI - wa
    wc = wa + PI
    wd = wb + PI
    if shifted:
        wa += theta
        wb += theta
        wc += theta
        wd += theta
    t = 0.5 * PI + 0.5 * theta
    if t <= wa:
        return True, True, t, wa, wb, t + PI
    elif t <= wb:
        return True, False, wb, wc, 0.0, 0.0
    elif t <= wc:
        return True, True, t, wc, wd, t + PI
    else:
        return True, False, wd, wa + 2.0 * PI, 0.0, 0.0


# ==========================================================================
#  The integrator: holds what the Fortran keeps in COMMON /CAP1/../CAP12/
# ==========================================================================
class _AtcorsIntegrator:
    """State + integrands of ATCORS.

    The Fortran passes information to the integrands F1..F17 and to the
    integration limits G1..G19 through COMMON blocks.  Here they are
    attributes of this object:

        self.i       <- COMMON /CAP1/ I    (loop index! selects which bodies
                                            attenuate: 1 can only, 2 +can,
                                            3 +sample, see F3, F7, F9 ...)
        self.cmu     <- /CAP2/ CMU1   container total mu
        self.hmu     <- /CAP3/ HMU1   furnace total mu
        self.smu     <- /CAP4/ SMU1   sample total mu
        self.r1..r4  <- /CAP5/../CAP8/ radii: sample, can outer, furnace in/out
        self.theta   <- /CAP9/ THET1  scattering angle used by F's and G's
        self.d       <- /CAP12/ D     beam half-width

    Every integrand F_k(R, OMEGA) is written vectorised: R and OMEGA may be
    NumPy arrays of the same shape, which lets the 9x9 Gauss grid be evaluated
    in one call instead of 81 scalar calls (much faster, same numbers).
    """

    def __init__(self, r1, r2, r3, r4, smu, cmu, hmu, theta, d, col):
        self.r1, self.r2, self.r3, self.r4 = r1, r2, r3, r4
        self.smu, self.cmu, self.hmu = smu, cmu, hmu
        self.theta = theta
        self.d = d
        self.col = col
        self.i = 10           # value left in /CAP1/ by "DO 10 I=6,9"

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _vis(self, R, W):
        return _vis(R, W, self.theta, self.col, self.d)

    def _ex(self, R, W, expo):
        """R * exp(-expo) * vis   -- the common form of all F's."""
        return R * np.exp(-expo) * self._vis(R, W)

    # Path in furnace used by several integrands
    def _hl_std(self, R, W):
        t = self.theta
        return (_aa(self.r4, R, W) - _aa(self.r3, R, W)
                + _bb(self.r4, R, W, t) - _bb(self.r3, R, W, t))

    def _hl_2(self, R, W):        # used in F9, F10, F11, F15, F16, F17
        t = self.theta
        return (_aa(self.r4, R, W) - 2 * _aa(self.r3, R, W)
                + _bb(self.r4, R, W, t) - 2 * _bb(self.r3, R, W, t)
                - _cc(R, W, t))

    def _hl_5(self, R, W):        # used in F5, F6, F7
        t = self.theta
        return (_bb(self.r4, R, W, t) - 2 * _bb(self.r3, R, W, t)
                + _aa(self.r4, R, W) - _cc(R, W, t))

    def _hl_12(self, R, W):       # used in F12, F13, F14
        t = self.theta
        return (_aa(self.r4, R, W) - 2 * _aa(self.r3, R, W)
                + _bb(self.r4, R, W, t) - _cc(R, W, t))

    # ------------------------------------------------------------------
    # Integrands  F1 .. F17   (Fortran REAL*4 FUNCTION ATCOR$Fk(R,OMEGA))
    # ------------------------------------------------------------------
    def F1(self, R, W):
        """Scattering in the sample (gives A_s,sch)."""
        t = self.theta
        sl1 = _aa(self.r1, R, W) + _bb(self.r1, R, W, t)
        cl = _aa(self.r2, R, W) + _bb(self.r2, R, W, t) - sl1
        hl = self._hl_std(R, W)
        sl = sl1 - _cc(R, W, t)
        return self._ex(R, W, self.smu * sl + self.cmu * cl + self.hmu * hl)

    def F2(self, R, W):
        t = self.theta
        cl = _aa(self.r2, R, W) + _bb(self.r2, R, W, t) - _cc(R, W, t)
        return self._ex(R, W, self.cmu * cl + self.hmu * self._hl_std(R, W))

    def F3(self, R, W):
        t = self.theta
        sl = 2 * _bb(self.r1, R, W, t)
        cl = _bb(self.r2, R, W, t) + _aa(self.r2, R, W) - _cc(R, W, t) - sl
        hl = self._hl_std(R, W)
        if self.i < 2:
            return self._ex(R, W, self.cmu * cl + self.hmu * hl)
        return self._ex(R, W, self.smu * sl + self.cmu * cl + self.hmu * hl)

    def F4(self, R, W):
        t = self.theta
        sl = 2 * _aa(self.r1, R, W) + 2 * _bb(self.r1, R, W, t)
        cl = _aa(self.r2, R, W) + _bb(self.r2, R, W, t) - _cc(R, W, t) - sl
        hl = self._hl_std(R, W)
        if self.i < 2:
            return self._ex(R, W, self.cmu * cl + self.hmu * hl)
        return self._ex(R, W, self.smu * sl + self.cmu * cl + self.hmu * hl)

    def F5(self, R, W):
        return self._ex(R, W, self.hmu * self._hl_5(R, W))

    def F6(self, R, W):
        cl = 2 * _bb(self.r2, R, W, self.theta)
        hl = self._hl_5(R, W)
        if self.i < 2:
            return self._ex(R, W, self.hmu * hl)
        return self._ex(R, W, self.hmu * hl + self.cmu * cl)

    def F7(self, R, W):
        t = self.theta
        sl = 2 * _bb(self.r1, R, W, t)
        cl = 2 * _bb(self.r2, R, W, t) - sl
        hl = self._hl_5(R, W)
        return self._ex(R, W, self._by_i(sl, cl, hl))

    def F8(self, R, W):
        t = self.theta
        hl = _aa(self.r4, R, W) + _bb(self.r4, R, W, t) - _cc(R, W, t)
        return self._ex(R, W, self.hmu * hl)

    def F9(self, R, W):
        t = self.theta
        sl = 2 * _bb(self.r1, R, W, t) + 2 * _aa(self.r1, R, W)
        hl = self._hl_2(R, W)
        cl = 2 * _aa(self.r2, R, W) + 2 * _bb(self.r2, R, W, t) - sl
        return self._ex(R, W, self._by_i(sl, cl, hl))

    def F10(self, R, W):
        cl = 2 * _aa(self.r2, R, W) - 2 * _bb(self.r2, R, W, self.theta)
        hl = self._hl_2(R, W)
        if self.i < 2:
            return self._ex(R, W, self.hmu * hl)
        return self._ex(R, W, self.cmu * cl + self.hmu * hl)

    def F11(self, R, W):
        return self._ex(R, W, self.hmu * self._hl_2(R, W))

    def F12(self, R, W):
        return self._ex(R, W, self.hmu * self._hl_12(R, W))

    def F13(self, R, W):
        cl = 2 * _aa(self.r2, R, W)
        hl = self._hl_12(R, W)
        if self.i < 2:
            return self._ex(R, W, self.hmu * hl)
        return self._ex(R, W, self.hmu * hl + self.cmu * cl)

    def F14(self, R, W):
        sl = 2 * _aa(self.r1, R, W)
        cl = 2 * _aa(self.r2, R, W) - sl
        hl = self._hl_12(R, W)
        return self._ex(R, W, self._by_i(sl, cl, hl))

    def F15(self, R, W):
        sl = 2 * _aa(self.r1, R, W)
        cl = 2 * _aa(self.r2, R, W) - sl
        hl = self._hl_2(R, W)
        return self._ex(R, W, self._by_i(sl, cl, hl))

    def F16(self, R, W):
        cl = 2 * _aa(self.r2, R, W)
        hl = self._hl_2(R, W)
        if self.i < 2:
            return self._ex(R, W, self.hmu * hl)
        return self._ex(R, W, self.hmu * hl + self.cmu * cl)

    def F17(self, R, W):
        return self._ex(R, W, self.hmu * self._hl_2(R, W))

    def _by_i(self, sl, cl, hl):
        """Fortran arithmetic IF  'IF(I-2)1,2,3' used in F7, F9, F14, F15:
        I=1 -> furnace only, I=2 -> +container, I>=3 -> +sample."""
        if self.i < 2:
            return self.hmu * hl
        if self.i == 2:
            return self.cmu * cl + self.hmu * hl
        return self.smu * sl + self.cmu * cl + self.hmu * hl

    # ------------------------------------------------------------------
    # Integration limits G1 .. G19 (Fortran REAL*4 FUNCTION ATCOR$Gk(R))
    # They give omega_min(r) or omega_max(r).  Vectorised in R.
    # ------------------------------------------------------------------
    def G1(self, R):  return np.full_like(R, 0.5 * (PI + self.theta))
    def G2(self, R):  return np.full_like(R, 1.5 * PI + 0.5 * self.theta)
    def G3(self, R):  return 0.5 * PI + self.theta + _arcc(self.r1 / R)
    def G4(self, R):  return 1.5 * PI + self.theta - _arcc(self.r1 / R)
    def G5(self, R):  return 1.5 * PI + _arcc(self.r1 / R)
    def G6(self, R):  return 0.5 * PI + _arcc(self.d / R)
    def G7(self, R):  return 0.5 * PI + _arcc(self.r2 / R)
    def G8(self, R):  return 0.5 * PI + _arcc(self.r1 / R)
    def G9(self, R):  return np.full_like(R, PI)
    def G10(self, R): return 1.5 * PI - _arcc(self.d / R)
    def G11(self, R): return np.full_like(R, 2.0 * PI)
    def G12(self, R): return 1.5 * PI + _arcc(self.r2 / R)
    def G13(self, R): return 1.5 * PI + _arcc(self.d / R)
    def G14(self, R): return PI + _arcc(self.r3 / R)
    def G15(self, R): return _arcs(self.r2 / R)
    def G16(self, R): return _arcs(self.d / R)
    def G17(self, R): return _arcs(self.r1 / R)
    def G18(self, R): return np.zeros_like(R)
    def G19(self, R): return 2.0 * PI - _arcc(self.r3 / R)

    # ------------------------------------------------------------------
    # Double integrals
    # ------------------------------------------------------------------
    def gxy3(self, F, psi1, psi2, xinf, xsup):
        """Integral_{xinf}^{xsup} dr  Integral_{psi1(r)}^{psi2(r)} dw  F(r, w)

        Fortran: SUBROUTINE ATCOR$GXY3(F,PSI1X,PSI2X,XINF,XSUP,NX,NY,S) with
        NX = NY = 1 (one Gauss panel of 9 points in each direction).
        Vectorised: the 9 r-nodes and the 9x9 omega-nodes are evaluated at
        once.  Note that, as in Fortran, a negative omega range (psi2<psi1)
        contributes negatively; nothing is clipped.
        """
        dx = 0.5 * (xsup - xinf)
        x = xinf + dx * (1.0 + GL_U)                    # 9 r-nodes
        p1 = psi1(x)
        p2 = psi2(x)
        dy = 0.5 * (p2 - p1)                            # 9 half-ranges
        y = p1[:, None] + dy[:, None] * (1.0 + GL_U[None, :])   # 9x9 omegas
        fv = F(np.broadcast_to(x[:, None], y.shape), y)
        inner = fv @ GL_H                               # sum_K H_K F
        return dx * np.sum(GL_H * dy * inner)

    def gxy3_beam(self, F, psi1, psi2, xinf, xsup, theta_arg, shifted):
        """Same double integral but with zero weight outside the beam.

        Fortran: ATCOR$GXY3x (shifted=False, incident-beam limits, BEAMWX)
                 ATCOR$GXY3z (shifted=True, limits reflected, BEAMWZ)
        For each r-node the omega range is intersected with the beam
        interval(s) returned by `_beamw`; there may be two intervals.
        """
        dx = 0.5 * (xsup - xinf)
        x = xinf + dx * (1.0 + GL_U)
        total = 0.0
        for L in range(9):
            xl = float(x[L])
            p1 = float(psi1(np.array(xl)))
            p2 = float(psi2(np.array(xl)))
            bw, bw2, w1, w2, w3, w4 = _beamw(xl, theta_arg, self.d, shifted)
            intervals = []
            if bw:
                intervals.append((max(p1, w1), min(p2, w2)))
                if bw2:
                    intervals.append((max(p1, w3), min(p2, w4)))
            else:
                intervals.append((p1, p2))
            sj_a = 0.0
            for a, b in intervals:
                dy = 0.5 * (b - a)
                if dy <= 0.0:                 # "if (dxli.le.0.) goto 5"
                    continue
                y = a + dy * (1.0 + GL_U)
                fv = F(np.full_like(y, xl), y)
                sj_a += GL_H[L] * dy * np.dot(GL_H, fv)
            total += sj_a
        return dx * total


# ==========================================================================
#  ATCORS -- the Paalman & Pings factors
# ==========================================================================
def atcors(theta, rr1, rr2, rr3, rr4, sw, smu, cmu, hmu,
           collimator: Collimator = NO_COLLIMATOR):
    """Paalman & Pings attenuation factors for a cylindrical geometry.

    Fortran: SUBROUTINE ATCORS(THETA,RR1,RR2,RR3,RR4,SW,SMU,CMU,HMU,
                               ASSCH,ACCH,ACSCH,AHSCH,AHCH,AHH)

    Parameters
    ----------
    theta : scattering angle 2theta [rad]
    rr1   : sample radius (= inner radius of the container) [cm]
    rr2   : container outer radius [cm]   (rr2 <= rr1  -> no container)
    rr3   : furnace inner radius [cm]
    rr4   : furnace outer radius [cm]     (rr4 <= rr3  -> no furnace)
    sw    : full beam width [cm]
    smu, cmu, hmu : TOTAL linear attenuation coefficients mu = n*(sigma_s +
                    sigma_a(lambda)) of sample, container, furnace [cm^-1]
    collimator    : optional radial collimator (Fortran /coll*/ commons)

    Returns
    -------
    (assch, acch, acsch, ahsch, ahch, ahh) : floats
        Same order as the Fortran output arguments.
    """
    r1, r2, r3, r4 = float(rr1), float(rr2), float(rr3), float(rr4)
    d = 0.5 * sw
    can = r2 > r1
    if not can:                          # no container: make it infinitely thin
        cmu = 0.0
        r2 = r1 * 1.000001
    fur = r4 > r3
    if not fur:                          # no furnace: put a dummy one outside beam
        hmu = 0.0
        r3 = math.sqrt(r2 * r2 + d * d) * 1.00001
        r4 = r3 * 1.000001

    # Consistency checks (only when the beam is wider than sample and can)
    if not (r1 >= d or r2 >= d):
        if r3 <= d:
            raise ValueError("ATCORS: the beam is wider than the furnace i.d.")
        if (r4 * r4 - r3 * r3) > r1 * r1:
            raise ValueError("ATCORS: r4**2-r3**2 > r1**2")
        if (r3 * r3 - d * d) < r2 * r2:
            raise ValueError("ATCORS: r3**2-r2**2 < (sw/2)**2")

    it = _AtcorsIntegrator(r1, r2, r3, r4, smu, cmu, hmu, abs(theta), d, collimator)
    g = it.gxy3

    # ---------------- A_s,sch : scattering in the sample -----------------
    if d >= r1:                                          # beam wider than sample
        assch = 2.0 * g(it.F1, it.G1, it.G2, 0.0, r1) / (PI * r1 * r1)
    else:                                                # beam narrower
        ax = it.gxy3_beam(it.F1, it.G1, it.G2, 0.0, r1, theta, False)
        az = it.gxy3_beam(it.F1, it.G1, it.G2, 0.0, r1, theta, True)
        assch = (ax + az) / area(r1, d)

    # ---------------- A_c,ch and A_c,sch : scattering in container --------
    if can:
        aaa = [0.0, 0.0]
        cth = math.cos(theta / 2.0)
        if d >= r2:                                      # beam wider than can
            a1 = g(it.F2, it.G1, it.G3, r1, r2)
            if cth > r1 / r2:
                rinf = r1 / cth
                a0 = g(it.F2, it.G4, it.G2, rinf, r2)
                for I in (1, 2):
                    it.i = I
                    if theta <= 0:
                        b2 = c = 0.0
                    else:
                        b2 = g(it.F3, it.G3, it.G5, r1, rinf)
                        c = g(it.F4, it.G5, it.G2, r1, rinf)
                    b3 = g(it.F3, it.G3, it.G4, rinf, r2)
                    aaa[I - 1] = a1 + a0 + b2 + b3 + c
            else:
                for I in (1, 2):
                    it.i = I
                    b23 = 0.0 if theta >= PI else g(it.F3, it.G3, it.G5, r1, r2)
                    c = g(it.F4, it.G5, it.G2, r1, r2)
                    aaa[I - 1] = a1 + b23 + c
            norm = 2.0 / (PI * (r2 * r2 - r1 * r1))
        else:                                            # can wider than beam
            def gb(F, p1, p2, xa, xb):
                return (it.gxy3_beam(F, p1, p2, xa, xb, theta, False)
                        + it.gxy3_beam(F, p1, p2, xa, xb, theta, True))
            a1 = gb(it.F2, it.G1, it.G3, r1, r2)
            if cth > r1 / r2:
                rinf = r1 / cth
                a0 = gb(it.F2, it.G4, it.G2, rinf, r2)
                for I in (1, 2):
                    it.i = I
                    if theta <= 0:
                        b2 = c = 0.0
                    else:
                        b2 = gb(it.F3, it.G3, it.G5, r1, rinf)
                        c = gb(it.F4, it.G5, it.G2, r1, rinf)
                    b3 = gb(it.F3, it.G3, it.G4, rinf, r2)
                    aaa[I - 1] = a1 + a0 + b2 + b3 + c
            else:
                for I in (1, 2):
                    it.i = I
                    b23 = 0.0 if theta >= PI else gb(it.F3, it.G3, it.G5, r1, r2)
                    c = gb(it.F4, it.G5, it.G2, r1, r2)
                    aaa[I - 1] = a1 + b23 + c
            norm = 1.0 / (area(r2, d) - area(r1, d))
        acch = norm * aaa[0]
        acsch = norm * aaa[1]
    else:
        acch = acsch = 1.0

    # ---------------- Furnace factors (computed at 0, 180, 90 deg) --------
    if fur:
        sh = d * math.sqrt(r4 * r4 - d * d) - d * math.sqrt(r3 * r3 - d * d)
        sh = 2.0 * (sh + r4 * r4 * float(_arcs(d / r4)) - r3 * r3 * float(_arcs(d / r3)))

        # --- 2theta = 0
        it.theta = 0.0
        bbb = [0.0, 0.0, 0.0]
        A1 = g(it.F5, it.G6, it.G7, r3, r4)
        A2 = 0.0
        for I in (1, 2, 3):
            it.i = I
            if I < 3:
                A2 = g(it.F6, it.G7, it.G8, r3, r4)
            A3 = g(it.F7, it.G8, it.G9, r3, r4)
            bbb[I - 1] = (A1 + A2 + A3) * (4.0 / sh)
        h00, ch00, sch00 = bbb

        # --- 2theta = 180
        it.theta = PI
        A1HPI = g(it.F8, it.G9, it.G10, r3, r4)
        A3 = g(it.F11, it.G13, it.G12, r3, r4)
        for I in (1, 2, 3):
            it.i = I
            if I < 3:
                A2 = g(it.F10, it.G12, it.G5, r3, r4)
            A1 = g(it.F9, it.G5, it.G11, r3, r4)
            bbb[I - 1] = (A1 + A2 + A3 + A1HPI) * (2.0 / sh)
        hpi, chpi, schpi = bbb

        # --- 2theta = 90
        it.theta = PI / 2.0
        A1 = g(it.F8, it.G6, it.G14, r3, r4)
        A2 = g(it.F5, it.G14, it.G10, r3, r4)
        A1H90 = A1 + A2
        A1 = g(it.F12, it.G15, it.G16, r3, r4)
        A7 = g(it.F17, it.G13, it.G12, r3, r4)
        A6 = 0.0
        for I in (1, 2, 3):
            it.i = I
            if I < 3:
                A2 = g(it.F13, it.G17, it.G15, r3, r4)
                A6 = g(it.F16, it.G12, it.G5, r3, r4)
            A3 = g(it.F14, it.G18, it.G17, r3, r4)
            A4 = g(it.F14, it.G19, it.G11, r3, r4)
            A5 = g(it.F15, it.G5, it.G19, r3, r4)
            bbb[I - 1] = (A1H90 + A1 + A7 + A2 + A6 + A3 + A4 + A5) / sh
        h90, ch90, sch90 = bbb

        # --- parabola through the three angles, evaluated at theta
        def parab(f0, f90, f180):
            a1 = (f90 - f0 - 0.5 * (f180 - f0)) / (-(PI * PI) / 4.0)
            a2 = (f180 - f0 - 4.0 * (f90 - f0)) / (-PI)
            return a1 * theta * theta + a2 * theta + f0
        ahh = parab(h00, h90, hpi)
        ahch = parab(ch00, ch90, chpi)
        ahsch = parab(sch00, sch90, schpi)
    else:
        ahh = ahch = ahsch = 1.0

    return float(assch), float(acch), float(acsch), float(ahsch), float(ahch), float(ahh)


# ==========================================================================
#  Blech & Averbach multiple scattering
# ==========================================================================
_BA_TABLE = None      # cache  (Fortran: SAVE reentry, rmu, ratio, d)


def _find_data_file(name: str) -> str:
    """Look for a data table first in the current directory (as the Fortran
    does), then next to this module, then in $PYCORRECT_DATA."""
    for folder in (os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
                   os.environ.get("PYCORRECT_DATA", "")):
        p = os.path.join(folder, name)
        if folder and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"cannot find {name}")


def load_double_scattering_table(path: str | None = None):
    """Read doublescattering.dat (Blech & Averbach delta_2 table).

    Fortran: first part of SUBROUTINE MSCAT:
        read (3,100) (rmu(i),i=2,38)          ! 37 values of mu*R
        read (3,110) (ratio(i),i=1,20)        ! 20 values of R/H
        read (3,*) ((d(i,j),j=1,20),i=2,38)   ! 37 x 20 table
    rmu(1)=0 and d(1,:)=0 are added (no multiple scattering at mu*R=0).

    Returns
    -------
    rmu   : array (38,)   mu*R grid, rmu[0] = 0
    ratio : array (20,)   R/H grid
    d     : array (38,20) delta_2(mu*R, R/H), row 0 = 0
    """
    global _BA_TABLE
    if path is None and _BA_TABLE is not None:
        return _BA_TABLE
    fname = path or _find_data_file("doublescattering.dat")
    rmu, ratio, vals = [], [], []
    in_table = False
    with open(fname) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("Mu*R:"):
                rmu += [float(v) for v in s[5:].split()]
            elif s.startswith("R/H"):
                ratio += [float(v) for v in s.split(":", 1)[1].split()]
            elif s.lower().startswith("double"):
                in_table = True
            elif in_table and s:
                vals += [float(v) for v in s.split()]
    rmu = np.concatenate(([0.0], rmu))
    ratio = np.array(ratio)
    d = np.zeros((len(rmu), len(ratio)))
    d[1:, :] = np.array(vals).reshape(len(rmu) - 1, len(ratio))
    table = (rmu, ratio, d)
    if path is None:
        _BA_TABLE = table
    return table


def mscat(r, h, tmu, smu, *, faithful=False, verbose=True, return_details=False):
    """Blech & Averbach multiple-scattering correction for a cylinder.

    Fortran: SUBROUTINE MSCAT(R,H,TMU,SMU,DELTA)

    Parameters
    ----------
    r   : radius of the cylinder [cm]
    h   : height of the illuminated part (beam height) [cm]
    tmu : total linear attenuation coefficient [cm^-1]
    smu : linear SCATTERING coefficient [cm^-1]
    faithful : False (default) -> corrected physics: proper bilinear
                        interpolation in (mu*R, R/H).
               True  -> reproduce CORRECT 2.24 exactly.  The Fortran line
                        `del=d1+(d1-d1)*(...)` means the table is interpolated
                        in mu*R ONLY; the R/H value used is the grid value just
                        below the true R/H.

    Returns
    -------
    delta : ratio (multiple scattering) / (single scattering), all orders.
            If return_details=True, returns (delta, delta2_table_value).
    """
    rmu, ratio, d = load_double_scattering_table()
    ratioex = r / h
    rmuex = r * tmu
    if ratio[0] <= ratioex <= ratio[-1] and rmuex <= rmu[-1]:
        # --- locate the cell (Fortran index search translated to 0-based)
        ir = 1
        while ratioex > ratio[ir]:
            ir += 1
        ir -= 1
        im = 1
        while rmuex > rmu[im]:
            im += 1
        im -= 1
        f_mu = (rmuex - rmu[im]) / (rmu[im + 1] - rmu[im])
        d1 = d[im, ir] + (d[im + 1, ir] - d[im, ir]) * f_mu
        d2 = d[im, ir + 1] + (d[im + 1, ir + 1] - d[im, ir + 1]) * f_mu
        if faithful:
            dl = d1                        # Fortran bug: (d1-d1) = 0
        else:
            dl = d1 + (d2 - d1) * (ratioex - ratio[ir]) / (ratio[ir + 1] - ratio[ir])
    else:
        if verbose:
            print("\n COMMENT - outside range tabulated: "
                  "proceeding with calculation of multiple scattering\n")
        dl = mulscs(rmuex, rmuex / ratioex)
    delta2 = dl
    dl = smu / tmu * dl
    delta = (dl + 0.693956559 * dl**2 + 0.370110165 * dl**3
             + 0.160525235 * dl**4 + 0.058706371 * dl**5)
    if return_details:
        return float(delta), float(delta2)
    return float(delta)


def h2el(k1, phi, rmu, hmu):
    """Blech-Averbach double-scattering kernel for a first-flight direction k1.

    Fortran: REAL FUNCTION H2EL(K1,PHI,RMU,HMU) -- four nested 9-point Gauss
    loops (z, theta, xi, eta).  Here the four loops are done at once with NumPy
    broadcasting on a 9x9x9x9 grid (axes: i1=z, i2=theta, i3=xi, i4=eta).
    All lengths are in units of 1/mu (rmu = mu*R, hmu = mu*H).
    """
    lam, w = GL_U, GL_H
    cphi, sphi = math.cos(phi), math.sin(phi)
    kx, ky, kz = k1
    z = (0.5 * hmu * (1 + lam))[:, None, None]                    # (9,1,1)
    th = (0.5 * (2 * phi + PI * lam))[None, :, None]              # (1,9,1)
    r1 = rmu * np.cos(th)
    r2 = rmu * np.sin(th)
    xilim = r1 * cphi + r2 * sphi + np.sqrt(np.maximum(rmu * rmu - (r1 * sphi - r2 * cphi) ** 2, 0))
    xi = 0.5 * xilim * (1 + lam[None, None, :])                   # (1,9,9)
    p1 = r1 - xi * cphi
    p2 = r2 - xi * sphi
    k12 = kx * kx + ky * ky
    if k12 != 0:
        etalim = (p1 * kx + p2 * ky + np.sqrt(np.maximum(
            rmu * rmu * k12 - (ky * p1 - kx * p2) ** 2, 0))) / k12   # (1,9,9)
        etalim = np.broadcast_to(etalim, (9, 9, 9)).copy()
        test = z - etalim * kz
        bad = (test < 0) | (test > hmu)
        if kz > 0:
            alt = np.broadcast_to(z / kz, (9, 9, 9))
        elif kz < 0:
            alt = np.broadcast_to((z - hmu) / kz, (9, 9, 9))
        else:
            alt = etalim
        etalim = np.where(bad, alt, etalim)
    elif kz > 0:
        etalim = np.broadcast_to(z, (9, 9, 9))
    else:
        etalim = np.broadcast_to(hmu - z, (9, 9, 9))
    eta = 0.5 * etalim[..., None] * (1 + lam)                      # (9,9,9,9)
    dd = p1[..., None] - eta * kx + np.sqrt(np.maximum(rmu * rmu - (p2[..., None] - eta * ky) ** 2, 0))
    W = (w[:, None, None, None] * w[None, :, None, None]
         * w[None, None, :, None] * w[None, None, None, :])
    integrand = (W * xilim[..., None] * etalim[..., None]
                 * np.cos(phi - th)[..., None] / np.exp(xi[..., None] + eta + dd))
    return float(integrand.sum() / (16.0 * rmu))


def mulscs(rmu, hmu):
    """Direct calculation of the Blech-Averbach ratio delta_2 = double/single.

    Fortran: SUBROUTINE MULSCS(RMU,HMU,DOUBLE).  Used by MSCAT when (mu*R,R/H)
    is outside the tabulated range.  The detector is at 2theta = 60 deg (phi),
    as in the original.
    """
    lam, w = GL_U, GL_H
    phi = 1.047197551
    cphi, sphi = 0.5, 0.866025403
    double = 0.0
    for i2 in range(9):
        kz = lam[i2]
        st1 = math.sin(math.acos(kz))
        for i1 in range(9):
            phi1 = PI * (1 + lam[i1])
            k1 = (st1 * math.cos(phi1), st1 * math.sin(phi1), kz)
            double += h2el(k1, phi, rmu, hmu) * w[i1] * w[i2]
    double /= 4.0
    # single scattering in the same units
    s = 0.0
    for i in range(9):
        th = phi + 0.5 * PI * lam[i]
        xilim = rmu * (math.cos(th) * cphi + math.sin(th) * sphi
                       + math.sqrt(1 - (sphi * math.cos(th) - cphi * math.sin(th)) ** 2))
        for j in range(9):
            xi = 0.5 * xilim * (1 + lam[j])
            dd = rmu * math.cos(th) - xi * cphi + math.sqrt(max(rmu * rmu - (rmu * math.sin(th) - xi * sphi) ** 2, 0))
            s += w[i] * w[j] * xilim * (cphi * math.cos(th) + sphi * math.sin(th)) / math.exp(dd + xi)
    single = s / (4.0 * rmu)
    return double / single
