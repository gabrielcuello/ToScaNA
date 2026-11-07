"""
routines.py  --  Python translation of  routines.f  (CORRECT 2.24)
====================================================================

Contents
--------
plate(...)    <- SUBROUTINE PLATE   flat-plate attenuation factors
polyfit(...)  <- SUBROUTINE POLYFIT (+ POLYFN, and the Harwell routines
                 VB01S / MA10A / FM02AS / SA01A it relies on)
polyval_c(...)   helper that evaluates c[0] + c[1] x + ... (the loops that
                 follow every CALL POLYFIT in the Fortran)

Why POLYFIT does not need the Harwell library here
--------------------------------------------------
In Fortran, POLYFIT calls VB01S, a general Gauss-Newton / Marquardt
non-linear least-squares minimiser (Harwell library), with the model
POLYFN = c1 + c2 x + ... + c_n x^(n-1).  VB01S minimises

        SS = sum_i  W_i * (f(x_i) - y_i)^2

Because the model is LINEAR in the coefficients, Gauss-Newton converges in
one step to the ordinary weighted linear least-squares solution.  We
therefore solve it directly with  numpy.linalg.lstsq  (double precision, so
it is slightly *more* accurate than the REAL*4 Fortran).

IMPORTANT: the third argument W multiplies the SQUARED residual directly,
it is NOT 1/sigma^2.  CORRECT passes the error bars themselves as W in
SMOOTH_VANADIUM (so points with larger errors get *larger* weight!).  This
is what the program did; we reproduce it.

NumPy functions used
--------------------
numpy.vander(x, n, increasing=True) : matrix [1, x, x^2, ..., x^(n-1)]
numpy.linalg.lstsq(A, b, rcond=None): least-squares solution of A c = b
numpy.sqrt, numpy.asarray           : element-wise sqrt; convert to array
"""

from __future__ import annotations

import math

import numpy as np


def polyfit(x, y, w, nc):
    """Weighted least-squares polynomial with `nc` coefficients.

    Fortran: SUBROUTINE POLYFIT(X,Y,W,NPTS,COEFF,NC,SUMSQ)

    Parameters
    ----------
    x, y : data
    w    : weights multiplying the squared residuals (see module docstring)
    nc   : number of coefficients (polynomial degree = nc-1)

    Returns
    -------
    coeff : array (nc,)   f(x) = coeff[0] + coeff[1] x + ... + coeff[nc-1] x^(nc-1)
    sumsq : float         UNweighted sum of squared residuals (as Fortran)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    A = np.vander(x, nc, increasing=True)          # columns 1, x, x^2, ...
    sw = np.sqrt(np.abs(w))
    coeff, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    fit = A @ coeff
    sumsq = float(np.sum((fit - y) ** 2))
    return coeff, sumsq


def polyval_c(coeff, x):
    """Evaluate coeff[0] + coeff[1]*x + ... (Fortran loops after POLYFIT:
    'a=coeff(1); do k=2,ncoeff: a=a+coeff(k)*x**(k-1)').

    numpy.polynomial.polynomial.polyval uses the same increasing-power order.
    """
    return np.polynomial.polynomial.polyval(np.asarray(x, float), coeff)


def plate(rad, phiang, sa_t, ca_t, sa_mu, ca_mu, faithful=False):
    """Paalman & Pings factors for a FLAT PLATE sample in a flat container.

    Fortran: SUBROUTINE PLATE(RAD,PHIANG,SA_T,CA_T,SA_MU,CA_MU,ASSC,ACC,ACSC)

    Parameters
    ----------
    rad    : scattering angle 2theta [rad]
    phiang : angle between the plate normal and the incident beam [DEGREES]
    sa_t   : sample thickness [cm]
    ca_t   : TOTAL thickness of the container (both walls + sample) [cm];
             each wall is (ca_t - sa_t)/2.  ca_t = 0 or = sa_t -> no container
    sa_mu, ca_mu : total linear attenuation coefficients [cm^-1]
    faithful : False (default) -> corrected formulas; True -> CORRECT 2.24.
             Two errors of the Fortran are corrected (found with the direct
             method of MCAMS 2.1, see README):
             * reflection geometry, A_c,sc: for neutrons scattered in the BACK
               wall the Fortran includes the sample on the way in only,
               acsc2 = e^{ca d (b-a)} e^{-sa t a} acsc1; the outgoing path
               crosses the sample too: acsc2 = e^{(ca d + sa t)(b-a)} acsc1
               (b < 0 in reflection). The error is large: 0.82 instead of 0.61
               for 2 mm of Ni between 0.2 mm V windows at 110 deg.
             * transmission with a == b (2theta = 0 at normal incidence),
               A_c,c: the two walls are added but not averaged (factor 2).

    Returns
    -------
    (assc, acc, acsc) : A_s,sc, A_c,c, A_c,sc
        All three are returned as 1 when the scattered beam is parallel to the
        plate (|phi +- 90deg - 2theta| < 0.01 rad), as in Fortran.

    Geometry: a = 1/cos(phi) (incident path factor), b = 1/cos(2theta-phi)
    (exit path factor).  "reflect" = scattered beam leaves through the same
    face it entered (reflection geometry); otherwise transmission geometry.
    """
    pi = 3.14159                       # same truncated pi as the Fortran
    assc = acc = acsc = 1.0
    sa_d = sa_t
    ca_d = (ca_t - sa_t) / 2.0
    phi = phiang * pi / 180.0
    if abs(phi + pi / 2 - rad) < 0.01 or abs(phi - pi / 2 - rad) < 0.01:
        return assc, acc, acsc
    a = 1.0 / math.cos(phi)
    b = 1.0 / math.cos(rad - phi)
    reflect = not ((phi + pi / 2) > rad and (phi - pi / 2) < rad)
    e = math.exp

    if sa_t == ca_t or ca_t == 0.0:               # no container
        acc = acsc = 1.0
        if reflect:
            assc = (e(sa_mu * sa_d * (b - a)) - 1) / (sa_mu * sa_d * (b - a))
        elif a == b:
            assc = e(-sa_mu * sa_d * b)
        else:
            assc = (e(-sa_mu * sa_d * b)
                    * (e(sa_mu * sa_d * (b - a)) - 1) / (sa_mu * sa_d * (b - a)))
    else:
        if reflect:
            assc = (e(-ca_mu * ca_d * (a - b))
                    * (e(sa_mu * sa_d * (b - a)) - 1) / (sa_mu * sa_d * (b - a)))
            acsc1 = (e(ca_mu * ca_d * (b - a)) - 1) / (2 * ca_mu * ca_d * (b - a))
            if faithful:     # CORRECT 2.24: sample attenuation on the way in only
                acsc2 = e(ca_mu * ca_d * (b - a)) * e(-sa_mu * sa_d * a) * acsc1
            else:            # back wall: front wall AND sample, in and out
                acsc2 = e((ca_mu * ca_d + sa_mu * sa_d) * (b - a)) * acsc1
            acsc = acsc1 + acsc2
            acc = (e(ca_mu * ca_d * (b - a)) + 1) * acsc1
        elif a == b:
            assc = e(-ca_mu * ca_d * (a + b) - sa_mu * sa_d * b)
            acsc1 = e(-sa_mu * sa_d * b - 2 * ca_mu * ca_d * b)
            acsc2 = e(-sa_mu * sa_d * a - ca_mu * ca_d * (a + b))
            acsc = (acsc1 + acsc2) / 2
            acc = e(-ca_mu * ca_d * (a + b)) + e(-2 * ca_mu * ca_d * b)
            if not faithful:  # average of the two walls (the Fortran adds them)
                acc /= 2
        else:
            f_s = (e(sa_mu * sa_d * (b - a)) - 1) / (sa_mu * sa_d * (b - a))
            f_c = (e(ca_mu * ca_d * (b - a)) - 1) / (ca_mu * ca_d * (b - a))
            assc = e(-ca_mu * ca_d * (a + b) - sa_mu * sa_d * b) * f_s
            acsc1 = e(-sa_mu * sa_d * b - 2 * ca_mu * ca_d * b) * f_c
            acsc2 = e(-sa_mu * sa_d * a - ca_mu * ca_d * (a + b)) * f_c
            acsc = (acsc1 + acsc2) / 2
            acc = (e(-ca_mu * ca_d * (a + b)) + e(-2 * ca_mu * ca_d * b)) * f_c / 2
    return assc, acc, acsc
