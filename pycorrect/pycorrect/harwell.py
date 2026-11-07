"""
harwell.py  --  replacement for the Harwell Subroutine Library routines
=======================================================================
(harwell.f in the "lib" folder of CORRECT)

harwell.f contains old (1970s) numerical routines.  Only some of them are
actually used by CORRECT, and every one of them has a standard, better
tested equivalent in NumPy/SciPy:

  Harwell routine     used by            Python replacement
  ---------------     -------            ------------------
  VB01S  (+ MA10A,    POLYFIT            numpy.linalg.lstsq   (routines.polyfit)
  FM02AS, SA01A,                          -- linear least squares
  OA01A, ERF, ERFC,
  GMMMA, ALGAMA)
  VB06A  + TG01B      SPLINE_SMOOTH(TOF) scipy.interpolate.LSQUnivariateSpline
                                          (lsq_cubic_spline below)
  VC03A               not used           (automatic knot placement; would be
                                          scipy.interpolate.UnivariateSpline)
  FA03A               not used           numpy.random.normal
  *D versions         not used           (double-precision copies)

About VB06A
-----------
VB06A fits a cubic spline with GIVEN knots XN to data (XD, YD) with weights
WD, minimising   sum WD^2 (s(x)-y)^2  +  sum THETA_j^2 (jump of s''' at knot j)^2.
SPLINE_SMOOTH (correct_tofdif_224.f) passes an array TH that is never
initialised.  In practice it is zero, i.e. VB06A then performs a pure weighted
least-squares cubic-spline fit with the given knots.  That is exactly what
`scipy.interpolate.LSQUnivariateSpline` does (B-spline basis, same function
space), so the results agree within rounding.  TG01B just evaluates the spline
(here: calling the spline object).

SciPy functions used
--------------------
scipy.interpolate.LSQUnivariateSpline(x, y, t, w=None, bbox=[a,b], k=3)
    x increasing data abscissae, t interior knots (strictly inside bbox),
    w weights (multiply the residual, like WD), k=3 cubic.
    The returned object is callable: spl(xnew) evaluates the spline.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def lsq_cubic_spline(x, y, knots, w=None):
    """Least-squares cubic spline through (x, y) with knots[0], knots[-1] as
    the end points and knots[1:-1] as interior knots.

    Equivalent of  CALL VB06A(...)  followed by  TG01B(...)  evaluations.
    Data outside [knots[0], knots[-1]] are clamped to the end knots, as VB06A
    does (XD(1)=MIN(MAX(XD(1),XN(1)),XN(NN))).
    Returns the spline object (callable).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x, kind="stable")       # VB06A requires ascending data
    x, y = x[order], y[order]
    w = None if w is None else np.asarray(w, float)[order]
    xn = np.asarray(knots, float)
    x = np.clip(x, xn[0], xn[-1])
    inner = np.unique(xn[1:-1])
    inner = inner[(inner > xn[0]) & (inner < xn[-1])]
    return LSQUnivariateSpline(x, y, inner, w=w, bbox=[xn[0], xn[-1]], k=3)
