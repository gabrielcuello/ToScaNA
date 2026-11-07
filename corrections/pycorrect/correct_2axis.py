"""
correct_2axis.py  --  Python translation of  correct_2axis_224.f
==================================================================

Full chain of corrections for constant-wavelength (two-axis) diffractometers
such as D4 or D20 at the ILL, or SLAD.

Fortran -> Python correspondence
--------------------------------
SUBROUTINE CORRECT_2AXIS        -> correct_2axis(state)
SUBROUTINE ABSOR                -> absor(...)
SUBROUTINE SMOOTH_VANADIUM      -> smooth_vanadium(...)
SUBROUTINE SMOOTH_VANADIUM_SG   -> smooth_vanadium_sg(...)
SUBROUTINE SMOOTH               -> _sg_smooth(...)        (Savitzky-Golay)
SUBROUTINE QUADFIT2             -> _quadfit2(...)
SUBROUTINE PLACZEK              -> placzek(...)

What the chain does, for every spectrum (block)
-----------------------------------------------
 1. read vanadium, furnace, container and sample (sample LAST: its 2theta
    grid is the one used, as in version 2.24)
 2. correct the sample for a partially filled container (fullness)
 3. subtract the backgrounds (empty instrument) of each measurement
 4. multiply container and vanadium by user factors
 5. optional low-angle (direct beam) correction, from cadmium data or by
    extrapolating a straight line fitted between lowang(3) and lowang(4)
 6. Paalman & Pings attenuation factors (ATCORS via ABSOR) and the
    standard subtraction:
        I_h  = I_h,raw / A_h,h
        I_c  = (I_ch - A_h,ch I_h) / A_c,ch
        I_s  = (I_sch - A_c,sch I_c - A_h,sch I_h) / A_s,sch
 7. vanadium: attenuation, Blech & Averbach multiple scattering and Placzek
       I_v  = I_v,abs / (1 - P_v + delta_v)
 8. smoothing of vanadium (polynomial or Savitzky-Golay) and normalisation
       S(Q) = I_s / I_v  *  (n_v R_v^2 sigma_v/4pi) / (n_s R_s^2)
    (if there is no vanadium, the container is used as calibrant)
 9. sample multiple scattering (Blech & Averbach) and Placzek:
       dsigma/dOmega = S(Q) - delta_s * sigma_s/4pi + P_s
10. write corrected data versus Q (or 2theta) to the output file

The function returns a list of dictionaries, one per spectrum, with all the
intermediate arrays (attenuation factors, delta, Placzek, normalised data,
...).  They are what you want to compare with Monte Carlo simulations.

NumPy functions used
--------------------
numpy.searchsorted(a, v) : index of the first a[i] >= v (replaces the Fortran
                           'do while (theta(n).lt.x) n=n+1' searches)
numpy.sin/cos/tan/arcsin, numpy.sqrt, numpy.abs, numpy.zeros, numpy.ones,
numpy.full, numpy.isfinite, numpy.radians/degrees (deg <-> rad)
"""

from __future__ import annotations

import math

import numpy as np

from .fileio import read_file, DataWriter
from .ndplib import atcors, mscat
from .routines import plate, polyfit, polyval_c
from .state import CorrectState, BKG_OF

PI = math.pi

# measured bodies, in Fortran index order isam = 1..4
BODIES = ["furnace", "container", "sample", "vanadium"]


# ==========================================================================
#  Attenuation factors on the measured 2theta grid
# ==========================================================================
def absor(geometry, plate_angle, r1, r2, r3, r4, sw, smu, cmu, hmu,
          angle_deg, collimator=None, faithful=False):
    """Paalman & Pings factors at every 2theta of `angle_deg`.

    Fortran: SUBROUTINE ABSOR.  Exactly as the Fortran, the factors are
    computed exactly at only 10 equally spaced angles between the first and
    last point, and a 4th-degree polynomial (5 coefficients) in
    (2theta - 2theta_first) [rad] is fitted to each factor and evaluated on
    the full grid.  This keeps the cost low and smooths quadrature noise.

    Parameters: see ndplib.atcors (cylinder) or routines.plate (plate).
        For PLATE geometry, r1 = sample thickness, r2 = total container
        thickness, plate_angle = angle of the plate normal [deg]; the furnace
        factors are set to 1.

    Returns: dict with arrays 'assch','acch','acsch','ahsch','ahch','ahh'.
    """
    from .ndplib import NO_COLLIMATOR
    col = collimator or NO_COLLIMATOR
    ang = np.asarray(angle_deg, float)
    nang = 10
    angrad = np.radians(ang[0] + np.arange(nang) * (ang[-1] - ang[0]) / (nang - 1))
    a = np.ones((6, nang))
    for i, t in enumerate(angrad):
        if geometry == "CYLINDRICAL":
            a[:, i] = atcors(t, r1, r2, r3, r4, sw, smu, cmu, hmu, col)
        elif geometry == "PLATE":
            a[:3, i] = plate(t, plate_angle, r1, r2, smu, cmu, faithful)
        else:
            raise ValueError(f"unknown geometry {geometry}")
    x = angrad - angrad[0]
    xx = np.radians(ang - ang[0])
    names = ["assch", "acch", "acsch", "ahsch", "ahch", "ahh"]
    out = {}
    for k, name in enumerate(names):
        c, _ = polyfit(x, a[k], np.ones(nang), 5)
        out[name] = polyval_c(c, xx)
    return out


# ==========================================================================
#  Vanadium smoothing
# ==========================================================================
def smooth_vanadium(theta, v, e, ideg, faithful=False):
    """Iterative polynomial fit of the vanadium with outlier rejection.

    Fortran: SUBROUTINE SMOOTH_VANADIUM(THETA,V,E,N,MAXPTS,ISAM,IDEG)

    ideg = number of polynomial coefficients (order ideg-1); 0 = no smoothing.
    Loop idev = 10, 9, ..., 3: fit, then flag as outliers (permanently) the
    points further than idev*e from the fit (Bragg peaks of V, spikes...).
    The final fit replaces the data.

    Weights: the Fortran passes the ERRORS as POLYFIT weights, i.e. points with
    larger errors get larger weight.  faithful=False (default) uses the
    statistically correct weights 1/e^2.
    """
    if ideg == 0:
        return v.copy()
    t = theta - theta[0]
    w_all = e if faithful else 1.0 / np.maximum(e, 1e-30) ** 2
    omit = np.zeros(len(v), bool)
    c = None
    for idev in range(10, 2, -1):
        keep = ~omit
        c, _ = polyfit(t[keep], v[keep], w_all[keep], ideg)
        vs = polyval_c(c, t)
        omit |= np.abs(v - vs) > idev * e
    return polyval_c(c, t)


def _sg_smooth(y1, npts, is1, is2, nsm):
    """Savitzky-Golay quadratic smoothing with special end treatment.

    Fortran: SUBROUTINE SMOOTH(MAXPTS,YOUT,NPTS,IS1,IS2,NSM,SFIRST)
    References: Savitzky & Golay, Anal. Chem. 36 (1964) 1627;
                Proctor & Sherwood, Anal. Chem. 52 (1980) 2315.
    Window = 2*nsm+1 points.  The first is1 and last npts-is2 points are
    computed from the polynomial fitted at points is1+nsm+1 and is2-nsm-1
    (used to skip vanadium Bragg peaks sitting on the detector edges).

    y1 : array with 1-BASED indexing (y1[0] unused), modified in place.
    """
    norm1 = nsm * (4 * nsm * nsm - 1) * (2 * nsm + 3) * (nsm + 1) / 3.0
    its = np.arange(-nsm, nsm + 1)
    pm0 = nsm * (nsm + 1) * (3 * nsm * (nsm + 1) - 1 - 5 * its * its)
    iu0l = is1 + nsm + 1
    iu0r = is2 - nsm - 1
    yin = y1.copy()
    for ii in range(1, iu0l):                         # left end
        iu = ii - iu0l
        pmut = (5 * (3 * its * its - nsm * (nsm + 1)) * iu * iu
                + (2 * nsm - 1) * (2 * nsm + 3) * its * iu + pm0)
        y1[ii] = np.dot(pmut, yin[iu0l + its]) / norm1
    for ii in range(iu0r + 1, npts + 1):              # right end
        iu = ii - iu0r
        pmut = (5 * (3 * its * its - nsm * (nsm + 1)) * iu * iu
                + (2 * nsm - 1) * (2 * nsm + 3) * its * iu + pm0)
        y1[ii] = np.dot(pmut, yin[iu0r + its]) / norm1
    for ii in range(iu0l, iu0r + 1):                  # central part
        y1[ii] = np.dot(pm0, yin[ii + its]) / norm1


def _quadfit2(y1, npts, iu0, n1, nsm):
    """Replace a vanadium Bragg peak by a parabola fitted on both sides.

    Fortran: SUBROUTINE QUADFIT2(MAXPTS,YOUT,NPTS,IU0,N1,NSM)
    Fit range: [max(1,iu0-(n1+nsm-1)), iu0-n1] and [iu0+n1, min(npts-1,
    iu0+n1+nsm-1)];  replaced range: [iu0-n1+1, iu0+n1-1].  1-BASED y1.
    The Fortran solves the 3x3 normal equations by hand; the same sums are
    used here (a..h) so that the result is identical.
    """
    im = max(1, iu0 - (n1 + nsm - 1))
    ip = min(npts - 1, iu0 + n1 + nsm - 1)
    idx = np.r_[np.arange(im, iu0 - n1 + 1), np.arange(iu0 + n1, ip + 1)]
    if len(idx) == 0:
        return
    it = (idx - iu0).astype(float)
    yy = y1[idx]
    a, b, c, d, e = len(idx), it.sum(), (it**2).sum(), (it**3).sum(), (it**4).sum()
    f, g, h = yy.sum(), (it * yy).sum(), (it**2 * yy).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        if iu0 - im == ip - iu0:                     # symmetric: b = d = 0
            den = a * e - c * c
            co = [(e * f - c * h) / den, g / c, (a * h - c * f) / den]
        else:
            den = (c * d - b * e) * (a * c - b * b) - (c * c - b * d) * (a * d - b * c)
            co0 = ((c * d - b * e) * (c * f - b * g) - (c * c - d * b) * (d * f - b * h)) / den
            co1 = ((b * c - a * d) * (c * g - b * h) - (c * d - b * e) * (b * f - a * g)) / den
            co2 = (-(b * b - c * a) * (c * g - b * h) + (c * c - b * d) * (b * f - a * g)) / den
            co = [co0, co1, co2]
    if not all(np.isfinite(co)):          # degenerate fit: leave data unchanged
        return
    for ii in range(iu0 - n1 + 1, iu0 + n1):
        t = float(ii - iu0)
        y1[ii] = co[0] + co[1] * t + co[2] * t * t


def smooth_vanadium_sg(theta, v, ideg, lamda, ur, vr, wr, faithful=False):
    """Savitzky-Golay smoothing of vanadium after removing its Bragg peaks.

    Fortran: SUBROUTINE SMOOTH_VANADIUM_SG(THETA,V,E,NPTS,MAXPTS,ISAM,IDEG,
             LAMDA,UR,VR,WR)

    * V is bcc, a = 3.024 A; peaks at h^2+k^2+l^2 = 2,4,...,18.
    * Their width is the Caglioti resolution FWHM^2 = U tan^2 + V tan + W
      (deg^2), parameters from the 'calibration' command.
    * Each peak fully inside the range is replaced by a parabola (QUADFIT2);
      peaks on the edges shift the start/end points of the smoothing.
    * Then Savitzky-Golay smoothing with window 2*|ideg|+1 points.

    faithful=True reproduces the edge search of the Fortran, which compares
    with the LEFT side of a peak on the low-angle edge (so the start point
    never moves).  faithful=False skips the peak properly (uses its right
    side on the low edge and its left side on the high edge).
    """
    npts = len(v)
    th1 = np.r_[np.nan, theta]           # 1-based copies
    y1 = np.r_[np.nan, v].astype(float)
    av = 3.024
    qfac = lamda / 2.0 / av
    nsm = abs(ideg)
    is1, is2 = 0, npts - 1
    for iv in range(1, 10):
        s = qfac * math.sqrt(2.0 * iv)
        if s >= 1.0:
            continue                      # peak beyond 2theta = 180
        th = math.asin(s)
        th2v = 2.0 * math.degrees(th)
        tant = math.tan(th)
        fw2 = ur * tant * tant + vr * tant + wr
        if fw2 <= 0:
            continue
        fwhm = math.sqrt(fw2)
        th2l, th2r = th2v - fwhm, th2v + fwhm
        nquad = int(fwhm / (30.0 / 900.0))   # Fortran integer assignment
        if th2l < th1[1] and th2r > th1[1]:                    # low edge
            is1 = 1
            lim = th2l if faithful else th2r
            while is1 + 1 <= npts and th1[is1 + 1] < lim:
                is1 += 1
        elif th2l > th1[1] and th2r < th1[npts - 1]:          # inside
            i1 = 1
            while th1[i1 + 1] < th2l:
                i1 += 1
            i2 = npts - 1
            while th1[i2 - 1] > th2r:
                i2 -= 1
            iu0 = (i1 + i2) // 2
            n1 = min(iu0 - i1, i2 - iu0)
            _quadfit2(y1, npts, iu0, n1, nquad)
        elif th2l < th1[npts - 1] and th2r > th1[npts - 1]:   # high edge
            lim = th2r if faithful else th2l
            while is2 - 1 >= 1 and th1[is2 - 1] > lim:
                is2 -= 1
    _sg_smooth(y1, npts, is1, is2, nsm)
    return y1[1:]


# ==========================================================================
#  Placzek (inelasticity) correction for a constant-wavelength instrument
# ==========================================================================
_PLACZEK_C = {"D4": (0.878, 0.211, 0.206),
              "D20": (0.793, 0.139, 0.265),
              "SLAD": (0.793, 0.139, 0.265)}      # SLAD: "temporary values"


def placzek(theta_deg, wavelength, atwt, temp, instrument, method, omegav=0.0):
    """Placzek self-scattering correction P(2theta) for one atomic species.

    Fortran: SUBROUTINE PLACZEK(THETA,NPTS,WAVELENGTH,ATWT,TEMP,INSTRUMENT,P)

    method = 'SERIES_EXPANSION' : Yarnell et al., Phys. Rev. A7 (1973) 2130
             'DIATOMIC'         : Powles (effective temperature of a
                                  vibrating molecule, frequency omegav [cm^-1?
                                  the constant 7.6385 converts it to K units])
             'NONE'             : zeros
    c1,c2,c3 are detector-efficiency constants of each instrument.
    Units: P is per unit (sigma/4pi) -- it is multiplied by frac*sigma_s/4pi
    by the caller.  kTe uses the factor 0.041229 = 2 m_n k_B / hbar^2 ...
    in A^-2 K^-1 units, with m_n = 1.008665 amu.
    """
    th = np.asarray(theta_deg, float)
    if method not in ("SERIES_EXPANSION", "DIATOMIC"):
        return np.zeros_like(th)
    c1, c2, c3 = _PLACZEK_C.get(instrument, _PLACZEK_C["D4"])
    nmass = 1.008665
    k0 = 2.0 * PI / wavelength
    kte = 0.041229412 * temp * nmass / k0 / k0
    if method == "DIATOMIC":
        betav = 7.6385392 * omegav / temp
        # betav / tanh(betav/2) -> 2 when betav -> 0 (Fortran gives 0/0 = NaN)
        ratio = betav / math.tanh(0.5 * betav) if betav != 0 else 2.0
        kte = kte * (10.0 + ratio) / 12.0
    a = -kte / 2.0
    b = c1 - 2.0 * c3 * a
    q = 2.0 * k0 * np.sin(np.radians(th) / 2.0)
    return nmass / atwt * (a + b * q * q / (k0 * k0)
                           - nmass / (atwt * k0 * k0) * (0.5 * q**4 / (k0 * k0) + c2 * q * q))


# ==========================================================================
#  Main driver
# ==========================================================================
def _sample_params(st: CorrectState):
    """Macroscopic cross sections.  First lines of CORRECT_2AXIS.
    mu_total(lambda) = n*(sigma_s + sigma_a(2200 m/s) * lambda/1.798)."""
    lam = st.wavelength
    if st.ca_r2 <= 0:
        st.ca_r2 = st.ca_r1
    p = {}
    for pre, cs_s, cs_a, rho in (("ca", st.ca_scat_cs, st.ca_abs_cs, st.ca_rho),
                                 ("fu", st.fu_scat_cs, st.fu_abs_cs, st.fu_rho),
                                 ("va", st.va_scat_cs, st.va_abs_cs, st.va_rho)):
        p[pre + "_smu"] = cs_s * rho
        p[pre + "_amu"] = cs_a * rho
        p[pre + "_tmu"] = p[pre + "_smu"] + p[pre + "_amu"] * lam / 1.798
        p[pre + "_b2"] = cs_s / 4.0 / PI
    ftot = sum(s.fraction for s in st.species) or 1.0
    fr = [s.fraction / ftot for s in st.species]
    sa_scat = sum(f * s.scatt_cs for f, s in zip(fr, st.species))
    sa_abs = sum(f * s.abs_cs for f, s in zip(fr, st.species))
    p["frac"] = fr
    p["sa_scat_cs"], p["sa_abs_cs"] = sa_scat, sa_abs
    p["sa_smu"] = sa_scat * st.sa_rho
    p["sa_amu"] = sa_abs * st.sa_rho
    p["sa_tmu"] = p["sa_smu"] + p["sa_amu"] * lam / 1.798
    p["sa_b2"] = sa_scat / 4.0 / PI
    return p


def correct_2axis(st: CorrectState, verbose: bool = True):
    """Run all corrections for every spectrum.  Fortran: CORRECT_2AXIS.

    Returns a list (one dict per spectrum) with keys:
      'block','theta','q','dcs','err'             final result
      'assch','acch','acsch','ahsch','ahch','ahh' sample P&P factors
      'avvch','acvch','acchv','ahvch','ahchv','ahhv' vanadium factors (if any)
      'delta_sa','delta_va'                       Blech-Averbach delta
      'pc_sa','pc_va'                             Placzek corrections
      'calib_raw','calib_smooth'                  corrected/smoothed V (or can)
      'sample_abs_corrected'                      I_s after P&P, before norm.
    """
    p = _sample_params(st)
    f = st.faithful
    fmt = st.data_format
    out_writer = None
    results = []
    lam = st.wavelength

    for ispec, spec in enumerate(st.spectra):
        blk = spec.block
        counts, errors = {}, {}
        # ---------------- 1. read data (V, furnace, can, sample last) -----
        cad = None
        if st.has("cadmium"):
            cad = read_file(st.files["cadmium"], blk, fmt)
        sample_blk = None
        for body in ("vanadium", "furnace", "container", "sample"):
            if st.has(body):
                b = read_file(st.files[body], blk, fmt)
                counts[body], errors[body] = b.y.astype(float), b.e.astype(float)
                theta = b.x.astype(float)
                if body == "sample":
                    sample_blk = b
        if sample_blk is None:
            raise RuntimeError("no sample file defined")
        npts = len(theta)

        # ---------------- 2. container not completely full ----------------
        if st.has("container"):
            ff = st.sa_fullness
            if not f:
                errors["sample"] = np.sqrt(errors["sample"] ** 2
                                           + ((1 - ff) * errors["container"]) ** 2) / ff
            counts["sample"] = (counts["sample"] - (1 - ff) * counts["container"]) / ff

        # ---------------- 3. background subtraction ----------------------
        bkg_c = bkg_e = None
        for body in BODIES:
            if st.has(body) and st.has(BKG_OF[body]):
                bb = read_file(st.files[BKG_OF[body]], blk, fmt)
                bkg_c, bkg_e = bb.y.astype(float), bb.e.astype(float)
                counts[body] = counts[body] - bkg_c
                errors[body] = np.sqrt(errors[body] ** 2 + bkg_e ** 2)

        q = 4 * PI / lam * np.sin(np.radians(theta) / 2)

        # ---------------- 4. multipliers ----------------------------------
        if st.has("container"):
            counts["container"] *= st.ca_multiplier
            errors["container"] *= st.ca_multiplier
        if st.has("vanadium"):
            counts["vanadium"] *= st.va_multiplier
            errors["vanadium"] *= st.va_multiplier

        if ispec == 0:
            out_writer = DataWriter(st.files["output"], "TEXT", st.overwrite)
        hdr = dict(instrument=sample_blk.instrument, user=sample_blk.user,
                   nrun=sample_blk.nrun, nspec=sample_blk.nspec,
                   run_date=sample_blk.run_date)
        fnx = sample_blk.fnx

        # ---------------- 5. low-angle (direct beam) correction ----------
        if st.has("cadmium") or st.low_ba or st.low_fu or st.low_ca or st.low_va:
            n1, n2, n3, n4 = (int(np.searchsorted(theta, x)) for x in st.lowang)
            if n2 == n1:
                n2 = -1                      # Fortran sets n2=0: empty loop
            if n4 == n3:
                n4 = -1

            def summed(i0, i1):
                sl = slice(i0, i1 + 1)
                if cad is not None:
                    return cad.y[sl].astype(float)
                s = np.zeros(max(0, i1 + 1 - i0))
                if st.low_fu and "furnace" in counts:   s = s + counts["furnace"][sl]
                if st.low_ca and "container" in counts: s = s + counts["container"][sl]
                if st.low_va and "vanadium" in counts:  s = s + counts["vanadium"][sl]
                if st.low_ba and bkg_c is not None:     s = s + bkg_c[sl]
                return s

            nfit = sum([st.low_ba, st.low_fu, st.low_ca, st.low_va]) or 1
            coeff = np.zeros(2)
            if n4 - n3 + 1 > 1:
                coeff, _ = polyfit(theta[n3:n4 + 1], summed(n3, n4), np.ones(n4 - n3 + 1), 2)
            if n2 >= n1:
                sl = slice(n1, n2 + 1)
                th = theta[sl]
                diff = summed(n1, n2)
                if cad is not None:
                    diff = diff - coeff[0] - coeff[1] * th
                    ediff = cad.e[sl].astype(float)
                else:
                    if f:      # Fortran operator precedence: only slope /nfit
                        diff = diff - coeff[0] - coeff[1] * th / nfit
                    else:      # per-spectrum excess over the extrapolated line
                        diff = (diff - coeff[0] - coeff[1] * th) / nfit
                    e2 = np.zeros(len(th))
                    if st.low_ba:   # Fortran uses errors(i,1) (furnace) here
                        src = errors.get("furnace") if f else bkg_e
                        if src is not None: e2 += src[sl] ** 2
                    if st.low_fu and "furnace" in errors:   e2 += errors["furnace"][sl] ** 2
                    if st.low_ca and "container" in errors: e2 += errors["container"][sl] ** 2
                    if st.low_va and "vanadium" in errors:  e2 += errors["vanadium"][sl] ** 2
                    ediff = np.sqrt(e2) / nfit
                for body in BODIES:            # not if background subtracted
                    if st.has(body) and not st.has(BKG_OF[body]):
                        counts[body][sl] -= diff
                        errors[body][sl] = np.sqrt(errors[body][sl] ** 2 + ediff ** 2)
            if st.write_low_angle:
                out_writer.append(theta, counts["sample"], errors["sample"], title=sample_blk.title,
                                  fnx=fnx, fny="Counts (low angle corrected)", **hdr)

        res = {"block": blk, "theta": theta, "q": q}
        col = st.collimator

        # ---------------- 6a. vanadium attenuation ------------------------
        if st.has("vanadium"):
            cv, ev = counts["vanadium"], errors["vanadium"]
            cf = counts.get("furnace", np.zeros(npts)); ef = errors.get("furnace", np.zeros(npts))
            cc = counts.get("container", np.zeros(npts)); ec = errors.get("container", np.zeros(npts))
            if st.va_furnace and not st.va_container:
                A = absor(st.geometry, st.van_angle, st.va_r, 0.0, st.fu_r1, st.fu_r2,
                          st.beam_width, p["va_tmu"], 0.0, p["fu_tmu"], theta, col, f)
                rh = A["ahsch"] / A["ahh"]
                cv2 = (cv - rh * cf) / A["assch"]
                ev2 = np.sqrt(ev**2 + (rh if f else rh**2) * ef**2) / A["assch"]
            elif st.va_furnace and st.va_container:
                A = absor(st.geometry, st.van_angle, st.ca_r1, st.ca_r2, st.fu_r1, st.fu_r2,
                          st.beam_width, p["va_tmu"], p["ca_tmu"], p["fu_tmu"], theta, col, f)
                rc = A["acsch"] / A["acch"]
                rh = A["ahsch"] / A["ahh"] - A["acsch"] * A["ahch"] / A["acch"] / A["ahh"]
                cv2 = (cv - rc * cc - rh * cf) / A["assch"]
                if f:
                    rhe = A["ahsch"] / A["ahh"] + A["acsch"] * A["ahch"] / A["acch"] / A["ahh"]
                    ev2 = np.sqrt(ev**2 + rc * ec**2 + rhe * ef**2) / A["assch"]
                else:
                    ev2 = np.sqrt(ev**2 + rc**2 * ec**2 + rh**2 * ef**2) / A["assch"]
            elif st.va_container:
                A = absor(st.geometry, st.van_angle, st.ca_r1, st.ca_r2, st.fu_r1, st.fu_r2,
                          st.beam_width, p["va_tmu"], p["ca_tmu"], 0.0, theta, col, f)
                rc = A["acsch"] / A["acch"]
                cv2 = (cv - rc * cc) / A["assch"]
                ev2 = np.sqrt(ev**2 + (rc if f else rc**2) * ec**2) / A["assch"]
            else:
                A = absor(st.geometry, st.van_angle, st.va_r, 0.0, st.fu_r1, st.fu_r2,
                          st.beam_width, p["va_tmu"], 0.0, 0.0, theta, col, f)
                cv2, ev2 = cv / A["assch"], ev / A["assch"]
            counts["vanadium"], errors["vanadium"] = cv2, ev2
            # names used by the Fortran listing for the vanadium factors
            res.update(avvch=A["assch"], acchv=A["acch"], acvch=A["acsch"],
                       ahvch=A["ahsch"], ahchv=A["ahch"], ahhv=A["ahh"])

        # ---------------- 6b. sample, container, furnace -----------------
        A = absor(st.geometry, st.sample_angle, st.ca_r1, st.ca_r2, st.fu_r1, st.fu_r2,
                  st.beam_width, p["sa_tmu"], p["ca_tmu"], p["fu_tmu"], theta, col, f)
        res.update(A)
        zero = np.zeros(npts)
        ch, eh = counts.get("furnace", zero) / A["ahh"], errors.get("furnace", zero) / A["ahh"]
        cc = (counts.get("container", zero) - A["ahch"] * ch) / A["acch"]
        ec = np.sqrt(errors.get("container", zero) ** 2 + (A["ahch"] * eh) ** 2) / A["acch"]
        cs = (counts["sample"] - A["acsch"] * cc - A["ahsch"] * ch) / A["assch"]
        es = np.sqrt(errors["sample"] ** 2 + (A["acsch"] * ec) ** 2
                     + (A["ahsch"] * eh) ** 2) / A["assch"]
        if st.has("furnace"):
            counts["furnace"], errors["furnace"] = ch, eh
        if st.has("container"):
            counts["container"], errors["container"] = cc, ec
        counts["sample"], errors["sample"] = cs, es
        res["sample_abs_corrected"] = cs.copy()

        if st.write_absorption:
            for key, ttl in (("assch", "As,sch"), ("acsch", "Ac,sch"), ("acch", "Ac,ch"),
                             ("ahsch", "Ah,sch"), ("ahch", "Ah,ch"), ("ahh", "Ah,h")):
                out_writer.append(theta, A[key], A[key], title=ttl, fnx=fnx, fny=key, **hdr)
            if st.has("vanadium"):
                out_writer.append(theta, res["avvch"], res["avvch"], title="Av",
                                  fnx=fnx, fny="Av", **hdr)

        # ---------------- 7. calibrant: multiple scattering + Placzek -----
        pc_va = np.zeros(npts)
        if st.placzek_correction != "NONE":
            pc_va = placzek(theta, lam, 50.0, 300.0, st.instrument,
                            st.placzek_correction, st.omegav)
        if st.has("vanadium"):
            delta_va = (mscat(st.va_r, st.beam_height, p["va_tmu"], p["va_smu"],
                              faithful=f, verbose=verbose)
                        if st.geometry != "PLATE" else 0.0)
            pc_va = pc_va * (1 + delta_va)
            counts["vanadium"] = counts["vanadium"] / (1 - pc_va + delta_va)
            errors["vanadium"] = errors["vanadium"] / (1 - pc_va + delta_va)
            calib = "vanadium"
        else:
            if st.geometry != "PLATE":
                shell = 1 - st.ca_r1 ** 2 / st.ca_r2 ** 2     # volume fraction of wall
                if f:   # Fortran passes (smu, amu) where (tmu, smu) are expected
                    delta_va = mscat(st.ca_r2, st.beam_height, p["ca_smu"] * shell,
                                     p["ca_amu"] * shell, faithful=True, verbose=verbose)
                else:
                    delta_va = mscat(st.ca_r2, st.beam_height, p["ca_tmu"] * shell,
                                     p["ca_smu"] * shell, faithful=False, verbose=verbose)
            else:
                delta_va = 0.0
            pc_va = pc_va * (1 + delta_va)
            counts["container"] = counts["container"] / (1 - pc_va + delta_va)
            errors["container"] = errors["container"] / (1 - pc_va + delta_va)
            calib = "container"
        res["delta_va"], res["pc_va"] = delta_va, pc_va
        res["calib_raw"] = counts[calib].copy()

        if st.write_calibration:
            out_writer.append(theta, counts[calib], errors[calib],
                              title="Vanadium" if calib == "vanadium" else "Can",
                              fnx=fnx, fny="Counts (corrected)", **hdr)

        # ---------------- 8. smoothing and normalisation ------------------
        vs = st.vanadium_smoothing
        if vs >= 0:
            counts[calib] = smooth_vanadium(theta, counts[calib], errors[calib], vs, f)
        else:
            counts[calib] = smooth_vanadium_sg(theta, counts[calib], vs, lam,
                                               st.calib_ur, st.calib_vr, st.calib_wr, f)
        res["calib_smooth"] = counts[calib].copy()
        if calib == "vanadium":
            factor = st.va_rho * st.va_r ** 2 * p["va_b2"] / (st.sa_rho * st.ca_r1 ** 2)
        else:
            factor = (st.ca_rho * (st.ca_r2 ** 2 - st.ca_r1 ** 2) * p["ca_b2"]
                      / (st.sa_rho * st.ca_r1 ** 2))
        counts["sample"] = counts["sample"] / counts[calib] * factor
        errors["sample"] = errors["sample"] / counts[calib] * factor

        if st.write_calibration:
            out_writer.append(theta, counts[calib], errors[calib],
                              title="Vanadium smoothed" if calib == "vanadium" else "Can smoothed",
                              fnx=fnx, fny="Counts (corrected)", **hdr)

        # ---------------- 9. sample multiple scattering and Placzek ------
        if st.geometry != "PLATE":
            fs = (st.ca_r1 / st.ca_r2) ** 2
            delta_sa = mscat(st.ca_r2, st.beam_height,
                             (1 - fs) * p["ca_tmu"] + fs * p["sa_tmu"],
                             (1 - fs) * p["ca_smu"] + fs * p["sa_smu"],
                             faithful=f, verbose=verbose)
        else:
            delta_sa = 0.0
        pc_sa = np.zeros(npts)
        if st.placzek_correction != "NONE":
            for fr, sp in zip(p["frac"], st.species):
                pp = placzek(theta, lam, sp.weight, st.sa_temp, st.instrument,
                             st.placzek_correction, st.omegav)
                pc_sa += fr * sp.scatt_cs / 4 / PI * pp * (1 + delta_sa)
        counts["sample"] = counts["sample"] - delta_sa * p["sa_b2"] + pc_sa
        res.update(delta_sa=delta_sa, pc_sa=pc_sa,
                   dcs=counts["sample"], err=errors["sample"])

        # ---------------- display (Fortran FORMAT 90..150) ----------------
        if verbose:
            _display(ispec + 1, st, p, res)

        if st.write_placzek:
            out_writer.append(q, pc_sa, errors["sample"], title="Inelasticity",
                              fnx="Momentum transfer (A**-1)", fny="P (barns)", **hdr)

        # ---------------- 10. output ---------------------------------------
        title = st.new_title or sample_blk.title
        if st.xouttype == "ANGLE":
            out_writer.append(theta, counts["sample"], errors["sample"], title=title,
                              fnx="Scattering angle (degrees)", fny="Cross-section", **hdr)
        else:
            out_writer.append(q, counts["sample"], errors["sample"], title=title,
                              fnx="Momentum transfer (A**-1)", fny="Cross-section", **hdr)
        results.append(res)

    if out_writer is not None:
        out_writer.close()
    return results


def _display(ispec, st, p, r):
    """Print the correction tables every 5 degrees (Fortran FORMATs 90-150)."""
    theta, q = r["theta"], r["q"]
    rows = []
    nxt = 5.0 * int(theta[0] / 5.0)
    for i, t in enumerate(theta):
        if t >= nxt:
            rows.append(i)
            nxt += 5.0
    if not st.has("vanadium"):
        print(f"\n\n\n Spectrum {ispec:2d}\n\n Calibration done with can. Can multiple "
              "scattering and Placzek corrections:\n\n  Angle  Q    m.s.     Placzek")
        for i in rows:
            print(f" {theta[i]:6.2f}{q[i]:6.2f}{r['delta_va']:9.5f}{r['pc_va'][i]:9.5f}")
    else:
        print(f"\n\n\n Spectrum {ispec:2d}\n\n Vanadium absorption, multiple scattering and "
              f"Placzek corrections:\n\n Linear absorption coefficient x radius ={p['va_tmu'] * st.va_r:5.3f}\n")
        print("  Angle  Q    Avvch   Acvch   Acch    Ahvch   Ahch    Ahh      m.s.     Placzek")
        for i in rows:
            print(f" {theta[i]:6.2f}{q[i]:6.2f}" + "".join(f"{r[k][i]:8.5f}" for k in
                  ("avvch", "acvch", "acchv", "ahvch", "ahchv", "ahhv"))
                  + f"{r['delta_va']:9.5f}{r['pc_va'][i]:9.5f}")
    print(f"\n Sample absorption, multiple scattering and Placzek corrections:\n\n"
          f" Linear absorption coefficient x radius ={p['sa_tmu'] * st.ca_r1:5.3f}\n")
    print("  Angle  Q    Assch   Acsch   Acch    Ahsch   Ahch    Ahh      m.s.     Placzek")
    for i in rows:
        print(f" {theta[i]:6.2f}{q[i]:6.2f}" + "".join(f"{r[k][i]:8.5f}" for k in
              ("assch", "acsch", "acch", "ahsch", "ahch", "ahh"))
              + f"{r['delta_sa']:9.5f}{r['pc_sa'][i]:9.5f}")
