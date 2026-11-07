"""
correct_tof.py  --  Python translation of  correct_tofdif_224.f
=================================================================

Corrections for time-of-flight (pulsed source) diffractometers: LAD,
SANDALS, SEPD (ISIS / IPNS).  The physics is the same as in the 2-axis
case, but every quantity depends on wavelength instead of angle: each
spectrum is one detector bank at a fixed 2theta, and mu(lambda) changes
along the spectrum.

Fortran -> Python correspondence
--------------------------------
SUBROUTINE CORRECT_TOF   -> correct_tof(state)
SUBROUTINE NORM          -> norm_to_monitor(...)
SUBROUTINE ABSORB        -> absorb(...)     P&P factors vs wavelength
SUBROUTINE MULSCAT       -> mulscat(...)    Blech-Averbach delta vs wavelength
SUBROUTINE SPLINE_SMOOTH -> spline_smooth(...)   (VB06A/TG01B -> SciPy)
SUBROUTINE PLAMOL        -> plamol(...)     Placzek, diatomic molecule
SUBROUTINE PLAATOM       -> plaatom(...)    Placzek, atomic (series)
SUBROUTINE LITLF         -> litlf(...)      incident flux spectrum + derivatives
SUBROUTINE BIGF          -> bigf(...)       detector efficiency + derivatives

Note on the original PC version
-------------------------------
CORRECT_TOF set the input format to 'GENIE' (VAX binary).  On the PC build
READ_FILE does nothing for GENIE, so the TOF branch could not actually run
on a PC.  Here TEXT/DATA files are read (state.data_format), so it works.

Known bugs (reproduced only with state.faithful = True)
-------------------------------------------------------
* after the wavelength-range cut the Fortran copies COUNTS into ERRORS
  (`errors(i,isam)=counts(i+imin-1,isam)`), and does not shift XDATA;
* vanadium in furnace: uses AHSCH (not yet computed / from the previous
  spectrum) instead of AHVH, and SUBTRACTS the furnace error in quadrature.
  The second point is reproduced; for the first, garbage cannot be
  reproduced, so AHVH is always used.
"""

from __future__ import annotations

import math

import numpy as np

from .fileio import read_file, DataWriter
from .harwell import lsq_cubic_spline
from .ndplib import atcors, mscat
from .routines import plate, polyfit, polyval_c
from .state import CorrectState, BKG_OF

PI = math.pi
BODIES = ["furnace", "container", "sample", "vanadium"]


def norm_to_monitor(xmon, mon, emon, x, c, e):
    """Divide a spectrum by the monitor linearly interpolated onto its x.

    Fortran: SUBROUTINE NORM.  Error propagation as in Fortran:
        e' = sqrt(e^2 + c^2 em^2/m^2) / m
    with em itself 'interpolated' the Fortran way.
    (numpy.interp could do the interpolation of `mon`; the explicit loop
    keeps the Fortran error formula.)
    """
    nmon = len(xmon)
    cn, en = np.empty_like(c), np.empty_like(e)
    i1 = 0
    for j in range(len(x)):
        while i1 < nmon - 2 and xmon[i1 + 1] < x[j]:
            i1 += 1
        fr = (x[j] - xmon[i1]) / (xmon[i1 + 1] - xmon[i1])
        m = mon[i1] + (mon[i1 + 1] - mon[i1]) * fr
        em = math.sqrt(emon[i1] ** 2 + ((emon[i1 + 1] - emon[i1]) * fr) ** 2)
        en[j] = math.sqrt(e[j] ** 2 + c[j] ** 2 * em * em / m / m) / m
        cn[j] = c[j] / m
    return cn, en


def absorb(geometry, plate_angle, r1, r2, r3, r4, sw, angle_deg,
           fu_smu, fu_amu, ca_smu, ca_amu, wavcs, sa_tcs, sa_rho, wav,
           collimator=None, faithful=False):
    """P&P factors as a function of wavelength at fixed 2theta.

    Fortran: SUBROUTINE ABSORB.  The factors are calculated exactly at the
    ncs (=21) wavelengths of `wavcs`, where the sample total cross section is
    sa_tcs, then a 4th-degree polynomial in lambda is fitted and evaluated
    at the data wavelengths `wav`.
    """
    from .ndplib import NO_COLLIMATOR
    col = collimator or NO_COLLIMATOR
    theta = angle_deg * PI / 180.0
    wavcs = np.asarray(wavcs, float)
    a = np.ones((6, len(wavcs)))
    for i, w in enumerate(wavcs):
        fu_mu = fu_smu + fu_amu * w / 1.798
        ca_mu = ca_smu + ca_amu * w / 1.798
        sa_mu = sa_tcs[i] * sa_rho
        if geometry == "CYLINDRICAL":
            a[:, i] = atcors(theta, r1, r2, r3, r4, sw, sa_mu, ca_mu, fu_mu, col)
        else:
            a[:3, i] = plate(theta, plate_angle, r1, r2, sa_mu, ca_mu, faithful)
    names = ["assch", "acch", "acsch", "ahsch", "ahch", "ahh"]
    out = {}
    for k, name in enumerate(names):
        c, _ = polyfit(wavcs, a[k], np.ones(len(wavcs)), 5)
        out[name] = polyval_c(c, wav)
    return out


def mulscat(wav, r, h, smu, amu, *, faithful=False, verbose=True):
    """Blech-Averbach delta(lambda): computed at 21 wavelengths spanning the
    data range (+1%), cubic polynomial fitted and evaluated on `wav`.

    Fortran: SUBROUTINE MULSCAT(WAV,NPTS,R,H,SMU,AMU,MS)
    smu, amu : scattering and absorption (at 1.798 A) linear coefficients.
    """
    wav = np.asarray(wav, float)
    wrange = abs(wav[-1] - wav[0]) * 1.01
    wmid = 0.5 * (wav[0] + wav[-1])
    x = wmid + (np.arange(1, 22) - 11) * wrange / 20.0
    d = np.array([mscat(r, h, smu + amu * xi / 1.798, smu, faithful=faithful,
                        verbose=verbose) for xi in x])
    c, _ = polyfit(x, d, np.ones(21), 4)
    return polyval_c(c, wav)


def spline_smooth(x, y, nknot=20):
    """Smooth y(x) with a least-squares cubic spline with `nknot` knots placed
    at equally spaced *indices* of the data.

    Fortran: SUBROUTINE SPLINE_SMOOTH(X,Y,ISAM,NPTS,MAXPTS,NKNOT) which calls
    the Harwell routines VB06A (fit) and TG01B (evaluation).
    Knot positions as in Fortran: the first/last knot slightly outside the
    data, interior knots at x(npts*i/nknots), i = 2..nknots-1 (1-based).
    """
    x = np.asarray(x, float)
    npts = len(x)
    dx = 0.001 * (x[-1] - x[0])
    xm = x.copy()
    xm[0] -= dx
    xm[-1] += dx
    xn = np.empty(nknot)
    xn[0] = xm[0] + 0.5 * dx
    for i in range(2, nknot):                     # Fortran i = 2..nknots-1
        xn[i - 1] = xm[(npts * i) // nknot - 1]
    xn[-1] = xm[-1] - 0.5 * dx
    if xn[-1] < xn[0]:                            # decreasing x: flip knots
        xn = xn[::-1]
    spl = lsq_cubic_spline(xm, y, xn)
    return spl(x)


# --------------------------------------------------------------------------
#  Placzek corrections for TOF
# --------------------------------------------------------------------------
FLUX_PARAS = (3.49, 0.09, 7.96, 2.74, 0.0, 1.14, 0.15, 0.0, 0.0)
# fepi, alpha, fmax, lambdat, cona, lambda1, lambda2, lambda3, lambda4


def litlf(wav, dat=FLUX_PARAS):
    """Model of the incident spectrum phi(lambda) (Maxwellian + epithermal
    1/lambda^(2 alpha+1) with a cut-off) and its logarithmic derivatives.

    Fortran: SUBROUTINE LITLF(NPTS,DAT,WAV,FL,FL1,FL2)
    Returns fl, fl1 = (lambda/phi) dphi/dlambda-type first-order term,
    fl2 second-order term, as used by the Placzek expansions.
    """
    fepi, alph, fmax, wat, cona, al1, al2 = dat[:7]
    al3, al4 = (dat[7], dat[8]) if cona >= 0.001 else (0.0, 0.00001)
    wav = np.asarray(wav, float)
    rw2 = wat * wat / (wav * wav)
    xpt = np.where(rw2 < 80, np.exp(-np.minimum(rw2, 80)), 0.0)
    fim = rw2 * rw2 * xpt / wav
    fim2 = fim * (30.0 - rw2 * (26.0 - 4.0 * rw2))
    fim1 = fim * (2.0 * rw2 - 5.0)
    wal = (wav - al1) / al2
    xp1 = np.where(wal > 80, 1.0e30, np.exp(np.minimum(wal, 80)))
    xp2 = np.exp((al3 - wav) / al4) if cona > 0.001 else 0.0
    dl = (1.0 / (1.0 + xp1)) * (1.0 + cona / (1.0 + xp2))
    tw1 = 2.0 * alph + 1.0
    fiep = dl / wav ** tw1
    g = wav / al2 / (1.0 / xp1 + 1.0)
    fep1 = -fiep * (tw1 - g)
    fep2 = fiep * ((tw1 + g) ** 2 + tw1
                   - wav * wav / al2 / al2 / xp1 / (1 + 1 / xp1) / (1 + 1 / xp1))
    fl = fim * fmax + fiep * fepi
    fl1 = (fim1 * fmax + fep1 * fepi) / fl
    fl2 = (fim2 * fmax + fep2 * fepi) / fl
    return fl, fl1, fl2


def bigf(k, det_type="HE_GAS"):
    """Detector efficiency eps(k) = 1 - exp(-dk/k) and its derivatives.

    Fortran: SUBROUTINE BIGF(NPTS,K,BF,BF1,BF2,DET_TYPE)
    HE_GAS: dk = 2 pi / 1.44 ;  SCINTILLATION: dk = 2 pi * 8.3 * 0.1 cm.
    """
    dk = 2 * PI / 1.44 if det_type == "HE_GAS" else 2 * PI * 8.3 * 0.1
    rk = dk / np.asarray(k, float)
    a = np.exp(-rk)
    b = 1 - a
    return b, -rk * a / b, rk * (2 - rk) * a / b


def plamol(wav, wt, scs, temp, angle, det_type, path1, path2):
    """Placzek correction for a homonuclear/heteronuclear DIATOMIC molecule
    (uses the first two species only).  Fortran: SUBROUTINE PLAMOL."""
    wav = np.asarray(wav, float)
    rad = abs(angle * PI / 180.0)
    as2 = math.sin(rad / 2) ** 2
    ac = math.cos(rad)
    r = path2 / path1
    rr = 1 + r
    k = 2 * PI / wav
    _, bf1, bf2 = bigf(k, det_type)
    _, fl1, fl2 = litlf(wav)
    d = 0.5 * (bf2 + (9 * r + 3) * bf1 + r * r * fl2 + r * (9 * r + 1) * fl1
               + 2 * r * bf1 * fl1 + (1 + 5 * r + 16 * r * r))
    t1 = (bf1 + r * fl1 + 2 + 3 * r) * as2 / rr
    energy = 24.08 * k * k / temp
    t2 = 0.25 * (ac + 4 * d * as2 / rr / rr) / energy
    w1, w2 = wt[0], wt[1]
    return ((-t1 / w1 + t2 * (1 + w2 / w1) / (1 + 0.5 * w2 / w1) / w1) * scs[0] / 4 / PI
            + (-t1 / w2 + t2 * (1 + w1 / w2) / (1 + 0.5 * w1 / w2) / w2) * scs[1] / 4 / PI)


def plaatom(wav, wgt, frac, scs, temp, angle, det_type, path1, path2):
    """Placzek correction (series expansion to second order) for atomic
    systems, including flux and detector-efficiency derivatives.
    Fortran: SUBROUTINE PLAATOM (Howe, McGreevy & Howells formulation)."""
    wav = np.asarray(wav, float)
    k = 2 * PI / wav
    _, bf1, bf2 = bigf(k, det_type)
    _, fl1, fl2 = litlf(wav)
    f = path1 / (path1 + path2)
    sth = math.sin(abs(angle) * PI / 360.0)
    s2 = sth * sth
    kte = 1.0534026e-3 * wav * wav * temp
    cor1 = (2 * s2 * ((f - 1) * fl1 - f * bf1 + f - 3) + kte / 2
            + kte * s2 * ((8 * f - 9) * (f - 1) * fl1 + 3 * f * (3 - 2 * f) * bf1
                          + 2 * f * (1 - f) * fl1 * bf1 + (1 - f) ** 2 * fl2
                          + f * f * bf2 + 3 * (4 * f - 5) * (f - 1)))
    cor2 = (2 * s2 + 2 * sth ** 4 * ((4 * f - 7) * (f - 1) * fl1 + f * (7 - 2 * f) * bf1
                                    + 2 * f * (1 - f) * bf1 * fl1 + (1 - f) ** 2 * fl2
                                    + f * f * bf2 + (2 * f * f - 7 * f + 8)))
    nmass = 1.0
    p = np.zeros_like(wav)
    for wj, fj, sj in zip(wgt, frac, scs):
        p += sj / 4 / PI * nmass / wj * fj * (cor1 + nmass / wj * cor2)
    return p


# --------------------------------------------------------------------------
#  Main driver
# --------------------------------------------------------------------------
def correct_tof(st: CorrectState, verbose: bool = True):
    """Run the TOF corrections for every spectrum.  Fortran: CORRECT_TOF.
    Returns a list of dictionaries (as correct_2axis)."""
    f = st.faithful
    fmt = st.data_format
    if st.ca_r2 <= 0:
        st.ca_r2 = st.ca_r1
    ca_smu, ca_amu = st.ca_scat_cs * st.ca_rho, st.ca_abs_cs * st.ca_rho
    ca_b2 = st.ca_scat_cs / 4 / PI
    fu_smu, fu_amu = st.fu_scat_cs * st.fu_rho, st.fu_abs_cs * st.fu_rho
    va_smu, va_amu = st.va_scat_cs * st.va_rho, st.va_abs_cs * st.va_rho
    va_b2 = st.va_scat_cs / 4 / PI
    ftot = sum(s.fraction for s in st.species) or 1.0
    frac = [s.fraction / ftot for s in st.species]
    sa_scat = sum(fr * s.scatt_cs for fr, s in zip(frac, st.species))
    sa_abs = sum(fr * s.abs_cs for fr, s in zip(frac, st.species))
    sa_smu, sa_amu = sa_scat * st.sa_rho, sa_abs * st.sa_rho
    sa_b2 = sa_scat / 4 / PI

    # ---- total cross sections at 21 wavelengths (or from cs_file) --------
    if st.has("cs_file"):
        with open(st.files["cs_file"]) as fh:
            vals = fh.read().split()
        ncs = int(vals[0])
        rang1, rang2 = float(vals[1]), float(vals[2])
        if rang1 > st.wav_min or rang2 < st.wav_max:
            raise ValueError("Absorption file wavelength range is inconsistent with data")
        if ncs != 21:
            raise ValueError(f"{ncs} data points in absorption file. 21 are required")
        arr = np.array(vals[3:3 + 2 * ncs], float).reshape(ncs, 2)
        wavcs, sa_tcs = arr[:, 0], arr[:, 1]
    else:
        wavcs = (st.wav_max - st.wav_min) / 20 * np.arange(21) + st.wav_min
        sa_tcs = sa_scat + sa_abs * wavcs / 1.798
    va_tcs = st.va_scat_cs + st.va_abs_cs * wavcs / 1.798
    if verbose:
        print("\n\n Total cross-sections:\n\n Wavelength  c-s ")
        for w, s in zip(wavcs, sa_tcs):
            print(f"      {w:5.2f}  {s:6.2f}")

    writer = DataWriter(st.files["output"], "TEXT", st.overwrite)
    results = []
    for spec in st.spectra:
        blk = spec.block
        counts, errors = {}, {}
        sample_blk = None
        xdata = None
        for body in BODIES:
            if not st.has(body):
                continue
            # monitor of data (and of background), converted to point mode
            if st.monitor_spectrum:
                mb = read_file(st.files[body], st.monitor_spectrum, fmt)
                xmon = mb.x_with_edge()
                mbb = None
                if st.has(BKG_OF[body]):
                    mbb = read_file(st.files[BKG_OF[body]], st.monitor_spectrum, fmt)
                if mb.histogram and mb.x_edge is not None:
                    xmon = 0.5 * (xmon[:-1] + xmon[1:])
            b = read_file(st.files[body], blk, fmt)
            if body == "sample":
                sample_blk = b
            c, e = b.y.astype(float), b.e.astype(float)
            bc = be = None
            if st.has(BKG_OF[body]):
                bb = read_file(st.files[BKG_OF[body]], blk, fmt)
                bc, be = bb.y.astype(float), bb.e.astype(float)
            x = b.x_with_edge()
            if b.histogram and b.x_edge is not None:
                x = 0.5 * (x[:-1] + x[1:])                 # point mode
            if st.monitor_spectrum:
                keep = (x >= xmon[0]) & (x <= xmon[-1])
                ns, ne = np.argmax(keep), len(keep) - np.argmax(keep[::-1]) - 1
                c, e = norm_to_monitor(xmon, mb.y, mb.e, x[ns:ne + 1], c[ns:ne + 1], e[ns:ne + 1])
                if bc is not None:
                    bc, be = norm_to_monitor(xmon, mbb.y, mbb.e, x[ns:ne + 1],
                                             bc[ns:ne + 1], be[ns:ne + 1])
                x = x[ns:ne + 1]
            if bc is not None:
                c = c - bc
                e = np.sqrt(e * e + be * be)
            if body == "container":
                c, e = c * st.ca_multiplier, e * st.ca_multiplier
            if body == "vanadium":
                c, e = c * st.va_multiplier, e * st.va_multiplier
            counts[body], errors[body] = c, e
            xdata = x
        npts = len(xdata)
        for body in BODIES:
            counts.setdefault(body, np.zeros(npts))
            errors.setdefault(body, np.zeros(npts))

        # ---- remove zero-count ends (and x = 0) ---------------------------
        ns = 1 if xdata[0] == 0 else 0
        while counts["sample"][ns] == 0:
            ns += 1
        ne = npts - 1
        while counts["sample"][ne] == 0:
            ne -= 1
        sl = slice(ns, ne + 1)
        xdata = xdata[sl]
        for body in BODIES:
            counts[body], errors[body] = counts[body][sl], errors[body][sl]

        # ---- wavelength and Q ----------------------------------------------
        sin_t = math.sin(abs(spec.angle) * PI / 360.0)
        if st.xtype == "TOF":
            wav = 3.956033433 * xdata / (spec.path1 + spec.path2)
            q = 4 * PI / wav * sin_t
        elif st.xtype == "WAVELENGTH":
            wav = xdata.copy()
            q = 4 * PI / wav * sin_t
        else:
            q = xdata.copy()
            wav = 4 * PI / q * sin_t

        # ---- wavelength range ---------------------------------------------
        imin, imax = 0, len(wav) - 1
        if wav[1] > wav[0]:
            for i, w in enumerate(wav):
                if w < st.wav_min: imin = i + 1
                if w < st.wav_max: imax = i
        else:
            for i, w in enumerate(wav):
                if w > st.wav_max: imin = i + 1
                if w > st.wav_min: imax = i
        sl = slice(imin, imax + 1)
        for body in BODIES:
            if f:
                errors[body] = counts[body][sl].copy()      # Fortran bug
            else:
                errors[body] = errors[body][sl]
            counts[body] = counts[body][sl]
        wav, q = wav[sl], q[sl]
        if not f:
            xdata = xdata[sl]
        npts = len(wav)

        res = {"block": blk, "wav": wav, "q": q}
        col = st.collimator
        # ---- vanadium attenuation -------------------------------------------
        if st.has("vanadium"):
            if st.va_furnace:
                A = absorb(st.geometry, st.van_angle, st.va_r, st.va_r, st.fu_r1, st.fu_r2,
                           st.beam_width, spec.angle, fu_smu, fu_amu, 0.0, 0.0,
                           wavcs, va_tcs, st.va_rho, wav, col, f)
                rh = A["ahsch"] / A["ahh"]                  # = ahvh/ahh
                counts["vanadium"] = (counts["vanadium"] - rh * counts["furnace"]) / A["assch"]
                sgn = -1.0 if f else 1.0
                errors["vanadium"] = np.sqrt(np.maximum(
                    errors["vanadium"] ** 2 + sgn * (rh * errors["furnace"]) ** 2, 0)) / A["assch"]
            else:
                A = absorb(st.geometry, st.van_angle, st.va_r, st.va_r, st.fu_r1, st.fu_r2,
                           st.beam_width, spec.angle, 0.0, 0.0, 0.0, 0.0,
                           wavcs, va_tcs, st.va_rho, wav, col, f)
                counts["vanadium"] /= A["assch"]
                errors["vanadium"] /= A["assch"]
            res.update(avvh=A["assch"], ahvh=A["ahsch"], ahhv=A["ahh"])

        A = absorb(st.geometry, st.sample_angle, st.ca_r1, st.ca_r2, st.fu_r1, st.fu_r2,
                   st.beam_width, spec.angle, fu_smu, fu_amu, ca_smu, ca_amu,
                   wavcs, sa_tcs, st.sa_rho, wav, col, f)
        res.update(A)
        ch, eh = counts["furnace"] / A["ahh"], errors["furnace"] / A["ahh"]
        cc = (counts["container"] - A["ahch"] * ch) / A["acch"]
        ec = np.sqrt(errors["container"] ** 2 + (A["ahch"] * eh) ** 2) / A["acch"]
        cs = (counts["sample"] - A["acsch"] * cc - A["ahsch"] * ch) / A["assch"]
        es = np.sqrt(errors["sample"] ** 2 + (A["acsch"] * ec) ** 2
                     + (A["ahsch"] * eh) ** 2) / A["assch"]
        counts.update(furnace=ch, container=cc, sample=cs)
        errors.update(furnace=eh, container=ec, sample=es)
        hdr = dict(instrument=sample_blk.instrument, user=sample_blk.user,
                   nrun=sample_blk.nrun, nspec=sample_blk.nspec, run_date=sample_blk.run_date)
        fnx = sample_blk.fnx
        if st.write_absorption:
            for key, ttl in (("assch", "As,sch"), ("acsch", "Ac,sch"), ("acch", "Ac,ch"),
                             ("ahsch", "Ah,sch"), ("ahch", "Ah,ch"), ("ahh", "Ah,h")):
                writer.append(xdata, A[key], A[key], title=ttl, fnx=fnx, fny=key, **hdr)

        # ---- multiple scattering of calibrant --------------------------------
        if st.has("vanadium"):
            delta_va = mulscat(wav, st.va_r, st.beam_height, va_smu, va_amu,
                               faithful=f, verbose=verbose)
            calib = "vanadium"
        else:
            shell = 1 - st.ca_r1 ** 2 / st.ca_r2 ** 2
            delta_va = mulscat(wav, st.ca_r2, st.beam_height, ca_smu * shell,
                               ca_amu * shell, faithful=f, verbose=verbose)
            calib = "container"
        counts[calib] = counts[calib] / (1 + delta_va)
        errors[calib] = errors[calib] / (1 + delta_va)
        res["delta_va"] = delta_va
        res["calib_raw"] = counts[calib].copy()
        if st.write_ms:
            writer.append(xdata, delta_va, delta_va, title="Multiple scattering",
                          fnx=fnx, fny="Fraction", **hdr)
        if st.write_calibration:
            writer.append(xdata, counts[calib], errors[calib], title=f"{calib} corrected",
                          fnx=fnx, fny="Counts (corrected)", **hdr)

        # ---- smoothing and normalisation ---------------------------------
        if calib == "vanadium":
            if st.vanadium_smoothing == 1:
                counts[calib] = spline_smooth(xdata if not f else xdata[:npts], counts[calib], 20)
            factor = st.va_rho * st.va_r ** 2 * va_b2 / (st.sa_rho * st.ca_r1 ** 2)
        else:
            counts[calib] = spline_smooth(xdata if not f else xdata[:npts], counts[calib], 20)
            factor = st.ca_rho * (st.ca_r2 ** 2 - st.ca_r1 ** 2) * ca_b2 / (st.sa_rho * st.ca_r1 ** 2)
        res["calib_smooth"] = counts[calib].copy()
        counts["sample"] = counts["sample"] / counts[calib] * factor
        errors["sample"] = errors["sample"] / counts[calib] * factor
        if st.write_calibration:
            writer.append(xdata, counts[calib], errors[calib], title=f"{calib} smoothed",
                          fnx=fnx, fny="Counts (corrected)", **hdr)

        # ---- sample multiple scattering and Placzek -----------------------
        fs = (st.ca_r1 / st.ca_r2) ** 2
        delta_sa = mulscat(wav, st.ca_r2, st.beam_height,
                           (1 - fs) * ca_smu + fs * sa_smu, (1 - fs) * ca_amu + fs * sa_amu,
                           faithful=f, verbose=verbose)
        wts = [s.weight for s in st.species]
        scs = [s.scatt_cs for s in st.species]
        if st.placzek_correction == "DIATOMIC":
            pc_sa = plamol(wav, wts, scs, st.sa_temp, spec.angle, spec.det_type,
                           spec.path1, spec.path2)
        elif st.placzek_correction == "SERIES_EXPANSION":
            pc_sa = plaatom(wav, wts, frac, scs, st.sa_temp, spec.angle, spec.det_type,
                            spec.path1, spec.path2)
        else:
            pc_sa = np.zeros(npts)
        counts["sample"] = counts["sample"] - delta_sa * sa_b2 + pc_sa
        res.update(delta_sa=delta_sa, pc_sa=pc_sa, dcs=counts["sample"], err=errors["sample"])

        if verbose:
            print(f"\n\n\n Spectrum {blk:2d} at a scattering angle of {spec.angle:6.1f} degrees:\n")
            print("  Wavel  Q    Assch   Acsch   Acch    Ahsch   Ahch    Ahh      m.s.     Placzek")
            step = max(1, npts // 20)
            for i in range(0, npts, step):
                print(f" {wav[i]:6.2f}{q[i]:6.2f}" + "".join(f"{A[k][i]:8.5f}" for k in
                      ("assch", "acsch", "acch", "ahsch", "ahch", "ahh"))
                      + f"{delta_sa[i]:9.5f}{pc_sa[i]:9.5f}")
        if st.write_ms:
            writer.append(xdata, delta_sa, delta_sa, title="Multiple scattering",
                          fnx=fnx, fny="Fraction", **hdr)
        if st.write_placzek:
            writer.append(xdata, pc_sa, pc_sa, title="Placzek correction",
                          fnx=fnx, fny="P (barns)", **hdr)

        # ---- output in increasing Q ------------------------------------------
        title = st.new_title or sample_blk.title
        if st.xtype == "Q":
            xo, yo, eo = q, counts["sample"], errors["sample"]
        else:
            xo, yo, eo = q[::-1], counts["sample"][::-1], errors["sample"][::-1]
        writer.append(xo, yo, eo, title=title, fnx="Momentum transfer",
                      fny="Cross-section", **hdr)
        results.append(res)
    writer.close()
    return results
