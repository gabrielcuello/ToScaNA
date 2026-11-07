"""
correct.py  --  Python translation of  correct_224.f  (main program)
=====================================================================

    python -m pycorrect                    interactive prompt  CORRECT>
    python -m pycorrect myrun.com          run a command file
    python -m pycorrect @myrun             same (".com" added if no extension)

Fortran -> Python correspondence
--------------------------------
PROGRAM CORRECT        -> main() and Correct.run_file()/interactive()
BLOCK DATA             -> state.CorrectState defaults
SUBROUTINE OBEY_LINE   -> Correct.obey_line()
SUBROUTINE CORRECT_LIST-> Correct.correct_list()
SUBROUTINE READ_ATOM_DATA -> read_atom_data()
FUNCTION FULLNAME      -> os.path.abspath
FUNCTION LENUSED       -> str.rstrip / len   (not needed in Python)

Differences with the Fortran behaviour
--------------------------------------
* After EXECUTE the Fortran program ends (CALL EXIT inside CORRECT_2AXIS).
  Here `execute` stores the results in Correct.results and the command file
  goes on with its next line (add 'quit' to stop explicitly).
* The qualifiers /furnace and /container of 'vanadium' work without a value
  (in Fortran they only worked if a stale value was left from a previous
  qualifier, because OBEY_LINE tests 'qval(i).ne.' '').
* New verb:  faithful [on|off]   (default off = corrected physics).
* 'placzek diatomic <omega>' reads omega from the 2nd parameter (Fortran read
  rparam(1), which is the word 'diatomic' -> 0).  faithful on restores it.

Using it from Python (e.g. in a notebook, for comparison with MC):

    from pycorrect import Correct
    c = Correct()
    c.run_file("d4run.com")              # or c.command("sample ...") line by line
    res = c.results[0]                   # dict of arrays for spectrum 1
    res["q"], res["dcs"], res["assch"], res["delta_sa"] ...
"""

from __future__ import annotations

import math
import os
import sys

from .fileio import get_input, pc_inquire
from .ndplib import _find_data_file
from .parse_line import parse_line
from .state import CorrectState, Species, Spectrum


def read_atom_data(symbol: str):
    """Look up an element in xsect.dat.

    Fortran: SUBROUTINE READ_ATOM_DATA(ATOM_SYMBOL,WEIGHT,SCATCS,ABSCS)
    File columns: Z, atomic weight, symbol, b (fm), sigma_scatt (barn),
                  sigma_abs at 2200 m/s (barn).
    Returns (weight, scatt_cs, abs_cs).  Raises KeyError if not found
    (the Fortran STOPs).
    """
    path = _find_data_file("xsect.dat")
    with open(path) as fh:
        for line in fh:
            v = line.split()
            if len(v) >= 6 and v[2].lower() == symbol.lower():
                return float(v[1]), float(v[4]), float(v[5])
    raise KeyError(f'Symbol "{symbol}" not found in x-section table')


class Correct:
    """The CORRECT interpreter: holds the state and obeys command lines."""

    def __init__(self, state: CorrectState | None = None, verbose=True, ask=True):
        self.st = state or CorrectState()
        self.verbose = verbose
        self.ask = ask              # ask "Continue (Y/N)?" after a listing
        self.results = None
        self.finished = False

    # ------------------------------------------------------------------
    def command(self, line: str):
        """Parse and obey one command line (PARSE_LINE + OBEY_LINE)."""
        pl = parse_line(line)
        if pl.verb:
            self.obey_line(pl)

    def run_file(self, fname: str):
        """Execute a command file; '@file' lines are nested command files.
        Fortran: main loop with 'comfile' and unit iread."""
        fname = pc_inquire(fname)
        base = os.path.dirname(os.path.abspath(fname))
        with open(fname) as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if self.finished:
                    break
                if line.strip().startswith("@"):
                    self.run_file(os.path.join(base, line.strip()[1:]))
                    continue
                if self.verbose and line.strip():
                    print(" CORRECT> " + line.rstrip())
                self.command(line)

    def interactive(self):
        """Prompt loop.  Fortran: GET_INPUT + PARSE_LINE + OBEY_LINE."""
        while not self.finished:
            line = get_input("CORRECT> ")
            if line.strip().startswith("@"):
                try:
                    self.run_file(line.strip()[1:])
                except OSError as exc:
                    print(" ERROR opening file", exc)
            else:
                self.command(line)

    # ------------------------------------------------------------------
    def obey_line(self, pl):
        """Act on one parsed command.  Fortran: SUBROUTINE OBEY_LINE."""
        st = self.st
        verb, qual, qval, rqval = pl.verb, pl.qual, pl.qval, pl.rqval
        param, rparam = pl.param, pl.rparam
        npar = len(param)
        p0 = param[0].lower() if param else ""

        def rp(i, default=None):          # rparam(i), 1-based like Fortran
            return rparam[i - 1] if npar >= i else default

        if verb == "background":
            keys = {"/furnace": "bkg_furnace", "/container": "bkg_container",
                    "/sample": "bkg_sample", "/vanadium": "bkg_vanadium"}
            if not qual:
                for k in keys.values():
                    st.files[k] = param[0]
            for q in qual:
                st.files[keys[q]] = param[0]

        elif verb == "beam":
            if npar >= 1: st.beam_height = rp(1)
            if npar >= 2: st.beam_width = rp(2)

        elif verb == "cadmium":
            st.files["cadmium"] = param[0]

        elif verb == "collimator":
            c = st.collimator
            c.enabled = True
            if npar >= 1: c.r_offset = rp(1)
            if npar >= 2: c.phi_offset = math.radians(rp(2))
            if npar >= 3: c.blade_space = rp(3)
            if npar >= 4: c.blade_thick = rp(4)
            if npar >= 5: c.r1 = rp(5)
            if npar >= 6: c.r2 = rp(6)
            if npar >= 7: c.rd = rp(7)
            c.update()

        elif verb == "calibration":
            if npar >= 1: st.calib_ur = rp(1)
            if npar >= 2: st.calib_vr = rp(2)
            if npar >= 3: st.calib_wr = rp(3)

        elif verb == "component":
            # component <fraction> <symbol>            (data from xsect.dat)
            # component <fraction> <weight> <sigma_s> <sigma_a>
            fraction = rp(1, 0.0)
            weight = rp(2, 0.0)
            scatcs = rp(3, 0.0)
            abscs = rp(4, 0.0)
            if weight == 0.0:
                sym = param[1][:2].lower()
                weight, scatcs, abscs = read_atom_data(sym)
                symbol = sym[0].upper() + sym[1:]
            else:
                symbol = "-"
            st.species.append(Species(symbol, fraction, weight, scatcs, abscs))

        elif verb == "container":
            st.files["container"] = param[0]
            if npar >= 2: st.ca_r2 = rp(2)
            for q, v, r in zip(qual, qval, rqval):
                if not v:
                    continue
                if q == "/density": st.ca_rho = r
                elif q == "/absorption": st.ca_abs_cs = r
                elif q == "/scattering": st.ca_scat_cs = r
                elif q == "/multiplier": st.ca_multiplier = r
                elif q == "/type":
                    t = v[0].lower()
                    if t == "v":   st.ca_abs_cs, st.ca_scat_cs, st.ca_rho = 5.08, 4.95, 0.0722
                    elif t == "s": st.ca_abs_cs, st.ca_scat_cs, st.ca_rho = 0.0571, 3.547, 0.0622
                    elif t == "t": st.ca_abs_cs, st.ca_scat_cs, st.ca_rho = 4.14, 5.01, 0.052

        elif verb == "execute":
            listing, cont = True, True
            for q, v in zip(qual, qval):
                if q == "/output":
                    cont = False
                    if v:
                        with open(v, "x") as fh:     # Fortran status='new'
                            old = sys.stdout
                            sys.stdout = fh
                            try:
                                self._execute(listing=True, cont=False)
                            finally:
                                sys.stdout = old
                        return
                elif q == "/nolist":
                    listing = False
            self._execute(listing, cont)

        elif verb == "faithful":                       # Python addition
            st.faithful = p0 not in ("off", "no", "0", "false", "n")
            if self.verbose:
                print(" faithful =", st.faithful)

        elif verb == "furnace":
            st.files["furnace"] = param[0]
            if npar >= 2: st.fu_r1 = rp(2)
            if npar >= 3: st.fu_r2 = rp(3)
            for q, v, r in zip(qual, qval, rqval):
                if not v:
                    continue
                if q == "/density": st.fu_rho = r
                elif q == "/absorption": st.fu_abs_cs = r
                elif q == "/scattering": st.fu_scat_cs = r

        elif verb == "geometry":
            if p0.startswith("c"): st.geometry = "CYLINDRICAL"
            if p0.startswith("p"): st.geometry = "PLATE"
            if npar == 2: st.sample_angle = rp(2)
            if npar == 3:
                st.sample_angle, st.van_angle = rp(2), rp(3)
            else:
                st.van_angle = st.sample_angle

        elif verb == "instrument":
            for pre, name in (("d2", "D20"), ("d4", "D4"), ("l", "LAD"),
                              ("sa", "SANDALS"), ("se", "SEPD"), ("sl", "SLAD")):
                if p0.startswith(pre):
                    st.instrument = name
            st.inst_type = "TOF" if st.instrument in ("LAD", "SANDALS", "SEPD") else "2-AXIS"

        elif verb == "list":
            self.correct_list()
            if self.ask and get_input("Continue(Y/N)? ").strip().lower().startswith("n"):
                self.finished = True

        elif verb == "low_angle":
            for i in range(4):
                if npar >= i + 1:
                    st.lowang[i] = rparam[i]
            for q in qual:
                if q == "/furnace": st.low_fu = True
                if q == "/container": st.low_ca = True
                if q == "/vanadium": st.low_va = True
                if q == "/background": st.low_ba = True

        elif verb == "monitor":
            if npar >= 1: st.monitor_spectrum = int(rp(1))

        elif verb == "output":
            st.files["output"] = param[0]

        elif verb == "placzek":
            if p0.startswith("d"): st.placzek_correction = "DIATOMIC"
            if p0.startswith("n"): st.placzek_correction = "NONE"
            if p0.startswith("s"): st.placzek_correction = "SERIES_EXPANSION"
            if st.faithful:
                if npar >= 1: st.omegav = rp(1)        # Fortran: always 0
            elif npar >= 2:
                st.omegav = rp(2)

        elif verb == "quit":
            self.finished = True

        elif verb == "range":
            if npar >= 1: st.wav_min = rp(1)
            if npar >= 2: st.wav_max = rp(2)

        elif verb == "sample":
            st.files["sample"] = param[0]
            if npar >= 2: st.ca_r1 = rp(2)
            for q, v, r in zip(qual, qval, rqval):
                if q == "/cs_file": st.files["cs_file"] = v
                if not v:
                    continue
                if q == "/density": st.sa_rho = r
                elif q == "/temperature": st.sa_temp = r
                elif q == "/fullness": st.sa_fullness = r

        elif verb == "spectrum":
            # spectrum <block> [angle path1 path2] [/detector=h|s]
            # spectrum /to=N        -> blocks 1..N
            # spectrum /default     -> 48 blocks (SLAD only)
            if qual and qual[0] == "/default" and st.instrument == "SLAD":
                st.spectra = [Spectrum(i) for i in range(1, 49)]
            elif qual and qual[0] == "/to":
                st.spectra = [Spectrum(i) for i in range(1, int(rqval[0]) + 1)]
            else:
                s = Spectrum(int(round(rp(1, 1))))
                if npar >= 2: s.angle = rp(2)
                if npar >= 3: s.path1 = rp(3)
                if npar >= 4: s.path2 = rp(4)
                if qual and qval[0]:
                    if qval[0][0].lower() == "h": s.det_type = "HE_GAS"
                    if qval[0][0].lower() == "s": s.det_type = "SCINTILLATION"
                st.spectra.append(s)

        elif verb == "title":
            st.new_title = param[0] if param else ""

        elif verb == "vanadium":
            st.files["vanadium"] = param[0]
            if npar >= 2: st.va_r = rp(2)
            for q, v, r in zip(qual, qval, rqval):
                if q == "/furnace": st.va_furnace = True
                elif q == "/container": st.va_container = True
                if not v:
                    continue
                if q == "/density": st.va_rho = r
                elif q == "/absorption": st.va_abs_cs = r
                elif q == "/scattering": st.va_scat_cs = r
                elif q == "/smoothing": st.vanadium_smoothing = int(round(r))
                elif q == "/multiplier": st.va_multiplier = r

        elif verb == "wavelength":
            if npar >= 1: st.wavelength = rp(1)

        elif verb == "write":
            for q, v in zip(qual, qval):
                if q == "/output" and v:
                    st.write_second_file = True
                    st.files["second"] = v
            for p in param:
                c = p[:1].lower()
                if c == "a": st.write_absorption = True
                if c == "c": st.write_calibration = True
                if c == "l": st.write_low_angle = True
                if c == "m": st.write_ms = True
                if c == "p": st.write_placzek = True

        elif verb == "xin":
            if p0.startswith("t"): st.xtype = "TOF"
            if p0.startswith("q"): st.xtype = "Q"
            if p0.startswith("w"): st.xtype = "WAVELENGTH"

        elif verb == "xout":
            if p0.startswith("a"): st.xouttype = "ANGLE"
            if p0.startswith("q"): st.xouttype = "Q"

    # ------------------------------------------------------------------
    def _execute(self, listing=True, cont=True):
        """'execute' verb: list, ask, then run the right correction routine."""
        if listing:
            self.correct_list()
            if cont and self.ask:
                if get_input("Continue(Y/N)? ").strip().lower().startswith("n"):
                    self.finished = True
                    return
        if self.st.inst_type == "2-AXIS":
            from .correct_2axis import correct_2axis
            self.results = correct_2axis(self.st, self.verbose)
        else:
            from .correct_tof import correct_tof
            self.results = correct_tof(self.st, self.verbose)

    # ------------------------------------------------------------------
    def correct_list(self):
        """Print the current set-up.  Fortran: SUBROUTINE CORRECT_LIST."""
        st = self.st
        fn = lambda k: os.path.abspath(st.files[k])
        print(f"\n CORRECT (Python)\n ========\n\n Correcting data from {st.inst_type} "
              f"diffractometer {st.instrument}\n")
        print(f" faithful (reproduce CORRECT 2.24 bugs): {st.faithful}")
        if st.has("sample"):
            print(f"\n Sample data is read from file {fn('sample')}")
            if st.has("bkg_sample"): print(f" with background from file {fn('bkg_sample')}")
        else:
            print("\n Sample file has not been defined")
        if st.has("container"):
            print(f"\n Container data is read from file {fn('container')}")
            if st.has("bkg_container"): print(f" with background from file {fn('bkg_container')}")
        if st.ca_multiplier != 1: print(f" Container data will be multiplied by {st.ca_multiplier:6.3f}")
        if st.has("furnace"):
            print(f"\n Furnace data is read from file {fn('furnace')}")
            if st.has("bkg_furnace"): print(f" with background from file {fn('bkg_furnace')}")
        if st.has("cadmium"):
            print(f"\n Cadmium data is read from file {fn('cadmium')}")
        if st.has("vanadium"):
            print(f"\n Vanadium data is read from file {fn('vanadium')}")
            if st.has("bkg_vanadium"): print(f" with background from file {fn('bkg_vanadium')}")
        else:
            print("\n Vanadium file has not been defined")
            if st.has("container"): print(" Calibration will be done with the container")
        if st.va_furnace: print(" Vanadium will be corrected for furnace scattering")
        if st.va_container: print(" Vanadium will be corrected for container scattering")
        if st.inst_type == "2-AXIS":
            vs = st.vanadium_smoothing
            if vs == 0: print(" No vanadium smoothing")
            elif vs > 0: print(f" Vanadium smoothing with polynomial order {vs - 1}")
            else: print(f" Vanadium smoothing with Savitzky-Golay filter with window {-vs * 2 + 1}")
        if st.va_multiplier != 1: print(f" Vanadium data will be multiplied by {st.va_multiplier:6.3f}")
        print(f" Calibration parameters used u={st.calib_ur:6.3f} v={st.calib_vr:6.3f} "
              f"w={st.calib_wr:6.3f} Degrees**2")
        print(f"\n Output will be to file {fn('output')}\n" if st.has("output")
              else "\n Output file has not been defined\n")
        if st.has("furnace"):
            print(f" Furnace abs. c/s (at 2200m/s)   : {st.fu_abs_cs:6.3f} barns\n"
                  f" Furnace scattering c/s          : {st.fu_scat_cs:6.3f} barns\n"
                  f" Furnace number density          : {st.fu_rho:6.4f} per cu. Angstrom")
        if st.has("container"):
            print(f" Can absorption c/s (at 2200m/s) : {st.ca_abs_cs:6.3f} barns\n"
                  f" Can scattering c/s              : {st.ca_scat_cs:6.3f} barns\n"
                  f" Can number density              : {st.ca_rho:6.4f} per cu. Angstrom")
        if st.has("vanadium"):
            print(f" V absorption c/s (at 2200m/s)   : {st.va_abs_cs:6.3f} barns\n"
                  f" Vanadium scattering c/s         : {st.va_scat_cs:6.3f} barns\n"
                  f" Vanadium number density         : {st.va_rho:6.4f} per cu. Angstrom")
        if st.has("sample"):
            print(f" Sample number density           : {st.sa_rho:6.4f} per cu. Angstrom\n"
                  f" Sample temperature              : {st.sa_temp:6.1f} K\n"
                  f" Sample new title                : {st.new_title}\n")
        print(f"\n Sample geometry is {st.geometry}")
        if st.geometry == "CYLINDRICAL":
            if st.has("sample"): print(f" Sample radius                   : {st.ca_r1:6.4f} cm")
            if st.has("container"): print(f" Can outer radius                : {st.ca_r2:6.4f} cm")
            if st.has("furnace"): print(f" Furnace inner radius            : {st.fu_r1:6.4f} cm\n"
                                        f" Furnace outer radius            : {st.fu_r2:6.4f} cm")
            if st.has("vanadium"): print(f" Vanadium radius                 : {st.va_r:6.4f} cm")
        else:
            if st.has("sample"): print(f" Sample thickness                : {st.ca_r1:6.4f} cm\n"
                                       f" Sample angle                    : {st.sample_angle:8.4f} degrees")
            if st.has("container"): print(f" Can thickness                   : {st.ca_r2:6.4f} cm")
            if st.has("vanadium"): print(f" Vanadium thickness              : {st.va_r:6.4f} cm\n"
                                         f" Vanadium angle                  : {st.van_angle:8.4f} degrees")
        c = st.collimator
        if c.enabled:
            print(f"\n Absorption corrections for a radial oscillating collimator\n"
                  f" Inner radius {c.r1:8.4f} cm, outer radius {c.r2:8.4f} cm, detector distance "
                  f"{c.rd:8.4f} cm, blade spacing {c.blade_space:8.4f} cm, thickness {c.blade_thick:8.4f} cm")
        if st.species:
            print(f"\n The sample consists of {len(st.species):2d} species:\n\n"
                  "  Symbol      Rel. Conc.    Atomic wt.      Cross-section  Absorption c-s")
            for s in st.species:
                print(f"    {s.symbol:2s}         {s.fraction:7.5f}        {s.weight:8.4f}       "
                      f"{s.scatt_cs:8.4f}       {s.abs_cs:8.4f}")
        else:
            print(" The sample composition has not been defined")
        print(f"\n Beam height                     : {st.beam_height:6.4f} cm\n"
              f" Beam width                      : {st.beam_width:6.4f} cm")
        print(f"\n Placzek correction will be {st.placzek_correction}\n")
        if st.inst_type == "TOF":
            print(f" Input data is a function of {st.xtype}\n\n"
                  f" Minimum wavelength used is {st.wav_min:4.1f}A\n"
                  f" Maximum wavelength used is {st.wav_max:4.1f}A\n")
            print(f" Data will be normalised using spectrum {st.monitor_spectrum} as monitor"
                  if st.monitor_spectrum else " There is no incident monitor spectrum")
            for s in st.spectra:
                print(f"     Spectrum {s.block:2d} at 2theta = {s.angle:7.2f} deg, detector "
                      f"{s.det_type}, L1 = {s.path1:6.3f} m, L2 = {s.path2:6.3f} m")
        else:
            print(f" Incident wavelength             : {st.wavelength:6.4f} Angstroms\n")
            if st.has("cadmium") or st.low_fu or st.low_ba or st.low_ca or st.low_va:
                la = st.lowang
                print(f" Low-angle correction between {la[0]:5.2f} and {la[1]:5.2f} deg, "
                      f"line fitted from {la[2]:5.2f} to {la[3]:5.2f} deg")
            if st.spectra:
                print(f"\n {len(st.spectra):2d} spectra will be corrected: "
                      + " ".join(str(s.block) for s in st.spectra))
            else:
                print("\n No spectra will be corrected")


def main(argv=None):
    """Command-line entry point.  Fortran: PROGRAM CORRECT."""
    argv = sys.argv[1:] if argv is None else argv
    print(" CORRECT version 2.24 -- Python translation")
    c = Correct()
    if argv:
        arg = argv[0][1:] if argv[0].startswith("@") else argv[0]
        c.run_file(arg)
    else:
        c.interactive()
    return c


if __name__ == "__main__":
    main()
