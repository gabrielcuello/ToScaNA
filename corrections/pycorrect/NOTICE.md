# Notice: origin and licensing status of `pycorrect`

## Origin

The code in this directory (`corrections/pycorrect/`) is a Python translation
of the Fortran program **CORRECT, version 2.24 (January 2000)**, written by
**M. A. Howe, R. L. McGreevy and P. Zetterström**.

The algorithms, the structure of the calculation and many routine names follow
the original Fortran sources. The correspondence between Fortran routines and
Python functions is documented in `docs/pycorrect/README.md` and in the
`pycorrect` user manual. The data files `xsect.dat` and `doublescattering.dat`
come from the CORRECT distribution.

The routines of the Harwell Subroutine Library used by CORRECT (`harwell.f`)
were **not** translated: they are replaced by NumPy/SciPy equivalents, and no
Harwell code is included here.

All credit for the original program, its physics and its algorithms belongs to
its authors. Errors introduced in the translation are the responsibility of the
ToScaNA developers. Differences from the original (double precision, corrected
errors) are documented in the `pycorrect` manual and in `docs/validation/`.

## Licensing status

The terms under which CORRECT was originally distributed have not yet been
established, and the authors' permission for this translation has not yet been
obtained. Until this is clarified:

- the contents of this directory are **not covered by the licence of the rest
  of ToScaNA**;
- copyright in the original program remains with its authors;
- **no licence is granted** for these files. They are made visible for
  scientific transparency and reproducibility of the results obtained with
  ToScaNA.

This notice will be replaced once the situation is clarified.

If you are one of the authors of CORRECT, or know the terms under which it was
distributed, please open an issue at <https://github.com/gabrielcuello/ToScaNA/issues>.

## Citation

If you use `pycorrect`, please cite the original CORRECT program as well as
ToScaNA.
