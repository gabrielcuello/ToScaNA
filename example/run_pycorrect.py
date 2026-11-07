"""
pycorrect (CORRECT 2.24) with its own command language, with THIS ToScaNA.

Same as ``python -m toscana.corrections.pycorrect``. From the data folder
(file names in .com files are relative to it):

    cd data
    python ../run_pycorrect.py Ni_powder.com      # writes Ni_powder_pycorrect.dat
    python ../run_pycorrect.py                    # interactive prompt CORRECT>
"""

import sys

import toscana_path  # noqa: F401
from toscana.corrections.pycorrect.correct import main

if __name__ == "__main__":
    main(sys.argv[1:])
