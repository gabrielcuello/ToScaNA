"""
ToScaNA correction from the command line, with THIS ToScaNA.

Same as ``python -m toscana.workflows.correct_cli`` (use that form when ToScaNA
is the ``toscana`` of your Python path). Examples, from this folder:

    python correct.py Ni_powder.par                               # MCAMS, .adat, outputs Q and angle
    python correct.py Ni_powder.par --input qdat --scale q --outdir results/cli
    python correct.py Ni_powder.par --method pycorrect --outdir results/cli_pycorrect
    python correct.py --help
"""

import sys

import toscana_path  # noqa: F401
from toscana.workflows.correct_cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
