"""
Run the ToScaNA test suite with the standard library only.

Usage (from the project root or anywhere else):

    python tests/run_tests.py            # all tests
    python tests/run_tests.py -v         # verbose
    python tests/run_tests.py test_fourier   # one module
"""

from __future__ import annotations

import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _pkg  # noqa: E402,F401  (registers `toscana`)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbosity = 2 if "-v" in sys.argv else 1
    loader = unittest.TestLoader()
    if args:
        suite = loader.loadTestsFromNames(args)
    else:
        suite = loader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(HERE))
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
