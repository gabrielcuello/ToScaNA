"""MCAMS self-checks (tests/mcams_checks.py), run as part of the suite.

`mcams_checks.py` is the original MCAMS test script: it runs all checks at
import time, prints PASS/FAIL for each and exits with status 1 if any fails.
It is executed here in a separate process, unchanged, and must end with
"ALL PASSED".
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest

import helpers  # noqa: F401

HERE = pathlib.Path(__file__).resolve().parent


class TestMCAMS(unittest.TestCase):
    def test_all_mcams_checks_pass(self) -> None:
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
        with helpers.in_tempdir() as tmp:
            out = subprocess.run(
                [sys.executable, str(HERE / "mcams_checks.py")],
                cwd=tmp, env=env, capture_output=True, text=True,
            )
        failed = [line for line in out.stdout.splitlines() if line.startswith("FAIL")]
        self.assertEqual(out.returncode, 0, "\n".join(failed) or out.stderr[-2000:])
        self.assertIn("ALL PASSED", out.stdout)


if __name__ == "__main__":
    unittest.main()
