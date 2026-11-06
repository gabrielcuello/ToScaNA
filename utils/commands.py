"""
Run an external program and capture its output in a file.

Public names
------------
execute_command_and_write_to_file
    Run a command (no shell), write its stdout to a file, print the return
    code or the error.
"""

from __future__ import annotations

import subprocess
from typing import Sequence


def execute_command_and_write_to_file(command: Sequence[str], output_file: str) -> None:
    """
    Run a command and write stdout to a file (legacy behavior).

    Runs ``subprocess.run(command, check=True, text=True, stdout=<file>,
    stderr=PIPE)``: no shell, standard output written to `output_file`,
    standard error captured.

    Parameters
    ----------
    command : sequence of str
        Program and its arguments, e.g. ``["ls", "-l"]``.
    output_file : str
        Path of the file receiving the standard output; created or
        OVERWRITTEN (it is created even if the command fails to start).

    Notes
    -----
    Never raises for a failing command; prints instead:

    - on success: "Return Code: 0";
    - on a non-zero exit status (``CalledProcessError``): the error, "Command
      failed with return code <n>" and "Error output: <stderr>";
    - on ``OSError`` (missing executable, unwritable `output_file`...):
      "Error: <exception>".

    Divergence vs legacy:
    - also catches `OSError` (e.g. `FileNotFoundError`) so missing executables
      do not crash tests/CI.
    """
    try:
        with open(output_file, "w") as file:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=file,
                stderr=subprocess.PIPE,
            )
            print("Return Code:", result.returncode)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("Command failed with return code", e.returncode)
        print("Error output:", e.stderr)
    except OSError as e:
        print("Error:", e)
