"""
Interactive pause for notebooks and scripts.

Public names
------------
stop
    Print a message and ask whether to continue.
"""

from __future__ import annotations

import sys


def stop(message: str) -> None:
    """
    Legacy interactive stop helper.

    Prints an empty line, a line of 30 ``"<>"``, and `message`; then asks
    "Do you want to continue? (y/n) :". Answering exactly ``"n"`` stops the
    program; any other answer (including empty) continues.

    Parameters
    ----------
    message : str
        Text to print.

    Raises
    ------
    RuntimeError
        "stop() called in non-interactive mode" if ``sys.stdin`` has no
        ``isatty`` or is not a terminal (the message is printed first).
        Note that in Jupyter ``sys.stdin`` is usually not a TTY, so this is
        raised there too.
    SystemExit
        "You stopped the program. Bye!" if the answer is ``"n"``.

    Divergence vs legacy:
    - In non-interactive environments (`sys.stdin.isatty() == False`), raises
      `RuntimeError` instead of prompting, to avoid CI hangs.
    """
    print()
    print(30 * "<>")
    print(message)

    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        raise RuntimeError("stop() called in non-interactive mode")

    answer = input("Do you want to continue? (y/n) :")
    if answer == "n":
        raise SystemExit("You stopped the program. Bye!")
    return None
