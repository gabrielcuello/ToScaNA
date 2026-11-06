"""
Location of the d4creg log file.

Position in the processing chain
--------------------------------
Infrastructure used by every step of the D4 regrouping (``dataio.acquisition``,
``dataio.corrections``, ``workflows.d4creg``). Those functions do not print the
detailed counting rates and correction factors; they append them to a single log
file whose location is fixed by this module.

Functions
---------
d4creg_log_path
    Return ``../logfiles/d4creg.log``, creating the folder and the file if needed.

Conventions
-----------
The path is RELATIVE TO THE CURRENT WORKING DIRECTORY (legacy layout: the
notebooks run in a sub-folder of the experiment directory, and the logs go to the
sibling folder ``logfiles``). ``dataio.parameters.readParam`` truncates this file
at start; ``workflows.d4creg.d4creg`` renames it at the end of each reduction.
"""

from __future__ import annotations

from pathlib import Path


def d4creg_log_path() -> Path:
    """
    Return the legacy log-file path and ensure that the folder and the file exist.

    Legacy uses a fixed relative path: `../logfiles/d4creg.log`.

    Returns
    -------
    pathlib.Path
        ``Path("../logfiles/d4creg.log")``, relative to the current working
        directory (not made absolute). Callers open it in append mode.

    Notes
    -----
    Side effects (file system):

    - creates the directory ``../logfiles`` (and parents) if it does not exist;
    - if ``../logfiles/d4creg.log`` does not exist, creates it with the content
      ``"# d4creg.log\\n\\n"`` (UTF-8). An existing file is left untouched.

    The function is called each time a message is logged, so if
    ``workflows.d4creg.d4creg`` has renamed the log file at the end of a
    reduction, the next message recreates a fresh ``d4creg.log``.

    See Also
    --------
    toscana.dataio.parameters.readParam : truncates the log file at start.
    toscana.workflows.d4creg.d4creg : renames the log file at the end.
    """
    log_path = Path("../logfiles/d4creg.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# d4creg.log\n\n", encoding="utf-8")
    return log_path
