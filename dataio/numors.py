"""
Expansion of numor lists.

Position in the processing chain
--------------------------------
First step of ``acquisition.getNumors``: the list ``runInfo["numorLst"]`` built by
``parameters.readParam`` from the ``<num>`` and ``<add>`` lines is expanded into the
explicit list of numors (raw-data file names without path or extension) to read.

Functions
---------
getNumorFiles
    Expand entries ``"123456"`` and ``"123456-123460"`` into a list of 6-digit
    numors.

Conventions
-----------
A numor is the 6-character run number used as file name of the raw data (e.g.
``"387229"``, completed with ``runInfo["ext"][4]`` = ``".nxs"`` for NeXus files).
"""

from __future__ import annotations


def getNumorFiles(numorLst: list[str]) -> list[str]:  # noqa: N802 (legacy API)
    """
    Expand a short syntax list of numors into a detailed list.

    Each entry of ``numorLst`` is handled according to its length only:

    - 6 characters (``"387229"``): kept as it is (the content is NOT checked to be
      numeric);
    - 13 characters (``"387229-387288"``): a range. Characters 0-5 give the first
      numor and characters 7-12 the last one (both converted with ``int``; the
      separator, character 6, is not checked). Every numor from first to last,
      both included, is appended, zero-padded to 6 digits.
    - any other length: error.

    The order of the entries is kept and duplicates are not removed.

    Parameters
    ----------
    numorLst : list of str
        Short-syntax numor entries, e.g. ``["387229-387231", "387240"]``
        (``runInfo["numorLst"]``).

    Returns
    -------
    list of str
        Explicit numors, each a 6-character string, e.g.
        ``["387229", "387230", "387231", "387240"]``.

    Raises
    ------
    ValueError
        If an entry has a length other than 6 or 13; if a 13-character entry has
        a first or last part that ``int`` cannot convert; if the last numor of a
        range is smaller than the first one.

    Notes
    -----
    This matches the legacy happy-path behavior and tightens error handling:
    invalid entries raise `ValueError` instead of printing/logging and continuing.
    No output is printed and no file is touched.

    Examples
    --------
    >>> getNumorFiles(["387229-387231", "387240"])
    ['387229', '387230', '387231', '387240']
    """
    numor_files: list[str] = []
    len_numor = 6

    for entry in numorLst:
        if len(entry) == len_numor:
            numor_files.append(entry)
            continue

        if len(entry) == 2 * len_numor + 1:
            first_num = int(entry[0:len_numor])
            last_num = int(entry[len_numor + 1 : 2 * len_numor + 1])
            if first_num > last_num:
                raise ValueError(
                    f"Invalid numor range {entry}: last numor must be >= first numor."
                )
            for num in range(first_num, last_num + 1):
                numor_files.append(str(num).zfill(len_numor))
            continue

        raise ValueError(
            "Invalid numor entry (getNumorFiles): expected 6-digit numor or '111111-222222' range."
        )

    return numor_files
