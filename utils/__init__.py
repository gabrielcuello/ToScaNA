"""
Small utilities.

Public names
------------
execute_command_and_write_to_file
    Run an external command, writing its standard output to a file; errors
    are printed, not raised.
getDate
    Current date and time as ``" Dayname DD/MM/YYYY HH:MM:SS"``.
extractDict
    Sub-dictionary ``{k: d[k][key]}`` of a dictionary of dictionaries.
extractAttr
    Dictionary ``{k: getattr(obj, attr)}`` of a dictionary of objects.
stop
    Print a message and ask whether to continue (interactive sessions only).
"""

from .commands import execute_command_and_write_to_file
from .dates import getDate
from .dicts import extractAttr, extractDict
from .interactive import stop

__all__ = [
    "execute_command_and_write_to_file",
    "extractAttr",
    "extractDict",
    "getDate",
    "stop",
]
