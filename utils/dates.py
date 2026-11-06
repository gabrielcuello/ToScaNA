"""
Date string in the format used in ToScaNA output headers.

Public names
------------
getDate
    Current local date and time as ``" Dayname DD/MM/YYYY HH:MM:SS"``.
"""

from __future__ import annotations

import calendar
from datetime import datetime


def getDate() -> str:
    """
    Produce a string with the current date in the legacy format:

        " Dayname DD/MM/YYYY HH:MM:SS"

    Returns
    -------
    str
        For example ``" Friday 06/11/2026 21:25:57"``: a leading space, the
        weekday name (``calendar.day_name``, hence in the current locale,
        English by default), a space, then ``%d/%m/%Y %H:%M:%S`` of the local
        time (``datetime.now()``).
    """
    now = datetime.now()
    day = " " + calendar.day_name[now.weekday()] + " "
    current_datetime = day + now.strftime("%d/%m/%Y %H:%M:%S")
    return current_datetime
