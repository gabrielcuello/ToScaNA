"""Allows ``python -m toscana.d4numor COMMAND ...`` (runs numorset.main)."""

import sys

from .numorset import main

sys.exit(main())
