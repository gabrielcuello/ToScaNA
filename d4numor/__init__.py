"""
d4numor: survey of D4 numor sets without a graphical interface.

Command-line and Jupyter companion of D4NumorGUI: list the variables of a set
of numors, weighted averages, plots, detector movement, preset calculator,
CSV/pandas export. The data are read with the bundled NeXus reader
``toscana.dataio.nexus``. Full user manual: ``docs/d4numor/MANUAL.md``;
examples: ``docs/d4numor/D4numor_examples.ipynb``.

Usage
-----
In Python / Jupyter::

    from toscana import d4numor as d4
    ns = d4.NumorSet.from_directory("../rawdata", "480652-480700")
    ns.summary()

From the command line (toscana importable)::

    python -m toscana.d4numor -d ../rawdata -n 480652-480700 summary

pandas is needed only for ``NumorSet.to_dataframe``; matplotlib only for
plots.
"""

from .numorset import (
    D4Error,
    NumorSet,
    build_file_list,
    build_parser,
    category,
    format_preset,
    format_variable_listing,
    get_reader,
    label,
    main,
    numor_from_path,
    numor_info,
    parse_numor_list,
)

__all__ = [
    "D4Error",
    "NumorSet",
    "build_file_list",
    "build_parser",
    "category",
    "format_preset",
    "format_variable_listing",
    "get_reader",
    "label",
    "main",
    "numor_from_path",
    "numor_info",
    "parse_numor_list",
]
