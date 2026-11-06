"""
Command files for the external program CORRECT.

Position in the processing chain
--------------------------------
After the regrouping (``.adat`` files of sample, container, environment/background
and vanadium written by ``d4creg_outputs.saveDiffAngle``), the corrections for
attenuation, multiple scattering and vanadium normalisation are made by the
external ILL program CORRECT, driven by a ``.com`` command file. This module
writes such files; it does not run CORRECT (``generate_com`` runs only a
placeholder command, see its Notes).

Functions
---------
generate_com
    Fill the ``{py...}`` placeholders of a CORRECT template and write
    ``<name>.com``.
saveCORRECT
    Write ``<sampleTitle>.com`` from a legacy-style experiment object.

Conventions
-----------
- Files are written in the current working directory (platform default encoding),
  overwriting existing files.
- CORRECT expects lengths in cm: ``saveCORRECT`` divides the radii and beam
  dimensions (in mm in the experiment object) by 10, and the diameters by 20.
"""

from __future__ import annotations

from typing import Any

from toscana.utils.commands import execute_command_and_write_to_file


def generate_com(
    CORRECT_template: str = "correct_template.com",
    name: str = "prueba",
    rsample: float = 0.5,
    density: float = 0.11,
    MTname: str = "MTBelljar_6mm",
    MTCanname: str = "MTCan_6mm",
    rcan: float = 0,
    Vananame: str = "Vanadium_7mm",
    rvana: float = 0.7,
    beam_width: float = 0.6,
    beam_height: float = 1.6,
) -> None:
    """
    Generate an input file for CORRECT starting from a template (legacy behavior).

    The template is read as text and the following placeholders are replaced
    (``str()`` of the value for numbers):

    ======================  ==============
    placeholder             argument
    ======================  ==============
    ``{pyfname}``           ``name``
    ``{pyradius_cm}``       ``rsample``
    ``{pydensity_at_AA3}``  ``density``
    ``{pymtbelljarname}``   ``MTname``
    ``{pycanname}``         ``MTCanname``
    ``{pyradius_can}``      ``rcan``
    ``{pyradius_vana}``     ``rvana``
    ``{pyvanadiumname}``    ``Vananame``
    ``{pybeam_w_cm}``       ``beam_width``
    ``{pybeam_h_cm}``       ``beam_height``
    ======================  ==============

    Parameters
    ----------
    CORRECT_template : str, optional
        Path of the template file. Default ``"correct_template.com"`` in the
        current working directory.
    name : str, optional
        Sample (file) name; also the base name of the output file. Default
        ``"prueba"``.
    rsample : float, optional
        Sample radius in cm. Default 0.5.
    density : float, optional
        Sample atomic number density in atoms/Å³. Default 0.11.
    MTname : str, optional
        Name of the empty-instrument (bell jar) measurement. Default
        ``"MTBelljar_6mm"``.
    MTCanname : str, optional
        Name of the empty-can measurement. Default ``"MTCan_6mm"``.
    rcan : float, optional
        Can radius (template placeholder ``{pyradius_can}``; units as expected
        by the template, presumably cm). Default 0.
    Vananame : str, optional
        Name of the vanadium measurement. Default ``"Vanadium_7mm"``.
    rvana : float, optional
        Vanadium radius (presumably cm). Default 0.7.
    beam_width, beam_height : float, optional
        Beam width and height in cm. Defaults 0.6 and 1.6.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the template does not exist.

    Notes
    -----
    Side effects:

    - prints the filled text;
    - writes ``<name>.com`` in the current working directory;
    - runs the PLACEHOLDER command ``["executable_name", "arg1"]`` through
      ``utils.commands.execute_command_and_write_to_file``, which creates (or
      truncates) ``output.txt`` in the current working directory and prints an
      error message because the executable does not exist (open item: this is
      not a real call to CORRECT).

    The legacy implementation also attempted to run CORRECT via a placeholder
    command; we preserve the call pattern but keep it non-crashing when the
    executable is missing (handled in `execute_command_and_write_to_file`).
    """
    with open(CORRECT_template, "r") as f:
        text = "".join(f.readlines())

    text = text.replace("{pyfname}", name)
    text = text.replace("{pyradius_cm}", str(rsample))
    text = text.replace("{pydensity_at_AA3}", str(density))
    text = text.replace("{pymtbelljarname}", MTname)
    text = text.replace("{pycanname}", MTCanname)
    text = text.replace("{pyradius_can}", str(rcan))
    text = text.replace("{pyradius_vana}", str(rvana))
    text = text.replace("{pyvanadiumname}", Vananame)
    text = text.replace("{pybeam_w_cm}", str(beam_width))
    text = text.replace("{pybeam_h_cm}", str(beam_height))
    print(text)

    with open(f"{name}.com", "w") as f:
        f.writelines(text)

    # Legacy placeholder "example usage"
    command_to_run = ["executable_name", "arg1"]
    output_file_path = "output.txt"
    execute_command_and_write_to_file(command_to_run, output_file_path)


def saveCORRECT(expt: Any) -> None:  # noqa: N802 (legacy API)
    """
    Write a CORRECT `.com` file from a legacy-style experiment object.

    Parity-first copy of `legacy/lib/toscana.py::saveCORRECT`.

    The file ``<expt.sampleTitle>.com`` contains, in this order::

        ! <file name>
        !
        instr <instrument>
        sample "<sampleTitle>.adat" <sampleOuterRadius/10> /temperature=<T>
               /density=<ρ> /packing=<packing fraction, 8.6f> /fullness=<f>
        container "<containerFile minus 5 chars>.adat" <container[3]/20>
        background "<environmentFile minus 5 chars>.adat" 0.8
        ! black "absorber.adat" 0.93
        vanadium "<vanadiumFile minus 5 chars>.adat" <vanadium[3]/20>
                 /smoothing=1 /multiplier=1.02
        background /vanadium "<environment file>.adat" 0.85
        wavelenght <wavelength>
        ! zeroangle = <zeroAngle> already subtracted
        zeroangle 0.0
        beam <beamHeight/10> <beamWidth/10>
        ! xout angle / ! output ... / ! title ...   (commented alternatives)
        xout q
        output <sampleTitle>.corr.q
        title "<sampleTitle>.corr.q (after correct)"
        spectrum 1
        execute/nopause
        quit

    (the ``sample`` and ``vanadium`` commands are each on ONE line in the file.)
    The last 5 characters of the container, environment and vanadium file names
    are removed (assumed to be an extension such as ``.adat`` or ``.qdat``) and
    ``.adat`` is appended. The zero angle is written as 0 because it has already
    been subtracted by the regrouping.

    Parameters
    ----------
    expt : object
        Experiment object with the attributes: ``sampleTitle`` (str),
        ``instrument`` (str), ``sampleOuterRadius`` (mm), ``sampleTemperature``
        (K), ``sampleDensity`` (as expected by CORRECT), ``samplePackingFraction``
        (float), ``sampleFullness``, ``containerFile``, ``environmentFile``,
        ``vanadiumFile`` (str), ``container`` and ``vanadium`` (sequences whose
        element 3 is the outer diameter in mm), ``wavelength`` (Å), ``zeroAngle``
        (deg), ``beamHeight`` and ``beamWidth`` (mm).

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        If ``expt`` lacks one of the attributes above.

    Notes
    -----
    Writes ``<sampleTitle>.com`` in the current working directory (overwritten),
    prints nothing. The keyword is written ``wavelenght`` (sic, as in the legacy
    code; open item: to be checked against what CORRECT accepts).

    The current ``toscana.experiment.Measurement`` class does not define most of
    these attributes (``sampleTitle``, ``sampleOuterRadius``, ``containerFile``,
    ...), so this function needs a legacy-style object.
    """
    parfile = expt.sampleTitle + ".com"
    with open(parfile, "w") as f:
        f.write("! " + parfile + "\n")
        f.write("!\n")
        f.write("instr " + expt.instrument + "\n")
        line = 'sample "{}.adat" {} /temperature={} /density={} /packing={:8.6f} /fullness={}'.format(
            expt.sampleTitle,
            expt.sampleOuterRadius / 10.0,
            expt.sampleTemperature,
            expt.sampleDensity,
            expt.samplePackingFraction,
            expt.sampleFullness,
        )
        f.write(line + "\n")
        can = expt.containerFile[0:-5] + ".adat"
        f.write('container "{}" {}\n'.format(can, expt.container[3] / 20.0))
        bckg = expt.environmentFile[0:-5] + ".adat"
        f.write('background "{}" 0.8\n'.format(bckg))
        f.write('! black "absorber.adat" 0.93\n')
        vana = expt.vanadiumFile[0:-5] + ".adat"
        f.write(
            'vanadium "{}" {} /smoothing=1 /multiplier=1.02\n'.format(
                vana, expt.vanadium[3] / 20.0
            )
        )
        f.write('background /vanadium "{}" 0.85\n'.format(bckg))
        f.write("wavelenght {}\n".format(expt.wavelength))
        f.write("! zeroangle = {} already subtracted\n".format(expt.zeroAngle))
        f.write("zeroangle {}\n".format(0.0))
        f.write("beam {} {}\n".format(expt.beamHeight / 10.0, expt.beamWidth / 10.0))
        f.write("! xout angle\n")
        f.write("! output " + expt.sampleTitle + ".corr\n")
        f.write('! title "' + expt.sampleTitle + '.corr (after correct)"\n')
        f.write("xout q\n")
        f.write("output " + expt.sampleTitle + ".corr.q\n")
        f.write('title "' + expt.sampleTitle + '.corr.q (after correct)"\n')
        f.write("spectrum 1\n")
        f.write("execute/nopause\n")
        f.write("quit\n")
    return
