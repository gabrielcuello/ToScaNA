MCAMS USER MANUAL - Monte Carlo Attenuation and Multiple Scattering (version 2.7)
=================================================================================

Compile:       latexmk -pdf main.tex        (or: make)    -> main.pdf
Alternative:   pdflatex main  (three times)

Content of this folder
  main.tex, preamble.tex    main file and packages/macros
  chapters/                 the 18 chapters (05b: vertical extent of the layers, new in 2.0
                            and extended in 2.2 with bottom and height; 05c: disc samples, new in 2.3;
                            05d: spherical samples, new in 2.4; 11b: worked examples
                            in the notation of CORRECT, new in 2.5 of the manual
                            (all the factors A_x,y and least squares: 2.6);
                            09b: direct method, new in 2.1)
  appendices/               A: reference of classes, functions, methods (generated)
                            B: NumPy functions   C: symbols   D: troubleshooting
                            E: listing of examples/correct_tools.py
  figures/                  all the figures (PDF), included by the chapters
  examples/                 the example scripts shown in the manual, and their outputs;
                            examples 13-18 use the helper file examples/correct_tools.py
                            (worked examples in the notation of CORRECT, 3-5 minutes each)
  tools/make_figures.py     regenerates the figures with the package itself
  tools/make_api_appendix.py regenerates appendix A from the source code
  tools/direct_method_analysis.py  validation of the direct method and comparison with
                            the Monte Carlo (numbers in tools/output/, figures fig_direct_*)

The manual is meant to be placed INSIDE the folder with the Python modules
(beam.py, slab.py, ...). The two tools and the examples look for the modules in
the parent folder; use MC_CODE_DIR=/path/to/code (figures) or an argument (api)
to use another location.

LaTeX packages used (all standard in TeX Live / MiKTeX): inputenc, fontenc, mathpazo,
txfonts (typewriter font txtt), microtype, babel, geometry, amsmath, amssymb, graphicx,
booktabs, tabularx, longtable, array, xcolor, listings, tcolorbox, enumitem, fancyhdr,
titlesec, caption, float, tikz, hyperref. The style is the same as the pycorrect manual.
