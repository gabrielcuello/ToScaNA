# Makefile of the manual.  Usage:
#   make            compile main.pdf
#   make figures    regenerate all the figures (needs the Python modules, see tools/)
#   make api        regenerate appendices/A_api_reference.tex from the source code
#   make direct     rerun the direct-method analysis (tables in tools/output, fig_direct_*)
#   make examples   run the examples and refresh their outputs
#   make clean      remove the auxiliary files of LaTeX
# The Python modules are expected in the folder above this one (override with CODE=...).
CODE ?= ..

all: main.pdf

main.pdf: main.tex preamble.tex $(wildcard chapters/*.tex appendices/*.tex figures/*.pdf examples/*.py examples/output/*.txt)
	latexmk -pdf -interaction=nonstopmode main.tex

figures:
	MC_CODE_DIR=$(CODE) python3 tools/make_figures.py

direct:
	MC_CODE_DIR=$(CODE) python3 tools/direct_method_analysis.py

api:
	python3 tools/make_api_appendix.py $(CODE)

examples:
	cd $(CODE) && for f in manual/examples/ex*.py; do b=$$(basename $$f .py); PYTHONPATH=. python3 $$f > manual/examples/output/$$b.txt; done

clean:
	latexmk -C
