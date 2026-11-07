"""
make_api_appendix.py - Write appendices/A_api_reference.tex: the list of all the
classes, functions and methods of the package with a one-sentence description.

The text is extracted from the source files (docstrings and signatures) with the
`ast` module of the standard library, WITHOUT running the code. Run it again each
time the code changes, so the appendix is always up to date:

    python make_api_appendix.py                (modules two levels above this file)
    python make_api_appendix.py /path/to/code  (modules in another folder)
"""

import ast
import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.abspath(os.path.join(HERE, "..", "appendices", "A_api_reference.tex"))

# Modules in the order of the manual, with a short title for each.
MODULES = [
    ("beam", "Incident beam"),
    ("sample_base", "Common ingredients of all samples"),
    ("slab", "Slab geometry"),
    ("disc", "Disc geometry (version 2.3)"),
    ("cylinder", "Cylindrical geometry"),
    ("sphere", "Spherical geometry (version 2.4)"),
    ("sample_factory", "Choice of the geometry"),
    ("transmission_vs_angle", "Transmission versus scattering angle"),
    ("multiple_scattering", "Single and multiple scattering"),
    ("transmission_by_layer", "Transmission by layer of last scattering"),
    ("direct_attenuation", "Direct method for single-scattering attenuation"),
    ("formula", "Chemical formulas"),
    ("nuclear_data", "Atomic and nuclear data"),
    ("layer_builder", "Layers from formula, density and cross sections"),
    ("run_simulation", "Main script"),
]

ASCII = {"\u00b7": "*", "\u00b0": " deg", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
         "\u201c": '"', "\u201d": '"', "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u00e9": "e",
         "\u03b8": "theta", "\u03bc": "mu", "\u03c3": "sigma", "\u03c6": "phi", "\u00c5": "Angstrom"}


def tex_escape(text):
    """Escape the characters that LaTeX treats specially."""
    for k, v in ASCII.items():
        text = text.replace(k, v)
    out = []
    for ch in text:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}.get(ch, ch))
    return "".join(out)


def code(text):
    """Text as typewriter font with the \\code macro (breakable at underscores)."""
    for k, v in ASCII.items():
        text = text.replace(k, v)
    esc = tex_escape(text)
    esc = esc.replace(r"\_", r"\_\allowbreak{}").replace(".", ".\\allowbreak{}")
    return r"\texttt{" + esc + "}"


def first_sentence(doc, limit=260):
    """First sentence of a docstring, as LaTeX text (`x` becomes typewriter)."""
    if not doc:
        return ""
    para = inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ")
    para = re.sub(r"\s+", " ", para).strip()
    parts = re.split(r"(?<=[a-z0-9\)\]`])\.\s+(?=[A-Z])", para, maxsplit=1)
    sentence = parts[0].rstrip(".") + "."
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + " ..."
    # `name` -> typewriter, the rest escaped
    pieces = re.split(r"(`[^`]+`)", sentence)
    return "".join(code(p[1:-1]) if p.startswith("`") and p.endswith("`") else tex_escape(p)
                   for p in pieces)


def signature(node):
    """Readable signature  name(arguments)  of a function definition."""
    args = ast.unparse(node.args)
    args = re.sub(r"\bself,?\s*", "", args, count=1).strip()
    return f"{node.name}({args})"


def dataclass_signature(node):
    """Constructor signature of a dataclass, built from its annotated fields."""
    fields = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            text = f"{item.target.id}: {ast.unparse(item.annotation)}"
            if item.value is not None:
                text += f" = {ast.unparse(item.value)}"
            fields.append(text)
    return f"{node.name}({', '.join(fields)})"


def decorators(node):
    return {ast.unparse(d) for d in node.decorator_list}


def rows_for_module(tree):
    """List of rows (kind, name/signature, description) of a parsed module."""
    rows = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            kind = "class"
            if "dataclass" in " ".join(decorators(node)):
                kind = "dataclass"
            title = dataclass_signature(node) if kind == "dataclass" else \
                node.name + (f"({bases})" if bases else "")
            rows.append((kind, title, first_sentence(ast.get_docstring(node))))
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    decs = decorators(item)
                    if item.name.startswith("__") and item.name != "__init__":
                        continue
                    if item.name == "__init__" and not ast.get_docstring(item) and not item.args.args[1:]:
                        continue
                    k = "property" if "property" in decs else ("static" if "staticmethod" in decs else "method")
                    if item.name == "__init__":
                        k = "init"
                    if any("abstractmethod" in d for d in decs):
                        k = "abstract"
                    label = (item.name if k == "property" else signature(item))
                    internal = "  (internal)" if item.name.startswith("_") and item.name != "__init__" else ""
                    desc = first_sentence(ast.get_docstring(item))
                    if not desc and item.name == "__init__":
                        desc = "Constructor: creates the object (see the description of the class)."
                    rows.append((k, node.name + "." + label, desc + (r" \emph{(internal)}" if internal else "")))
        elif isinstance(node, ast.FunctionDef):
            internal = node.name.startswith("_")
            rows.append(("function", signature(node),
                         first_sentence(ast.get_docstring(node)) + (r" \emph{(internal)}" if internal else "")))
    return rows


def main():
    lines = ["% ---------------------------------------------------------------------",
             "% GENERATED by tools/make_api_appendix.py: do not edit by hand,",
             "% run the script again after changing the code.",
             "% ---------------------------------------------------------------------",
             r"\chapter{Reference of classes, functions and methods}",
             r"\label{app:api}",
             "",
             "This appendix lists, module by module, every class, function and method of the "
             "package, with the signature and a one-sentence description taken from the "
             "docstrings. Methods are written as \\code{Class.method(arguments)}. "
             "Names beginning with an underscore are internal helpers, listed for completeness. "
             "The full explanation of each item is in its docstring and in the chapters of this manual.",
             ""]
    for modname, title in MODULES:
        path = os.path.join(CODE_DIR, modname + ".py")
        tree = ast.parse(open(path, encoding="utf-8").read())
        doc = ast.get_docstring(tree) or ""
        head = doc.split("\n")[0]
        head = head.split(" - ", 1)[1] if " - " in head else head
        lines.append(rf"\section{{\texorpdfstring{{\texttt{{{tex_escape(modname + '.py')}}}}}{{{modname}.py}}: {tex_escape(title)}}}")
        lines.append(rf"\label{{app:api:{modname}}}")
        lines.append(tex_escape(head.strip()) + ("" if head.strip().endswith((".", "?")) else "."))
        lines.append("")
        lines.append(r"\begin{longtable}{@{}p{1.75cm}p{6.35cm}p{6.45cm}@{}}")
        lines.append(r"\toprule \textbf{Kind} & \textbf{Name and arguments} & \textbf{What it does} \\ \midrule \endhead")
        for kind, name, desc in rows_for_module(tree):
            lines.append(rf"\small {kind} & \raggedright\arraybackslash\small {code(name)} & "
                         rf"\raggedright\arraybackslash\small {desc} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
        lines.append("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written", OUT)


if __name__ == "__main__":
    main()
