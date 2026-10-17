#!/usr/bin/env python3
"""Execute the code cells of $NOTEBOOK (default project1_v8.ipynb) up to (and including) the first cell containing MARKER.

Used to run the data-generation part of the notebook as a CPU batch job, so the generated
files come from exactly the notebook's code.  Usage: run_notebook_prefix.py MARKER
"""
import os
import sys
import nbformat

marker = sys.argv[1]
import __main__
namespace = __main__.__dict__  # run as a notebook kernel would, so worker functions pickle by reference
for cell in nbformat.read(os.environ.get("NOTEBOOK", "project1_v8.ipynb"), as_version=4).cells:
    if cell.cell_type != "code" or "%pip" in cell.source:
        continue
    exec(compile(cell.source, "<notebook>", "exec"), namespace)
    if marker in cell.source:
        break
