#!/usr/bin/env python3
"""Execute a notebook with the current directory as its working directory (nbconvert would use the notebook's own
folder), and save the executed copy; on an error, the partly executed copy is saved before failing.
Usage: execute_notebook.py NOTEBOOK.ipynb OUTPUT.ipynb [KERNEL]
"""
import os, sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

source, output = sys.argv[1], sys.argv[2]
nb = nbformat.read(source, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name=sys.argv[3] if len(sys.argv) > 3 else "csce636-p1",
                        resources={"metadata": {"path": os.getcwd()}})
try:
    client.execute()
finally:
    nbformat.write(nb, output)
    print(f"saved {output}")
