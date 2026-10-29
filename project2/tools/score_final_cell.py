#!/usr/bin/env python3
"""End-to-end check of a submission: run NOTEBOOK's final cell with MODEL on a labelled test file, then score it.

The test file is a pickled list of [n, k, m, P] with a matching pickled list of true m-heights (the format of the
provided training files). Rows are grouped by our 80/10/10 matrix-level split (same keyed hash as the notebook): only
rows whose matrix is in our validation or test split were never trained on, so those scores are fair.
Usage: score_final_cell.py NOTEBOOK MODEL TEST_INPUT TEST_HEIGHTS OUT
"""
import hashlib, pickle, re, sys, time
import __main__
import nbformat
import numpy as np

notebook, model, test_input, test_heights, out = sys.argv[1:6]
cell = nbformat.read(notebook, as_version=4).cells[-1].source
for a, b in (('TEST_INPUT_PATH = "CSCE-636-Project-2-Test-n_k_m_P"', f'TEST_INPUT_PATH = "{test_input}"'),
             ('OUTPUT_PATH = "CSCE-636-Project-2-Test-mHeights"', f'OUTPUT_PATH = "{out}"')):
    assert a in cell, a
    cell = cell.replace(a, b)
cell, found = re.subn(r"MODEL_FILES = \[[^\]]*\]", f'MODEL_FILES = ["{model}"]', cell)
assert found == 1, "MODEL_FILES"
cell = "\n".join(l.replace("%pip", "pass  # %pip", 1) if l.lstrip().startswith("%pip") else l for l in cell.splitlines())
t0 = time.time(); exec(compile(cell, "<final cell>", "exec"), __main__.__dict__); print(f"final cell: {time.time() - t0:.0f} s")

T = pickle.load(open(test_input, "rb"))
pred = pickle.load(open(out, "rb"))
assert isinstance(pred, list) and len(pred) == len(T) and all(type(v) is float for v in pred), "format"
p = np.array(pred); assert np.all(np.isfinite(p)) and p.min() >= 1, "values"
y = np.log2(np.array(pickle.load(open(test_heights, "rb")), float)); z = np.log2(p)
key = (636).to_bytes(8, "little")
bucket = np.array([int.from_bytes(hashlib.blake2b(bytes((int(n), int(k))) + np.asarray(P, dtype="<f8").tobytes(order="C"),
                                                  digest_size=8, key=key).digest(), "little") % 10 for n, k, m, P in T])
km = np.array([(s[1], s[2]) for s in T]); settings = sorted({(k, m) for k, m in km})
print(f"format OK: {len(pred):,} Python floats, all finite, min {p.min():.3f}")
for label, mask in (("rows in our TEST split (fair)", bucket == 9), ("rows in our VALIDATION split (fair)", bucket == 8),
                    ("all rows (optimistic: 80% were trained on)", np.ones(len(T), bool))):
    per = [float(((y - z)[mask & (km[:, 0] == k) & (km[:, 1] == m)] ** 2).mean()) for k, m in settings]
    print(f"{label}: {mask.sum():,} rows | mean {np.mean(per):.4f} | " + " ".join(f"(9,{k},{m}) {v:.4f}" for (k, m), v in zip(settings, per)))
