#!/usr/bin/env python3
"""Run the submitted notebook's final cell on a test file, label the test file exactly, and score the predictions.

Exact labels come from vertex enumeration (equivalent to the LP algorithm; a random sample is cross-checked with the
LP itself). Also reports how the test matrices relate to the provided training data.
Usage: score_test_set.py NOTEBOOK TEST_INPUT PREDICTIONS_OUT
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import hashlib, multiprocessing as mp, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_fast_data import vertex_heights
from generate_data import compute_m_height

N = 9


def label_chunk(task):
    k, P = task
    return vertex_heights(P)


def lp_label(task):
    P, k, m = task
    return compute_m_height(np.hstack([np.eye(k), P]), m)


if __name__ == "__main__":
    notebook, test_input, out = sys.argv[1:4]
    T = pickle.load(open(test_input, "rb"))
    km = np.array([(s[1], s[2]) for s in T]); settings = sorted({(int(k), int(m)) for k, m in km})

    # 1. Exact labels (vertex enumeration), grouped by k so each matrix is solved once for all m.
    labels_path = Path("data") / (Path(test_input).name + "-exact-heights.npy")
    if labels_path.exists():
        y = np.load(labels_path)
    else:
        t0 = time.time(); y = np.empty(len(T))
        with mp.get_context("fork").Pool(min(120, os.cpu_count())) as pool:
            for k in (4, 5, 6):
                rows = np.flatnonzero(km[:, 0] == k)
                P = np.stack([np.asarray(T[i][3], np.float64) for i in rows])
                chunks = np.array_split(np.arange(len(rows)), max(1, len(rows) // 2000))
                for c, h in zip(chunks, pool.imap(label_chunk, [(k, P[c]) for c in chunks])):
                    y[rows[c]] = h[np.arange(len(c)), km[rows[c], 1]]
            sample = np.random.default_rng(0).choice(len(T), 300, replace=False)
            lp = np.array(pool.map(lp_label, [(np.asarray(T[i][3], np.float64), int(T[i][1]), int(T[i][2])) for i in sample]))
        np.save(labels_path, y)
        print(f"exact labels: {len(y):,} samples in {time.time() - t0:.0f} s | finite {np.isfinite(y).all()} | min {y.min():.3f} | "
              f"LP cross-check on 300 random samples: max |log2 LP - log2 vertex| = {np.abs(np.log2(lp) - np.log2(y[sample])).max():.1e}")

    # 2. The submitted final cell, unchanged except for the input/output paths.
    import nbformat, __main__
    cell = nbformat.read(notebook, as_version=4).cells[-1].source
    for a, b in (('TEST_INPUT_PATH = "CSCE-636-Project-2-Test-n_k_m_P"', f'TEST_INPUT_PATH = "{test_input}"'),
                 ('OUTPUT_PATH = "CSCE-636-Project-2-Test-mHeights"', f'OUTPUT_PATH = "{out}"')):
        assert a in cell, a
        cell = cell.replace(a, b)
    cell = "\n".join(l.replace("%pip", "pass  # %pip", 1) if l.lstrip().startswith("%pip") else l for l in cell.splitlines())
    os.chdir(Path(notebook).resolve().parent)          # the cell finds the final model next to the notebook
    t0 = time.time(); exec(compile(cell, "<final cell>", "exec"), __main__.__dict__); print(f"final cell: {time.time() - t0:.0f} s")
    pred = pickle.load(open(out, "rb"))
    assert isinstance(pred, list) and len(pred) == len(T) and all(type(v) is float for v in pred)
    p = np.array(pred); assert np.isfinite(p).all() and p.min() >= 1
    print(f"format OK: list of {len(pred):,} Python floats, all finite, min {p.min():.4f}, max {p.max():.1f}")

    # 3. Scores.
    err = (np.log2(y) - np.log2(p)) ** 2
    per = {s: float(err[(km[:, 0] == s[0]) & (km[:, 1] == s[1])].mean()) for s in settings}
    print(f"\nCOST on the sample test set (mean over 9 settings): {np.mean(list(per.values())):.4f}")
    for (k, m), v in per.items():
        sel = (km[:, 0] == k) & (km[:, 1] == m)
        within2 = np.mean(np.abs(np.log2(y[sel]) - np.log2(p[sel])) <= 1)
        print(f"  (9,{k},{m}): {sel.sum():7,} samples | cost {v:.4f} | prediction within 2x of true height: {within2:.1%}")

    # 4. Relation to the provided training data.
    F = pickle.load(open("data/CSCE-636-Project-2-Train-n_k_m_P", "rb"))
    key = (636).to_bytes(8, "little")
    def bucket(n, k, P):
        return int.from_bytes(hashlib.blake2b(bytes((int(n), int(k))) + np.asarray(P, dtype="<f8").tobytes(order="C"),
                                              digest_size=8, key=key).digest(), "little") % 10
    provided = {}
    for n, k, m, P in F:
        provided.setdefault((int(k), np.asarray(P, np.int8).tobytes()), bucket(n, k, P))
    status = np.array([provided.get((int(s[1]), np.asarray(s[3], np.int8).tobytes()), -1) for s in T])
    print(f"\nTest samples whose matrix is in the provided data: {np.mean(status >= 0):.1%} "
          f"(in our training split: {np.mean((status >= 0) & (status < 8)):.1%}; in our validation/test splits: {np.mean(status >= 8):.1%})")
    for label, mask in (("matrix NOT in the provided data", status < 0), ("matrix in our training split", (status >= 0) & (status < 8)),
                        ("matrix in our validation/test splits", status >= 8)):
        if mask.any():
            pm = [float(err[mask & (km[:, 0] == k) & (km[:, 1] == m)].mean()) for k, m in settings if (mask & (km[:, 0] == k) & (km[:, 1] == m)).any()]
            print(f"  {label}: {mask.sum():,} samples | mean cost over settings {np.mean(pm):.4f}")
    rng = np.random.default_rng(1)
    for k in (4, 5, 6):
        train = np.unique(np.stack([np.asarray(P, np.int16).ravel() for n, kk, m, P in F if kk == k and m == 2]), axis=0)
        rows = np.flatnonzero((km[:, 0] == k) & (status < 0))
        if len(rows) == 0:
            continue
        q = np.stack([np.asarray(T[i][3], np.int16).ravel() for i in rng.choice(rows, min(500, len(rows)), replace=False)])
        near = np.mean([np.abs(train - v).max(1).min() <= 2 for v in q])
        print(f"  k={k}: of test matrices not in the provided data, {near:.0%} are within ±2 (every entry) of a provided matrix")
