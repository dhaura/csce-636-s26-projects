#!/usr/bin/env python3
"""Generate exact m-heights for all settings by enumerating LP vertices (no LP solver).

LP_{S,j} maximises c_j = u g_j subject to |u g_t| <= 1 for t outside S (|S| = m). When the
m-height is finite, the feasible region is a bounded polytope, and an optimum lies at a
vertex: u = s G_T^{-1} for k columns T outside S and signs s in {-1, 1}^k. Such a vertex,
with codeword c = u G, is feasible for some S of size m exactly when at most m entries of c
exceed 1 in magnitude, and its best objective is then max_t |c_t|. Hence

    h_m(C) = max over vertices with #{t : |c_t| > 1} <= m of max_t |c_t|.

All C(9, k) column sets and 2^(k-1) sign patterns (u and -u are equivalent) are enumerated
in a vectorised way. The m-height is infinite when some n - m columns of G do not span R^k;
those rows are dropped, as in the provided data and the test set. This reproduces the
provided labels to ~1e-11 in log2. It only produces training labels; the model never uses it.

Matrices are drawn with generate_data.py's empirical sampler, cycling k = 4, 5, 6 so each k
gets the same number of matrices. Each matrix contributes its finite rows m = 2, 3, ...
Output: one .npz with P zero-padded to 6 x 5 (int8), k, m (int8) and the height (float64).
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from itertools import combinations, product
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import FEATURES, N, compute_m_height, empirical_scales, sample_matrix  # noqa: E402

COLUMN_SETS = {k: list(combinations(range(N), k)) for k in (4, 5, 6)}
SIGNS = {k: np.array([(1,) + s for s in product((1, -1), repeat=k - 1)], dtype=np.float64) for k in (4, 5, 6)}


def vertex_heights(P):
    """m-heights for m = 0..n-k of a batch P (B, k, n-k); np.inf where infinite."""
    batch, k = P.shape[0], P.shape[1]
    G = np.concatenate((np.broadcast_to(np.eye(k), (batch, k, k)), P), axis=2)
    best = np.zeros((batch, N - k + 1))
    for T in COLUMN_SETS[k]:
        block = G[:, :, T]
        ok = np.abs(np.linalg.det(block)) >= 0.5          # integer P: nonsingular => |det| >= 1
        X = np.linalg.solve(np.where(ok[:, None, None], block, np.eye(k)), G)
        C = np.abs(np.einsum("sk,bkn->bsn", SIGNS[k], X))  # |codeword| at each vertex
        above = (C > 1 + 1e-9).sum(axis=2)
        top = C.max(axis=2)
        for m in range(1, N - k + 1):
            value = np.where(above <= m, top, 0.0).max(axis=1)
            best[:, m] = np.maximum(best[:, m], np.where(ok, value, 0.0))
    best[:, 0] = 1.0
    # Infinite when some n - m columns have rank < k (then the LP is unbounded).
    for m in range(1, N - k + 1):
        for R in combinations(range(N), N - m):
            deficient = np.linalg.matrix_rank(G[:, :, list(R)]) < k
            best[deficient, m] = np.inf
    return best


def generate_chunk(task):
    k, chunk_id, size, seed, scale_counts = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, k, chunk_id]))
    P = np.stack([sample_matrix(rng, k, "empirical", scale_counts) for _ in range(size)])
    heights = vertex_heights(P)
    rows_P, rows_m, rows_h = [], [], []
    for m in range(2, N - k + 1):
        keep = np.isfinite(heights[:, m])
        rows_P.append(P[keep]); rows_m.append(np.full(keep.sum(), m, dtype=np.int8)); rows_h.append(heights[keep, m])
    return chunk_id, np.concatenate(rows_P), np.concatenate(rows_m), np.concatenate(rows_h)


def verify(scale_counts, seed, count):
    """Compare all m-heights with the LP labeller on fresh matrices, infinite cases included."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 999]))
    for k in (4, 5, 6):
        P = np.stack([sample_matrix(rng, k, "empirical", scale_counts[str(k)]) for _ in range(count)])
        fast = vertex_heights(P)
        for m in range(2, N - k + 1):
            lp = np.array([compute_m_height(np.hstack([np.eye(k), p]), m) for p in P])
            both = np.isfinite(fast[:, m]) & np.isfinite(lp)
            agree_inf = np.array_equal(np.isinf(fast[:, m]), np.isinf(lp))
            err = np.abs(np.log2(fast[both, m]) - np.log2(lp[both])).max() if both.any() else 0.0
            print(f"verify (9,{k},{m}): {count} matrices, {int((~np.isfinite(lp)).sum())} infinite, "
                  f"infinite cases agree: {agree_inf}, max |log2 fast - log2 LP| = {err:.2e}", flush=True)
            if not agree_inf or err > 1e-6:
                raise RuntimeError("Vertex enumeration disagrees with the LP")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "CSCE-636-Project-1-Train-RandomExact.npz")
    parser.add_argument("--matrices-per-k", type=int, default=4_000_000)
    parser.add_argument("--chunk", type=int, default=10_000, help="Matrices per task")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=638, help="Differs from the other generators (636, 637)")
    parser.add_argument("--verify", type=int, default=100, help="Matrices per k checked against the LP first")
    args = parser.parse_args()

    start = time.time()
    with (args.input_dir / FEATURES).open("rb") as file:
        scale_counts = empirical_scales(pickle.load(file))
    print(f"Loaded scale histograms in {time.time() - start:.0f}s", flush=True)
    if args.verify:
        verify(scale_counts, args.seed, args.verify)

    parts = {"P": [], "k": [], "m": [], "height": []}
    summary = {}
    chunks = -(-args.matrices_per_k // args.chunk)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k in (4, 5, 6):
            t0, rows = time.time(), 0
            tasks = [(k, c, min(args.chunk, args.matrices_per_k - c * args.chunk), args.seed, scale_counts[str(k)])
                     for c in range(chunks)]
            # imap keeps chunk order, so the output does not depend on the worker count.
            for _, P, m, h in pool.imap(generate_chunk, tasks):
                padded = np.zeros((len(P), 6, 5), dtype=np.int8)
                padded[:, :k, :N - k] = P
                parts["P"].append(padded); parts["k"].append(np.full(len(P), k, dtype=np.int8))
                parts["m"].append(m); parts["height"].append(h)
                rows += len(P)
            summary[f"k={k}"] = {"matrices": args.matrices_per_k, "rows": rows}
            print(f"k={k}: {rows:,} finite rows from {args.matrices_per_k:,} matrices in {time.time() - t0:.0f}s", flush=True)

    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    counts = {f"(9,{k},{m})": int(np.sum((arrays['k'] == k) & (arrays['m'] == m))) for k in (4, 5, 6) for m in range(2, N - k + 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(args.output)
    meta = {"matrices_per_k": args.matrices_per_k, "seed": args.seed, "chunk": args.chunk,
            "sampler": "generate_data.py empirical", "label": "LP vertex enumeration (all m)",
            "numpy": np.__version__, "per_k": summary, "rows_per_setting": counts,
            "total_rows": int(len(arrays["height"]))}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Rows per setting: {counts}", flush=True)
    print(f"Wrote {len(arrays['height']):,} rows to {args.output} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
