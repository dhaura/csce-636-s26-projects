#!/usr/bin/env python3
"""Generate exact m-heights for the hardest settings, m = n - k, in closed form.

For m = n - k, LP_{S,j} constrains |c_t| <= 1 on the k positions T = [n] \\ S. A finite
m-height means the code is MDS: every k columns of G are linearly independent. Then
c_T determines the codeword (u = c_T G_T^{-1}), so c_j = c_T G_T^{-1} g_j, and the LP
optimum over the box c_T in [-1, 1]^k is ||G_T^{-1} g_j||_1. Hence

    h_{n-k}(C) = max over k-subsets T and j not in T of ||G_T^{-1} g_j||_1.

This reproduces the provided labels to ~1e-11 in log2 and runs hundreds of times faster
than the LP. It only produces training labels; the model never uses it.

Matrices are drawn with generate_data.py's empirical sampler (per-matrix scale from the
provided data, integer entries). Matrices with a singular k-column block have an infinite
m-height and are skipped, as in the provided data and the test set.

Output: one .npz with P zero-padded to 6 x 5 (int8), k, m (int8) and the height (float64).
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from itertools import combinations
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import FEATURES, N, compute_m_height, empirical_scales, sample_matrix  # noqa: E402

SUBSETS = {k: list(combinations(range(N), k)) for k in (4, 5, 6)}


def closed_form_heights(P):
    """h_{n-k} for a batch of P (B, k, n-k); inf where some k-column block is singular."""
    batch, k = P.shape[0], P.shape[1]
    G = np.concatenate((np.broadcast_to(np.eye(k), (batch, k, k)), P), axis=2)
    best = np.zeros(batch)
    singular = np.zeros(batch, dtype=bool)
    for T in SUBSETS[k]:
        rest = [j for j in range(N) if j not in T]
        block = G[:, :, T]
        # Integer matrices: a nonsingular block has |det| >= 1.
        bad = np.abs(np.linalg.det(block)) < 0.5
        singular |= bad
        block = np.where(bad[:, None, None], np.eye(k), block)
        X = np.linalg.solve(block, G[:, :, rest])
        best = np.maximum(best, np.abs(X).sum(axis=1).max(axis=1))
    best[singular] = np.inf
    return best


def generate_chunk(task):
    k, chunk_id, size, seed, scale_counts = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, k, chunk_id]))
    P = np.stack([sample_matrix(rng, k, "empirical", scale_counts) for _ in range(size)])
    heights = closed_form_heights(P)
    keep = np.isfinite(heights)
    return chunk_id, P[keep], heights[keep]


def verify(scale_counts, seed, count):
    """Compare the closed form with the LP labeller on fresh matrices."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 999]))
    for k in (4, 5, 6):
        P = np.stack([sample_matrix(rng, k, "empirical", scale_counts[str(k)]) for _ in range(count)])
        closed = closed_form_heights(P)
        lp = np.array([compute_m_height(np.hstack([np.eye(k), p]), N - k) for p in P])
        both = np.isfinite(closed) & np.isfinite(lp)
        agree_inf = np.array_equal(np.isinf(closed), np.isinf(lp))
        err = np.abs(np.log2(closed[both]) - np.log2(lp[both])).max()
        print(f"verify k={k} m={N - k}: {count} matrices, infinite cases agree: {agree_inf}, "
              f"max |log2 closed - log2 LP| = {err:.2e}", flush=True)
        if not agree_inf or err > 1e-6:
            raise RuntimeError("Closed form disagrees with the LP")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "CSCE-636-Project-2-Train-RandomExact-MaxM.npz")
    parser.add_argument("--rows-per-setting", type=int, default=10_000_000)
    parser.add_argument("--chunk", type=int, default=20_000, help="Matrices per task")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=637, help="Differs from generate_data.py's 636")
    parser.add_argument("--verify", type=int, default=200, help="Matrices per k checked against the LP first")
    args = parser.parse_args()

    start = time.time()
    with (args.input_dir / FEATURES).open("rb") as file:
        scale_counts = empirical_scales(pickle.load(file))
    print(f"Loaded scale histograms in {time.time() - start:.0f}s", flush=True)
    if args.verify:
        verify(scale_counts, args.seed, args.verify)

    parts = {"P": [], "k": [], "m": [], "height": []}
    summary = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k in (4, 5, 6):
            t0, rows, drawn, next_chunk = time.time(), 0, 0, 0
            while rows < args.rows_per_setting:
                # Bounded waves of chunks; imap keeps chunk order, so the output does not
                # depend on the worker count.
                wave = [(k, chunk, args.chunk, args.seed, scale_counts[str(k)])
                        for chunk in range(next_chunk, next_chunk + 2 * args.workers)]
                next_chunk += len(wave)
                for chunk_id, P, heights in pool.imap(generate_chunk, wave):
                    take = min(len(P), args.rows_per_setting - rows)
                    if take <= 0:
                        break
                    padded = np.zeros((take, 6, 5), dtype=np.int8)
                    padded[:, :k, :N - k] = P[:take]
                    parts["P"].append(padded)
                    parts["k"].append(np.full(take, k, dtype=np.int8))
                    parts["m"].append(np.full(take, N - k, dtype=np.int8))
                    parts["height"].append(heights[:take])
                    rows += take
                    drawn += args.chunk
            summary[f"(9,{k},{N - k})"] = {"rows": rows, "matrices_drawn": drawn,
                                           "finite_fraction": round(rows / drawn, 4)}
            print(f"k={k} m={N - k}: {rows:,} rows from {drawn:,} matrices "
                  f"in {time.time() - t0:.0f}s", flush=True)

    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(args.output)
    meta = {"rows_per_setting": args.rows_per_setting, "seed": args.seed, "chunk": args.chunk,
            "sampler": "generate_data.py empirical", "label": "closed form h_{n-k}",
            "numpy": np.__version__, "settings": summary, "total_rows": int(len(arrays["height"]))}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(arrays['height']):,} rows to {args.output} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
