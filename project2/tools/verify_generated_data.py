#!/usr/bin/env python3
"""Independent check of the generated training data against the project's LP algorithm.

For every generated file and setting, random rows are relabelled with the LP method of the
project description (generate_data.compute_m_height: SciPy HiGHS LPs over all S and j), and
compared with the stored labels. Also checks basic properties of each file, and that no
augmented-provided matrix equals a validation or test matrix of its split.
"""
import hashlib, multiprocessing as mp, pickle, sys
import numpy as np
sys.path.insert(0, ".")
from generate_data import compute_m_height

N, PER_CELL = 9, 150
FILES = {"RandomExact": "data/CSCE-636-Project-2-Train-RandomExact.npz",
         "RandomExact-MaxM": "data/CSCE-636-Project-2-Train-RandomExact-MaxM.npz",
         "AugmentedNear-80-10-10": "data/CSCE-636-Project-2-Train-AugmentedNear-80-10-10.npz"}


def lp_label(task):
    P, k, m = task
    return compute_m_height(np.hstack([np.eye(k), P[:k, :N - k].astype(np.float64)]), m)


def buckets(features):
    key, cache, out = (636).to_bytes(8, "little"), {}, np.empty(len(features), np.int8)
    for i, (n, k, m, P) in enumerate(features):
        ident = (n, k, id(P))
        if ident not in cache:
            payload = bytes((int(n), int(k))) + np.asarray(P, dtype="<f8").tobytes(order="C")
            cache[ident] = int.from_bytes(hashlib.blake2b(payload, digest_size=8, key=key).digest(), "little") % 10
        out[i] = cache[ident]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    F = pickle.load(open("data/CSCE-636-Project-2-Train-n_k_m_P", "rb"))
    B = buckets(F)
    pool = mp.get_context("spawn").Pool(120)
    for name, path in FILES.items():
        d = np.load(path); P, k, m, h = d["P"], d["k"], d["m"], d["height"]
        ok_entries = bool(np.all(np.abs(P.astype(int)) <= 100))
        zero_col = sum(int(np.any(np.all(P[sel][:, :kk, :N - kk] == 0, axis=1))) for kk in (4, 5, 6) for sel in [k == kk])
        print(f"\n== {name}: {len(h):,} rows | labels finite {np.isfinite(h).all()} | min label {h.min():.3f} | "
              f"entries within [-100,100] {ok_entries} | matrices with an all-zero column: {zero_col}", flush=True)
        if name.startswith("Augmented"):
            held_buckets = (6, 7, 8, 9) if name.endswith("60-20-20") else (8, 9)
            held = {(kk, F[i][3].astype(np.int8).tobytes()) for i in np.flatnonzero(np.isin(B, held_buckets)) for kk in [F[i][1]]}
            sample = rng.choice(len(h), size=min(len(h), 2_000_000), replace=False)
            leaks = sum((int(k[i]), P[i][:k[i], :N - k[i]].astype(np.int8).tobytes()) in held for i in sample)
            print(f"   leakage check on {len(sample):,} sampled rows: {leaks} equal a validation/test matrix of this split", flush=True)
        for kk in (4, 5, 6):
            for mm in range(2, N - kk + 1):
                rows = np.flatnonzero((k == kk) & (m == mm))
                if len(rows) == 0:
                    continue
                pick = rng.choice(rows, size=min(PER_CELL, len(rows)), replace=False)
                lp = np.array(pool.map(lp_label, [(P[i], kk, mm) for i in pick]))
                err = np.abs(np.log2(lp) - np.log2(h[pick]))
                print(f"   (9,{kk},{mm}): {len(rows):10,d} rows | {len(pick)} relabelled by LP | infinite by LP: {int(np.isinf(lp).sum())} | "
                      f"max |log2 LP - log2 stored| = {np.nanmax(np.where(np.isfinite(err), err, np.nan)):.1e}", flush=True)
