#!/usr/bin/env python3
"""Score final models on matrices from outside the provided clusters: fresh independent random matrices.

20,000 matrices per k from the empirical sampler (seed 639, unused by any training file), labelled exactly by
vertex enumeration, all finite m. No model has trained on them, and they are not near any provided matrix.
Usage: offcluster_probe.py MODEL.keras [MODEL.keras ...]
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import multiprocessing as mp, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_data import FEATURES, N, empirical_scales, sample_matrix
from generate_fast_data import vertex_heights

PROBE = Path("data/CSCE-636-Project-2-Probe-RandomOffCluster.npz")


def chunk(task):
    k, i, counts = task
    rng = np.random.default_rng(np.random.SeedSequence([639, k, i]))
    P = np.stack([sample_matrix(rng, k, "empirical", counts) for _ in range(1000)])
    return k, P, vertex_heights(P)


if __name__ == "__main__":
    if not PROBE.exists():
        with open(Path("data") / FEATURES, "rb") as f:
            counts = empirical_scales(pickle.load(f))
        X, S, y = [], [], []
        with mp.get_context("spawn").Pool(min(64, os.cpu_count())) as pool:
            for k, P, h in pool.imap(chunk, [(k, i, counts[str(k)]) for k in (4, 5, 6) for i in range(20)]):
                for m in range(2, N - k + 1):
                    ok = np.isfinite(h[:, m]); big = np.zeros((ok.sum(), 6, 5), np.int8); big[:, :k, :N - k] = P[ok]
                    X.append(big); S.append(np.tile([9, k, m], (ok.sum(), 1)).astype(np.int8)); y.append(h[ok, m])
        np.savez(PROBE, P=np.concatenate(X), setting=np.concatenate(S), height=np.concatenate(y))
    d = np.load(PROBE); X, Pm, y = d["P"].astype(np.float32), d["setting"].astype(np.float32), np.log2(d["height"])
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    settings = [(k, m) for k in (4, 5, 6) for m in range(2, 10 - k)]
    print(f"Off-cluster probe: {len(y):,} rows of independent random matrices (20,000 per k)")
    print("model".ljust(52) + "".join(f"(9,{k},{m})".rjust(9) for k, m in settings) + "     mean")
    for path in sys.argv[1:]:
        net = tf.keras.models.load_model(path, compile=False); z = np.empty(len(y)); costs = []
        for k, m in settings:
            sel = np.flatnonzero((Pm[:, 1] == k) & (Pm[:, 2] == m)); sub = net.get_layer(f"network_k{k}_m{m}")
            f = tf.function(lambda b: sub(b, training=False), jit_compile=True)
            for s in range(0, len(sel), 4096):
                idx = sel[s:s + 4096]; z[idx] = np.asarray(f({"matrix": X[idx], "parameters": Pm[idx]})).reshape(-1)
            costs.append(float(((y[sel] - z[sel]) ** 2).mean()))
        print(path.ljust(52) + "".join(f"{c:9.4f}" for c in costs) + f"{np.mean(costs):9.4f}", flush=True)
