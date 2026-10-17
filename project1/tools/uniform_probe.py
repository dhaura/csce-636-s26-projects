#!/usr/bin/env python3
"""Score final models on completely random matrices: every entry of P independent and uniform on [-100, 100].

Two probe sets, 20,000 matrices per k each (seed 640), labelled exactly by vertex enumeration for every finite m:
  - integer: uniform integers in [-100, 100] (like the provided data, which has integer entries);
  - real:    uniform real numbers in [-100, 100] (the specification allows any real entry in that range).
Matrices with an all-zero column are redrawn; infinite heights are dropped.
Usage: uniform_probe.py MODEL.keras [MODEL.keras ...]
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import multiprocessing as mp, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_fast_data import vertex_heights

N = 9


def chunk(task):
    kind, k, i = task
    rng = np.random.default_rng(np.random.SeedSequence([640, {"integer": 0, "real": 1}[kind], k, i]))
    draw = (lambda s: rng.integers(-100, 101, s).astype(np.float64)) if kind == "integer" else (lambda s: rng.uniform(-100, 100, s))
    P = draw((1000, k, N - k))
    while (bad := ~np.any(P != 0, axis=1).all(axis=1)).any():
        P[bad] = draw((bad.sum(), k, N - k))
    return kind, k, P, vertex_heights(P)


if __name__ == "__main__":
    probes = {}
    for kind in ("integer", "real"):
        path = Path(f"data/CSCE-636-Project-1-Probe-Uniform-{kind}.npz")
        if not path.exists():
            X, S, y = [], [], []
            with mp.get_context("spawn").Pool(min(64, os.cpu_count())) as pool:
                for _, k, P, h in pool.imap(chunk, [(kind, k, i) for k in (4, 5, 6) for i in range(20)]):
                    for m in range(2, N - k + 1):
                        ok = np.isfinite(h[:, m]); big = np.zeros((ok.sum(), 6, 5), np.float32); big[:, :k, :N - k] = P[ok]
                        X.append(big); S.append(np.tile([9, k, m], (ok.sum(), 1)).astype(np.float32)); y.append(h[ok, m])
            np.savez(path, P=np.concatenate(X), setting=np.concatenate(S), height=np.concatenate(y))
        probes[kind] = np.load(path)
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    settings = [(k, m) for k in (4, 5, 6) for m in range(2, 10 - k)]
    nets = {p: tf.keras.models.load_model(p, compile=False) for p in sys.argv[1:]}
    for kind, d in probes.items():
        X, Pm, y = d["P"].astype(np.float32), d["setting"].astype(np.float32), np.log2(d["height"])
        print(f"\nUniform {kind} entries in [-100, 100]: {len(y):,} rows (20,000 matrices per k)")
        print("model".ljust(52) + "".join(f"(9,{k},{m})".rjust(9) for k, m in settings) + "     mean")
        for path, net in nets.items():
            z = np.empty(len(y)); costs = []
            for k, m in settings:
                sel = np.flatnonzero((Pm[:, 1] == k) & (Pm[:, 2] == m)); sub = net.get_layer(f"network_k{k}_m{m}")
                f = tf.function(lambda b: sub(b, training=False), jit_compile=True)
                for s in range(0, len(sel), 4096):
                    idx = sel[s:s + 4096]; z[idx] = np.asarray(f({"matrix": X[idx], "parameters": Pm[idx]})).reshape(-1)
                costs.append(float(((y[sel] - z[sel]) ** 2).mean()))
            print(path.ljust(52) + "".join(f"{c:9.4f}" for c in costs) + f"{np.mean(costs):9.4f}", flush=True)
