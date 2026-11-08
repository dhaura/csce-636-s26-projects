#!/usr/bin/env python3
"""Score final models on the held-out validation and test splits of the provided data (80/10/10 keyed matrix hash).

Usage: score_split.py MODEL.keras [MODEL.keras ...]
"""
import hashlib, pickle, sys
import numpy as np
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)
F = pickle.load(open("data/CSCE-636-Project-2-Train-n_k_m_P", "rb"))
y = np.log2(np.array(pickle.load(open("data/CSCE-636-Project-2-Train-mHeights", "rb")), float))
key, cache, bucket = (636).to_bytes(8, "little"), {}, np.empty(len(F), np.int8)
for i, (n, k, m, P) in enumerate(F):
    ident = (int(n), int(k), id(P))
    if ident not in cache:
        payload = bytes((int(n), int(k))) + np.asarray(P, dtype="<f8").tobytes(order="C")
        cache[ident] = int.from_bytes(hashlib.blake2b(payload, digest_size=8, key=key).digest(), "little") % 10
    bucket[i] = cache[ident]
X = np.zeros((len(F), 6, 5), np.float32); S = np.zeros((len(F), 3), np.float32)
for i, (n, k, m, P) in enumerate(F):
    X[i, :k, :n - k] = P; S[i] = (n, k, m)
settings = [(k, m) for k in (4, 5, 6) for m in range(2, 10 - k)]
print("model".ljust(58) + "split".ljust(12) + "".join(f"(9,{k},{m})".rjust(9) for k, m in settings) + "     mean")
for path in sys.argv[1:]:
    net = tf.keras.models.load_model(path, compile=False)
    for name, b in (("validation", 8), ("test", 9)):
        costs = []
        for k, m in settings:
            sel = np.flatnonzero((bucket == b) & (S[:, 1] == k) & (S[:, 2] == m)); sub = net.get_layer(f"network_k{k}_m{m}")
            f = tf.function(lambda batch: sub(batch, training=False), jit_compile=True); z = np.empty(len(sel))
            for s in range(0, len(sel), 4096):
                idx = sel[s:s + 4096]; z[s:s + 4096] = np.asarray(f({"matrix": X[idx], "parameters": S[idx]})).reshape(-1)
            costs.append(float(((y[sel] - z) ** 2).mean()))
        print(path.ljust(58) + name.ljust(12) + "".join(f"{c:9.4f}" for c in costs) + f"{np.mean(costs):9.4f}", flush=True)
