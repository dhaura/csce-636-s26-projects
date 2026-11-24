#!/usr/bin/env python3
"""The two codes of Project 3, the noisy-codeword distribution, and the correctness and format rules.

Everything follows Section 1 of the project description exactly:
  y = c + eps + e,  c = u G,  u ~ U[-1, 1]^k,  eps ~ U[-delta, delta]^n,
  e: A in {3, 2, 1, 0} with probabilities 1/2, 1/4, 1/8, 1/8 nonzero entries at A distinct uniformly chosen
     locations, each uniform on [-Y, Y] without zero, Y = 3 Delta.
A decoding output S is correct when R(e) <= S <= R(e) u O(e), i.e. it contains every location with |e_j| > Delta
and no location with e_j = 0.

numpy is used for the fixed evaluation sets (float64, seeded) and tf_sample() draws fresh training batches on the
GPU (float32). Run this file to print the constants and check them against the decimals in the description.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
_P = 1 / PHI

G1 = np.array([[1, 1, 1, -1, 0, 0, _P, -_P, PHI, -PHI],
               [1, 1, -1, 1, PHI, PHI, 0, 0, _P, _P],
               [1, -1, 1, 1, _P, -_P, PHI, PHI, 0, 0]], dtype=np.float64)
G2 = np.zeros((3, 16))
G2[:, :4] = -1
for _i in range(3):
    G2[_i, 4 + 4 * _i:8 + 4 * _i] = 1


def _code(G, multiple):
    delta = float(np.linalg.norm(G, axis=0).mean())   # average column length
    Delta = multiple * delta
    return dict(G=G, k=G.shape[0], n=G.shape[1], delta=delta, Delta=Delta, Y=3 * Delta)


CODES = {1: _code(G1, 6 + 2 * np.sqrt(5)), 2: _code(G2, 8.0)}
A_VALUES = np.array([3, 2, 1, 0])
A_PROBS = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])


def sample(code, size, rng):
    """Draw `size` noisy codewords of one code; returns y (size, n) and the UME vector e (size, n), float64."""
    if code in HARMONIC_A:
        return sample_harmonic(code, size, rng)
    C = CODES[code]; n = C["n"]
    u = rng.uniform(-1, 1, (size, C["k"]))
    eps = rng.uniform(-C["delta"], C["delta"], (size, n))
    A = rng.choice(A_VALUES, size=size, p=A_PROBS)
    chosen = np.argsort(rng.random((size, n)), axis=1).argsort(axis=1) < A[:, None]   # A uniform distinct locations
    # Uniform on [-Y, Y] without zero: magnitude Y (1 - U) lies in (0, Y] for U in [0, 1), with an independent sign.
    values = C["Y"] * (1 - rng.random((size, n))) * rng.choice((-1.0, 1.0), (size, n))
    e = np.where(chosen, values, 0.0)
    return u @ C["G"] + eps + e, e


def targets(code, e):
    """Per-location training targets: 1 required (|e| > Delta), 0 forbidden (e = 0), and a mask that is 0 at the
    optional locations (0 < |e| <= Delta), where either answer is correct."""
    required = np.abs(e) > CODES[code]["Delta"]
    optional = (e != 0) & ~required
    return required.astype(np.float32), (~optional).astype(np.float32)


def correct(code, e, reported):
    """Per-sample correctness of boolean decisions `reported` (size, n): all required and no forbidden location."""
    required = np.abs(e) > CODES[code]["Delta"]
    return ~(required & ~reported).any(axis=1) & ~(reported & (e == 0)).any(axis=1)


def to_sets(reported):
    """Boolean decisions (size, n) -> list of sets of Python ints, the required output format."""
    return [set(int(j) for j in np.flatnonzero(row)) for row in reported]


def check_output(outputs, num_inputs, n):
    """Raise ValueError unless `outputs` is a list of `num_inputs` sets of ints in [0, n) (no bools)."""
    if type(outputs) is not list or len(outputs) != num_inputs:
        raise ValueError(f"expected a list of {num_inputs} sets, got {type(outputs).__name__} of length {len(outputs)}")
    for r, s in enumerate(outputs):
        if type(s) is not set:
            raise ValueError(f"output {r} is a {type(s).__name__}, not a set")
        for j in s:
            if type(j) is not int or not 0 <= j < n:
                raise ValueError(f"output {r} contains {j!r}, not an int location in [0, {n})")


def tf_sample(code, size):
    """The same distribution as sample(), drawn in TensorFlow (float32) for on-the-fly training batches."""
    import tensorflow as tf
    C = CODES[code]; n = C["n"]
    G = tf.constant(C["G"], tf.float32)
    u = tf.random.uniform((size, C["k"]), -1.0, 1.0)
    eps = tf.random.uniform((size, n), -C["delta"], C["delta"])
    A = tf.gather(tf.constant(A_VALUES, tf.int32),
                  tf.random.categorical(tf.math.log(tf.constant(A_PROBS[None], tf.float32)), size)[0])
    ranks = tf.argsort(tf.argsort(tf.random.uniform((size, n)), axis=1), axis=1)
    chosen = ranks < A[:, None]
    sign = tf.where(tf.random.uniform((size, n)) < 0.5, -1.0, 1.0)
    values = C["Y"] * (1.0 - tf.random.uniform((size, n))) * sign
    e = tf.where(chosen, values, 0.0)
    return u @ G + eps + e, e


# ----------------------------------------------------------------------------------------------------------------
# Codes 3-6 (added in the project update): matrices and Delta from data/codes_3_to_6.pkl, delta = 1, a fixed number
# A of UMEs, u ~ U[-2 Delta, 2 Delta]^k, LME magnitudes ~ N(delta, 0.3 delta) on [0, delta] with random signs, and
# UME magnitudes ~ N(Delta, 0.3 Delta) on (Delta, 2 Delta] (Group I, ceil(N/2) samples) or on (0, 2 Delta] with at
# least one above Delta (Group II), with random signs; the two groups are shuffled together.
#
# Every G has rows 0.5... cos(m theta_j), sin(m theta_j) for the odd harmonics m = 1, 3, ..., k-1 at theta_j = pi j / n
# (checked below). The odd harmonics k+1, ..., n-1 that G leaves out give an orthonormal parity-check matrix H
# (r x n, H G^T = 0), so the syndrome s = H y = H (eps + e) does not depend on u. ROTATIONS[j] turns s into the frame
# of location j: an error at j alone gives the same pattern (1, 0, 1, 0, ...) in every frame.
# ----------------------------------------------------------------------------------------------------------------
HARMONIC_FILE = Path(__file__).resolve().parent / "data" / "codes_3_to_6.pkl"
HARMONIC_A = {3: 4, 4: 5, 5: 10, 6: 20}
HARMONIC = tuple(HARMONIC_A)


def _harmonic_code(code, entry):
    G = np.asarray(entry["G"], dtype=np.float64); k, n = G.shape
    theta = np.pi * np.arange(n) / n
    harmonic = np.array([f(m * theta) for m in range(1, k, 2) for f in (np.cos, np.sin)]) / np.sqrt(k / 2)
    assert np.abs(G - harmonic).max() < 1e-12, f"code {code}: G is not the odd-harmonic matrix"
    freqs = np.arange(k + 1, n, 2)
    H = np.array([f(m * theta) for m in freqs for f in (np.cos, np.sin)]) / np.sqrt(n / 2)
    assert np.abs(G @ H.T).max() < 1e-12 and np.allclose(H @ H.T, np.eye(n - k))
    # ROTATIONS[j] (r x r): rotation of every (cos, sin) pair of frequency m by -m theta_j, so ROTATIONS[j] @ H[:, j]
    # = H[:, 0] and ROTATIONS[j] @ s depends only on the error positions relative to j.
    rot = np.zeros((n, n - k, n - k))
    for p, m in enumerate(freqs):
        c, s = np.cos(m * theta), np.sin(m * theta)
        rot[:, 2 * p, 2 * p], rot[:, 2 * p, 2 * p + 1] = c, s
        rot[:, 2 * p + 1, 2 * p], rot[:, 2 * p + 1, 2 * p + 1] = -s, c
    assert np.allclose(np.einsum("jqr,rj->jq", rot, H), H[:, :1].T)
    delta = float(np.linalg.norm(G, axis=0).mean())
    assert abs(delta - 1) < 1e-12
    return dict(G=G, k=k, n=n, r=n - k, delta=1.0, Delta=float(entry["Delta"]), A=HARMONIC_A[code], H=H,
                ROTATIONS=rot, family="harmonic")


if HARMONIC_FILE.exists():
    with open(HARMONIC_FILE, "rb") as _f:
        _entries = pickle.load(_f)
    for _c in HARMONIC:
        CODES[_c] = _harmonic_code(_c, _entries[f"code_{_c}"])


def _truncated_normal(rng, mean, sd, shape, low, high, low_open):
    """Rejection sampling as in the description: draw N(mean, sd), redraw values outside the interval
    [low, high] (or (low, high] if low_open) until all are inside."""
    x = rng.normal(mean, sd, shape)
    bad = ~(((x > low) if low_open else (x >= low)) & (x <= high))
    while bad.any():
        x[bad] = rng.normal(mean, sd, int(bad.sum()))
        bad[bad] = ~(((x[bad] > low) if low_open else (x[bad] >= low)) & (x[bad] <= high))
    return x


def sample_harmonic(code, size, rng, return_group=False):
    """Codes 3-6: `size` noisy codewords drawn exactly as in the description (both groups, shuffled jointly).
    Returns y and e (size, n), float64, and with return_group the group of every sample (1 = I, 2 = II)."""
    C = CODES[code]; n, k, A, D = C["n"], C["k"], C["A"], C["Delta"]
    u = rng.uniform(-2 * D, 2 * D, (size, k))
    eps = _truncated_normal(rng, 1.0, 0.3, (size, n), 0.0, 1.0, False) * rng.choice((-1.0, 1.0), (size, n))
    locations = np.argsort(rng.random((size, n)), axis=1)[:, :A]        # A distinct, uniformly chosen locations
    n1 = (size + 1) // 2                                                # ceil(size / 2) samples in Group I
    mags = np.empty((size, A))
    mags[:n1] = _truncated_normal(rng, D, 0.3 * D, (n1, A), D, 2 * D, True)
    mags[n1:] = _truncated_normal(rng, D, 0.3 * D, (size - n1, A), 0.0, 2 * D, True)
    redo = n1 + np.flatnonzero(mags[n1:].max(axis=1) <= D)              # Group II: none above Delta -> redraw all A
    while len(redo):
        mags[redo] = _truncated_normal(rng, D, 0.3 * D, (len(redo), A), 0.0, 2 * D, True)
        redo = redo[mags[redo].max(axis=1) <= D]
    e = np.zeros((size, n))
    np.put_along_axis(e, locations, mags * rng.choice((-1.0, 1.0), (size, A)), axis=1)
    y = u @ C["G"] + eps + e
    group = np.where(np.arange(size) < n1, 1, 2)
    order = rng.permutation(size)                                       # joint shuffle of both groups
    return (y[order], e[order], group[order]) if return_group else (y[order], e[order])


def syndrome(code, y):
    """The parity checks of codes 3-6 in units of Delta, s = y H^T / Delta (float64; one matrix multiplication)."""
    C = CODES[code]
    return np.asarray(y, np.float64) @ C["H"].T / C["Delta"]


def tf_sample_harmonic(code, size):
    """The codes 3-6 distribution drawn in TensorFlow for training batches. Everything is drawn and computed in
    float64 (truncated normals by the inverse CDF, the same distribution as rejection sampling; Group II sets with no
    magnitude above Delta are redrawn); returns the syndrome s = y H^T / Delta as float32 and e as float64."""
    import tensorflow as tf
    C = CODES[code]; n, k, A, D = C["n"], C["k"], C["A"], C["Delta"]
    f64 = tf.float64
    phi = lambda z: 0.5 * tf.math.erfc(-z / np.sqrt(2.0))

    def truncated(shape, mean, sd, low, high):
        a, b = phi(tf.constant((low - mean) / sd, f64)), phi(tf.constant((high - mean) / sd, f64))
        p = b - tf.random.uniform(shape, dtype=f64) * (b - a)          # in (a, b]
        return tf.clip_by_value(mean + sd * tf.math.ndtri(p), low, high)

    sign = lambda shape: tf.where(tf.random.uniform(shape) < 0.5, tf.constant(-1.0, f64), tf.constant(1.0, f64))
    u = tf.random.uniform((size, k), -2 * D, 2 * D, dtype=f64)
    eps = truncated((size, n), 1.0, 0.3, 0.0, 1.0) * sign((size, n))
    locations = tf.argsort(tf.random.uniform((size, n)), axis=1)[:, :A]
    n1 = (size + 1) // 2
    mags1 = truncated((n1, A), D, 0.3 * D, D, 2 * D)
    mags1 = tf.where(mags1 > D, mags1, tf.constant(np.nextafter(D, 2 * D), f64))   # (Delta, 2 Delta]
    draw2 = lambda: truncated((size - n1, A), D, 0.3 * D, 0.0, 2 * D)
    mags2 = tf.while_loop(lambda m: tf.reduce_any(tf.reduce_max(m, 1) <= D),
                          lambda m: [tf.where(tf.reduce_max(m, 1, keepdims=True) <= D, draw2(), m)], [draw2()])[0]
    mags2 = tf.where(mags2 > 0, mags2, tf.constant(np.nextafter(0.0, 1.0), f64))
    values = tf.concat([mags1, mags2], 0) * sign((size, A))
    e = tf.einsum("ba,ban->bn", values, tf.one_hot(locations, n, dtype=f64))
    y = u @ tf.constant(C["G"], f64) + eps + e
    s = y @ tf.constant(C["H"].T / D, f64)
    return tf.cast(s, tf.float32), e


if __name__ == "__main__":
    reference = {1: dict(delta=1.732050808, Delta=18.138271538, Y=54.414814613),
                 2: dict(delta=1.183012702, Delta=9.464101615, Y=28.392304845)}
    rounded_G1 = np.array([[1.00, 1.00, 1.00, -1.00, 0.00, 0.00, 0.62, -0.62, 1.62, -1.62],
                           [1.00, 1.00, -1.00, 1.00, 1.62, 1.62, 0.00, 0.00, 0.62, 0.62],
                           [1.00, -1.00, 1.00, 1.00, 0.62, -0.62, 1.62, 1.62, 0.00, 0.00]])
    assert np.array_equal(np.round(G1, 2), rounded_G1), "G1 does not round to the matrix in the description"
    assert np.allclose(np.linalg.norm(G1, axis=0), np.sqrt(3))
    assert np.allclose(np.linalg.norm(G2, axis=0), [np.sqrt(3)] * 4 + [1] * 12)
    assert abs(CODES[1]["delta"] - np.sqrt(3)) < 1e-15 and abs(CODES[2]["delta"] - (3 + np.sqrt(3)) / 4) < 1e-15
    for code, C in CODES.items():
        if code in HARMONIC_A:
            print(f"Code {code}: k={C['k']} n={C['n']} r={C['r']} A={C['A']} delta={C['delta']} Delta={C['Delta']!r}")
            continue
        for name, value in reference[code].items():
            assert abs(C[name] - value) < 1e-8, (code, name, C[name], value)
        print(f"Code {code}: k={C['k']} n={C['n']} delta={C['delta']:.9f} Delta={C['Delta']:.9f} Y={C['Y']:.9f}")
    print("All constants match the project description.")
