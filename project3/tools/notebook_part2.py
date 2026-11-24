"""Part II of the V4 notebook (codes 3-6 of the updated project), used by tools/build_notebook.py.

The code cells are ported from codes.py (harmonic codes, samplers) and train36.py (networks, training), with the same
functions and seeds. part2_cells(md, code, tag, versions) returns the notebook cells of sections 10 to 16.
"""

RECIPES36 = {
    "H2": '    # H2: transformer over the location tokens (width 128, 6 blocks, 8 heads), plus the value head; all four codes.\n'
          '    "H2": dict(width=128, blocks=6, heads=8, lr=5e-4, aux=1.0, epochs={3: 60, 4: 120, 5: 150, 6: 240}),',
    "H3": '    # H3: the same with width 192, for the two hardest codes.\n'
          '    "H3": dict(width=192, blocks=6, heads=8, lr=4e-4, aux=1.0, epochs={5: 90, 6: 150}),',
}


def part2_cells(md, code, tag, versions):
    cells = []
    cells.append(md(r"""# Part II: Codes 3 to 6

## 10. The Four Codes, Their Structure and Their Noisy-Codeword Distribution

The file `codes_3_to_6.pkl` provides $G$ and $\Delta$ for codes 3 to 6. `sample_harmonic` draws noisy codewords with
NumPy in float64 exactly as in the project description (rejection sampling, both sample groups, joint shuffle; for the
evaluation sets). `tf_sample_harmonic` draws the same distribution with TensorFlow in float64 on the GPU (truncated
normals by the inverse CDF) and returns the syndrome, for the training batches.

**Structure.** Every row of these four matrices is a cosine or sine of an *odd* harmonic $m = 1, 3, \dots, k-1$
sampled at $\theta_j = \pi j / n$ (scaled so that every column has length 1); the cell below checks this to $10^{-12}$.
The odd harmonics that $G$ leaves out, $m = k+1, \dots, n-1$, give a parity-check matrix $H$ ($r \times n$, $r = n - k$,
orthonormal rows, $GH^T = 0$). The **syndrome** $s = yH^T / \Delta$ therefore does not depend on $u$:
$s = (\epsilon + e)H^T / \Delta$. It is the input of the networks. It is one matrix multiplication, done in float64,
because $y$ is about $10^6$ times larger than the noise for code 6, and float32 rounding of $uG$ would swamp $\epsilon$.

**Location frames.** `ROTATIONS[j]` rotates every (cos, sin) pair of $s$ by $-m\theta_j$; then an error at location
$j$ alone gives the same pattern $(1, 0, 1, 0, \dots)$ in the frame of location $j$, wherever $j$ is, and the frame of
$j$ depends only on the error positions relative to $j$. The network receives $s$ in all $n$ frames (one token per
location; again a fixed matrix multiplication)."""))

    cells.append(code(r'''HARMONIC_A = {3: 4, 4: 5, 5: 10, 6: 20}               # fixed number of UMEs per codeword
DESCRIPTION_36 = {3: (16, 8, 2773.9396039298235), 4: (20, 10, 25631.144050459065), 5: (40, 20, 32768.0), 6: (80, 40, 262144.0)}
codes_file = find_file("codes_3_to_6.pkl")
assert codes_file is not None, "codes_3_to_6.pkl (provided with the project) must be in the notebook folder or data/"
ENTRIES_36 = load_pickle(codes_file)


def harmonic_code(code, entry):
    G = np.asarray(entry["G"], dtype=np.float64); k, n = G.shape
    theta = np.pi * np.arange(n) / n
    # G = the odd harmonics 1, 3, ..., k-1 (a cos row and a sin row each) at theta_j, scaled to unit-length columns.
    harmonic = np.array([f(m * theta) for m in range(1, k, 2) for f in (np.cos, np.sin)]) / np.sqrt(k / 2)
    assert np.abs(G - harmonic).max() < 1e-12, f"code {code}: G is not the odd-harmonic matrix"
    # H = the odd harmonics k+1, ..., n-1 that G leaves out: orthonormal rows, G H^T = 0.
    freqs = np.arange(k + 1, n, 2)
    H = np.array([f(m * theta) for m in freqs for f in (np.cos, np.sin)]) / np.sqrt(n / 2)
    assert np.abs(G @ H.T).max() < 1e-12 and np.allclose(H @ H.T, np.eye(n - k))
    # ROTATIONS[j] (r x r): rotates every (cos, sin) pair of frequency m by -m theta_j, so ROTATIONS[j] @ H[:, j] = H[:, 0].
    rot = np.zeros((n, n - k, n - k))
    for p, m in enumerate(freqs):
        c, s = np.cos(m * theta), np.sin(m * theta)
        rot[:, 2 * p, 2 * p], rot[:, 2 * p, 2 * p + 1] = c, s
        rot[:, 2 * p + 1, 2 * p], rot[:, 2 * p + 1, 2 * p + 1] = -s, c
    assert np.allclose(np.einsum("jqr,rj->jq", rot, H), H[:, :1].T)
    delta = float(np.linalg.norm(G, axis=0).mean())          # average column length: 1
    return dict(G=G, k=k, n=n, r=n - k, delta=delta, Delta=float(entry["Delta"]), A=HARMONIC_A[code], H=H, ROTATIONS=rot)


for code in HARMONIC_A:
    CODES[code] = harmonic_code(code, ENTRIES_36[f"code_{code}"])


def truncated_normal(rng, mean, sd, shape, low, high, low_open):
    """Rejection sampling as described: draw N(mean, sd) and redraw values outside [low, high] ((low, high] if
    low_open) until all lie inside."""
    x = rng.normal(mean, sd, shape)
    bad = ~(((x > low) if low_open else (x >= low)) & (x <= high))
    while bad.any():
        x[bad] = rng.normal(mean, sd, int(bad.sum()))
        bad[bad] = ~(((x[bad] > low) if low_open else (x[bad] >= low)) & (x[bad] <= high))
    return x


def sample_harmonic(code, size, rng, return_group=False):
    """Codes 3-6: `size` noisy codewords exactly as described (both groups, shuffled jointly). Returns y and e
    (size, n), float64, and with return_group the group of every sample (1 = Group I, 2 = Group II)."""
    C = CODES[code]; n, k, A, D = C["n"], C["k"], C["A"], C["Delta"]
    u = rng.uniform(-2 * D, 2 * D, (size, k))
    eps = truncated_normal(rng, 1.0, 0.3, (size, n), 0.0, 1.0, False) * rng.choice((-1.0, 1.0), (size, n))
    locations = np.argsort(rng.random((size, n)), axis=1)[:, :A]         # A distinct, uniformly chosen locations
    n1 = (size + 1) // 2                                                 # ceil(size / 2) samples in Group I
    mags = np.empty((size, A))
    mags[:n1] = truncated_normal(rng, D, 0.3 * D, (n1, A), D, 2 * D, True)
    mags[n1:] = truncated_normal(rng, D, 0.3 * D, (size - n1, A), 0.0, 2 * D, True)
    redo = n1 + np.flatnonzero(mags[n1:].max(axis=1) <= D)               # Group II: none above Delta -> redraw all A
    while len(redo):
        mags[redo] = truncated_normal(rng, D, 0.3 * D, (len(redo), A), 0.0, 2 * D, True)
        redo = redo[mags[redo].max(axis=1) <= D]
    e = np.zeros((size, n))
    np.put_along_axis(e, locations, mags * rng.choice((-1.0, 1.0), (size, A)), axis=1)
    y = u @ C["G"] + eps + e
    group = np.where(np.arange(size) < n1, 1, 2)
    order = rng.permutation(size)                                        # joint shuffle of both groups
    return (y[order], e[order], group[order]) if return_group else (y[order], e[order])


def syndrome(code, y):
    """s = y H^T / Delta in float64 (one matrix multiplication); independent of u."""
    return np.asarray(y, np.float64) @ CODES[code]["H"].T / CODES[code]["Delta"]


def tf_sample_harmonic(code, size):
    """The same distribution in TensorFlow, everything in float64 (truncated normals by the inverse CDF, the same
    distribution as rejection sampling; Group II sets without a magnitude above Delta are redrawn). Returns the
    syndrome s = y H^T / Delta as float32 and e as float64, for the training batches."""
    C = CODES[code]; n, k, A, D = C["n"], C["k"], C["A"], C["Delta"]
    f64 = tf.float64
    phi = lambda z: 0.5 * tf.math.erfc(-z / np.sqrt(2.0))

    def truncated(shape, mean, sd, low, high):
        a, b = phi(tf.constant((low - mean) / sd, f64)), phi(tf.constant((high - mean) / sd, f64))
        p = b - tf.random.uniform(shape, dtype=f64) * (b - a)           # in (a, b]
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


for code in HARMONIC_A:
    C = CODES[code]
    assert (C["n"], C["k"], C["Delta"]) == DESCRIPTION_36[code] and abs(C["delta"] - 1) < 1e-12
    print(f"Code {code}: n = {C['n']}, k = {C['k']}, r = {C['r']}, A = {C['A']}, delta = {C['delta']:.12f}, "
          f"Delta = {C['Delta']!r}; G = odd harmonics 1..{C['k'] - 1}, H = odd harmonics {C['k'] + 1}..{C['n'] - 1}")
print("All matrices, thresholds and UME counts match the project description.")
'''))

    cells.append(md(r"""### 10.1 Checks of the sampler and the provided example files

For each code: 200,000 codewords from `sample_harmonic` (Group I share, magnitudes inside their intervals, at least
one required UME per Group II sample, the noise $s - eH^T/\Delta$ at the level of $\epsilon$), the same statistics from
`tf_sample_harmonic`, and a comparison with the 1,000 provided example inputs: quantiles of $|y_j|$ and of the
syndrome norm $\|s\|$. The example outputs pass `check_output`; half of them (Group I) have exactly $A$ locations.
"""))

    cells.append(code(r'''for code in HARMONIC_A:
    C = CODES[code]; n, A, D = C["n"], C["A"], C["Delta"]
    y, e, g = sample_harmonic(code, 200_000, np.random.default_rng([3600, code]), return_group=True)
    mags = np.abs(e[e != 0]).reshape(len(e), A) / D
    s = syndrome(code, y)
    noise = s - e @ C["H"].T / D
    assert ((e != 0).sum(1) == A).all() and np.mean(g == 1) == 0.5
    assert (mags[g == 1] > 1).all() and (mags[g == 1] <= 2).all() and (mags[g == 2] > 0).all() and (mags <= 2).all()
    assert (mags[g == 2].max(1) > 1).all()
    s_tf, e_tf = tf_sample_harmonic(code, 200_000)
    m_tf = np.abs(e_tf.numpy()[e_tf.numpy() != 0]) / D
    print(f"Code {code}: Group II share with |e| > Delta: {np.mean(mags[g == 2] > 1):.4f} (TensorFlow sampler: "
          f"{2 * np.mean(m_tf > 1) - 1:.4f}); mean |e| / Delta {mags.mean():.4f} (TensorFlow {m_tf.mean():.4f}); "
          f"noise in s: std {noise.std() * D:.3f} / Delta")
    inputs = find_file(f"Noisy-Codeword-Examples-for-Code-{code}.pkl")
    outputs = find_file(f"Decode-results-examples-for-code-{code}.pkl")
    if inputs is None or outputs is None:
        print("  example files not found (optional)"); continue
    X, S = np.asarray(load_pickle(inputs), dtype=np.float64), load_pickle(outputs)
    assert X.shape == (len(X), n) and np.isfinite(X).all()
    check_output(S, len(X), n)
    EXAMPLES[code] = (X, S)
    q = [0.1, 0.5, 0.9, 0.99]
    print(f"  {len(X)} examples; outputs pass the format check; outputs with exactly A = {A} locations: "
          f"{np.mean([len(x) == A for x in S]):.3f}")
    print(f"  |y_j| / Delta quantiles {q}: examples {np.round(np.quantile(np.abs(X) / D, q), 3).tolist()}, "
          f"sampled {np.round(np.quantile(np.abs(y) / D, q), 3).tolist()}")
    print(f"  ||s|| quantiles:              examples {np.round(np.quantile(np.linalg.norm(syndrome(code, X), axis=1), q), 3).tolist()}, "
          f"sampled {np.round(np.quantile(np.linalg.norm(s, axis=1), q), 3).tolist()}")
'''))

    cells.append(md(r"""## 11. The Validation and Test Sets (Codes 3 to 6)

As in Part I: one fixed set of 1,000,000 validation and one of 1,000,000 test codewords per code, drawn with
`sample_harmonic` (the exact two-group mixture) from seeds used nowhere else; training draws fresh batches with
TensorFlow. The networks receive the syndromes (computed in float64, passed as float32); the final models (section 16)
take $y$ itself.
"""))

    cells.append(code(r'''EVAL36 = {}
for code in CODES_36:
    entry = {}
    for split, seed in (("val", 3601), ("test", 3602)):
        y, e, g = sample_harmonic(code, EVAL_SIZE, np.random.default_rng([seed, code]), return_group=True)
        entry[f"s_{split}"], entry[f"e_{split}"], entry[f"g_{split}"] = syndrome(code, y).astype(np.float32), e, g
        if split == "test":
            entry["y_test"] = y
    EVAL36[code] = entry
    C = CODES[code]; e = entry["e_test"]
    required = (np.abs(e) > C["Delta"]).sum(1); optional = ((e != 0) & (np.abs(e) <= C["Delta"])).sum(1)
    print(f"Code {code}: {EVAL_SIZE:,} validation and {EVAL_SIZE:,} test codewords | test: Group I share "
          f"{np.mean(entry['g_test'] == 1):.3f}; mean required {required.mean():.3f}, optional {optional.mean():.3f} locations per codeword")
'''))

    cells.append(md(r"""## 12. The Network Architecture (Codes 3 to 6)

Input: the syndrome $s$ ($r$ numbers, in units of $\Delta$). Output: $n$ logits ("report location $j$").

1. **Location tokens** (fixed, not trained): token $j$ = `ROTATIONS[j]` $s$, the syndrome in the frame of location $j$
   ($n$ tokens of $r$ numbers; one matrix multiplication). An isolated UME at $j$ shows up in token $j$ as the
   constant pattern $(1, 0, 1, 0, \dots) \cdot e_j \sqrt{2/n}$; UMEs elsewhere add rotated patterns that depend only
   on their offset from $j$.
2. **Embedding:** a shared linear layer to the width, a learned per-location bias (location embedding), and a GELU
   layer.
3. **Six transformer blocks** (pre-normalised multi-head self-attention between the $n$ tokens and a feed-forward layer
   of four times the width). Whether location $j$ holds a UME depends on how well the rest of the syndrome can be
   explained by UMEs elsewhere; attention lets every location compare its view with all the others, which is the
   "explaining away" that clusters of adjacent UMEs require.
4. **Heads:** a shared dense layer gives every token its logit. A second dense layer (the value head) estimates
   $e_j / \Delta$ at every location; it is used only for an auxiliary training loss (section 13), and the final
   models output only the logits.

**Versions:** H2 (width 128, about 1.2 million parameters) for all four codes; for codes 5 and 6, where the networks
are furthest from perfect decoding, also H3 (width 192, about 2.7 million parameters, trained for about as long as
H2). The number of training epochs grows with the difficulty of the code; the final model of each code is chosen on
the validation set (section 15).

**Development.** Shorter runs of the same code (`train36.py`; 200,000 validation codewords) compared this design
with two refinement networks that feed the syndrome left unexplained by the current value estimates (a fixed matrix
multiplication) back into later blocks, with fixed (R1) or growing, additive (R4) amplification. They were not better:

| Code 5, validation accuracy | epoch 7 | epoch 11 | epoch 17 |
|---|---|---|---|
| H2 (45-epoch schedule) | 0.776 | 0.881 | 0.935 |
| R1 refinement (45-epoch schedule) | 0.687 | 0.731 | 0.792 |
| R4 additive refinement (30-epoch schedule) | 0.688 | 0.805 | (stopped) |

The fixed layers are matrix multiplications with constant matrices derived from $G$; all decisions are made by the
trained layers.
"""))

    cells.append(code(r'''VERSIONS36 = {
''' + "\n".join(RECIPES36[v] for v in versions) + r'''
}


def recipe36(version, code):
    """The recipe of a version for one code (the number of epochs depends on the code); None if the version is not
    used for that code."""
    R = dict(VERSIONS36[version])
    if code not in R["epochs"]:
        return None
    R["epochs"] = EPOCHS if EPOCHS is not None else R["epochs"][code]
    return R


def transformer_block(h, width, heads, name):
    z = tf.keras.layers.LayerNormalization(name=f"{name}_attention_norm")(h)
    z = tf.keras.layers.MultiHeadAttention(heads, width // heads, name=f"{name}_attention")(z, z)
    h = tf.keras.layers.Add()([h, z])
    z = tf.keras.layers.LayerNormalization(name=f"{name}_ffn_norm")(h)
    z = tf.keras.layers.Dense(4 * width, activation="gelu", name=f"{name}_ffn_hidden")(z)
    return tf.keras.layers.Add()([h, tf.keras.layers.Dense(width, name=f"{name}_ffn_out")(z)])


def fixed_layer(equation, output_shape, kernel, name):
    """A linear layer with a constant (not trained) kernel."""
    layer = tf.keras.layers.EinsumDense(equation, output_shape=output_shape, name=name, trainable=False)
    layer.build((None, kernel.shape[0]))
    layer.set_weights([kernel])
    return layer


def build_network36(code, width, blocks, heads, aux=0.0, logits_only=False, **_):
    """Input: syndrome s (r numbers). Output: n logits, followed by n value estimates (e_j / Delta) when aux > 0
    (training), or only the logits with logits_only (final models)."""
    C = CODES[code]; n, r = C["n"], C["r"]
    inp = tf.keras.Input(shape=(r,), name="syndrome")
    # Fixed (not trained): token j = ROTATIONS[j] @ s, the syndrome seen from location j.
    x = fixed_layer("br,rjq->bjq", (n, r), np.transpose(C["ROTATIONS"], (2, 0, 1)), "frames")(inp)
    # Shared linear embedding of every token plus a learned per-location bias (location embedding).
    h = tf.keras.layers.EinsumDense("bjq,qd->bjd", output_shape=(n, width), bias_axes="jd", name="embed")(x)
    h = tf.keras.layers.Dense(width, activation="gelu", name="embed_mlp")(h)
    for b in range(blocks):
        h = transformer_block(h, width, heads, f"block{b}")
    h = tf.keras.layers.LayerNormalization(name="final_norm")(h)
    logits = tf.keras.layers.Reshape((n,), name="logits")(tf.keras.layers.Dense(1, name="token_logit")(h))
    if aux == 0 or logits_only:
        return tf.keras.Model(inp, logits, name=f"ume_locator_code{code}")
    value = tf.keras.layers.Reshape((n,), name="value")(tf.keras.layers.Dense(1, name="token_value")(h))
    return tf.keras.Model(inp, tf.keras.layers.Concatenate(name="logits_and_values")([logits, value]),
                          name=f"ume_locator_code{code}")


for version in VERSIONS36:
    for code in HARMONIC_A:
        if recipe36(version, code) is not None:
            net = build_network36(code, **recipe36(version, code))
            print(f"{version} code {code}: {net.count_params():,} parameters "
                  f"({sum(int(np.prod(w.shape)) for w in net.trainable_weights):,} trained), "
                  f"{recipe36(version, code)['epochs']} epochs")
'''))

    cells.append(md(r"""## 13. Training (Codes 3 to 6)

- **Loss:** the masked binary cross-entropy of Part I on the logits (target 1 at required, 0 at forbidden locations,
  optional locations masked out), plus `aux` times the mean squared error of the value estimates ($e_j / \Delta$ at
  every location, averaged over the stages). The value heads make the network track every UME's value, which the
  refinement stages need.
- **Data:** every batch of 4,096 codewords is drawn anew by `tf_sample_harmonic` (float64 on the GPU), half Group I and
  half Group II, as in the description.
- **Optimiser, schedule and validation** as in Part I (AdamW, warm-up and cosine decay; after every epoch the first
  200,000 validation codewords are decoded and the epoch with the fewest errors is kept), with 1,000 steps per epoch
  and a number of epochs that grows with the difficulty of the code. Seed 636 + code.
- **Resumable:** the training state is backed up after every epoch (`tf.keras.callbacks.BackupAndRestore`), so an
  interrupted run continues where it stopped when the notebook is run again.
"""))

    cells.append(code(r'''VAL_EPOCH_SIZE = min(200_000, EVAL_SIZE)        # validation codewords checked after every epoch


def make_loss36(n, aux):
    def loss(y_true, y_pred):
        target, mask, value = y_true[:, :n], y_true[:, n:2 * n], y_true[:, 2 * n:]
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=y_pred[:, :n])
        total = tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
        if aux > 0:
            stages = y_pred.shape[1] // n - 1
            for t in range(stages):
                total += aux / stages * tf.reduce_mean(tf.square(y_pred[:, n * (t + 1):n * (t + 2)] - value))
        return total
    return loss


def make_training_batch36(code):
    Delta = CODES[code]["Delta"]

    def training_batch(_):
        s, e = tf_sample_harmonic(code, BATCH)
        required = tf.cast(tf.abs(e) > Delta, tf.float32)
        optional = tf.cast((e != 0) & (tf.abs(e) <= Delta), tf.float32)
        return s, tf.concat([required, 1.0 - optional, tf.cast(e / Delta, tf.float32)], axis=1)
    return training_batch


_LOGIT_FUNCTIONS = {}                                     # one compiled function per network (kept with the network)


def logits36(network, code, s):
    """The network's logits (float32, GPU if available) for syndromes s; the value outputs are dropped."""
    if id(network) not in _LOGIT_FUNCTIONS:
        _LOGIT_FUNCTIONS[id(network)] = (network, tf.function(lambda b: network(b, training=False), jit_compile=True))
    f = _LOGIT_FUNCTIONS[id(network)][1]
    out = np.concatenate([np.asarray(f(tf.constant(s[i:i + 8192], tf.float32))) for i in range(0, len(s), 8192)])
    return out[:, :CODES[code]["n"]]


class Validate36(tf.keras.callbacks.Callback):
    """Validation after every epoch; keeps the epoch with the fewest wrong codewords (ties: lower validation loss).
    The history is written after every epoch, so an interrupted run can resume it."""

    def __init__(self, code, recipe, s, e, g, path, history_path):
        super().__init__(); self.code, self.recipe, self.s, self.e, self.g = code, recipe, s, e, g
        self.path, self.history_path = path, history_path
        self.history = json.loads(history_path.read_text())["epochs"] if history_path.exists() else []
        self.best = min(((r["val_errors"], r["val_loss"]) for r in self.history), default=(np.inf, np.inf))
        self.start = time.time() - 60 * (self.history[-1]["minutes"] if self.history else 0)

    def on_epoch_end(self, epoch, logs=None):
        logits = logits36(self.model, self.code, self.s)
        ok = correct(self.code, self.e, logits > 0); errors = int((~ok).sum()); loss = bce_np(self.code, logits, self.e)
        lr = float(np.asarray(self.model.optimizer.learning_rate))
        row = dict(epoch=epoch + 1, loss=float(logs["loss"]), val_loss=loss, val_errors=errors,
                   val_errors_group1=int((~ok[self.g == 1]).sum()), val_errors_group2=int((~ok[self.g == 2]).sum()),
                   lr=lr, minutes=(time.time() - self.start) / 60)
        self.history = [r for r in self.history if r["epoch"] < epoch + 1] + [row]
        improved = (errors, loss) < self.best
        if improved:
            self.best = (errors, loss); self.model.save(self.path)
        self.history_path.write_text(json.dumps(dict(recipe=self.recipe, steps=STEPS36, batch=BATCH, epochs=self.history), indent=1))
        print(f"epoch {epoch + 1:3d}  loss {row['loss']:.3e}  val_loss {loss:.3e}  val_errors {errors:7d} "
              f"({errors / len(self.s):.2e}; Group I {row['val_errors_group1']}, II {row['val_errors_group2']})  "
              f"lr {lr:.2e}  {row['minutes']:.1f} min{'  *' if improved else ''}", flush=True)


def model_path36(version, code):
    return OUT / f"model_{version}_code{code}.keras"


def train_network36(version, code):
    R = recipe36(version, code); n = CODES[code]["n"]
    history_path = OUT / f"history_{version}_code{code}.json"
    backup_dir = OUT / f"backup_{version}_code{code}"
    done = len(json.loads(history_path.read_text())["epochs"]) if history_path.exists() else 0
    if done >= R["epochs"] and not backup_dir.exists():
        print(f"{version} code {code}: already trained ({R['epochs']} epochs)"); return
    # Seed 636 + code; a resumed run (after an interruption) draws a new stream of training batches.
    tf.keras.utils.set_random_seed(636 + code + (1009 * done if backup_dir.exists() else 0))
    network = build_network36(code, **R)
    steps = R["epochs"] * STEPS36; warmup = min(2000, steps // 10)
    lr = tf.keras.optimizers.schedules.CosineDecay(R["lr"] * 0.01, steps - warmup, alpha=0.01, warmup_target=R["lr"],
                                                   warmup_steps=warmup)
    network.compile(optimizer=tf.keras.optimizers.AdamW(lr, weight_decay=1e-5, beta_2=0.95, clipnorm=1.0),
                    loss=make_loss36(n, R["aux"]))
    print(f"{version} code {code}: {R} | {STEPS36} steps x {BATCH} codewords per epoch", flush=True)
    data = tf.data.Dataset.range(1).repeat().map(make_training_batch36(code), num_parallel_calls=tf.data.AUTOTUNE).prefetch(8)
    E = EVAL36[code]; m = VAL_EPOCH_SIZE                  # the first 200,000 validation codewords after every epoch
    val = Validate36(code, R, E["s_val"][:m], E["e_val"][:m], E["g_val"][:m], model_path36(version, code), history_path)
    backup = tf.keras.callbacks.BackupAndRestore(str(backup_dir))
    network.fit(data, epochs=R["epochs"], steps_per_epoch=STEPS36, verbose=0,
                callbacks=[val, backup, tf.keras.callbacks.TerminateOnNaN()])


NETWORKS36 = {}
for version in VERSIONS36_TO_RUN:
    for code in CODES_36:
        if recipe36(version, code) is None:              # this version is not used for this code
            continue
        if TRAIN:
            train_network36(version, code)
        if model_path36(version, code).exists():
            NETWORKS36[version, code] = tf.keras.models.load_model(model_path36(version, code), compile=False)
            print(f"Loaded {model_path36(version, code)}")
        else:
            print(f"{model_path36(version, code)} not found: train it with P3_TRAIN=1")
'''))

    cells.append(md(r"""## 14. Training Curves (Codes 3 to 6)

Training loss (masked BCE plus the auxiliary value loss, on the current batches), validation BCE and the number of
wrongly decoded validation codewords (out of 1,000,000) after every epoch.
"""))

    cells.append(code(r'''HISTORIES36 = {}
for version in VERSIONS36_TO_RUN:
    for code in CODES_36:
        path = OUT / f"history_{version}_code{code}.json"
        if path.exists():
            HISTORIES36[version, code] = json.loads(path.read_text())["epochs"]

if HISTORIES36:
    fig, axes = plt.subplots(len(CODES_36), 2, figsize=(13, 4.2 * len(CODES_36)), squeeze=False)
    for row, code in enumerate(CODES_36):
        for version in VERSIONS36_TO_RUN:
            if (version, code) not in HISTORIES36:
                continue
            h = HISTORIES36[version, code]; epochs = [r["epoch"] for r in h]
            line, = axes[row, 0].plot(epochs, [r["loss"] for r in h], "--", alpha=0.7, label=f"{version} training (BCE + aux)")
            axes[row, 0].plot(epochs, [r["val_loss"] for r in h], color=line.get_color(), label=f"{version} validation BCE")
            axes[row, 1].plot(epochs, [max(r["val_errors"], 0.5) for r in h], "o-", ms=3, color=line.get_color(), label=version)
        axes[row, 0].set(yscale="log", xlabel="epoch", ylabel="loss", title=f"Code {code}: loss")
        axes[row, 1].set(yscale="log", xlabel="epoch", ylabel="wrong codewords (of 1M)", title=f"Code {code}: validation errors")
        for ax in axes[row]:
            ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.show()
    for (version, code), h in HISTORIES36.items():
        best = min(h, key=lambda r: (r["val_errors"], r["val_loss"]))
        print(f"{version} code {code}: {len(h)} epochs in {h[-1]['minutes']:.0f} min; best epoch {best['epoch']} "
              f"(val errors {best['val_errors']:,} = {best['val_errors'] / EVAL_SIZE:.2e}, val BCE {best['val_loss']:.2e})")
'''))

    cells.append(md(r"""## 15. Validation and Test Performance (Codes 3 to 6)

**Symmetry averaging.** These codes have exact symmetries that map codewords to codewords, UMEs to UMEs at moved
locations, and leave the distribution of the noisy codewords unchanged:

- the negacyclic **shift** ($y_j \to y_{j-1}$, $y_0 \to -y_{n-1}$; location $j \to j + 1$),
- the **reflection** ($y_0 \to y_0$, $y_j \to -y_{n-j}$; location $j \to n - j$), since the cosines are even and the
  sines odd,
- the **sign** change $y \to -y$ (locations unchanged).

The networks are trained on every shift equally often, but nothing makes them exactly symmetric, and their errors on
the transformed copies of a codeword are partly independent. The final decoders therefore average the logits over
several transformed copies ("views"; each view's logits are moved back to the original locations). In the syndrome
domain every view is an orthogonal matrix $M_v = H V H^T$ ($V$ the signed permutation that transforms $y$), and
moving the logits back is a permutation, so a decoder is the same trained network applied to fixed linear transforms
of its input, with the outputs averaged. Tests on development networks showed that reflection and sign change help
much more than more shifts (code 6: one view 73.3%, four shifts 76.1%, the eight views below 78.3%).

- **Single networks:** 8 views, the shifts by $0$ and $n/2$, each with and without reflection, each with both signs.
- **Ensembles:** where two versions are trained (codes 5 and 6), the two networks together with 4 views each (with
  and without reflection, both signs), so that every candidate costs 8 network evaluations per codeword.

For every candidate the threshold is chosen on the validation set as in Part I, and the final model of a code is the
candidate with the fewest validation errors (ties: fewer parameters). The test set is decoded once. The test accuracy
of every single network *without* averaging is shown for comparison.
"""))

    cells.append(code(r'''VIEWS_SINGLE = [(reflect, shift, sign) for reflect in (False, True) for shift in (0, 1 / 2) for sign in (1, -1)]
VIEWS_PAIR = [(reflect, 0, sign) for reflect in (False, True) for sign in (1, -1)]
# Ensembles of the m best single networks with 8 / m views each (VIEWS_PAIR within VIEWS_SINGLE).
ENSEMBLES = {"pair": (2, VIEWS_PAIR)}


def view_matrices(code, views):
    """For each view (reflect, shift as a fraction of n, sign): M (r x r) with syndrome(view of y) = s M, and the
    location map loc (original location i is location loc[i] of the view)."""
    C = CODES[code]; n, H = C["n"], C["H"]
    reflection = np.zeros((n, n)); reflection[0, 0] = 1.0; reflection[n - np.arange(1, n), np.arange(1, n)] = -1.0
    out = []
    for reflect, shift, sign in views:
        k = round(shift * n)
        P = np.zeros((n, n)); P[np.arange(n - k), np.arange(k, n)] = 1.0; P[np.arange(n - k, n), np.arange(k)] = -1.0
        V = sign * ((reflection if reflect else np.eye(n)) @ P)              # y -> y V
        loc = (((n - np.arange(n)) % n if reflect else np.arange(n)) + k) % n
        out.append((H @ V @ H.T, loc))
    return out


def averaged_logits36(networks, code, s, views):
    """Logits averaged over the networks and the views, each view's logits moved back to the original locations."""
    total = 0.0
    for M, loc in view_matrices(code, views):
        sv = (s.astype(np.float64) @ M).astype(np.float32)
        for network in networks:
            total = total + logits36(network, code, sv)[:, loc]
    return total / (len(networks) * len(views))


def choose_threshold(code, e, logits):
    errs = np.array([(~correct(code, e, logits > t)).sum() for t in GRID])
    best = GRID[errs == errs.min()]
    return float(best[len(best) // 2]), int(errs.min())


def view_logits36(version, code):
    """For the validation and test sets: the network's logits without averaging (the first view is the identity), and
    its averages over all VIEWS_SINGLE ("single") and over the views of each ensemble ("pair")."""
    E, net = EVAL36[code], NETWORKS36[version, code]
    groups = {"single": VIEWS_SINGLE, **{key: views for key, (m, views) in ENSEMBLES.items()}}
    out = {}
    for split in ("val", "test"):
        sums = {key: 0.0 for key in groups}
        for view, (M, loc) in zip(VIEWS_SINGLE, view_matrices(code, VIEWS_SINGLE)):
            lv = logits36(net, code, (E[f"s_{split}"].astype(np.float64) @ M).astype(np.float32))[:, loc]
            if view == (False, 0, 1):
                out[split, "plain"] = lv
            for key, views in groups.items():
                if view in views:
                    sums[key] = sums[key] + lv
        for key, views in groups.items():
            out[split, key] = sums[key] / len(views)
    return out


def summarize36(name, code, versions, views, lv, lt, plain=None):
    E = EVAL36[code]; ev, et, gt = E["e_val"], E["e_test"], E["g_test"]
    t, val_errors = choose_threshold(code, ev, lv)
    rep = lt > t; ok = correct(code, et, rep)
    test_plain = None
    if plain is not None:                        # the same network without averaging, for comparison
        t1, _ = choose_threshold(code, ev, plain[0])
        test_plain = float(correct(code, et, plain[1] > t1).mean())
    return dict(
        name=name, versions=versions, views=views, code=code,
        parameters=sum(NETWORKS36[v, code].count_params() for v in versions),
        threshold=t, val_errors=val_errors, val_accuracy=1 - val_errors / len(ev), test_errors=int((~ok).sum()),
        test_accuracy=float(ok.mean()), test_accuracy_plain=test_plain,
        test_accuracy_group1=float(ok[gt == 1].mean()), test_accuracy_group2=float(ok[gt == 2].mean()),
        test_missed_required=int(((np.abs(et) > CODES[code]["Delta"]) & ~rep).any(1).sum()),
        test_forbidden_reported=int((rep & (et == 0)).any(1).sum()))


# Every single network (8 views), then (codes with enough versions) the ensembles of the m best ones on the validation
# set (8 / m views each; their averages over those views are kept from the single-network pass).
RESULTS36, ENSEMBLE_LOGITS = {}, {}
for (version, code) in NETWORKS36:
    L = view_logits36(version, code)
    RESULTS36[version, code] = summarize36(version, code, [version], VIEWS_SINGLE, L["val", "single"], L["test", "single"],
                                           plain=(L["val", "plain"], L["test", "plain"]))
    for key in ENSEMBLES:
        ENSEMBLE_LOGITS[version, code, key] = (L["val", key], L["test", key])
    del L
for code in CODES_36:
    singles = sorted((r for (v, c), r in RESULTS36.items() if c == code), key=lambda r: (r["val_errors"], r["parameters"]))
    for key, (m, views) in ENSEMBLES.items():
        if len(singles) >= m:
            names = [r["name"] for r in singles[:m]]
            lv = sum(ENSEMBLE_LOGITS[v, code, key][0] for v in names) / m
            lt = sum(ENSEMBLE_LOGITS[v, code, key][1] for v in names) / m
            RESULTS36["+".join(names), code] = summarize36("+".join(names), code, names, views, lv, lt)
del ENSEMBLE_LOGITS

label_width = max(len(f"{name} code {code}") for name, code in RESULTS36) + 1 if RESULTS36 else 16
print(f"{'':{label_width}s} {'views':>5s} {'params':>10s} {'threshold':>9s} {'val accuracy':>12s} {'test accuracy':>13s} {'(no averaging)':>14s} "
      f"{'Group I':>9s} {'Group II':>9s} {'missed required':>15s} {'forbidden reported':>18s}")
for (name, code), r in sorted(RESULTS36.items(), key=lambda kv: (kv[0][1], len(kv[1]["versions"]), kv[0][0])):
    plain = "" if r["test_accuracy_plain"] is None else f"{r['test_accuracy_plain']:.6f}"
    print(f"{name + ' code ' + str(code):{label_width}s} {len(r['views']) * len(r['versions']):5d} {r['parameters']:10,d} {r['threshold']:9.2f} "
          f"{r['val_accuracy']:12.6f} {r['test_accuracy']:13.6f} {plain:>14s} {r['test_accuracy_group1']:9.6f} "
          f"{r['test_accuracy_group2']:9.6f} {r['test_missed_required']:15,d} {r['test_forbidden_reported']:18,d}")
print(f"({EVAL_SIZE:,} validation and test codewords per code; views = network evaluations per codeword; missed "
      "required / forbidden reported: test codewords with at least one such location, a codeword can have both)")

FINAL36 = {}
for code in CODES_36:
    candidates = [r for (name, c), r in RESULTS36.items() if c == code]
    if candidates:
        FINAL36[code] = min(candidates, key=lambda r: (r["val_errors"], r["parameters"]))
        print(f"Code {code}: final model {FINAL36[code]['name']} with {len(FINAL36[code]['views'])} views per network "
              f"(threshold {FINAL36[code]['threshold']:g}, test accuracy {FINAL36[code]['test_accuracy']:.6f})")
'''))

    cells.append(md(rf"""## 16. The Final Models (Codes 3 to 6)

As in Part I (threshold folded in, float64), with the symmetry averaging built in. The final model takes $y$ ($n$
numbers), computes the syndrome $s = yH^T / \Delta$ (a constant float64 matrix multiplication), applies the view
matrices $M_v$ (constant matrix multiplications), runs the chosen network(s) on each view (shared weights), moves each
output back to the original locations (a constant permutation matrix), averages them and subtracts the threshold. Only the
logits are used (no value heads). The cell saves `final_model_{tag}_code3.keras` to `final_model_{tag}_code6.keras`,
checks on the first 20,000 test codewords that the final models decode like the float32 networks, compares them with
the provided example outputs (which include some optional locations as well), and runs them on these 20,000 test
codewords written in the format of the test files. These models run on a GPU when one is available (float64 either
way), because the averaging multiplies the work by eight; like the final cell, the checks decode them in batches of
1,024 (codes 3 and 4), 512 (code 5) and 256 (code 6) codewords, so that they also fit smaller GPUs.
"""))

    cells.append(code(r'''FINAL_CHECK_36 = 20_000


def make_final_model36(code, r):
    C = CODES[code]; n = C["n"]
    tf.keras.config.set_dtype_policy("float64")
    try:
        nets = [build_network36(code, **recipe36(v, code), logits_only=True) for v in r["versions"]]
        named = [tf.keras.Model(net.input, net.output, name=f"{v}_code{code}") for v, net in zip(r["versions"], nets)]
        inp = tf.keras.Input(shape=(n,), dtype="float64", name="y")   # y is not rounded to float32
        s = fixed_layer("bi,iq->bq", (C["r"],), C["H"].T / C["Delta"], "syndrome")(inp)
        outputs = []
        for v, (M, loc) in enumerate(view_matrices(code, r["views"])):
            sv = fixed_layer("bq,qp->bp", (C["r"],), M, f"view_{v}")(s)
            back = np.zeros((n, n)); back[loc, np.arange(n)] = 1.0          # view location loc[i] -> location i
            for net in named:
                outputs.append(fixed_layer("bj,ji->bi", (n,), back, f"view_{v}_back_{net.name}")(net(sv)))
        mean = tf.keras.layers.Average(name="average")(outputs)
        out = tf.keras.layers.Rescaling(1.0, offset=-r["threshold"], name="minus_threshold")(mean)
        final = tf.keras.Model(inp, out, name=f"final_ume_locator_code{code}")
    finally:
        tf.keras.config.set_dtype_policy("float32")
    for v, net in zip(r["versions"], nets):         # copy the trained weights layer by layer (no value heads)
        trained = {layer.name: layer for layer in NETWORKS36[v, code].layers}
        for layer in net.layers:
            if layer.weights:
                layer.set_weights([w.astype(np.float64) for w in trained[layer.name].get_weights()])
    return final


FINAL_BATCH = {1: 4096, 2: 4096, 3: 1024, 4: 1024, 5: 512, 6: 256}     # codewords per call, as in the final cell


def final_outputs(final, y, code):
    """As the final cell: the final model's float64 outputs, in batches (GPU if available)."""
    b = FINAL_BATCH[code]
    out = [np.asarray(final(tf.constant(y[i:i + b], tf.float64), training=False)) for i in range(0, len(y), b)]
    return np.concatenate(out) if out else np.zeros((0, CODES[code]["n"]))


def decode36(final, y, code):
    """As the final cell: location reported when the final model's output is positive."""
    return final_outputs(final, y, code) > 0


for code, r in FINAL36.items():
    final = make_final_model36(code, r)
    final.save(final_model_path(code))
    final = tf.keras.models.load_model(final_model_path(code), compile=False)
    E = EVAL36[code]; m = FINAL_CHECK_36; y, e = E["y_test"][:m], E["e_test"][:m]
    start = time.time(); rep64 = decode36(final, y, code); seconds = time.time() - start
    rep32 = averaged_logits36([NETWORKS36[v, code] for v in r["versions"]], code, E["s_test"][:m], r["views"]) > r["threshold"]
    print(f"Code {code}: saved {final_model_path(code)} ({r['name']}, threshold {r['threshold']:g}, {len(final.layers)} layers); "
          f"accuracy on the first {m:,} test codewords {correct(code, e, rep64).mean():.6f} (float32 networks "
          f"{correct(code, e, rep32).mean():.6f}); decoded differently: {(rep64 != rep32).any(1).sum()}; "
          f"{m / seconds:,.0f} codewords/s")
    if code in EXAMPLES:
        X, S = EXAMPLES[code]
        ours = to_sets(decode36(final, X, code)); check_output(ours, len(X), CODES[code]["n"])
        print(f"  examples: identical to the example output {np.mean([a == b for a, b in zip(ours, S)]):.3f}, "
              f"subset of it {np.mean([a <= b for a, b in zip(ours, S)]):.3f}, superset {np.mean([a >= b for a, b in zip(ours, S)]):.3f}")
    heldout = OUT / f"Noisy-Codeword-Heldout-for-Code-{code}.pkl"
    with open(heldout, "wb") as file:
        pickle.dump([[float(v) for v in row] for row in y], file)
    X = np.asarray(load_pickle(heldout), dtype=np.float64)
    ours = to_sets(decode36(final, X, code)); check_output(ours, len(X), CODES[code]["n"])
    reported = np.zeros((len(X), CODES[code]["n"]), bool)
    for row, s in enumerate(ours):
        reported[row, list(s)] = True
    print(f"  {heldout.name}: {len(ours):,} outputs pass the format check, accuracy {correct(code, e, reported).mean():.6f}")
'''))

    cells.append(md(rf"""### 16.1 One File for the Six Final Models

The final models of the six codes (sections 9 and 16) are stored together in one file, `final_model_{tag}.keras`: a
Keras model whose six parts are the unchanged final models, named `code1` to `code6` (one input and one output per
code). The final cell loads this one file and takes the part of each code by its number (`get_layer("code3")`, for
instance), never by the input length. The cell reloads the file and checks, on the held-out codewords of every code,
that each part gives exactly the same float64 outputs as the separately saved final model.
"""))

    cells.append(code(r'''ALL_CODES = [1, 2, 3, 4, 5, 6]


def bundle_path():
    return OUT / f"final_model_{NOTEBOOK_TAG}.keras"


if sorted(CODES_TO_RUN) == ALL_CODES and all(final_model_path(c).exists() for c in ALL_CODES):
    finals = {c: tf.keras.models.load_model(final_model_path(c), compile=False) for c in ALL_CODES}
    inputs = [tf.keras.Input(shape=(CODES[c]["n"],), dtype="float64", name=f"y_code{c}") for c in ALL_CODES]
    parts = [tf.keras.Model(finals[c].inputs, finals[c].outputs, name=f"code{c}") for c in ALL_CODES]
    bundle = tf.keras.Model(inputs, [part(x) for part, x in zip(parts, inputs)], name=f"final_ume_locators_{NOTEBOOK_TAG}")
    bundle.save(bundle_path())
    bundle = tf.keras.models.load_model(bundle_path(), compile=False)
    print(f"Saved {bundle_path()} ({bundle_path().stat().st_size / 1e6:.1f} MB): parts "
          f"{[layer.name for layer in bundle.layers if isinstance(layer, tf.keras.Model)]}")
    for c in ALL_CODES:
        X = np.asarray(load_pickle(OUT / f"Noisy-Codeword-Heldout-for-Code-{c}.pkl"), dtype=np.float64)[:FINAL_CHECK_36]
        separate, part = final_outputs(finals[c], X, c), final_outputs(bundle.get_layer(f"code{c}"), X, c)
        print(f"Code {c}: part code{c} on {len(X):,} held-out codewords: outputs identical to {final_model_path(c).name}: "
              f"{np.array_equal(separate, part)}; decisions identical: {np.array_equal(separate > 0, part > 0)}")
    del finals, parts, bundle
else:
    print("One-file bundle skipped: it needs the final models of all six codes from this run.")
'''))
    return cells
