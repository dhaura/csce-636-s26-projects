#!/usr/bin/env python3
"""Train one network (version, code) for codes 3-6 on fresh noisy codewords drawn for every batch.

Input: the syndrome s = y H^T / Delta (r numbers; H is the orthonormal parity-check matrix of the code, so s does not
depend on u; one fixed matrix multiplication, done in float64 when the batch is drawn). Network: a fixed linear layer
turns s into one token per location (s in the frame of location j, codes.ROTATIONS), then a learned embedding, a stack
of transformer blocks and one logit per location ("report this location").
Loss: masked binary cross-entropy per location (target 1 at required locations |e| > Delta, 0 where e = 0, optional
locations masked out); with aux > 0, plus aux times the mean squared error of a second per-location output that
predicts e_j / Delta (a training-only head that helps the network keep track of every error's value).
Validation: a fixed set (seed [3601, code], the exact two-group mixture) after every epoch; the epoch with the fewest
wrongly decoded codewords is kept (ties: lower validation loss). At the end the logit threshold is chosen on
validation and accuracy is measured once on the test set (seed [3602, code]).

Configuration (environment): P3_CODE (3-6), P3_VERSION (key of VERSIONS), P3_RUN (artifacts/<run>), overrides
P3_EPOCHS, P3_STEPS, P3_BATCH, P3_LR, P3_WIDTH, P3_BLOCKS, P3_HEADS, P3_AUX, P3_VAL_SIZE, P3_TEST_SIZE.
Outputs in artifacts/<run>/: model_<V>_code<c>.keras, history_<V>_code<c>.json, results_<V>_code<c>.json.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codes import CODES, correct, sample_harmonic, syndrome, tf_sample_harmonic  # noqa: E402

VERSIONS = {
    # H1: transformer over location tokens (rotated syndromes), learned location embedding.
    "H1": dict(width=128, blocks=6, heads=8, lr=5e-4, epochs=60, aux=0.0),
    # H2: H1 with the auxiliary value head.
    "H2": dict(width=128, blocks=6, heads=8, lr=5e-4, epochs=60, aux=1.0),
    # H3: H2 with width 192.
    "H3": dict(width=192, blocks=6, heads=8, lr=4e-4, epochs=60, aux=1.0),
    # H5: H2 plus a support head (binary cross-entropy on e_j != 0 at every location, optional UMEs included), so that
    # small UMEs are located as well (the value loss barely notices them).
    "H5": dict(width=128, blocks=6, heads=8, lr=5e-4, epochs=60, aux=1.0, support=1.0),
    # H6: H2 without the learned location embedding: every location is treated alike (each token sees the syndrome in
    # its own frame), so the network is equivariant under shifts up to the sign of wrapped tokens.
    "H6": dict(width=128, blocks=6, heads=8, lr=5e-4, epochs=60, aux=1.0, pos=False),
    # R1: refinement network, 3 stages of 2 transformer blocks. After each stage a value head estimates e / Delta at
    # every location; a fixed linear layer (the same rotation/parity-check matrices) maps these estimates to the
    # syndrome they explain, and the next stage receives what is left unexplained (amplified by `gain`).
    "R1": dict(arch="refine", width=128, blocks=2, stages=3, heads=8, lr=5e-4, epochs=60, aux=1.0, gain=8.0),
    # R2: R1 with 4 stages (8 blocks).
    "R2": dict(arch="refine", width=128, blocks=2, stages=4, heads=8, lr=5e-4, epochs=60, aux=1.0, gain=8.0),
    # R3: R1 with width 192.
    "R3": dict(arch="refine", width=192, blocks=2, stages=3, heads=8, lr=4e-4, epochs=60, aux=1.0, gain=8.0),
    # R4: additive refinement with zoom. Stage 0 estimates the values v0; stage t >= 1 receives the unexplained rest
    # amplified by gain^t (bounded by tanh) and adds a correction scaled by 1 / gain^t, so the precision of the value
    # estimates can grow by about `gain` per stage. Value losses are Huber losses in each stage's own (zoomed) units.
    "R4": dict(arch="zoom", width=128, blocks=2, stages=3, heads=8, lr=5e-4, epochs=60, aux=1.0, gain=16.0),
}

CODE = int(os.environ.get("P3_CODE", "3"))
VERSION = os.environ.get("P3_VERSION", "H1")
RUN_DIR = Path("artifacts") / os.environ.get("P3_RUN", "h_dev")
R = dict(VERSIONS[VERSION])
for key, name, cast in (("epochs", "P3_EPOCHS", int), ("lr", "P3_LR", float), ("width", "P3_WIDTH", int),
                        ("blocks", "P3_BLOCKS", int), ("heads", "P3_HEADS", int), ("aux", "P3_AUX", float)):
    if name in os.environ:
        R[key] = cast(os.environ[name])
STEPS = int(os.environ.get("P3_STEPS", "1000"))
BATCH = int(os.environ.get("P3_BATCH", "4096"))
VAL_SIZE = int(os.environ.get("P3_VAL_SIZE", "200000"))
TEST_SIZE = int(os.environ.get("P3_TEST_SIZE", "1000000"))
C = CODES[CODE]; N, RR, DELTA = C["n"], C["r"], C["Delta"]
TAG = f"{VERSION}_code{CODE}"

tf.keras.utils.set_random_seed(636 + CODE)
tf.config.experimental.enable_tensor_float_32_execution(False)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


def transformer_block(h, width, heads, name):
    z = tf.keras.layers.LayerNormalization(name=f"{name}_attention_norm")(h)
    z = tf.keras.layers.MultiHeadAttention(heads, width // heads, name=f"{name}_attention")(z, z)
    h = tf.keras.layers.Add()([h, z])
    z = tf.keras.layers.LayerNormalization(name=f"{name}_ffn_norm")(h)
    z = tf.keras.layers.Dense(4 * width, activation="gelu", name=f"{name}_ffn_hidden")(z)
    return tf.keras.layers.Add()([h, tf.keras.layers.Dense(width, name=f"{name}_ffn_out")(z)])


def fixed_layer(equation, output_shape, kernel, name):
    layer = tf.keras.layers.EinsumDense(equation, output_shape=output_shape, name=name, trainable=False)
    layer.build((None, kernel.shape[0]))
    layer.set_weights([kernel])
    return layer


def build_network(code, width, blocks, heads, aux=0.0, arch="transformer", stages=1, gain=8.0, support=0.0, pos=True, **_):
    C = CODES[code]; n, r = C["n"], C["r"]
    inp = tf.keras.Input(shape=(r,), name="syndrome")
    # Fixed (not trained): token j = ROTATIONS[j] @ s, the syndrome seen from location j.
    rot_kernel = np.transpose(C["ROTATIONS"], (2, 0, 1))                       # (r, n, r)
    x = fixed_layer("br,rjq->bjq", (n, r), rot_kernel, "frames")(inp)
    # Shared linear embedding of every token plus a learned per-location bias (location embedding; none if not pos).
    h = tf.keras.layers.EinsumDense("bjq,qd->bjd", output_shape=(n, width), bias_axes="jd" if pos else "d", name="embed")(x)
    h = tf.keras.layers.Dense(width, activation="gelu", name="embed_mlp")(h)
    values = []
    if arch == "refine":
        # Fixed: values v (n) -> ROTATIONS[j] @ H v for every j, the syndrome that v explains, seen from each location.
        explain = fixed_layer("bi,ijq->bjq", (n, r), np.einsum("jqr,ri->ijq", C["ROTATIONS"], C["H"]), "explain")
        for t in range(stages):
            for b in range(blocks):
                h = transformer_block(h, width, heads, f"stage{t}_block{b}")
            hn = tf.keras.layers.LayerNormalization(name=f"stage{t}_norm")(h)
            v = tf.keras.layers.Reshape((n,), name=f"stage{t}_value")(tf.keras.layers.Dense(1, name=f"stage{t}_token_value")(hn))
            values.append(v)
            if t < stages - 1:
                rest = tf.keras.layers.Subtract(name=f"stage{t}_unexplained")([x, explain(v)])
                rest = tf.keras.layers.Rescaling(gain ** (t + 1), name=f"stage{t}_amplify")(rest)
                h = tf.keras.layers.Add(name=f"stage{t}_inject")([h, tf.keras.layers.EinsumDense(
                    "bjq,qd->bjd", output_shape=(n, width), bias_axes="jd", name=f"stage{t}_embed")(rest)])
        h = hn
    elif arch == "zoom":
        explain = fixed_layer("bi,ijq->bjq", (n, r), np.einsum("jqr,ri->ijq", C["ROTATIONS"], C["H"]), "explain")
        v = None
        for t in range(stages):
            if t > 0:
                rest = tf.keras.layers.Subtract(name=f"stage{t}_unexplained")([x, explain(v)])
                rest = tf.keras.layers.Activation("tanh", name=f"stage{t}_bounded")(
                    tf.keras.layers.Rescaling(gain ** t, name=f"stage{t}_zoom")(rest))
                h = tf.keras.layers.Add(name=f"stage{t}_inject")([h, tf.keras.layers.EinsumDense(
                    "bjq,qd->bjd", output_shape=(n, width), bias_axes="jd", name=f"stage{t}_embed")(rest)])
            for b in range(blocks):
                h = transformer_block(h, width, heads, f"stage{t}_block{b}")
            hn = tf.keras.layers.LayerNormalization(name=f"stage{t}_norm")(h)
            dv = tf.keras.layers.Reshape((n,), name=f"stage{t}_correction")(tf.keras.layers.Dense(1, name=f"stage{t}_token_value")(hn))
            if t == 0:
                v = dv
            else:
                v = tf.keras.layers.Add(name=f"stage{t}_value")([v, tf.keras.layers.Rescaling(gain ** -t, name=f"stage{t}_unzoom")(dv)])
            values.append(v)
        h = hn
    else:
        for b in range(blocks):
            h = transformer_block(h, width, heads, f"block{b}")
        h = tf.keras.layers.LayerNormalization(name="final_norm")(h)
        if aux > 0:
            values.append(tf.keras.layers.Reshape((n,), name="value")(tf.keras.layers.Dense(1, name="token_value")(h)))
    logits = tf.keras.layers.Reshape((n,), name="logits")(tf.keras.layers.Dense(1, name="token_logit")(h))
    if support > 0:   # placed last; make_loss reads it from the end of the outputs
        values.append(tf.keras.layers.Reshape((n,), name="support")(tf.keras.layers.Dense(1, name="token_support")(h)))
    outputs = tf.keras.layers.Concatenate(name="logits_and_values")([logits] + values) if values else logits
    return tf.keras.Model(inp, outputs, name=f"ume_locator_code{code}_{VERSION.lower()}")


def make_loss(n, aux, zoom=None, support=0.0):
    """Masked BCE on the logits plus aux times the value losses averaged over the stages: MSE, or with zoom (R4)
    Huber(zoom^t * (v_t - e / Delta)) / zoom^t for stage t: quadratic with curvature zoom^t for errors below its
    resolution 1 / zoom^t, linear (gradient magnitude at most 1) above it. With support > 0 (H5), the last n outputs
    are support logits, trained with BCE on e_j != 0 at every location."""
    def loss(y_true, y_pred):
        target, mask, value = y_true[:, :n], y_true[:, n:2 * n], y_true[:, 2 * n:]
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=y_pred[:, :n])
        total = tf.reduce_sum(bce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
        if support > 0:
            present = tf.cast(value != 0, tf.float32)
            total += support * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=present, logits=y_pred[:, -n:]))
            y_pred = y_pred[:, :-n]
        if aux > 0:
            stages = y_pred.shape[1] // n - 1
            for t in range(stages):
                err = y_pred[:, n * (t + 1):n * (t + 2)] - value
                if zoom:
                    a = tf.abs(err * zoom ** t)
                    total += aux / stages * tf.reduce_mean(tf.where(a < 1.0, 0.5 * a * a, a - 0.5)) / zoom ** t
                else:
                    total += aux / stages * tf.reduce_mean(tf.square(err))
        return total
    return loss


def training_batch(_):
    s, e = tf_sample_harmonic(CODE, BATCH)
    required = tf.cast(tf.abs(e) > DELTA, tf.float32)
    optional = tf.cast((e != 0) & (tf.abs(e) <= DELTA), tf.float32)
    return s, tf.concat([required, 1.0 - optional, tf.cast(e / DELTA, tf.float32)], axis=1)


def predict_logits(network, s, n=N):
    f = tf.function(lambda b: network(b, training=False)[:, :n], jit_compile=True)
    return np.concatenate([np.asarray(f(tf.constant(s[i:i + 8192], tf.float32))) for i in range(0, len(s), 8192)])


def bce_np(logits, e):
    required = np.abs(e) > DELTA; mask = ~((e != 0) & ~required)
    loss = np.maximum(logits, 0) - logits * required + np.log1p(np.exp(-np.abs(logits)))
    return float((loss * mask).sum() / mask.sum())


class Validate(tf.keras.callbacks.Callback):
    def __init__(self, s, e, group, path):
        super().__init__(); self.s, self.e, self.group, self.path = s, e, group, path
        self.best = (np.inf, np.inf); self.history = []; self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logits = predict_logits(self.model, self.s)
        ok = correct(CODE, self.e, logits > 0); errors = int((~ok).sum()); loss = bce_np(logits, self.e)
        lr = float(np.asarray(self.model.optimizer.learning_rate))
        row = dict(epoch=epoch + 1, loss=float(logs["loss"]), val_loss=loss, val_errors=errors,
                   val_errors_group1=int((~ok[self.group == 1]).sum()), val_errors_group2=int((~ok[self.group == 2]).sum()),
                   lr=lr, minutes=(time.time() - self.start) / 60)
        self.history.append(row)
        improved = (errors, loss) < self.best
        if improved:
            self.best = (errors, loss); self.model.save(self.path)
        (RUN_DIR / f"history_{TAG}.json").write_text(json.dumps(dict(recipe=R, steps=STEPS, batch=BATCH,
                                                                     epochs=self.history), indent=1))
        print(f"epoch {epoch + 1:3d}  loss {row['loss']:.3e}  val_loss {loss:.3e}  val_errors {errors:6d} "
              f"({errors / len(self.s):.2e}; I {row['val_errors_group1']}, II {row['val_errors_group2']})  lr {lr:.2e}  "
              f"{row['minutes']:.1f} min{'  *' if improved else ''}", flush=True)


if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    yv, ev, gv = sample_harmonic(CODE, VAL_SIZE, np.random.default_rng([3601, CODE]), return_group=True)
    sv = syndrome(CODE, yv).astype(np.float32); del yv
    gpus = tf.config.list_physical_devices("GPU")
    # Several GPUs: data-parallel training (BATCH is the global batch, split evenly over the GPUs).
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
    with strategy.scope():
        network = build_network(CODE, **R)
        steps = R["epochs"] * STEPS; warmup = min(2000, steps // 10)
        lr = tf.keras.optimizers.schedules.CosineDecay(R["lr"] * 0.01, steps - warmup, alpha=0.01,
                                                       warmup_target=R["lr"], warmup_steps=warmup)
        network.compile(optimizer=tf.keras.optimizers.AdamW(lr, weight_decay=1e-5, beta_2=0.95, clipnorm=1.0),
                        loss=make_loss(N, R["aux"], R["gain"] if R.get("arch") == "zoom" else None, R.get("support", 0.0)))
    network.summary(print_fn=lambda s, **_: print(s))
    print(f"{TAG}: {R} | {STEPS} steps x {BATCH} codewords per epoch | GPUs {gpus} ({strategy.num_replicas_in_sync} replicas)",
          flush=True)
    data = tf.data.Dataset.range(1).repeat().map(training_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(8)
    path = RUN_DIR / f"model_{TAG}.keras"
    val = Validate(sv, ev, gv, path)
    network.fit(data, epochs=R["epochs"], steps_per_epoch=STEPS, callbacks=[val, tf.keras.callbacks.TerminateOnNaN()],
                verbose=0)
    (RUN_DIR / f"history_{TAG}.json").write_text(json.dumps(dict(recipe=R, steps=STEPS, batch=BATCH, epochs=val.history), indent=1))

    network = tf.keras.models.load_model(path, compile=False)
    yt, et, gt = sample_harmonic(CODE, TEST_SIZE, np.random.default_rng([3602, CODE]), return_group=True)
    st = syndrome(CODE, yt).astype(np.float32); del yt
    lv, lt = predict_logits(network, sv), predict_logits(network, st)
    grid = np.linspace(-8, 8, 161)
    errs = np.array([(~correct(CODE, ev, lv > t)).sum() for t in grid])
    best = grid[errs == errs.min()]; threshold = float(best[len(best) // 2])
    ok = correct(CODE, et, lt > threshold); rep = lt > threshold
    res = dict(version=VERSION, code=CODE, recipe=R, parameters=int(network.count_params()), threshold=threshold,
               val_errors_at_0=int((~correct(CODE, ev, lv > 0)).sum()), val_errors=int(errs.min()),
               test_size=TEST_SIZE, test_errors=int((~ok).sum()), test_accuracy=float(ok.mean()),
               test_accuracy_group1=float(ok[gt == 1].mean()), test_accuracy_group2=float(ok[gt == 2].mean()),
               test_missed_required=int(((np.abs(et) > DELTA) & ~rep).any(1).sum()),
               test_forbidden_reported=int((rep & (et == 0)).any(1).sum()))
    (RUN_DIR / f"results_{TAG}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
