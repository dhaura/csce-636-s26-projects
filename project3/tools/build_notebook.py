#!/usr/bin/env python3
"""Build project3_<tag>.ipynb, the self-contained submission notebook for Project 3 (decoding noisy codewords).

The notebook contains all code to generate data, build, train and test the networks, the selection of the final
network per code, the final (float64, threshold-folded) models, and the final cell that decodes the test files.
The code cells are self-contained (the same functions and seeds as codes.py and train36.py).
Run:  .conda-env/bin/python tools/build_notebook.py V2     (writes project3_v2.ipynb; versions V1..V2)
      .conda-env/bin/python tools/build_notebook.py V4     (writes project3_v4.ipynb: codes 1-2 with V1..V3 as in V3,
                                                            plus Part II for codes 3-6 of the updated project)
"""
import sys

import nbformat

FINAL = sys.argv[1] if len(sys.argv) > 1 else "V2"
PART2 = FINAL == "V4"                      # the updated project: codes 3-6 in a second part
ALL_VERSIONS = ["V1", "V2", "V3"]
VERSIONS = ALL_VERSIONS if PART2 else ALL_VERSIONS[:ALL_VERSIONS.index(FINAL) + 1]
HARD = "V3" in VERSIONS
TAG = FINAL.lower()
NOTEBOOK = f"project3_{TAG}.ipynb"

md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
cells = []


def p1(source):
    """Part I cells (codes 1 and 2): in V4 their loops run over CODES_12 instead of all codes to run."""
    return source.replace("CODES_TO_RUN", "CODES_12") if PART2 else source


# Part II (codes 3-6): the versions shown in the notebook.
VERSIONS36 = ["H2", "H3"]

version_rows = {
    "V1": "| **V1** | starting point | residual MLP on the whole vector $y$ | width 512, 4 blocks | 60 epochs, peak 1e-3 | exact distribution |",
    "V2": "| **V2** | one token per location, attention between locations | transformer | width 128, 4 blocks, 4 heads | 60 epochs, peak 5e-4 | exact distribution |",
    "V3": "| **V3** | harder training data | as V2 | as V2 | 60 epochs, peak 5e-4 | 75% exact distribution, 25% hard variants (section 5.1) |",
}

DATA_TEXT = (r"""- **Unlimited fresh training data.** Every training batch is drawn anew from the exact distribution of the project
  description, on the GPU, so the networks never see the same codeword twice (about 490 million training codewords per
  network) and cannot overfit.""" + (r""" V3 draws a quarter of every batch from harder variants of the distribution
  (UMEs near $\Delta$, $u$ and $\epsilon$ at the corners of their ranges), which widens its decision margins.""" if HARD else ""))
SELECTION_TEXT = (r"""- **Final models.** For each code, the version with the fewest validation errors (ties: the fewest errors on hard
  variants of the distribution, then the larger decision margin; section 7.1) is chosen.
 """ if HARD else
                  r"""- **Final models.** For each code, the version with the fewest validation errors (ties: the larger decision margin) is
  chosen.""")
SELECTION_TEXT = SELECTION_TEXT.rstrip() if not HARD else SELECTION_TEXT.rstrip() + chr(10) + " "
# The opening summary of the V4 notebook: the final model of each code only, without version names (the versions are
# described in sections 4, 7, 12 and 15).
OVERVIEW_V4 = r"""# Project 3: Deep Neural Networks for Locating Unbounded-Magnitude Errors in Noisy Codewords

## The Model

- **One final model per code:** It outputs one number per location, and location $j$ is
  reported when that number is positive. All decisions are made by the trained networks and the other layers are fixed
  matrix multiplications, the averaging over views and the decision threshold.
- **Inputs:** Codes 1 and 2 take $y / \Delta$. Codes 3 to 6 take the syndrome $s = yH^T / \Delta$, a fixed float64
  multiplication with the parity-check matrix $H$ of the code that removes the codeword exactly, seen from every
  location (one token per location).
- **Networks:** Transformers with one token per location. Self-attention lets every location compare its value with
  all the others, which is what decides whether it holds a UME.
- **Training:** Binary cross-entropy per location (target 1 at required and 0 at forbidden locations; optional
  locations are masked out, because either answer is correct there), AdamW with a warm-up and a cosine-decaying
  learning rate. Every batch of 4,096 codewords is drawn fresh from the distribution of the project description, on
  the GPU, so the networks never see the same codeword twice and cannot overfit. Codes 3 to 6 add an auxiliary loss
  on per-location estimates of $e_j / \Delta$.
- **Symmetry averaging (codes 3 to 6):** The codes are unchanged by negacyclic shifts, reflection and sign change, so
  the final model runs its network(s) on 8 transformed copies of the codeword ("views") and averages the outputs.
- **Final models:** For each code, the candidate with the fewest errors on 1,000,000 validation codewords, with its
  threshold folded in, saved in float64 so that its decisions are the same on any machine. The six final models are
  stored together in one file, `final_model_v4.keras`, which the final cell loads.

## The Final Model of Each Code

| Code | $n$ | $k$ | Network | Parameters | Training |
|---|---|---|---|---|---|
| 1 | 10 | 3 | transformer, width 128, 4 blocks, 4 heads | 0.8 M | 60 epochs, a quarter of every batch from harder variants of the distribution |
| 2 | 16 | 3 | as code 1 | 0.8 M | as code 1 |
| 3 | 16 | 8 | transformer, width 128, 6 blocks, 8 heads; 8 views | 1.2 M | 60 epochs |
| 4 | 20 | 10 | as code 3 | 1.2 M | 120 epochs |
| 5 | 40 | 20 | two transformers, widths 128 and 192, 6 blocks, 8 heads; 4 views each, outputs averaged | 1.2 M + 2.7 M | 150 and 90 epochs |
| 6 | 80 | 40 | transformer, width 192, 6 blocks, 8 heads; 8 views | 2.7 M | 150 epochs |

An epoch is 2,000 steps (codes 1 and 2) or 1,000 steps (codes 3 to 6) of 4,096 codewords. Sections 7 and 15 show how
the final models were chosen.

## Outline

1. Configuration.

Part I (codes 1 and 2):

2. The two codes and the noisy-codeword distribution.
3. The validation and test sets.
4. The network architectures.
5. Training.
6. Training curves.
7. Validation, test performance and the choice of the final network.
8. Large-scale test of the final networks.
9. The final models.

Part II (codes 3 to 6):

10. The four codes, their structure and their noisy-codeword distribution.
11. The validation and test sets.
12. The network architecture.
13. Training.
14. Training curves.
15. Validation and test performance.
16. The final models.

17. **Final cell:** load the trained model and decode the six test files.
"""
if PART2:
    cells.append(md(OVERVIEW_V4))
else:
  cells.append(md(rf"""# Project 3: A Deep Neural Network for Locating Unbounded-Magnitude Errors in Noisy Codewords

## The Task

A noisy codeword is $y = uG + \epsilon + e \in \mathbb{{R}}^n$: a codeword $uG$ of an analog code with generator matrix $G$
($k = 3$), bounded noise $\epsilon$ ($|\epsilon_j| \le \delta$) and a vector $e$ of up to three unbounded-magnitude errors
(UMEs). The decoder receives $y$ and returns a set $S$ of locations. $S$ is correct when it contains every
**required** location ($|e_j| > \Delta$) and no **forbidden** location ($e_j = 0$); locations with $0 < |e_j| \le \Delta$
are **optional**. Accuracy is the share of correctly decoded codewords, computed separately for the two codes
(Code 1: $n = 10$; Code 2: $n = 16$).

## The Model

- **One network per code.** Each network takes the $n$ values of $y$ (divided by $\Delta$) and outputs $n$ logits, one
  per location. The only post-processing is converting the outputs into a set: location $j$ is reported when its
  logit is positive. All the decoding is done by the trained network.
- **Training as $n$ binary decisions.** Each output is trained with binary cross-entropy: target 1 at required
  locations, 0 at forbidden ones. Optional locations are masked out of the loss, because either answer is correct
  there. A codeword is decoded correctly exactly when all its required and forbidden locations are on the right side.
{DATA_TEXT}
- **Evaluation.** Fixed, seeded sets of 1,000,000 validation and 1,000,000 test codewords per code (never used for
  training), then a large-scale test on {{STRESS}} million further fresh test codewords per code, because the networks make
  fewer than one error per million codewords.
{SELECTION_TEXT} Its decision threshold is folded into the network, and the network is saved in float64, so that its
  decisions are reproducible on any machine (section 9).

## Network Versions

| Version | Change | Architecture | Size | Training | Training data |
|---|---|---|---|---|---|
{chr(10).join(version_rows[v] for v in VERSIONS)}

All versions use AdamW (weight decay 1e-5) with a linear warm-up and a cosine-decaying learning rate, 2,000 steps
of 4,096 codewords per epoch.

## Outline

1. Configuration.
2. The two codes and the noisy-codeword distribution.
3. The validation and test sets.
4. The network architectures.
5. Training.
6. Training curves.
7. Validation, test performance and the choice of the final network.
8. Large-scale test of the final networks.
9. The final models.
10. **Final cell:** load the trained models and decode the two test files.
""".replace("{STRESS}", "100")))

cells.append(code(r'''# Install dependencies (Google Colab only; elsewhere the project environment is used).

import sys

if "google.colab" in sys.modules:
    %pip install -q "numpy>=2,<3" "tensorflow==2.21.0" "keras==3.15.1" matplotlib
'''))

cells.append(md(rf"""## 1. Configuration

The defaults are the configuration of the submitted run. Environment variables were used to train the networks as
separate batch jobs (one per version and code, with `P3_TRAIN=1`); the submitted notebook was then executed with
`P3_TRAIN=0`, which loads those trained networks and runs every evaluation.

| Setting | Default | Environment variable |
|---|---|---|
| Output directory | `artifacts/notebook_{TAG}` | `P3_OUT` |
| Train the networks (otherwise load trained ones) | `1` | `P3_TRAIN` |
| Codes | `{"1,2,3,4,5,6" if PART2 else "1,2"}` | `P3_CODES` |
| {"Versions to train or load (codes 1 and 2)" if PART2 else "Versions to train or load"} | `{",".join(VERSIONS)}` | `P3_VERSIONS` |
""" + (f"| Versions to train or load (codes 3 to 6) | `{','.join(VERSIONS36)}` | `P3_VERSIONS36` |\n" if PART2 else "") + rf"""| Epochs, steps per epoch, batch size (overrides, for quick tests) | from the version {"recipes (sections 4 and 12)" if PART2 else "table"}, {"2000 (codes 1, 2) and 1000 (codes 3 to 6)" if PART2 else "2000"}, 4096 | `P3_EPOCHS`, `P3_STEPS`, `P3_BATCH` |
| Size of the validation and test sets | 1,000,000 each | `P3_EVAL_SIZE` |
| Millions of codewords in the large-scale test (section 8) | 100 | `P3_STRESS_MILLIONS` |
""" + ("| Millions of codewords per hard variant (section 7.1) | 10 | `P3_HARD_MILLIONS` |\n" if HARD else "")))

cells.append(code(rf'''import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*shuffle=True.*")   # Keras notes that tf.data input is not reshuffled
tf.get_logger().setLevel("ERROR")                                # no retracing notices from the evaluation helpers

OUT = Path(os.environ.get("P3_OUT", "artifacts/notebook_{TAG}"))
TRAIN = os.environ.get("P3_TRAIN", "1") == "1"
CODES_TO_RUN = [int(c) for c in os.environ.get("P3_CODES", "{"1,2,3,4,5,6" if PART2 else "1,2"}").split(",")]
{'CODES_12 = [c for c in CODES_TO_RUN if c in (1, 2)]                  # Part I' + chr(10) + 'CODES_36 = [c for c in CODES_TO_RUN if c in (3, 4, 5, 6)]            # Part II' + chr(10) if PART2 else ''}VERSIONS_TO_RUN = os.environ.get("P3_VERSIONS", "{",".join(VERSIONS)}").split(",")
{'VERSIONS36_TO_RUN = os.environ.get("P3_VERSIONS36", "' + ",".join(VERSIONS36) + '").split(",")' + chr(10) if PART2 else ''}EPOCHS = int(os.environ["P3_EPOCHS"]) if "P3_EPOCHS" in os.environ else None
STEPS = int(os.environ.get("P3_STEPS", "2000"))
{'STEPS36 = int(os.environ.get("P3_STEPS", "1000"))' + chr(10) if PART2 else ''}BATCH = int(os.environ.get("P3_BATCH", "4096"))
EVAL_SIZE = int(os.environ.get("P3_EVAL_SIZE", "1000000"))
STRESS_MILLIONS = int(os.environ.get("P3_STRESS_MILLIONS", "100"))
{'HARD_MILLIONS = int(os.environ.get("P3_HARD_MILLIONS", "10"))' + chr(10) if HARD else ''}NOTEBOOK_TAG = "{TAG}"
OUT.mkdir(parents=True, exist_ok=True)

# Full float32 precision on GPUs (no TF32 matrix multiplication), and GPU memory allocated as needed.
tf.config.experimental.enable_tensor_float_32_execution(False)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"TensorFlow {{tf.__version__}}, NumPy {{np.__version__}} | GPUs: {{tf.config.list_physical_devices('GPU')}}")
print(f"Output {{OUT}} | train {{TRAIN}} | codes {{CODES_TO_RUN}} | versions {{VERSIONS_TO_RUN}}"{' f" (codes 1, 2), {VERSIONS36_TO_RUN} (codes 3 to 6)"' if PART2 else ''})
'''))

cells.append(md((r"""# Part I: Codes 1 and 2

""" if PART2 else "") + r"""## 2. The Two Codes and the Noisy-Codeword Distribution

Both codes have $k = 3$; $\phi = (1 + \sqrt 5) / 2$.

- **Code 1** ($n = 10$): the columns of $G_1$ are the ten axes of a regular dodecahedron (all of length $\sqrt 3$).
- **Code 2** ($n = 16$): four columns $(-1, -1, -1)^T$, then each unit vector four times.

$\delta$ is the average column length of $G$, $\Delta_1 = (6 + 2\sqrt 5)\,\delta_1$, $\Delta_2 = 8\,\delta_2$ and
$Y = 3\Delta$. A noisy codeword is $y = uG + \epsilon + e$ with $u \sim U[-1, 1]^k$, $\epsilon \sim U[-\delta, \delta]^n$,
and $e$ with $A \in \{3, 2, 1, 0\}$ nonzero entries (probabilities $1/2, 1/4, 1/8, 1/8$) at $A$ distinct, uniformly
chosen locations, each uniform on $[-Y, Y] \setminus \{0\}$.

`sample` draws codewords with NumPy in float64 (for the fixed evaluation sets); `tf_sample` draws the same
distribution with TensorFlow on the GPU (for the training batches). `correct` implements the correctness rule and
`check_output` the output-format rules. The cell checks every constant against the decimals given in the project
description.
"""))

cells.append(code(r'''PHI = (1 + np.sqrt(5)) / 2
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
    C = CODES[code]; n = C["n"]
    u = rng.uniform(-1, 1, (size, C["k"]))
    eps = rng.uniform(-C["delta"], C["delta"], (size, n))
    A = rng.choice(A_VALUES, size=size, p=A_PROBS)
    chosen = np.argsort(rng.random((size, n)), axis=1).argsort(axis=1) < A[:, None]   # A uniform distinct locations
    # Uniform on [-Y, Y] without zero: magnitude Y (1 - U) lies in (0, Y] for U in [0, 1), with an independent sign.
    values = C["Y"] * (1 - rng.random((size, n))) * rng.choice((-1.0, 1.0), (size, n))
    e = np.where(chosen, values, 0.0)
    return u @ C["G"] + eps + e, e


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


# Check the constants against the project description.
reference = {1: dict(delta=1.732050808, Delta=18.138271538, Y=54.414814613),
             2: dict(delta=1.183012702, Delta=9.464101615, Y=28.392304845)}
rounded_G1 = np.array([[1.00, 1.00, 1.00, -1.00, 0.00, 0.00, 0.62, -0.62, 1.62, -1.62],
                       [1.00, 1.00, -1.00, 1.00, 1.62, 1.62, 0.00, 0.00, 0.62, 0.62],
                       [1.00, -1.00, 1.00, 1.00, 0.62, -0.62, 1.62, 1.62, 0.00, 0.00]])
assert np.array_equal(np.round(G1, 2), rounded_G1), "G1 does not round to the matrix in the description"
for code, C in CODES.items():
    for name, value in reference[code].items():
        assert abs(C[name] - value) < 1e-8, (code, name, C[name], value)
    print(f"Code {code}: k = {C['k']}, n = {C['n']}, delta = {C['delta']:.9f}, Delta = {C['Delta']:.9f}, Y = {C['Y']:.9f}")
print("All constants match the project description.")
'''))

cells.append(md(r"""### 2.1 The provided example files

The provided examples (1,000 noisy codewords per code with one permitted decoding each) show the input and output
formats; they come without their UME vectors, so they cannot measure accuracy. The cell checks that the example outputs
pass `check_output`, and compares the distribution of the example inputs with codewords drawn by `sample`: the
quantiles of $|y_j|$ and of the largest $|y_j|$ per codeword should agree up to sampling error (1,000 examples).
"""))

cells.append(code(r'''def find_file(name, folders=(".", "data")):
    """The first existing path for a provided file (the notebook folder or data/), or None."""
    return next((Path(f) / name for f in folders if (Path(f) / name).exists()), None)


class CompatUnpickler(pickle.Unpickler):
    # Pickles written with NumPy 2 reference numpy._core; NumPy 1.x only has numpy.core.
    def find_class(self, module, name):
        if int(np.__version__.split(".")[0]) < 2 and module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, "rb") as file:
        return CompatUnpickler(file).load()


EXAMPLES = {}
for code in CODES:
    inputs = find_file(f"Noisy-Codeword-Examples-for-Code-{code}.pkl")
    outputs = find_file(f"Decode-results-examples-for-code-{code}.pkl")
    if inputs is None or outputs is None:
        print(f"Code {code}: example files not found (optional)"); continue
    X, S = np.asarray(load_pickle(inputs), dtype=np.float64), load_pickle(outputs)
    n = CODES[code]["n"]
    assert X.shape == (len(X), n) and np.isfinite(X).all()
    check_output(S, len(X), n)
    EXAMPLES[code] = (X, S)
    y, _ = sample(code, 200_000, np.random.default_rng([3600, code]))
    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print(f"Code {code}: {len(X)} examples, outputs pass the format check; mean set size {np.mean([len(s) for s in S]):.2f}")
    print(f"  |y_j| quantiles {q}:  examples {np.round(np.quantile(np.abs(X), q), 2).tolist()}")
    print(f"                        sampled  {np.round(np.quantile(np.abs(y), q), 2).tolist()}")
    print(f"  max_j |y_j| quantiles: examples {np.round(np.quantile(np.abs(X).max(1), q), 2).tolist()}")
    print(f"                         sampled  {np.round(np.quantile(np.abs(y).max(1), q), 2).tolist()}")
'''))

PART1_START = len(cells)                   # from here to section 9, loops run over the Part I codes (p1)
cells.append(md(r"""## 3. The Validation and Test Sets

One fixed set of 1,000,000 validation codewords and one of 1,000,000 test codewords per code, drawn with `sample`
from seeds used nowhere else. Training never uses them: it draws fresh batches with TensorFlow's random generator.
The validation set selects the best epoch, the decision threshold and the final version; the test set is used only
to report performance.
"""))

cells.append(code(r'''EVAL = {}
for code in CODES_TO_RUN:
    y_val, e_val = sample(code, EVAL_SIZE, np.random.default_rng([3601, code]))
    y_test, e_test = sample(code, EVAL_SIZE, np.random.default_rng([3602, code]))
    EVAL[code] = dict(y_val=y_val, e_val=e_val, y_test=y_test, e_test=e_test)
    C = CODES[code]
    A = (e_test != 0).sum(1)
    required = (np.abs(e_test) > C["Delta"]).sum(1)
    optional = ((e_test != 0) & (np.abs(e_test) <= C["Delta"])).sum(1)
    print(f"Code {code}: {len(y_val):,} validation and {len(y_test):,} test codewords | test: share with A = 0..3: "
          + " ".join(f"{(A == a).mean():.4f}" for a in range(4))
          + f" | mean required {required.mean():.3f}, optional {optional.mean():.3f} locations per codeword")
'''))

cells.append(md(r"""## 4. The Network Architectures

Both architectures take $y / \Delta$ and output $n$ logits.

- **Residual MLP (V1):** a dense embedding of the whole vector, residual blocks (layer normalisation, a dense layer
  of twice the width with GELU, a dense layer back to the width), and a dense output layer with $n$ logits.
- **Transformer (V2):** one token per location. Token $j$ is $y_j w_j + b_j$ followed by a GELU layer, where $w_j$
  and $b_j$ are learned per location ($b_j$ acts as a location embedding). Pre-normalised transformer blocks
  (multi-head self-attention between the $n$ tokens, and a feed-forward layer of four times the width) let every
  location compare its value with all the others, which is what decoding requires: whether $y_j$ is a UME depends on
  how well the other values fit a codeword. A shared dense layer gives each token its logit.
"""))

recipe_lines = {
    "V1": '    # V1: residual MLP on the whole vector y.\n    "V1": dict(arch="mlp", width=512, blocks=4, heads=0, lr=1e-3, epochs=60),',
    "V2": '    # V2: transformer with one token per location (value embedding + learned location embedding).\n    "V2": dict(arch="transformer", width=128, blocks=4, heads=4, lr=5e-4, epochs=60),',
    "V3": '    # V3: V2 trained with a quarter of every batch drawn from the hard variants (tf_sample_hard).\n    "V3": dict(arch="transformer", width=128, blocks=4, heads=4, lr=5e-4, epochs=60, hard=0.25),',
}
cells.append(code(r'''VERSIONS = {
''' + "\n".join(recipe_lines[v] for v in VERSIONS) + r'''
}


def recipe(version):
    R = dict(VERSIONS[version])
    if EPOCHS is not None:
        R["epochs"] = EPOCHS
    return R


def build_network(code, arch, width, blocks, heads, **_):
    n = CODES[code]["n"]
    inp = tf.keras.Input(shape=(n,), name="y")
    x = tf.keras.layers.Rescaling(1 / CODES[code]["Delta"], name="scale")(inp)
    if arch == "mlp":
        h = tf.keras.layers.Dense(width, activation="gelu", name="embed")(x)
        for b in range(blocks):
            z = tf.keras.layers.LayerNormalization(name=f"block{b}_norm")(h)
            z = tf.keras.layers.Dense(2 * width, activation="gelu", name=f"block{b}_hidden")(z)
            h = tf.keras.layers.Add()([h, tf.keras.layers.Dense(width, name=f"block{b}_out")(z)])
        logits = tf.keras.layers.Dense(n, name="logits")(tf.keras.layers.LayerNormalization(name="final_norm")(h))
    else:
        # Token j = y_j * w_j + b_j: a location-specific embedding of the value; b_j acts as a learned location
        # embedding. A per-token feed-forward layer then makes the embedding nonlinear in y_j.
        h = tf.keras.layers.EinsumDense("bn,nd->bnd", output_shape=(n, width), bias_axes="nd", name="embed")(x)
        h = tf.keras.layers.Dense(width, activation="gelu", name="embed_mlp")(h)
        for b in range(blocks):
            z = tf.keras.layers.LayerNormalization(name=f"block{b}_attention_norm")(h)
            z = tf.keras.layers.MultiHeadAttention(heads, width // heads, name=f"block{b}_attention")(z, z)
            h = tf.keras.layers.Add()([h, z])
            z = tf.keras.layers.LayerNormalization(name=f"block{b}_ffn_norm")(h)
            z = tf.keras.layers.Dense(4 * width, activation="gelu", name=f"block{b}_ffn_hidden")(z)
            h = tf.keras.layers.Add()([h, tf.keras.layers.Dense(width, name=f"block{b}_ffn_out")(z)])
        z = tf.keras.layers.Dense(1, name="token_logit")(tf.keras.layers.LayerNormalization(name="final_norm")(h))
        logits = tf.keras.layers.Reshape((n,), name="logits")(z)
    return tf.keras.Model(inp, logits, name=f"ume_locator_code{code}_{arch}")


for version in VERSIONS:
    for code in CODES:
        print(f"{version} code {code}: {build_network(code, **recipe(version)).count_params():,} parameters")
'''))

train_md = r"""## 5. Training

- **Loss** (`masked_bce`): binary cross-entropy per location with target 1 at required and 0 at forbidden locations,
  averaged over the non-optional locations. The training labels are passed as the targets followed by the mask.
- **Data** (`training_batch`): every batch of 4,096 codewords is drawn anew by `tf_sample` inside a `tf.data`
  pipeline on the GPU.
- **Optimiser:** AdamW (weight decay 1e-5, $\beta_2 = 0.95$, gradients clipped to norm 1), learning rate warming up
  linearly over 2,000 steps from 1% of its peak, then decaying along a cosine to 1% of the peak.
- **Validation** (`Validate`): after every epoch, the network decodes the full validation set (logit > 0); the epoch
  with the fewest wrongly decoded codewords (ties: lower validation loss) is saved.
- Each network is trained with the random seed 636 + code.
"""
if HARD:
    train_md += r"""
### 5.1 Hard variants of the training data (V3)

The earlier versions decode the exact distribution without errors (sections 7 and 8), but tests on harder variants of
the distribution showed that their decision margins shrink, and some codewords are decoded wrongly, when the UMEs
lie close to $\Delta$ or when $u$ and $\epsilon$ sit at corners of their ranges. V3 therefore draws a quarter of
every training batch from three hard variants (`tf_sample_hard`, one variant per codeword, equally likely):

- **near:** always $A = 3$ UMEs, each magnitude uniform on $[0.7\Delta, 1.3\Delta]$, so required locations sit just
  above $\Delta$ next to optional ones just below it;
- **corner:** the usual $A$ and $e$, with $u \in \{-1, 1\}^k$ and $\epsilon \in \{-\delta, \delta\}^n$;
- **near-corner:** both.

Only the training data change; validation, the threshold and the test use the exact distribution.
"""
cells.append(md(train_md))

hard_code = r'''

def tf_sample_hard(code, size):
    """Training-only hard variants of the distribution, one of three per codeword (equally likely):
    near (A = 3, every UME magnitude uniform in [0.7, 1.3] Delta), corner (the usual A and e, with u in {-1, 1}^k and
    eps in {-delta, delta}^n), and near-corner (both)."""
    C = CODES[code]; n, k = C["n"], C["k"]
    variant = tf.random.uniform((size, 1), 0, 3, tf.int32)
    near, corner = variant != 2, variant != 0                         # 0 near, 1 near-corner, 2 corner
    sign = lambda shape: tf.where(tf.random.uniform(shape) < 0.5, -1.0, 1.0)
    u = tf.where(corner, sign((size, k)), tf.random.uniform((size, k), -1.0, 1.0))
    eps = tf.where(corner, C["delta"] * sign((size, n)), tf.random.uniform((size, n), -C["delta"], C["delta"]))
    A = tf.gather(tf.constant(A_VALUES, tf.int32),
                  tf.random.categorical(tf.math.log(tf.constant(A_PROBS[None], tf.float32)), size)[0])[:, None]
    A = tf.where(near, 3, A)
    mags = tf.where(near, C["Delta"] * tf.random.uniform((size, n), 0.7, 1.3),
                    C["Y"] * (1.0 - tf.random.uniform((size, n))))
    chosen = tf.argsort(tf.argsort(tf.random.uniform((size, n)), axis=1), axis=1) < A
    e = tf.where(chosen, mags * sign((size, n)), 0.0)
    return u @ tf.constant(C["G"], tf.float32) + eps + e, e
''' if HARD else ""

batch_code = r'''

def make_training_batch(code, hard_fraction=0.0):
    Delta = CODES[code]["Delta"]
    hard = int(round(hard_fraction * BATCH))

    def training_batch(_):
        y, e = tf_sample(code, BATCH - hard)
        if hard:
            yh, eh = tf_sample_hard(code, hard)
            y, e = tf.concat([y, yh], 0), tf.concat([e, eh], 0)
        required = tf.cast(tf.abs(e) > Delta, tf.float32)
        optional = tf.cast((e != 0) & (tf.abs(e) <= Delta), tf.float32)
        return y, tf.concat([required, 1.0 - optional], axis=1)
    return training_batch
''' if HARD else r'''

def make_training_batch(code):
    Delta = CODES[code]["Delta"]

    def training_batch(_):
        y, e = tf_sample(code, BATCH)
        required = tf.cast(tf.abs(e) > Delta, tf.float32)
        optional = tf.cast((e != 0) & (tf.abs(e) <= Delta), tf.float32)
        return y, tf.concat([required, 1.0 - optional], axis=1)
    return training_batch
'''

cells.append(code(r'''def make_masked_bce(n):
    def masked_bce(y_true, logits):
        target, mask = y_true[:, :n], y_true[:, n:]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
        return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
    return masked_bce
''' + hard_code + batch_code + r'''

def predict_logits(network, y):
    f = tf.function(lambda b: network(b, training=False), jit_compile=True)
    return np.concatenate([np.asarray(f(tf.constant(y[s:s + 16384], tf.float32))) for s in range(0, len(y), 16384)])


def bce_np(code, logits, e):
    required = np.abs(e) > CODES[code]["Delta"]; mask = ~((e != 0) & ~required)
    loss = np.maximum(logits, 0) - logits * required + np.log1p(np.exp(-np.abs(logits)))
    return float((loss * mask).sum() / mask.sum())


class Validate(tf.keras.callbacks.Callback):
    """Full-validation-set evaluation after every epoch; keeps the epoch with the fewest wrong codewords."""

    def __init__(self, code, y, e, path):
        super().__init__(); self.code, self.y, self.e, self.path = code, y, e, path
        self.best = (np.inf, np.inf); self.history = []; self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logits = predict_logits(self.model, self.y)
        errors = int((~correct(self.code, self.e, logits > 0)).sum()); loss = bce_np(self.code, logits, self.e)
        lr = float(np.asarray(self.model.optimizer.learning_rate))
        row = dict(epoch=epoch + 1, loss=float(logs["loss"]), val_loss=loss, val_errors=errors, lr=lr,
                   minutes=(time.time() - self.start) / 60)
        self.history.append(row)
        improved = (errors, loss) < self.best
        if improved:
            self.best = (errors, loss); self.model.save(self.path)
        print(f"epoch {epoch + 1:3d}  loss {row['loss']:.3e}  val_loss {loss:.3e}  val_errors {errors:6d} "
              f"({errors / len(self.y):.2e})  lr {lr:.2e}  {row['minutes']:.1f} min{'  *' if improved else ''}", flush=True)


def model_path(version, code):
    return OUT / f"model_{version}_code{code}.keras"


def train_network(version, code):
    R = recipe(version); n = CODES[code]["n"]
    tf.keras.utils.set_random_seed(636 + code)
    network = build_network(code, **R)
    steps = R["epochs"] * STEPS; warmup = min(2000, steps // 10)
    lr = tf.keras.optimizers.schedules.CosineDecay(R["lr"] * 0.01, steps - warmup, alpha=0.01, warmup_target=R["lr"],
                                                   warmup_steps=warmup)
    network.compile(optimizer=tf.keras.optimizers.AdamW(lr, weight_decay=1e-5, beta_2=0.95, clipnorm=1.0),
                    loss=make_masked_bce(n))
    print(f"{version} code {code}: {R} | {STEPS} steps x {BATCH} codewords per epoch", flush=True)
''' + (r'''    data = tf.data.Dataset.range(1).repeat().map(make_training_batch(code, R.get("hard", 0.0)),
                                                 num_parallel_calls=tf.data.AUTOTUNE).prefetch(8)
''' if HARD else r'''    data = tf.data.Dataset.range(1).repeat().map(make_training_batch(code), num_parallel_calls=tf.data.AUTOTUNE).prefetch(8)
''') + r'''    val = Validate(code, EVAL[code]["y_val"], EVAL[code]["e_val"], model_path(version, code))
    network.fit(data, epochs=R["epochs"], steps_per_epoch=STEPS, callbacks=[val, tf.keras.callbacks.TerminateOnNaN()],
                verbose=0)
    history = dict(recipe=R, steps=STEPS, batch=BATCH, epochs=val.history)
    (OUT / f"history_{version}_code{code}.json").write_text(json.dumps(history, indent=1))


NETWORKS = {}
for version in VERSIONS_TO_RUN:
    for code in CODES_TO_RUN:
        if TRAIN:
            train_network(version, code)
        if model_path(version, code).exists():
            NETWORKS[version, code] = tf.keras.models.load_model(model_path(version, code), compile=False)
            print(f"Loaded {model_path(version, code)}")
        else:
            print(f"{model_path(version, code)} not found: train it with P3_TRAIN=1")
'''))

cells.append(md(r"""## 6. Training Curves

Training loss (on the current batches), validation loss and the number of wrongly decoded validation codewords
(out of 1,000,000) after every epoch. Validation errors of zero cannot be drawn on the logarithmic scale; those epochs
are shown at the bottom edge of the plot.
"""))

cells.append(code(r'''HISTORIES = {}
for version in VERSIONS_TO_RUN:
    for code in CODES_TO_RUN:
        path = OUT / f"history_{version}_code{code}.json"
        if path.exists():
            HISTORIES[version, code] = json.loads(path.read_text())["epochs"]

if HISTORIES:
    fig, axes = plt.subplots(len(CODES_TO_RUN), 2, figsize=(13, 4.2 * len(CODES_TO_RUN)), squeeze=False)
    for row, code in enumerate(CODES_TO_RUN):
        for version in VERSIONS_TO_RUN:
            if (version, code) not in HISTORIES:
                continue
            h = HISTORIES[version, code]; epochs = [r["epoch"] for r in h]
            line, = axes[row, 0].plot(epochs, [r["loss"] for r in h], "--", alpha=0.7, label=f"{version} training")
            axes[row, 0].plot(epochs, [r["val_loss"] for r in h], color=line.get_color(), label=f"{version} validation")
            axes[row, 1].plot(epochs, [max(r["val_errors"], 0.5) for r in h], "o-", ms=3, color=line.get_color(), label=version)
        axes[row, 0].set(yscale="log", xlabel="epoch", ylabel="masked BCE loss", title=f"Code {code}: loss")
        axes[row, 1].set(yscale="log", xlabel="epoch", ylabel="wrong codewords (of 1M)", ylim=(0.4, None),
                         title=f"Code {code}: validation errors (0 drawn at the bottom edge)")
        for ax in axes[row]:
            ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.show()
    for (version, code), h in HISTORIES.items():
        best = min(h, key=lambda r: (r["val_errors"], r["val_loss"]))
        first0 = next((r["epoch"] for r in h if r["val_errors"] == 0), None)
        print(f"{version} code {code}: {len(h)} epochs in {h[-1]['minutes']:.0f} min; best epoch {best['epoch']} "
              f"(val errors {best['val_errors']}, val loss {best['val_loss']:.2e}); first epoch with 0 val errors: {first0}")
'''))

cells.append(md(r"""## 7. Validation, Test Performance and the Choice of the Final Network

For every trained network:

- **Threshold:** location $j$ is reported when its logit exceeds a threshold $t$, chosen on the validation set as
  the middle of the range of thresholds (from $-8$ to $8$ in steps of 0.1) with the fewest wrongly decoded codewords.
- **Decision margin:** for one codeword, the smallest distance of a logit from $t$ on the correct side, over its
  required locations (logit $- t$) and forbidden locations ($t -$ logit); optional locations do not count. The codeword
  is decoded correctly exactly when its margin is positive. When the networks make no errors, the low end of the
  margin distribution shows how far they are from making one.
""" + (r"""- **Choice** (section 7.1): for each code, the final network is the version with the fewest validation errors;
  ties are broken by the fewest errors on the hard validation variants, then by the larger 10th-smallest margin on them.
""" if HARD else r"""- **Choice:** for each code, the final network is the version with the fewest validation errors, with ties broken by
  the larger 10th-smallest validation margin.
""") + r"""- **Test:** the test set is decoded once, with the chosen threshold.
"""))

cells.append(code(r'''def margins(code, e, logits, t):
    """Per-codeword decision margin: min over required (logit - t) and forbidden (t - logit) locations."""
    required = np.abs(e) > CODES[code]["Delta"]
    per_location = np.where(required, logits - t, np.where(e == 0, t - logits, np.inf))
    return per_location.min(axis=1)


GRID = np.round(np.linspace(-8, 8, 161), 2)
RESULTS = {}
for (version, code), network in NETWORKS.items():
    ev, et = EVAL[code]["e_val"], EVAL[code]["e_test"]
    lv, lt = predict_logits(network, EVAL[code]["y_val"]), predict_logits(network, EVAL[code]["y_test"])
    errs = np.array([(~correct(code, ev, lv > t)).sum() for t in GRID])
    best = GRID[errs == errs.min()]; t = float(best[len(best) // 2])
    rep = lt > t; ok = correct(code, et, rep); A = (et != 0).sum(1)
    mv = np.sort(margins(code, ev, lv, t)); mt = np.sort(margins(code, et, lt, t))
    RESULTS[version, code] = dict(
        version=version, code=code, parameters=network.count_params(), threshold=t,
        zero_error_range=(float(best.min()), float(best.max())) if errs.min() == 0 else None,
        val_errors=int(errs.min()), val_margin_10th=float(mv[9]), test_errors=int((~ok).sum()),
        test_accuracy=float(ok.mean()), test_margin_10th=float(mt[9]),
        test_missed_required=int(((np.abs(et) > CODES[code]["Delta"]) & ~rep).any(1).sum()),
        test_forbidden_reported=int((rep & (et == 0)).any(1).sum()),
        test_accuracy_by_A=[float(ok[A == a].mean()) for a in range(4)])

print(f"{'':12s} {'params':>9s} {'threshold':>9s} {'0-error thresholds':>19s} {'val errors':>10s} {'val margin':>10s} "
      f"{'test errors':>11s} {'test accuracy':>13s} {'test margin':>11s}   test accuracy for A = 0, 1, 2, 3")
for (version, code), r in sorted(RESULTS.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    zr = "none" if r["zero_error_range"] is None else f"{r['zero_error_range'][0]:g} to {r['zero_error_range'][1]:g}"
    print(f"{version} code {code}  {r['parameters']:9,d} {r['threshold']:9.2f} {zr:>19s} {r['val_errors']:10d} "
          f"{r['val_margin_10th']:10.2f} {r['test_errors']:11d} {r['test_accuracy']:13.6f} {r['test_margin_10th']:11.2f}   "
          + ", ".join(f"{a:.6f}" for a in r["test_accuracy_by_A"]))
print("(margins: 10th-smallest per-codeword decision margin, in logits)")
''' + ("" if HARD else r'''
FINAL = {}
for code in CODES_TO_RUN:
    candidates = [r for (v, c), r in RESULTS.items() if c == code]
    if candidates:
        FINAL[code] = min(candidates, key=lambda r: (r["val_errors"], -r["val_margin_10th"]))
        print(f"Code {code}: final network {FINAL[code]['version']} (threshold {FINAL[code]['threshold']:g})")
''')))

if HARD:
    cells.append(md(r"""### 7.1 Hard variants of the distribution (validation) and the choice

All versions decode the 1,000,000 validation and test codewords without errors, so the exact distribution alone
cannot separate them. This section decodes three harder variants of the distribution, the same ones V3 partly trains
on (section 5.1): **near** (always 3 UMEs, magnitudes uniform on $[0.7\Delta, 1.3\Delta]$), **near-corner** (as near,
with $u$ and $\epsilon$ at random corners of their ranges) and **corner** (the usual UMEs, with $u$ and $\epsilon$ at
corners). For each variant, 10 million codewords are drawn with NumPy from validation seeds used nowhere else, and
every network decodes them with its threshold from section 7. These codewords serve evaluation only; they are never
part of the decoding. A brute-force decoder that is told the true UME locations decodes such codewords correctly, so
every error here is an error of the network, not an ambiguous codeword.
"""))

    cells.append(code(r'''VARIANTS = ("near", "near-corner", "corner")


def draw_variant(code, size, rng, variant):
    """Evaluation-only hard variants (NumPy, float64), the same distributions as tf_sample_hard's."""
    C = CODES[code]; n, k = C["n"], C["k"]
    corner = variant.endswith("corner")
    u = rng.choice((-1.0, 1.0), (size, k)) if corner else rng.uniform(-1, 1, (size, k))
    eps = C["delta"] * rng.choice((-1.0, 1.0), (size, n)) if corner else rng.uniform(-C["delta"], C["delta"], (size, n))
    if variant.startswith("near"):
        A = np.full(size, 3)
        mags = C["Delta"] * rng.uniform(0.7, 1.3, (size, n))
    else:
        A = rng.choice(A_VALUES, size=size, p=A_PROBS)
        mags = C["Y"] * (1 - rng.random((size, n)))
    chosen = np.argsort(rng.random((size, n)), axis=1).argsort(axis=1) < A[:, None]
    e = np.where(chosen, mags * rng.choice((-1.0, 1.0), (size, n)), 0.0)
    return u @ C["G"] + eps + e, e


HARD_RESULTS = {}
for code in CODES_TO_RUN:
    nets = [(v, NETWORKS[v, code], RESULTS[v, code]["threshold"]) for v in VERSIONS_TO_RUN if (v, code) in NETWORKS]
    fs = {v: tf.function(lambda b, net=net: net(b, training=False), jit_compile=True) for v, net, _ in nets}
    for v, _, _ in nets:
        HARD_RESULTS[v, code] = dict(wrong={x: 0 for x in VARIANTS}, lowest=np.full(10, np.inf))
    for index, variant in enumerate(VARIANTS, start=1):
        for chunk in range(HARD_MILLIONS):
            y, e = draw_variant(code, 1_000_000, np.random.default_rng([3603, code, index, chunk]), variant)
            for v, _, t in nets:
                logits = np.concatenate([np.asarray(fs[v](tf.constant(y[s:s + 65536], tf.float32)))
                                         for s in range(0, len(y), 65536)])
                h = HARD_RESULTS[v, code]
                h["wrong"][variant] += int((~correct(code, e, logits > t)).sum())
                h["lowest"] = np.sort(np.concatenate([h["lowest"], np.sort(margins(code, e, logits, t))[:10]]))[:10]

print(f"Wrongly decoded codewords out of {HARD_MILLIONS} million per variant (validation seeds), "
      "and the 10th-smallest margin over all of them:")
print(f"{'':12s} " + " ".join(f"{x:>12s}" for x in VARIANTS) + f" {'total':>8s} {'margin':>8s}")
for (v, code), h in sorted(HARD_RESULTS.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    h["total"] = sum(h["wrong"].values()); h["margin_10th"] = float(h["lowest"][9])
    print(f"{v} code {code}  " + " ".join(f"{h['wrong'][x]:12d}" for x in VARIANTS) + f" {h['total']:8d} {h['margin_10th']:8.2f}")

FINAL = {}
for code in CODES_TO_RUN:
    candidates = [r for (v, c), r in RESULTS.items() if c == code]
    if candidates:
        FINAL[code] = min(candidates, key=lambda r: (r["val_errors"], HARD_RESULTS[r["version"], code]["total"],
                                                     -HARD_RESULTS[r["version"], code]["margin_10th"]))
        print(f"Code {code}: final network {FINAL[code]['version']} (threshold {FINAL[code]['threshold']:g})")
'''))

cells.append(md(r"""## 8. Large-Scale Test of the Final Networks

With no errors on 1,000,000 test codewords, the test set only shows that the error rate is below about $3 \times
10^{-6}$ (95% confidence). This section decodes 100 million further fresh test codewords per code (seeded chunks of
1,000,000, drawn with `sample` from seeds used nowhere else) with each final network and its threshold, counts the
wrongly decoded codewords, and records the decision margins. With $W$ errors in $N$ codewords, the error rate is
below $3/N$ when $W = 0$ (95% confidence).
"""))

cells.append(code(r'''STRESS = {}
for code, r in FINAL.items() if STRESS_MILLIONS > 0 else []:   # P3_STRESS_MILLIONS=0 skips this test
    network, t = NETWORKS[r["version"], code], r["threshold"]
    f = tf.function(lambda b, network=network: network(b, training=False), jit_compile=True)
    wrong, lowest, start = 0, np.full(10, np.inf), time.time()
    for chunk in range(STRESS_MILLIONS):
        y, e = sample(code, 1_000_000, np.random.default_rng([3604, code, chunk]))
        logits = np.concatenate([np.asarray(f(tf.constant(y[s:s + 65536], tf.float32))) for s in range(0, len(y), 65536)])
        wrong += int((~correct(code, e, logits > t)).sum())
        lowest = np.sort(np.concatenate([lowest, np.sort(margins(code, e, logits, t))[:10]]))[:10]
    STRESS[code] = dict(version=r["version"], codewords=STRESS_MILLIONS * 1_000_000, wrong=wrong,
                        lowest_margins=lowest.round(2).tolist())
    bound = f"error rate below {3 / (STRESS_MILLIONS * 1e6):.0e} (95%)" if wrong == 0 else f"error rate {wrong / (STRESS_MILLIONS * 1e6):.2e}"
    print(f"Code {code} ({r['version']}): {wrong} wrong of {STRESS_MILLIONS:,} million codewords -> {bound}; "
          f"lowest margins {lowest.round(2).tolist()} ({(time.time() - start) / 60:.1f} min)")
'''))

cells.append(md(rf"""## 9. The Final Models

Each final model is the chosen network with two changes that do not alter what it computes:

- **The threshold is folded in:** a final layer subtracts $t$ from the logits, so the final model reports location $j$
  exactly when its output is positive.
- **float64:** the network is rebuilt with float64 layers and the trained weights are copied into it. The
  submitted results are compared with the outputs of the model when it is run again; logits of optional
  locations can lie arbitrarily close to the threshold, where float32 rounding differences between machines (up to
  about $10^{{-4}}$) could flip a decision. In float64 these differences shrink to about $10^{{-12}}$.

The cell saves `final_model_{TAG}_code1.keras` and `final_model_{TAG}_code2.keras`, checks that the final models decode the
test set exactly like the selected networks, and runs them on the provided example inputs and on the first 100,000
test codewords written in the same format as the test files.
"""))

cells.append(code(r'''def final_model_path(code):
    return OUT / f"final_model_{NOTEBOOK_TAG}_code{code}.keras"


def make_final_model(code, r):
    tf.keras.config.set_dtype_policy("float64")
    try:
        network = build_network(code, **recipe(r["version"]))
        inp = tf.keras.Input(shape=(CODES[code]["n"],), name="y")
        out = tf.keras.layers.Rescaling(1.0, offset=-r["threshold"], name="minus_threshold")(network(inp))
        final = tf.keras.Model(inp, out, name=f"final_ume_locator_code{code}")
    finally:
        tf.keras.config.set_dtype_policy("float32")
    network.set_weights([w.astype(np.float64) for w in NETWORKS[r["version"], code].get_weights()])
    return final


def decode(final, y):
    """What the final cell does: float64 logits on the CPU in batches of 4,096, location reported when positive."""
    with tf.device("/CPU:0"):
        out = np.concatenate([np.asarray(final(tf.constant(y[s:s + 4096], tf.float64), training=False))
                              for s in range(0, len(y), 4096)]) if len(y) else np.zeros((0, final.output_shape[-1]))
    return out > 0


for code, r in FINAL.items():
    final = make_final_model(code, r)
    final.save(final_model_path(code))
    final = tf.keras.models.load_model(final_model_path(code), compile=False)
    y, e = EVAL[code]["y_test"], EVAL[code]["e_test"]
    start = time.time(); rep64 = decode(final, y); seconds = time.time() - start
    rep32 = predict_logits(NETWORKS[r["version"], code], y) > r["threshold"]
    optional = (e != 0) & (np.abs(e) <= CODES[code]["Delta"])
    print(f"Code {code}: saved {final_model_path(code)} ({r['version']}, threshold {r['threshold']:g}, output dtype "
          f"{final.output.dtype}); test accuracy {correct(code, e, rep64).mean():.6f}; codewords decoded differently from "
          f"the float32 network: {(rep64 != rep32).any(1).sum()} (all at optional locations: {not (rep64 != rep32)[~optional].any()}); "
          f"CPU speed {len(y) / seconds:,.0f} codewords/s")
    if code in EXAMPLES:
        X, S = EXAMPLES[code]
        ours = to_sets(decode(final, X)); check_output(ours, len(X), CODES[code]["n"])
        print(f"  examples: identical to the example output {np.mean([a == b for a, b in zip(ours, S)]):.3f}, "
              f"subset of it {np.mean([a <= b for a, b in zip(ours, S)]):.3f}, superset {np.mean([a >= b for a, b in zip(ours, S)]):.3f} "
              "(differences are expected at optional locations)")
    heldout = OUT / f"Noisy-Codeword-Heldout-for-Code-{code}.pkl"
    with open(heldout, "wb") as file:
        pickle.dump([[float(v) for v in row] for row in y[:100_000]], file)
    X = np.asarray(load_pickle(heldout), dtype=np.float64)
    ours = to_sets(decode(final, X)); check_output(ours, len(X), CODES[code]["n"])
    reported = np.zeros((len(X), CODES[code]["n"]), bool)
    for row, s in enumerate(ours):
        reported[row, list(s)] = True
    print(f"  {heldout.name}: {len(ours):,} outputs pass the format check, accuracy {correct(code, e[:len(X)], reported).mean():.6f}")
'''))

for _cell in cells[PART1_START:]:
    if _cell.cell_type == "code":
        _cell.source = p1(_cell.source)

if PART2:
    sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
    from notebook_part2 import part2_cells
    cells += part2_cells(md, code, TAG, VERSIONS36)
    cells.append(md(rf"""## 17. Final Cell: Decode the Six Test Files

This cell loads the one model file `final_model_{TAG}.keras` (the six final models, section
16.1; from the notebook folder or `artifacts/submission/`), reads `Noisy-Codeword-Test-for-Code-1.pkl` to
`Noisy-Codeword-Test-for-Code-6.pkl` (lists of lists of $n$ real numbers), decodes every codeword in float64 (on the
GPU if one is available, otherwise on the CPU) and writes
`Decode-Results-Test-for-Code-1.pkl` to `Decode-Results-Test-for-Code-6.pkl` (lists of sets of integer locations, in
the input order) with `pickle.dump`. The model of each code is selected by the code number (`get_layer("code3")`, for
instance), not by the input length (codes 2 and 3 both have $n = 16$). The codewords are decoded in batches whose
size depends on the code (halved if a GPU runs out of memory; this does not change any decision). The cell performs
no training. A missing input file is skipped with a message.
"""))
else:
    cells.append(md(rf"""## 10. Final Cell: Decode the Two Test Files

This cell is self-contained: it loads the two final models (`final_model_{TAG}_code1.keras`, `final_model_{TAG}_code2.keras`,
from the notebook folder or `artifacts/submission/`), reads `Noisy-Codeword-Test-for-Code-1.pkl` and
`Noisy-Codeword-Test-for-Code-2.pkl` (lists of lists of $n$ real numbers), decodes every codeword on the CPU in
float64 and writes `Decode-Results-Test-for-Code-1.pkl` and `Decode-Results-Test-for-Code-2.pkl` (lists of sets of
integer locations, in the input order) with `pickle.dump`. It performs no training. A missing input file is skipped
with a message.
"""))

# Final cell, decoding loop. V4: batches sized per code (the code 5 and 6 models evaluate several networks and views in
# float64), halved if the GPU runs out of memory; every codeword is decoded independently, so the batch size does not
# change the decisions.
FINAL_BATCH = ("BATCH = {1: 4096, 2: 4096, 3: 1024, 4: 1024, 5: 512, 6: 256}   # codewords per call (halved if the GPU "
               "runs out of memory)" if PART2 else "BATCH = 4096")
FINAL_LOOP = (
"""    batch, start = BATCH[code], 0
    with tf.device(DEVICE):
        while start < len(y):
            try:
                outputs = network(tf.constant(y[start:start + batch], tf.float64), training=False)
            except tf.errors.ResourceExhaustedError:      # GPU out of memory: smaller batches, same decisions
                if batch == 1:
                    raise
                batch //= 2
                continue
            reported[start:start + batch] = np.asarray(outputs) > 0
            start += batch
""" if PART2 else
"""    with tf.device("/CPU:0"):
        for start in range(0, len(y), BATCH):
            outputs = network(tf.constant(y[start:start + BATCH], tf.float64), training=False)
            reported[start:start + BATCH] = np.asarray(outputs) > 0
""")

cells.append(code(rf'''# ===== Final cell: load the trained DNNs and decode the {"six" if PART2 else "two"} test files =====
# On Google Colab, first install the package versions used for training, so the saved networks load and
# decode identically (elsewhere the project environment is used).
import sys

if "google.colab" in sys.modules:
    %pip install -q "numpy>=2,<3" "tensorflow==2.21.0" "keras==3.15.1"

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

if "google.colab" in sys.modules and tf.__version__ != "2.21.0":
    raise RuntimeError(f"TensorFlow {{tf.__version__}} was already loaded: restart the runtime (Runtime > Restart session) and run this cell again.")

CODE_LENGTHS = {{{"1: 10, 2: 16, 3: 16, 4: 20, 5: 40, 6: 80}   # code number -> codeword length n" if PART2 else "1: 10, 2: 16}"}
MODEL_FOLDERS = [".", "artifacts/submission"]
{FINAL_BATCH}
{'DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"   # float64 on either device' + chr(10) if PART2 else ''}

class CompatUnpickler(pickle.Unpickler):
    # Pickles written with NumPy 2 reference numpy._core; NumPy 1.x only has numpy.core.
    def find_class(self, module, name):
        if int(np.__version__.split(".")[0]) < 2 and module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


for code, n in CODE_LENGTHS.items():
    input_path = Path(f"Noisy-Codeword-Test-for-Code-{{code}}.pkl")
    output_path = Path(f"Decode-Results-Test-for-Code-{{code}}.pkl")
    if not input_path.exists():
        print(f"Code {{code}}: {{input_path}} not found, skipped"); continue
    model_file = next((Path(f) / f"final_model_{TAG}_code{{code}}.keras" for f in MODEL_FOLDERS
                       if (Path(f) / f"final_model_{TAG}_code{{code}}.keras").exists()), None)
    if model_file is None:
        raise FileNotFoundError(f"final_model_{TAG}_code{{code}}.keras not found in {{MODEL_FOLDERS}}")
    with tf.device({'DEVICE' if PART2 else '"/CPU:0"'}):
        network = tf.keras.models.load_model(model_file, compile=False)

    with open(input_path, "rb") as file:
        samples = CompatUnpickler(file).load()
    y = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1) if len(samples) else np.zeros((0, n))
    if y.shape[1] != n or not np.isfinite(y).all():
        raise ValueError(f"Code {{code}}: expected {{n}} finite real numbers per codeword, got shape {{y.shape}}")

    # Decoding: the network's outputs (float64, threshold folded in); location j is reported when output j > 0.
    reported = np.zeros(y.shape, dtype=bool)
{FINAL_LOOP}    results = [set(int(j) for j in np.flatnonzero(row)) for row in reported]

    # Format check: a list with one set of int locations in [0, n) per input codeword, in input order.
    assert type(results) is list and len(results) == len(samples)
    assert all(type(s) is set and all(type(j) is int and 0 <= j < n for j in s) for s in results)
    with open(output_path, "wb") as file:
        pickle.dump(results, file)
    print(f"Code {{code}}: {{model_file}} decoded {{len(results):,}} codewords{' on {DEVICE}' if PART2 else ''} -> {{output_path}} "
          f"(mean set size {{np.mean([len(s) for s in results]) if results else 0:.3f}}; first five {{results[:5]}})")
'''))

# V4: the final cell loads the one model file with the six final models (section 16.1) and takes each code's part by
# its number.
FINAL_CELL_V4 = """# ===== Final cell: load the trained DNNs and decode the six test files =====
# On Google Colab, first install the package versions used for training, so the saved networks load and
# decode identically (elsewhere the project environment is used).
import sys

if "google.colab" in sys.modules:
    %pip install -q "numpy>=2,<3" "tensorflow==2.21.0" "keras==3.15.1"

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

if "google.colab" in sys.modules and tf.__version__ != "2.21.0":
    raise RuntimeError(f"TensorFlow {tf.__version__} was already loaded: restart the runtime (Runtime > Restart session) and run this cell again.")

CODE_LENGTHS = {1: 10, 2: 16, 3: 16, 4: 20, 5: 40, 6: 80}   # code number -> codeword length n
MODEL_FILE = "final_model_TAG.keras"   # the six final models in one file, as the parts "code1" to "code6"
MODEL_FOLDERS = [".", "artifacts/submission"]
BATCH = {1: 4096, 2: 4096, 3: 1024, 4: 1024, 5: 512, 6: 256}   # codewords per call (halved if the GPU runs out of memory)
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"   # float64 on either device


class CompatUnpickler(pickle.Unpickler):
    # Pickles written with NumPy 2 reference numpy._core; NumPy 1.x only has numpy.core.
    def find_class(self, module, name):
        if int(np.__version__.split(".")[0]) < 2 and module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


models, model_file = None, None
for code, n in CODE_LENGTHS.items():
    input_path = Path(f"Noisy-Codeword-Test-for-Code-{code}.pkl")
    output_path = Path(f"Decode-Results-Test-for-Code-{code}.pkl")
    if not input_path.exists():
        print(f"Code {code}: {input_path} not found, skipped"); continue
    if models is None:                               # loaded once, when the first test file is found
        model_file = next((Path(f) / MODEL_FILE for f in MODEL_FOLDERS if (Path(f) / MODEL_FILE).exists()), None)
        if model_file is None:
            raise FileNotFoundError(f"{MODEL_FILE} not found in {MODEL_FOLDERS}")
        with tf.device(DEVICE):
            models = tf.keras.models.load_model(model_file, compile=False)
    network = models.get_layer(f"code{code}")        # the final model of this code, selected by the code number
    if network.inputs[0].shape[-1] != n:
        raise ValueError(f"Code {code}: the part code{code} of {model_file} expects {network.inputs[0].shape[-1]} numbers, not {n}")

    with open(input_path, "rb") as file:
        samples = CompatUnpickler(file).load()
    y = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1) if len(samples) else np.zeros((0, n))
    if y.shape[1] != n or not np.isfinite(y).all():
        raise ValueError(f"Code {code}: expected {n} finite real numbers per codeword, got shape {y.shape}")

    # Decoding: the network's outputs (float64, threshold folded in); location j is reported when output j > 0.
    reported = np.zeros(y.shape, dtype=bool)
    batch, start = BATCH[code], 0
    with tf.device(DEVICE):
        while start < len(y):
            try:
                outputs = network(tf.constant(y[start:start + batch], tf.float64), training=False)
            except tf.errors.ResourceExhaustedError:      # GPU out of memory: smaller batches, same decisions
                if batch == 1:
                    raise
                batch //= 2
                continue
            reported[start:start + batch] = np.asarray(outputs) > 0
            start += batch
    results = [set(int(j) for j in np.flatnonzero(row)) for row in reported]

    # Format check: a list with one set of int locations in [0, n) per input codeword, in input order.
    assert type(results) is list and len(results) == len(samples)
    assert all(type(s) is set and all(type(j) is int and 0 <= j < n for j in s) for s in results)
    with open(output_path, "wb") as file:
        pickle.dump(results, file)
    print(f"Code {code}: {model_file} (part code{code}) decoded {len(results):,} codewords on {DEVICE} -> {output_path} "
          f"(mean set size {np.mean([len(s) for s in results]) if results else 0:.3f}; first five {results[:5]})")
""".replace("final_model_TAG.keras", f"final_model_{TAG}.keras")
if PART2:
    assert cells[-1].cell_type == "code" and cells[-1].source.startswith("# ===== Final cell")
    cells[-1] = code(FINAL_CELL_V4)

# Wording edits made by hand in the submitted V4 notebook (Part I text shared with the V3 notebook, which stays as it was).
V4_WORDING = [
    ("`sample` draws codewords with NumPy in float64 (for the fixed evaluation sets); `tf_sample` draws the same",
     "`sample` draws codewords with NumPy in float64 (for the fixed evaluation sets) and `tf_sample` draws the same"),
    ("formats; they come without their UME vectors, so they cannot measure accuracy.",
     "formats and they come without their UME vectors, so they cannot measure accuracy."),
    ("Only the training data change; validation, the threshold and the test use the exact distribution.",
     "Only the training data change and for validation, the threshold and the test use the exact distribution."),
]
if PART2:
    for old, new in V4_WORDING:
        hits = [c for c in cells if c.cell_type == "markdown" and old in c.source]
        assert len(hits) == 1, old
        hits[0].source = hits[0].source.replace(old, new)

nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": ".conda-env", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nb.cells = cells
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
print(f"{NOTEBOOK} written with {len(cells)} cells (versions {VERSIONS})")
