# Project 1: DNN estimation of the m-height of analog codes

Given a systematic generator matrix `G = [I_k | P]` with `n = 9`, `k ∈ {4, 5, 6}` and `m ∈ {2, …, n−k}`, the model predicts the m-height `h_m(C)`. The exact value requires solving `m·C(n, m)` linear programs. The network replaces that with one forward pass. There are 9 `(n, k, m)` settings, and each one is graded separately with the cost

```
σ(y, ŷ) = (log2 y − log2 ŷ)²
```

Every prediction must be finite and ≥ 1.

## The model

`project1.ipynb` contains the complete pipeline. It is a single TensorFlow/Keras multilayer perceptron that handles all 9 settings.

| Stage | What it does |
|---|---|
| Inputs | `matrix`: `P` placed in the upper-left corner of a zero-padded `6 × 5` array. `parameters`: `[n, k, m]`. |
| Scaling (inside the model) | `Rescaling(1/100)` on the matrix, `Rescaling(1/10)` on the parameters. These are fixed layers that are saved with the model. |
| Body | Flatten + concatenate (33 values) → Dense 256 → Dense 128 → Dense 64, all ReLU (about 50k parameters) |
| Output | Dense 1 with Softplus, then `ReLU(max_value=100)`. This is `z = log2(ŷ)`, so it lies in `[0, 100]`. |
| Post-processing | `ŷ = 2^z`, which is always finite and ≥ 1 |

There is no other pre- or post-processing, and no linear-algebra features such as SVD, as the project rules require.

### Training

- **Target and loss:** `log2(m-height)` with mean squared error, which is exactly the grading cost.
- **Equal weight per setting:** each row is weighted by the inverse of its setting's frequency, so all 9 `(n, k, m)` settings count equally, as in the grade.
- **Split:** 60 / 20 / 20 train / validation / test, assigned per distinct matrix with a seeded hash of `(n, k, P)`. A matrix that appears under several `m` values never ends up in two splits.
- **Optimiser:** Adam (lr 1e-3), batch size 4096, up to 50 epochs (`MHEIGHT_MAX_EPOCHS`).
  - `ReduceLROnPlateau` halves the learning rate after 3 epochs without improvement.
  - Early stopping waits 8 epochs.
  - The checkpoint with the best validation loss is kept.
- **Reproducibility:** seed 636 and TensorFlow op determinism.
- **Invalid labels:** rows with non-finite labels or labels below 1 are reported and excluded.

## Data

| File (in `data/`, not tracked by Git) | Contents |
|---|---|
| `CSCE-636-Project-1-Train-n_k_m_P` / `-mHeights` | The provided training set: 3,833,203 rows of `[n, k, m, P]` and their m-heights |
| `CSCE-636-Project-1-Train-Extended-n_k_m_P` / `-mHeights` | The provided rows followed by 3,833,203 rows labelled by the LP generator (7,666,406 rows in total) |

`generate_data.py` produces the extended rows with the LP algorithm from the project description, using SciPy HiGHS on CPUs. Re-solving random provided rows reproduces the instructor's labels to about 1e-13 in log2.

The new matrices are drawn to resemble the provided ones:
- Each matrix gets its own scale `a`, drawn from the original data's distribution of per-matrix max |P|.
- Its entries are integers drawn uniformly from `[−a, a]`.
- Matrices with an all-zero `P` column are rejected and redrawn.
- Only finite m-heights are kept, as in the provided data and the test set.

[GENERATION.md](GENERATION.md) covers the details, checkpointing and resume.

The extended pickles are written with NumPy 2. The notebook's `load_pickle` also reads them under NumPy 1.x.

## Running

### On NERSC Perlmutter (batch)

```bash
cd project1
# Train/evaluate on the provided data -> artifacts/dnn_v1_keras3/
sbatch --account=m4012_g run_notebook.sh
# Train/evaluate on the extended data -> artifacts/dnn_v2/
MHEIGHT_DATASET=extended MHEIGHT_RUN=dnn_v2 MHEIGHT_MAX_EPOCHS=150 sbatch --account=m4012_g run_notebook.sh
# Regenerate the extended dataset (about 9 hours on 64 cores; resubmit the same command to resume)
sbatch --account=m4012_g run_data_gen.sh
```

`run_notebook.sh` runs the notebook headlessly on one shared A100 using the project environment.

Each run directory `artifacts/<run>/` collects:
- `project1_executed.ipynb`, the notebook with all its outputs;
- the Slurm logs;
- `best_model.keras`, `history.json`, `costs.json` and `split_indices.npz`.

The source notebook is not modified.

### Environment

The project environment is `project1/.conda-env` (Python 3.11, TensorFlow 2.21, Keras 3.15, NumPy 2.4, SciPy). Its Jupyter kernel is **CSCE 636 Project 1** (`csce636-p1`).

TensorFlow uses the CUDA 12 / cuDNN 9 libraries that `tensorflow[and-cuda]` installs into the environment. For GPU use outside the batch script, do the same as the script:

```bash
module unload cudatoolkit   # the default CUDA 13 module conflicts with TF 2.21
export LD_LIBRARY_PATH=$(ls -d $PWD/.conda-env/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:):$LD_LIBRARY_PATH
```

Notebook settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `MHEIGHT_DATASET` | `original` | `original` or `extended` training data |
| `MHEIGHT_RUN` | `dnn_v1_keras3` | output directory under `artifacts/` |
| `MHEIGHT_MAX_EPOCHS` | `50` | epoch budget (early stopping may end sooner) |

Run the notebook from the `project1/` directory so that the `data/` and `artifacts/` paths resolve.

### On Google Colab

The first cell installs `numpy>=2,<3` and `tensorflow`, and only does so on Colab. Upload `data/` and the chosen `artifacts/<run>/best_model.keras` alongside the notebook.

## Files

| Path | Purpose |
|---|---|
| `project1.ipynb` | Data loading, model, training, evaluation, inference helpers |
| `generate_data.py`, `run_data_gen.sh`, `GENERATION.md` | LP-based generation of extended training data |
| `run_notebook.sh` | Slurm script that executes the notebook on a GPU |
| `TRAINING_REVIEW.md` | Review of the first (`dnn_v1`) training run |
| `artifacts/` | Trained models, logs and executed notebooks (not tracked by Git) |
