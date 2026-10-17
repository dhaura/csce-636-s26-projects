# Project 1: DNN estimation of the m-height of analog codes

Given a systematic generator matrix `G = [I_k | P]` with `n = 9`, `k ∈ {4, 5, 6}` and `m ∈ {2, …, n−k}`, the model predicts the m-height `h_m(C)`. The exact value requires solving `m·C(n, m)` linear programs. The network replaces that with one forward pass. There are 9 `(n, k, m)` settings, and each one is graded separately with the cost

```
σ(y, ŷ) = (log2 y − log2 ŷ)²
```

Every prediction must be finite and ≥ 1.

## The model

The submission model is **V2**, in `project1_v2.ipynb`, trained into `artifacts/dnn_v2_finetune/`. It is a single TensorFlow/Keras residual multilayer perceptron that handles all 9 settings. `project1_v1.ipynb` is the earlier version: the same pipeline with a small MLP.

| Stage | What it does |
|---|---|
| Inputs | `matrix`: `P` placed in the upper-left corner of a zero-padded `6 × 5` array. `parameters`: `[n, k, m]`. |
| Scaling (inside the model) | `Rescaling(1/100)` on the matrix, `Rescaling(1/10)` on the parameters. These are fixed layers that are saved with the model. |
| Body (V2) | Flatten + concatenate (33 values) → Dense 512 → 4 residual blocks `x + Dense(512)(Dense(1024, GELU)(LayerNorm(x)))` → LayerNorm → ReLU (about 4.2M parameters) |
| Body (V1) | Flatten + concatenate → Dense 256 → Dense 128 → Dense 64, all ReLU (about 50k parameters) |
| Output | Dense 1 with Softplus, then `ReLU(max_value=100)`. This is `z = log2(ŷ)`, so it lies in `[0, 100]`. |
| Post-processing | `ŷ = 2^z`, which is always finite and ≥ 1 |

There is no other pre- or post-processing, and no linear-algebra features such as SVD, as the project rules require.

### Training

- **Target and loss:** `log2(m-height)` with mean squared error, which is exactly the grading cost.
- **Equal weight per setting:** each row is weighted by the inverse of its setting's frequency, so all 9 `(n, k, m)` settings count equally, as in the grade.
- **Split:** 60 / 20 / 20 train / validation / test, assigned per distinct matrix with a seeded hash of `(n, k, P)`. A matrix that appears under several `m` values never ends up in two splits.
- **Two training phases:**
  1. **Pre-training** on the extended dataset (provided + generated rows): Adam with learning rate 1e-3, validated on all extended validation rows. The best weights are saved as `pretrained_model.keras`.
  2. **Fine-tuning** on the provided rows only: a new Adam optimiser with learning rate 3e-4, validated on the provided-data validation rows. The best weights are saved as `best_model.keras`, the final model.

  Pre-training learns the general structure from more matrices. Fine-tuning adapts the model to the instructor's distribution, which the test set follows.
- **In both phases:** batch size 4096; `ReduceLROnPlateau` halves the learning rate after 3 epochs without improvement; early stopping waits 8 epochs; the checkpoint with the best validation loss is kept.
- **Evaluation:** costs are reported on the provided-data validation and test rows, for both the pre-trained and the fine-tuned model.
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

The extended pickles are written with NumPy 2. The notebook's `load_pickle` also reads them under NumPy 1.x. The notebook always loads the extended files and uses their first `N_PROVIDED` rows (the provided data) for fine-tuning and evaluation.

## Running

### On NERSC Perlmutter (batch)

```bash
cd project1
# V2 (submission model): pre-train + fine-tune -> artifacts/dnn_v2_finetune/ (about 30 minutes)
NOTEBOOK=project1_v2.ipynb MHEIGHT_RUN=dnn_v2_finetune sbatch --account=m4012_g --time=03:00:00 run_notebook.sh
# V1 -> artifacts/dnn_v1_finetune/ (the saved model used 100 + 150 epochs)
MHEIGHT_PRETRAIN_EPOCHS=100 MHEIGHT_FINETUNE_EPOCHS=150 sbatch --account=m4012_g run_notebook.sh
# Regenerate the extended dataset on a CPU node (about 4.5 hours; resubmit the same command to resume)
sbatch --account=m4012 run_data_gen_cpu.sh
# ...or on a GPU node (about 9 hours on 64 cores)
sbatch --account=m4012_g run_data_gen.sh
```

`run_notebook.sh` runs the notebook headlessly on one shared A100 using the project environment. `NOTEBOOK=` selects another notebook (default `project1_v1.ipynb`).

Each run directory `artifacts/<run>/` collects:
- `project1_v1_executed.ipynb`, the notebook with all its outputs;
- the Slurm logs;
- `pretrained_model.keras` and `best_model.keras` (the fine-tuned, final model);
- `pretrain_history.json`, `finetune_history.json`, `costs.json`, `split_indices.npz` and `local_test_predictions.pkl`.

Runs made with earlier notebook versions (`dnn_v1`, `dnn_v1_keras3`, `dnn_v1_extended`) contain `project1_executed.ipynb` and a single `history.json` instead.

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
| `MHEIGHT_RUN` | `dnn_v1_finetune` | output directory under `artifacts/` |
| `MHEIGHT_PRETRAIN_EPOCHS` | `50` | pre-training epoch budget (early stopping may end sooner) |
| `MHEIGHT_FINETUNE_EPOCHS` | `50` | fine-tuning epoch budget (early stopping may end sooner) |
| `MHEIGHT_FINETUNE_LR` | `3e-4` | fine-tuning learning rate |

Run the notebook from the `project1/` directory so that the `data/` and `artifacts/` paths resolve.

### On Google Colab

The first cell runs only on Colab. It installs exactly the versions the model was trained with: `tensorflow==2.21.0` and `keras==3.15.1` (with NumPy 2). These have wheels for Colab's Python 3.12. A fresh Python 3.12 environment with only these packages reproduced the project environment's predictions byte for byte. For training on Colab, also upload `data/`.

## Predicting for a test file (the final cell)

The last cell of `project1_v2.ipynb` is the one the graders run. It is self-contained: in a fresh session, run the first cell (on Colab) and then the final cell. No training cells or training data are needed.

1. Put the submitted model `best_model.keras` next to the notebook. When run from `project1/`, the cell also finds it at `artifacts/dnn_v2_finetune/best_model.keras`.
2. Set `TEST_INPUT_PATH` (default `CSCE-636-Project-1-Test-n_k_m_P`) to a pickled list of `[n, k, m, P]` samples.
3. Run the cell. It writes `OUTPUT_PATH` (default `CSCE-636-Project-1-Test-mHeights`): a pickled list of Python floats, one per input sample in input order. This is the format of `CSCE-636-Project-1-Train-mHeights`. Every value is finite and ≥ 1.

Inference runs on the CPU in fixed batches of 1024, so repeated runs give identical files. This was tested on the full provided training file (3,833,203 rows, about 6 minutes on a login node): two runs were byte-identical and matched the notebook's evaluation exactly.

For the post-deadline sample-test submission, generate `CSCE-636-Project-1-Test-mHeights` with this cell **on Colab**, the environment the graders are likely to use. Different CPU types can change the last bits of floating-point results.

## Files

| Path | Purpose |
|---|---|
| `project1_v1.ipynb` | Data loading, model, pre-training and fine-tuning, evaluation, inference helpers |
| `project1_v2.ipynb` | **Submission notebook**: V1's pipeline with a larger residual MLP (about 4.2M parameters), plus the final prediction cell; trains into `artifacts/dnn_v2_finetune` |
| `generate_data.py`, `GENERATION.md` | LP-based generation of extended training data |
| `run_data_gen_cpu.sh`, `run_data_gen.sh` | Slurm scripts for data generation on a CPU node or a GPU node |
| `run_notebook.sh` | Slurm script that executes a notebook on a GPU |
| `artifacts/` | Trained models, logs and executed notebooks (not tracked by Git) |
