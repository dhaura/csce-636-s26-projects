# Additional m-height data

The standalone `generate_data.py` uses NumPy and **SciPy** (`scipy.optimize.linprog`, HiGHS dual simplex). SciPy is the only new direct dependency. `multiprocessing`, `pickle`, and other utilities are part of Python. The notebook and `requirements.txt` are not changed by this generation setup.

Install into the existing environment if needed:

```bash
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
module load python
conda activate "$PWD/.conda-env"
python -m pip install scipy
```

## Generation and labels

- Defaults to **3,833,203 new rows**, yielding **7,666,406 total** with the original dataset.
- Uses `n=9` and cycles `k=4,5,6`, so each k gets the same number of sampled matrices.
- **Matrix distribution (default `--distribution empirical`)**: the instructor's matrices are not uniform on [−100, 100]. Each has its own scale (per-matrix max |P| is multimodal: 5th percentile 9, median 81). The generator therefore counts the per-matrix max |P| of the original data for each k (stored as `scale_counts` in `run.json`). For each new matrix it draws a scale `a` from that histogram and draws integer entries uniformly from `[-a, a]`, stored as float64. This matches the original value histograms, zero fraction and row/column spread. `--distribution integer` (fixed `a=100`) and `uniform` (continuous) remain available.
- Matrices with an all-zero `P` column are resampled, since the specification excludes all-zero columns of `G`.
- **Finite labels only**, like the original data and the test set. For each matrix, m=2,3,... are labelled until the first infinite m-height; since m-heights are nondecreasing in m, every larger m is infinite and is not solved. A matrix therefore contributes 0 to 9−k−1 rows, always the prefix m=2,...; the original data has the same structure. Generation stops after the matrix that reaches the target, and the merge drops the overshoot rows (largest m of the last matrix).
- Seeds are determined by the run seed and matrix ID. Changing the process count or checkpoint frequency on resume does not change the rows. Matrices are sampled independently; no deduplication against the original dataset is performed.
- Constructs `G=[I|P]` and solves the specified LP for every subset and selected column. Variables `u` are explicitly free in sign. Central symmetry makes a second LP for the negative objective unnecessary. Re-solving random original rows reproduces the instructor labels to about 1e-13 in log2.
- An LP is treated as unbounded (infinite height) only after a confirming solve without presolve. Numerical solver failures stop generation with the offending matrix ID instead of producing a made-up label. LP labels use floating-point solver tolerances, not symbolic exact arithmetic.
- The outputs are pickled with NumPy 2. The notebook's `load_pickle` also reads them under NumPy 1.x (for example the TensorFlow 2.15 module).

## Perlmutter submission

The supplied Slurm script requests **one GPU node, four GPUs, 128 logical CPUs (64 physical cores), all node memory, and 16 hours** (a full run took about 8.7 hours). It starts **64 CPU worker processes**, each with one HiGHS/BLAS thread. SciPy HiGHS does not use GPUs; the GPU allocation is included as requested. For this workload, a CPU-node allocation would use resources more efficiently.

These settings follow the [NERSC Perlmutter job guide](https://docs.nersc.gov/systems/perlmutter/running-jobs/), [CPU affinity guide](https://docs.nersc.gov/jobs/affinity/), and [queue limits](https://docs.nersc.gov/jobs/policy/). The LP interface and status codes are documented in [SciPy linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).

Run from the project directory, substituting your actual GPU allocation account:

```bash
sbatch --account=YOUR_GPU_ACCOUNT run_data_gen.sh
```

The job activates `project1/.conda-env`. Checkpoints and final files default to `$SCRATCH/csce636-project1-generation-v2`. After a completed run the script copies the two extended files and `complete.json` (as `Extended-complete.json`) into `project1/data/`. To use CFS instead:

```bash
export GENERATION_OUTPUT_DIR="$CFS/my-project/mheight-generation"
sbatch --account=YOUR_GPU_ACCOUNT run_data_gen.sh
```

Use an actual project directory to which you have write access. The script rejects output paths in HOME or outside SCRATCH/CFS. Slurm logs are written to this repository's `project1/data/` on `/pscratch`; those paths in the Slurm header are specific to this checkout. Adjust the absolute header paths if relocating the project. The checkpoint directory must support file locking (Perlmutter scratch does).

A 10-minute generation benchmark on a compute node can use the same job script:

```bash
sbatch --account=YOUR_GPU_ACCOUNT --time=00:20:00 \
  run_data_gen.sh --max-seconds 600
```

This saves useful progress for the full target. Subsequent normal submissions continue it. A short independent end-to-end trial can instead use a separate output directory and `--new-samples 9`. Such a trial still loads the originals during the final merge; never change `--new-samples` in an existing run directory.

The full schedule is roughly a billion LP solves (fewer, since solving stops at the first infinite height). Completion within one job is not guaranteed. Progress logs report actual rows/second and an estimated remaining time. Benchmark on the allocated node before extrapolating runtime.

### CPU-node variant

`run_data_gen_cpu.sh` runs the same generation on a CPU node: 128 workers on 128 physical cores, 10-hour limit. Submit it with your CPU account (`sbatch --account=m4012 run_data_gen_cpu.sh`). It writes to `$SCRATCH/csce636-project1-generation-v2-cpu`, so it can run alongside the GPU-node job. Both produce identical rows and copy the same files into `project1/data/`. Once one finishes, cancel the other.

## Checkpoint and resume behavior

Only the parent process writes files. It keeps at most twice the worker count of outstanding matrix tasks. Checkpoints are atomic single pickle files containing **both** feature and label lists, complete matrix groups, and contiguous matrix IDs. They are flushed after approximately **50,000 rows**, after **five minutes** of accumulated results, and on a handled stop signal. A group is never split across checkpoints.

A timeout warning (`SIGUSR1`), `SIGTERM`, or Ctrl+C saves the completed prefix and exits with **code 75**. A hard kill can lose work since the last checkpoint; it cannot commit a half-written checkpoint. In-flight matrices are regenerated deterministically. Temporary files ending in `.tmp` are ignored.

**Resubmit the same `sbatch` command to resume automatically.** This does not automatically submit another charged job. Exit code 75 means a resumable stop, not a completed dataset. Slurm sends a warning three minutes before the time limit. Seed, sample target, distribution, input hashes, and NumPy/SciPy versions must match the saved `run.json`; worker count and checkpoint frequency may change. A lock prevents concurrent jobs from writing into the same directory. Unreadable or structurally inconsistent checkpoints cause an explicit error instead of silently dropping rows.

After generation, workers exit and the original lists are loaded once. Checkpoints are appended in matrix order, preserving feature/label alignment and leaving the input files intact. The combined lists are written with `pickle.dump()` as:

- `CSCE-636-Project-1-Train-Extended-n_k_m_P`
- `CSCE-636-Project-1-Train-Extended-mHeights`

Each output is replaced atomically, and **`complete.json` is written last**, with counts and SHA-256 checksums. Require that marker before consuming the pair: the two output files cannot be committed together in one filesystem operation. If merging is interrupted, the next invocation repeats the merge from the checkpoints. Successful runs keep checkpoints for recovery. Allow space for the originals, shards, final outputs, and temporary files; the final merge holds both combined lists in node memory. No outputs are stored in HOME. SCRATCH remains subject to NERSC's purge policy.
