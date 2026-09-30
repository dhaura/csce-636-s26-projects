#!/bin/bash
#SBATCH --job-name=mheight-generation
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@180
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/data/generation-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/data/generation-%j.err

# Submit with: sbatch --account=YOUR_GPU_ACCOUNT run_generation.slurm
# HiGHS is CPU-only: this reserves a GPU node as requested, but does not use GPUs.
# One process per physical CPU core, with one solver/BLAS thread per process.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1}"
: "${SCRATCH:?SCRATCH must be set}"
GENERATION_OUTPUT_DIR="${GENERATION_OUTPUT_DIR:-$SCRATCH/csce636-project1-generation}"
module load python
conda activate "$PROJECT_DIR/.conda-env"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
python -c 'import numpy, scipy; print("NumPy", numpy.__version__, "SciPy", scipy.__version__)'

# A nonzero exit of 75 means the checkpoint is safe and this job needs resubmission.
# The pre-timeout USR1 goes to the srun job step, whose Python parent checkpoints.
srun --cpu-bind=cores python generate_data.py \
    --input-dir "$PROJECT_DIR/data" \
    --output-dir "$GENERATION_OUTPUT_DIR" \
    --workers 64 \
    --new-samples 3833203 \
    --batch-size 50000 \
    --checkpoint-seconds 300 \
    "$@"
