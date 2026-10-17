#!/bin/bash
#SBATCH --job-name=mheight-fast-gen
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/data/fast-gen-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/data/fast-gen-%j.err

# Exact labels for all settings by LP vertex enumeration (see generate_fast_data.py), on a CPU node.
# Submit with: sbatch --account=YOUR_CPU_ACCOUNT run_fast_gen_cpu.sh [generate_fast_data.py options]
# It first checks vertex enumeration against the LP on fresh matrices and stops on any mismatch.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1}"
module load python
conda activate "$PROJECT_DIR/.conda-env"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --cpu-bind=cores python generate_fast_data.py --workers 128 --chunk 10000 --matrices-per-k 5000000 "$@"
