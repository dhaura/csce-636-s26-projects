#!/bin/bash
#SBATCH --job-name=mheight-hard-gen
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/data/hard-gen-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/data/hard-gen-%j.err

# Closed-form labels for the m = n-k settings (see generate_hard_data.py), on a CPU node.
# Submit with: sbatch --account=YOUR_CPU_ACCOUNT run_hard_gen_cpu.sh [generate_hard_data.py options]
# It first checks the closed form against the LP on fresh matrices and stops on any mismatch.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1}"
module load python
conda activate "$PROJECT_DIR/.conda-env"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --cpu-bind=cores python generate_hard_data.py --workers 128 --chunk 20000 --rows-per-setting 10000000 "$@"
