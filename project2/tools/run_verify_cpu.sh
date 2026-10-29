#!/bin/bash
#SBATCH --job-name=mheight-verify
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=01:00:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2/data/verify-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2/data/verify-%j.err
# Independent LP check of the generated training data.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 
.conda-env/bin/python tools/verify_generated_data.py
