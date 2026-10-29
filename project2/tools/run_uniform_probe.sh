#!/bin/bash
#SBATCH --job-name=mheight-uniform
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2/artifacts/uniform-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2/artifacts/uniform-%j.err
# Usage: sbatch --account=m4012 --qos=debug tools/run_uniform_probe.sh MODEL.keras [MODEL.keras ...]
set -euo pipefail
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
.conda-env/bin/python tools/uniform_probe.py "$@"
