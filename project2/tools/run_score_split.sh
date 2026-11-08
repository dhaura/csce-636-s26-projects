#!/bin/bash
#SBATCH --job-name=mheight-split
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2/artifacts/split-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2/artifacts/split-%j.err
# Usage: sbatch --account=m4012 --qos=debug tools/run_score_split.sh MODEL.keras [MODEL.keras ...]
set -euo pipefail
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
.conda-env/bin/python tools/score_split.py "$@"
