#!/bin/bash
#SBATCH --job-name=mheight-offcluster
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/artifacts/offcluster-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/artifacts/offcluster-%j.err
# Usage: sbatch --account=m4012 --qos=debug tools/run_offcluster_probe.sh MODEL.keras [MODEL.keras ...]
set -euo pipefail
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
.conda-env/bin/python tools/offcluster_probe.py "$@"
