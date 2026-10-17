#!/bin/bash
#SBATCH --job-name=mheight-testset
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/artifacts/testset-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/artifacts/testset-%j.err
# Usage: sbatch --account=m4012 tools/run_score_test_set.sh NOTEBOOK TEST_INPUT PREDICTIONS_OUT
set -euo pipefail
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
.conda-env/bin/python tools/score_test_set.py "$@"
