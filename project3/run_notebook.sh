#!/bin/bash
#SBATCH --job-name=p3-notebook
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project3
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project3/artifacts/notebook-%j.out
# Executes the submission notebook ($NOTEBOOK, default project3_v2.ipynb), from this folder, on one shared A100 with the P3_* settings from
# the environment (sbatch passes it through), writing the executed copy to $P3_OUT as $NB_NAME.
# Training jobs (one per version and code):
#   P3_OUT=artifacts/notebook_v2 P3_TRAIN=1 P3_VERSIONS=V2 P3_CODES=1 P3_STRESS_MILLIONS=0 NB_NAME=train_V2_code1.ipynb sbatch --account=m4012_g run_notebook.sh
# Final executed notebook (loads all trained networks, runs every evaluation):
#   P3_OUT=artifacts/notebook_v2 P3_TRAIN=0 NB_NAME=project3_v2_executed.ipynb sbatch --account=m4012_g run_notebook.sh
set -euo pipefail
module unload cudatoolkit 2>/dev/null || true
ENV_DIR="$PWD/.conda-env"
NVIDIA_LIBS=$(ls -d "$ENV_DIR"/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:)
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ENV_DIR/bin:$PATH" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
NOTEBOOK=${NOTEBOOK:-project3_v2.ipynb}
: "${P3_OUT:?set P3_OUT}"; mkdir -p "$P3_OUT"
echo "$(date) $NOTEBOOK on $(hostname): P3_TRAIN=${P3_TRAIN:-1} P3_VERSIONS=${P3_VERSIONS:-default} P3_CODES=${P3_CODES:-1,2} -> $P3_OUT/$NB_NAME"
python tools/execute_notebook.py "$NOTEBOOK" "$P3_OUT/$NB_NAME"
echo "$(date) done"
