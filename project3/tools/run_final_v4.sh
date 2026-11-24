#!/bin/bash
# The final V4 run: execute project3_v4.ipynb with P3_TRAIN=0 (loads every trained network from artifacts/notebook_v4,
# runs all evaluations, chooses the final models and writes final_model_v4_code{1..6}.keras) on one GPU of a node
# from NERSC's interactive queue. Output: artifacts/notebook_v4/project3_v4_executed.ipynb.
set -euo pipefail
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project3
export P3_OUT=artifacts/notebook_v4 P3_TRAIN=0
cp project3_v4.ipynb "$P3_OUT/project3_v4_source.ipynb"
salloc --immediate=1800 --qos=interactive --constraint=gpu --account=m4012_g --nodes=1 --time=04:00:00 \
    srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=64 bash -c '
  module unload cudatoolkit 2>/dev/null || true
  export LD_LIBRARY_PATH="$(ls -d $PWD/.conda-env/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="$PWD/.conda-env/bin:$PATH" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
  echo "$(date) final run on $(hostname)"
  python tools/execute_notebook.py "$P3_OUT/project3_v4_source.ipynb" "$P3_OUT/project3_v4_executed.ipynb"
  echo "$(date) done"'
