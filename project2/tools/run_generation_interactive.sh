#!/bin/bash
# Run the notebook's data-generation cells (sections 2-3) on one CPU node from NERSC's interactive queue.
# salloc gives up if the node is not granted within 10 minutes.
set -euo pipefail
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MHEIGHT_RUN=generation
salloc --immediate=600 --qos=interactive --constraint=cpu --account=m4012 --nodes=1 --time=04:00:00 \
    srun --ntasks=1 --cpus-per-task=256 .conda-env/bin/python tools/run_notebook_prefix.py "AUG_PATH = "
