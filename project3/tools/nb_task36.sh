#!/bin/bash
# One task of tools/final36_driver.sh (one GPU): runs project3_v4.ipynb with P3_TRAIN=1 for the (code, version) at
# position SLURM_PROCID of NB_TASKS, which trains (or resumes) that network; the executed copy is kept per session.
set -uo pipefail
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project3
read -ra T <<< "$NB_TASKS"
IFS=: read -r CODE VERSION <<< "${T[$SLURM_PROCID]}"
module unload cudatoolkit 2>/dev/null || true
export LD_LIBRARY_PATH="$(ls -d "$PWD"/.conda-env/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$PWD/.conda-env/bin:$PATH" PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export P3_TRAIN=1 P3_CODES=$CODE P3_VERSIONS36=$VERSION P3_STRESS_MILLIONS=0 P3_HARD_MILLIONS=0
LOG="$P3_OUT/train_${VERSION}_code${CODE}.log"
echo "$(date) session $NB_SESSION task $SLURM_PROCID on $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?}: code $CODE version $VERSION" >> "$LOG"
python tools/execute_notebook.py "${NB_SOURCE:-artifacts/notebook_v4/project3_v4_train.ipynb}" "$P3_OUT/train_${VERSION}_code${CODE}_s${NB_SESSION}.ipynb" >> "$LOG" 2>&1
echo "$(date) notebook finished (exit $?)" >> "$LOG"
