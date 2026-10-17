#!/bin/bash
# One task of the interactive V8 run: trains the setting given by SLURM_PROCID on this task's GPU (1 seed).
# Launched by tools/run_v8_interactive.sh via srun; one task per GPU.
set -euo pipefail
SETTINGS_LIST=(4-2 4-3 4-4 4-5 5-2 5-3 5-4 6-2 6-3)
S=${SETTINGS_LIST[$SLURM_PROCID]}
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
module unload cudatoolkit 2>/dev/null || true
ENV_DIR="$PWD/.conda-env"
export LD_LIBRARY_PATH="$(ls -d "$ENV_DIR"/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ENV_DIR/bin:$PATH" PYTHONUNBUFFERED=1
export MHEIGHT_RUN=submission_v8_1seed MHEIGHT_SETTINGS=$S MHEIGHT_SEEDS=1 MHEIGHT_SPLIT=80-10-10 MHEIGHT_TRAIN=1 MHEIGHT_DATASET=extended
RUN_DIR="artifacts/$MHEIGHT_RUN"; mkdir -p "$RUN_DIR"
echo "$(date) task $SLURM_PROCID on $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?}: setting $S"
jupyter nbconvert --to notebook --execute project1_v8.ipynb --ExecutePreprocessor.kernel_name=csce636-p1 \
    --ExecutePreprocessor.timeout=-1 --output-dir "$RUN_DIR" --output "project1_v8_executed_$S.ipynb" \
    > "$RUN_DIR/task_$S.out" 2> "$RUN_DIR/task_$S.err"
echo "$(date) setting $S finished"
