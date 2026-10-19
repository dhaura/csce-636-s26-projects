#!/bin/bash
# Gessert et al. 2018 literature baseline: 5-fold CV, one array task per fold.
# Submit from hwx/:  sbatch slurm/gessert2018_cv.sbatch
# Afterwards:        .conda-env/bin/python literature_baseline_gessert2018.py --summarize
#SBATCH -J isic-gessert2018
#SBATCH -A m4012
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH --array=0-4
#SBATCH -o slurm/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.conda-env/bin:$PATH"
export TORCH_HOME="$SCRATCH/.cache/torch" HF_HOME="$SCRATCH/.cache/huggingface"

nvidia-smi -L
python literature_baseline_gessert2018.py --fold "$SLURM_ARRAY_TASK_ID" --num-workers 16 "$@"
