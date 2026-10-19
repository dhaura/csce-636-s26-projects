#!/bin/bash
# Run one or more job files (see run_queue.py) one after another on ONE 4-GPU node.
# Usage (from hwx/):
#   salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 01:00:00 \
#     srun -n 1 -c 128 -G 4 bash slurm/run_jobs_node.sh slurm/jobs/round3_pilot.txt
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # hwx/
export PATH="$PWD/.conda-env/bin:$PATH"
export TORCH_HOME="$SCRATCH/.cache/torch" HF_HOME="$SCRATCH/.cache/huggingface"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
nvidia-smi -L
for jobs in "$@"; do
    python -u slurm/run_queue.py "$jobs" --gpus 4
done
echo "$(date +%T) all done"
