#!/bin/bash
# Execute 01_baseline_resnet50.ipynb on a Perlmutter GPU node; outputs are written into the notebook.
# Submit from hwx/:  sbatch slurm/baseline_notebook.sbatch
#SBATCH -J isic-baseline
#SBATCH -A m4012
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -o slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.conda-env/bin:$PATH"
export TORCH_HOME="$SCRATCH/.cache/torch" HF_HOME="$SCRATCH/.cache/huggingface"

nvidia-smi -L
jupyter nbconvert --to notebook --execute --inplace 01_baseline_resnet50.ipynb \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=isic-hwx
echo "Finished. Metrics: results/baseline_resnet50/metrics.json"
