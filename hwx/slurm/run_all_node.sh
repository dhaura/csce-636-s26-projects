#!/bin/bash
# Run the baseline notebook + all 5 Gessert et al. 2018 folds on ONE 4-GPU Perlmutter node, then summarize.
#   GPU0: baseline notebook -> fold 3     GPU1: fold 0 -> fold 4     GPU2: fold 1     GPU3: fold 2
# Usage (from hwx/), e.g. through the interactive QOS which usually starts within minutes:
#   salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 03:00:00 srun -n 1 -c 128 -G 4 bash slurm/run_all_node.sh
#SBATCH -J isic-all
#SBATCH -A m4012
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # hwx/
export PATH="$PWD/.conda-env/bin:$PATH"
export TORCH_HOME="$SCRATCH/.cache/torch" HF_HOME="$SCRATCH/.cache/huggingface"
export TQDM_DISABLE=1   # keep logs readable
LOGS=slurm/logs; mkdir -p "$LOGS"
nvidia-smi -L

fold() {  # fold <gpu> <k>
    CUDA_VISIBLE_DEVICES=$1 python -u literature_baseline_gessert2018.py --fold "$2" --num-workers 16 \
        > "$LOGS/gessert2018-fold$2.out" 2>&1
    echo "$(date +%T) fold $2 finished (exit $?)"
}
baseline() {
    CUDA_VISIBLE_DEVICES=0 jupyter nbconvert --to notebook --execute --inplace 01_baseline_resnet50.ipynb \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=isic-hwx > "$LOGS/baseline.out" 2>&1
    echo "$(date +%T) baseline finished (exit $?)"
}

echo "$(date +%T) starting"
(baseline; fold 0 3) &
(fold 1 0; fold 1 4) &
fold 2 1 &
fold 3 2 &
wait

python literature_baseline_gessert2018.py --summarize > "$LOGS/gessert2018-summary.out" 2>&1
echo "$(date +%T) all done"
