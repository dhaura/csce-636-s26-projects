#!/bin/bash
# Experiment: does Gessert-style inverse-frequency loss weighting help our ResNet-50 baseline?
# 10 runs = {plain CE, weighted CE} x 5 folds, same seed -> paired comparison. ~20 min on one 4-GPU node.
# Usage (from hwx/):
#   salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 01:00:00 srun -n 1 -c 128 -G 4 bash slurm/run_loss_weighting_node.sh
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # hwx/
export PATH="$PWD/.conda-env/bin:$PATH"
export TORCH_HOME="$SCRATCH/.cache/torch" HF_HOME="$SCRATCH/.cache/huggingface"
export TQDM_DISABLE=1
LOGS=slurm/logs; mkdir -p "$LOGS"
nvidia-smi -L

JOBS=()
for k in 0 1 2 3 4; do
    JOBS+=("resnet50_ce none $k" "resnet50_wce inverse $k")
done

worker() {  # worker <gpu>: runs every 4th job
    local gpu=$1
    for ((i = gpu; i < ${#JOBS[@]}; i += 4)); do
        read -r name weighting k <<< "${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$gpu python -u train_classifier.py --run-name "$name" --loss-weighting "$weighting" \
            --fold "$k" --num-workers 16 > "$LOGS/$name-fold$k.out" 2>&1
        echo "$(date +%T) $name fold $k finished (exit $?) on GPU $gpu"
    done
}

echo "$(date +%T) starting ${#JOBS[@]} runs"
for g in 0 1 2 3; do worker "$g" & done
wait

python compare_runs.py resnet50_ce resnet50_wce gessert2018_densenet121 \
    --out results/comparisons/loss_weighting.json > "$LOGS/compare_loss_weighting.out" 2>&1
echo "$(date +%T) all done"
