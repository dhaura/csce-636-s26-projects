#!/bin/bash
# Round 2 of baseline optimisation on ONE 4-GPU node:
#   1. train 4 one-factor variants of resnet50_ce x 5 folds (res448, res384, ep60, trivial aug)
#   2. test-time augmentation (4 and 20 views) on every run's checkpoints, including resnet50_ce
#   3. post-hoc logit adjustment of every run and TTA variant (tau tuned on CV folds)
#   4. comparisons on the CV folds (for selection) and on the official test set (for reporting)
# Usage (from hwx/):
#   salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 02:30:00 srun -n 1 -c 128 -G 4 bash slurm/run_round2_node.sh
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # hwx/
export PATH="$PWD/.conda-env/bin:$PATH"
export TORCH_HOME="$SCRATCH/.cache/torch" HF_HOME="$SCRATCH/.cache/huggingface"
export TQDM_DISABLE=1
LOGS=slurm/logs; mkdir -p "$LOGS"
nvidia-smi -L

python -u slurm/run_queue.py slurm/jobs/round2_train.txt --gpus 4
python -u slurm/run_queue.py slurm/jobs/round2_tta.txt --gpus 4

echo "$(date +%T) logit adjustment"
RUNS=()
for r in resnet50_ce resnet50_ce_res384 resnet50_ce_res448 resnet50_ce_ep60 resnet50_ce_trivial; do
    for v in "" _tta4 _tta20; do
        python posthoc_logit_adjustment.py "$r$v" > "$LOGS/la-$r$v.out" 2>&1 \
            && RUNS+=("$r${v}_la") || echo "logit adjustment failed for $r$v"
    done
done

echo "$(date +%T) comparisons"
python compare_runs.py "${RUNS[@]}" --split cv_val --out results/comparisons/round2_cv.json > "$LOGS/compare_round2_cv.out" 2>&1
python compare_runs.py "${RUNS[@]}" --split official_test --out results/comparisons/round2_test.json > "$LOGS/compare_round2_test.out" 2>&1
echo "$(date +%T) all done"
