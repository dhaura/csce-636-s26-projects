#!/bin/bash
#SBATCH --job-name=mheight-notebook
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --chdir=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
#SBATCH --output=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/artifacts/notebook-%j.out
#SBATCH --error=/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1/artifacts/notebook-%j.err

# Executes $NOTEBOOK (default project1_v1.ipynb) headlessly on one shared A100 with the project environment
# (TensorFlow 2.21 / Keras 3). Everything from a run goes to artifacts/$MHEIGHT_RUN/:
# the executed notebook (<notebook>_executed.ipynb), the Slurm .out/.err logs, the model,
# history, costs and split indices. The source notebook itself is not modified.
#
# Submit with, e.g.:
#   sbatch --account=m4012_g run_notebook.sh                                   # project1_v1.ipynb -> dnn_v1_finetune
#   MHEIGHT_PRETRAIN_EPOCHS=100 MHEIGHT_FINETUNE_EPOCHS=150 sbatch --account=m4012_g run_notebook.sh
#   NOTEBOOK=other.ipynb MHEIGHT_RUN=other_run sbatch --account=m4012_g run_notebook.sh
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1}"
NOTEBOOK="${NOTEBOOK:-project1_v1.ipynb}"
export MHEIGHT_DATASET="${MHEIGHT_DATASET:-extended}"
export MHEIGHT_RUN="${MHEIGHT_RUN:-dnn_v1_finetune}"
cd "$PROJECT_DIR"

# TensorFlow 2.21 is built for CUDA 12: drop the default CUDA 13 module and use the
# CUDA 12 / cuDNN 9 libraries pip installed into the environment.
module unload cudatoolkit 2>/dev/null || true
ENV_DIR="$PROJECT_DIR/.conda-env"
NVIDIA_LIBS=$(ls -d "$ENV_DIR"/lib/python3.11/site-packages/nvidia/*/lib | paste -sd:)
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

python -c 'import tensorflow as tf, numpy; print("TF", tf.__version__, "NumPy", numpy.__version__, "GPUs", tf.config.list_physical_devices("GPU"))'
RUN_DIR="artifacts/$MHEIGHT_RUN"
mkdir -p "$RUN_DIR"

for ext in out err; do
    mv "artifacts/notebook-$SLURM_JOB_ID.$ext" "$RUN_DIR/notebook-$SLURM_JOB_ID.$ext"
done
echo "Notebook: $NOTEBOOK  Dataset: $MHEIGHT_DATASET  Run: $MHEIGHT_RUN"
jupyter nbconvert --to notebook --execute "$NOTEBOOK" \
    --ExecutePreprocessor.kernel_name=csce636-p1 \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir "$RUN_DIR" --output "${NOTEBOOK%.ipynb}_executed.ipynb"
echo "Executed notebook: $RUN_DIR/${NOTEBOOK%.ipynb}_executed.ipynb"
