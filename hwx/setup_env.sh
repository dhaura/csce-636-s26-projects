#!/usr/bin/env bash
# Create the project conda env at hwx/.conda-env and register it as a Jupyter kernel.
# Usage (from anywhere):  bash hwx/setup_env.sh
set -euo pipefail

HWX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$HWX_DIR/.conda-env"
KERNEL_NAME="isic-hwx"

# On NERSC Perlmutter conda comes from a module; elsewhere assume conda is on PATH.
if ! command -v conda >/dev/null 2>&1; then
    module load conda
fi

if [ -d "$ENV_DIR" ]; then
    echo "Env already exists at $ENV_DIR - updating it"
    conda env update -p "$ENV_DIR" -f "$HWX_DIR/environment.yml" --prune
else
    conda env create -p "$ENV_DIR" -f "$HWX_DIR/environment.yml"
fi

# Keep pretrained-weight caches on scratch instead of the small $HOME quota.
CACHE_DIR="${SCRATCH:-$HWX_DIR}/.cache"
conda env config vars set -p "$ENV_DIR" \
    TORCH_HOME="$CACHE_DIR/torch" \
    HF_HOME="$CACHE_DIR/huggingface" >/dev/null

"$ENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" \
    --display-name "Python (isic-hwx)" \
    --env TORCH_HOME "$CACHE_DIR/torch" --env HF_HOME "$CACHE_DIR/huggingface"

"$ENV_DIR/bin/python" -c "import torch, torchvision, timm; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| timm', timm.__version__, '| CUDA available:', torch.cuda.is_available())"

echo
echo "Done. Activate with:  module load conda && conda activate $ENV_DIR"
echo "Jupyter kernel:       Python (isic-hwx)"
