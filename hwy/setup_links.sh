#!/usr/bin/env bash
# Share the Homework X data, conda env and finished runs with hwy/ (no copies). Run once from anywhere.
set -euo pipefail
HWY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ln -sfn ../../hwx/data/isic2018 "$HWY/data/isic2018"   # ISIC 2018 Task 3 images + labels
ln -sfn ../hwx/.conda-env "$HWY/.conda-env"             # same environment as Homework X
mkdir -p "$HWY/results"
ln -sfn ../../hwx/results "$HWY/results/hwx"           # Homework X runs (baseline), read as results/hwx/<run>
echo "links ready under $HWY"
