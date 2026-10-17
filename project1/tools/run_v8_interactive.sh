#!/bin/bash
# Train all 9 V8 settings at once (1 seed each) in NERSC's interactive GPU queue: 3 nodes x 3 GPUs.
# salloc gives up if the nodes are not granted within 10 minutes.
set -euo pipefail
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project1
salloc --immediate=600 --qos=interactive --constraint=gpu --account=m4012_g --nodes=3 --time=04:00:00 \
    srun --ntasks=9 --ntasks-per-node=3 --gpus-per-task=1 --cpus-per-task=32 --gpu-bind=single:1 \
    bash tools/run_v8_interactive_task.sh
