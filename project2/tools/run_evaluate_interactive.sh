#!/bin/bash
# On one interactive CPU node: execute NOTEBOOK with MHEIGHT_TRAIN=0 (assembly, full logs and results) into
# artifacts/$RUN, then score $RUN's model against the comparison model on the held-out splits and the probe sets.
# Usage: NOTEBOOK=project2_v7.ipynb RUN=p2_v7 MODEL=final_model_v7.keras COMPARE=artifacts/p2_v6/final_model_v6.keras bash tools/run_evaluate_interactive.sh
# (COMPARE may list several models; NODE=gpu uses an interactive GPU node, see below.)
set -euo pipefail
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project2
: "${NOTEBOOK:?}" "${RUN:?}" "${MODEL:?}" "${COMPARE:?}"
export PATH="$PWD/.conda-env/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MHEIGHT_RUN=$RUN MHEIGHT_SPLIT=80-10-10 MHEIGHT_TRAIN=0
# NODE=gpu runs on an interactive GPU node instead (GPUs hidden; same EPYC 7763 CPUs) when no CPU node is free.
if [ "${NODE:-cpu}" = gpu ]; then NODE_ARGS="--constraint=gpu --account=m4012_g --gpus=1"; CPUS=128; else NODE_ARGS="--constraint=cpu --account=m4012"; CPUS=256; fi
salloc --immediate=600 --qos=interactive $NODE_ARGS --nodes=1 --time=02:00:00 srun --ntasks=1 --cpus-per-task=$CPUS bash -c "
  set -e
  jupyter nbconvert --to notebook --execute $NOTEBOOK --ExecutePreprocessor.kernel_name=csce636-p1 --ExecutePreprocessor.timeout=-1 \
      --output-dir artifacts/$RUN --output ${NOTEBOOK%.ipynb}_executed.ipynb > artifacts/$RUN/assembly.log 2>&1
  echo \"\$(date) assembly done\"
  python tools/score_split.py artifacts/$RUN/$MODEL $COMPARE > artifacts/$RUN/eval_split.txt 2> artifacts/$RUN/eval_split.err
  python tools/offcluster_probe.py artifacts/$RUN/$MODEL $COMPARE > artifacts/$RUN/eval_offcluster.txt 2> artifacts/$RUN/eval_offcluster.err
  python tools/uniform_probe.py artifacts/$RUN/$MODEL $COMPARE > artifacts/$RUN/eval_uniform.txt 2> artifacts/$RUN/eval_uniform.err
  echo \"\$(date) evaluation done\"
"
