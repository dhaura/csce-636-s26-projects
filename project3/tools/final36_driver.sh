#!/bin/bash
# Train the final codes 3-6 networks THROUGH THE NOTEBOOK (P3_TRAIN=1) in consecutive 4-hour interactive sessions.
# TASKS="code:version ..." lists the trainings (one GPU each, at most 4 per session, in the order given).
# A task is finished when its history exists and its backup folder does not (Keras deletes the backup when training
# completes); an interrupted training resumes from its backup in the next session.
# Usage: TASKS="6:H2 5:H2 4:H2 3:H2" [P3_OUT=artifacts/notebook_v4] [MAX_SESSIONS=12] [SESSION_START=0] tools/final36_driver.sh
# (SESSION_START: sessions already run by an earlier driver for these tasks; only the session numbers in the logs and
# executed copies, and the MAX_SESSIONS count, depend on it.)
set -uo pipefail
cd /pscratch/sd/d/dhaura/repos/csce-636-s26-projects/project3
export P3_OUT=${P3_OUT:-artifacts/notebook_v4}
read -ra ALL <<< "${TASKS:?set TASKS}"
mkdir -p "$P3_OUT"
session=${SESSION_START:-0}
while [ "$session" -lt "${MAX_SESSIONS:-12}" ]; do
  todo=()
  for t in "${ALL[@]}"; do
    IFS=: read -r c v <<< "$t"
    if [ -f "$P3_OUT/history_${v}_code${c}.json" ] && [ ! -d "$P3_OUT/backup_${v}_code${c}" ]; then continue; fi
    todo+=("$t")
  done
  if [ ${#todo[@]} -eq 0 ]; then echo "$(date) all tasks finished"; exit 0; fi
  run=("${todo[@]:0:4}")
  # The interactive QOS allows 2 jobs per user: wait for a free slot.
  until [ "$(squeue -u "$USER" -h -o %q | grep -c interactive)" -lt 2 ]; do sleep 60; done
  export NB_TASKS="${run[*]}" NB_SESSION=$((session + 1))
  echo "$(date) session $NB_SESSION: ${run[*]}"
  start=$(date +%s)
  salloc --immediate=1800 --qos=interactive --constraint=gpu --account=m4012_g --nodes=1 --time=04:00:00 \
      srun --ntasks=${#run[@]} --gpus-per-task=1 --cpus-per-task=32 --gpu-bind=single:1 bash tools/nb_task36.sh
  status=$?
  if [ $(( $(date +%s) - start )) -lt 300 ]; then
    quick=$(( ${quick:-0} + 1 ))
    echo "$(date) no session (allocation failed or tasks ended at once, exit $status; $quick in a row)"
    [ "$quick" -ge 6 ] && { echo "$(date) giving up"; exit 1; }
    sleep 300; continue
  fi
  quick=0
  session=$((session + 1))
  echo "$(date) session $session ended (exit $status)"
  sleep 30
done
echo "$(date) stopped after $session sessions"
