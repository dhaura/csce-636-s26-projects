#!/usr/bin/env python
"""Run a list of jobs on a node's GPUs: one job per GPU, the next job goes to the first GPU that frees up.

Job file: one job per line, "<log-name> <shell command>"; blank lines and # comments are ignored.
Put the longest jobs first so the tail of the queue is short. Each job gets CUDA_VISIBLE_DEVICES
set and writes to <log-dir>/<log-name>.out. Exits non-zero if any job failed.

    python slurm/run_queue.py slurm/jobs/round2_train.txt --gpus 4
"""
import argparse
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobfile")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--log-dir", default="slurm/logs")
    args = p.parse_args()

    jobs = queue.Queue()
    n = 0
    for line in Path(args.jobfile).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, cmd = line.split(maxsplit=1)
            jobs.put((name, cmd))
            n += 1
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    print(f"{time.strftime('%H:%M:%S')} {args.jobfile}: {n} jobs on {args.gpus} GPUs", flush=True)

    failed, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                name, cmd = jobs.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            with open(Path(args.log_dir) / f"{name}.out", "w") as log:
                rc = subprocess.call(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            print(f"{time.strftime('%H:%M:%S')} {name} finished (exit {rc}) on GPU {gpu} "
                  f"in {(time.time() - t0) / 60:.1f} min", flush=True)
            if rc:
                with lock:
                    failed.append(name)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(args.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"{time.strftime('%H:%M:%S')} done: {n - len(failed)}/{n} succeeded" + (f"; failed: {failed}" if failed else ""),
          flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
