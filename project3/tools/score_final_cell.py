#!/usr/bin/env python3
"""Run the notebook's final cell exactly as graded and score its output files.

In a fresh folder, it places the final models from MODEL_DIR and the held-out codewords (the first 100,000 test
codewords per code, seeds as in the notebook, written in the instructor's input format) under the official test-file
names, executes the final cell's source there in a fresh Python process, checks the output files' format strictly,
and scores them with the known UME vectors. With --threads, the cell is run once per CPU thread count and the outputs
are compared, to check that the decisions do not depend on the machine's parallelism.
With --gpu, the cell runs with the GPU visible (otherwise CPU only), and --codes restricts the codes written.
Usage: score_final_cell.py NOTEBOOK.ipynb MODEL_DIR [--size N] [--threads 1,8,32] [--codes 1,2,...] [--gpu]
       [--models GLOB]  (model files copied from MODEL_DIR; default final_model_*_code*.keras)
"""
import argparse, os, pickle, shutil, subprocess, sys, tempfile
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codes import CODES, check_output, correct, sample  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("notebook"); parser.add_argument("model_dir")
parser.add_argument("--size", type=int, default=100_000)
parser.add_argument("--threads", default="")
parser.add_argument("--codes", default="")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--models", default="final_model_*_code*.keras", help="glob of the model files to copy")
args = parser.parse_args()
CODES_RUN = [int(c) for c in args.codes.split(",")] if args.codes else list(CODES)

cells = [c for c in nbformat.read(args.notebook, as_version=4).cells if c.cell_type == "code"]
final = cells[-1].source
assert final.startswith("# ===== Final cell"), "last code cell is not the final cell"
script = "\n".join(line.replace("%pip", "pass  # %pip") if "%pip" in line else line for line in final.splitlines())

truth = {}
work = Path(tempfile.mkdtemp(prefix="final_cell_", dir=ROOT / "artifacts"))
for code in CODES_RUN:
    y, e = sample(code, args.size, np.random.default_rng([3602, code]))
    truth[code] = e
    with open(work / f"Noisy-Codeword-Test-for-Code-{code}.pkl", "wb") as f:
        pickle.dump([[float(v) for v in row] for row in y], f)
models = sorted(Path(args.model_dir).glob(args.models))
for m in models:
    shutil.copy(m, work / m.name)
(work / "final_cell.py").write_text(script)
print(f"Folder {work}: models {[m.name for m in models]}, {args.size:,} codewords per code {CODES_RUN}")

runs = {}
for threads in ([int(t) for t in args.threads.split(",")] if args.threads else [None]):
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    if not args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if threads:
        env.update(TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS=str(threads))
    for code in CODES_RUN:
        (work / f"Decode-Results-Test-for-Code-{code}.pkl").unlink(missing_ok=True)
    out = subprocess.run([sys.executable, "final_cell.py"], cwd=work, env=env, capture_output=True, text=True)
    print(f"--- threads {threads or 'default'}: exit {out.returncode}\n{out.stdout.strip()}")
    if out.returncode:
        print(out.stderr[-3000:]); sys.exit(1)
    runs[threads] = {}
    for code in CODES_RUN:
        C = CODES[code]
        with open(work / f"Decode-Results-Test-for-Code-{code}.pkl", "rb") as f:
            results = pickle.load(f)
        check_output(results, args.size, C["n"])
        reported = np.zeros((args.size, C["n"]), bool)
        for r, s in enumerate(results):
            reported[r, list(s)] = True
        runs[threads][code] = results
        ok = correct(code, truth[code], reported)
        print(f"Code {code}: format OK, accuracy {ok.mean():.6f} ({(~ok).sum()} wrong of {args.size:,})")
keys = list(runs)
for k in keys[1:]:
    for code in CODES_RUN:
        same = runs[k][code] == runs[keys[0]][code]
        print(f"Code {code}: outputs with {k} threads identical to {keys[0] or 'default'} threads: {same}")
