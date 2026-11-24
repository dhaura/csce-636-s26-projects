#!/usr/bin/env python3
"""Make a test pack for trying the submitted notebook's final cell on Google Colab.

Writes to OUT (default artifacts/colab_test):
  Noisy-Codeword-Test-for-Code-{1..6}.pkl   SIZE fresh codewords per code (seeds [3700, code], used nowhere else),
                                            in the instructor's input format, under the official test-file names
  answers/Answers-for-Code-{c}.pkl          per codeword: the required and the forbidden locations
  reference/Decode-Results-Test-for-Code-{c}.pkl
                                            the outputs of the submitted final cell and model, run here
  check_results.py                          checks the Colab outputs: format, accuracy, identical to the reference
  README.txt                                how to use it
and OUT.zip with all of it (not the model file).
Usage: make_colab_test.py [--out DIR] [--size N] [--notebook NB] [--model MODEL]
"""
import argparse, os, pickle, shutil, subprocess, sys, tempfile
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codes import CODES, check_output, sample  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=str(ROOT / "artifacts/colab_test"))
parser.add_argument("--size", type=int, default=1000)
parser.add_argument("--notebook", default=str(ROOT / "artifacts/submission/project3_v4_executed.ipynb"))
parser.add_argument("--model", default=str(ROOT / "artifacts/submission/final_model_v4.keras"))
args = parser.parse_args()
out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
(out / "answers").mkdir(parents=True); (out / "reference").mkdir()

for code, C in CODES.items():
    y, e = sample(code, args.size, np.random.default_rng([3700, code]))
    with open(out / f"Noisy-Codeword-Test-for-Code-{code}.pkl", "wb") as f:
        pickle.dump([[float(v) for v in row] for row in y], f)
    required = [set(int(j) for j in np.flatnonzero(row)) for row in np.abs(e) > C["Delta"]]
    forbidden = [set(int(j) for j in np.flatnonzero(row)) for row in e == 0]
    with open(out / "answers" / f"Answers-for-Code-{code}.pkl", "wb") as f:
        pickle.dump({"required": required, "forbidden": forbidden}, f)

# Reference outputs: the submitted final cell, unchanged (without the Colab install line), with the submitted model.
cells = [c for c in nbformat.read(args.notebook, as_version=4).cells if c.cell_type == "code"]
final = cells[-1].source
assert final.startswith("# ===== Final cell"), "last code cell is not the final cell"
script = "\n".join(line.replace("%pip", "pass  # %pip") if "%pip" in line else line for line in final.splitlines())
work = Path(tempfile.mkdtemp(prefix="colab_ref_", dir=ROOT / "artifacts"))
for code in CODES:
    shutil.copy(out / f"Noisy-Codeword-Test-for-Code-{code}.pkl", work)
shutil.copy(args.model, work)
(work / "final_cell.py").write_text(script)
env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
run = subprocess.run([sys.executable, "final_cell.py"], cwd=work, env=env, capture_output=True, text=True)
print(run.stdout.strip())
if run.returncode:
    print(run.stderr[-3000:]); sys.exit(1)
for code, C in CODES.items():
    name = f"Decode-Results-Test-for-Code-{code}.pkl"
    with open(work / name, "rb") as f:
        check_output(pickle.load(f), args.size, C["n"])
    shutil.move(work / name, out / "reference" / name)
shutil.rmtree(work)

(out / "check_results.py").write_text(r'''"""Check the six Decode-Results-Test-for-Code-j.pkl files written by the final cell.

Run in the folder of the test files, after the final cell:  !python check_results.py
For each code: the strict output format (a list with one set of int locations in [0, n) per input codeword), the
accuracy against answers/, and whether the outputs are identical to reference/ (the same model run on Perlmutter).
"""
import pickle
from pathlib import Path

N = {1: 10, 2: 16, 3: 16, 4: 20, 5: 40, 6: 80}
all_ok = True
for code, n in N.items():
    with open(f"Noisy-Codeword-Test-for-Code-{code}.pkl", "rb") as f:
        num_inputs = len(pickle.load(f))
    path = Path(f"Decode-Results-Test-for-Code-{code}.pkl")
    if not path.exists():
        print(f"Code {code}: {path} MISSING"); all_ok = False; continue
    with open(path, "rb") as f:
        results = pickle.load(f)
    problem = None
    if type(results) is not list or len(results) != num_inputs:
        problem = f"expected a list of {num_inputs} sets, got {type(results).__name__} of length {len(results)}"
    else:
        for r, s in enumerate(results):
            if type(s) is not set:
                problem = f"output {r} is a {type(s).__name__}, not a set"; break
            bad = [j for j in s if type(j) is not int or not 0 <= j < n]
            if bad:
                problem = f"output {r} contains {bad[0]!r}, not an int location in [0, {n})"; break
    if problem:
        print(f"Code {code}: INVALID FORMAT: {problem}"); all_ok = False; continue
    with open(f"answers/Answers-for-Code-{code}.pkl", "rb") as f:
        answers = pickle.load(f)
    right = sum(req <= s and not (s & forb) for s, req, forb in zip(results, answers["required"], answers["forbidden"]))
    line = f"Code {code}: format OK; accuracy {right / num_inputs:.4f} ({num_inputs - right} wrong of {num_inputs:,})"
    reference = Path(f"reference/Decode-Results-Test-for-Code-{code}.pkl")
    if reference.exists():
        with open(reference, "rb") as f:
            ref = pickle.load(f)
        differ = sum(a != b for a, b in zip(results, ref)) + abs(len(results) - len(ref))
        line += f"; identical to the reference outputs: {differ == 0}" + (f" ({differ} codewords differ)" if differ else "")
        all_ok &= differ == 0
    print(line)
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
''')

(out / "README.txt").write_text(f"""Test pack for the final cell of project3_v4_executed.ipynb ({args.size:,} fresh codewords per code, drawn
from the distribution of the project description with seeds the notebook never uses; the same size as the
instructor's Sample Test Set).

On Google Colab:
 1. Upload project3_v4_executed.ipynb and open it; Runtime > Change runtime type > a GPU (T4 is enough).
 2. Upload final_model_v4.keras and this zip to /content (the Files pane), then in a new cell:  !unzip -o colab_test.zip
    (the six Noisy-Codeword-Test-for-Code-j.pkl files must sit next to final_model_v4.keras).
 3. Run only the final cell (the last cell), unchanged. It installs the training versions of TensorFlow and Keras;
    if it asks for a restart, use Runtime > Restart session and run it again. Its lines should say /GPU:0.
 4. In a new cell:  !python check_results.py
    Expected: "format OK" for all six codes, "identical to the reference outputs: True" for all six, and
    ALL CHECKS PASSED. The reference outputs were made by the same final cell and model on Perlmutter, so this
    is the check the grader makes: the outputs do not change between machines.
""")

zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
print(f"Wrote {out} and {zip_path}")
