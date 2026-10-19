"""Generate 07_colour_constancy_round5.ipynb (run from hwx/: python research/make_notebook_07.py)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "07_colour_constancy_round5.ipynb"

cells = [
("markdown", r'''
# 07 — Colour constancy and colour augmentation (round 5): a pre-registered negative result

**Hypothesis** (`research/round5_plan.md`, written before the runs): part of the CV → test drop comes from acquisition shift
(other clinics and devices in the test set). **Shades-of-Grey colour constancy** (p = 6, as in the ISIC 2019 winner) and
**stronger colour augmentation** (jitter 0.3 / 0.3 / 0.3, hue 0.1) should therefore help on test.

**Design.** One factor at a time on the optimised ResNet-50 (448 px, plain CE, 30 epochs, 5 folds, TTA-20 + logit adjustment):
the primary run combines both changes; two ablations isolate each. CV (HAM10000 only) cannot see the shift, so it is a
**non-inferiority check** (margin 0.005); adoption for later rounds depends on CV alone, and the test comparison is a single
pre-registered contrast.

**Reproduce:** `salloc ... srun ... bash slurm/run_jobs_node.sh slurm/jobs/round5_train.txt slurm/jobs/round5_eval.txt`
(~85 min), then `python research/round5_analysis.py`.
'''),
("code", r'''
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HWX_DIR = Path.cwd() if (Path.cwd() / "isic").is_dir() else Path.cwd().parent
sys.path.insert(0, str(HWX_DIR))
from isic import analysis as A
from isic.data import CLASSES, ShadesOfGray, load_split

pd.set_option("display.width", 200)
RUN_TRAINING = False   # True on a GPU node: re-create one fold of the primary variant (~18 min)

def run(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=HWX_DIR)

if RUN_TRAINING:
    run("python train_classifier.py --run-name resnet50_cc_coloraug_res448 --img-size 448 --color-constancy --augment color --fold 0")
    run("python evaluate_tta.py resnet50_cc_coloraug_res448 --crops 5 --folds 0")
    run("python posthoc_logit_adjustment.py resnet50_cc_coloraug_res448_tta20")
'''),
("markdown", "## 1. What Shades of Grey does to the images, and how different train and test colours are"),
("code", r'''
sog = ShadesOfGray()
tr, te = load_split("train"), load_split("test")
examples = [tr[tr.dx == c].iloc[0].path for c in ["MEL", "NV", "BKL", "VASC"]] + [te.iloc[i].path for i in (0, 1)]
fig, axes = plt.subplots(2, len(examples), figsize=(3 * len(examples), 5.2))
for j, p in enumerate(examples):
    im = Image.open(p).convert("RGB")
    axes[0, j].imshow(im); axes[1, j].imshow(sog(im))
    axes[0, j].set_title(("train · " + tr.set_index("path").dx.get(p, "")) if p in set(tr.path) else "test", fontsize=9)
    for a in axes[:, j]: a.axis("off")
axes[0, 0].text(-40, 225, "original", rotation=90, va="center"); axes[1, 0].text(-40, 225, "Shades of Grey", rotation=90, va="center")
plt.tight_layout()
'''),
("code", r'''
def chroma(df, cc, n=300):
    out = []
    for p in df.sample(n, random_state=0).path:
        im = Image.open(p).convert("RGB")
        x = np.asarray(sog(im) if cc else im, dtype=np.float32).mean((0, 1))
        out.append(x / x.sum())
    return np.array(out)
rows = []
for cc in (False, True):
    a, b = chroma(tr, cc), chroma(te, cc)
    rows.append({"images": "Shades of Grey" if cc else "original", "train chromaticity spread (r,g,b)": a.std(0).round(4),
                 "test chromaticity spread": b.std(0).round(4), "train-test mean chromaticity gap": np.abs(a.mean(0) - b.mean(0)).sum().round(4)})
pd.DataFrame(rows).set_index("images")
'''),
("markdown", "## 2. Results (pre-registered analysis)"),
("code", r'''
BASE = "resnet50_ce_res448_tta20_la"
RUNS = {"baseline (no CC, basic aug)": BASE,
        "primary: CC + strong colour aug": "resnet50_cc_coloraug_res448_tta20_la",
        "ablation: CC only": "resnet50_cc_res448_tta20_la",
        "ablation: strong colour aug only": "resnet50_coloraug_res448_tta20_la"}
display(A.summary(list(RUNS.values()), names=list(RUNS)))
A.paired_table(list(RUNS.values())[1:], BASE, names=list(RUNS)[1:])
'''),
("code", r'''
rec = pd.DataFrame({n: A.per_class_recall(r) for n, r in RUNS.items()})
fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
rec.plot.bar(ax=axes[0], rot=0); axes[0].set_ylim(0.5, 1); axes[0].grid(axis="y", alpha=.3)
axes[0].set_title("Per-class recall, test (5-fold mean)"); axes[0].legend(fontsize=8)
gap = pd.Series({n: A.per_fold(r, "cv_val").bal_acc.mean() - A.per_fold(r).bal_acc.mean() for n, r in RUNS.items()})
gap.plot.barh(ax=axes[1], color="grey"); axes[1].axvline(0, c="k", lw=1)
axes[1].set_title("CV minus test balanced accuracy (positive = drop on test)"); axes[1].grid(axis="x", alpha=.3)
plt.tight_layout()
'''),
("markdown", r'''
## Takeaways
* **Hypothesis rejected.** Every colour variant is worse than the baseline on the test set on **all 5 folds** (CC + strong
  aug −0.016, CC only −0.032, strong aug only −0.019) and none passes the CV non-inferiority check, so nothing is adopted.
* **Colour is signal here, not just nuisance.** Vascular lesions are red, keratoses brown, dermatofibromas pinkish-brown;
  normalising the global colour cast or jittering hue by ±0.1 removes information the classifier uses. HAM10000 and the test
  set also differ little in mean colour (chromaticity gap 0.003–0.012 before CC, depending on the image sample; table above),
  so there was little shift for CC to fix.
* **The premise was only partly right.** The optimised ResNet-50 has no CV → test drop (test is 0.003 *higher*); the drop we
  targeted belonged to other models. Pre-registering the rule kept us from searching on the test set for a colour setting that
  happens to work.
'''),
]


def main():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python (isic-hwx)", "language": "python", "name": "isic-hwx"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(cells):
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n"), "id": f"cell-{i:02d}"}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
