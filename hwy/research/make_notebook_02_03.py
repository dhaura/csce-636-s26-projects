"""Generate 02_round2_combination_noise.ipynb and 03_round3_model_soup.ipynb (run from hwy/: python research/make_notebook_02_03.py)."""
import json
from pathlib import Path

HWY_DIR = Path(__file__).resolve().parents[1]

SETUP = r'''
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HWY_DIR = Path.cwd() if (Path.cwd() / "isic").is_dir() else Path.cwd().parent
sys.path.insert(0, str(HWY_DIR)); sys.path.insert(0, str(HWY_DIR / "research"))
from isic import analysis as A
from isic.data import CLASSES

pd.set_option("display.width", 200)
la = lambda r: f"{r}_tta20_la"
BASE, R1_RECT, ADOPTED = "hwx/convnext_s_kd_res448_tta20_la", "cnx_kd_r1_rect_tta20_la", "cnx_kd_r1_rect_ema_tta20_la"

def run(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=HWY_DIR)
'''

NB02 = [
("markdown", r'''
# HW Y · 02 — Round 2: combine the adopted changes, tune drop path, measure noise

**Round 1 adopted** drop path 0.2 and full-frame 448 × 600 input. The plan (`research/plan.md`, written before these runs)
specified four runs, all with full-frame input and an EMA copy:
- drop path 0.1 / 0.2 / 0.3
- drop path 0.2 again with seed 1, as a noise floor

**Rules:**
1. Pick the drop path rate by CV.
2. Keep the chosen run only if it beats full frame alone (paired CV gain > 0, ≥ 3/5 folds).
3. Re-test EMA with a cross-fitted CV score that removes best-epoch selection bias.
4. Call gains smaller than the seed-to-seed spread "within noise".
'''),
("code", SETUP + r'''
RUNS = {"full frame + dp 0.1": "cnx_kd_r2_rect_dp01", "full frame + dp 0.2": "cnx_kd_r2_rect_dp02",
        "full frame + dp 0.3": "cnx_kd_r2_rect_dp03", "full frame + dp 0.2, seed 1": "cnx_kd_r2_rect_dp02_s1"}
RUN_TRAINING = False   # True on a GPU node: one fold of the seed-1 replicate (~42 min)
if RUN_TRAINING:
    run("python scripts/train_classifier.py --run-name cnx_kd_r2_rect_dp02_s1 --model convnext_small.fb_in22k_ft_in1k --lr 2e-4 "
        "--img-size 448 --img-width 600 --soft-labels results/hwx/soft_labels/bb5_calib_oof.csv --kd-alpha 0.5 --ema 0.999 "
        "--drop-path 0.2 --seed 1 --fold 0")
'''),
("code", r'''
names = ["baseline (HW X)", "full frame (Round 1)"] + list(RUNS)
runs = [BASE, R1_RECT] + [la(r) for r in RUNS.values()]
display(A.summary(runs, names=names)[["bal_acc", "auc", "mel_sens"]])
display(A.summary(runs, split="cv_val", names=names, ensemble=False)[["bal_acc"]].rename(columns={"bal_acc": "cv_bal_acc"}))
A.paired_table([la(r) for r in RUNS.values()], R1_RECT, names=[f"{n} vs full frame alone" for n in RUNS])
'''),
("markdown", "## Noise floor: the same recipe with a different seed"),
("code", r'''
a, b = la(RUNS["full frame + dp 0.2"]), la(RUNS["full frame + dp 0.2, seed 1"])
t = pd.DataFrame({s: [A.per_fold(a, s).bal_acc.mean(), A.per_fold(b, s).bal_acc.mean()] for s in ("cv_val", "official_test")}, index=["seed 0", "seed 1"])
t.loc["seed 1 - seed 0"] = t.loc["seed 1"] - t.loc["seed 0"]
display(t.round(4))
fig, ax = plt.subplots(figsize=(7, 3.2))
for j, (s, col) in enumerate((("cv_val", "C0"), ("official_test", "C3"))):
    ax.scatter(A.per_fold(a, s).bal_acc, [j - 0.12] * 5, color=col, marker="o", label=f"{s} seed 0")
    ax.scatter(A.per_fold(b, s).bal_acc, [j + 0.12] * 5, color=col, marker="x", label=f"{s} seed 1")
ax.set_yticks([0, 1], ["CV", "test"]); ax.set_xlabel("balanced accuracy per fold"); ax.legend(fontsize=8, ncol=2); ax.grid(axis="x", alpha=.3)
ax.set_title("Same recipe, two seeds"); plt.tight_layout()
'''),
("markdown", "## EMA re-tested with a cross-fitted (selection-free) CV score"),
("code", r'''
from round2_analysis import cross_fitted
rows = []
for n, r in RUNS.items():
    raw, ema = cross_fitted(r), cross_fitted(f"{r}_ema")
    best_raw, best_ema = A.history(r).groupby("fold").val_balanced_acc.max().mean(), A.history(r).groupby("fold").val_balanced_acc_ema.max().mean()
    rows.append({"run": n, "best-epoch raw": best_raw, "best-epoch EMA": best_ema, "cross-fitted raw": raw.mean(), "cross-fitted EMA": ema.mean(),
                 "cross-fitted EMA-raw": (ema - raw).mean(), "folds": f"{int(((ema - raw) > 0).sum())}/5",
                 "test EMA-raw (TTA+LA)": A.paired(la(f"{r}_ema"), la(r))[0]})
pd.DataFrame(rows).set_index("run").round(4)
'''),
("markdown", r'''
## Takeaways
* **Drop path 0.2** is the best rate (CV 0.894). Combined with full frame, it does **not** beat full frame alone on CV
  (−0.002), so full frame alone stays.
* **EMA** wins the cross-fitted CV score in 3 of 4 runs, so it is adopted, but only by +0.0002. The best-epoch scores
  (first two columns) are 0.01 higher than the cross-fitted ones; that difference is the selection bonus.
* **Noise floor:** the seed-1 replicate moves the 5-fold mean by −0.007 on CV and −0.005 on test. Every gain from Rounds
  1–2 is this size. **Round-2 model: full frame + EMA, CV 0.894, test 0.867.**
'''),
]

NB03 = [
("markdown", r'''
# HW Y · 03 — Round 3: model soups (weight averaging), still one network

A model soup (Wortsman et al., ICML 2022) averages the *weights* of several fine-tuned copies of one network. All our
ConvNeXt-S runs start from the same ImageNet-22k checkpoint and use LayerNorm, so a per-fold average is well defined.
The result is one network with the same inference cost. The plan, written before any soup was evaluated, uses uniform
soups with no selection:
- **10 full-frame fine-tunes:** 5 runs × raw/EMA weights.
- **6 square fine-tunes:** 3 runs × raw/EMA weights.

Each soup is compared with the Round-2 model by the usual CV rule. `scripts/model_soup.py` writes each soup as an
ordinary run.
'''),
("code", SETUP + r'''
SOUPS = {"soup of 10 full-frame fine-tunes": ("soup_rect10", "ens_rect10_tta20_la",
          ["cnx_kd_r1_rect", "cnx_kd_r2_rect_dp01", "cnx_kd_r2_rect_dp02", "cnx_kd_r2_rect_dp03", "cnx_kd_r2_rect_dp02_s1"]),
         "soup of 6 square fine-tunes": ("soup_square6", "ens_square6_tta20_la", ["cnx_kd_r1_base", "cnx_kd_r1_dp02", "cnx_kd_r1_lld"])}
RUN_SOUP = False   # True on a GPU node: rebuild one soup (a minute) and its TTA-20 evaluation (~18 min)
if RUN_SOUP:
    members = " ".join(f"{m} {m}_ema" for m in SOUPS["soup of 6 square fine-tunes"][2])
    run(f"python scripts/model_soup.py --name soup_square6 --members {members}")
    run("python scripts/evaluate_tta.py soup_square6 --crops 5 && python scripts/posthoc_logit_adjustment.py soup_square6_tta20")
'''),
("code", r'''
rows = []
for n, (soup, ens, members) in SOUPS.items():
    ing = [la(m) for m in members] + [la(f"{m}_ema") for m in members]
    for label, r in [(f"{n} (1 network)", la(soup)), (f"  its ingredients, mean of {len(ing)}", None), (f"  prediction ensemble of the {len(ing)} ({len(ing)} networks)", ens)]:
        if r is None:
            rows.append({"model": label, "cv": np.mean([A.per_fold(x, "cv_val").bal_acc.mean() for x in ing]),
                         "test": np.mean([A.per_fold(x).bal_acc.mean() for x in ing])})
        else:
            rows.append({"model": label, "cv": A.per_fold(r, "cv_val").bal_acc.mean(), "test": A.per_fold(r).bal_acc.mean()})
rows.insert(0, {"model": "adopted Round-2 model (full frame + EMA)", "cv": A.per_fold(ADOPTED, "cv_val").bal_acc.mean(), "test": A.per_fold(ADOPTED).bal_acc.mean()})
display(pd.DataFrame(rows).set_index("model").round(4))
A.paired_table([la(s) for s, _, _ in SOUPS.values()], ADOPTED, names=[f"{n} vs adopted" for n in SOUPS])
'''),
("code", r'''
from round3_analysis import bootstrap_ci
lo, hi = bootstrap_ci(ADOPTED)
print(f"adopted model test {A.per_fold(ADOPTED).bal_acc.mean():.4f}, 95% CI {lo:.3f}-{hi:.3f} (baseline {A.per_fold(BASE).bal_acc.mean():.4f})")
pd.DataFrame({"baseline": A.per_class_recall(BASE), "adopted": A.per_class_recall(ADOPTED),
              "soup of 10": A.per_class_recall(la("soup_rect10"))}).round(3).T
'''),
("markdown", r'''
## Takeaways
* **Neither soup passes the rule** (CV −0.014 and −0.010 against the Round-2 model, 0/5 folds). On test, each soup is
  at or slightly above its ingredients' mean (0.870 vs 0.868; 0.869 vs 0.867).
* The CV deficit (about 0.012 below the ingredients) matches the best-epoch selection bonus found in Rounds 1–2. Each
  ingredient is its fold's best epoch; the soup is not selected at all. This is the same bias that hid EMA.
* Prediction ensembles of the same networks reach test 0.875 / 0.883. The gap to them, and to the 0.892 teacher
  ensemble, comes from combining different networks, not from training one network better.
* **Adopted model after 3 rounds:** full frame + EMA, CV 0.894, test 0.867 (95% CI 0.841–0.890). That is +0.003 over the
  baseline, within seed noise.
'''),
]


def write(cells, name):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python (isic-hwx)", "language": "python", "name": "isic-hwx"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(cells):
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n"), "id": f"cell-{i:02d}"}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    (HWY_DIR / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", HWY_DIR / name)


if __name__ == "__main__":
    write(NB02, "02_round2_combination_noise.ipynb")
    write(NB03, "03_round3_model_soup.ipynb")
