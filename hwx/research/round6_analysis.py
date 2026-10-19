"""Round 6 analysis, following research/round6_plan.md. Run from hwx/ after the round-6 node job."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HWX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HWX))
from isic import analysis as A  # noqa: E402
from isic.data import CLASSES  # noqa: E402

PY = sys.executable
PAIRS = [("ConvNeXt-S", "convnext_s"), ("ResNet-50", "resnet50"), ("EfficientNetV2-S", "effv2_s"),
         ("DINOv2 ViT-B/14", "dinov2_b14"), ("ViT-B/16", "vit_b16")]
HARD = {"resnet50": "resnet50_ce_res448_tta20_la"}


def hard(key):
    return HARD.get(key, f"{key}_res448_tta20_la")


def kd(key):
    return f"{key}_kd_res448_tta20_la"


def sh(*args):
    subprocess.run([PY, *args], cwd=HWX, check=True, capture_output=True)


# 1. logit adjustment of the new students
for _, key in PAIRS[3:]:
    if not (A.RESULTS / kd(key)).exists():
        sh("posthoc_logit_adjustment.py", f"{key}_kd_res448_tta20")

# 2. primary: each student vs its hard-label twin (test, paired)
rows, diffs = [], []
for name, key in PAIRS:
    h, k = A.per_fold(hard(key)), A.per_fold(kd(key))
    mu, sd, better, n = A.paired(kd(key), hard(key))
    auc = A.paired(kd(key), hard(key), metric="macro_auc"); mel = A.paired(kd(key), hard(key), metric="mel_sensitivity")
    diffs += list(k.bal_acc.values - h.bal_acc.values)
    rows.append({"architecture": name, "round": 6 if key in ("dinov2_b14", "vit_b16") else 4, "hard": h.bal_acc.mean(),
                 "distilled": k.bal_acc.mean(), "gain": f"{mu:+.4f} ± {sd:.4f} ({better}/{n})",
                 "AUC gain": f"{auc[0]:+.4f} ({auc[2]}/{auc[3]})", "MEL sens gain": f"{mel[0]:+.4f} ({mel[2]}/{mel[3]})"})
print(pd.DataFrame(rows).set_index("architecture").round(4).to_string())
d = np.array(diffs)
print(f"\npooled over {len(d)} fold pairs: mean gain {d.mean():+.4f}, improved in {(d > 0).sum()}/{len(d)}")

# 3. ensembles (reporting only)
BB = ["resnet50_ce_res448_tta20", "convnext_s_res448_tta20", "effv2_s_res448_tta20_ts512", "dinov2_b14_res448_tta20", "vit_b16_res448_tta20"]
KD5 = [f"{key}_kd_res448_tta20" for _, key in PAIRS]
for tag, members in [("kd5", KD5), ("bb5kd5", BB + KD5)]:
    if not (A.RESULTS / f"dl_{tag}_calib_tau").exists():
        sh("decision_layer.py", "--members", *members, "--tag", tag, "--save", "calib_tau")


def all_models(run):
    ps = [pd.read_csv(A.RESULTS / run / f"fold{k}" / "pred_official_test.csv") for k in range(5)]
    return ps[0]["label"].map(CLASSES.index).to_numpy(), np.mean([p[CLASSES].to_numpy() for p in ps], 0).argmax(1)


def bacc(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(len(CLASSES))]))


y0, _ = all_models("dl_final8_calib_tau")
rng = np.random.default_rng(0)
B = [np.concatenate([rng.choice(np.where(y0 == c)[0], int((y0 == c).sum())) for c in range(7)]) for _ in range(2000)]
rows = []
for label, run in [("5 backbones (round 3)", "ens5_res448_tta20_la"), ("final, pre-registered (5 bb + 3 students)", "dl_final8_calib_tau"),
                   ("5 students (calib tau)", "dl_kd5_calib_tau"), ("5 backbones + 5 students (calib tau)", "dl_bb5kd5_calib_tau")]:
    _, p = all_models(run); bs = np.array([bacc(y0[b], p[b]) for b in B])
    cv = A.per_fold(run, "cv_val").bal_acc.mean()
    rows.append({"ensemble": label, "CV (students optimistic)": round(cv, 4), "test per fold": round(A.per_fold(run).bal_acc.mean(), 4),
                 "test all models": round(bacc(y0, p), 4), "95% CI": f"{np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}",
                 "P(>0.895)": round(float((bs > 0.895).mean()), 2)})
print("\n" + pd.DataFrame(rows).set_index("ensemble").to_string())
