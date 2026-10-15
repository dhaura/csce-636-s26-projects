#!/usr/bin/env python
"""Out-of-fold ensemble soft labels for knowledge distillation.

Each training image gets the calibrated ensemble probability from the fold models that never saw it:
every member's cv_val log-probs are temperature-scaled (one T per member, fitted on all OOF
predictions), averaged, and renormalised. No logit adjustment: the targets keep the training prior.

Caveat: the soft label of an image in fold j comes from models trained on the other folds, which
include any student's own CV fold k, so a distilled student's CV score is mildly optimistic.

    python make_soft_labels.py --members resnet50_ce_res448_tta20 convnext_s_res448_tta20 effv2_s_res448_tta20 \
        dinov2_b14_res448_tta20 vit_b16_res448_tta20 --out results/soft_labels/bb5_calib_oof.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from isic.data import CLASSES
from isic.decision import EPS, balanced_accuracy, fit_temperature, log_softmax

HWX_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    Z, imgs, ys = {}, None, None
    for m in args.members:
        dfs = [pd.read_csv(HWX_DIR / "results" / m / f"fold{k}" / "pred_cv_val.csv") for k in range(5)]
        df = pd.concat(dfs, ignore_index=True)
        if imgs is None:
            imgs, ys = df["image"], df["label"].map(CLASSES.index).to_numpy()
        assert (df["image"] == imgs).all()
        Z[m] = np.log(np.clip(df[CLASSES].to_numpy(), EPS, 1.0))
    T = {m: fit_temperature(Z[m], ys) for m in args.members}
    ens = np.exp(log_softmax(np.mean([log_softmax(Z[m] / T[m]) for m in args.members], 0)))
    out = pd.DataFrame(ens, columns=CLASSES); out.insert(0, "image", imgs.values); out["label"] = [CLASSES[i] for i in ys]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    conf = ens.max(1)
    print("temperatures:", {m: round(t, 3) for m, t in T.items()})
    print(f"{len(out)} images | OOF balanced acc (argmax, unadjusted) {balanced_accuracy(ys, ens.argmax(1), len(CLASSES)):.4f} | "
          f"mean prob on true class {ens[np.arange(len(ys)), ys].mean():.3f} | mean max prob {conf.mean():.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
