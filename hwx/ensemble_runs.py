#!/usr/bin/env python
"""Average several finished runs fold by fold and save the result as a new run.

For each fold k, the members' saved probabilities (pred_<split>.csv) are averaged and scored,
so the ensemble can be judged on the CV folds like any single model. compare_runs.py then
also reports the average of all members x folds on the official test set.

    python ensemble_runs.py --name ens5_res448_tta20_la resnet50_ce_res448_tta20_la convnext_s_res448_tta20_la \
        effv2_s_res448_tta20_la dinov2_b14_res448_tta20_la vit_b16_res448_tta20_la
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from isic.data import CLASSES
from isic.engine import save_json, save_predictions
from isic.metrics import compute_metrics, format_metrics

HWX_DIR = Path(__file__).resolve().parent
SPLITS = ["cv_val", "official_val", "official_test"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="member runs under results/")
    p.add_argument("--name", required=True, help="output run name under results/")
    args = p.parse_args()

    roots = [HWX_DIR / "results" / r for r in args.runs]
    folds = sorted(set.intersection(*[{int(d.name[4:]) for d in r.glob("fold[0-9]")} for r in roots]))
    if not folds:
        raise SystemExit("members share no folds")
    for k in folds:
        out = HWX_DIR / "results" / args.name / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        res = {"config": {"method": "fold-wise probability average", "members": args.runs}, "best_checkpoint": {}}
        for s in SPLITS:
            preds = [pd.read_csv(r / f"fold{k}" / f"pred_{s}.csv") for r in roots]
            assert all((q["image"] == preds[0]["image"]).all() for q in preds), f"image order differs in fold {k} {s}"
            probs = np.mean([q[CLASSES].to_numpy() for q in preds], axis=0)
            labels = preds[0]["label"].map(CLASSES.index).to_numpy()
            res["best_checkpoint"][s] = compute_metrics(labels, probs)
            save_predictions(out / f"pred_{s}.csv", preds[0]["image"], probs, labels)
        save_json(res, out / "metrics.json")
        print(f"fold {k}\n" + format_metrics(res["best_checkpoint"]["cv_val"], "cv_val") + "\n"
              + format_metrics(res["best_checkpoint"]["official_test"], "official_test"))
    print(f"wrote results/{args.name} ({len(args.runs)} members x {len(folds)} folds)")


if __name__ == "__main__":
    main()
