#!/usr/bin/env python
"""Compare finished multi-fold runs on the official test set, with paired per-fold differences.

Each run is a directory with fold<k>/metrics.json (written by train_classifier.py or
literature_baseline_gessert2018.py). Runs are given as names under results/ or as paths.
The first run is the reference for the paired differences (same folds, same seed).

    python compare_runs.py resnet50_ce resnet50_wce gessert2018_densenet121
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from isic.data import CLASSES
from isic.metrics import compute_metrics

HWX_DIR = Path(__file__).resolve().parent
METRICS = ["balanced_acc", "accuracy", "macro_f1", "macro_auc", "mel_sensitivity", "mel_specificity"]


def load_run(run: str) -> dict:
    root = Path(run) if Path(run).is_dir() else HWX_DIR / "results" / run
    folds = {}
    for d in sorted(root.glob("fold[0-9]")):
        if (d / "metrics.json").exists():
            folds[d.name] = json.loads((d / "metrics.json").read_text())["best_checkpoint"]
    if not folds:
        raise SystemExit(f"no finished folds under {root}")
    # Ensemble = average of the fold models' test probabilities.
    preds = [pd.read_csv(root / f / "pred_official_test.csv") for f in folds]
    ens = compute_metrics(preds[0]["label"].map(CLASSES.index).to_numpy(),
                          np.mean([p[CLASSES].to_numpy() for p in preds], axis=0))
    return {"name": root.name, "folds": folds, "ensemble": ens}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--split", default="official_test", choices=["official_test", "cv_val", "official_val"])
    p.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    args = p.parse_args()
    runs = [load_run(r) for r in args.runs]
    s = args.split
    out = {"split": s, "runs": {}}

    print(f"=== {s}: mean +- std over folds (n folds), plus {s == 'official_test' and 'fold-ensemble' or ''} ===")
    rows = {}
    for r in runs:
        vals = {m: np.array([f[s][m] for f in r["folds"].values()]) for m in METRICS}
        rows[r["name"]] = {m: f"{v.mean():.4f} +- {v.std():.4f}" for m, v in vals.items()}
        rows[r["name"]]["n"] = len(r["folds"])
        out["runs"][r["name"]] = {"mean": {m: float(v.mean()) for m, v in vals.items()},
                                  "std": {m: float(v.std()) for m, v in vals.items()},
                                  "ensemble_official_test": r["ensemble"]}
    print(pd.DataFrame(rows).T.to_string())

    print(f"\n=== {s}: mean per-class recall over folds ===")
    recall = {r["name"]: {c: np.mean([f[s]["per_class_recall"][c] for f in r["folds"].values()]) for c in CLASSES}
              for r in runs}
    print(pd.DataFrame(recall).T.round(3).to_string())

    if s == "official_test":
        print("\n=== fold-ensemble (average of fold models' probabilities) on official_test ===")
        print(pd.DataFrame({r["name"]: {m: r["ensemble"][m] for m in METRICS} for r in runs}).T.round(4).to_string())

    ref = runs[0]
    for r in runs[1:]:
        common = sorted(set(ref["folds"]) & set(r["folds"]))
        if not common:
            continue
        print(f"\n=== paired difference {r['name']} - {ref['name']} on {s} ({len(common)} common folds) ===")
        diffs = {}
        for m in METRICS:
            d = np.array([r["folds"][f][s][m] - ref["folds"][f][s][m] for f in common])
            diffs[m] = {"mean_diff": float(d.mean()), "std_diff": float(d.std()), "folds_better": int((d > 0).sum())}
            print(f"  {m:<16s} {d.mean():+.4f} +- {d.std():.4f}   better on {(d > 0).sum()}/{len(d)} folds")
        out.setdefault("paired_vs_" + ref["name"], {})[r["name"]] = diffs

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
