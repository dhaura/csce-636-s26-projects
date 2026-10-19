#!/usr/bin/env python
"""Post-hoc logit adjustment of a finished run's saved predictions (no retraining).

    A. K. Menon et al., "Long-tail learning via logit adjustment", ICLR 2021.

A model trained with plain cross-entropy on imbalanced data learns p(y|x) under the
training class prior pi. For balanced accuracy we instead predict

    argmax_y  log p(y|x) - tau * log pi_y

with pi taken from each fold's training split. tau = 0 leaves the model unchanged;
tau = 1 is the textbook correction for a calibrated model. One global tau is tuned by
mean balanced accuracy on the CV folds (pred_cv_val.csv) and never on the test set.

The adjusted predictions are written as a new run (results/<run>_la/fold<k>/) in the
same format as train_classifier.py, so compare_runs.py can include it.

    python posthoc_logit_adjustment.py resnet50_ce
    python compare_runs.py resnet50_ce resnet50_ce_la resnet50_wce
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from isic.data import CLASSES, add_cv_folds, class_counts, load_split
from isic.engine import save_json, save_predictions
from isic.metrics import compute_metrics

HWX_DIR = Path(__file__).resolve().parent
SPLITS = ["cv_val", "official_val", "official_test"]


def load_preds(path):
    df = pd.read_csv(path)
    return df["image"], df[CLASSES].to_numpy(), df["label"].map(CLASSES.index).to_numpy()


def adjust(probs, log_prior, tau):
    """Subtract tau * log prior from log-probabilities and renormalise."""
    z = np.log(np.clip(probs, 1e-12, None)) - tau * log_prior
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="finished run under results/ (needs fold<k>/pred_<split>.csv)")
    p.add_argument("--taus", type=float, nargs="+", default=[round(t, 2) for t in np.arange(-1.0, 3.01, 0.1)])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=0)
    p.add_argument("--out-name", default=None, help="default: <run>_la")
    args = p.parse_args()

    src = HWX_DIR / "results" / args.run
    folds = sorted(int(d.name[4:]) for d in src.glob("fold[0-9]") if (d / "pred_cv_val.csv").exists())
    if not folds:
        raise SystemExit(f"no fold predictions under {src}")

    train_all = add_cv_folds(load_split("train"), n_folds=args.n_folds, seed=args.cv_seed)
    data = {}
    for k in folds:
        counts = class_counts(train_all[train_all.fold != k].label)
        data[k] = {"log_prior": np.log(counts / counts.sum())}
        data[k].update({s: load_preds(src / f"fold{k}" / f"pred_{s}.csv") for s in SPLITS})

    def fold_metrics(split, tau):
        return [compute_metrics(data[k][split][2], adjust(data[k][split][1], data[k]["log_prior"], tau)) for k in folds]

    sweep = []
    for tau in args.taus:
        cv, test = fold_metrics("cv_val", tau), fold_metrics("official_test", tau)
        row = {"tau": tau}
        for name, ms in (("cv", cv), ("test", test)):
            for m in ("balanced_acc", "accuracy", "mel_sensitivity", "mel_specificity"):
                row[f"{name}_{m}"] = float(np.mean([x[m] for x in ms]))
        sweep.append(row)
    table = pd.DataFrame(sweep).set_index("tau")
    best_tau = float(table["cv_balanced_acc"].idxmax())

    print(f"[{args.run}] tau sweep, mean over {len(folds)} folds "
          "(tau is selected on cv_* only; test_* columns are shown for analysis, not used for selection)")
    print(table.round(4).to_string())
    print(f"\nselected tau = {best_tau} (best mean cv balanced accuracy)")

    out_root = HWX_DIR / "results" / (args.out_name or f"{args.run}_la")
    for k in folds:
        (out_root / f"fold{k}").mkdir(parents=True, exist_ok=True)
        res = {"config": {"method": "posthoc_logit_adjustment", "source_run": args.run, "tau": best_tau,
                          "selected_on": "mean cv_val balanced accuracy over folds"},
               "best_checkpoint": {}}
        for s in SPLITS:
            images, probs, labels = data[k][s]
            adj = adjust(probs, data[k]["log_prior"], best_tau)
            res["best_checkpoint"][s] = compute_metrics(labels, adj)
            save_predictions(out_root / f"fold{k}" / f"pred_{s}.csv", images, adj, labels)
        save_json(res, out_root / f"fold{k}" / "metrics.json")
    save_json({"source_run": args.run, "selected_tau": best_tau, "sweep": table.reset_index().to_dict("records")},
              out_root / "tau_sweep.json")
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
