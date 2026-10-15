#!/usr/bin/env python
"""Fit an ensemble decision layer on out-of-fold (OOF) predictions and compare variants (nested CV).

Every training image has exactly one prediction from a model that never saw it (its fold's
cv_val prediction), so the members' cv_val files form an OOF set of 10,015 images.

Protocol: for each fold k, a decision layer is fitted on the OOF predictions of the other 4 folds
and scored on fold k (nested CV, used for selection). The same layer is applied to the fold-k
models' test predictions (reporting only). `--save METHOD` writes that method as a new run under
results/ (fold-wise, like any other run) so compare_runs.py and the notebooks can use it.

    python decision_layer.py --members resnet50_ce_res448_tta20 convnext_s_res448_tta20 effv2_s_res448_tta20 \
        dinov2_b14_res448_tta20 vit_b16_res448_tta20 --tag bb5
    python decision_layer.py ... --tag bb5 --save calib_bias
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from isic.data import CLASSES, add_cv_folds, class_counts, load_split
from isic.decision import (EPS, balanced_accuracy, fit_class_bias, fit_stacker, fit_tau, fit_temperature,
                           log_softmax)
from isic.engine import save_json, save_predictions
from isic.metrics import compute_metrics

HWX_DIR = Path(__file__).resolve().parent
R = HWX_DIR / "results"
NC = len(CLASSES)
SPLITS = ["cv_val", "official_val", "official_test"]

METHODS = {
    "member_tau": "per-member tau, then average probabilities (rounds 2-3)",
    "avg_tau": "average log-probs, single tau",
    "avg_bias": "average log-probs, per-class bias",
    "calib_tau1": "temperature-calibrate members, average, tau = 1 (no tuning)",
    "calib_tau": "temperature-calibrate members, average, single tau",
    "calib_bias": "temperature-calibrate members, average, per-class bias",
    "stack": "class-balanced logistic-regression stacker on member log-probs",
}


def load_run(run, k, split):
    df = pd.read_csv(R / run / f"fold{k}" / f"pred_{split}.csv")
    return df["image"], np.log(np.clip(df[CLASSES].to_numpy(), EPS, 1.0)), df["label"].map(CLASSES.index).to_numpy()


def build_layer(method, Z, y, log_prior):
    """Fit `method` on member log-probs Z (dict member -> (n, C)); return f(dict) -> log-probs."""
    members = list(Z)
    if method == "member_tau":
        taus = {m: fit_tau(Z[m], y, log_prior) for m in members}
        return lambda W: np.log(np.mean([np.exp(log_softmax(W[m] - taus[m] * log_prior)) for m in members], 0) + EPS), {"taus": taus}
    if method in ("avg_tau", "avg_bias"):
        g = lambda W: log_softmax(np.mean([W[m] for m in members], 0))
        zf = g(Z); t = fit_tau(zf, y, log_prior)
        if method == "avg_tau":
            return lambda W: g(W) - t * log_prior, {"tau": t}
        b = fit_class_bias(zf, y, -t * log_prior)
        return lambda W: g(W) + b, {"tau_init": t, "bias": b.tolist()}
    if method.startswith("calib"):
        T = {m: fit_temperature(Z[m], y) for m in members}
        g = lambda W: log_softmax(np.mean([log_softmax(W[m] / T[m]) for m in members], 0))
        zf = g(Z)
        if method == "calib_tau1":
            return lambda W: g(W) - log_prior, {"temperatures": T}
        t = fit_tau(zf, y, log_prior)
        if method == "calib_tau":
            return lambda W: g(W) - t * log_prior, {"temperatures": T, "tau": t}
        b = fit_class_bias(zf, y, -t * log_prior)
        return lambda W: g(W) + b, {"temperatures": T, "tau_init": t, "bias": b.tolist()}
    if method == "stack":
        lr = fit_stacker(np.concatenate([Z[m] for m in members], 1), y)
        return lambda W: lr.predict_log_proba(np.concatenate([W[m] for m in members], 1)), {}
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", nargs="+", required=True, help="runs with raw (unadjusted) predictions")
    p.add_argument("--tag", required=True, help="name for outputs, e.g. bb5")
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--save", default=None, help="write this method as results/dl_<tag>_<method>/")
    args = p.parse_args()

    data = {s: {k: {} for k in range(5)} for s in SPLITS}
    labels, images = {s: {} for s in SPLITS}, {s: {} for s in SPLITS}
    for m in args.members:
        for k in range(5):
            for s in SPLITS:
                images[s][k], data[s][k][m], labels[s][k] = load_run(m, k, s)
    train_all = add_cv_folds(load_split("train"))
    counts = class_counts(train_all.label)
    log_prior = np.log(counts / counts.sum())

    rows, details = [], {}
    methods = [args.save] if args.save else args.methods
    for method in methods:
        t0 = time.time()
        cv, te, te_probs, out = [], [], [], {}
        for k in range(5):
            fit = [j for j in range(5) if j != k]
            Z = {m: np.concatenate([data["cv_val"][j][m] for j in fit]) for m in args.members}
            y = np.concatenate([labels["cv_val"][j] for j in fit])
            f, info = build_layer(method, Z, y, log_prior)
            outs = {s: f(data[s][k]) for s in SPLITS}
            cv.append(balanced_accuracy(labels["cv_val"][k], outs["cv_val"].argmax(1), NC))
            te.append(balanced_accuracy(labels["official_test"][k], outs["official_test"].argmax(1), NC))
            te_probs.append(np.exp(log_softmax(outs["official_test"])))
            out[k] = (outs, info)
        allm = balanced_accuracy(labels["official_test"][0], np.mean(te_probs, 0).argmax(1), NC)
        rows.append({"method": method, "description": METHODS[method], "cv_nested": np.mean(cv), "cv_std": np.std(cv),
                     "test_per_fold": np.mean(te), "test_std": np.std(te), "test_all_models": allm, "sec": time.time() - t0})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        details[method] = {k: out[k][1] for k in out}

        if args.save:
            name = f"dl_{args.tag}_{method}"
            for k, (outs, info) in out.items():
                d = R / name / f"fold{k}"; d.mkdir(parents=True, exist_ok=True)
                res = {"config": {"method": f"decision layer: {METHODS[method]}", "members": args.members,
                                  "fitted_on": "OOF predictions of the other 4 folds", "params": info}, "best_checkpoint": {}}
                for s in SPLITS:
                    pr = np.exp(log_softmax(outs[s]))
                    res["best_checkpoint"][s] = compute_metrics(labels[s][k], pr)
                    save_predictions(d / f"pred_{s}.csv", images[s][k], pr, labels[s][k])
                save_json(res, d / "metrics.json")
            print(f"wrote results/{name}")

    table = pd.DataFrame(rows).set_index("method")
    print("\n" + table.drop(columns="description").round(4).to_string())
    if not args.save:
        save_json({"members": args.members, "rows": table.reset_index().to_dict("records"),
                   "details": json.loads(json.dumps(details, default=float))},
                  R / "comparisons" / f"decision_layer_{args.tag}.json")


if __name__ == "__main__":
    main()
