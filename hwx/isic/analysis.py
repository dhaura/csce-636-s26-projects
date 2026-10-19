"""Read finished runs under results/ and summarise them (used by the version notebooks)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES
from .metrics import compute_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"
METRICS = ["balanced_acc", "accuracy", "macro_f1", "macro_auc", "mel_sensitivity", "mel_specificity"]
SHORT = {"balanced_acc": "bal_acc", "accuracy": "acc", "macro_f1": "f1", "macro_auc": "auc",
         "mel_sensitivity": "mel_sens", "mel_specificity": "mel_spec"}


def folds(run):
    return sorted(int(d.name[4:]) for d in (RESULTS / run).glob("fold[0-9]") if (d / "metrics.json").exists())


def fold_metrics(run, k):
    return json.loads((RESULTS / run / f"fold{k}" / "metrics.json").read_text())


def per_fold(run, split="official_test"):
    """One row per fold with the main metrics on `split`."""
    rows = []
    for k in folds(run):
        m = fold_metrics(run, k)
        rows.append({"fold": k, **{SHORT[x]: m["best_checkpoint"][split][x] for x in METRICS},
                     "best_epoch": m.get("best_epoch")})
    return pd.DataFrame(rows).set_index("fold")


def fold_ensemble(run, split="official_test"):
    """Metrics of the average of all fold models' probabilities on `split` (official sets only)."""
    preds = [pd.read_csv(RESULTS / run / f"fold{k}" / f"pred_{split}.csv") for k in folds(run)]
    labels = preds[0]["label"].map(CLASSES.index).to_numpy()
    return compute_metrics(labels, np.mean([p[CLASSES].to_numpy() for p in preds], axis=0))


def summary(runs, split="official_test", names=None, ensemble=True):
    """Mean ± std over folds per run; optionally the fold-ensemble balanced accuracy (test only)."""
    rows = {}
    for i, run in enumerate(runs):
        df = per_fold(run, split).drop(columns="best_epoch")
        row = {c: f"{df[c].mean():.3f} ± {df[c].std(ddof=0):.3f}" for c in df.columns}
        if ensemble and split == "official_test":
            row["ensemble_bal_acc"] = round(fold_ensemble(run)["balanced_acc"], 3)
        row["n_folds"] = len(df)
        rows[names[i] if names else run] = row
    return pd.DataFrame(rows).T


def paired(run, ref, split="official_test", metric="balanced_acc"):
    """Per-fold difference run - ref: (mean, std, folds better, n)."""
    common = sorted(set(folds(run)) & set(folds(ref)))
    d = np.array([fold_metrics(run, k)["best_checkpoint"][split][metric] - fold_metrics(ref, k)["best_checkpoint"][split][metric]
                  for k in common])
    return d.mean(), d.std(), int((d > 0).sum()), len(d)


def paired_table(runs, ref, names=None, splits=("cv_val", "official_test"), metrics=("balanced_acc", "mel_sensitivity", "macro_auc")):
    rows = {}
    for i, run in enumerate(runs):
        row = {}
        for s in splits:
            for m in metrics:
                mu, sd, better, n = paired(run, ref, s, m)
                row[f"{'cv' if s == 'cv_val' else 'test'}_{SHORT[m]}"] = f"{mu:+.3f} ± {sd:.3f} ({better}/{n})"
        rows[names[i] if names else run] = row
    return pd.DataFrame(rows).T


def per_class_recall(run, split="official_test"):
    return pd.DataFrame([fold_metrics(run, k)["best_checkpoint"][split]["per_class_recall"] for k in folds(run)]).mean()


def confusion(run, k=0, split="official_test"):
    return pd.DataFrame(fold_metrics(run, k)["best_checkpoint"][split]["confusion_matrix"], index=CLASSES, columns=CLASSES)


def history(run):
    """Per-epoch CV metrics of every fold (training runs only)."""
    hs = []
    for k in folds(run):
        f = RESULTS / run / f"fold{k}" / "history.csv"
        if f.exists():
            hs.append(pd.read_csv(f).assign(fold=k))
    return pd.concat(hs, ignore_index=True) if hs else pd.DataFrame()


def tau(run):
    f = RESULTS / run / "tau_sweep.json"
    return json.loads(f.read_text()) if f.exists() else None


def plot_curves(ax, runs, names=None, metric="val_balanced_acc"):
    """Mean over folds with a min-max band, per run."""
    for i, run in enumerate(runs):
        h = history(run).dropna(subset=[metric])
        g = h.groupby("epoch")[metric].agg(["mean", "min", "max"])
        ax.plot(g.index, g["mean"], lw=2, label=names[i] if names else run)
        ax.fill_between(g.index, g["min"], g["max"], alpha=0.15)
    ax.set_xlabel("epoch"); ax.set_ylabel("CV balanced accuracy"); ax.grid(alpha=0.3); ax.legend()


def plot_confusion(ax, run, k=0, title=None):
    cm = confusion(run, k).to_numpy()
    frac = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45); ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title or f"{run} · fold {k}")
