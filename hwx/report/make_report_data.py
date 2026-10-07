"""Collect every number the report page shows from hwx/results into one JSON (stdout)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HWX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HWX))
from isic.data import CLASSES, add_cv_folds, class_counts, load_split  # noqa: E402
from isic.metrics import compute_metrics  # noqa: E402

R = HWX / "results"
KEYS = ["balanced_acc", "accuracy", "macro_f1", "macro_auc", "mel_sensitivity", "mel_specificity"]

# key -> (results run, run whose history.csv / training metadata to use)
RUNS = {
    "ce": ("resnet50_ce", "resnet50_ce"),
    "wce": ("resnet50_wce", "resnet50_wce"),
    "gessert": ("gessert2018_densenet121", "gessert2018_densenet121"),
    "opt": ("resnet50_ce_res448_tta20_la", "resnet50_ce_res448"),
}


def metrics(run, k):
    return json.loads((R / run / f"fold{k}" / "metrics.json").read_text())


def run_block(run, src):
    folds = []
    for k in range(5):
        m, s = metrics(run, k), metrics(src, k)
        b = m["best_checkpoint"]
        folds.append({"fold": k, "best_epoch": s.get("best_epoch"), "time_min": round(s.get("train_time_min", 0), 1),
                      "cv": {x: round(b["cv_val"][x], 4) for x in KEYS},
                      "test": {x: round(b["official_test"][x], 4) for x in KEYS},
                      "test_recall": {c: round(v, 4) for c, v in b["official_test"]["per_class_recall"].items()}})
    preds = [pd.read_csv(R / run / f"fold{k}" / "pred_official_test.csv") for k in range(5)]
    ens = compute_metrics(preds[0]["label"].map(CLASSES.index).to_numpy(), np.mean([p[CLASSES].to_numpy() for p in preds], axis=0))
    hist = pd.concat([pd.read_csv(R / src / f"fold{k}" / "history.csv") for k in range(5)]).dropna(subset=["val_balanced_acc"])
    h = hist.groupby("epoch")["val_balanced_acc"].agg(["mean", "min", "max"]).reset_index()
    return {"name": run, "folds": folds, "ensemble": {x: round(ens[x], 4) for x in KEYS},
            "curve": [[int(a), round(b, 4), round(c, 4), round(d, 4)] for a, b, c, d in h.values],
            "cm_fold0": metrics(run, 0)["best_checkpoint"]["official_test"]["confusion_matrix"]}


def summary(run):
    out = {}
    for split in ("cv_val", "official_test"):
        vals = {x: [metrics(run, k)["best_checkpoint"][split][x] for k in range(5)] for x in KEYS}
        out[split] = {x: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] for x, v in vals.items()}
    preds = [pd.read_csv(R / run / f"fold{k}" / "pred_official_test.csv") for k in range(5)]
    ens = compute_metrics(preds[0]["label"].map(CLASSES.index).to_numpy(), np.mean([p[CLASSES].to_numpy() for p in preds], axis=0))
    out["ensemble"] = {x: round(ens[x], 4) for x in KEYS}
    tau = R / run / "tau_sweep.json"
    out["tau"] = json.loads(tau.read_text())["selected_tau"] if tau.exists() else None
    return out


def paired(a, b, split, m="balanced_acc"):
    d = np.array([metrics(a, k)["best_checkpoint"][split][m] - metrics(b, k)["best_checkpoint"][split][m] for k in range(5)])
    return [round(float(d.mean()), 4), round(float(d.std()), 4), int((d > 0).sum())]


out = {"classes": CLASSES, "runs": {k: run_block(*v) for k, v in RUNS.items()}}
nb = json.loads((R / "baseline_resnet50" / "metrics.json").read_text())
out["notebook"] = {x: round(nb["metrics"]["official_test"][x], 4) for x in KEYS}

tr = add_cv_folds(load_split("train"))
out["dist"] = {"train": class_counts(tr.label).tolist(), "val": class_counts(load_split("val").label).tolist(),
               "test": class_counts(load_split("test").label).tolist()}

la_runs = ["resnet50_ce", "resnet50_ce_la", "resnet50_wce", "resnet50_wce_la", "gessert2018_densenet121", "gessert2018_densenet121_la"]
out["la"] = {r: summary(r) for r in la_runs}

r2 = [f"resnet50_ce{v}{t}_la" for v in ("", "_res384", "_res448", "_ep60", "_trivial") for t in ("", "_tta4", "_tta20")]
out["round2"] = {r: summary(r) for r in r2}

steps = [("resnet50_ce_la", "resnet50_ce", "+ logit adjustment"),
         ("resnet50_ce_res448_la", "resnet50_ce_la", "+ 448 px input"),
         ("resnet50_ce_res448_tta20_la", "resnet50_ce_res448_la", "+ 20-view TTA"),
         ("resnet50_ce_res448_tta20_la", "resnet50_ce", "Total vs plain baseline"),
         ("resnet50_ce_tta20_la", "resnet50_ce_la", "20-view TTA alone (224 px)"),
         ("resnet50_ce_res384_la", "resnet50_ce_la", "384 px alone"),
         ("resnet50_ce_ep60_la", "resnet50_ce_la", "60 epochs alone"),
         ("resnet50_ce_trivial_la", "resnet50_ce_la", "TrivialAugment alone"),
         ("resnet50_ce_res448_tta20_la", "gessert2018_densenet121", "Optimised vs Gessert"),
         ("resnet50_ce_res448_tta20_la", "gessert2018_densenet121_la", "Optimised vs Gessert + adjustment")]
out["steps"] = [{"label": n, "cv": paired(a, b, "cv_val"), "test": paired(a, b, "official_test"),
                 "test_mel": paired(a, b, "official_test", "mel_sensitivity")} for a, b, n in steps]

# ---------- round 3: stronger backbones ----------
BACKBONES = [  # key, label, timm model, run prefix
    ("resnet50", "ResNet-50", "resnet50.tv2_in1k", "resnet50_ce_res448"),
    ("convnext_s", "ConvNeXt-S", "convnext_small.fb_in22k_ft_in1k", "convnext_s_res448"),
    ("effv2_s", "EfficientNetV2-S", "tf_efficientnetv2_s.in21k_ft_in1k", "effv2_s_res448"),
    ("dinov2_b14", "DINOv2 ViT-B/14", "vit_base_patch14_dinov2.lvd142m", "dinov2_b14_res448"),
    ("vit_b16", "ViT-B/16", "vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit_b16_res448"),
]
ref = "resnet50_ce_res448_tta20_la"
r3 = []
for key, label, model, prefix in BACKBONES:
    run = f"{prefix}_tta20_la"
    raw = metrics(prefix, 0)
    pil = {}
    for lr in ("2e-4", "5e-5"):
        pr = R / f"pilot224_{key}_lr{lr}_la" / "fold0" / "metrics.json"
        if pr.exists():
            pil[lr] = round(json.loads(pr.read_text())["best_checkpoint"]["cv_val"]["balanced_acc"], 4)
    if key == "resnet50":
        pil = {"2e-4": round(metrics("resnet50_ce_la", 0)["best_checkpoint"]["cv_val"]["balanced_acc"], 4)}
    r3.append({"key": key, "label": label, "model": model, "params_m": round(raw["n_params"] / 1e6, 1),
               "lr": raw["config"]["lr"], "min_per_fold": round(float(np.mean([metrics(prefix, k)["train_time_min"] for k in range(5)])), 1),
               "pilot": pil, "summary": summary(run),
               "test_folds": [round(metrics(run, k)["best_checkpoint"]["official_test"]["balanced_acc"], 4) for k in range(5)],
               "paired": None if key == "resnet50" else {
                   "cv": paired(run, ref, "cv_val"), "test": paired(run, ref, "official_test"),
                   "test_auc": paired(run, ref, "official_test", "macro_auc"),
                   "test_mel": paired(run, ref, "official_test", "mel_sensitivity")}})
out["round3"] = r3


def cross_ensemble(members):
    """Per fold: average the members' probabilities; plus all members x folds averaged on test."""
    per = {}
    for split in ("cv_val", "official_test"):
        vals = []
        for k in range(5):
            ps = [pd.read_csv(R / r / f"fold{k}" / f"pred_{split}.csv") for r in members]
            vals.append(compute_metrics(ps[0]["label"].map(CLASSES.index).to_numpy(), np.mean([q[CLASSES].to_numpy() for q in ps], axis=0)))
        per[split] = {x: [round(float(np.mean([v[x] for v in vals])), 4), round(float(np.std([v[x] for v in vals])), 4)] for x in KEYS}
    big = [pd.read_csv(R / r / f"fold{k}" / "pred_official_test.csv") for r in members for k in range(5)]
    allm = compute_metrics(big[0]["label"].map(CLASSES.index).to_numpy(), np.mean([q[CLASSES].to_numpy() for q in big], axis=0))
    return {"per_fold": per, "all": {x: round(allm[x], 4) for x in KEYS},
            "all_recall": {c: round(v, 4) for c, v in allm["per_class_recall"].items()}, "all_cm": allm["confusion_matrix"],
            "n_models": len(big)}


members = [f"{b[3]}_tta20_la" for b in BACKBONES]
out["xens"] = {"all5": cross_ensemble(members), "no_vit": cross_ensemble(members[:4])}

print(json.dumps(out))
