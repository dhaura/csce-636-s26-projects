#!/usr/bin/env python
"""Literature baseline: single-model recipe of Gessert et al. (2018), ISIC 2018 Task 3.

    N. Gessert et al., "Skin Lesion Diagnosis using Ensembles, Unscaled Multi-Crop
    Evaluation and Loss Weighting", arXiv:1808.01694 (2nd place ISIC 2018 Task 3,
    best entry using only public data; official test balanced accuracy 0.856 with a
    54-model ensemble). Reference code: https://github.com/ngessert/isic2018

What is reproduced (their reference single model, Densenet121, Table II: 5-fold CV
WACC 0.795 / mean AUC 0.967 / accuracy 0.823):
  * ImageNet-pretrained DenseNet-121, fully fine-tuned
  * full-resolution 600x450 images, *unscaled* random 224x224 crops, H/V flips,
    ColorJitter(brightness=32/255, saturation=0.5)
  * cross-entropy weighted by inverse class frequency  w_i = N / n_i
  * Adam, lr 5e-4, batch 40, lr x0.2 at epochs 50/75/100, 125 epochs
  * validation every 5 epochs with 16 grid crops; keep the best-WACC checkpoint
  * final evaluation with 36 unscaled grid crops, probabilities averaged
  * 5-fold CV split by lesion (one fold per run; use --fold 0..4)

Deviations (documented in README): our own lesion-grouped stratified folds (shared
with all other experiments) instead of the authors' pickle; ImageNet mean/std
normalization (timm weights) instead of mean subtraction; bf16 autocast; no
ensemble / SVM meta-learner (single model only).

Usage:
    python literature_baseline_gessert2018.py --fold 0          # one CV fold (~30-40 min on an A100)
    python literature_baseline_gessert2018.py --smoke-test      # 2-minute pipeline check
    python literature_baseline_gessert2018.py --summarize       # aggregate finished folds
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from isic.data import (CLASSES, ISICDataset, GridCrops, add_cv_folds, class_counts, class_weights,
                       gessert_train_transform, load_split)
from isic.engine import get_device, predict, save_json, save_predictions, seed_everything, train_one_epoch
from isic.metrics import compute_metrics, format_metrics

HWX_DIR = Path(__file__).resolve().parent

# Numbers reported in the paper (Table II, 5-fold CV, 36-crop evaluation).
PAPER_REFERENCE = {
    "densenet121_cv": {"balanced_acc": 0.795, "accuracy": 0.823, "macro_auc": 0.967},
    "densenet121_no_weighting_cv": {"balanced_acc": 0.757, "accuracy": 0.883, "macro_auc": 0.976},
    "best_single_model_senet154_cv": {"balanced_acc": 0.817, "accuracy": 0.854, "macro_auc": 0.974},
    "ensemble_cv": {"balanced_acc": 0.851},
    "ensemble_official_test": {"balanced_acc": 0.856},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="densenet121.tv_in1k", help="timm model name (pretrained weights)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=0, help="seed of the lesion-grouped fold assignment")
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--epochs", type=int, default=125)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-milestones", type=int, nargs="+", default=[50, 75, 100])
    p.add_argument("--lr-gamma", type=float, default=0.2)
    p.add_argument("--weighting", choices=["inverse", "inverse_c", "none"], default="inverse",
                   help="inverse: w=N/n_i (paper default); inverse_c: w=N/(C n_i); none: plain CE")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--val-crops-per-side", type=int, default=4, help="4 -> 16 crops during training")
    p.add_argument("--test-crops-per-side", type=int, default=6, help="6 -> 36 crops for final evaluation")
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--smoke-test", action="store_true", help="tiny subset, 2 epochs - checks the pipeline")
    p.add_argument("--summarize", action="store_true", help="aggregate finished folds and exit")
    return p.parse_args()


def default_out_root(arch: str) -> Path:
    return HWX_DIR / "results" / f"gessert2018_{arch.split('.')[0]}"


def run_fold(args):
    seed_everything(args.seed)
    device = get_device()
    amp = not args.no_amp
    out_dir = args.out_dir or default_out_root(args.arch) / (f"fold{args.fold}" + ("_smoke" if args.smoke_test else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    train_all = add_cv_folds(load_split("train"), n_folds=args.n_folds, seed=args.cv_seed)
    train_df = train_all[train_all.fold != args.fold].reset_index(drop=True)
    val_df = train_all[train_all.fold == args.fold].reset_index(drop=True)
    off_val_df, test_df = load_split("val"), load_split("test")
    if args.smoke_test:
        train_df = train_df.groupby("label").head(40).reset_index(drop=True)
        val_df, off_val_df, test_df = (d.groupby("label").head(8).reset_index(drop=True)
                                       for d in (val_df, off_val_df, test_df))
        args.epochs, args.eval_every, args.lr_milestones = 2, 1, [1]
    print(f"fold {args.fold}: train {len(train_df)} | cv-val {len(val_df)} | official val {len(off_val_df)} | test {len(test_df)}")
    print("train class counts:", dict(zip(CLASSES, class_counts(train_df.label).tolist())))

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(ISICDataset(train_df, gessert_train_transform()), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kw)
    # Multi-crop eval: keep ~512-600 crops per forward pass.
    val_tf, test_tf = GridCrops(args.val_crops_per_side), GridCrops(args.test_crops_per_side)
    eval_bs = max(1, 576 // args.test_crops_per_side ** 2)

    def eval_loader(df, tf, bs):
        return DataLoader(ISICDataset(df, tf), batch_size=bs, shuffle=False, **loader_kw)

    val_loader_16 = eval_loader(val_df, val_tf, max(1, 512 // args.val_crops_per_side ** 2))

    # ---------------- model / optimisation ----------------
    model = timm.create_model(args.arch, pretrained=True, num_classes=len(CLASSES))
    model = model.to(device).to(memory_format=torch.channels_last)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    w = class_weights(train_df.label, args.weighting)
    print("class weights:", None if w is None else dict(zip(CLASSES, w.round(decimals=2).tolist())))
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    # ---------------- training ----------------
    history, best_wacc, best_epoch = [], -1.0, -1
    ckpt_best = out_dir / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp=amp, desc=f"epoch {epoch}")
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            probs, labels = predict(model, val_loader_16, device, amp=amp, desc="val16")
            m = compute_metrics(labels, probs)
            row.update(val_balanced_acc=m["balanced_acc"], val_accuracy=m["accuracy"], val_auc=m["macro_auc"],
                       val_mel_sens=m["mel_sensitivity"])
            if m["balanced_acc"] > best_wacc:
                best_wacc, best_epoch = m["balanced_acc"], epoch
                torch.save(model.state_dict(), ckpt_best)
            print(f"[{time.time() - t0:7.0f}s] epoch {epoch:3d} loss {train_loss:.4f} lr {row['lr']:.2e} | "
                  f"val16 WACC {m['balanced_acc']:.4f} acc {m['accuracy']:.4f} AUC {m['macro_auc']:.4f} "
                  f"(best {best_wacc:.4f} @ {best_epoch})", flush=True)
        else:
            print(f"[{time.time() - t0:7.0f}s] epoch {epoch:3d} loss {train_loss:.4f} lr {row['lr']:.2e}", flush=True)
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # ---------------- final evaluation (36 crops) ----------------
    results = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "best_epoch": best_epoch, "train_time_min": (time.time() - t0) / 60,
               "paper_reference": PAPER_REFERENCE}
    splits = {"cv_val": val_df, "official_val": off_val_df, "official_test": test_df}

    # Last checkpoint on the CV fold (paper notes a large best-vs-last WACC gap).
    probs, labels = predict(model, eval_loader(val_df, test_tf, eval_bs), device, amp=amp, desc="last cv_val")
    results["last_checkpoint"] = {"cv_val": compute_metrics(labels, probs)}
    print(format_metrics(results["last_checkpoint"]["cv_val"], "last cv_val"))

    model.load_state_dict(torch.load(ckpt_best, map_location=device))
    results["best_checkpoint"] = {}
    for name, df in splits.items():
        probs, labels = predict(model, eval_loader(df, test_tf, eval_bs), device, amp=amp, desc=name)
        results["best_checkpoint"][name] = compute_metrics(labels, probs)
        save_predictions(out_dir / f"pred_{name}.csv", df.image, probs, labels)
        print(format_metrics(results["best_checkpoint"][name], name))
    save_json(results, out_dir / "metrics.json")
    print(f"Saved results to {out_dir}")


def summarize(args):
    """Mean +- std over finished folds, plus a 5-fold-ensemble on the official test set."""
    root = args.out_dir or default_out_root(args.arch)
    fold_dirs = sorted(d for d in root.glob("fold[0-9]") if (d / "metrics.json").exists())
    if not fold_dirs:
        raise SystemExit(f"No finished folds under {root}")
    rows = []
    for d in fold_dirs:
        r = json.loads((d / "metrics.json").read_text())["best_checkpoint"]
        rows.append({"fold": d.name, **{f"{s}_{k}": r[s][k] for s in r
                                        for k in ("balanced_acc", "accuracy", "macro_auc", "mel_sensitivity")}})
    table = pd.DataFrame(rows).set_index("fold")
    print(table.round(4).to_string())
    print("\nmean +- std over folds:")
    for col in table.columns:
        print(f"  {col:<32s} {table[col].mean():.4f} +- {table[col].std(ddof=0):.4f}")

    preds = [pd.read_csv(d / "pred_official_test.csv") for d in fold_dirs]
    probs = np.mean([p[CLASSES].to_numpy() for p in preds], axis=0)
    labels = preds[0]["label"].map(CLASSES.index).to_numpy()
    ens = compute_metrics(labels, probs)
    print("\n" + format_metrics(ens, f"{len(fold_dirs)}-fold ens. test"))
    print(f"\nPaper: DenseNet121 CV WACC {PAPER_REFERENCE['densenet121_cv']['balanced_acc']}, "
          f"54-model ensemble official test {PAPER_REFERENCE['ensemble_official_test']['balanced_acc']}")
    save_json({"per_fold": table.reset_index().to_dict("records"),
               "mean": table.mean().to_dict(), "std": table.std(ddof=0).to_dict(),
               "fold_ensemble_official_test": ens, "paper_reference": PAPER_REFERENCE}, root / "summary.json")


if __name__ == "__main__":
    a = parse_args()
    summarize(a) if a.summarize else run_fold(a)
