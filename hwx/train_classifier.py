#!/usr/bin/env python
"""Fine-tune an ImageNet-pretrained classifier on ISIC 2018 Task 3 (one CV fold per run).

The default recipe is exactly the baseline of 01_baseline_resnet50.ipynb (ResNet-50, 224 px
downscaled crops, AdamW 2e-4, 30 epochs, warm-up + cosine, model selection on the CV fold's
balanced accuracy). Flags change one factor at a time so experiments stay controlled.

Examples:
    python train_classifier.py --run-name resnet50_ce  --fold 0                         # baseline
    python train_classifier.py --run-name resnet50_wce --fold 0 --loss-weighting inverse  # + Gessert loss weighting
    python train_classifier.py --run-name tmp --smoke-test                               # 1-minute pipeline check

Outputs: results/<run-name>/fold<k>/{best.pt, history.csv, metrics.json, pred_<split>.csv}.
Compare runs with compare_runs.py.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from isic.data import (CLASSES, ISICDataset, add_cv_folds, baseline_transforms, class_counts, class_weights,
                       load_split)
from isic.engine import get_device, predict, save_json, save_predictions, seed_everything, train_one_epoch
from isic.metrics import compute_metrics, format_metrics
from isic.models import build_model

HWX_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="results/<run-name>/fold<k>/")
    p.add_argument("--model", default="resnet50.tv2_in1k", help="timm model name (ImageNet-pretrained)")
    p.add_argument("--loss-weighting", choices=["none", "inverse"], default="none",
                   help="none: plain CE (baseline); inverse: CE weighted by w_i = N/n_i (Gessert et al. 2018)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--resize", type=int, default=None,
                   help="short-side resize before cropping; default keeps the baseline framing, round(img_size * 256/224)")
    p.add_argument("--augment", choices=["basic", "trivial"], default="basic",
                   help="basic: mild color jitter (baseline); trivial: TrivialAugmentWide")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade compute for memory (large transformers)")
    p.add_argument("--smoke-test", action="store_true", help="tiny subset, 2 epochs")
    args = p.parse_args()
    if args.resize is None:
        args.resize = round(args.img_size * 256 / 224)
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    amp = not args.no_amp
    out_dir = HWX_DIR / "results" / args.run_name / (f"fold{args.fold}" + ("_smoke" if args.smoke_test else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    train_all = add_cv_folds(load_split("train"), n_folds=args.n_folds, seed=args.cv_seed)
    splits = {
        "train": train_all[train_all.fold != args.fold].reset_index(drop=True),
        "cv_val": train_all[train_all.fold == args.fold].reset_index(drop=True),
        "official_val": load_split("val"),
        "official_test": load_split("test"),
    }
    if args.smoke_test:
        splits = {k: d.groupby("label").head(40 if k == "train" else 8).reset_index(drop=True) for k, d in splits.items()}
        args.epochs = 2
    train_df = splits["train"]
    print(f"[{args.run_name}] fold {args.fold}: " + " | ".join(f"{k} {len(d)}" for k, d in splits.items()))

    train_tf, eval_tf = baseline_transforms(args.img_size, args.resize, args.augment)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(ISICDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    eval_loaders = {k: DataLoader(ISICDataset(d, eval_tf), batch_size=128, shuffle=False, **loader_kw)
                    for k, d in splits.items() if k != "train"}

    # ---------------- model / loss / optimiser ----------------
    model = build_model(args.model, len(CLASSES), args.img_size, pretrained=True,
                        grad_checkpointing=args.grad_checkpointing)
    model = model.to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())

    w = class_weights(train_df.label, args.loss_weighting)
    print("train class counts:", dict(zip(CLASSES, class_counts(train_df.label).tolist())))
    print("class weights:", None if w is None else dict(zip(CLASSES, w.round(decimals=2).tolist())))
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------- training ----------------
    history, best_bacc, best_epoch = [], -1.0, -1
    ckpt = out_dir / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler=scheduler, amp=amp,
                                     desc=f"epoch {epoch}")
        probs, labels = predict(model, eval_loaders["cv_val"], device, amp=amp, desc="cv_val")
        m = compute_metrics(labels, probs)
        history.append({"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"],
                        "val_balanced_acc": m["balanced_acc"], "val_accuracy": m["accuracy"],
                        "val_macro_auc": m["macro_auc"], "val_mel_sens": m["mel_sensitivity"]})
        if m["balanced_acc"] > best_bacc:
            best_bacc, best_epoch = m["balanced_acc"], epoch
            torch.save(model.state_dict(), ckpt)
        print(f"[{time.time() - t0:6.0f}s] epoch {epoch:2d} | loss {train_loss:.4f} | val BalAcc {m['balanced_acc']:.4f} "
              f"Acc {m['accuracy']:.4f} AUC {m['macro_auc']:.4f} MEL sens {m['mel_sensitivity']:.3f} | "
              f"best {best_bacc:.4f} @ {best_epoch}", flush=True)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # ---------------- evaluation ----------------
    results = {"config": vars(args), "n_params": n_params, "best_epoch": best_epoch,
               "train_time_min": (time.time() - t0) / 60,
               "last_checkpoint": {"cv_val": m}, "best_checkpoint": {}}
    model.load_state_dict(torch.load(ckpt, map_location=device))
    for name, loader in eval_loaders.items():
        probs, labels = predict(model, loader, device, amp=amp, desc=name)
        results["best_checkpoint"][name] = compute_metrics(labels, probs)
        save_predictions(out_dir / f"pred_{name}.csv", splits[name].image, probs, labels)
        print(format_metrics(results["best_checkpoint"][name], name))
    save_json(results, out_dir / "metrics.json")
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
