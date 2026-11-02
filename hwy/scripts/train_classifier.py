#!/usr/bin/env python
"""Fine-tune an ImageNet-pretrained classifier on ISIC 2018 Task 3 (one CV fold per run).

The default recipe is exactly the baseline of 01_baseline_resnet50.ipynb (ResNet-50, 224 px
downscaled crops, AdamW 2e-4, 30 epochs, warm-up + cosine, model selection on the CV fold's
balanced accuracy). Flags change one factor at a time so experiments stay controlled.

Examples:
    python scripts/train_classifier.py --run-name resnet50_ce  --fold 0                         # baseline
    python scripts/train_classifier.py --run-name resnet50_wce --fold 0 --loss-weighting inverse  # + Gessert loss weighting
    python scripts/train_classifier.py --run-name tmp --smoke-test                               # 1-minute pipeline check

Homework Y additions (all off by default, so the default recipe is unchanged):
    --ema 0.999 (EMA copy of the weights, written as a second run <run-name>_ema), --drop-path, --layer-decay,
    --img-width (rectangular full-frame input), --lesion-boxes (lesion-centred crops), --metadata (age/sex/site fusion),
    --teacher-views (consistent distillation: soft labels per stored training view, scripts/teacher_views.py).

Outputs: results/<run-name>/fold<k>/{best.pt, history.csv, metrics.json, pred_<split>.csv}.
Compare runs with compare_runs.py.
"""
import sys
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hwy/, so the isic package imports
from timm.utils import ModelEmaV3

from isic.data import (CLASSES, ISICDataset, LesionCrop, MetaDataset, SoftLabelDataset, ViewSoftLabelDataset, add_cv_folds,
                       baseline_transforms, class_counts, class_weights, load_split)
from isic.engine import get_device, predict, save_json, save_predictions, seed_everything, train_one_epoch
from isic.metadata import FIELD_SLICES, N_META, metadata_for
from isic.metrics import compute_metrics, format_metrics
from isic.models import build_model, layer_decay_param_groups

HWY_DIR = Path(__file__).resolve().parents[1]


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
    p.add_argument("--augment", choices=["basic", "trivial", "color"], default="basic",
                   help="basic: mild color jitter (baseline); trivial: TrivialAugmentWide; color: stronger color jitter")
    p.add_argument("--color-constancy", action="store_true", help="Shades-of-Grey colour constancy on every image (train + eval)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade compute for memory (large transformers)")
    p.add_argument("--soft-labels", type=Path, default=None,
                   help="CSV (image + one probability column per class) of teacher soft labels -> knowledge distillation")
    p.add_argument("--teacher-views", type=Path, default=None,
                   help="teacher_views.py output dir -> consistent distillation (one soft label per stored training view)")
    p.add_argument("--kd-alpha", type=float, default=0.5, help="weight of the distillation term (1 - alpha on hard-label CE)")
    p.add_argument("--kd-temp", type=float, default=1.0, help="distillation temperature")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 = off); EMA weights are selected and saved separately")
    p.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth rate")
    p.add_argument("--layer-decay", type=float, default=0.0, help="layer-wise LR decay (ConvNeXt only; --lr is the top layer's)")
    p.add_argument("--img-width", type=int, default=None, help="rectangular input img_size x img_width (full frame)")
    p.add_argument("--lesion-boxes", type=Path, default=None, help="CSV image,x0,y0,x1,y1 -> lesion-centred crops")
    p.add_argument("--metadata", action="store_true", help="fuse age / sex / body site (isic/metadata.py)")
    p.add_argument("--meta-drop-all", type=float, default=0.25, help="modality dropout: P(all metadata removed)")
    p.add_argument("--meta-drop-field", type=float, default=0.10, help="modality dropout: P(one field removed)")
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
    fold_dir = f"fold{args.fold}" + ("_smoke" if args.smoke_test else "")
    out_dir = HWY_DIR / "results" / args.run_name / fold_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ema_dir = HWY_DIR / "results" / f"{args.run_name}_ema" / fold_dir

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

    train_tf, eval_tf = baseline_transforms(args.img_size, args.resize, args.augment, args.color_constancy,
                                            width=args.img_width)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_kw, eval_kw = {}, {}
    if args.lesion_boxes:
        bx = pd.read_csv(args.lesion_boxes)
        boxes = dict(zip(bx["image"], bx[["x0", "y0", "x1", "y1"]].to_numpy(float)))
        train_kw, eval_kw = dict(lesion_crop=LesionCrop(train=True), boxes=boxes), dict(lesion_crop=LesionCrop(train=False), boxes=boxes)
        print(f"lesion-centred crops from {args.lesion_boxes}")
    if args.teacher_views:
        tv = pd.read_csv(args.teacher_views / "images.csv").image
        params, soft = np.load(args.teacher_views / "params.npy"), np.load(args.teacher_views / "soft_views.npy")
        train_ds = ViewSoftLabelDataset(train_df, dict(zip(tv, params)), dict(zip(tv, soft)),
                                        (args.img_size, args.img_width or args.img_size))
        print(f"consistent distillation from {args.teacher_views}: {params.shape[1]} stored views per image (alpha {args.kd_alpha})")
    elif args.soft_labels:
        sl = pd.read_csv(args.soft_labels)
        soft = dict(zip(sl["image"], sl[CLASSES].to_numpy()))
        train_ds = SoftLabelDataset(train_df, train_tf, soft, **train_kw)
        print(f"knowledge distillation from {args.soft_labels} (alpha {args.kd_alpha}, T {args.kd_temp})")
    else:
        train_ds = ISICDataset(train_df, train_tf, **train_kw)
    # eval sets: name -> (DataFrame, metadata removed?); metadata models are also scored with all metadata removed
    eval_sets = {k: (d, False) for k, d in splits.items() if k != "train"}
    if args.metadata:
        train_ds = MetaDataset(train_ds, metadata_for(train_df.image), FIELD_SLICES, args.meta_drop_all, args.meta_drop_field)
        eval_sets.update({f"{k}_nometa": (splits[k], True) for k in ("cv_val", "official_test")})
        print(f"metadata fusion, modality dropout: all {args.meta_drop_all}, per field {args.meta_drop_field}")

    def eval_dataset(df, nometa):
        ds = ISICDataset(df, eval_tf, **eval_kw)
        return MetaDataset(ds, metadata_for(df.image, missing=nometa), FIELD_SLICES) if args.metadata else ds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    eval_loaders = {k: DataLoader(eval_dataset(d, nm), batch_size=128, shuffle=False, **loader_kw)
                    for k, (d, nm) in eval_sets.items()}

    # ---------------- model / loss / optimiser ----------------
    model = build_model(args.model, len(CLASSES), args.img_size, pretrained=True,
                        grad_checkpointing=args.grad_checkpointing, drop_path_rate=args.drop_path,
                        n_meta=N_META if args.metadata else 0)
    model = model.to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())

    w = class_weights(train_df.label, args.loss_weighting)
    print("train class counts:", dict(zip(CLASSES, class_counts(train_df.label).tolist())))
    print("class weights:", None if w is None else dict(zip(CLASSES, w.round(decimals=2).tolist())))
    ce = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    if args.soft_labels or args.teacher_views:
        T, a = args.kd_temp, args.kd_alpha

        def loss_fn(logits, target):  # target = [hard label, teacher probs]
            y, p_t = target[:, 0].long(), target[:, 1:]
            if T != 1.0:
                p_t = torch.softmax(torch.log(p_t.clamp_min(1e-7)) / T, dim=1)
            kd = -(p_t * torch.log_softmax(logits.float() / T, dim=1)).sum(1).mean() * T * T
            return (1 - a) * ce(logits, y) + a * kd
    else:
        loss_fn = ce
    if args.layer_decay:
        optimizer = torch.optim.AdamW(layer_decay_param_groups(model, args.lr, args.layer_decay, args.weight_decay))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEmaV3(model, decay=args.ema, use_warmup=True) if args.ema else None

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------- training ----------------
    history, best_bacc, best_epoch = [], -1.0, -1
    epoch_probs = {"raw": [], "ema": []}   # per-epoch CV predictions, for selection-free (cross-fitted) CV estimates
    ckpt = out_dir / "best.pt"
    if ema is not None:
        ema_dir.mkdir(parents=True, exist_ok=True)
        best_ema, best_ema_epoch = -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler=scheduler, amp=amp,
                                     desc=f"epoch {epoch}", ema=ema)
        probs, labels = predict(model, eval_loaders["cv_val"], device, amp=amp, desc="cv_val")
        m = compute_metrics(labels, probs)
        epoch_probs["raw"].append(probs.astype(np.float16))
        history.append({"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[-1]["lr"],
                        "val_balanced_acc": m["balanced_acc"], "val_accuracy": m["accuracy"],
                        "val_macro_auc": m["macro_auc"], "val_mel_sens": m["mel_sensitivity"]})
        if m["balanced_acc"] > best_bacc:
            best_bacc, best_epoch = m["balanced_acc"], epoch
            torch.save(model.state_dict(), ckpt)
        ema_msg = ""
        if ema is not None:
            probs_ema, labels_ema = predict(ema.module, eval_loaders["cv_val"], device, amp=amp, desc="cv_val ema")
            m_ema = compute_metrics(labels_ema, probs_ema)
            epoch_probs["ema"].append(probs_ema.astype(np.float16))
            history[-1].update(val_balanced_acc_ema=m_ema["balanced_acc"], val_macro_auc_ema=m_ema["macro_auc"])
            if m_ema["balanced_acc"] > best_ema:
                best_ema, best_ema_epoch = m_ema["balanced_acc"], epoch
                torch.save(ema.module.state_dict(), ema_dir / "best.pt")
            ema_msg = f" | EMA {m_ema['balanced_acc']:.4f} best {best_ema:.4f} @ {best_ema_epoch}"
        print(f"[{time.time() - t0:6.0f}s] epoch {epoch:2d} | loss {train_loss:.4f} | val BalAcc {m['balanced_acc']:.4f} "
              f"Acc {m['accuracy']:.4f} AUC {m['macro_auc']:.4f} MEL sens {m['mel_sensitivity']:.3f} | "
              f"best {best_bacc:.4f} @ {best_epoch}{ema_msg}", flush=True)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    np.save(out_dir / "cv_val_epoch_probs.npy", np.stack(epoch_probs["raw"]))   # (epochs, images, classes)
    if ema is not None:
        np.save(ema_dir / "cv_val_epoch_probs.npy", np.stack(epoch_probs["ema"]))

    # ---------------- evaluation ----------------
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    final = [(model, ckpt, out_dir, {"best_epoch": best_epoch, "last_checkpoint": {"cv_val": m}})]
    if ema is not None:
        final.append((ema.module, ema_dir / "best.pt", ema_dir, {"best_epoch": best_ema_epoch, "ema_of": args.run_name}))
    for net, path, out, extra in final:
        results = {"config": config, "n_params": n_params, "train_time_min": (time.time() - t0) / 60, **extra,
                   "best_checkpoint": {}}
        net.load_state_dict(torch.load(path, map_location=device))
        for name, loader in eval_loaders.items():
            probs, labels = predict(net, loader, device, amp=amp, desc=name)
            results["best_checkpoint"][name] = compute_metrics(labels, probs)
            save_predictions(out / f"pred_{name}.csv", eval_sets[name][0].image, probs, labels)
            print(format_metrics(results["best_checkpoint"][name], f"{name}{' (EMA)' if net is not model else ''}"))
        save_json(results, out / "metrics.json")
        print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
