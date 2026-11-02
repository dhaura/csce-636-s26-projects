#!/usr/bin/env python
"""Re-evaluate a finished train_classifier.py run with test-time augmentation (no retraining).

Loads each fold's best.pt and averages softmax probabilities over several views of
every image: crops (center, or center + 4 corners) x flips (none, H, V, H+V).
The result is written as a new run, results/<run>_tta<views>/fold<k>/, in the same
format as train_classifier.py, so posthoc_logit_adjustment.py and compare_runs.py
work on it unchanged.

    python scripts/evaluate_tta.py resnet50_ce --crops 1     # 4 views  -> results/resnet50_ce_tta4
    python scripts/evaluate_tta.py resnet50_ce --crops 5     # 20 views -> results/resnet50_ce_tta20
"""
import sys
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hwy/, so the isic package imports
from isic.data import CLASSES, ISICDataset, LesionCrop, MetaDataset, TTAViews, add_cv_folds, load_split
from isic.engine import get_device, predict, save_json, save_predictions
from isic.metadata import FIELD_SLICES, N_META, metadata_for
from isic.metrics import compute_metrics, format_metrics
from isic.models import build_model

HWY_DIR = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="finished train_classifier.py run under results/")
    p.add_argument("--crops", type=int, choices=[1, 5], default=5)
    p.add_argument("--no-flips", action="store_true")
    p.add_argument("--img-size", type=int, default=None,
                   help="evaluate at this crop size instead of the training size (FixRes); output gets suffix _ts<size>")
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--folds", type=int, nargs="+", default=None, help="default: every fold with a best.pt")
    args = p.parse_args()

    device = get_device()
    src = HWY_DIR / "results" / args.run
    folds = args.folds or sorted(int(d.name[4:]) for d in src.glob("fold[0-9]") if (d / "best.pt").exists())
    if not folds:
        raise SystemExit(f"no checkpoints under {src}")

    for k in folds:
        cfg = json.loads((src / f"fold{k}" / "metrics.json").read_text())["config"]
        size = args.img_size or cfg["img_size"]
        resize = round(size * 256 / 224) if args.img_size else (cfg.get("resize") or round(size * 256 / 224))
        width = cfg.get("img_width") and round(cfg["img_width"] * size / cfg["img_size"])
        tf = TTAViews(size, resize, crops=args.crops, flips=not args.no_flips, color_constancy=cfg.get("color_constancy", False),
                      width=width)
        suffix = f"_tta{tf.n_views}" + (f"_ts{size}" if args.img_size else "")
        out = HWY_DIR / "results" / f"{args.run}{suffix}" / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)

        train_all = add_cv_folds(load_split("train"), n_folds=cfg["n_folds"], seed=cfg["cv_seed"])
        splits = {"cv_val": train_all[train_all.fold == k].reset_index(drop=True),
                  "official_val": load_split("val"), "official_test": load_split("test")}
        eval_sets = {name: (df, False) for name, df in splits.items()}
        if cfg.get("metadata"):
            eval_sets.update({f"{s}_nometa": (splits[s], True) for s in ("cv_val", "official_test")})
        crop_kw = {}
        if cfg.get("lesion_boxes"):
            bx = pd.read_csv(HWY_DIR / cfg["lesion_boxes"])
            crop_kw = dict(lesion_crop=LesionCrop(train=False), boxes=dict(zip(bx["image"], bx[["x0", "y0", "x1", "y1"]].to_numpy(float))))

        model = build_model(cfg["model"], len(CLASSES), cfg["img_size"], pretrained=False,
                            n_meta=N_META if cfg.get("metadata") else 0)
        model.load_state_dict(torch.load(src / f"fold{k}" / "best.pt", map_location="cpu"))
        model = model.to(device).to(memory_format=torch.channels_last)

        # Keep roughly 512 views of 224 px per forward pass.
        bs = max(2, int(512 / tf.n_views * (224 / size) ** 2 * size / (width or size)))
        res = {"config": {**cfg, "tta": {"crops": args.crops, "flips": not args.no_flips, "n_views": tf.n_views, "test_size": size, "test_resize": resize},
                          "source_run": args.run},
               "best_checkpoint": {}}
        for name, (df, nometa) in eval_sets.items():
            ds = ISICDataset(df, tf, **crop_kw)
            if cfg.get("metadata"):
                ds = MetaDataset(ds, metadata_for(df.image, missing=nometa), FIELD_SLICES)
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            probs, labels = predict(model, loader, device, desc=f"fold{k} {name}")
            res["best_checkpoint"][name] = compute_metrics(labels, probs)
            save_predictions(out / f"pred_{name}.csv", df.image, probs, labels)
        save_json(res, out / "metrics.json")
        print(f"[{args.run}{suffix}] fold {k}\n" + format_metrics(res["best_checkpoint"]["cv_val"], "cv_val")
              + "\n" + format_metrics(res["best_checkpoint"]["official_test"], "official_test"), flush=True)


if __name__ == "__main__":
    main()
