#!/usr/bin/env python
"""Re-evaluate a finished train_classifier.py run with test-time augmentation (no retraining).

Loads each fold's best.pt and averages softmax probabilities over several views of
every image: crops (center, or center + 4 corners) x flips (none, H, V, H+V).
The result is written as a new run, results/<run>_tta<views>/fold<k>/, in the same
format as train_classifier.py, so posthoc_logit_adjustment.py and compare_runs.py
work on it unchanged.

    python evaluate_tta.py resnet50_ce --crops 1     # 4 views  -> results/resnet50_ce_tta4
    python evaluate_tta.py resnet50_ce --crops 5     # 20 views -> results/resnet50_ce_tta20
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from isic.data import CLASSES, ISICDataset, TTAViews, add_cv_folds, load_split
from isic.engine import get_device, predict, save_json, save_predictions
from isic.metrics import compute_metrics, format_metrics
from isic.models import build_model

HWX_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="finished train_classifier.py run under results/")
    p.add_argument("--crops", type=int, choices=[1, 5], default=5)
    p.add_argument("--no-flips", action="store_true")
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--folds", type=int, nargs="+", default=None, help="default: every fold with a best.pt")
    args = p.parse_args()

    device = get_device()
    src = HWX_DIR / "results" / args.run
    folds = args.folds or sorted(int(d.name[4:]) for d in src.glob("fold[0-9]") if (d / "best.pt").exists())
    if not folds:
        raise SystemExit(f"no checkpoints under {src}")

    for k in folds:
        cfg = json.loads((src / f"fold{k}" / "metrics.json").read_text())["config"]
        resize = cfg.get("resize") or round(cfg["img_size"] * 256 / 224)
        tf = TTAViews(cfg["img_size"], resize, crops=args.crops, flips=not args.no_flips)
        out = HWX_DIR / "results" / f"{args.run}_tta{tf.n_views}" / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)

        train_all = add_cv_folds(load_split("train"), n_folds=cfg["n_folds"], seed=cfg["cv_seed"])
        splits = {"cv_val": train_all[train_all.fold == k].reset_index(drop=True),
                  "official_val": load_split("val"), "official_test": load_split("test")}

        model = build_model(cfg["model"], len(CLASSES), cfg["img_size"], pretrained=False)
        model.load_state_dict(torch.load(src / f"fold{k}" / "best.pt", map_location="cpu"))
        model = model.to(device).to(memory_format=torch.channels_last)

        # Keep roughly 512 views of 224 px per forward pass.
        bs = max(2, int(512 / tf.n_views * (224 / cfg["img_size"]) ** 2))
        res = {"config": {**cfg, "tta": {"crops": args.crops, "flips": not args.no_flips, "n_views": tf.n_views},
                          "source_run": args.run},
               "best_checkpoint": {}}
        for name, df in splits.items():
            loader = DataLoader(ISICDataset(df, tf), batch_size=bs, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
            probs, labels = predict(model, loader, device, desc=f"fold{k} {name}")
            res["best_checkpoint"][name] = compute_metrics(labels, probs)
            save_predictions(out / f"pred_{name}.csv", df.image, probs, labels)
        save_json(res, out / "metrics.json")
        print(f"[{args.run} tta{tf.n_views}] fold {k}\n" + format_metrics(res["best_checkpoint"]["cv_val"], "cv_val")
              + "\n" + format_metrics(res["best_checkpoint"]["official_test"], "official_test"), flush=True)


if __name__ == "__main__":
    main()
