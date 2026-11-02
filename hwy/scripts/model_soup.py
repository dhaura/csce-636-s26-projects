#!/usr/bin/env python
"""Uniform model soup: average the weights of several fine-tuned runs, fold by fold, into ONE network per fold.

    M. Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy
    without increasing inference time", ICML 2022.

Every member must be the same architecture fine-tuned from the same pretrained checkpoint, with the same input
framing (img_size / img_width). The soup is written as a normal run (results/<name>/fold<k>/{best.pt, metrics.json,
pred_<split>.csv}, single-view scores), so evaluate_tta.py and posthoc_logit_adjustment.py work on it unchanged.

    python scripts/model_soup.py --name soup_rect10 --members cnx_kd_r1_rect cnx_kd_r1_rect_ema ...
"""
import sys
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hwy/, so the isic package imports
from isic.data import CLASSES, ISICDataset, add_cv_folds, baseline_transforms, load_split
from isic.engine import get_device, predict, save_json, save_predictions
from isic.metrics import compute_metrics, format_metrics
from isic.models import build_model

HWY_DIR = Path(__file__).resolve().parents[1]
SAME = ("model", "img_size", "img_width", "resize", "n_folds", "cv_seed", "metadata", "lesion_boxes", "color_constancy")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--members", nargs="+", required=True, help="runs under results/ with fold<k>/best.pt")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    args = p.parse_args()
    device = get_device()

    for k in args.folds:
        cfgs = [json.loads((HWY_DIR / "results" / m / f"fold{k}" / "metrics.json").read_text())["config"] for m in args.members]
        for key in SAME:
            vals = {json.dumps(c.get(key)) for c in cfgs}
            assert len(vals) == 1, f"members differ in {key}: {vals}"
        cfg = {**cfgs[0], "run_name": args.name, "soup_of": args.members, "ema": 0.0}

        soup = None
        for m in args.members:
            sd = torch.load(HWY_DIR / "results" / m / f"fold{k}" / "best.pt", map_location="cpu")
            if soup is None:
                soup = {n: t.float().clone() for n, t in sd.items()}
            else:
                assert sd.keys() == soup.keys(), f"{m}: parameter names differ"
                for n, t in sd.items():
                    soup[n] += t.float()
        soup = {n: t / len(args.members) for n, t in soup.items()}

        out = HWY_DIR / "results" / args.name / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(soup, out / "best.pt")

        model = build_model(cfg["model"], len(CLASSES), cfg["img_size"], pretrained=False)
        model.load_state_dict(soup)
        model = model.to(device).to(memory_format=torch.channels_last)
        _, eval_tf = baseline_transforms(cfg["img_size"], cfg["resize"], cfg.get("augment", "basic"),
                                         cfg.get("color_constancy", False), width=cfg.get("img_width"))
        train_all = add_cv_folds(load_split("train"), n_folds=cfg["n_folds"], seed=cfg["cv_seed"])
        splits = {"cv_val": train_all[train_all.fold == k].reset_index(drop=True),
                  "official_val": load_split("val"), "official_test": load_split("test")}
        res = {"config": cfg, "best_epoch": None, "best_checkpoint": {}}
        for name, df in splits.items():
            loader = DataLoader(ISICDataset(df, eval_tf), batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            probs, labels = predict(model, loader, device, desc=name)
            res["best_checkpoint"][name] = compute_metrics(labels, probs)
            save_predictions(out / f"pred_{name}.csv", df.image, probs, labels)
        save_json(res, out / "metrics.json")
        print(f"[{args.name}] fold {k}: soup of {len(args.members)}\n" + format_metrics(res["best_checkpoint"]["cv_val"], "cv_val")
              + "\n" + format_metrics(res["best_checkpoint"]["official_test"], "official_test"), flush=True)


if __name__ == "__main__":
    main()
