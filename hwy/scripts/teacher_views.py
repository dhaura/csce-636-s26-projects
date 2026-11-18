#!/usr/bin/env python
"""Consistent distillation targets: teacher predictions on the exact training views the student will see.

The Homework X soft labels give one fixed target per image (the TTA-20 ensemble prediction), while the student sees a
different random crop / flip / colour jitter every epoch. Here every training image gets K stored random views (same
distribution as the training pipeline). The out-of-fold 5-backbone teacher, the HW X fold models that never saw the
image, predicts each view. The student (train_classifier.py --teacher-views) then trains on those views with their own
soft labels (Beyer et al., "Knowledge distillation: A good teacher is patient and consistent", CVPR 2022).

    python scripts/teacher_views.py params  --out results/teacher_views/bb5_rect_v16 --views 16 --img-size 448 --img-width 600
    python scripts/teacher_views.py predict --out results/teacher_views/bb5_rect_v16 --fold 0      # one job per CV fold (GPU)
    python scripts/teacher_views.py merge   --out results/teacher_views/bb5_rect_v16

merge fits one temperature per member on its per-view out-of-fold log-probs (training labels only), averages the
calibrated log-probs and renormalises, as make_soft_labels.py does for the fixed labels. Outputs in --out:
params.npy (N, K, VIEW_PARAMS), images.csv, logp_fold<k>.npy, soft_views.npy (N, K, C), summary.json.
Same caveat as the fixed labels: a fold-j image's teacher was trained on the other folds, incl. the student's CV fold.
"""
import sys
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hwy/, so the isic package imports
from isic.data import CLASSES, add_cv_folds, apply_view, load_split, sample_view_params
from isic.decision import balanced_accuracy, fit_temperature, log_softmax
from isic.engine import get_device
from isic.models import _SIZED_FAMILIES

HWY_DIR = Path(__file__).resolve().parents[1]
MEMBERS = ["resnet50_ce_res448", "convnext_s_res448", "effv2_s_res448", "dinov2_b14_res448", "vit_b16_res448"]
FIXED = HWY_DIR / "results" / "hwx" / "soft_labels" / "bb5_calib_oof.csv"


class StoredViews(Dataset):
    """All K stored views of one image per item: (K, 3, h, w)."""

    def __init__(self, paths, params, out_size):
        self.paths, self.params, self.out_size = paths, params, out_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return torch.stack([apply_view(img, p, self.out_size) for p in self.params[i]])


def train_folds():
    return add_cv_folds(load_split("train"), n_folds=5, seed=0)   # the HW X teacher's folds


def cmd_params(args):
    df = train_folds()
    out_size = (args.img_size, args.img_width or args.img_size)
    params = np.stack([sample_view_params(np.random.default_rng([args.seed, i]), *Image.open(p).size, out_size, args.views)
                       for i, p in enumerate(df.path)])
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "params.npy", params)
    df[["image", "label", "fold"]].to_csv(args.out / "images.csv", index=False)
    (args.out / "config.json").write_text(json.dumps({"views": args.views, "out_size": out_size, "seed": args.seed,
                                                      "members": args.members}, indent=2))
    print(f"{params.shape} view parameters for {len(df)} images -> {args.out}")


def build_teacher(run, k, device):
    cfg = json.loads((HWY_DIR / "results" / "hwx" / run / f"fold{k}" / "metrics.json").read_text())["config"]
    # ViTs: position embeddings are resampled to the view's patch grid at run time (input padded to a patch multiple).
    kw = dict(img_size=cfg["img_size"], dynamic_img_size=True, dynamic_img_pad=True) if cfg["model"].startswith(_SIZED_FAMILIES) else {}
    model = timm.create_model(cfg["model"], pretrained=False, num_classes=len(CLASSES), **kw)
    model.load_state_dict(torch.load(HWY_DIR / "results" / "hwx" / run / f"fold{k}" / "best.pt", map_location="cpu"))
    return model.to(device).eval().to(memory_format=torch.channels_last)


@torch.no_grad()
def cmd_predict(args):
    cfg = json.loads((args.out / "config.json").read_text())
    df = pd.read_csv(args.out / "images.csv")
    params = np.load(args.out / "params.npy")
    idx = np.where(df.fold == args.fold)[0]
    paths = load_split("train").set_index("image").loc[df.image[idx], "path"].tolist()
    loader = DataLoader(StoredViews(paths, params[idx], tuple(cfg["out_size"])), batch_size=args.batch_images,
                        num_workers=args.num_workers, pin_memory=True)
    device = get_device()
    teachers = [build_teacher(m, args.fold, device) for m in cfg["members"]]
    K = params.shape[1]
    out = np.zeros((len(idx), K, len(teachers), len(CLASSES)), np.float32)
    i = 0
    for b, x in enumerate(loader):
        n = len(x)
        x = x.flatten(0, 1).to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for t, net in enumerate(teachers):
                out[i:i + n, :, t] = torch.log_softmax(net(x).float(), 1).view(n, K, -1).cpu().numpy()
        i += n
        if b % 50 == 0:
            print(f"fold {args.fold}: {i}/{len(idx)} images", flush=True)
    np.save(args.out / f"logp_fold{args.fold}.npy", out)
    y = df.label.to_numpy()[idx]
    for t, m in enumerate(cfg["members"]):
        print(f"{m}: per-view balanced acc {balanced_accuracy(np.repeat(y, K), out[:, :, t].reshape(-1, len(CLASSES)).argmax(1), len(CLASSES)):.4f}")
    print(f"wrote {args.out / f'logp_fold{args.fold}.npy'} {out.shape}")


def cmd_merge(args):
    cfg = json.loads((args.out / "config.json").read_text())
    df = pd.read_csv(args.out / "images.csv")
    y, C = df.label.to_numpy(), len(CLASSES)
    K, M = cfg["views"], len(cfg["members"])
    Z = np.zeros((len(df), K, M, C), np.float32)
    for k in range(5):
        Z[np.where(df.fold == k)[0]] = np.load(args.out / f"logp_fold{k}.npy")
    yk = np.repeat(y, K)
    T = [fit_temperature(Z[:, :, m].reshape(-1, C).astype(np.float64), yk) for m in range(M)]
    ens = np.mean([log_softmax(Z[:, :, m].reshape(-1, C) / T[m]) for m in range(M)], 0)
    soft = np.exp(log_softmax(ens)).reshape(len(df), K, C).astype(np.float32)
    np.save(args.out / "soft_views.tmp.npy", soft)   # renamed when complete: other nodes wait for soft_views.npy
    os.replace(args.out / "soft_views.tmp.npy", args.out / "soft_views.npy")

    # How much do the per-view targets differ from the fixed per-image target used so far?
    fixed = pd.read_csv(FIXED).set_index("image").loc[df.image, CLASSES].to_numpy()
    tv = 0.5 * np.abs(soft - fixed[:, None]).sum(-1)                        # total variation distance, per view
    p_true = soft[np.arange(len(df)), :, y]                                 # (N, K) teacher prob on the true class
    summary = {
        "temperatures": dict(zip(cfg["members"], [round(t, 3) for t in T])),
        "per_view_member_bacc": {m: balanced_accuracy(yk, Z[:, :, i].reshape(-1, C).argmax(1), C) for i, m in enumerate(cfg["members"])},
        "per_view_ensemble_bacc": balanced_accuracy(yk, soft.reshape(-1, C).argmax(1), C),
        "view_avg_ensemble_bacc": balanced_accuracy(y, soft.mean(1).argmax(1), C),
        "fixed_label_bacc": balanced_accuracy(y, fixed.argmax(1), C),
        "tv_to_fixed_mean": float(tv.mean()), "tv_to_fixed_median": float(np.median(tv)),
        "p_true_view_mean": float(p_true.mean()), "p_true_fixed_mean": float(fixed[np.arange(len(df)), y].mean()),
        "p_true_within_image_sd_mean": float(p_true.std(1).mean()),
        "argmax_changes_across_views": float((soft.argmax(-1) != soft.argmax(-1)[:, :1]).any(1).mean()),
        "per_view_argmax_vs_fixed_disagree": float((soft.argmax(-1) != fixed.argmax(-1)[:, None]).mean()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'soft_views.npy'} {soft.shape}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["params", "predict", "merge"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--views", type=int, default=16)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--img-width", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--members", nargs="+", default=MEMBERS, help="HW X runs (results/hwx/<run>) forming the teacher")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-images", type=int, default=4, help="images per batch (x K views)")
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    args = p.parse_args()
    if not args.out.is_absolute():
        args.out = HWY_DIR / args.out
    {"params": cmd_params, "predict": cmd_predict, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
