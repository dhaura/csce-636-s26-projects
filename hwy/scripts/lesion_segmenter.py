#!/usr/bin/env python
"""Lesion segmenter for lesion-centred crops: train per CV fold, then turn predicted masks into lesion boxes.

Model: ImageNet ResNet-50 encoder (timm, features at strides 4-32) + a small FPN decoder, 384 x 288 input,
BCE + Dice loss, trained on the HAM10000 lesion masks (Tschandl et al., data/ham10000_extra/masks).
Folds are the classifier's CV folds, so every training image gets its mask from the model that did not see it.
Validation / test images get the mean of the 5 fold models. Boxes always come from predicted masks.

    python scripts/lesion_segmenter.py train --fold 0      # -> results/lesion_seg/fold0/{best.pt, metrics.json, ...}
    python scripts/lesion_segmenter.py boxes               # -> results/lesion_seg/boxes.csv
"""
import sys
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hwy/, so the isic package imports
from isic.data import IMAGENET_MEAN, IMAGENET_STD, add_cv_folds, load_split
from isic.engine import get_device, save_json, seed_everything

HWY_DIR = Path(__file__).resolve().parents[1]
MASK_DIR = HWY_DIR / "data" / "ham10000_extra" / "masks" / "HAM10000_segmentations_lesion_tschandl"
OUT = HWY_DIR / "results" / "lesion_seg"
H, W = 288, 384


class SegDataset(Dataset):
    def __init__(self, df, train=False, masks=True):
        self.images, self.paths, self.train, self.masks = df["image"].tolist(), df["path"].tolist(), train, masks

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = TF.to_tensor(Image.open(self.paths[i]).convert("RGB").resize((W, H), Image.BILINEAR))
        y = (TF.to_tensor(Image.open(MASK_DIR / f"{self.images[i]}_segmentation.png").convert("L")
                          .resize((W, H), Image.NEAREST)) > 0.5).float() if self.masks else torch.zeros(1, H, W)
        if self.train:
            if torch.rand(()) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(()) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            x = TF.adjust_brightness(x, float(np.random.uniform(0.8, 1.2)))
            x = TF.adjust_saturation(x, float(np.random.uniform(0.8, 1.2)))
        return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD), y


class FPNSeg(nn.Module):
    def __init__(self, encoder="resnet50.tv2_in1k", pretrained=True, ch=128):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        self.lat = nn.ModuleList(nn.Conv2d(c, ch, 1) for c in self.enc.feature_info.channels())
        self.smooth = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                                    nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        feats = self.enc(x)
        y = self.lat[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), reversed(self.lat[:-1])):
            y = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False) + lat(f)
        return F.interpolate(self.out(self.smooth(y)), size=x.shape[-2:], mode="bilinear", align_corners=False)


def loss_fn(logits, y):
    p = logits.sigmoid()
    dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, y) + dice.mean()


@torch.no_grad()
def predict_probs(model, loader, device):
    """Sigmoid maps (N, H, W), averaged with the horizontal flip."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = (model(x).float().sigmoid() + model(x.flip(-1)).float().sigmoid().flip(-1)) / 2
        out.append(p[:, 0].cpu())
    return torch.cat(out).numpy()


def jaccard(prob, mask):
    pred = prob > 0.5
    inter, union = (pred & mask).sum((1, 2)), (pred | mask).sum((1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def train(args):
    seed_everything(args.seed)
    device = get_device()
    df = add_cv_folds(load_split("train"))
    tr, va = df[df.fold != args.fold], df[df.fold == args.fold]
    if args.smoke_test:
        tr, va, args.epochs = tr.head(64), va.head(32), 1
    out = OUT / (f"fold{args.fold}" + ("_smoke" if args.smoke_test else ""))
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(SegDataset(tr, train=True), batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(SegDataset(va), batch_size=64, **kw)

    model = FPNSeg().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / len(train_loader)) * 0.5 * (1 + np.cos(np.pi * s / total)))
    masks_va = np.stack([SegDataset(va)[i][1][0].numpy() > 0.5 for i in range(len(va))]) if not args.smoke_test else None
    best, t0, history = -1, time.time(), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        probs = predict_probs(model, val_loader, device)
        jac = float(jaccard(probs, masks_va).mean()) if masks_va is not None else float("nan")
        history.append({"epoch": epoch, "loss": loss.item(), "val_jaccard": jac})
        if jac > best or masks_va is None:
            best = jac
            torch.save(model.state_dict(), out / "best.pt")
        print(f"[{time.time() - t0:5.0f}s] epoch {epoch:2d} | loss {loss.item():.4f} | val Jaccard {jac:.4f} | best {best:.4f}", flush=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # predictions with the best checkpoint: out-of-fold maps for this fold, full maps for official val + test
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    np.save(out / "probs_cv_val.npy", (predict_probs(model, val_loader, device) * 255).astype(np.uint8))
    ev = pd.concat([load_split("val"), load_split("test")])
    if args.smoke_test:
        ev = ev.head(16)
    np.save(out / "probs_official.npy", (predict_probs(model, DataLoader(SegDataset(ev, masks=False), batch_size=64, **kw), device) * 255).astype(np.uint8))
    pd.Series(va.image.tolist()).to_csv(out / "images_cv_val.csv", index=False, header=["image"])
    pd.Series(ev.image.tolist()).to_csv(out / "images_official.csv", index=False, header=["image"])
    save_json({"fold": args.fold, "best_val_jaccard": best, "epochs": args.epochs, "train_time_min": (time.time() - t0) / 60}, out / "metrics.json")


def box_from_prob(prob, size):
    """Largest connected component of prob > 0.5 -> (x0, y0, x1, y1) in original pixels; empty -> whole image."""
    w, h = size
    lab, n = ndimage.label(prob > 127)
    if n == 0:
        return (0.0, 0.0, float(w), float(h)), 0.0
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    ys, xs = np.where(lab == 1 + int(np.argmax(sizes)))
    sx, sy = w / W, h / H
    return (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy), float(sizes.max() / prob.size)


def boxes(args):
    rows, jac = [], []
    paths = {r.image: r.path for d in (load_split("train"), load_split("val"), load_split("test")) for r in d.itertuples()}
    official = []
    for k in range(5):
        d = OUT / f"fold{k}"
        images = pd.read_csv(d / "images_cv_val.csv").image
        probs = np.load(d / "probs_cv_val.npy")
        for im, p in zip(images, probs):
            m = np.asarray(Image.open(MASK_DIR / f"{im}_segmentation.png").convert("L").resize((W, H), Image.NEAREST)) > 127
            jac.append(jaccard(p[None] / 255.0, m[None])[0])
            box, area = box_from_prob(p, Image.open(paths[im]).size)
            rows.append({"image": im, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "area": area, "source": f"oof_fold{k}"})
        official.append(np.load(d / "probs_official.npy").astype(np.float32))
    images = pd.read_csv(OUT / "fold0" / "images_official.csv").image
    for im, p in zip(images, np.mean(official, axis=0)):
        box, area = box_from_prob(p, Image.open(paths[im]).size)
        rows.append({"image": im, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "area": area, "source": "mean5"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "boxes.csv", index=False)
    save_json({"oof_jaccard_mean": float(np.mean(jac)), "oof_jaccard_median": float(np.median(jac)),
               "oof_jaccard_below_0.65": float(np.mean(np.array(jac) < 0.65)), "n_boxes": len(df),
               "empty_masks": int((df.area == 0).sum())}, OUT / "boxes_summary.json")
    print(f"wrote {OUT / 'boxes.csv'}: {len(df)} boxes | OOF Jaccard mean {np.mean(jac):.4f} median {np.median(jac):.4f} "
          f"| empty masks {(df.area == 0).sum()}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "boxes"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    train(args) if args.command == "train" else boxes(args)


if __name__ == "__main__":
    main()
