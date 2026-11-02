"""Training / inference loops shared by the notebook and the literature-baseline script."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import CLASSES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device, amp: bool):
    # bf16 autocast on GPU: no GradScaler needed, numerically safe on A100.
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda")


def _forward(model, x, device):
    """Run `model` on a batch. x is images (B, 3, H, W) or (B, K, 3, H, W) for K views per image, or a pair
    (images, metadata) for MetaFusion models. Returns (logits (B*K, C), K)."""
    img, meta = x if isinstance(x, (list, tuple)) else (x, None)
    img = img.to(device, non_blocking=True)
    n_views = img.shape[1] if img.ndim == 5 else 1
    if img.ndim == 5:
        img = img.flatten(0, 1)
    img = img.to(memory_format=torch.channels_last)
    if meta is None:
        return model(img), n_views
    return model(img, meta.to(device, non_blocking=True).repeat_interleave(n_views, dim=0)), n_views


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None, amp=True, desc="train", ema=None) -> float:
    """One pass over `loader`. `scheduler` (if given) is stepped every iteration; `ema` (timm ModelEmaV3) updated too."""
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        y = y.to(device, non_blocking=True)
        with _autocast(device, amp):
            loss = loss_fn(_forward(model, x, device)[0], y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model, step=scheduler.last_epoch if scheduler is not None else None)
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, amp=True, desc="eval"):
    """Return (probs (N, C), labels (N,)).

    Inputs of shape (B, K, 3, H, W) are treated as K crops per image; softmax
    probabilities are averaged over the crops (multi-crop evaluation). (images, metadata) pairs are supported.
    """
    model.eval()
    all_probs, all_labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        with _autocast(device, amp):
            logits, n_crops = _forward(model, x, device)
        probs = logits.float().softmax(dim=1).view(-1, n_crops, logits.shape[1]).mean(dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def save_predictions(path, images, probs, labels=None) -> None:
    """Save per-image class probabilities in the ISIC submission format (+ true label if known)."""
    df = pd.DataFrame(probs, columns=CLASSES)
    df.insert(0, "image", list(images))
    if labels is not None:
        df["label"] = [CLASSES[i] for i in labels]
    df.to_csv(path, index=False)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
