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


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None, amp=True, desc="train") -> float:
    """One pass over `loader`. `scheduler` (if given) is stepped every iteration."""
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with _autocast(device, amp):
            loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, amp=True, desc="eval"):
    """Return (probs (N, C), labels (N,)).

    Inputs of shape (B, K, 3, H, W) are treated as K crops per image; softmax
    probabilities are averaged over the crops (multi-crop evaluation).
    """
    model.eval()
    all_probs, all_labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        n_crops = x.shape[1] if x.ndim == 5 else 1
        if x.ndim == 5:
            x = x.flatten(0, 1)
        with _autocast(device, amp):
            logits = model(x.to(memory_format=torch.channels_last))
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
