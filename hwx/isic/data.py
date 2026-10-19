"""ISIC 2018 Task 3 data loading: label CSVs, lesion-aware CV folds, datasets, transforms."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# Column order of the official ground-truth CSVs.
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis / Bowen's",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
}
NUM_CLASSES = len(CLASSES)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "isic2018"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_SPLIT_DIRS = {"train": "Training", "val": "Validation", "test": "Test"}


def load_split(split: str, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Return a DataFrame with columns [image, path, label, dx] for an official split.

    split: "train" (10015 HAM10000 images), "val" (193, official validation) or "test" (1512, official test).
    For "train" the lesion grouping (lesion_id, diagnosis_confirm_type) is merged in.
    """
    name = _SPLIT_DIRS[split]
    root = Path(root)
    gt = pd.read_csv(root / f"ISIC2018_Task3_{name}_GroundTruth" / f"ISIC2018_Task3_{name}_GroundTruth.csv")
    df = pd.DataFrame({
        "image": gt["image"],
        "path": [str(root / f"ISIC2018_Task3_{name}_Input" / f"{im}.jpg") for im in gt["image"]],
        "label": gt[CLASSES].to_numpy().argmax(axis=1),
    })
    df["dx"] = [CLASSES[i] for i in df["label"]]
    if split == "train":
        groups = pd.read_csv(root / "ISIC2018_Task3_Training_LesionGroupings.csv")
        df = df.merge(groups[["image", "lesion_id", "diagnosis_confirm_type"]], on="image", how="left")
        assert df["lesion_id"].notna().all(), "every training image should have a lesion_id"
    return df


def add_cv_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Assign a `fold` column using class-stratified folds grouped by lesion_id.

    Grouping by lesion prevents near-duplicate images of the same lesion from
    landing in both training and validation (a common source of inflated scores).
    """
    df = df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["lesion_id"])):
        df.loc[df.index[val_idx], "fold"] = k
    return df


def class_counts(labels) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=NUM_CLASSES)


def class_weights(labels, mode: str = "inverse"):
    """Per-class loss weights (Gessert et al. 2018).

    inverse:   w_i = N / n_i        inverse_c: w_i = N / (C * n_i)        none: None
    Note: nn.CrossEntropyLoss(weight=w) normalises by the summed weights of the batch,
    so `inverse` and `inverse_c` give identical gradients; only the class ratios matter.
    """
    if mode == "none":
        return None
    counts = class_counts(labels).astype(np.float64)
    w = counts.sum() / counts
    if mode == "inverse_c":
        w /= len(counts)
    elif mode != "inverse":
        raise ValueError(f"unknown class weighting {mode!r}")
    return torch.tensor(w, dtype=torch.float32)


class ISICDataset(Dataset):
    """Image dataset over a DataFrame with `path` and `label` columns."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), int(self.labels[idx])


class SoftLabelDataset(ISICDataset):
    """ISICDataset whose target is a vector [hard label, p_1, ..., p_C] (teacher soft labels).

    `soft` maps image id -> length-C probability vector; every image in `df` must have one.
    """

    def __init__(self, df: pd.DataFrame, transform, soft: dict):
        super().__init__(df, transform)
        self.soft = np.stack([soft[im] for im in df["image"]]).astype(np.float32)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, torch.cat([torch.tensor([float(y)]), torch.from_numpy(self.soft[idx])])


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class ShadesOfGray:
    """Shades-of-Grey colour constancy (Finlayson & Trezzi 2004).

    Estimates the illuminant per channel as the Minkowski p-norm mean of the pixel values and rescales the
    channels so that the illuminant becomes grey. p = 6 follows the ISIC 2019 winning entry (Gessert et al.).
    """

    def __init__(self, p: float = 6.0):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        # Illuminant from a 4x-downsampled copy (the p-mean barely changes), then a per-channel lookup table.
        small = np.asarray(img.reduce(4), dtype=np.float64)
        illum = np.power(np.mean(np.power(small, self.p), axis=(0, 1)), 1.0 / self.p)
        gain = 1.0 / (illum / np.sqrt(np.sum(illum ** 2)) * np.sqrt(3.0))
        lut = [min(255, int(round(v * g))) for g in gain for v in range(256)]
        return img.point(lut)


def baseline_transforms(img_size: int = 224, resize: int = 256, augment: str = "basic", color_constancy: bool = False):
    """Standard ImageNet-style fine-tuning pipeline (downscale, then crop).

    augment="basic":   mild color jitter (the baseline).
    augment="trivial": TrivialAugmentWide (one random op per image, random magnitude) instead of the jitter.
    augment="color":   stronger color jitter (0.3 / 0.3 / 0.3, hue 0.1), for robustness to acquisition shift.
    color_constancy:   apply Shades-of-Grey colour constancy to every image first (train and eval).
    """
    if augment == "basic":
        color = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
    elif augment == "trivial":
        color = T.TrivialAugmentWide()
    elif augment == "color":
        color = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    else:
        raise ValueError(f"unknown augment {augment!r}")
    pre = [ShadesOfGray()] if color_constancy else []
    train = T.Compose(pre + [
        T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        color,
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluation = T.Compose(pre + [
        T.Resize(resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluation


def gessert_train_transform(crop: int = 224):
    """Gessert et al. 2018: unscaled random crops from the full 600x450 image + flips + color jitter."""
    return T.Compose([
        T.RandomCrop(crop),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TTAViews:
    """Test-time augmentation for the baseline pipeline: resize, then crops x flips.

    crops=1: center crop only; crops=5: center + 4 corners (FiveCrop).
    flips=True adds horizontal, vertical and both flips of every crop (x4 views).
    Returns a tensor (n_views, 3, img_size, img_size); predict() averages over the views.
    """

    def __init__(self, img_size: int = 224, resize: int = 256, crops: int = 1, flips: bool = True,
                 color_constancy: bool = False):
        assert crops in (1, 5)
        self.img_size, self.resize, self.crops, self.flips = img_size, resize, crops, flips
        self.cc = ShadesOfGray() if color_constancy else None
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    @property
    def n_views(self) -> int:
        return self.crops * (4 if self.flips else 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.cc is not None:
            img = self.cc(img)
        x = TF.to_tensor(TF.resize(img, self.resize))
        crops = list(TF.five_crop(x, self.img_size)) if self.crops == 5 else [TF.center_crop(x, self.img_size)]
        views = []
        for c in crops:
            views.append(c)
            if self.flips:
                views += [TF.hflip(c), TF.vflip(c), TF.vflip(TF.hflip(c))]
        return self.normalize(torch.stack(views))


class GridCrops:
    """Unscaled multi-crop evaluation: n x n equally spaced crops covering the image.

    Returns a tensor of shape (n*n, 3, crop, crop). n=6 gives the paper's 36 crops,
    n=4 the 16 crops used for validation during training.
    """

    def __init__(self, n_per_side: int, crop: int = 224):
        self.n = n_per_side
        self.crop = crop
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        x = TF.to_tensor(img)
        if min(w, h) < self.crop:  # never expected for ISIC (600x450) but keep it safe
            x = TF.resize(x, self.crop)
            h, w = x.shape[-2:]
        tops = np.linspace(0, h - self.crop, self.n).round().astype(int)
        lefts = np.linspace(0, w - self.crop, self.n).round().astype(int)
        crops = [x[:, t:t + self.crop, l:l + self.crop] for t in tops for l in lefts]
        return self.normalize(torch.stack(crops))
