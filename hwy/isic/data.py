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
    """Image dataset over a DataFrame with `path` and `label` columns.

    lesion_crop (optional): a LesionCrop applied before `transform`, with `boxes` mapping image id -> lesion box.
    """

    def __init__(self, df: pd.DataFrame, transform, lesion_crop=None, boxes: dict = None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy()
        self.transform = transform
        self.lesion_crop = lesion_crop
        self.boxes = np.stack([boxes[im] for im in df["image"]]) if lesion_crop is not None else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.lesion_crop is not None:
            img = self.lesion_crop(img, self.boxes[idx])
        return self.transform(img), int(self.labels[idx])


class SoftLabelDataset(ISICDataset):
    """ISICDataset whose target is a vector [hard label, p_1, ..., p_C] (teacher soft labels).

    `soft` maps image id -> length-C probability vector; every image in `df` must have one.
    """

    def __init__(self, df: pd.DataFrame, transform, soft: dict, **kwargs):
        super().__init__(df, transform, **kwargs)
        self.soft = np.stack([soft[im] for im in df["image"]]).astype(np.float32)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, torch.cat([torch.tensor([float(y)]), torch.from_numpy(self.soft[idx])])


class ViewSoftLabelDataset(ISICDataset):
    """Consistent distillation: each item is one of K stored training views of the image, with the teacher's soft
    label for exactly that view (scripts/teacher_views.py). Target is [hard label, p_1, ..., p_C], as SoftLabelDataset.

    `params` maps image id -> (K, VIEW_PARAMS) view parameters and `soft` maps image id -> (K, C) teacher probabilities.
    """

    def __init__(self, df: pd.DataFrame, params: dict, soft: dict, out_size):
        super().__init__(df, transform=None)
        self.params = np.stack([params[im] for im in df["image"]])
        self.soft = np.stack([soft[im] for im in df["image"]]).astype(np.float32)
        self.out_size = out_size

    def __getitem__(self, idx):
        v = int(torch.randint(self.params.shape[1], ()))
        x = apply_view(Image.open(self.paths[idx]).convert("RGB"), self.params[idx, v], self.out_size)
        return x, torch.cat([torch.tensor([float(self.labels[idx])]), torch.from_numpy(self.soft[idx, v])])


class MetaDataset(Dataset):
    """Wraps an image dataset so that each input is (image, metadata vector) (see isic/metadata.py).

    Modality dropout (training only): with probability drop_all every field is set to missing, otherwise each field
    (age, sex, site) independently with probability drop_field. Missing = zero values + the field's 'missing' flag.
    """

    def __init__(self, base: Dataset, meta: np.ndarray, field_slices, drop_all: float = 0.0, drop_field: float = 0.0):
        self.base, self.meta = base, torch.from_numpy(meta)
        self.field_slices = field_slices   # [(value slice, missing-flag index)] per field
        self.drop_all, self.drop_field = drop_all, drop_field

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        m = self.meta[idx].clone()
        drop_all = self.drop_all > 0 and torch.rand(()) < self.drop_all
        for values, flag in self.field_slices:
            if drop_all or (self.drop_field > 0 and torch.rand(()) < self.drop_field):
                m[values] = 0.0
                m[flag] = 1.0
        return (x, m), y


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


class LesionCrop:
    """Square crop centred on the lesion box (x0, y0, x1, y1), before the usual pipeline.

    side = margin x the longer box side, clamped to [min_side, image height]; the square is shifted to lie inside the
    image. train=True draws margin ~ U(margin_range) and shifts the centre by up to `jitter` x side.
    """

    def __init__(self, train: bool, margin: float = 1.3, margin_range=(1.1, 1.6), jitter: float = 0.1,
                 min_side: int = 225):
        self.train, self.margin, self.margin_range, self.jitter, self.min_side = train, margin, margin_range, jitter, min_side

    def __call__(self, img: Image.Image, box) -> Image.Image:
        w, h = img.size
        x0, y0, x1, y1 = box
        margin = np.random.uniform(*self.margin_range) if self.train else self.margin
        side = float(np.clip(margin * max(x1 - x0, y1 - y0), self.min_side, min(w, h)))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if self.train:
            cx, cy = cx + np.random.uniform(-1, 1) * self.jitter * side, cy + np.random.uniform(-1, 1) * self.jitter * side
        left = float(np.clip(cx - side / 2, 0, w - side))
        top = float(np.clip(cy - side / 2, 0, h - side))
        return img.crop((round(left), round(top), round(left + side), round(top + side)))


def baseline_transforms(img_size: int = 224, resize: int = 256, augment: str = "basic", color_constancy: bool = False,
                        width: int = None):
    """Standard ImageNet-style fine-tuning pipeline (downscale, then crop).

    augment="basic":   mild color jitter (the baseline).
    augment="trivial": TrivialAugmentWide (one random op per image, random magnitude) instead of the jitter.
    augment="color":   stronger color jitter (0.3 / 0.3 / 0.3, hue 0.1), for robustness to acquisition shift.
    color_constancy:   apply Shades-of-Grey colour constancy to every image first (train and eval).
    width:             crop width for rectangular (full-frame) input; default square. Same pixel scale either way.
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
    aspect = (width or img_size) / img_size
    size = (img_size, width) if width else img_size
    train = T.Compose(pre + [
        T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(aspect * 3 / 4, aspect * 4 / 3)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        color,
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluation = T.Compose(pre + [
        T.Resize(resize),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluation


# Stored training views (consistent distillation): one row per view =
# top, left, height, width (crop box), hflip, vflip, brightness, contrast, saturation, hue, order of the 4 colour ops.
VIEW_PARAMS = 14


def sample_view_params(rng: np.random.Generator, img_w: int, img_h: int, out_size, n_views: int,
                       scale=(0.4, 1.0), jitter=(0.2, 0.2, 0.2, 0.02)) -> np.ndarray:
    """n_views random training views as parameter rows (n_views, VIEW_PARAMS).

    Same distribution as the train pipeline of baseline_transforms(augment="basic") at crop size out_size = (h, w):
    RandomResizedCrop (torchvision's sampling rule, incl. its centre-crop fallback), H and V flips, ColorJitter.
    Drawn from a numpy Generator so the views can be stored, given to a teacher, and replayed by apply_view.
    """
    aspect = out_size[1] / out_size[0]
    ratio = (aspect * 3 / 4, aspect * 4 / 3)
    area = img_w * img_h
    rows = np.zeros((n_views, VIEW_PARAMS), np.float32)
    for v in range(n_views):
        for _ in range(10):
            target = area * rng.uniform(*scale)
            r = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
            w, h = round(np.sqrt(target * r)), round(np.sqrt(target / r))
            if 0 < w <= img_w and 0 < h <= img_h:
                top, left = rng.integers(0, img_h - h + 1), rng.integers(0, img_w - w + 1)
                break
        else:
            in_ratio = img_w / img_h
            w, h = (img_w, round(img_w / ratio[0])) if in_ratio < ratio[0] else \
                   (round(img_h * ratio[1]), img_h) if in_ratio > ratio[1] else (img_w, img_h)
            top, left = (img_h - h) // 2, (img_w - w) // 2
        b, c, s, hue = jitter
        rows[v] = [top, left, h, w, rng.random() < 0.5, rng.random() < 0.5,
                   rng.uniform(1 - b, 1 + b), rng.uniform(1 - c, 1 + c), rng.uniform(1 - s, 1 + s), rng.uniform(-hue, hue),
                   *rng.permutation(4)]
    return rows


def apply_view(img: Image.Image, p, out_size) -> torch.Tensor:
    """Replay one stored view (a row from sample_view_params) on a PIL image -> normalised tensor (3, h, w)."""
    top, left, h, w = (int(v) for v in p[:4])
    img = TF.resized_crop(img, top, left, h, w, list(out_size), antialias=True)
    if p[4] > 0.5:
        img = TF.hflip(img)
    if p[5] > 0.5:
        img = TF.vflip(img)
    ops = (lambda x: TF.adjust_brightness(x, float(p[6])), lambda x: TF.adjust_contrast(x, float(p[7])),
           lambda x: TF.adjust_saturation(x, float(p[8])), lambda x: TF.adjust_hue(x, float(p[9])))
    for o in p[10:14]:
        img = ops[int(o)](img)
    return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)


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
    width gives rectangular crops (img_size x width) for full-frame models.
    Returns a tensor (n_views, 3, img_size, width); predict() averages over the views.
    """

    def __init__(self, img_size: int = 224, resize: int = 256, crops: int = 1, flips: bool = True,
                 color_constancy: bool = False, width: int = None):
        assert crops in (1, 5)
        self.img_size = (img_size, width) if width else img_size
        self.resize, self.crops, self.flips = resize, crops, flips
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
