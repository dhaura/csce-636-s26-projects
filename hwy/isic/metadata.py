"""Patient metadata (age, sex, body site) for HAM10000 training images and the ISIC 2018 test images.

Source: HAM10000 on Harvard Dataverse (doi:10.7910/DVN/DBW86T), downloaded to data/ham10000_extra/.
Only fields available at diagnosis time are used. `dx_type` (how the diagnosis was confirmed) and
`dataset` (which clinic / device) are never model inputs: both are tied to the label or the acquisition.
The official validation images have no metadata; they are encoded as all-missing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

EXTRA_ROOT = Path(__file__).resolve().parents[1] / "data" / "ham10000_extra"
SEXES = ["male", "female"]
SITES = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand", "lower extremity", "neck",
         "scalp", "trunk", "upper extremity"]
# [age/100, age missing] + sex one-hot + [sex missing] + site one-hot + [site missing]
N_META = 2 + len(SEXES) + 1 + len(SITES) + 1


def load_metadata(root: Path = EXTRA_ROOT) -> pd.DataFrame:
    """One row per image (train + test) with columns [image, age, sex, localization, dataset]."""
    tabs = [pd.read_csv(root / f) for f in ("HAM10000_metadata.tab", "ISIC2018_Task3_Test_GroundTruth.tab")]
    df = pd.concat(tabs, ignore_index=True).rename(columns={"image_id": "image"})
    return df[["image", "age", "sex", "localization", "dataset"]].drop_duplicates("image")


def encode_metadata(df: pd.DataFrame) -> np.ndarray:
    """(N, N_META) float32 array; unknown or missing fields set their 'missing' flag and leave the rest zero."""
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy()
    sex = df["sex"].fillna("unknown").astype(str).to_numpy()
    site = df["localization"].fillna("unknown").astype(str).to_numpy()
    out = np.zeros((len(df), N_META), np.float32)
    out[:, 0] = np.nan_to_num(age / 100.0)
    out[:, 1] = np.isnan(age)
    for j, s in enumerate(SEXES):
        out[:, 2 + j] = sex == s
    out[:, 2 + len(SEXES)] = ~np.isin(sex, SEXES)
    o = 3 + len(SEXES)
    for j, s in enumerate(SITES):
        out[:, o + j] = site == s
    out[:, o + len(SITES)] = ~np.isin(site, SITES)
    return out


# (value columns, missing-flag column) per field, for modality dropout in MetaDataset
FIELD_SLICES = [(slice(0, 1), 1), (slice(2, 2 + len(SEXES)), 2 + len(SEXES)),
                (slice(3 + len(SEXES), 3 + len(SEXES) + len(SITES)), N_META - 1)]


def metadata_for(images, missing: bool = False) -> np.ndarray:
    """Encoded metadata for a list of image ids; missing=True encodes every field as missing (image-only input)."""
    df = pd.DataFrame({"image": list(images)}).merge(load_metadata(), on="image", how="left")
    if missing:
        df[["age", "sex", "localization"]] = np.nan
    return encode_metadata(df)
