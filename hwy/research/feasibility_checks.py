"""CPU-only feasibility checks for Homework Y, run before any GPU round (run from hwy/).

1. Metadata: how much do age / sex / body site add on top of the baseline's predictions?
   A class-balanced logistic regression on [baseline log-probs, metadata] is fitted on the baseline's
   out-of-fold CV predictions (nested: fit on 4 folds, score the 5th) and applied once to the test set.
2. Lesion crops: how much would a crop around the lesion zoom in? Bounding boxes of the HAM10000 masks.

    python research/feasibility_checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

HWY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HWY_DIR))
from isic import analysis as A
from isic.data import CLASSES, add_cv_folds, load_split
from isic.metadata import encode_metadata, load_metadata

BASE = "hwx/convnext_s_kd_res448_tta20"   # probabilities before logit adjustment


def bacc(y, p):
    return np.mean([(p[y == c] == c).mean() for c in np.unique(y)])


def oof(run, split):
    dfs = [pd.read_csv(A.RESULTS / run / f"fold{k}" / f"pred_{split}.csv").assign(fold=k) for k in range(5)]
    return pd.concat(dfs) if split == "cv_val" else dfs


def metadata_check():
    meta = load_metadata()
    cv = oof(BASE, "cv_val").merge(meta, on="image")
    te = oof(BASE, "official_test")
    y_cv, y_te = cv["label"].map(CLASSES.index).to_numpy(), te[0]["label"].map(CLASSES.index).to_numpy()
    X_img_cv = np.log(cv[CLASSES].to_numpy().clip(1e-7))
    M_cv = encode_metadata(cv)
    M_te = encode_metadata(te[0][["image"]].merge(meta, on="image", how="left"))
    rows = []
    for name, use_img, use_meta in [("image probs only", 1, 0), ("metadata only", 0, 1), ("image probs + metadata", 1, 1)]:
        feats = lambda Xi, M: np.hstack([Xi] * use_img + [M] * use_meta)
        # nested CV over the baseline's own folds
        pred = np.zeros(len(cv), int)
        for k in range(5):
            tr, va = cv.fold.to_numpy() != k, cv.fold.to_numpy() == k
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(feats(X_img_cv, M_cv)[tr], y_cv[tr])
            pred[va] = lr.predict(feats(X_img_cv, M_cv)[va])
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(feats(X_img_cv, M_cv), y_cv)
        test = [bacc(y_te, lr.predict(feats(np.log(p[CLASSES].to_numpy().clip(1e-7)), M_te))) for p in te]
        rows.append({"stacker": name, "cv nested": bacc(y_cv, pred), "test (per-fold mean)": np.mean(test)})
    print("1. Metadata on top of the baseline's OOF predictions")
    print(f"   reference: baseline + logit adjustment, test {A.per_fold('hwx/convnext_s_kd_res448_tta20_la').bal_acc.mean():.4f}")
    print(pd.DataFrame(rows).set_index("stacker").round(4).to_string(), "\n")
    return rows


def metadata_missing_check():
    """Same stacker, scored separately on the HAM-source and the metadata-free 'external' test images,
    with and without training copies whose metadata is removed (the idea behind modality dropout)."""
    meta = load_metadata()
    cv = oof(BASE, "cv_val").merge(meta, on="image")
    te = oof(BASE, "official_test")
    tm = te[0][["image"]].merge(meta, on="image", how="left")
    ext = (tm.dataset == "external").to_numpy()
    y_cv, y_te = cv["label"].map(CLASSES.index).to_numpy(), te[0]["label"].map(CLASSES.index).to_numpy()
    Xc, Mc, Mt = np.log(cv[CLASSES].to_numpy().clip(1e-7)), encode_metadata(cv), encode_metadata(tm)
    none = encode_metadata(pd.DataFrame({"age": [np.nan], "sex": [None], "localization": [None]}))[0]
    lr0 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Xc, y_cv)
    rows = []
    for aug in (False, True):
        X, y = np.hstack([Xc, Mc]), y_cv
        if aug:
            X, y = np.vstack([X, np.hstack([Xc, np.tile(none, (len(Xc), 1))])]), np.concatenate([y_cv, y_cv])
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X, y)
        for name, m in (("all test", np.ones(len(ext), bool)), ("HAM sources", ~ext), ("external (no metadata)", ext)):
            score = lambda f: np.mean([bacc(y_te[m], f(np.log(p[CLASSES].to_numpy().clip(1e-7)))[m]) for p in te])
            rows.append({"training": "with metadata-removed copies" if aug else "metadata as given", "test subset": name,
                         "image + metadata": score(lambda x: lr.predict(np.hstack([x, Mt]))),
                         "image only": score(lambda x: lr0.predict(x))})
    print("1b. Metadata stacker by test source (test used for diagnosis only, not for choosing models)")
    print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")
    return rows


def lesion_box_check(n=2000):
    tr = load_split("train").sample(n, random_state=0)
    mask_dir = HWY_DIR / "data" / "ham10000_extra" / "masks" / "HAM10000_segmentations_lesion_tschandl"
    rows = []
    for im in tr.image:
        m = np.asarray(Image.open(mask_dir / f"{im}_segmentation.png")) > 127
        ys, xs = np.where(m)
        h, w = m.shape
        bw, bh = (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) if len(xs) else (w, h)
        rows.append({"area": m.mean(), "box_side": max(bw, bh), "touches_border": bool(len(xs)) and
                     (xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1)})
    d = pd.DataFrame(rows)
    side = (d.box_side * 1.3).clip(upper=450)   # square crop with a 30% margin, at most the image height
    zoom = 394 / side   # the baseline's centre view covers 394 px of the original image
    print(f"2. Lesion boxes ({n} training masks, image 600x450)")
    print(f"   lesion area fraction: median {d.area.median():.2f}, 10th pct {d.area.quantile(.1):.2f}, 90th pct {d.area.quantile(.9):.2f}")
    print(f"   lesion touches the image border: {d.touches_border.mean():.0%}")
    print(f"   zoom vs the baseline view (crop = 1.3 x box side): >1.25x for {(zoom > 1.25).mean():.0%}, "
          f">1.5x for {(zoom > 1.5).mean():.0%}, median {zoom.median():.2f}x")
    return {"area_median": d.area.median(), "touches_border": d.touches_border.mean(), "zoom_gt_1.25": (zoom > 1.25).mean(),
            "zoom_gt_1.5": (zoom > 1.5).mean(), "zoom_median": zoom.median()}


if __name__ == "__main__":
    out = {"metadata_stacker": metadata_check(), "metadata_by_source": metadata_missing_check(), "lesion_boxes": lesion_box_check()}
    f = HWY_DIR / "results" / "comparisons" / "feasibility.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=2, default=float))
