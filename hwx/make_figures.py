#!/usr/bin/env python
"""Slide-ready figures for the presentation, regenerated from results/ (and one checkpoint for Grad-CAM).

    python make_figures.py                 # all figures -> figures/*.png (+ .pdf for charts)
    python make_figures.py --only fig03 fig08
    python make_figures.py --no-gradcam    # skip the GPU figure

Every number comes from saved runs; the "final model" is the pre-registered round-4 ensemble (dl_final8_calib_tau).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from isic import analysis as A  # noqa: E402
from isic.data import CLASS_NAMES, CLASSES, load_split  # noqa: E402

HWX = Path(__file__).resolve().parent
OUT = HWX / "figures"

# Palette (validated categorical slots, text in ink colours, never in series colours)
INK, INK2, MUTED, GRID, BASE = "#15171b", "#4b5059", "#80858e", "#e3e5e9", "#c3c6cc"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ACCENT = "#a3315b"
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 12,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "text.color": INK,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15, "legend.frameon": False,
})

FINAL = "dl_final8_calib_tau"
BASELINE = "resnet50_ce"
SOTA = [("Gessert et al. 2018\n(2nd place, HAM10000)", 0.856), ("ISIC 2018 winner\n(external data)", 0.885),
        ("Sun et al. 2021\n(ext. data + metadata)", 0.895)]


def save(fig, name, pdf=True):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    if pdf:
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)


def vgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)


def mean_test(run):
    return A.per_fold(run).bal_acc.mean()


def all_models(run, split="official_test"):
    ps = [pd.read_csv(A.RESULTS / run / f"fold{k}" / f"pred_{split}.csv") for k in A.folds(run)]
    probs = np.mean([p[CLASSES].to_numpy() for p in ps], 0)
    return ps[0]["image"].to_numpy(), ps[0]["label"].map(CLASSES.index).to_numpy(), probs


def bacc(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(len(CLASSES))]))


def boot_ci(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = [np.where(y == c)[0] for c in range(len(CLASSES))]
    bs = [bacc(y[b], pred[b]) for b in (np.concatenate([rng.choice(i, len(i)) for i in idx]) for _ in range(n))]
    return np.percentile(bs, [2.5, 97.5])


# ---------------------------------------------------------------------------------------------
def fig01_classes():
    tr = load_split("train")
    counts = tr.dx.value_counts()
    # deterministic, typical-looking example per class: the image the final model is most confident about (CV fold preds)
    picks = {}
    for k in A.folds(FINAL):
        df = pd.read_csv(A.RESULTS / FINAL / f"fold{k}" / "pred_cv_val.csv")
        for c in CLASSES:
            d = df[df.label == c]
            best = d.loc[d[c].idxmax()]
            if c not in picks or best[c] > picks[c][1]:
                picks[c] = (best["image"], best[c])
    fig, axes = plt.subplots(1, 7, figsize=(17, 3.6))
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.06)
    path = dict(zip(tr.image, tr.path))
    for ax, c in zip(axes, CLASSES):
        ax.imshow(Image.open(path[picks[c][0]]).convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(c, fontsize=14, color=INK, pad=6)
        ax.text(0.5, -0.08, f"{CLASS_NAMES[c]}\n{counts[c]:,} training images", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=INK2)
    fig.suptitle("Seven diagnoses from one dermoscopy image (ISIC 2018 Task 3 / HAM10000)", x=0.01, ha="left", fontsize=15, fontweight="bold", y=1.04)
    save(fig, "fig01_classes", pdf=False)


def fig02_imbalance():
    tr = load_split("train"); n = tr.dx.value_counts().reindex(CLASSES)
    order = n.sort_values().index
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(range(7), n[order], color=BLUE, height=0.6)
    for i, c in enumerate(order):
        ax.text(n[c] + 80, i, f"{n[c]:,}  ({n[c] / n.sum():.1%})", va="center", fontsize=11, color=INK2)
    ax.set_yticks(range(7), [f"{c}  {CLASS_NAMES[c]}" for c in order]); ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(0, 8200); vgrid(ax); ax.set_xlabel("training images")
    ax.set_title("Two-thirds of the images are moles", loc="left")
    ax.text(0, -0.22, "Always predicting NV gives 67% accuracy but only 14% balanced accuracy (mean per-class recall),\n"
            "which is why balanced accuracy is the official metric.", transform=ax.transAxes, fontsize=10.5, color=INK2, va="top")
    save(fig, "fig02_imbalance")


def fig03_progress():
    _, ye, pe = all_models("ens5_res448_tta20_la")
    steps = [("Plain ResNet-50\n(224 px)", mean_test("resnet50_ce"), "single"),
             ("+ logit\nadjustment", mean_test("resnet50_ce_la"), "single"),
             ("+ 448 px\ninput", mean_test("resnet50_ce_res448_la"), "single"),
             ("+ 20-view\nTTA", mean_test("resnet50_ce_res448_tta20_la"), "single"),
             ("Distilled\nConvNeXt-S", mean_test("convnext_s_kd_res448_tta20_la"), "single"),
             ("Ensemble of\n5 backbones", bacc(ye, pe.argmax(1)), "ens")]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(steps))
    vals = [s[1] for s in steps]
    colors = [BLUE if s[2] == "single" else ACCENT for s in steps]
    ax.bar(x, vals, color=colors, width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold", color=INK)
    for (lab, v), ls in zip(SOTA, [":", "--", "-"]):
        ax.axhline(v, color=MUTED, lw=1.3, ls=ls, zorder=0)
        ax.text(-0.42, v + 0.001, f"{lab.split(chr(10))[0]} · {v:.3f}", va="bottom", ha="left", fontsize=10, color=MUTED)
    ax.set_xticks(x, [s[0] for s in steps], fontsize=11); ax.tick_params(axis="x", colors=INK, length=0)
    ax.set_ylim(0.65, 0.92); ax.set_ylabel("test balanced accuracy"); hgrid(ax)
    ax.set_title("From 0.724 to 0.892 on the official ISIC 2018 test set", loc="left")
    ax.text(0, -0.2, "Blue: one network (mean of 5 fold models). Rose: average of 25 networks. Dashed lines: published results.\n"
            "Every choice made on cross-validation folds, never on the test set.", transform=ax.transAxes, fontsize=10, color=INK2, va="top")
    save(fig, "fig03_progress")


def fig04_sota():
    rows = []
    for label, run in [("Ours: optimised ResNet-50\n(5 fold models)", "resnet50_ce_res448_tta20_la"),
                       ("Ours: distilled ConvNeXt-S\n(5 fold models)", "convnext_s_kd_res448_tta20_la"),
                       ("Ours: final model, pre-registered\n(5 backbones + 3 students)", FINAL),
                       ("Ours: 5 backbones\n(25 networks)", "ens5_res448_tta20_la")]:
        _, y, p = all_models(run); pred = p.argmax(1)
        rows.append((label, bacc(y, pred), *boot_ci(y, pred), "ours"))
    rows += [(lab, v, np.nan, np.nan, "pub") for lab, v in SOTA]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for i, (lab, v, lo, hi, kind) in enumerate(rows):
        if kind == "ours":
            ax.plot([lo, hi], [i, i], color=BLUE, lw=3, solid_capstyle="round", alpha=0.45)
            ax.plot(v, i, "o", ms=10, color=BLUE, mec="white", mew=2)
        else:
            ax.plot(v, i, "D", ms=9, color=MUTED, mec="white", mew=1.5)
        ax.text(0.925, i, f"{v:.3f}", va="center", fontsize=11, color=INK, fontweight="bold" if kind == "ours" else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10); ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(0.82, 0.935); vgrid(ax); ax.set_xlabel("test balanced accuracy (official ISIC 2018 Task 3 test set)")
    ax.set_title("Our ensembles match the best published result using HAM10000 alone", loc="left")
    ax.text(0, -0.15, "Bars: 95% interval from a stratified bootstrap over the 1,512 test images (±0.025). Published entries report no interval.",
            transform=ax.transAxes, fontsize=10, color=INK2, va="top")
    save(fig, "fig04_sota")


def fig05_distillation():
    pairs = [("ConvNeXt-S", "convnext_s_res448_tta20_la", "convnext_s_kd_res448_tta20_la"),
             ("ResNet-50", "resnet50_ce_res448_tta20_la", "resnet50_kd_res448_tta20_la"),
             ("EfficientNetV2-S", "effv2_s_res448_tta20_la", "effv2_s_kd_res448_tta20_la"),
             ("DINOv2 ViT-B/14", "dinov2_b14_res448_tta20_la", "dinov2_b14_kd_res448_tta20_la"),
             ("ViT-B/16", "vit_b16_res448_tta20_la", "vit_b16_kd_res448_tta20_la")]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    for i, (name, hard, kd) in enumerate(pairs):
        h, k = A.per_fold(hard).bal_acc, A.per_fold(kd).bal_acc
        for hv, kv in zip(h, k):
            ax.plot([hv, kv], [i - 0.12, i + 0.12], color=GRID, lw=1.2, zorder=1)
        ax.scatter(h, [i - 0.12] * 5, s=28, color=MUTED, zorder=2)
        ax.scatter(k, [i + 0.12] * 5, s=28, color=ACCENT, alpha=0.55, zorder=2)
        ax.plot(h.mean(), i - 0.12, "o", ms=12, mfc="white", mec=INK, mew=2, zorder=3)
        ax.plot(k.mean(), i + 0.12, "o", ms=12, color=ACCENT, mec="white", mew=2, zorder=3)
        mu, _, better, n = A.paired(kd, hard)
        ax.text(0.905, i, f"+{mu:.3f}\n{better}/{n} folds", va="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(pairs)), [p[0] for p in pairs], fontsize=12); ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(0.78, 0.93); vgrid(ax); ax.set_xlabel("test balanced accuracy (one network = one dot; big dots = mean of 5 folds)")
    ax.invert_yaxis()
    ax.set_title("Out-of-fold ensemble distillation improves every network: 25 of 25 fold pairs", loc="left")
    ax.plot([], [], "o", ms=10, mfc="white", mec=INK, mew=2, label="trained on hard labels")
    ax.plot([], [], "o", ms=10, color=ACCENT, label="distilled from the ensemble's out-of-fold predictions")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.22), ncol=2, fontsize=10.5)
    save(fig, "fig05_distillation")


def fig06_effects():
    E = [  # (label, run, reference, adopted)
        ("Logit adjustment after training", "resnet50_ce_la", "resnet50_ce", True),
        ("Class-weighted loss", "resnet50_wce", "resnet50_ce", False),
        ("448 px input (vs 224)", "resnet50_ce_res448_la", "resnet50_ce_la", True),
        ("384 px input (vs 224)", "resnet50_ce_res384_la", "resnet50_ce_la", False),
        ("20-view test-time augmentation", "resnet50_ce_res448_tta20_la", "resnet50_ce_res448_la", True),
        ("60 epochs (vs 30)", "resnet50_ce_ep60_la", "resnet50_ce_la", False),
        ("TrivialAugment", "resnet50_ce_trivial_la", "resnet50_ce_la", False),
        ("ConvNeXt-S backbone (vs ResNet-50)", "convnext_s_res448_tta20_la", "resnet50_ce_res448_tta20_la", False),
        ("ViT-B/16 backbone (vs ResNet-50)", "vit_b16_res448_tta20_la", "resnet50_ce_res448_tta20_la", False),
        ("DINOv2 backbone (vs ResNet-50)", "dinov2_b14_res448_tta20_la", "resnet50_ce_res448_tta20_la", False),
        ("Average of 5 backbones (vs ResNet-50)", "ens5_res448_tta20_la", "resnet50_ce_res448_tta20_la", True),
        ("Distillation, ConvNeXt-S", "convnext_s_kd_res448_tta20_la", "convnext_s_res448_tta20_la", True),
        ("Distillation, ResNet-50", "resnet50_kd_res448_tta20_la", "resnet50_ce_res448_tta20_la", True),
        ("Distillation, DINOv2 ViT-B/14", "dinov2_b14_kd_res448_tta20_la", "dinov2_b14_res448_tta20_la", True),
        ("Distillation, ViT-B/16", "vit_b16_kd_res448_tta20_la", "vit_b16_res448_tta20_la", True),
        ("Test at 512 px (FixRes), ResNet-50", "resnet50_ce_res448_tta20_ts512_la", "resnet50_ce_res448_tta20_la", False),
        ("Colour constancy", "resnet50_cc_res448_tta20_la", "resnet50_ce_res448_tta20_la", False),
        ("Stronger colour augmentation", "resnet50_coloraug_res448_tta20_la", "resnet50_ce_res448_tta20_la", False),
    ]
    rows = [(lab, *A.paired(run, ref), adopted) for lab, run, ref, adopted in E]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11, 8.8))
    for i, (lab, mu, sd, better, n, adopted) in enumerate(rows):
        c = BLUE if adopted else ORANGE
        ax.plot([mu - sd, mu + sd], [i, i], color=c, lw=2.5, alpha=0.45, solid_capstyle="round")
        ax.plot(mu, i, "o", ms=9, color=c, mec="white", mew=1.5)
        ax.text(0.105, i, f"{mu:+.3f}  ({better}/{n})", va="center", fontsize=10.5, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=11); ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(-0.06, 0.135); vgrid(ax)
    ax.set_xlabel("change in test balanced accuracy, paired over the same 5 folds (mean ± SD; folds improved)")
    ax.set_title("What worked and what did not", loc="left")
    ax.plot([], [], "o", color=BLUE, ms=9, label="kept (chosen on CV / pre-registered)")
    ax.plot([], [], "o", color=ORANGE, ms=9, label="not kept")
    ax.legend(loc="upper left", fontsize=10.5)
    save(fig, "fig06_effects")


def fig07_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, (title, run) in zip(axes, [("Plain ResNet-50 baseline (5 fold models averaged)", BASELINE), ("Final model (40 networks averaged)", FINAL)]):
        _, y, p = all_models(run); pred = p.argmax(1)
        cm = np.zeros((7, 7), int)
        np.add.at(cm, (y, pred), 1)
        frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10.5, color="white" if frac[i, j] > 0.55 else INK)
        ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec=ACCENT, lw=2.5))
        ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False, ec=ACCENT, lw=2.5))
        ax.set_xticks(range(7), CLASSES); ax.set_yticks(range(7), CLASSES); ax.tick_params(colors=INK, length=0)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title}\nbalanced accuracy {bacc(y, pred):.3f} · melanoma recall {frac[0, 0]:.2f}", loc="left", fontsize=13)
        for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.01, -0.02, "Official test set (1,512 images); shading = share of each true class. Outlined: melanoma ↔ mole confusions.",
             fontsize=10.5, color=INK2)
    save(fig, "fig07_confusion")


def fig09_missed_melanomas():
    imgs, y, p = all_models(FINAL)
    te = load_split("test"); path = dict(zip(te.image, te.path))
    pred = p.argmax(1); mel = CLASSES.index("MEL")
    miss = np.where((y == mel) & (pred != mel))[0]
    miss = miss[np.argsort(p[miss, mel])][:12]
    fig, axes = plt.subplots(2, 6, figsize=(17, 6.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.06, wspace=0.06, hspace=0.32)
    for ax, i in zip(axes.ravel(), miss):
        ax.imshow(Image.open(path[imgs[i]]).convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"called {CLASSES[pred[i]]} ({p[i, pred[i]]:.0%})\nP(melanoma) {p[i, mel]:.0%}", fontsize=10.5, color=INK)
    for ax in axes.ravel()[len(miss):]:
        ax.axis("off")
    n_mel, n_miss = int((y == mel).sum()), int(((y == mel) & (pred != mel)).sum())
    fig.suptitle(f"Melanomas the final model missed: {n_miss} of {n_mel} (the 12 it was surest about)", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", y=1.0)
    by = pd.Series([CLASSES[k] for k in pred[(y == mel) & (pred != mel)]]).value_counts()
    fig.text(0.01, 0.0, "Missed melanomas were called: " + ", ".join(f"{c} {v}" for c, v in by.items()) +
             ". The test set can hold several photos of one lesion (e.g. the two 'VASC' images), so one hard lesion can count twice.",
             fontsize=10.5, color=INK2)
    save(fig, "fig09_missed_melanomas", pdf=False)


def fig08_gradcam():
    import torch
    from isic.data import baseline_transforms
    from isic.models import build_model

    run, k = "convnext_s_kd_res448", 0
    cfg = json.loads((A.RESULTS / run / f"fold{k}" / "metrics.json").read_text())["config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["model"], 7, cfg["img_size"], pretrained=False)
    model.load_state_dict(torch.load(A.RESULTS / run / f"fold{k}" / "best.pt", map_location="cpu")); model.eval().to(dev)
    layer = model.stages[-1]
    store = {}
    layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    _, eval_tf = baseline_transforms(cfg["img_size"], cfg["resize"])

    # examples chosen from the final model's test predictions: confident correct per class + characteristic errors
    imgs, y, p = all_models(FINAL); pred = p.argmax(1)
    te = load_split("test"); path = dict(zip(te.image, te.path))
    picks = []
    for c in ["MEL", "BCC", "VASC", "DF"]:
        ci = CLASSES.index(c); ok = np.where((y == ci) & (pred == ci))[0]
        picks.append((ok[np.argmax(p[ok, ci])], f"{c} correct"))
    for t, q, lab in [("MEL", "NV", "melanoma called mole"), ("NV", "MEL", "mole called melanoma")]:
        ti, qi = CLASSES.index(t), CLASSES.index(q); bad = np.where((y == ti) & (pred == qi))[0]
        if len(bad):
            picks.append((bad[np.argmax(p[bad, qi])], lab))

    fig, axes = plt.subplots(2, len(picks), figsize=(3.1 * len(picks), 6.6))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.07, wspace=0.06, hspace=0.18)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for j, (i, lab) in enumerate(picks):
        x = eval_tf(Image.open(path[imgs[i]]).convert("RGB")).unsqueeze(0).to(dev).requires_grad_(True)
        logits = model(x); cls = int(logits.argmax(1))
        model.zero_grad(); logits[0, cls].backward()
        a, g = store["act"][0], store["grad"][0]
        cam = torch.relu((g.mean((1, 2), keepdim=True) * a).sum(0)).detach().cpu().numpy()
        cam = np.array(Image.fromarray(cam / (cam.max() + 1e-8)).resize((x.shape[-1], x.shape[-2]), Image.BILINEAR))
        rgb = np.clip(x[0].detach().cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        axes[0, j].imshow(rgb); axes[1, j].imshow(rgb); axes[1, j].imshow(cam, cmap="magma", alpha=0.5)
        axes[0, j].set_title(f"{lab}\ntrue {CLASSES[y[i]]} · model {CLASSES[cls]}", fontsize=10.5, color=INK)
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
    axes[0, 0].set_ylabel("image", fontsize=11, color=INK2); axes[1, 0].set_ylabel("Grad-CAM", fontsize=11, color=INK2)
    fig.suptitle("Where the best single network looks (distilled ConvNeXt-S, fold 0)", x=0.01, ha="left", fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.01, 0.0, "Grad-CAM for the predicted class (last ConvNeXt stage; bright = raises that class's score). Correct cases focus on the lesion; "
             "on the missed melanoma the network looks at the background instead.", fontsize=10.5, color=INK2)
    save(fig, "fig08_gradcam", pdf=False)


FIGS = {"fig01": fig01_classes, "fig02": fig02_imbalance, "fig03": fig03_progress, "fig04": fig04_sota,
        "fig05": fig05_distillation, "fig06": fig06_effects, "fig07": fig07_confusion, "fig08": fig08_gradcam,
        "fig09": fig09_missed_melanomas}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--no-gradcam", action="store_true")
    a = ap.parse_args()
    for key, fn in FIGS.items():
        if (a.only and key not in a.only) or (a.no_gradcam and key == "fig08"):
            continue
        fn()
