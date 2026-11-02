#!/usr/bin/env python
"""Slide-ready figures for the Homework Y presentation, regenerated from results/ (run from hwy/).

    python scripts/make_figures.py                 # all figures -> figures/*.png (+ .pdf for charts)
    python scripts/make_figures.py --only fig01 fig07

Same palette and diagram style as the Homework X figures (../hwx/scripts/make_figures.py). Every number comes from saved
runs; scores are 5-fold means after TTA-20 + logit adjustment, EMA weights where the run name ends in _ema.
"""
import sys
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # hwy/, so the isic package imports
from isic import analysis as A  # noqa: E402
from isic.data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, apply_view, load_split  # noqa: E402

HWY = Path(__file__).resolve().parents[1]
OUT = HWY / "figures"
TV = HWY / "results" / "teacher_views" / "bb5_rect_v16"
R4 = HWY / "results" / "comparisons" / "hwy_round4.json"

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

BASE_RUN = "hwx/convnext_s_kd_res448_tta20_la"
R3_RUN, CONS_RUN, A09_RUN = "cnx_kd_r1_rect_ema_tta20_la", "cnx_kd_r4_cons_ema_tta20_la", "cnx_kd_r4_cons_a09_ema_tta20_la"
FIXED = ["cnx_kd_r1_rect_ema_tta20_la", "cnx_kd_r1_rect_s1_ema_tta20_la"]
CONS = ["cnx_kd_r4_cons_ema_tta20_la", "cnx_kd_r4_cons_s1_ema_tta20_la"]
TEACHER_TEST = 0.892   # HW X 5-backbone ensemble (25 networks), official test
TARGET = 0.88
# Every single network evaluated in HW Y: (label, round, run, decision)
ALL_RUNS = [("Drop path 0.2", "R1", "cnx_kd_r1_dp02_tta20_la", "adopted"),
            ("Full-frame 448 × 600", "R1", "cnx_kd_r1_rect_tta20_la", "adopted"),
            ("Layer-wise LR decay", "R1", "cnx_kd_r1_lld_tta20_la", "rejected"),
            ("Lesion-centred crop", "R1", "cnx_kd_r1_lesion_tta20_la", "rejected"),
            ("Metadata fusion", "R1", "cnx_kd_r1_meta_tta20_la", "rejected"),
            ("EMA (square input)", "R1", "cnx_kd_r1_base_ema_tta20_la", "rejected"),
            ("Full frame + drop path 0.1", "R2", "cnx_kd_r2_rect_dp01_tta20_la", "rejected"),
            ("Full frame + drop path 0.2", "R2", "cnx_kd_r2_rect_dp02_tta20_la", "rejected"),
            ("Full frame + drop path 0.3", "R2", "cnx_kd_r2_rect_dp03_tta20_la", "rejected"),
            ("Full frame + EMA", "R2", R3_RUN, "adopted"),
            ("Soup of 10 full-frame runs", "R3", "soup_rect10_tta20_la", "rejected"),
            ("Soup of 6 square runs", "R3", "soup_square6_tta20_la", "rejected"),
            ("Consistent distillation", "R4", CONS_RUN, "adopted"),
            ("Consistent distillation, α 0.9", "R4", A09_RUN, "adopted, within noise")]


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


def folds(run, split="official_test"):
    return A.per_fold(run, split).bal_acc.to_numpy()


def r4():
    return json.loads(R4.read_text())


def seed_rms(split="official_test"):
    """Typical change of a 5-fold mean when only the seed changes (RMS over the seed pairs)."""
    d = np.array(r4()["noise"]["pooled"][split]["seed_diffs"])
    return float(np.sqrt((d ** 2).mean()))


def fig01_ladder():
    steps = [("Baseline\n(end of HW X)", BASE_RUN, BLUE), ("+ full-frame\n448 × 600", "cnx_kd_r1_rect_tta20_la", BLUE),
             ("+ weight\nEMA", R3_RUN, BLUE), ("+ consistent\ndistillation", CONS_RUN, ACCENT),
             ("+ teacher weight\nα 0.9 (rule's pick)", A09_RUN, BASE)]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(steps))
    b = folds(BASE_RUN).mean()
    ax.axhspan(b - seed_rms(), b + seed_rms(), color=MUTED, alpha=0.13, lw=0, zorder=0)
    for xi, (lab, run, c) in zip(x, steps):
        f = folds(run)
        ax.bar(xi, f.mean(), color=c, width=0.62, zorder=1, hatch="//" if c == BASE else None, edgecolor="white" if c != BASE else MUTED, lw=0)
        ax.scatter(np.full(5, xi) + np.linspace(-0.16, 0.16, 5), f, s=16, color="white", edgecolor=INK2, lw=0.8, zorder=3)
        ax.text(xi, 0.824, f"{f.mean():.3f}", ha="center", va="bottom", fontsize=14, fontweight="bold", color="white" if c != BASE else INK)
    for lab, v, ls in ((f"Target · {TARGET:.2f}", TARGET, "--"), (f"HW X teacher ensemble (25 networks) · {TEACHER_TEST:.3f}", TEACHER_TEST, ":")):
        ax.axhline(v, color=MUTED, lw=1.3, ls=ls, zorder=0)
        ax.text(len(steps) - 0.45, v + 0.0006, lab, va="bottom", ha="right", fontsize=10, color=MUTED)
    ax.set_xticks(x, [s[0] for s in steps], fontsize=11); ax.tick_params(axis="x", colors=INK, length=0)
    ax.set_ylim(0.82, 0.90); ax.set_ylabel("test balanced accuracy"); hgrid(ax)
    ax.set_title("One network, step by step: 0.864 → 0.870 on the official test set", loc="left")
    ax.text(0, -0.2, "Bars: one ConvNeXt-S (mean of 5 fold models, dots). Each step was adopted by the pre-registered CV rule.\n"
            "Rose: the one change that beat its control on CV and test with 2 seeds per arm. Hatched: adopted by the rule, worse on test.\n"
            f"Grey band: baseline ± {seed_rms():.3f}, the typical change from re-running with another seed.",
            transform=ax.transAxes, fontsize=10, color=INK2, va="top")
    save(fig, "fig01_ladder")


def fig02_all_changes():
    base_cv, base_te = folds(BASE_RUN, "cv_val"), folds(BASE_RUN)
    rows = [(f"{lab}  ({rnd})", folds(run, "cv_val") - base_cv, folds(run) - base_te, dec) for lab, rnd, run, dec in ALL_RUNS][::-1]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    band = seed_rms()
    ax.axvspan(-band, band, color=MUTED, alpha=0.13, lw=0)
    for i, (lab, dcv, dte, dec) in enumerate(rows):
        c = ACCENT if lab.startswith("Consistent distillation  ") else BLUE if dec == "adopted" else MUTED if "noise" in dec else ORANGE
        ax.plot([dte.mean() - dte.std(), dte.mean() + dte.std()], [i, i], color=c, lw=2.5, alpha=0.45, solid_capstyle="round")
        ax.plot(dte.mean(), i, "o", ms=9, color=c, mec="white", mew=1.5, zorder=3)
        ax.plot(dcv.mean(), i, "o", ms=7, mfc="white", mec=c, mew=1.6, zorder=2)
        ax.text(0.048, i, f"test {dte.mean():+.3f}   CV {dcv.mean():+.3f}", va="center", fontsize=10.5, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=11); ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(-0.03, 0.045); ax.set_xlabel("change in balanced accuracy vs the HW X baseline (paired over the same 5 folds)"); vgrid(ax)
    ax.set_title("13 changes, one network each: almost all inside the seed noise", loc="left", pad=12)
    ax.plot([], [], "o", color=ACCENT, ms=9, label="adopted, beats its control on CV and test")
    ax.plot([], [], "o", color=BLUE, ms=9, label="adopted by the CV rule")
    ax.plot([], [], "o", color=MUTED, ms=9, label="adopted by the rule, worse on test")
    ax.plot([], [], "o", color=ORANGE, ms=9, label="not adopted")
    ax.plot([], [], "o", ms=7, mfc="white", mec=INK2, label="CV (hollow)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=10.5)
    ax.text(-band + 0.0005, -0.75, f"grey band ± {band:.3f}: typical change from the seed alone", fontsize=9.5, color=MUTED, va="bottom")
    ax.set_ylim(-0.9, len(rows) - 0.5)
    save(fig, "fig02_all_changes")


def fig03_consistent_vs_fixed():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, split, name in ((axes[0], "cv_val", "Cross-validation (decides)"), (axes[1], "official_test", "Official test (reported only)")):
        fx, cs = np.array([folds(r, split) for r in FIXED]), np.array([folds(r, split) for r in CONS])
        for k in range(5):
            ax.plot([fx[:, k].mean(), cs[:, k].mean()], [k, k], color=GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.scatter(fx[:, k], [k - 0.12] * 2, s=26, color=MUTED, zorder=2)
            ax.scatter(cs[:, k], [k + 0.12] * 2, s=26, color=ACCENT, alpha=0.55, zorder=2)
            ax.plot(fx[:, k].mean(), k - 0.12, "o", ms=11, color=MUTED, mec="white", mew=2, zorder=3)
            ax.plot(cs[:, k].mean(), k + 0.12, "o", ms=11, color=ACCENT, mec="white", mew=2, zorder=3)
        d = cs.mean(0) - fx.mean(0)
        ax.set_title(f"{name}\n{d.mean():+.3f} on average, {(d > 0).sum()}/5 folds better", loc="left", fontsize=13)
        vgrid(ax); ax.tick_params(axis="y", colors=INK, length=0)
    axes[0].set_yticks(range(5), [f"fold {k}" for k in range(5)], fontsize=12); axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlabel("balanced accuracy")
    axes[0].plot([], [], "o", ms=10, color=MUTED, label="fixed soft label per image")
    axes[0].plot([], [], "o", ms=10, color=ACCENT, label="consistent: a soft label per view")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, fontsize=11)
    fig.text(0.11, -0.16, "Small dots: seeds 0 and 1; big dots: their mean. Same folds, same recipe; only the distillation targets change.",
             ha="left", fontsize=10, color=INK2)
    fig.suptitle("Consistent distillation beats fixed labels with 2 seeds per arm", x=0.01, ha="left", fontsize=15, fontweight="bold", y=1.06)
    save(fig, "fig03_consistent_vs_fixed")


def fig04_seed_noise():
    p = r4()["noise"]["pooled"]
    prim = r4()["primary"]
    rows = [("Same recipe, seed 1 − seed 0:\nfixed labels", p["cv_val"]["seed_diffs"][0], p["official_test"]["seed_diffs"][0], MUTED),
            ("Same recipe, seed 1 − seed 0:\nconsistent labels", p["cv_val"]["seed_diffs"][1], p["official_test"]["seed_diffs"][1], MUTED),
            ("Same recipe, seed 1 − seed 0:\nRound 2 drop path 0.2", p["cv_val"]["seed_diffs"][2], p["official_test"]["seed_diffs"][2], MUTED),
            ("Consistent − fixed labels\n(2 seeds each)", prim["cv_val"]["diff"], prim["official_test"]["diff"], ACCENT)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(rows))
    for i, (lab, cv, te, c) in enumerate(rows):
        ax.barh(i - 0.18, cv, height=0.32, color="white", edgecolor=c, lw=2)
        ax.barh(i + 0.18, te, height=0.32, color=c)
        for v, dy in ((cv, -0.18), (te, 0.18)):
            ax.text(v + (0.0008 if v >= 0 else -0.0008), i + dy, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=10.5, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=11); ax.tick_params(axis="y", colors=INK, length=0); ax.invert_yaxis()
    ax.set_xlim(-0.024, 0.024); ax.set_xlabel("change in the 5-fold mean balanced accuracy"); vgrid(ax)
    ax.set_title("A different random seed alone moves the score as much as most changes", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="white", ec=INK2, lw=2, label="CV"), Patch(fc=INK2, label="test")], loc="lower left", ncol=2, fontsize=11)
    ax.text(0, -0.2, f"Pooled over three seed pairs, one run's 5-fold mean has SD {p['cv_val']['single_run_sd']:.3f} (CV) and "
            f"{p['official_test']['single_run_sd']:.3f} (test).\nThe Round 3 model (fixed labels, seed 0) was a lucky seed.",
            transform=ax.transAxes, fontsize=10, color=INK2, va="top")
    save(fig, "fig04_seed_noise")


def fig05_cv_vs_test():
    pts = [(lab, folds(run, "cv_val").mean(), folds(run).mean(), dec) for lab, _, run, dec in ALL_RUNS]
    pts += [("Baseline (HW X)", folds(BASE_RUN, "cv_val").mean(), folds(BASE_RUN).mean(), "base"),
            ("Full frame + EMA, seed 1", folds(FIXED[1], "cv_val").mean(), folds(FIXED[1]).mean(), "seed"),
            ("Consistent distillation, seed 1", folds(CONS[1], "cv_val").mean(), folds(CONS[1]).mean(), "seed")]
    gaps = r4()["cv_test_gap"]
    gap = np.mean([gaps["cnx_kd_r1_rect_ema"], gaps["cnx_kd_r1_rect_s1_ema"]])
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.linspace(0.875, 0.92, 2)
    ax.plot(xs, xs - gap, color=MUTED, lw=1.2, ls="--")
    ax.text(0.8995, 0.8995 - gap + 0.0009, f"test = CV − {gap:.3f}", fontsize=10, color=MUTED, rotation=37, rotation_mode="anchor")
    color = lambda lab, dec: ORANGE if "α" in lab else ACCENT if lab.startswith("Consistent") else INK2 if dec == "base" else BLUE if dec == "adopted" else MUTED
    for lab, cv, te, dec in pts:
        c = color(lab, dec)
        ax.plot(cv, te, "o", ms=10 if c in (ACCENT, ORANGE, INK2) else 7, color=c, mec="white", mew=1.5, zorder=3)
    for lab, txt, dx, dy, ha in (("Consistent distillation, α 0.9", "consistent, α 0.9", 0, -0.0028, "center"),
                                 ("Consistent distillation", "consistent, seed 0", 0, 0.0016, "center"),
                                 ("Consistent distillation, seed 1", "consistent, seed 1", 0.0007, 0, "left"),
                                 ("Full frame + EMA, seed 1", "fixed labels, seed 1", 0.0006, -0.0003, "left"),
                                 ("Full frame + EMA", "fixed labels, seed 0\n(Round 3 model)", 0.0006, 0.0006, "left"),
                                 ("Baseline (HW X)", "baseline", 0.0006, -0.0012, "left")):
        cv, te = next((c, t) for l, c, t, _ in pts if l == lab)
        ax.text(cv + dx, te + dy, txt, fontsize=10.5, color=INK, ha=ha, va="center")
    for c, lab in ((INK2, "baseline"), (BLUE, "adopted in Rounds 1–2"), (MUTED, "not adopted / seed replicate"),
                   (ACCENT, "consistent distillation, α 0.5"), (ORANGE, "consistent, α 0.9")):
        ax.plot([], [], "o", ms=9, color=c, label=lab)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.set_xlabel("cross-validation balanced accuracy"); ax.set_ylabel("test balanced accuracy"); hgrid(ax); vgrid(ax)
    ax.set_xlim(0.875, 0.92); ax.set_ylim(0.845, 0.885)
    ax.set_title("Leaning harder on the teacher raises CV, not test", loc="left")
    ax.text(0, -0.13, "Each dot is one network (5-fold mean); dashed line: the CV − test gap of the fixed-label runs. The teacher was\n"
            f"trained on the student's CV images, so copying it more (α 0.9) flatters CV: its CV − test gap is "
            f"{gaps['cnx_kd_r4_cons_a09_ema']:.3f}, twice the {gap:.3f} of the fixed-label runs.",
            transform=ax.transAxes, fontsize=10, color=INK2, va="top")
    save(fig, "fig05_cv_vs_test")


def fig06_recall():
    cols = [("Baseline (HW X)", [BASE_RUN], BASE), ("Fixed labels, 2 seeds", FIXED, BLUE), ("Consistent, 2 seeds", CONS, ACCENT), ("Consistent, α 0.9", [A09_RUN], ORANGE)]
    rec = {name: np.mean([np.array([A.per_class_recall(r)[c] for c in CLASSES]) for r in runs], 0) for name, runs, _ in cols}
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(CLASSES)); w = 0.2
    for j, (name, _, c) in enumerate(cols):
        ax.bar(x + (j - 1.5) * w, rec[name], width=w * 0.92, color=c, label=name)
    ax.set_xticks(x, CLASSES, fontsize=12); ax.tick_params(axis="x", colors=INK, length=0)
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("test recall (5-fold mean)"); hgrid(ax)
    ax.legend(loc="upper left", ncol=4, fontsize=10.5, bbox_to_anchor=(0, 1.02))
    ax.set_title("Where the gain comes from: DF, VASC and AKIEC, not melanoma", loc="left", pad=28)
    ax.text(0, -0.13, "Test set: 171 MEL, 909 NV, 93 BCC, 43 AKIEC, 217 BKL, 44 DF, 35 VASC, so one DF or VASC image moves its recall by 2–3 points.",
            transform=ax.transAxes, fontsize=10, color=INK2, va="top")
    save(fig, "fig06_recall")


def fig07_views():
    df = pd.read_csv(TV / "images.csv"); params = np.load(TV / "params.npy"); soft = np.load(TV / "soft_views.npy")
    fixed = pd.read_csv(HWY / "results/hwx/soft_labels/bb5_calib_oof.csv").set_index("image").loc[df.image, CLASSES].to_numpy()
    y = df.label.to_numpy(); p_true = soft[np.arange(len(df)), :, y]
    i = int(np.argmax(p_true.std(1) * (y == 0)))   # the melanoma whose views disagree most
    img = Image.open(load_split("train").set_index("image").path[df.image[i]]).convert("RGB")
    order = np.argsort(p_true[i])
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.9), gridspec_kw={"hspace": 0.28})
    for ax, v in zip(axes.flat, order[[0, 1, 2, 3, -4, -3, -2, -1]]):
        x = apply_view(img, params[i, v], (448, 600)).permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        ax.imshow(x.clip(0, 1)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(ACCENT if p_true[i, v] > 0.5 else BASE); s.set_linewidth(2.5)
        ax.set_title(f"teacher: P(melanoma) = {p_true[i, v]:.2f}", fontsize=12, color=INK)
    fig.suptitle(f"One melanoma, 8 of its 16 stored training views. The fixed label says {fixed[i, 0]:.2f} for every crop;\n"
                 f"the per-view labels range from {p_true[i].min():.2f} (dark border cropped out) to {p_true[i].max():.2f} (full lesion).",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.04)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.08)
    save(fig, "fig07_views", pdf=False)


# ---------------------------------------------------------------------------------------------
# Architecture diagrams: the pretrained backbone is one grey block; what Homework Y added is highlighted.
KIND = {"data": ("#FDFCF9", BASE), "op": ("#E8F0FB", BLUE), "backbone": ("#E9EAEC", "#9AA0A8"), "ours": ("#F3E3EA", ACCENT)}


def _box(ax, x, y, w, h, title, sub=None, kind="op", fs=13):
    from matplotlib.patches import FancyBboxPatch
    fill, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=edge, lw=2 if kind == "ours" else 1.4))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK, wrap=True)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=fs - 2.5, color=INK2, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK, linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2, label=None, color=INK2, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=18, linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", va="bottom", fontsize=11, color=INK2)


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def _legend(ax, x, y, ours="what Homework Y added or changed"):
    from matplotlib.patches import Patch
    items = [("backbone", "pretrained backbone (internals not shown)"), ("ours", ours), ("op", "fixed operation"), ("data", "data")]
    ax.legend(handles=[Patch(fc=KIND[k][0], ec=KIND[k][1], label=l) for k, l in items], loc="lower left",
              bbox_to_anchor=(x, y), bbox_transform=ax.transData, ncol=4, fontsize=11, handlelength=1.6)


def _row(ax, boxes, y, h, fs=12.5):
    for x, w, t, s, k in boxes:
        _box(ax, x, y, w, h, t, s, k, fs=fs)
    for (x1, w1, *_), (x2, *_r) in zip(boxes, boxes[1:]):
        _arrow(ax, x1 + w1 + 0.05, y + h / 2, x2 - 0.05, y + h / 2)


def fig08_arch_consistent():
    fig, ax = _canvas(16, 8.6)
    ax.text(0.2, 8.2, "Fixed vs consistent distillation targets", fontsize=16, fontweight="bold", color=INK)
    h = 1.7
    ax.text(0.2, 7.55, "ROUNDS 1–3: ONE FIXED LABEL PER IMAGE", fontsize=12, fontweight="bold", color=MUTED)
    _row(ax, [(0.2, 1.9, "Training\nimage", "600 × 450", "data"),
              (2.5, 3.1, "Teacher, once", "25 out-of-fold models\nTTA-20 on the whole image", "op"),
              (6.0, 2.75, "One soft label q", "e.g. P(MEL) = 0.66\nfor every epoch", "op"),
              (9.15, 2.95, "Random crop\neach epoch", "may cut off the\nlesion border", "op"),
              (12.5, 3.35, "Student loss", "0.5 · CE(z, y) + 0.5 · CE(z, q)\nq ignores what the crop shows", "op")], 5.5, h)
    ax.text(0.2, 4.55, "ROUND 4: ONE LABEL PER VIEW (CONSISTENT)", fontsize=12, fontweight="bold", color=ACCENT)
    _row(ax, [(0.2, 1.9, "Training\nimage", "600 × 450", "data"),
              (2.5, 3.1, "16 stored views", "drawn once: crop 0.4–1 of area,\nflips, colour jitter → 448 × 600", "ours"),
              (6.0, 2.75, "Teacher, per view", "same 25 out-of-fold\nmodels, calibrated", "ours"),
              (9.15, 2.95, "16 soft labels q_v", "one melanoma:\nP(MEL) 0.07 … 0.87", "ours"),
              (12.5, 3.35, "Student loss", "each epoch: one view v\n0.5 · CE(z_v, y) + 0.5 · CE(z_v, q_v)", "ours")], 2.5, h)
    ax.text(0.2, 1.55, "Cost: one teacher pass over 160,240 views (about 25 GPU-minutes); student training time is unchanged.\n"
            "Idea from Beyer et al., CVPR 2022, \"Knowledge distillation: a good teacher is patient and consistent\".",
            fontsize=11.5, color=INK2, va="top", linespacing=1.5)
    _legend(ax, 0.2, 0.05, ours="new in Round 4")
    save(fig, "fig08_arch_consistent")


def fig09_arch_final():
    fig, ax = _canvas(16, 9.4)
    ax.text(0.2, 9.0, "Our HW Y single model: ConvNeXt-S with consistent distillation", fontsize=16, fontweight="bold", color=INK)
    ax.text(0.2, 8.45, "TRAINING", fontsize=12, fontweight="bold", color=ACCENT)
    y, h = 5.15, 1.9
    _row(ax, [(0.2, 1.75, "Training\nimage", "600 × 450", "data"),
              (2.25, 2.25, "Stored view v", "1 of 16 per image\n448 × 600 (full frame)", "ours"),
              (4.8, 2.6, "ConvNeXt-S\nbackbone", "ImageNet-22k, 49.5M\nall layers fine-tuned", "backbone"),
              (7.7, 1.95, "Pool +\nLayerNorm", "768-d feature", "op"),
              (9.95, 1.8, "Head", "Linear 768 → 7\n→ logits z", "op"),
              (12.05, 3.75, "Distillation loss", "0.5 · CE(z, y) + 0.5 · CE(z, q_v)\nweights tracked by EMA 0.999", "ours")], y, h)
    _box(ax, 4.8, 7.35, 6.6, 1.05, "Teacher on the same view (5 backbones × 5 folds)",
         "calibrated average from the models that never saw this image", "ours", fs=12)
    _box(ax, 12.05, 7.45, 1.8, 0.85, "soft label q_v", None, "ours", fs=12)
    _box(ax, 14.05, 7.45, 1.75, 0.85, "true label y", None, "data", fs=12)
    _arrow(ax, 11.45, 7.88, 12.0, 7.88)
    _arrow(ax, 12.95, 7.4, 12.95, y + h + 0.05)
    _arrow(ax, 14.92, 7.4, 14.92, y + h + 0.05)
    tau = A.tau(CONS_RUN)["selected_tau"]
    ax.text(0.2, 4.35, "PREDICTION", fontsize=12, fontweight="bold", color=BLUE)
    _row(ax, [(0.2, 1.75, "Test image", "600 × 450", "data"),
              (2.25, 2.25, "20 views", "5 crops × 4 flips\n448 × 600 each", "ours"),
              (4.8, 2.6, "EMA network", "ConvNeXt-S + head\n(averaged weights)", "backbone"),
              (7.7, 1.95, "Average", "softmax over\nthe 20 views", "op"),
              (9.95, 3.6, "Logit adjustment", f"log p − τ · log π_class\nτ = {tau:.1f}, chosen on CV", "op"),
              (13.85, 2.0, "Diagnosis", "argmax of 7", "op")], 1.6, h)
    _legend(ax, 0.2, 0.2)
    save(fig, "fig09_arch_final")


FIGS = {"fig01": fig01_ladder, "fig02": fig02_all_changes, "fig03": fig03_consistent_vs_fixed, "fig04": fig04_seed_noise,
        "fig05": fig05_cv_vs_test, "fig06": fig06_recall, "fig07": fig07_views, "fig08": fig08_arch_consistent, "fig09": fig09_arch_final}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    for key, fn in FIGS.items():
        if not a.only or key in a.only:
            fn()
