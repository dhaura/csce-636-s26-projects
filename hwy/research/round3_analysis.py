"""Round 3 analysis (model soups), as pre-registered in research/plan.md (run from hwy/: python research/round3_analysis.py).

Rule: a soup is adopted if its paired CV gain over the Round-2 model (cnx_kd_r1_rect_ema, TTA-20 + logit adjustment) is
> 0 with >= 3/5 folds better; if both soups pass, the higher CV mean wins. Test is reported only. For context (not for
the decision) each soup is also compared with its own ingredients and with the prediction ensemble of the same
ingredients (10x / 6x the inference cost of the soup).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HWY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HWY_DIR))
from isic import analysis as A
from isic.data import CLASSES

pd.set_option("display.width", 220)
BASE, R2 = "hwx/convnext_s_kd_res448_tta20_la", "cnx_kd_r1_rect_ema_tta20_la"
SOUPS = {
    "soup_rect10": ["cnx_kd_r1_rect", "cnx_kd_r2_rect_dp01", "cnx_kd_r2_rect_dp02", "cnx_kd_r2_rect_dp03", "cnx_kd_r2_rect_dp02_s1"],
    "soup_square6": ["cnx_kd_r1_base", "cnx_kd_r1_dp02", "cnx_kd_r1_lld"],
}
ENS = {"soup_rect10": "ens_rect10_tta20_la", "soup_square6": "ens_square6_tta20_la"}
la = lambda r: f"{r}_tta20_la"


def bootstrap_ci(run, n=2000, seed=0):
    """95% CI of the fold-mean test balanced accuracy, resampling test images within each class."""
    ps = [pd.read_csv(A.RESULTS / run / f"fold{k}" / "pred_official_test.csv") for k in A.folds(run)]
    y = ps[0]["label"].map(CLASSES.index).to_numpy()
    preds = np.stack([p[CLASSES].to_numpy().argmax(1) for p in ps])
    rng = np.random.default_rng(seed)
    idx_c = [np.where(y == c)[0] for c in range(len(CLASSES))]
    stats = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ic, len(ic)) for ic in idx_c])
        yy = y[idx]
        stats.append(np.mean([np.mean([(pr[idx][yy == c] == c).mean() for c in range(len(CLASSES))]) for pr in preds]))
    return np.percentile(stats, [2.5, 97.5])


def main():
    rows, decisions = [], {}
    for name, run in [("baseline (HW X)", BASE), ("Round-2 model: full frame + EMA", R2)]:
        rows.append({"model": name, "run": run, "cv": A.per_fold(run, "cv_val").bal_acc.mean(), "test": A.per_fold(run).bal_acc.mean()})
    passed = {}
    for soup, members in SOUPS.items():
        run = la(soup)
        mu, sd, b, n = A.paired(run, R2, "cv_val")
        tmu, _, tb, _ = A.paired(run, R2)
        bmu, _, bb, _ = A.paired(run, BASE, "cv_val")
        tbmu, _, tbb, _ = A.paired(run, BASE)
        ing = [la(m) for m in members] + [la(f"{m}_ema") for m in members]
        ing_cv = [A.per_fold(r, "cv_val").bal_acc.mean() for r in ing]
        ing_te = [A.per_fold(r).bal_acc.mean() for r in ing]
        rows.append({"model": soup, "run": run, "cv": A.per_fold(run, "cv_val").bal_acc.mean(), "test": A.per_fold(run).bal_acc.mean(),
                     "test_auc": A.per_fold(run).auc.mean(), "test_mel_sens": A.per_fold(run).mel_sens.mean(),
                     "cv vs R2": f"{mu:+.4f} ({b}/{n})", "test vs R2": f"{tmu:+.4f} ({tb}/{n})",
                     "cv vs baseline": f"{bmu:+.4f} ({bb}/{n})", "test vs baseline": f"{tbmu:+.4f} ({tbb}/{n})",
                     "ingredients cv mean / best": f"{np.mean(ing_cv):.4f} / {max(ing_cv):.4f}",
                     "ingredients test mean / best": f"{np.mean(ing_te):.4f} / {max(ing_te):.4f}",
                     f"ensemble of ingredients ({len(ing)}x cost) cv / test": f"{A.per_fold(ENS[soup], 'cv_val').bal_acc.mean():.4f} / {A.per_fold(ENS[soup]).bal_acc.mean():.4f}",
                     **A.per_source(run).add_prefix("test ").to_dict()})
        if mu > 0 and b >= 3:
            passed[soup] = A.per_fold(run, "cv_val").bal_acc.mean()
    t = pd.DataFrame(rows).set_index("model")
    print(t.drop(columns="run").T.to_string(), "\n")
    for soup in SOUPS:
        print(f"{soup}: {'passes' if soup in passed else 'fails'} the rule against the Round-2 model")
    chosen = max(passed, key=passed.get) if passed else None
    final = la(chosen) if chosen else R2
    lo, hi = bootstrap_ci(final)
    print(f"\nRound-3 model: {final} | CV {A.per_fold(final, 'cv_val').bal_acc.mean():.4f} | test {A.per_fold(final).bal_acc.mean():.4f} "
          f"(95% CI {lo:.3f}-{hi:.3f}) | baseline test {A.per_fold(BASE).bal_acc.mean():.4f}")
    print("per-class recall (test):"); print(pd.DataFrame({"baseline": A.per_class_recall(BASE), "final": A.per_class_recall(final)}).round(3).T.to_string())
    out = HWY_DIR / "results" / "comparisons" / "hwy_round3.json"
    out.write_text(json.dumps({"final": final, "chosen_soup": chosen, "ci": [lo, hi], "rows": t.reset_index().to_dict("records")}, indent=2, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
