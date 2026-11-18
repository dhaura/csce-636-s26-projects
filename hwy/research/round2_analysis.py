"""Round 2 analysis, as pre-registered in research/plan.md (run from hwy/: python research/round2_analysis.py).

1. Drop path rate: pick 0.1 / 0.2 / 0.3 (full frame, seed 0) by CV mean (TTA-20 + logit adjustment).
2. The chosen run must beat cnx_kd_r1_rect (best Round-1 single factor) by the Round-1 rule, else keep cnx_kd_r1_rect.
3. EMA: cross-fitted CV score (epoch picked on one class-stratified half of each CV fold, scored on the other half);
   adopt if EMA beats the raw weights in >= 3 of the 4 Round-2 runs with a positive mean.
4. Noise floor: seed 1 minus seed 0 of the combination, paired by fold.
Test numbers are reported only.
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
BASE, R1_BEST = "hwx/convnext_s_kd_res448_tta20_la", "cnx_kd_r1_rect_tta20_la"
DP = {0.1: "cnx_kd_r2_rect_dp01", 0.2: "cnx_kd_r2_rect_dp02", 0.3: "cnx_kd_r2_rect_dp03"}
SEED1 = "cnx_kd_r2_rect_dp02_s1"
la = lambda r: f"{r}_tta20_la"


def bacc(y, pred):
    return np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])


def cross_fitted(run, seed=0):
    """Selection-free CV score per fold: pick the epoch on one stratified half, score that epoch on the other half."""
    out = []
    for k in A.folds(run):
        d = A.RESULTS / run / f"fold{k}"
        P = np.load(d / "cv_val_epoch_probs.npy").astype(np.float32)          # (epochs, images, classes)
        y = pd.read_csv(d / "pred_cv_val.csv")["label"].map(CLASSES.index).to_numpy()
        rng = np.random.default_rng(seed + k)
        half = np.zeros(len(y), bool)
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            half[idx[: len(idx) // 2]] = True
        pred = np.zeros(len(y), int)
        for sel, ev in ((half, ~half), (~half, half)):
            e = int(np.argmax([bacc(y[sel], P[t, sel].argmax(1)) for t in range(len(P))]))
            pred[ev] = P[e, ev].argmax(1)
        out.append(bacc(y, pred))
    return np.array(out)


def row(name, run, ref):
    cv, te = A.per_fold(run, "cv_val"), A.per_fold(run)
    mu, _, b, n = A.paired(run, ref, "cv_val")
    tmu, _, tb, _ = A.paired(run, ref)
    return {"run": name, "cv": cv.bal_acc.mean(), "cv_std": cv.bal_acc.std(ddof=0), "test": te.bal_acc.mean(),
            "test_std": te.bal_acc.std(ddof=0), "test_auc": te.auc.mean(), "test_mel_sens": te.mel_sens.mean(),
            "ref": ref, "cv_paired": mu, "cv_better": f"{b}/{n}", "test_paired": tmu, "test_better": f"{tb}/{n}",
            **A.per_source(run).add_prefix("test ").to_dict()}


def main():
    out = {"decisions": {}}
    rows = [row("baseline (HW X)", BASE, BASE), row("R1 full frame (best R1 single factor)", R1_BEST, BASE),
            row("R1 drop path 0.2", la("cnx_kd_r1_dp02"), BASE)]
    rows += [row(f"R2 full frame + drop path {p}", la(r), R1_BEST) for p, r in DP.items()]
    rows += [row("R2 full frame + drop path 0.2, seed 1", la(SEED1), R1_BEST)]
    t = pd.DataFrame(rows)
    print(t.drop(columns="ref").round(4).to_string(index=False), "\n(paired columns: R1 rows vs the baseline, R2 rows vs R1 full frame)\n")

    # 1. drop path rate by CV mean
    cvs = {p: A.per_fold(la(r), "cv_val").bal_acc.mean() for p, r in DP.items()}
    p_best = max(cvs, key=cvs.get)
    print("1. drop path rate by CV mean:", {p: round(v, 4) for p, v in cvs.items()}, "-> choose", p_best)
    # 2. chosen vs best Round-1 single factor
    mu, _, b, n = A.paired(la(DP[p_best]), R1_BEST, "cv_val")
    keep = mu > 0 and b >= 3
    chosen = DP[p_best] if keep else "cnx_kd_r1_rect"
    print(f"2. {DP[p_best]} vs cnx_kd_r1_rect: CV paired {mu:+.4f} ({b}/{n}) -> {'adopt the combination' if keep else 'keep cnx_kd_r1_rect'}")
    out["decisions"].update(drop_path_rate=p_best, combination_adopted=bool(keep), chosen=chosen)

    # 3. EMA with the cross-fitted CV score
    ema = []
    for r in [*DP.values(), SEED1]:
        raw, em = cross_fitted(r), cross_fitted(f"{r}_ema")
        tmu, _, tb, _ = A.paired(la(f"{r}_ema"), la(r))
        mu, _, b, _ = A.paired(la(f"{r}_ema"), la(r), "cv_val")
        ema.append({"run": r, "cross-fitted raw": raw.mean(), "cross-fitted EMA": em.mean(), "cross-fitted EMA-raw": (em - raw).mean(),
                    "folds": f"{int(((em - raw) > 0).sum())}/5", "best-epoch CV EMA-raw (TTA+LA)": mu, "best-epoch folds": f"{b}/5",
                    "test EMA-raw": tmu, "test folds": f"{tb}/5"})
    ema = pd.DataFrame(ema)
    print("\n3. EMA vs raw weights")
    print(ema.round(4).to_string(index=False))
    wins = int((ema["cross-fitted EMA-raw"] > 0).sum())
    adopt_ema = wins >= 3 and ema["cross-fitted EMA-raw"].mean() > 0
    print(f"   EMA wins the cross-fitted score in {wins}/4 runs, mean {ema['cross-fitted EMA-raw'].mean():+.4f} -> {'ADOPT' if adopt_ema else 'reject'}")
    out["decisions"]["EMA"] = bool(adopt_ema)

    # 4. noise floor
    mu, sd, b, n = A.paired(la(SEED1), la(DP[0.2]), "cv_val")
    tmu, tsd, tb, _ = A.paired(la(SEED1), la(DP[0.2]))
    print(f"\n4. noise floor (seed 1 - seed 0, drop path 0.2 + full frame): CV {mu:+.4f} (per-fold sd {sd:.4f}), "
          f"test {tmu:+.4f} (per-fold sd {tsd:.4f})")
    out["noise"] = {"cv_mean_diff": mu, "cv_fold_sd": sd, "test_mean_diff": tmu, "test_fold_sd": tsd}

    final = la(f"{chosen}_ema") if adopt_ema else la(chosen)
    print(f"\nRound-2 model: {final} | CV {A.per_fold(final, 'cv_val').bal_acc.mean():.4f} | test {A.per_fold(final).bal_acc.mean():.4f}")
    out.update(final=final, rows=rows, ema=ema.to_dict("records"))
    f = HWY_DIR / "results" / "comparisons" / "hwy_round2.json"
    f.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", f)


if __name__ == "__main__":
    main()
