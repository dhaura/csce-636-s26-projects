"""Round 4 analysis (consistent distillation), as pre-registered in research/plan.md (run from hwy/: python research/round4_analysis.py).

1. Primary: consistent (per-view) vs fixed teacher targets, 2 seeds per arm. Per fold, the two seeds of each arm are
   averaged; adopt consistent targets if the paired CV gain is > 0 with >= 3/5 folds better.
2. Secondary: alpha 0.9 vs alpha 0.5 with consistent targets (seed 0), only if rule 1 adopts them.
3. Noise: single-seed gains must exceed 0.007 CV to be called improvements; the 2-seed comparison 0.005.
All scores: EMA weights, TTA-20 + logit adjustment. Test is reported only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HWY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HWY_DIR)); sys.path.insert(0, str(HWY_DIR / "research"))
from isic import analysis as A
from round2_analysis import cross_fitted
from round3_analysis import bootstrap_ci

pd.set_option("display.width", 220)
BASE, R3 = "hwx/convnext_s_kd_res448_tta20_la", "cnx_kd_r1_rect_ema_tta20_la"
FIXED = ["cnx_kd_r1_rect_ema", "cnx_kd_r1_rect_s1_ema"]           # control arm, seeds 0 and 1
CONS = ["cnx_kd_r4_cons_ema", "cnx_kd_r4_cons_s1_ema"]            # consistent distillation, seeds 0 and 1
A09 = "cnx_kd_r4_cons_a09_ema"
TV = HWY_DIR / "results" / "teacher_views" / "bb5_rect_v16" / "summary.json"
la = lambda r: f"{r}_tta20_la"
NOISE_1, NOISE_2 = 0.007, 0.005


def cf(run):
    """Cross-fitted CV per fold; NaN for runs trained before per-epoch CV probabilities were saved (Round 1)."""
    return cross_fitted(run) if (A.RESULTS / run / "fold0" / "cv_val_epoch_probs.npy").exists() else np.full(5, np.nan)


def fold_scores(run, split="cv_val"):
    return A.per_fold(la(run), split).bal_acc.to_numpy()


def arm(runs, split="cv_val"):
    return np.mean([fold_scores(r, split) for r in runs], 0)


def row(name, run, ref=R3):
    cv, te = A.per_fold(la(run), "cv_val"), A.per_fold(la(run))
    mu, _, b, n = A.paired(la(run), ref, "cv_val")
    tmu, _, tb, _ = A.paired(la(run), ref)
    return {"model": name, "run": la(run), "cv": cv.bal_acc.mean(), "test": te.bal_acc.mean(), "test_auc": te.auc.mean(),
            "test_mel_sens": te.mel_sens.mean(), "cv vs R3 model": f"{mu:+.4f} ({b}/{n})", "test vs R3 model": f"{tmu:+.4f} ({tb}/{n})",
            "cross-fitted cv": cf(run).mean(), **A.per_source(la(run)).add_prefix("test ").to_dict()}


def main():
    out = {"decisions": {}}
    rows = [row("fixed targets, seed 0 (Round-3 model)", FIXED[0]), row("fixed targets, seed 1", FIXED[1]),
            row("consistent targets, seed 0", CONS[0]), row("consistent targets, seed 1", CONS[1]),
            row("consistent targets, alpha 0.9, seed 0", A09)]
    t = pd.DataFrame(rows).set_index("model")
    print(t.drop(columns="run").T.to_string(), "\n")

    # 1. primary: 2-seed arms, paired by fold
    res = {}
    for split in ("cv_val", "official_test"):
        d = arm(CONS, split) - arm(FIXED, split)
        res[split] = {"diff": d.mean(), "fold_diffs": d.tolist(), "better": int((d > 0).sum()),
                      "fixed": arm(FIXED, split).mean(), "consistent": arm(CONS, split).mean()}
    # seed 1 only: the seed-0 control is the Round-1 run, which has no per-epoch CV probabilities
    x = {"fixed": cf(FIXED[1]), "consistent": cf(CONS[1])}
    res["cross_fitted"] = {"diff": (x["consistent"] - x["fixed"]).mean(), "better": int((x["consistent"] > x["fixed"]).sum()),
                           "fixed": x["fixed"].mean(), "consistent": x["consistent"].mean(), "seeds": "seed 1 only"}
    adopt = res["cv_val"]["diff"] > 0 and res["cv_val"]["better"] >= 3
    beyond = res["cv_val"]["diff"] > NOISE_2
    print(f"1. consistent - fixed targets (2 seeds per arm): CV {res['cv_val']['diff']:+.4f} ({res['cv_val']['better']}/5), "
          f"test {res['official_test']['diff']:+.4f} ({res['official_test']['better']}/5), "
          f"cross-fitted CV, seed 1 {res['cross_fitted']['diff']:+.4f} ({res['cross_fitted']['better']}/5)"
          f" -> {'ADOPT' if adopt else 'reject'}{' (beyond noise)' if adopt and beyond else ' (within noise)' if adopt else ''}")
    out["primary"] = res
    out["decisions"].update(consistent_adopted=bool(adopt), consistent_beyond_noise=bool(adopt and beyond))

    # 2. secondary: alpha 0.9
    mu, _, b, n = A.paired(la(A09), la(CONS[0]), "cv_val")
    tmu, _, tb, _ = A.paired(la(A09), la(CONS[0]))
    adopt09 = bool(adopt and mu > 0 and b >= 3)
    print(f"2. alpha 0.9 - alpha 0.5 (consistent, seed 0): CV {mu:+.4f} ({b}/{n}), test {tmu:+.4f} ({tb}/{n})"
          f" -> {'ADOPT' if adopt09 else 'not adopted' if adopt else 'reported only (rule 1 failed)'}")
    out["alpha09"] = {"cv_diff": mu, "cv_better": b, "test_diff": tmu, "test_better": tb}
    out["decisions"]["alpha09_adopted"] = adopt09

    # 3. noise: seed 1 - seed 0 within each arm
    out["noise"] = {}
    for name, (r0, r1) in (("fixed", FIXED), ("consistent", CONS)):
        cmu, csd, _, _ = A.paired(la(r1), la(r0), "cv_val")
        tmu, tsd, _, _ = A.paired(la(r1), la(r0))
        out["noise"][name] = {"cv_mean_diff": cmu, "cv_fold_sd": csd, "test_mean_diff": tmu, "test_fold_sd": tsd}
        print(f"3. seed 1 - seed 0, {name} targets: CV {cmu:+.4f} (fold sd {csd:.4f}), test {tmu:+.4f} (fold sd {tsd:.4f})")

    # pooled seed noise: seed 1 - seed 0 of the 5-fold mean, over every seed pair so far (incl. Round 2's drop path 0.2)
    pairs = [FIXED, CONS, ["cnx_kd_r2_rect_dp02_ema", "cnx_kd_r2_rect_dp02_s1_ema"]]
    pooled = {}
    for split in ("cv_val", "official_test"):
        d = np.array([A.paired(la(r1), la(r0), split)[0] for r0, r1 in pairs])
        pooled[split] = {"seed_diffs": d.tolist(), "single_run_sd": float(np.sqrt((d ** 2).mean() / 2))}
    out["noise"]["pooled"] = pooled
    print(f"   pooled single-run SD of the 5-fold mean ({len(pairs)} seed pairs): CV {pooled['cv_val']['single_run_sd']:.4f}, "
          f"test {pooled['official_test']['single_run_sd']:.4f} (= SD of a 2-seed vs 2-seed comparison)")
    gaps = {r: A.per_fold(la(r), "cv_val").bal_acc.mean() - A.per_fold(la(r)).bal_acc.mean() for r in FIXED + CONS + [A09]}
    out["cv_test_gap"] = gaps
    print("   CV - test gap: " + ", ".join(f"{r.replace('cnx_kd_', '')} {g:+.4f}" for r, g in gaps.items()))

    final = (A09 if adopt09 else CONS[0]) if adopt else FIXED[0]
    lo, hi = bootstrap_ci(la(final))
    print(f"\nRound-4 model: {la(final)} | CV {A.per_fold(la(final), 'cv_val').bal_acc.mean():.4f} | test {A.per_fold(la(final)).bal_acc.mean():.4f} "
          f"(95% CI {lo:.3f}-{hi:.3f}) | baseline test {A.per_fold(BASE).bal_acc.mean():.4f}")
    print("per-class recall (test):")
    print(pd.DataFrame({"baseline": A.per_class_recall(BASE), "fixed 2 seeds": np.mean([A.per_class_recall(la(r)) for r in FIXED], 0),
                        "consistent 2 seeds": np.mean([A.per_class_recall(la(r)) for r in CONS], 0)}).round(3).T.to_string())
    if TV.exists():
        out["teacher"] = json.loads(TV.read_text())
    ci_runs = {la(r): ([lo, hi] if r == final else list(bootstrap_ci(la(r)))) for r in (FIXED[0], CONS[0], A09)}
    print("test 95% CI: " + ", ".join(f"{r}: {a:.3f}-{b:.3f}" for r, (a, b) in ci_runs.items()))
    out.update(final=la(final), ci=[lo, hi], ci_runs=ci_runs, rows=t.reset_index().to_dict("records"))
    f = HWY_DIR / "results" / "comparisons" / "hwy_round4.json"
    f.write_text(json.dumps(out, indent=2, default=float))
    print("wrote", f)


if __name__ == "__main__":
    main()
