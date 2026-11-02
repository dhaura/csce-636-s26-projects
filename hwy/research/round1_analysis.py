"""Round 1 analysis, as pre-registered in research/plan.md (run from hwy/: python research/round1_analysis.py).

Decision rule: adopt a factor if its paired CV gain over the Homework X baseline (TTA-20 + logit adjustment) is > 0
with >= 3/5 folds better. EMA is decided within runs (EMA copy vs raw weights of the same training run).
Metadata must in addition stay within 0.005 of the baseline's CV with all metadata removed.
Test numbers are printed for reporting only.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HWY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HWY_DIR))
from isic import analysis as A

pd.set_option("display.width", 220)
BASE = "hwx/convnext_s_kd_res448_tta20_la"
FACTORS = {"re-run (noise floor)": "cnx_kd_r1_base", "drop path 0.2": "cnx_kd_r1_dp02", "layer-wise LR decay": "cnx_kd_r1_lld",
           "rectangular frame": "cnx_kd_r1_rect", "lesion-centred crop": "cnx_kd_r1_lesion", "metadata fusion": "cnx_kd_r1_meta"}
la = lambda r: f"{r}_tta20_la"


def ok(run):
    return len(A.folds(run)) == 5


def main():
    rows, decisions = [], {}
    for name, r in FACTORS.items():
        for tag, run in (("raw", la(r)), ("EMA", la(f"{r}_ema"))):
            if not ok(run):
                continue
            cv, te = A.per_fold(run, "cv_val"), A.per_fold(run)
            mu, sd, better, n = A.paired(run, BASE, "cv_val")
            tmu, tsd, tbetter, _ = A.paired(run, BASE)
            row = {"factor": name, "weights": tag, "cv": cv.bal_acc.mean(), "cv_std": cv.bal_acc.std(ddof=0),
                   "test": te.bal_acc.mean(), "test_std": te.bal_acc.std(ddof=0), "test_auc": te.auc.mean(),
                   "cv_paired": mu, "cv_better": f"{better}/{n}", "test_paired": tmu, "test_better": f"{tbetter}/{n}",
                   "tau": A.tau(run)["selected_tau"], **A.per_source(run).add_prefix("test ").to_dict()}
            rows.append(row)
    t = pd.DataFrame(rows)
    print(f"Baseline {BASE}: CV {A.per_fold(BASE, 'cv_val').bal_acc.mean():.4f}, test {A.per_fold(BASE).bal_acc.mean():.4f}")
    print(t.round(4).to_string(index=False), "\n")

    # EMA: within-run paired comparison
    ema = []
    for name, r in FACTORS.items():
        if ok(la(r)) and ok(la(f"{r}_ema")):
            mu, sd, better, n = A.paired(la(f"{r}_ema"), la(r), "cv_val")
            tmu, _, tb, _ = A.paired(la(f"{r}_ema"), la(r))
            ema.append({"run": r, "cv EMA - raw": mu, "folds": f"{better}/{n}", "test EMA - raw": tmu, "test folds": f"{tb}/{n}"})
    ema = pd.DataFrame(ema)
    print("EMA vs raw weights of the same run (decides EMA)")
    print(ema.round(4).to_string(index=False), "\n")
    if len(ema):
        cv_gain = ema["cv EMA - raw"].mean()
        n_pos = int((ema["cv EMA - raw"] > 0).sum())
        decisions["EMA"] = bool(cv_gain > 0 and n_pos >= (len(ema) + 1) // 2)
        print(f"EMA: mean CV gain over {len(ema)} runs {cv_gain:+.4f}, positive in {n_pos}/{len(ema)} -> "
              f"{'adopt' if decisions['EMA'] else 'reject'}\n")

    # factor decisions (raw weights vs the baseline)
    for name, r in list(FACTORS.items())[1:]:
        if not ok(la(r)):
            print(f"{name}: not finished")
            continue
        mu, _, better, n = A.paired(la(r), BASE, "cv_val")
        adopt = mu > 0 and better >= 3
        if r.endswith("_meta"):
            cv_nometa = A.per_fold(la(r), "cv_val_nometa").bal_acc.mean()
            cv_base = A.per_fold(BASE, "cv_val").bal_acc.mean()
            nonin = cv_nometa >= cv_base - 0.005
            te_nm = A.per_fold(la(r), "official_test_nometa").bal_acc.mean()
            print(f"{name}: CV without metadata {cv_nometa:.4f} vs baseline {cv_base:.4f} -> non-inferior: {nonin} "
                  f"(test without metadata {te_nm:.4f}, reporting only)")
            adopt = adopt and nonin
        decisions[name] = bool(adopt)
        print(f"{name}: CV paired {mu:+.4f} ({better}/{n} folds) -> {'ADOPT' if adopt else 'reject'}")

    seg = HWY_DIR / "results" / "lesion_seg" / "boxes_summary.json"
    if seg.exists():
        print("\nlesion segmenter (out-of-fold, vs HAM10000 masks):", json.loads(seg.read_text()))
    out = HWY_DIR / "results" / "comparisons" / "hwy_round1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"baseline": BASE, "rows": rows, "ema": ema.to_dict("records"), "decisions": decisions}, indent=2))
    print(f"\ndecisions: {decisions}\nwrote {out}")


if __name__ == "__main__":
    main()
