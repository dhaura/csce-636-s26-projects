"""Round 5 analysis, following research/round5_plan.md. Run from hwx/ after the round-5 node job.

1. Logit-adjust every round-5 TTA-20 run (tau tuned on CV).
2. CV non-inferiority check (margin 0.005) -> adoption decision (test not used).
3. The single primary test comparison + the two ablations (reporting).
"""
import subprocess
import sys
from pathlib import Path

HWX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HWX))
from isic import analysis as A  # noqa: E402

BASE = "resnet50_ce_res448_tta20_la"
RUNS = {"primary: colour constancy + strong colour aug": "resnet50_cc_coloraug_res448",
        "ablation: colour constancy only": "resnet50_cc_res448",
        "ablation: strong colour aug only": "resnet50_coloraug_res448"}
MARGIN = 0.005

for run in RUNS.values():
    if not (A.RESULTS / f"{run}_tta20_la").exists():
        subprocess.run([sys.executable, "posthoc_logit_adjustment.py", f"{run}_tta20"], cwd=HWX, check=True, capture_output=True)

print(f"{'variant':<48s} {'tau':>4s} {'CV':>7s} {'CV paired vs base':>24s} {'non-inferior':>12s} | {'TEST':>7s} {'TEST paired':>22s} {'AUC paired':>22s} {'MEL paired':>22s} | 5-model")
b_cv, b_te = A.per_fold(BASE, "cv_val").bal_acc.mean(), A.per_fold(BASE).bal_acc.mean()
print(f"{'baseline (resnet50_ce_res448)':<48s} {A.tau(BASE)['selected_tau']:>4} {b_cv:7.4f} {'':>24s} {'':>12s} | {b_te:7.4f} {'':>22s} {'':>22s} {'':>22s} | {A.fold_ensemble(BASE)['balanced_acc']:.4f}")
fmt = lambda t: f"{t[0]:+.4f}±{t[1]:.4f} ({t[2]}/{t[3]})"
for name, run in RUNS.items():
    r = f"{run}_tta20_la"
    cv = A.paired(r, BASE, "cv_val")
    ok = cv[0] >= -MARGIN
    print(f"{name:<48s} {A.tau(r)['selected_tau']:>4} {A.per_fold(r, 'cv_val').bal_acc.mean():7.4f} {fmt(cv):>24s} {('yes' if ok else 'NO'):>12s} | "
          f"{A.per_fold(r).bal_acc.mean():7.4f} {fmt(A.paired(r, BASE)):>22s} {fmt(A.paired(r, BASE, metric='macro_auc')):>22s} "
          f"{fmt(A.paired(r, BASE, metric='mel_sensitivity')):>22s} | {A.fold_ensemble(r)['balanced_acc']:.4f}")
primary = f"{RUNS['primary: colour constancy + strong colour aug']}_tta20_la"
adopt = A.paired(primary, BASE, "cv_val")[0] >= -MARGIN
print(f"\nAdoption (pre-registered, CV non-inferiority only): {'ADOPT colour constancy + strong colour aug' if adopt else 'adopt nothing'}")
print("\nCV - test gap (mean over folds; smaller = less shift):")
for name, r in [("baseline", BASE)] + [(n, f"{run}_tta20_la") for n, run in RUNS.items()]:
    print(f"  {name:<48s} {A.per_fold(r, 'cv_val').bal_acc.mean() - A.per_fold(r).bal_acc.mean():+.4f}")
