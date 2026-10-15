# Round 5 plan (written 2026-10-12, before any round-5 result was seen)

## Hypothesis
Part of the CV → test drop comes from acquisition shift: the official test set includes clinics and devices that are
absent from HAM10000. **Colour constancy** (Shades of Grey, p = 6; Finlayson & Trezzi 2004, used by the ISIC 2019
winner) normalises the illuminant of every image. **Stronger colour augmentation** makes the model robust to the
colour variation that remains. Both should help on the test set more than on the HAM10000-only CV folds.

## Experiment (one factor at a time, vs `resnet50_ce_res448`)
All runs use ResNet-50 at 448 px, plain CE, 30 epochs, 5 folds, seed 0, then 20-view TTA and logit adjustment
(τ tuned on CV).

| Run | Colour constancy | Colour augmentation |
|---|---|---|
| `resnet50_ce_res448` (exists) | no | basic: jitter 0.2 / 0.2 / 0.2, hue 0.02 |
| `resnet50_cc_res448` | **Shades of Grey** (train + test) | basic |
| `resnet50_coloraug_res448` | no | **strong**: jitter 0.3 / 0.3 / 0.3, hue 0.1 |
| `resnet50_cc_coloraug_res448` | **Shades of Grey** | **strong** |

## Decision rules, fixed now
* **Primary comparison (one test):** `resnet50_cc_coloraug_res448` vs `resnet50_ce_res448` on the official test set,
  paired over the 5 folds after TTA-20 + LA. The other two runs are ablations that attribute any effect.
* **CV cannot measure the shift**, so CV is a non-inferiority check only: a variant is acceptable if its CV balanced
  accuracy is at most 0.005 below the baseline's (fold noise is about ±0.016).
* **Adoption for the next round** (re-distillation and the ensemble): adopt the primary variant if it passes the CV
  non-inferiority check. If it fails, adopt nothing. Test results do not influence adoption.
