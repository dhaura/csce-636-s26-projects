# Round 4 plan (written 2026-10-12, before any round-4 result was seen)

## Questions
1. **Decision layer (done, CPU).** Can a better ensemble decision layer, fitted on out-of-fold (OOF) predictions,
   beat per-member logit adjustment? Selection is by nested CV: fit on 4 folds' OOF predictions, score the 5th.
2. **Distillation.** Does a single student trained on OOF-ensemble soft labels beat the same architecture trained on
   hard labels? Students: ResNet-50, EfficientNetV2-S and ConvNeXt-S at 448 px. Settings are fixed in advance:
   alpha = 0.5, T = 1, same lr and recipe as round 3, TTA-20 plus the selected decision layer.
3. **FixRes.** Does testing the round-3 CNN checkpoints at 512 px instead of 448 px help?

## Known bias
A student's soft labels come from fold models that were trained on the student's own CV fold. Student CV scores,
and the CV scores of any ensemble that contains students, are therefore optimistic and are not used to choose.

## Decision rules, fixed now
* **Decision layer:** the nested-CV winner, `calib_tau` (temperature-calibrate each member, average log-probs,
  single tau). Selected 2026-10-12 with nested CV 0.8746 vs 0.8719 for per-member tau.
* **FixRes:** replace a CNN member's 448-px evaluation with 512 px only if that member's own CV balanced accuracy
  improves. This is clean, because FixRes uses no soft labels.
* **Students, single model:** reported on the official test set against their hard-label twins. No variants are
  tried, so there is no selection.
* **Final model (fixed in advance):** `calib_tau` over all round-3 backbones, with the FixRes choice applied, plus
  the three distilled students. This model is reported on test whatever its CV score. The 5-backbone ensemble without
  students is reported next to it.
