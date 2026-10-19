# Round 6 plan (written 2026-10-12, before any round-6 result was seen)

## Question
Does out-of-fold ensemble distillation, which improved ConvNeXt-S, ResNet-50 and EfficientNetV2-S on every fold in
round 4, also help the two transformer backbones?

## Experiment (an exact replication of the round-4 protocol)
* Students: DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`, lr 5e-5) and ViT-B/16
  (`vit_base_patch16_224.augreg2_in21k_ft_in1k`, lr 5e-5). The learning rates are the round-3 pilot choices,
  unchanged.
* Soft labels: `results/soft_labels/bb5_calib_oof.csv`, the same file as round 4.
* Loss: 0.5·CE(hard) + 0.5·CE(soft), T = 1. Recipe: 448 px, 30 epochs, 5 folds, seed 0, TTA-20, logit adjustment.
* Known bias: a student's CV score is optimistic, because the teacher saw the student's own fold.

## Decision rules, fixed now
* **Primary comparison:** each student vs its hard-label twin on the official test set, paired over the 5 folds
  (`dinov2_b14_res448_tta20_la`, `vit_b16_res448_tta20_la`). No variants are tried.
* **Pooled replication:** the paired test gain across all five architectures (25 fold pairs), reported with the
  number of pairs that improved.
* **Ensembles (reporting only, nothing selected):** (a) all 5 students with `calib_tau`; (b) 5 backbones +
  5 students with `calib_tau`. The pre-registered round-4 final model stays the official final model. These are
  reported next to it, with bootstrap CIs.
