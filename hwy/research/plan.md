# Homework Y plan (written before any Homework Y run)

## Goal and baseline
Improve one network for ISIC 2018 Task 3 (7 classes, balanced multiclass accuracy).

**Baseline = best single network from Homework X:** ConvNeXt-S (ImageNet-22k), 448 px, distilled from the out-of-fold
5-backbone ensemble (0.5 CE + 0.5 soft CE), 30 epochs, AdamW 2e-4, TTA-20, logit adjustment (τ tuned on CV).
Run `hwx/convnext_s_kd_res448_tta20_la`: CV 0.888 ± 0.020, test 0.864 ± 0.010 (5-fold means).
Everything in Homework X (the protocol, distillation, TTA, logit adjustment) is previous work; Homework Y counts only
changes made on top of this model.

**Target:** one network at ≥ 0.88 test balanced accuracy (the 5-backbone ensemble reached 0.892).

## Evaluation protocol (unchanged from Homework X)
* Same 5 lesion-grouped CV folds (`add_cv_folds(seed=0)`); every variant trains all 5 folds.
* Score = TTA-20 + logit adjustment (τ picked on CV), 5-fold mean balanced accuracy.
* **Decisions use CV only**, as paired per-fold differences against the reference. Adopt a change if the mean paired CV
  gain is > 0 and at least 3 of 5 folds improve. The test set (1,512 images, 95% CI ≈ ±0.025) is reported for every
  variant and never used to choose.
* Caveat inherited from Homework X: the soft labels leak some CV-fold knowledge, so student CV is optimistic
  (CV − test ≈ 0.025 for both hard and distilled ConvNeXt-S, so the leak is small). All Homework Y variants share it.
* Noise floor: Round 1 re-runs the baseline unchanged (with an EMA copy tracked alongside), which measures run-to-run noise.
  *Note added during Round 1:* the re-run's fold 0 is bit-identical to Homework X (same seed, same node type), so an
  identical re-run measures nothing. Round 2 adds a re-run with a different seed (`--seed 1`) as the noise floor.

## Feasibility checks (CPU, `research/feasibility_checks.py`)
* **Metadata** (age, sex, body site from the HAM10000 Dataverse tables, test included). `dx_type` and `dataset` are never
  inputs (label- or acquisition-linked). The 316 test images from the 'external' source (21%) have no metadata at all;
  training images almost always do. A logistic-regression stacker on the baseline's OOF predictions plus metadata gained
  +0.001 on CV. On test it gained +0.013 to +0.018 on the HAM-source images, but collapsed on the metadata-free
  'external' images (0.889 → 0.783). Training it with copies whose metadata was removed fixed most of this.
  *Disclosure:* this diagnostic looked at test results. The design choice it led to (train for missing metadata) depends
  only on which test inputs have metadata, which is known at prediction time. It does not depend on any test labels.
* **Lesion crops:** in the HAM10000 masks, the median lesion covers 24% of the image and 20% of lesions touch the image
  border. A lesion-centred square crop (1.3 × box) zooms in more than 1.25× for 26% of images. For large lesions it
  zooms *out*, so the whole lesion is visible; the baseline's centre view cuts off border-touching lesions.

## Round 1 — one factor at a time against the baseline (all distilled, same soft labels and seed)
| Run | Change | Why |
|---|---|---|
| `cnx_kd_r1_base` | none (re-run); EMA copy tracked | noise floor; EMA decided by the paired comparison **within the same run** |
| `cnx_kd_r1_dp02` | stochastic depth (drop path) 0.2 | the standard ConvNeXt fine-tuning regulariser, unused in Homework X |
| `cnx_kd_r1_lld` | layer-wise LR decay 0.8 (14 groups, as in the ConvNeXt paper), top LR 4e-4 | protect the pretrained low-level features while the top layers adapt |
| `cnx_kd_r1_rect` | rectangular full-frame input 448 × 600 (same pixel scale, 34% more field of view) | the square crop discards the image sides |
| `cnx_kd_r1_lesion` | lesion-centred square crop from a lesion segmenter (train: random margin 1.1–1.6 × box, ±10% shift; eval: 1.3 ×; side 225–450 px) | Grad-CAM showed background attention; small lesions get more pixels |
| `cnx_kd_r1_meta` | image + metadata fusion (MLP 20 → 64 → 64, concatenated with the pooled image features), **modality dropout**: all metadata removed for 25% of training samples, each field for a further 10% | metadata adds information that is not in the image. Modality dropout is needed because 21% of test images have no metadata |

Every run also tracks an EMA copy of the weights (decay 0.999, timm warm-up). Its predictions are written as
`<run>_ema` so that EMA × factor can be read off for free.

**Lesion segmenter:** a U-Net-style decoder on an ImageNet ResNet-50, at 384 × 288 px, BCE + Dice loss. It is trained per
CV fold on the HAM10000 masks (the same folds as the classifier). Each training image gets its box from the fold model
that did not see it. Validation and test images use the 5-fold mean mask. Boxes always come from *predicted* masks (never
the ground truth), so training and test inputs match. Box = bounding box of the largest connected component; an empty
mask falls back to the full image.

**Metadata adoption rule (extra):** CV with metadata must improve (the rule above). In addition, CV with **all metadata
removed** must not fall below the baseline by more than 0.005. Test is also reported per source (HAM sources vs
'external').

## Round 1 outcome (`research/round1_analysis.py`)
Adopted: **drop path 0.2** (CV +0.007, 4/5 folds) and **full-frame 448 × 600** (CV +0.008, 3/5). Not adopted:
layer-wise LR decay (−0.002), lesion-centred crop (−0.001), metadata fusion (−0.001), and EMA (worse than the raw weights
in 5 of 6 runs on CV). Two follow-up checks used CV data only:
* **EMA vs checkpoint selection.** The raw weights' CV score fluctuates about twice as much between epochs as the EMA
  weights' (sd 0.0067 vs 0.0030). So picking the best epoch on the CV fold lifts raw CV 0.004–0.006 more than EMA CV. At
  the last epoch, where nothing is selected, the two are equal. Our CV rule is therefore tilted against EMA by about
  the size of its deficit.
* **Metadata unused.** Removing all metadata from the fusion model changes its prediction for 0.24% of CV images. The
  stacker check had already found that metadata adds nothing on CV on top of a strong image model. Metadata is dropped.

## Round 2 — combine the adopted factors, tune drop path, measure noise (written before any Round 2 run)
All runs use full-frame 448 × 600 input, distillation as before, and an EMA copy.
| Run | Change |
|---|---|
| `cnx_kd_r2_rect_dp02` | drop path 0.2 + full frame (the combination) |
| `cnx_kd_r2_rect_dp01` | drop path 0.1 + full frame |
| `cnx_kd_r2_rect_dp03` | drop path 0.3 + full frame |
| `cnx_kd_r2_rect_dp02_s1` | same as the combination with `--seed 1`: noise floor (run-to-run spread of the 5-fold mean) |

**Rules.**
1. **Drop path rate:** choose 0.1 / 0.2 / 0.3 by CV mean (TTA-20 + logit adjustment; seed-0 runs only).
2. **Combination vs best single factor:** compare the chosen run with `cnx_kd_r1_rect` (the best Round-1 single factor
   on CV), using the Round-1 rule (paired CV gain > 0, ≥ 3/5 folds). If it fails, keep `cnx_kd_r1_rect`.
3. **EMA, re-tested without selection bias:** every Round-2 run saves its per-epoch CV predictions (raw and EMA). The
   EMA decision uses a *cross-fitted* CV score. Each CV fold is split into two class-stratified halves. The epoch is
   picked on one half and scored on the other, then the halves swap. Adopt EMA if its cross-fitted score beats the raw
   weights' in at least 3 of the 4 Round-2 runs, with a positive mean.
4. **Noise floor:** report the seed-1 minus seed-0 paired difference. A gain smaller than this spread is called
   "within noise" in the write-up, even if a rule adopted it.

## Round 2 outcome (`research/round2_analysis.py`)
* Drop path rate: 0.2 is best on CV (0.1: 0.889, 0.2: 0.894, 0.3: 0.890).
* The combination (full frame + drop path 0.2) does not beat full frame alone: CV −0.002 (3/5). So full frame alone stays.
* EMA, cross-fitted CV: better in 3 of 4 runs, mean +0.0002, so EMA is adopted. The rule is met, but the margin is tiny.
* **Noise floor:** seed 1 − seed 0 of the same configuration is −0.007 on CV and −0.005 on test (5-fold means, per-fold
  sd 0.018 and 0.011). Every Round-1/2 gain is about this size, so by rule 4 they are all **within noise**.
* Round-2 model: `cnx_kd_r1_rect_ema` (full frame + EMA): CV 0.894, test 0.867.

## Round 3 — weight averaging ("model soup") of our fine-tuned networks (written before any soup is evaluated)
The Round-2 noise floor says single fine-tuning runs differ by about ±0.007. Averaging the *weights* of several
fine-tuned copies of one network (Wortsman et al., ICML 2022) gives **one network** with the same inference cost. All our
ConvNeXt-S runs start from the same ImageNet-22k checkpoint and use LayerNorm (no batch statistics), so per-fold weight
averages are well defined. No new training.
| Soup | Ingredients per CV fold (uniform average, no selection) | Evaluated at |
|---|---|---|
| `soup_rect10` | full-frame runs `cnx_kd_r1_rect`, `cnx_kd_r2_rect_dp01/dp02/dp03/dp02_s1`, raw + EMA weights (10) | 448 × 600 |
| `soup_square6` | square runs `cnx_kd_r1_base`, `cnx_kd_r1_dp02`, `cnx_kd_r1_lld`, raw + EMA weights (6) | 448 × 448 |
The lesion-crop run is left out because it was trained on a different input view. The metadata run is left out because
its head is different. Each ingredient is that run's selected checkpoint for the fold, so a soup carries the same
best-epoch selection as any single run, and no extra selection.
**Rule:** compare each soup with the Round-2 model (`cnx_kd_r1_rect_ema`) using the usual rule (TTA-20 + logit
adjustment, paired CV gain > 0, ≥ 3/5 folds). If both pass, the one with the higher CV mean is adopted. Each soup is
also compared with the noise floor from Round 2.

## Round 3 outcome (`research/round3_analysis.py`)
Both soups fail the rule against the Round-2 model. `soup_rect10`: CV 0.880 (−0.014, 0/5 folds), test 0.870.
`soup_square6`: CV 0.884 (−0.010, 0/5), test 0.869. On test each soup is at or slightly above its ingredients' mean
(0.870 vs 0.868; 0.869 vs 0.867). On CV it is 0.012 below. The likely reason is the same one found for EMA: every
ingredient's checkpoint is its best CV epoch (worth 0.010–0.016 of CV in Rounds 1–2), while the soup itself is never
selected. The pre-registered decision stands. **Adopted model after Round 3: `cnx_kd_r1_rect_ema` (full frame + EMA),
CV 0.894, test 0.867 (95% CI 0.841–0.890).** For context, the prediction ensembles of the same ingredients reach test 0.875
(10 networks) and 0.883 (6 networks).

## Round 4 — consistent distillation (written before any Round 4 run)
**Motivation.** Rounds 1–3 changed the student's training recipe and found nothing beyond seed noise; the remaining gap
is to the teacher (0.892 test) and to ensembles (0.875–0.883). So Round 4 changes the *distillation target* instead.
Until now every image has one fixed soft label: the teacher ensemble's TTA-20 prediction on the whole image. The student,
however, sees a different random crop / flip / colour jitter every epoch, so it is asked to predict the whole-image label
from a crop that may not contain the evidence for it. Beyer et al. (CVPR 2022, "A good teacher is patient and
consistent") find that the teacher should see the *same* view as the student.

**Method** (`scripts/teacher_views.py`, `--teacher-views` in `train_classifier.py`):
1. For every training image, 16 random training views are drawn once and stored as parameters
   (448 × 600 RandomResizedCrop, H/V flips, colour jitter; the same distribution as the training pipeline). Seed 0.
2. The out-of-fold 5-backbone teacher (the same HW X fold models as for the fixed labels; an image's teacher never
   saw it) predicts every view. ViT members get position embeddings resampled to the 448 × 600 grid (login-node check on
   352 fold-0 images: every member's accuracy on 448 × 600 centre views is within ±0.012 of its square 448 accuracy).
3. One temperature per member is fitted on its per-view out-of-fold log-probs; calibrated log-probs are averaged and
   renormalised (as `make_soft_labels.py`). No logit adjustment.
4. Each epoch the student takes one of the 16 stored views per image (uniformly) with the soft label for that view.
   Everything else is the adopted recipe: ConvNeXt-S, 448 × 600, AdamW 2e-4, 30 epochs, EMA 0.999, α = 0.5.

**Runs** (all full frame + EMA; the EMA weights are the model, as adopted in Round 2):
| Run | Targets | Seed | Role |
|---|---|---|---|
| `cnx_kd_r1_rect` (exists) | fixed labels | 0 | control |
| `cnx_kd_r1_rect_s1` | fixed labels | 1 | control, second seed |
| `cnx_kd_r4_cons` | per-view labels, α 0.5 | 0 | consistent distillation |
| `cnx_kd_r4_cons_s1` | per-view labels, α 0.5 | 1 | consistent distillation, second seed |
| `cnx_kd_r4_cons_a09` | per-view labels, α 0.9 | 0 | more weight on the (now consistent) teacher |

**Rules** (CV = TTA-20 + logit adjustment, EMA weights, as always; test reported only):
1. **Primary — consistent vs fixed targets, 2 seeds per arm.** Per fold, average the two seeds of each arm; adopt
   consistent distillation if the paired CV gain is > 0 with ≥ 3/5 folds better. Two seeds per arm halve the
   seed variance of the comparison. The cross-fitted (selection-free) CV score is reported alongside.
2. **Secondary — α 0.9.** Compared with the seed-0 run of the primary's winning arm *with the same targets*
   (`cnx_kd_r4_cons` if rule 1 adopts consistent targets; otherwise α 0.9 is reported only). Adopt if paired CV gain
   > 0 with ≥ 3/5 folds.
3. **Noise.** A single-seed gain is called an improvement only if it exceeds the Round-2 noise floor (0.007 CV);
   for the 2-seed comparison the threshold is 0.007 / √2 ≈ 0.005.
4. The final model and its bootstrap 95% CI are reported for the seed-0 run of the adopted recipe, as before.

Diagnostics (training data only): per-view teacher accuracy, how far the per-view targets are from the fixed target
(total variation), and how often the teacher's top class changes between views of one image.

**Budget.** Two 4-GPU nodes for up to 4 h: teacher pass (5 fold jobs, about 12 min each), then 20 student fold
trainings (about 42 min each) and 8 TTA evaluations.

### Round 4 outcome (2026-11-02; `research/round4_analysis.py`, `results/comparisons/hwy_round4.json`, notebook 04)
All 28 jobs succeeded (teacher pass 5 × 5 min, 20 trainings × 40 min, 8 TTA evaluations). Scores: EMA, TTA-20 + LA.

| Run | CV | Test | Test MEL sens. | Cross-fitted CV |
|---|---|---|---|---|
| fixed labels, seed 0 (Round-3 model) | 0.894 | 0.867 | 0.816 | — (Round-1 run, no per-epoch probs) |
| fixed labels, seed 1 | 0.879 | 0.850 | 0.828 | 0.812 |
| consistent labels, seed 0 | 0.907 | 0.870 | 0.799 | 0.830 |
| consistent labels, seed 1 | 0.900 | 0.866 | 0.801 | 0.820 |
| consistent labels, α 0.9, seed 0 | 0.914 | 0.860 | 0.742 | 0.839 |

1. **Primary: adopted.** Consistent − fixed (2 seeds per arm): CV **+0.017 (4/5 folds)**, test +0.009 (5/5),
   cross-fitted CV (seed 1 only) +0.008 (4/5). Beyond the pre-registered 0.005 threshold. 2-seed means on test:
   fixed 0.858, consistent 0.868.
2. **α 0.9: adopted by the rule, within noise, and worse on test.** CV +0.007 (5/5) vs `cnx_kd_r4_cons`, below the
   0.007 single-seed floor; test −0.009 (0/5) and melanoma sensitivity 0.80 → 0.74. Per rule 4 the Round-4 model is
   `cnx_kd_r4_cons_a09_ema_tta20_la`: CV 0.914, test 0.860 (95% CI 0.833–0.885), baseline 0.864.
3. **Noise was underestimated.** Seed 1 − seed 0 of the 5-fold mean: fixed −0.015 CV / −0.017 test, consistent
   −0.007 / −0.004, Round-2 drop path 0.2 −0.014 / −0.008. Pooled single-run SD of the 5-fold mean: **0.009 CV,
   0.008 test** (the Round-2 floor of 0.007 was one pair). A 2-seed vs 2-seed comparison has the same SD (≈ 0.009), so
   the primary CV gain is ≈ 1.9 SD, and test agrees. The Round-3 model (fixed labels, seed 0) was a lucky seed:
   its seed-1 twin scores 0.850 on test. Training losses of the two match (0.2174 vs 0.2170 at epoch 30), so the code
   path is not at fault.
4. **CV is becoming optimistic as the student leans on the teacher.** CV − test gap: fixed 0.027–0.029,
   consistent 0.034–0.037, α 0.9 0.054. A likely cause is the known leak in the out-of-fold teacher: a training image's
   teacher was trained on the other folds, including the student's CV fold, so the soft labels carry information
   about the CV images; more weight on the teacher (α 0.9) means more of it. Fixing this needs a teacher per
   (student fold, image fold) pair (20 teachers × 5 backbones), which is out of budget for HW Y.

**Decision for the HW Y deliverable.** Rule 4 names the α 0.9 run as the final model, and it is reported as such.
The finding that holds up is rule 1: consistent targets beat fixed targets by about +0.01 balanced accuracy averaged
over two seeds, on both CV and test. The ≥ 0.88 single-network target was not reached (best adopted test 0.870, consistent
α 0.5, seed 0).

## Later rounds (plans written before each round, in this file)
* **Round 2:** combine all adopted Round-1 factors in one model. If the combination loses on CV to the best single
  factor, keep the best single factor. *(Specified above.)*
* **Round 3:** second-generation distillation (a teacher ensemble that includes the improved students), and/or a larger
  student (ConvNeXt-B) with the final recipe. Out of scope unless rounds 1–2 leave time: self-supervised pretraining on
  unlabeled ISIC images (it would have to exclude every test image).

## Reporting
For each step: CV and test (5-fold mean ± std), paired differences against the previous step, per-class recall,
per-source test breakdown, and a bootstrap 95% CI for the final model. Negative results are reported as such.
