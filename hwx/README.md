# HW X — Deep Learning Exploration: Skin-Lesion Diagnosis (ISIC 2018 / HAM10000)

Classify dermoscopic images of skin lesions into the 7 diagnostic classes of the
[ISIC 2018 Challenge, Task 3](https://challenge.isic-archive.com/landing/2018/47/):

| Code | Class | Train images |
|---|---|---|
| MEL | Melanoma | 1,113 |
| NV | Melanocytic nevus (mole) | 6,705 |
| BCC | Basal cell carcinoma | 514 |
| AKIEC | Actinic keratosis / Bowen's disease | 327 |
| BKL | Benign keratosis | 1,099 |
| DF | Dermatofibroma | 115 |
| VASC | Vascular lesion | 142 |

The data is heavily imbalanced (67% NV), so the official metric is **balanced multiclass accuracy**
(mean per-class recall). We additionally report accuracy, macro-F1, macro-AUC and melanoma sensitivity/specificity.

## Layout

```
hwx/
├── README.md
├── requirements.txt               # pip dependencies
├── environment.yml                # conda env spec (python 3.11 + requirements.txt)
├── setup_env.sh                   # creates hwx/.conda-env and the "Python (isic-hwx)" Jupyter kernel
├── data/
│   ├── download_isic2018.py       # downloads + verifies the dataset into data/isic2018/
│   └── isic2018/                  # (created; git-ignored, ~3.1 GB)
├── isic/                          # shared code
│   ├── data.py                    # label loading, lesion-grouped CV folds, Dataset, transforms, multi-crop
│   ├── engine.py                  # train / predict loops (bf16 autocast), I/O helpers
│   └── metrics.py                 # balanced accuracy, AUC, per-class recall, melanoma sens/spec
├── 01_baseline_resnet50.ipynb     # our baseline: ImageNet ResNet-50, plain cross-entropy
├── literature_baseline_gessert2018.py   # literature baseline: Gessert et al. 2018 single-model recipe
├── train_classifier.py            # notebook-01 recipe as a CLI (one fold per run, one factor changed per flag)
├── compare_runs.py                # multi-fold comparison on the official test set + paired per-fold differences
├── posthoc_logit_adjustment.py    # re-weights a finished run's saved predictions by class prior (tau tuned on CV)
├── evaluate_tta.py                # re-evaluates a run's checkpoints with test-time augmentation (4 or 20 views)
├── slurm/                         # Perlmutter batch jobs (logs in slurm/logs/)
└── results/                       # (created; git-ignored) checkpoints, metrics.json, prediction CSVs
```

## Setup

```bash
cd hwx
bash setup_env.sh                       # conda env at hwx/.conda-env + Jupyter kernel "Python (isic-hwx)"
module load conda && conda activate $PWD/.conda-env
python data/download_isic2018.py        # ~3.3 GB download, verifies 10015 / 193 / 1512 images
```

`setup_env.sh` points `TORCH_HOME` / `HF_HOME` at `$SCRATCH/.cache` so pretrained weights don't fill `$HOME`.

**Fastest way to run everything** (baseline + 5 Gessert folds + summary in ~75 min on one 4-GPU node; the
`interactive` QOS usually starts within minutes while `shared`/`regular` can queue for days):

```bash
salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 03:00:00 srun -n 1 -c 128 -G 4 bash slurm/run_all_node.sh
```

> **Perlmutter notes.** Login nodes are shared — run training on a compute node, either with the batch
> scripts below or interactively (`salloc -A m4012 -C gpu -q interactive -G 1 -c 32 -t 1:00:00`, or a GPU
> node on jupyter.nersc.gov with the *Python (isic-hwx)* kernel). `$PSCRATCH` purges files not accessed for
> 8 weeks — copy `results/` somewhere permanent before the deadline.

## Data and splits

| Split | Images | Used for |
|---|---|---|
| Training (HAM10000) | 10,015 | training; fold 0 of a 5-fold CV held out for model selection |
| Official validation | 193 | reporting only |
| Official test | 1,512 | **headline number** (labels are public; comparable to the challenge leaderboard) |

HAM10000 contains several images of the same lesion (10,015 images, 7,470 lesions). Folds are built with
`StratifiedGroupKFold` grouped by `lesion_id` (from `ISIC2018_Task3_Training_LesionGroupings.csv`), so no lesion
appears in both training and validation. All experiments share the same folds (`cv_seed=0`).

## 1. Baseline — `01_baseline_resnet50.ipynb`

The "default recipe", with no class-imbalance handling, so later experiments have a clean reference point.

| | |
|---|---|
| Model | ResNet-50 (`timm` `resnet50.tv2_in1k`, 23.5M params), ImageNet-pretrained, fully fine-tuned |
| Input | resize short side to 256 → 224 crops (random-resized + flips + mild color jitter for training; center crop for eval) |
| Loss | plain cross-entropy |
| Optimiser | AdamW lr 2e-4, wd 0.01, batch 64, 30 epochs, 1-epoch warm-up + cosine |
| Selection | best balanced accuracy on CV fold 0 |

Run interactively in Jupyter, or headless (outputs are saved into the notebook):

```bash
sbatch slurm/baseline_notebook.sbatch
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute 01_baseline_resnet50.ipynb --output /tmp/smoke.ipynb  # 1-min check
```

Outputs go to `results/baseline_resnet50/` (`best.pt`, `history.csv`, `metrics.json`, `pred_<split>.csv`).

## 2. Literature baseline — Gessert et al. (2018)

> N. Gessert, T. Sentker, F. Madesta, R. Schmitz, H. Kniep, I. Baltruschat, R. Werner, A. Schlaefer.
> *Skin Lesion Diagnosis using Ensembles, Unscaled Multi-Crop Evaluation and Loss Weighting.*
> ISIC 2018 challenge report, [arXiv:1808.01694](https://arxiv.org/abs/1808.01694). Code: [ngessert/isic2018](https://github.com/ngessert/isic2018)

**Why this paper.**
- It placed **2nd in ISIC 2018 Task 3** (official test balanced accuracy **0.856**) and was the
  **best entry using only public data**.
- It reports **single-model results under leakage-free, lesion-grouped 5-fold CV** with all hyperparameters
  given. It also includes ablations (no pretraining, no loss weighting, number of crops) that map directly onto
  the questions this project studies.
- Public code is available to check the details the paper leaves out.

**Reproduced recipe** (`literature_baseline_gessert2018.py`, reference model *DenseNet-121*):

| | Paper | This script |
|---|---|---|
| Backbone | DenseNet-121, ImageNet-pretrained, full fine-tune | same (`densenet121.tv_in1k`, torchvision weights) |
| Input | full 600×450 images, **unscaled** random 224 crops, H/V flips, brightness (32/255) + saturation (0.5) jitter | same |
| Loss | CE weighted by inverse class frequency, wᵢ = N/nᵢ | same (`--weighting inverse`; also `inverse_c`, `none`) |
| Optimiser | Adam, lr 5e-4, batch 40, ×0.2 at epochs 50/75/100, 125 epochs | same |
| Selection | eval every 5 epochs with 16 grid crops, keep best WACC | same |
| Final eval | 36 unscaled grid crops, averaged | same |
| CV folds | authors' 5-fold lesion-grouped split | **our** 5-fold lesion-grouped split (shared with all experiments) |
| Normalisation | per-channel mean subtraction | ImageNet mean/std (matches the pretrained weights) |
| Precision | fp32 | bf16 autocast |
| Ensemble / SVM meta-learner | 54 models + SVM | not reproduced (single model; a 5-fold average is reported by `--summarize`) |

**Reported numbers to compare against** (Table II, 5-fold CV, 36-crop eval):

| Model | Balanced acc. (WACC) | Accuracy | Mean AUC |
|---|---|---|---|
| DenseNet-121 (reference recipe) | **0.795** | 0.823 | 0.967 |
| DenseNet-121, no loss weighting | 0.757 | 0.883 | 0.976 |
| DenseNet-121, no pretraining | 0.694 | 0.678 | 0.931 |
| DenseNet-121, 4-crop eval | 0.662 | 0.737 | 0.932 |
| SENet-154 (best single model) | 0.817 | 0.854 | 0.974 |
| 54-model ensemble | 0.851 | – | – |
| 54-model ensemble, **official test set** | **0.856** | – | – |

Run:

```bash
sbatch slurm/gessert2018_cv.sbatch                          # 5 folds as an array job
python literature_baseline_gessert2018.py --fold 0          # or a single fold interactively
python literature_baseline_gessert2018.py --smoke-test      # quick pipeline check
python literature_baseline_gessert2018.py --summarize       # mean ± std over folds + fold-ensemble on official test
```

Per-fold outputs go to `results/gessert2018_densenet121/fold<k>/`. `metrics.json` includes the best-checkpoint
metrics on the CV fold, official validation and official test sets. It also includes last-checkpoint metrics on
the CV fold, since the paper notes a large gap between best and last checkpoints.

## 3. Experiment: loss weighting on the baseline

Question: how much of the Gessert recipe's behaviour comes from **inverse-frequency loss weighting** alone?
`train_classifier.py` reproduces the notebook-01 baseline exactly; the only change is `--loss-weighting inverse`
(wᵢ = N/nᵢ). Both variants run on all 5 folds with the same seed, so differences are paired per fold.

```bash
salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 01:00:00 srun -n 1 -c 128 -G 4 bash slurm/run_loss_weighting_node.sh
python compare_runs.py resnet50_ce resnet50_wce gessert2018_densenet121    # re-print the comparison
```

Runs: `results/resnet50_ce/` (plain CE) and `results/resnet50_wce/` (weighted CE). The comparison is saved to
`results/comparisons/loss_weighting.json`.

**Results (official test set, mean ± std over 5 folds; ensemble = average of the 5 fold models)**

| Model | Balanced acc. | Accuracy | AUC | MEL sens. | MEL spec. | 5-fold ensemble balanced acc. |
|---|---|---|---|---|---|---|
| ResNet-50, plain CE | 0.724 ± 0.012 | 0.834 | 0.966 | 0.599 | 0.953 | 0.766 |
| ResNet-50, weighted CE | 0.773 ± 0.011 | 0.810 | 0.965 | 0.674 | 0.929 | **0.827** |
| Gessert DenseNet-121 (weighted CE + full-res crops + 36-crop eval) | 0.770 ± 0.012 | 0.818 | 0.965 | **0.749** | 0.916 | 0.787 |

Paired difference (weighted - plain, same fold and seed): balanced acc. **+0.050 ± 0.016 (better on 5/5 folds)**,
melanoma sensitivity +0.075 (5/5), accuracy -0.024, melanoma specificity -0.024, AUC -0.001.

Note: re-running an identical configuration (fold 0, same seed) gave 0.769 vs 0.743 test balanced accuracy, from GPU
non-determinism. Compare runs over folds, not single runs.

## 4. Experiment: post-hoc logit adjustment (no retraining)

`posthoc_logit_adjustment.py <run>` rescales a finished run's saved probabilities. It predicts
argmax log p(y|x) − τ·log πᵧ, with π taken from each fold's training split (Menon et al., ICLR 2021).
One τ is chosen by mean CV balanced accuracy, never on test. The output is written as a new run, `results/<run>_la/`.

| Model (official test, 5-fold mean) | τ (picked on CV) | Balanced acc. | Accuracy | MEL sens. | 5-fold ensemble balanced acc. |
|---|---|---|---|---|---|
| ResNet-50 plain CE | – | 0.724 ± 0.012 | 0.834 | 0.599 | 0.766 |
| ResNet-50 plain CE + logit adjustment | 1.6 | **0.786 ± 0.013** | 0.776 | 0.678 | 0.843 |
| ResNet-50 weighted CE | – | 0.773 ± 0.011 | 0.810 | 0.674 | 0.827 |
| ResNet-50 weighted CE + logit adjustment | 0.5 | 0.785 ± 0.014 | 0.770 | 0.695 | **0.853** |
| Gessert DenseNet-121 | – | 0.770 ± 0.012 | 0.818 | 0.749 | 0.787 |
| Gessert DenseNet-121 + logit adjustment | 0.2 | 0.793 ± 0.008 | 0.790 | 0.773 | 0.818 |

Adjusting the plain-CE model after training beats retraining with a weighted loss (+0.063 vs +0.050 over plain CE,
5/5 folds). Weighting plus adjustment gives no further gain. Imbalance handling therefore plateaus near 0.785 for
ResNet-50 at 224 px. The cost is accuracy and macro-F1: more moles get flagged as other classes.

```bash
python posthoc_logit_adjustment.py resnet50_ce
python compare_runs.py resnet50_ce resnet50_ce_la resnet50_wce resnet50_wce_la gessert2018_densenet121 gessert2018_densenet121_la
```

## 5. Round 2: optimising the baseline

**Protocol.** Each variant changes one factor of `resnet50_ce`: 384 px, 448 px, 60 epochs, or TrivialAugment.
Each is trained on all 5 folds with the same seed. Every run is then re-evaluated with test-time augmentation
(`evaluate_tta.py`: 4 views = flips, or 20 views = 5 crops × flips) and logit-adjusted. The configuration is
**selected by mean CV balanced accuracy only**. Test numbers are reported for every variant but never used for choices.

```bash
salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 02:30:00 srun -n 1 -c 128 -G 4 bash slurm/run_round2_node.sh
```

**Selection on the CV folds (top rows, all logit-adjusted):**

| Variant | CV balanced acc. | Test balanced acc. | Test MEL sens. | Test 5-fold ensemble |
|---|---|---|---|---|
| **448 px + TTA-20** (selected) | **0.831 ± 0.016** | **0.835 ± 0.014** | 0.759 | **0.870** |
| 448 px + TTA-4 | 0.829 ± 0.016 | 0.837 ± 0.012 | 0.779 | 0.871 |
| 448 px | 0.817 ± 0.017 | 0.826 ± 0.012 | 0.770 | 0.863 |
| 384 px + TTA-4 | 0.810 ± 0.021 | 0.798 ± 0.018 | 0.773 | 0.847 |
| 224 px + TTA-20 | 0.803 ± 0.017 | 0.804 ± 0.009 | 0.695 | 0.838 |
| 224 px (round-1 best) | 0.787 ± 0.018 | 0.786 ± 0.013 | 0.678 | 0.843 |

**Where the gain comes from (paired over 5 folds, test balanced accuracy):**

| Step | CV | Test |
|---|---|---|
| Plain-CE baseline → + logit adjustment | +0.028 (5/5) | +0.063 (5/5) |
| → + 448 px | +0.031 (5/5) | +0.040 (5/5) |
| → + TTA-20 | +0.014 (5/5) | +0.009 (3/5) |
| **Total vs. raw baseline** (0.724 → 0.835) | +0.073 (5/5) | **+0.111 (5/5)**; MEL sens. +0.160 |
| 60 epochs alone | −0.000 | +0.010 (not selected) |
| TrivialAugment alone | +0.007 | +0.010 (not selected) |

**Optimised baseline** = ResNet-50, 448 px crops (resize 512), 30 epochs, mild color jitter, plain CE;
20-view TTA; logit adjustment with τ = 1.6. Runs: `results/resnet50_ce_res448_tta20_la/`.
It beats the Gessert DenseNet-121 reproduction by +0.064 test balanced accuracy (5/5 folds; +0.041 vs Gessert + LA).
Its 5-fold ensemble (0.870) exceeds the paper's 54-model ensemble (0.856).
Resolution is the factor that improves the features: AUC rises, unlike with loss weighting. Longer training and
stronger augmentation did not help on the CV folds.

## Other reference points

- ISIC 2018 Task 3 official test leaderboard: winner 0.885 (MetaOptima, used external data); top-10 ≈ 0.78–0.88.
- Tschandl et al., *Lancet Oncology* 2019: experienced dermatologists got 18.8 / 30 correct, versus 25.4 / 30 for the top-3 algorithms.
- MedMNIST v2, DermaMNIST (same images, low resolution, plain accuracy): ResNet-18 at 224 px gets 0.754.

## Data license and citations

ISIC 2018 data is CC-BY-NC 4.0. Please cite:
- P. Tschandl, C. Rosendahl, H. Kittler. *The HAM10000 dataset…* Sci. Data 5, 180161 (2018).
- N. Codella et al. *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by ISIC.* arXiv:1902.03368 (2019).
