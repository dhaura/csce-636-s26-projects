# HW Y — Deep Learning Performance Improvement: Skin-Lesion Diagnosis (ISIC 2018 / HAM10000)

Homework Y continues the Homework X project (`../hwx/`, frozen as submitted). The task is the same: classify dermoscopic
images into the 7 diagnostic classes of the [ISIC 2018 Challenge, Task 3](https://challenge.isic-archive.com/landing/2018/47/),
scored by **balanced multiclass accuracy** (mean per-class recall).

**Baseline (end of Homework X):** the best single network, a ConvNeXt-S at 448 px. It is distilled from the out-of-fold
5-backbone ensemble and uses TTA-20 and logit adjustment. Scores: **CV 0.888, official test 0.864** (5-fold means).
Run `results/hwx/convnext_s_kd_res448_tta20_la`.

**Goal:** improve that single network with new techniques of our own, showing each step's gain separately.
Target ≥ 0.88 on the official test set.

## What is new in Homework Y
Every change is a flag of `scripts/train_classifier.py` (off by default, so the Homework X recipe is unchanged):

| Flag | Idea |
|---|---|
| `--ema 0.999` | exponential moving average of the weights, saved as a second run `<run>_ema` |
| `--drop-path 0.2` | stochastic depth |
| `--layer-decay 0.8` | layer-wise learning-rate decay (ConvNeXt, 14 groups) |
| `--img-width 600` | rectangular full-frame input (448 × 600 instead of a 448 × 448 crop) |
| `--lesion-boxes results/lesion_seg/boxes.csv` | lesion-centred crops from our lesion segmenter (`scripts/lesion_segmenter.py`) |
| `--metadata` | fusion of age, sex and body site, with modality dropout for images without metadata |
| `--teacher-views results/teacher_views/bb5_rect_v16` | consistent distillation: per-view teacher labels from `scripts/teacher_views.py` |

Plans are written before each round's results: [`research/plan.md`](research/plan.md).

## Setup
```bash
cd hwy
bash setup_links.sh                       # shares the HW X conda env, ISIC 2018 images and finished runs (no copies)
python data/download_ham10000_extra.py    # HAM10000 metadata, test metadata, lesion masks (~12 MB)
```
`results/hwx/` is a link to the Homework X runs, so the baseline and the distillation soft labels
(`results/hwx/soft_labels/bb5_calib_oof.csv`) are read in place.

## Layout
```
hwy/
├── README.md
├── setup_links.sh                 # links .conda-env, data/isic2018 and results/hwx to the Homework X folder
├── data/
│   ├── download_ham10000_extra.py # metadata + lesion masks from Harvard Dataverse
│   └── ham10000_extra/            # (created; git-ignored)
├── isic/                          # shared library (Homework X version + the additions below)
│   ├── data.py                    # + LesionCrop, MetaDataset (modality dropout), rectangular framing, stored views
│   ├── metadata.py                # (new) loads and encodes age / sex / body site
│   ├── models.py                  # + drop path, MetaFusion (image + metadata), layer-wise LR decay groups
│   ├── engine.py                  # + EMA updates, (image, metadata) inputs
│   └── metrics.py, decision.py, analysis.py
├── scripts/                       # run from hwy/ as: python scripts/<name>.py
│   ├── train_classifier.py        # Homework X trainer + the new flags
│   ├── lesion_segmenter.py        # (new) per-fold lesion segmenter -> lesion boxes
│   ├── model_soup.py              # (new) uniform weight averaging of fine-tuned runs, fold by fold
│   ├── teacher_views.py           # (new) 16 stored views per image, out-of-fold teacher labels for each view
│   ├── evaluate_tta.py, posthoc_logit_adjustment.py, compare_runs.py, ...  # as in Homework X
├── 01_round1_single_factors.ipynb      # round 1: six changes, one at a time
├── 02_round2_combination_noise.ipynb   # round 2: combination, drop path rate, seed noise, cross-fitted EMA
├── 03_round3_model_soup.ipynb          # round 3: model soups vs ingredients vs ensembles
├── 04_round4_consistent_distillation.ipynb  # round 4: per-view teacher labels, 2 seeds per arm, α 0.9
├── research/                      # plans (written before results), feasibility checks, round analyses, notebook generators
├── slurm/                         # node driver + GPU job queue; job lists per round in slurm/jobs/
├── report/                        # (local only, git-ignored) builds the online results page and guide
└── results/                       # (git-ignored) run outputs; results/hwx -> ../../hwx/results
```

## Running a round on Perlmutter
```bash
salloc -A m4012 -C gpu -q interactive -N 1 -G 4 -t 04:00:00 \
  srun -n 1 -c 128 -G 4 bash slurm/run_jobs_node.sh slurm/jobs/round1a_train.txt slurm/jobs/round1a_eval.txt
```

## Results
Report page: https://claude.ai/artifact/74uPfHdK8a8vkLkvKonSHo (updated after every round).
Guide with explanations and a slide outline: https://claude.ai/artifact/EMWvjjTDyN6JKqPMzvQTzb

**Round 1** (`research/round1_analysis.py`, notebook `01_round1_single_factors.ipynb`): six changes, each tested on its own against
the baseline. All numbers are 5-fold means after TTA-20 + logit adjustment.

| Change | CV | Test | CV paired | Decision |
|---|---|---|---|---|
| Baseline (HW X) | 0.888 | 0.864 | — | reference |
| Drop path 0.2 | 0.895 | **0.869** | +0.007 (4/5) | adopted |
| Full-frame 448 × 600 | **0.896** | 0.864 | +0.008 (3/5) | adopted |
| Layer-wise LR decay 0.8 | 0.886 | 0.864 | −0.002 (2/5) | not adopted |
| Lesion-centred crop | 0.887 | 0.862 | −0.001 (2/5) | not adopted |
| Metadata fusion | 0.887 | 0.863 | −0.001 (2/5) | not adopted |
| EMA (vs raw weights, 6 runs) | −0.002 | +0.002 to +0.007 | better in 1/6 runs | not adopted |

Findings:
* The lesion segmenter works well (out-of-fold Jaccard 0.912; boxes have median IoU 0.95 with the mask boxes), but lesion crops do not help.
* The metadata model learned to ignore its metadata: removing it changes 0.24% of CV predictions.
* EMA is worse on CV but better on test in every run. The cause is best-epoch selection: the raw weights' CV score is
  twice as noisy, so taking its maximum flatters it. Round 2 re-tests EMA with a selection-free (cross-fitted) CV score.

**Round 2** (`research/round2_analysis.py`, notebook `02_round2_combination_noise.ipynb`): full-frame runs with drop path
0.1 / 0.2 / 0.3, plus a seed-1 replicate.

| Run | CV | Test |
|---|---|---|
| Full frame + drop path 0.1 / 0.2 / 0.3 | 0.889 / 0.894 / 0.890 | 0.869 / 0.866 / 0.870 |
| Full frame + drop path 0.2, seed 1 | 0.887 | 0.861 |

* Drop path 0.2 is the best rate, but the combination does not beat full frame alone (CV −0.002), so full frame alone stays.
* EMA is adopted on a cross-fitted CV score (3 of 4 runs, +0.0002).
* **Noise floor:** a different seed moves the 5-fold mean by −0.007 (CV) and −0.005 (test). Every gain so far is within this.

**Round 3** (`research/round3_analysis.py`, `scripts/model_soup.py`, notebook `03_round3_model_soup.ipynb`): uniform weight
averages ("model soups") of our fine-tunes, still one network each. Soup of 10 full-frame runs: CV 0.880, test 0.870. Soup
of 6 square runs: CV 0.884, test 0.869. Both fail the CV rule against the Round-2 model. On test they match their
ingredients, but they lose on CV because they lack the best-epoch selection bonus.

**Round 4** (`research/round4_analysis.py`, `scripts/teacher_views.py`, notebook `04_round4_consistent_distillation.ipynb`):
consistent distillation. Every training image gets 16 stored random views; the out-of-fold teacher labels each view, and the
student trains on those views with their own labels (Beyer et al., CVPR 2022). Two seeds per arm. EMA weights throughout.

| Run | CV | Test | Test MEL sens. |
|---|---|---|---|
| Fixed labels, seed 0 / 1 (Round-3 model / replicate) | 0.894 / 0.879 | 0.867 / 0.850 | 0.816 / 0.828 |
| Consistent labels, seed 0 / 1 | 0.907 / 0.900 | **0.870** / 0.866 | 0.799 / 0.801 |
| Consistent labels, α 0.9, seed 0 | 0.914 | 0.860 | 0.742 |

* **Consistent vs fixed labels (2 seeds each): CV +0.017 (4/5 folds), test +0.009 (5/5)** — adopted, and the only change in
  HW Y that clears the noise on both CV and test.
* α 0.9 passes the CV rule (+0.007, 5/5) but is within noise and worse on test (−0.009, 0/5). Its CV − test gap (0.054, vs
  0.027 for fixed labels) points to the teacher's knowledge of the CV folds.
* **Noise, pooled over three seed pairs:** 1 SD of a 5-fold mean is 0.009 (CV) and 0.008 (test). The Round-3 model was a lucky seed.

**Final model by the pre-registered rules:** consistent distillation with α 0.9 (`cnx_kd_r4_cons_a09_ema`), CV 0.914, test
0.860 (95% CI 0.833–0.885). The step that held up is consistent distillation with α 0.5 (`cnx_kd_r4_cons_ema`): CV 0.907,
test **0.870** (95% CI 0.845–0.893), +0.006 over the baseline. The 0.88 target is not reached.
