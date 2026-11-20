# Data-side and training-objective levers for the 52-point plateau

Scope: small labelled set (3,573 frames; about 1,900 count for first-batch-like images), six defects with ordinal grades,
noisy human labels, objective Doc2_Strict (25 positive macro-F1 over 14 labels + 20 P0 + 20 clean). "Points" below mean
points of the 65-point objective. The project's own documents are treated as authoritative for what was tested. The
current deliverable `primary` scores 51.45 nested out-of-fold (OOF) (52.21–52.30 on first-batch rows) and 51.78 on the
practice reference.

Local evidence used throughout (paths relative to `/wanqing-develop/luowenjing/18`):
- `docs/REASSESSMENT_AND_PLAN_2026-11-12.md` (abbreviated REASSESS) §1–2, §7–11
- `docs/REVIEW_AND_ROADMAP_2026-11-11.md` (REVIEW) §2–4, §9
- `docs/SYNTHLAB.md` §1, §7–14
- `results/next55/LEDGER.md` (LEDGER)
- `code/r55/train_testbed.py`, `code/qingyu/unimodal_xbm_losses.py`

What the current training objective already contains. This was verified in code: `train_testbed.py` around line 449
instantiates `QingyuUnimodalCompositeLoss(soft_sigma=0.35, lambda_rps=0.45, lambda_rank=0.25, ...)`, and
`unimodal_xbm_losses.py` around lines 448–498 defines it. The loss has five parts:
- soft ordinal Gaussian targets (σ = 0.35), with no mass leaked onto "clean". This is ordinal label smoothing.
- squared-EMD / RPS loss.
- a Thurstone pairwise ranking loss with a 512-item cross-batch memory, doubled on adjacent grades.
- an all-clean BCE term.
- fixed rarity weights per (defect, grade): P0 ×1.65, 过度生成 ×1.85, 过锐 P1/P0 ×1.75, 花屏 / 虚化突兀 P0 ×1.90.

Around it, the trainer adds:
- a unimodal quadratic residual head and a prototype bank;
- weight averaging (`swa_sds`): the mean of the last 2 epochs, plus the best epoch under `--select val`;
- optional cross-fitted teacher KL on labelled rows (`--lambda-kd-lab`, the `xd` run) and soft-label self-training (`--ssl`).

Thresholds are not learned in the loss. A multi-start decoder (Decoder55) fits per-label thresholds for Doc2_Strict on
pooled OOF probabilities.

---

## Q1. Targeted synthetic data / augmentation for first-batch failure modes

### Takeaway
The IQA literature and this project's own fidelity gate agree. Synthetic distortions transfer to real defects only when
the simulated process is physically the same as the real one: here, decoder corruption (花屏), which passed. Graded,
content-dependent defects (过度生成, 虚化突兀, mid-grade 模糊/过锐) cannot be simulated with a faithful grade. The
project has closed every such recipe on evidence, and I found no untested recipe with real evidence of a score gain.
Expected value for the score is below 0.2 points; the lever is **closed for score** and open only for robustness.

### Cited Findings
- Synthetic distortions are usually applied globally, while authentic distortions are often non-uniform. This mismatch is
  given as the reason models trained on synthetic data underperform on real-world images — [Learning from Synthetic Data for Opinion-free BIQA in the Wild](https://ar5iv.labs.arxiv.org/html/2106.14076)
- Recent synthetic-to-authentic methods report modest gains. SynDR-IQA (trained on KADID-10k, tested on authentic sets)
  improves average SRCC/PLCC by 2.7 % over the next best method, DGQA. It is slightly behind FreqAlign on KADID→KonIQ SRCC
  and behind Q-Align on KADID→LIVEC PLCC — [SynDR-IQA, arXiv 2601.00225](https://arxiv.org/pdf/2601.00225)
- SLIDE-IQA pretrains only on synthetic degradations to learn degradation patterns rather than content, and is evaluated on
  four authentic and four synthetic benchmarks — [SLIDE-IQA, arXiv 2606.29162](https://arxiv.org/pdf/2606.29162)
  (preprint; tables not verified).
- Humans' accuracy drops from 83.6 % when identifying the distortion type to 69.9 % when judging severity, so graded
  labels carry inherent ambiguity that simulation cannot remove — [DistortBench](https://arxiv.org/abs/2604.19966)
  (as cited in REVIEW §4; not re-fetched).
- NTIRE 2025 KwaiSR describes the same defect classes as this task (face deformation, "AIGC-style textures" from
  generative enhancement) — [KwaiSR](https://arxiv.org/abs/2504.15003) (as cited in REVIEW §4).

Project record:
- **Behavioural probe** (SYNTHLAB §8). Blur, motion blur, codec loss and USM/Laplacian sharpening are already detected in
  order, with ≥95 % monotone ladders (Gaussian σ 0.7→3.5 called 模糊 8→97 %; x264 crf 30→44 called 纹理损失 57→97 %). So
  there is nothing to teach for realistic blur, compression or sharpening.
- **The two blind spots fail the real-data relevance check.** Resolution loss (fine-detail measure AUC 0.57) and
  edge-preserving smoothing (AUC 0.49–0.55) are not what annotators label, so teaching them would create false alarms.
- **Benign-processing invariance** (`benign1`, `baug`, `tw-benign1`; SYNTHLAB §7). Null to harmful in distribution and
  harmful on unseen pipelines (+0.0022 to +0.0055 mean RPS).
- **过度生成**
  - The FLUX ladder (`og`) was not recognised (P 0.16–0.25) and scored −0.40 (LEDGER 2026-11-11).
  - GAN restoration (`ogsr1`): fidelity 23.7 % vs 3.1 % on matched negatives. Target RPS −0.00004 (n.s.) (SYNTHLAB §10–11).
  - Restoration idempotence features were null (SYNTHLAB §12).
  - Ranking ladders `rank` / `rank25` scored −0.72 / −0.90 (LEDGER).
- **虚化突兀** (REASSESS §10.7; SYNTHLAB §14). Depth bokeh was unfaithful. `bkseg1` (Mask R-CNN subject mask) had
  fidelity 12.1 % vs 8.3 % and a null target Brier (−0.00046, CI90 [−0.0043, +0.0032]).
- **花屏 decoder corruption** (`glitch2`, with matched undamaged twins; REASSESS §6). Held-out family AUC 0.885 → 0.9999,
  no harm in distribution, argmax 49.37 = 49.37. **Adopted as robustness only** (`primary_v2`). `primary_v2` is −0.55 on
  first-batch draws, partly a recipe artefact (REASSESS §11.8).

Why the attempts failed. The loss is grade confusion, not detection: 78–92 % of errors are one grade off, and
过度生成 detection AUC is already about 0.96 (REASSESS §2.5; SYNTHLAB §11). Synthetic recipes can give "present, grade
unknown" or a recipe-assigned grade. Neither matches the annotators' grading convention, and the confident errors are
partly label-convention choices (REASSESS §10.1: 25 of 86 过度生成 false alarms carry 虚化突兀 P0 instead).

### Inferences
- The fidelity condition that worked here matches the one the literature names: a generative process identical to the
  real one (bitstream corruption), plus matched negatives that remove the source shortcut. v1 without negatives failed
  the no-harm check with 7 false alarms. No other defect in the taxonomy has such a process available.
- One recipe remains untested: localised generative edits (diffusion inpainting of small regions such as captions, faces
  or sky) for the "small local invention" 过度生成 P0 misses (REASSESS §10.1). Even if faithful, it targets only 12 of 190
  P0 frames called clean. A P0 label is worth about 2.79 points of F1 per unit (REASSESS §2.5), so recovering a few of
  those 12 frames is worth ≲0.1–0.2 points. Pre-screen-test risk is high (fidelity has failed 3 times for this defect).
  Cost would be about 3 h on 2 GPUs, as for E3. **Not recommended**; it is mentioned only as the one untested variant.
- Expected score gain from any further synthetic recipe: <0.2 points. Closed.

### Gaps
- I found no published study that trains graded (multi-level) real-world enhancement artefacts from synthetic data and
  reports a gain on human-graded labels. The literature reports MOS correlation, not ordinal-grade F1.
- SLIDE-IQA and SynDR-IQA numbers come from abstracts and snippets; full tables were not verified.

---

## Q2. Domain-adaptive sample weighting, importance weighting, hard-example mining / focal re-weighting, curriculum

### Takeaway
Theory and large-scale empirical work both predict small or vanishing effects for re-weighting in over-parameterised
networks trained to near-zero loss, and curricula help only with noisy labels or a limited budget. The project has
already measured the two endpoints of batch weighting, first batch only versus both batches, and they are equal after
ensembling. Any intermediate weight is therefore expected to be about 0. Focal-type and hard-example emphasis would
up-weight exactly the ambiguous mid-grade labels. Expected gain is below 0.2 points; **closed**, with one cheap
noise-oriented variant optional.

### Cited Findings
- Importance weighting has a strong effect early in training, but the effect fades over successive epochs once the
  network can separate the training data. L2 regularisation and batch normalisation partly restore it; dropout does not.
  The authors call the standard use of importance weighting in deep networks into question —
  [Byrd & Lipton, ICML 2019](https://proceedings.mlr.press/v97/byrd19a.html)
  ([arXiv](https://arxiv.org/abs/1812.03372))
- In over 25,000 models on CIFAR10/100, FOOD101 and FOOD101N, curricula give little or no gain in standard training;
  random order is as good or better. Curricula help with a limited training-time budget or with noisy labels —
  [Wu, Dyer & Neyshabur, ICLR 2021](https://arxiv.org/abs/2012.03107)
- Plain group balancing by re-sampling matches specialised methods for worst-group accuracy —
  [Idrissi et al.](https://arxiv.org/abs/2110.14503) (as cited in REVIEW §4).

Project record:
- **Training loss** drops 1.00 → 0.06 by epoch 8 (REVIEW §2.4). This is the separable regime in which Byrd & Lipton find
  that weighting effects vanish.
- **Batch weighting endpoints** (REASSESS §11.1, §11.5–11.7). Same-recipe first-batch-only vs both-batch towers on
  first-batch rows:
  - pe: +0.18 / +0.05 decoded (n.s.) over two seeds.
  - cradio: −0.61; siglip: −0.26 (n.s.).
  - Ensemble with all three tier-1 towers swapped: 51.69 vs 51.57 (+0.12, CI90 [−0.23, +0.54]).
  - First-batch-only twins added as extra members: −0.19 decoded. Per-batch decoder: −0.11.
- **Group up-weighting** (`boost`, clean+processed rows ×8): rejected (LEDGER, REVIEW §9). Clean+processed flagged
  58 % → 52 % (n = 31, about 2 images); grade AUC −0.0040 (CI90 [−0.0071, −0.0009]).
- **Rarity weighting** is already in the loss (fixed class weights 1.65–1.90).
- Older code paths used asymmetric focal BCE (`docs/TECHNICAL_REPORT.md` §L90). That was not a controlled comparison
  under the current protocol.
- **Duplicated hard-labelled rows hurt.** `benign1` extra rows: in-distribution RPS +0.0021 (CI excludes 0), 过度生成 F1
  0.46 → 0.39 (SYNTHLAB §7).
- **Residual grading errors have no content structure.** Clustering explains no more variance than a random partition
  (REASSESS §7 item 1), so there is no "hard content cluster" for mining to target.

### Inferences
- **Domain weighting** of supplement vs first batch: ≈0 points. Both endpoints are measured and equal within noise, and
  the per-tower effect is mixed in sign. Closed.
- **Hard-example mining / focal.** With label noise concentrated at mid-grade boundaries, up-weighting high-loss examples
  amplifies noisy labels. The CIFAR-N evidence (Q4) is that real human noise is instance-dependent and is exactly what
  such methods mis-handle. Expected ≤0 points. Closed.
- **Curriculum.** The one condition with evidence, noisy labels, applies here. However, the closest project analogue,
  cross-fitted soft labels (`xd`), which also de-emphasises noisy hard labels, helped one tower (RPS −8 %) and vanished in
  the ensemble. A curriculum or small-loss down-weighting variant would most likely behave the same: <0.1 points pooled.
  - Optional cheap test: one pe tower, folds 0–1, 3 seeds vs the 3-seed control (the SYNTHLAB gate). Cost about 6 tower
    runs, roughly 2–3 GPU-hours. The bar would be a pooled RPS gain, not a single-tower gain.

### Gaps
- No published study on importance or domain weighting for ordinal IQA with two release batches was found. The transfer
  from Byrd & Lipton (classification, separable data) is an inference.

---

## Q3. Pseudo-labelling / self-training without test images

### Takeaway
Self-training works here only on frames from the same sources and videos as the labelled data. `ssl` recovered 25–32 %
of the label value. KVQ (other sources) gave nothing, and nothing would come from other public video corpora. Test and
practice images are excluded by AGENTS.md rule 2, and the owner has said no more data is available. So **no legitimate
in-domain unlabelled pool currently exists in the repo**. If the owner could export unlabelled frames of the first-batch
videos, the plausible pooled gain is about +0.1 to +0.4 points. That figure is an estimate; it has not been measured
inside the ensemble.

### Cited Findings
- Noisy Student: an un-noised teacher pseudo-labels unlabelled images and a noised student learns from labels plus soft
  pseudo-labels; soft labels are safer for out-of-domain data — [Xie et al.](https://arxiv.org/abs/1911.04252)
  (as cited in REVIEW §4). An NR-IQA instance of this placed third at NTIRE 2022 —
  [Conde et al.](https://arxiv.org/abs/2204.12819) (as cited in REVIEW §4).
- CReST observes that SSL methods perform poorly on minority classes but still produce high-precision pseudo-labels for
  them. Re-sampling pseudo-labelled minority examples improves FixMatch by up to 11.8 % on imbalanced benchmarks, with gains
  mainly in minority-class recall — [CReST, CVPR 2021](https://arxiv.org/abs/2102.09559)
- NTIRE 2024 KVQ challenge: top teams used ensembles plus external pretraining; no team reported pseudo-labelling —
  [report](https://arxiv.org/abs/2404.11313) (as cited in REVIEW §4).

Project record:
- **`ssl`** (LEDGER 2026-11-11; REVIEW §9). Half A labelled plus half B with teacher soft labels: argmax 46.69 → 47.34
  vs 48.74 with all labels; RPS 0.05460 → 0.05365 → 0.05101. This is 25–32 % recovery on same-distribution frames, for a
  single tower.
- **KVQ self-training** (`kvq`, `shiftkvq`, `kvqcert`): no in-distribution gain (RPS −0.0002, n.s.) and none on unseen
  pipelines. It was harmful where the teacher is blind (花屏 family RPS 0.138 → 0.161). Rejected (LEDGER 2026-11-12).
- REASSESS §3 rules out "pseudo-labelling by our own models on frames from other sources". The E5 ranking tool found
  KVQ to be the wrong pool for the rare defects (REASSESS §6).
- **Sibling exposure** (training on other labelled frames of the same video) gives RPS −10 % (seed 2026) and −8.1 %
  (seed 2027) on held-out sibling frames. The effect is already included in the full-fit models (REASSESS §11.1, §11.5).
- REASSESS §11.4 item 3: unlabelled test frames would fit the `ssl` mechanism (+0.1 to +0.3, not measured) but violate
  rule 2. The legal alternative is unlabelled frames exported from the first-batch videos, whose video ids are in the
  file names. This needs the owner, who has said "no more data on my side" (§11 header).

### Inferences
Legitimate pools, ranked:
1. **Other frames of first-batch videos, exported by the owner.** This is the same mechanism as `ssl` plus sibling
   exposure, and it is the only pool with evidence.
2. **Frames of production pipelines not in the labelled set.** These help robustness only if the teacher already knows
   the defects there. The `shiftkvq` evidence says it does not.
3. **Public corpora (KVQ, Inter4K, KonIQ).** Tested or equivalent: null.

Estimate for pool 1:
- Halving labels costs about 2 points in both single towers and two-seed ensembles (REASSESS §2.4).
- If the unlabelled frames equal the labelled first-batch count and recover 25–32 % of the label value, a single tower
  gains about 0.5.
- Inside the ensemble the gain should shrink but not vanish. This mechanism adds new images, unlike `xd`, which only
  reduced variance and vanished. Pooled estimate: +0.1 to +0.4 points.
- Concentrating pseudo-labels on rare positives (CReST-style resampling) is the variant with literature support for
  minority recall, i.e. 过度生成 and 虚化突兀, the data-limited defects.
- Risk: medium. A teacher must be cross-fitted per fold to avoid leakage, as in the existing `ssl` code. Frames 50–400 s
  apart are different pictures (median thumbnail correlation 0.27), so diversity is real.
- Cost: retraining the three tier-1 towers × 8 folds plus full fit. The tier-1 full-fit of 8 models took 10,770 s on
  2 GPUs (LEDGER), so the total is on the order of tens of GPU-hours.
- **Blocked on data access.** Without new frames this lever is closed.

Mean teacher, FixMatch-style consistency and EMA teachers add no information beyond the soft-label variant already
tested. Under the evidence here the gain comes from new in-domain images, not the SSL algorithm. Expected difference
between SSL algorithms: <0.1 points.

### Gaps
- The `ssl` recovery was measured for one tower, not the pooled ensemble. The pooled range above is extrapolated.
- How many unlabelled frames exist per first-batch video is unknown from the repo.

---

## Q4. Losses: alignment with Doc2_Strict / RPS; ordinal, metric-surrogate, noise-robust, EMA/SWA, multi-teacher distillation

### Takeaway
The training objective is already ordinal and RPS-aligned: Gaussian soft ordinal targets, squared-EMD/RPS, adjacent-grade
ranking, rarity weights, and averaging of the last epochs. Thresholds for Doc2_Strict are fitted exactly by the decoder,
and in-sample decoder headroom is only about 0.4. Swapping in CORN/CORAL, sigmoidF1/Lovász or noise-robust losses
therefore changes calibration and variance, the mechanism the ensemble already absorbs. Each is expected to be worth
<0.2 points. The one untested loss-side item with a concrete use is **cross-fitted multi-teacher distillation of the pool
into one or two towers**. Its value is cost and deployability (the 20 cost points), not objective score: expect about
−0.3 to −1.0 points against the full ensemble.

### Cited Findings
**Ordinal losses**
- CORN replaces CORAL's weight-sharing constraint with conditional binary tasks; its loss is the sum of K−1 binary
  cross-entropies over conditional subsets. Reported gains are in MAE/RMSE on age-type datasets versus classification,
  OR-CNN and CORAL — [Shi, Cao & Raschka, arXiv 2111.08851](https://arxiv.org/abs/2111.08851v5)
  ([secondary summary](https://aclanthology.org/2026.aimecon-main.55.pdf)). Exact head-to-head numbers were not retrieved.
- Unimodal output distributions beat a cross-entropy baseline on Adience top-3 accuracy, even in the weakest variant —
  [Beckham & Pal, ICML 2017](https://arxiv.org/pdf/1705.05278)
- Plain cross-entropy does not yield unimodal ordinal posteriors, which motivates calibration-plus-unimodality losses —
  [ORCU, arXiv 2410.15658](https://arxiv.org/pdf/2410.15658)
- Class-distance-weighted CE is benchmarked against CE, MSE, CORN and others on one medical ordinal dataset —
  [Polat et al., arXiv 2412.01246](https://arxiv.org/html/2412.01246v1)
- Exact squared EMD as a loss for ordered classes gives state-of-the-art results on age estimation and aesthetics at
  little extra cost — [Hou, Yu & Samaras, arXiv 1611.05916](https://arxiv.org/abs/1611.05916v1)

**Metric surrogates**
- sigmoidF1 approximates F1 for SGD training in multilabel classification and reports outperforming other losses on text
  and image datasets. Dataset counts differ between versions —
  [Bénédict et al., TMLR 2022](https://arxiv.org/abs/2108.10566v3)

**Noise-robust losses**
- On CIFAR-N, real human label noise is instance-dependent. State-of-the-art robust methods show a significant gap between
  synthetic class-dependent noise and real human noise, especially on CIFAR-10N —
  [Wei et al., ICLR 2022](https://arxiv.org/abs/2110.12088)
- Ordinal label-noise correction and early-learning regularisation report large gains on synthetic noise and weak ones on
  real noise — [ELR](https://arxiv.org/abs/2007.00151), [ORDAC](https://arxiv.org/abs/2509.02351) (as cited in REVIEW §4).

**Distillation**
- Knowledge distillation improves student generalisation, but a large teacher–student discrepancy often remains even when
  the student has capacity to match. The best fidelity policy reached only about 85 % agreement, and the best policies for
  accuracy differ from those for fidelity — [Stanton et al., NeurIPS 2021](https://arxiv.org/pdf/2106.05945)
  ([poster](https://izmailovpavel.github.io/files/gnosis/poster.pdf))
- High student accuracy need not mean high fidelity in self and ensemble distillation —
  [Guo et al., arXiv 2405.00739](https://arxiv.org/html/2405.00739v1)
- A student trained on an ensemble's outputs can inherit the ensemble's gain —
  [Allen-Zhu & Li](https://arxiv.org/abs/2012.09816) (as cited in REVIEW §4).

**Project record**
- **Loss composition**: see the header of this file (`unimodal_xbm_losses.py` ~L448–498). Soft ordinal σ = 0.35 is an
  ordinal form of label smoothing. RPS (the gate metric) is directly in the loss at λ = 0.45. Ranking has been in the loss
  since Tier-1.
- **CORN** was used in older pipelines: the V13 VLM with "soft CORN" (`docs/D1A437A_FINAL_RESULT_REPORT.md` §3: s16–s20
  mean 47.19), and CORN towers in older reports. No controlled CORN-vs-current comparison exists under the R55 protocol.
- **Decoder.** The multi-start decoder optimises thresholds directly for Doc2_Strict. In-sample it reaches 51.87 vs
  nested 51.45 vs argmax 51.28, so at most about 0.4 is lost to threshold estimation (REASSESS §11.1). Five decoding rules
  do not differ significantly (REASSESS §8).
- **Cross-fitted soft labels (`xd`).** Single tower RPS 0.0510 → 0.0468 (−8 %), NLL 0.583 → 0.429, making it equal to
  V16. Inside the ensemble: 0.04192 → 0.04201 (slightly worse). Not adopted for the ensemble; "recipe of choice for a
  single small model" (LEDGER; REVIEW §9).
- **Weight averaging and checkpoint selection.**
  - `--select fixed` (mean of last 2 epochs) vs val: +0.14 (n.s.) (LEDGER P1).
  - Three-seed averaging of the pe member: tower −5.7 % RPS, primary pool −0.09 % (REASSESS §10.8). The pool has already
    absorbed member noise, so EMA/SWA variants have nothing left to remove.
  - Best-epoch selection optimism is about 0.5 per tower (REVIEW §2.4), and the `fixed`-recipe tier-1 family is 0.6
    below the deployed members (REASSESS §11.7). This is a measurement issue, not a lever.
- **Earlier teacher distillation.** Phoenix stage 2 (OOF teacher distillation) moved pillars 2 and 3 by +0.6 and +0.8,
  but its 5-fold OOF teacher leaks validation labels at second order (REVIEW §4).
- **Fallback without the VLM member**: 51.35 OOF at 14.8 GB / 3.7 s per image, vs `primary` 51.45 at 20.4 GB with 4-bit
  quantisation. The cost-tier gap of 2–14 points dwarfs the objective difference (REASSESS §11.8).

### Inferences
- **Misalignment.** Little remains: the loss already targets RPS and ordinal smoothness, and the decoder targets
  Doc2_Strict exactly.
  - The one structural mismatch is that macro-F1 weights rare P2/P1 labels such as 过度生成 equally with common ones.
    Fixed rarity weights only partly address this. Because the decoder re-thresholds each label, the residual is ranking
    quality (adjacent-grade AUC), which loss re-weighting barely changes (Q2).
- **Ordinal losses (CORN/CORAL/EMD).** The current loss already includes EMD², and the literature shows gains mainly in
  MAE relative to plain CE, not relative to a soft-ordinal + EMD² composite. Pooled gain <0.1 points. Closed.
- **sigmoidF1 / Lovász-type surrogates.** Their benefit in the literature comes largely from implicit threshold and
  label-count handling, which the decoder already does exactly on OOF. Remaining headroom is ≤0.4 in-sample and
  noise-level when nested. Gain <0.2. Risk: these losses destabilise calibration, and RPS-gated decisions would penalise
  them. Closed.
- **Noise-robust losses / more label smoothing.** CIFAR-N shows weak gains on real human noise, and the project's
  strongest noise-handling variant (`xd`) vanished in the pool. Gain <0.1 pooled. Closed.
- **EMA/SWA.** Closed by the seed-bagging result: −0.09 % pool RPS for three full seeds, which is more averaging than any
  EMA gives.
- **Multi-teacher distillation into a single student.** This is the one testable item. Its use is a cheaper engine, not
  a higher objective score.
  - Hypothesis H-KD: a pe or cradio tower trained on all labels plus **cross-fitted pooled-ensemble soft labels**, using
    the `xd` scheme (half-A/half-B ensemble teachers so that no teacher saw the row or the validation fold), reaches pooled
    RPS of about 0.044–0.046.
  - For reference: `xd`-pe alone 0.0468, fallback 0.04220, primary 0.04192. As one model or a two-model engine this would
    sit about 0.3–1.0 points below `primary`.
  - Literature (Stanton et al.) predicts imperfect fidelity, so matching the pool is unlikely.
  - Benefit: an unquantised single-4090 engine of roughly 1–2 GB, which could move the cost tier.
  - Risk: medium. Second-order leakage occurs if the pooled OOF is used directly as the teacher; it must be cross-fitted.
  - Cost: ensemble teachers on two halves (6 members × 2 halves × 8 folds) are the dominant cost, likely 1–3 GPU-days.
    A cheaper approximation reuses the existing half-A/half-B pe teachers plus Phoenix/V16 OOF only where leakage-safe.
  - Objective score: expected negative versus `primary`. Worth running only if the owner values the cost points.

### Gaps
- No primary-source table comparing CORN/CORAL to an EMD² + soft-ordinal composite was found.
- sigmoidF1's exact gains per dataset were not retrieved (abstract-level only).
- No published result on ensemble-to-single-student distillation for NR-IQA grade classification was found.

---

## Q5. Realistic ceiling given noisy labels

### Takeaway
The noise cannot be measured from this data: only 13 near-identical pairs exist, and sibling frames are genuinely
different pictures. Several bounds and analogies converge, though:
- 52.25 ± 0.56 for first-batch-like 654-image draws with the current models;
- about 54.4 if every graded defect reached the F1 of the best-learned, saturated defect;
- a pessimistic sibling-based macro-F1 bound of 0.73 against the model's 0.63.

A realistic reachable range with the released labels is **52–53.5**. 55 is not reachable by data-side or loss-side
levers; only new same-source labels have shown multi-point value (about +0.8 per doubling of rare-defect positives).

### Cited Findings
- Human severity judgement is markedly less reliable than type identification (83.6 % → 69.9 %) —
  [DistortBench](https://arxiv.org/abs/2604.19966) (via REVIEW §4).
- Real human annotation noise is instance-dependent; the CIFAR-10N aggregate label has a 9.03 % noise rate, the random
  label about 18 %, and the worst label 40.21 % — [CIFAR-N, Wei et al.](https://arxiv.org/abs/2110.12088)

Project record:
- **Sibling consistency.** Two frames of one video agree on all six defects in 44 % of pairs, and on 76 / 65 / 77 / 86 %
  of 模糊 / 纹理损失 / 过锐 / 过度生成 grades. The ensemble agrees with the frame's own label at 80 / 77 / 80 / 85 %
  (REASSESS §11.3).
- **Sibling "√ bound"**: macro F1 0.728 vs model 0.634. It is pessimistic, and the model already exceeds it on P0
  (REVIEW §2.3). Averaging over siblings lowers the score by 2.06, so sibling disagreement is not pure noise
  (REASSESS §2 item 6).
- **Rating spread.** Published subjective studies put the spread of individual ratings at about 0.6 of a grade at
  mid-scale (SOS ≈ 0.08–0.11), which caps exact-grade agreement near one half. This is an analogy, not a measurement
  (REASSESS §7 item 2).
- **Saturated-defect ceiling.** 纹理损失 reaches F1 0.65 / 0.65 / 0.78. If every graded defect matched it, the total
  would be about 54.4 (REASSESS §11.4).
- **Points still lost per defect**: 过度生成 3.55, 过锐 2.44, 模糊 2.31, 纹理损失 1.86, 虚化突兀 0.46, 花屏 0.05
  (REASSESS §2.5).
- **Learning curve** (REASSESS §2.4). Per doubling of labels: 模糊 / 纹理损失 / 过锐 +0.003 to +0.017 F1; 过度生成
  +0.068, 虚化突兀 +0.076.
- **Practical range.** First-batch draws give 52.25 ± 0.56, P(≥54) = 0.2 %, P(≥55) = 0. The honest centre for a
  first-batch-like final set is 51.6–52.3 (REASSESS §11.2, §11.7).

### Inferences
- With 模糊, 纹理损失 and 过锐 saturated, label noise or convention is the most plausible limiter for them. Those three
  defects hold 6.6 of the lost points, which no loss trick has moved.
- The remaining data-limited defects (过度生成, 虚化突兀) respond only to real labels.

Summary of lever values (points on the 65-point objective, pooled ensemble):

| Lever | Project status | Expected gain | Risk | GPU cost | Verdict |
|---|---|---|---|---|---|
| Further synthetic recipes (blur / codec / sharpen / bokeh / generative) | 9 recipes tested; only 花屏 passed (robustness) | <0.2 | high | ~3 h × 2 GPU per screen | closed |
| Localised inpainting for 过度生成 P0 small inventions | untested | ≲0.1–0.2 | high (fidelity) | ~6 GPU-h | not recommended |
| Batch/domain weighting | endpoints measured equal | ≈0 | low | — | closed |
| Focal / hard-example mining | `boost` rejected; rarity weights present | ≤0 | medium | — | closed |
| Curriculum / small-loss down-weighting | untested; `xd` analogue vanished | <0.1 | low | ~3 GPU-h screen | optional only |
| Self-training on unlabelled first-batch-video frames | `ssl` 25–32 % recovery (single tower) | +0.1 to +0.4 | medium | tens of GPU-h | **only open data lever; blocked on owner export** |
| Self-training on other-source pools (KVQ etc.) | rejected | 0 | — | — | closed |
| SSL algorithm choice (FixMatch / mean teacher vs soft labels) | — | <0.1 | — | — | closed |
| CORN / CORAL / other ordinal heads | legacy only, no controlled test | <0.1 | low | — | closed |
| sigmoidF1 / Lovász surrogate | untested; decoder already optimises F1 | <0.2 | medium | — | closed |
| Noise-robust losses / more smoothing | `xd` null in pool | <0.1 | low | — | closed |
| EMA / SWA | seed bag −0.09 % pool RPS | ≈0 | low | — | closed |
| Cross-fitted ensemble → single-student distillation | untested at pool level | −0.3 to −1.0 vs `primary` (for cost, not score) | medium (leakage) | 1–3 GPU-days | only if cost points matter |
| New same-source labels for 过度生成 / 虚化突兀 | measured | ≈+0.8 per doubling | — | — | owner-dependent |

### Gaps
- The true inter-rater noise of this dataset is unmeasured. A small re-annotation study (the same frames graded twice)
  is the only way to turn the ceiling from an analogy into a measurement (REASSESS §7).
- The pooled effect of in-domain self-training has not been measured.
