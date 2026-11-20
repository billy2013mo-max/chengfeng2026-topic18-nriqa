# 命题18 — critical reassessment and new plan (2026-11-12)

Scope: what the completed experiments (`docs/REVIEW_AND_ROADMAP_2026-11-11.md` §9–§10, `results/next55/LEDGER.md`)
actually show, which of my earlier assumptions they refute, what that leaves, and a new plan. Numbers are nested
out-of-fold (OOF) on the 3,573 labelled images, metric Doc2_Strict (65 max = 25 positive + 20 P0 + 20 clean), unless
stated. RPS = ranked probability score of the pooled probabilities (lower is better). Nothing here uses practice-set
labels; the one online number (`s1` = 51.23) is used only as a check of the offline protocol.

## 1. Verdict on the previous plan

The plan of 2026-11-11 assumed that single-model training changes (unlabelled in-domain frames, soft labels, crops,
rebalancing) would each add tenths of a point. **That assumption was wrong.** Ten such changes were tested; none
improved the ensemble. What did improve it was not in the plan's core: pooling unrelated model families (+0.4–0.6).

| Result | Evidence | Status |
|---|---|---|
| Cross-family ensemble, family-balanced, one seed per model | RPS 0.04298 → 0.04192, CI excludes 0; OOF 51.04 → 51.45 | **worked**, reproducible |
| Offline protocol predicts online | `s1` online 51.23 vs offline 51.04 (140-image subsets 50.6 ± 1.8) | **confirmed** |
| Deployment (bf16, merged 4-bit pillar 1) | 19.3 GB, 5.5 s per image, grades unchanged | **worked** |
| Random crops, crop averaging, clean boosting, ranking ladders, tag head, generated 过度生成 ladder | single-tower null or negative | failed |
| Cross-fitted soft labels (`xd`) | single tower RPS −8 %, ensemble 0.04192 → 0.04201 | failed for the ensemble |
| Pseudo-labels on in-domain frames (`ssl`) | recover 25–32 % of the label value on same-distribution frames | worked only as a measurement |
| KVQ self-training (`kvq`, `shiftkvq`, `kvqcert`) | no RPS gain in distribution, none on unseen pipelines | failed |
| Second seeds, learned weights/temperatures, per-defect family weights | ΔRPS ≤ 0.0001 | no value |

## 2. Why they failed — what the evidence supports

New diagnostics run for this reassessment (all CPU, all OOF) are marked ◆.

1. **Everything that reduces the variance of one model is already done by the ensemble.** Calibration, crop noise,
   label-noise overfitting: `xd`, `rcrop` and crop averaging improve a single tower and vanish in the pool.
   *Evidence-supported.*
2. **The model families do not have different skills.** ◆ Per-defect skill is the same for every family
   (过度生成 adjacent-grade AUC 0.70–0.75 for all seven members; pool 0.74). ◆ Per-defect family weights fitted in
   a nested loop gain ΔRPS −0.00004. ◆ 61 % of the pool's cell errors are wrong in all three families; in the
   other 39 % a family is right only by chance (an oracle that picks the right family would score 56.9, but nothing
   learnable predicts which one). So more families of the same kind buy variance reduction only (≈−0.0004 RPS
   each, falling). *Evidence-supported.*
3. **Pseudo-labels cannot add what the teacher does not know.** `ssl` works on frames from the same sources (the
   student sees new *images*); KVQ frames come from other sources where the teacher is no better than the student,
   and where the teacher is blind (花屏 on a family that lacks it) self-training makes it worse.
   *Evidence-supported.*
4. **The data limit is concentrated in the rare defects.** ◆ Doubling the labels (pe_core_large, half → all):

   | Defect | positives | mean label F1 half → all | RPS change |
   |---|---|---|---|
   | 模糊 | 1,711 | 0.564 → 0.581 (+0.017) | −2 % |
   | 纹理损失 | 2,803 | 0.666 → 0.679 (+0.014) | −4 % |
   | 过锐 | 1,283 | 0.531 → 0.533 (+0.003) | −5 % |
   | 花屏 | 213 | 0.949 → 0.966 (+0.017) | −41 % |
   | 过度生成 | 684 | 0.353 → 0.421 (**+0.068**) | −8 % |
   | 虚化突兀 | 225 | 0.738 → 0.815 (**+0.076**) | −24 % |

   The three common defects are close to saturated in data; the "≈2 points per doubling" comes from 过度生成,
   虚化突兀, 花屏 and from clean images (other-clean false alarms 5.8 % → 2.7 %). A two-seed ensemble shows the
   same slope (47.5 → 49.5). *Evidence-supported for one backbone; assumed for the full ensemble.*
5. **Where the points are lost** (primary ensemble; one label = 1.79 points of F1, P0 labels 2.79):
   过度生成 3.55 (F1 0.32 / 0.27 / 0.63), 过锐 2.44 (0.53 / 0.48 / 0.76), 模糊 2.31 (0.48 / 0.52 / 0.82),
   纹理损失 1.86, 虚化突兀 0.46, 花屏 0.05. 78–92 % of errors are one grade off.
6. **A claim I made that the evidence does not support.** I wrote that the ceiling is "label consistency",
   using frames of the same video as repeated labels. ◆ Averaging the model's own probabilities over the frames of
   a video *lowers* the score on those 844 images by 2.06 (CI90 [−2.95, −1.19]): frames of one video genuinely
   differ, so sibling disagreement is not label noise and the "sibling bound" is not a ceiling. The size of the
   label noise is **unknown**. What remains true: for 过度生成 P2/P1 another frame's human label predicts the
   grade better than the whole ensemble (F1 0.48 / 0.34 vs 0.43 / 0.17 on those images), so that defect has a
   video-level component the models miss. *Hypothesis*: it is the strength of the enhancement applied to the video.
7. **Unseen sources** (shift test): tagged pipelines +10 % RPS; 209 of 211 花屏 positives come from one source and
   detection collapses without it (AUC 0.995 → 0.885); processed-but-clean frames are flagged 32 % of the time
   (31 examples). None of the tested ideas moved these. *Evidence-supported.*

## 3. What the negative results rule out

Not to be repeated without a new, specific hypothesis:
- any change whose mechanism is variance reduction, regularisation or calibration of one tower;
- more seeds, more backbones of the kinds already present, learned stackers/gates of any granularity tested;
- pseudo-labelling by our own models on frames from other sources;
- synthetic *generative* defects labelled by our own teachers;
- video-level aggregation (hurts; and the practice set has no sibling frames ◆);
- tuning the decoder on the practice set (offline it only trades positive score for clean score one for one).

What is left has to bring **information the current models do not have**: real labels (ours or related human
labels), a different kind of evidence, or training examples of situations absent from the data.

## 4. Alternatives

Each is judged by the same test: does it add information the pool lacks, and can a cheap controlled experiment
refute it?

### A. External human supervision through pretrained IQA experts (new)
- **Mechanism.** NR-IQA models trained on hundreds of thousands of human opinion scores (KonIQ, SPAQ, FLIVE, PIPAL,
  KADID, AGIQA…) encode severity judgements from data we do not have. Used frozen, as extra features.
- **Evidence for.** The grades already track independent human MOS (Spearman 0.74 on KVQ) but not perfectly, so MOS
  carries related, not identical, information. **Against:** blur/texture are data-saturated (item 4), so the upside
  is mainly in 过锐 and under source shift.
- **Limits.** Different label type; photo datasets rather than short-video frames.
- **Test (E1).** Score all labelled images with ~10 pretrained metrics; grouped nested gradient boosting per
  defect on those scores alone and on scores + pool probabilities. Cheap, no training of ours.

### B. Generative-prior evidence for 过度生成 (new)
- **Mechanism.** 过度生成 is the biggest loss, data-limited, and every discriminative family plateaus at the same
  AUC. Output of a generative enhancer lies closer to a generative model's manifold than camera texture does; the
  reconstruction error of a latent autoencoder (FLUX AE, local) measures that (the AEROBLADE idea).
- **Evidence for.** Only indirect: a video-level component the models miss (item 6). **Against:** the production
  enhancer may not be a latent-diffusion model.
- **Test (E2).** AE encode–decode at native resolution, error statistics per frequency band and region → nested
  boosting for 过度生成 → does it add to the pool on that defect?

### C. Simulate what can be simulated faithfully: decoding corruption (new)
- **Mechanism.** 花屏 is decoder corruption (smeared macroblocks). Unlike generative defects it can be produced
  exactly, by corrupting the bitstream of real short videos (KVQ) and decoding. This gives 花屏 on many sources.
- **Evidence for.** The shift test shows the failure it would fix; contact sheet of 花屏 positives confirms the type.
- **Limits.** No gain expected on OOF or the practice set (花屏 F1 is 0.98 there); the gain is robustness.
- **Test (E3).** Shift fold that holds out the 花屏 family, with vs without simulated 花屏 in training.

### D. Real MOS labels as an auxiliary task (KVQ), not pseudo-labels (new, conditional)
- **Mechanism.** 2,926 KVQ videos carry human scores across many processing pipelines: real supervision on
  sources the labelled set lacks. A regression/ranking head on the shared trunk.
- **Evidence for / against.** Same as A; KVQ self-training failed because it carried no human information — this
  does. Run **only if E1 shows external human scores add information**; otherwise it is the same bet at 20× the cost.
- **Test (E4).** Shift folds: tagged-pipeline RPS with vs without the MOS head.

### E. Targeted labelling (the only lever with direct evidence)
- **Mechanism / evidence.** Item 4: doubling 过度生成 and 虚化突兀 positives is worth ≈+0.07 F1 on four labels
  (≈+0.8 points); processed-but-clean and unseen-pipeline frames have no other fix.
- **Limits.** Needs annotators and frames from the production pipelines; not something I can do from here.
- **What I can do (E5).** A ranking tool so that every labelled frame is spent where the ensemble is uncertain
  about 过度生成 grade, likely 虚化突兀, or processed-and-probably-clean.

### Considered and dropped
- Bigger or more VLMs: the 8B VLM pillars are no better at any defect (item 2).
- Per-image expert gating: nothing learnable in the disagreement (item 2).
- Content-controlled pairs across pipelines: no video appears under two pipelines in the data ◆.
- Processing-type pretext task on KVQ: file names are anonymised, no pipeline metadata ◆.
- Sending frames to an external vision API as a teacher: publishes internal data; not done without the owner.

## 5. Execution plan

Common rules. One hypothesis per experiment; control = the same code path without the change; decisions on RPS /
adjacent-grade AUC / label F1 with paired cluster bootstrap over videos; a result counts only if it survives
**inside the primary pool**; nothing is selected on the practice set. Every result is appended to the ledger.

| # | Experiment | Control | Success criterion | Stop condition | Cost |
|---|---|---|---|---|---|
| E1 | IQA-expert score bank → nested boosting, per defect | pool alone | pool + bank: ΔRPS ≤ −0.0003 with CI90 excluding 0, **or** adjacent-grade AUC of any defect +0.01 (P ≤ 0.05) | neither → drop A **and** D | ~1 h, 1 GPU |
| E2 | FLUX-AE reconstruction features → nested boosting for 过度生成 | pool alone | 过度生成 adjacent-grade AUC of pool + features ≥ +0.015 (P ≤ 0.05) | below → drop B | ~1 h, 1 GPU |
| E3 | Simulated 花屏 from corrupted KVQ bitstreams, shift fold 4 | `shift` fold 4 | 花屏 detection AUC on the held-out family ≥ 0.97 (now 0.885) with no loss elsewhere (ΔRPS other defects ≤ +0.001); and no change on two standard folds | AUC < 0.93 → drop C | ~3 h, 2 GPU |
| E4 | KVQ MOS head, shift folds (only if E1 passes) | `shift` | tagged-pipeline ΔRPS ≤ −0.002 (≈ a third of the gap), CI90 excluding 0 | no → drop D | ~3 h, 8 GPU |
| E5 | Labelling shortlist tool | — | delivered; owner decides | — | CPU |

Order: E1 and E2 first (cheap, decisive for the two "new information" hypotheses), E3 in parallel (independent,
addresses a measured failure), E4 only on E1's outcome, E5 whenever the GPUs are busy.

**Global stop rule.** If E1 and E2 are both null, no further modelling experiment on the score is justified by the
evidence: ship the current ensemble (plus E3 if it passes) and put the effort into E5. I will say so rather than
propose a sixth variation.

**What success would look like, honestly.** E1/E2 passing would add tenths of a point each, not points. The only
multi-point lever with evidence is labels for the rare defects (≈+0.8 per doubling of those positives) — and a
second doubling should be expected to give less.

## 6. Execution log

| Time (UTC) | Item | Status |
|---|---|---|
| 11-12 01:00 | E1 as planned (public pretrained IQA metrics via `pyiqa`) | **blocked**: the library is installed in `third_party/pydeps_iqa`, but running it downloads and loads third-party weights, which this session was not permitted to do. Needs the owner's go-ahead. |
| 11-12 01:05 | E1, local variant (`probe_mos_expert.py`): a KVQ-MOS regressor on frozen PE-Core-L / DINOv3-L features, tested as a feature bank against the pool | running. First fit leaked content across folds (KVQ has ~7 versions per source clip: frame-level SRCC 0.96); refit with folds grouped by source clip |
| 11-12 01:00 | E2 (`probe_ae_recon.py` + `probe_eval.py`): FLUX-AE reconstruction features, 59 per image | features done for 3,573 images; evaluation running |
| 11-12 01:15 | E3 (`build_glitch.py`): 1,151 simulated 花屏 frames from 900 KVQ videos (bit flips in key / inter frames, dropped inter frames, spliced key frame); presets `shiftglitch`, `glitch` | training: shift fold 4 (the held-out 花屏 family), shift fold 0 and standard folds 0–1 as no-harm checks |
| 11-12 01:25 | E5 (`rank_for_labelling.py`) | delivered; on the 800 scored KVQ frames it finds 24 过度生成-undecided and no 虚化突兀 candidates — KVQ is the wrong pool for the rare defects, frames from the production pipelines are needed |
| 11-12 01:40 | **E2 result** (`logs/r55/probes/ae_recon_eval.log`) | **failed, B dropped.** Reconstruction features alone: 过度生成 detection AUC 0.766, adjacent-grade AUC 0.62 / 0.60. Added to the pool: 过度生成 ΔRPS +0.0016 (worse, CI excludes 0), adjacent-grade AUC −0.009; nothing on any other defect. The generative-prior signal is weak and already contained in the pool |
| 11-12 01:45 | **E1 local variant result** (`logs/r55/probes/mos_expert_eval.log`) | **failed.** The KVQ-MOS expert predicts MOS at SRCC 0.82 on held-out source clips (0.96 was a content leak). As features it knows blur and texture loss (detection AUC 0.83 / 0.85) but adds nothing to the pool: mean ΔRPS +0.0002; best single effect 模糊 adjacent-grade AUC +0.0035 (bar: +0.01) |
| 11-12 01:50 | **E3 result, first version** (`shiftglitch`, `glitch`) | **main criterion passed, no-harm check failed.** Held-out 花屏 family: detection AUC 0.885 → **0.9985**, recall 0.36 → 0.97 at precision 1.00, argmax Doc2S on that family 40.5 → 43.3; shift fold 0 unchanged. But on standard folds 0–1: 7 花屏 false alarms on 872 images (0 before), mean RPS 0.0509 → 0.0531. The false alarms are blurry frames and screen-recording layouts: every simulated positive came from a re-encoded KVQ clip, so "looks like a re-encoded KVQ clip" became a shortcut |
| 11-12 02:00 | E3 second version (`shiftglitch2`, `glitch2`): each corrupted frame is paired with the same re-encoded window decoded without damage, labelled not-花屏 (1,152 + 1,152 frames) | training; same four checks |
| 11-12 01:55 | E1 as planned, after the owner's permission: 19 public metrics load in tensors-only mode (MUSIQ ×3, MANIQA ×3 incl. PIPAL and KADID, ARNIQA ×4, HyperIQA, DBCNN, TReS, NIMA ×2, PaQ-2-PiQ, NIQE, BRISQUE, PIQE); TOPIQ, the CLIP-based metrics, A-FINE and FGResQ do not load in this environment and were left out | scoring 3,573 images; evaluation follows |
| 11-12 02:40 | **E1 result, public metrics** (`logs/r55/probes/iqa_bank_eval.log`; 19 metrics × [whole frame, crop mean / min / max] = 76 features) | **failed; A and D dropped, E4 not run.** The bank alone is informative (detection AUC 模糊 0.91, 纹理损失 0.90, 过锐 0.85, 过度生成 0.86, 花屏 0.96, 虚化突兀 0.93), so the metrics do measure these defects. Added to the pool it gains nothing on any defect: mean ΔRPS +0.0004; 模糊 −0.0001 (n.s.), 过锐 +0.0012 and 过度生成 +0.0009 (worse) |
| 11-12 02:40 | **Global stop rule triggered** (E1 and E2 both null) | Three independent sources of new evidence — public human-opinion models, a KVQ-trained human-score model, a generative autoencoder — each know something about the defects, and none knows anything the pool does not. No further modelling experiment on the score is justified. Remaining open item: E3 v2 (robustness of 花屏, not score) |
| 11-12 03:20 | **E3 second version result** (`shiftglitch2`, `glitch2`: every corrupted frame paired with the same window decoded without damage) | **passed all four checks.** Held-out 花屏 family: detection AUC 0.885 → **0.9999**, recall 0.36 → 0.97, precision 1.00. Shift fold 0: no false alarm, RPS unchanged. Standard folds 0–1: 花屏 F1 0.980 → 0.990 (false alarms 0 → 1, misses 2 → 0), clean images flagged 5.2 % → 3.7 %, mean RPS +0.0004 (CI90 [−0.0012, +0.0018]), argmax 49.37 = 49.37. **Adopted** for the pe tower; remaining folds and a 100 %-data model are training. The pool needs a rule so that one member trained on other-source 花屏 is not outvoted (mean pooling would give 0.15 when only it fires) |

## 7. Is 51–52 a ceiling? (owner's question, 2026-11-12)

New evidence gathered for this question (all offline, CPU):

1. **Residual grading errors carry no content structure.** Clustering images by what they show (PE-Core global features, 20 and 60
   clusters) explains no more of the ensemble's signed grade residual than a random partition does (e.g. 过锐: 1.0 % vs null
   1.7 %; 过度生成: 3.5 % vs 3.0 %). The statement says grades depend on "内容主体、场景复杂度、观看预期"; whatever that
   dependence is, the models already have it.
2. **The label noise cannot be measured from this data.** Only 4 pairs of labelled images are near-identical in pixels and 11 are
   very similar; the set was built without duplicates. Published subjective-quality studies put the spread of individual ratings at
   about 0.6 of a grade at mid-scale even in controlled tests (SOS parameter ≈ 0.08–0.11), which by itself caps exact-grade
   agreement between two raters near one half. That is the same order as our P2/P1 label F1 (0.27–0.65), but it is an analogy,
   not a measurement of this dataset.
3. **The competition has pass/fail gates, and one is binding.** All four must hold: total > 40, positive-defect ≥ 15, P0 ≥ 13,
   clean ≥ 12. We have 15.5 / 16.3 / 19.6 (`s1` online: 15.08 / 16.15 / 20.00): 0.5 points of margin on the positive gate, 7.6 on
   the clean gate. On 654-image sets drawn from the labelled data the current decoder falls below 15 in **6.3 %** of draws.

   | Operating point (primary ensemble, nested OOF) | labels per image | total | positive | P0 | clean | P(positive < 15) on 654 images |
   |---|---|---|---|---|---|---|
   | Decoder fitted for the total (current) | 1.86 | 51.45 | 15.53 | 16.31 | 19.61 | 6.3 % |
   | Same decoder + defect shift 0.2 | 1.92 | 51.50 | 15.64 | 16.29 | 19.57 | 2.2 % |
   | Gate-aware decoder (clean capped at 19.0, defect-prone start) | 2.11 | 51.22 | 15.76 | 16.32 | 19.14 | 0.8 % |
   | Argmax + defect shift 0.6 | 1.98 | 51.16 | 15.93 | 16.05 | 19.18 | 0.0 % |
   | Argmax + defect shift 0.9 | 2.08 | 51.15 | 16.04 | 16.13 | 18.99 | 0.0 % |

   An unseen source lowers the positive score further (one tower: 15.04 → 13.44 when every image is from an unseen family;
   13.96 → 13.61 without the 花屏 family), so the margin matters more than these in-distribution numbers say.
4. **35 + 5 points of the 100 are not the objective score**: deployability 15, cost 20, optional explanation 5 (0 if no
   explanation is output; we output none).

Conclusion recorded for the owner: with these 3,573 labels, single frames and public models, every route converges on 51–53 for the
65 objective points; that plateau is consistent with a label-noise floor for the common defects but is not proven to be one. What can
still move the outcome is not a better classifier: the operating point against the gates, robustness on unseen sources (E3), the 40
non-objective points, labels for the rare defects, and a small re-annotation study to measure the real ceiling.

## 8. Plan for the highest honest objective score (owner's instruction, 2026-11-12)

Target: the 65 objective points on the final set; constraints: published rules (≤30B trained parameters, ≤15 s per image on
one 24 GB card), no use of practice-set feedback or test images, every choice validated out of fold.

**What was checked for this target today (all OOF, paired cluster bootstrap):**

| Lever | Result | Decision |
|---|---|---|
| Composition over all 28 models with complete OOF (incl. ConvNeXt-V2, DINOv3, second/third seeds, pillar 2, second V16) | best set that stays resident on 24 GB is the six-model primary (RPS 0.04192). Adding the second V16 and pillar 2 gives −0.00018 (CI90 [−0.00030, −0.00005]), all eleven −0.00020, greedy to 29.6B −0.00032 — at most ≈+0.1 Doc2S, and none of these fits 24 GB resident | keep the primary set; the larger sets are not worth losing single-card deployment |
| Decoding rule on the same probabilities (argmax, one-parameter shift, Decoder55, multi-start, bagged) | 51.28 / 51.33 / 51.45 / **51.52** / 51.45; no pair differs significantly (multi-start vs single start +0.06, CI90 [−0.04, +0.15]) | ship the multi-start decoder; decoding is not a lever |
| Full-data models vs the fold models the decoder was fitted on (800 unlabelled frames) | full-vs-fold TV 0.085–0.089 = fold-vs-fold 0.086–0.100; defect rates unchanged; slightly more confident (0.93 vs 0.91) | decoder transfers to the full-data models |
| Simulated 虚化突兀 (depth-based subject cut-out on KVQ frames) | first renders do not resemble the real defect (the depth split rarely isolates a person on this content; blur mostly destroys overlays) | stopped before training; not faithful, low expected value |
| Simulated 花屏 with matched negatives (E3) | passed (§6) | integrate: `primary_v2` uses the 花屏-robust pe tower and lets it raise the pooled 花屏 probability |

**The model to ship** (`final_engine.py --set primary_v2 --half`): cradio_v4_h + pe_core_large (trained with simulated 花屏) +
siglip2 + v16_full8 + Phoenix p1 (merged, 4-bit) + p3, all trained on 100 % of the labels, family-balanced pooling, multi-start
decoder fitted on pooled OOF. Expected objective score: OOF 51.5, plus ≈0.4 for training on all labels instead of 7/8
(learning curve), i.e. about 52 on data like ours; less on unseen sources.

**What would raise it further, in order of evidence:**
1. Labels for the rare defects (过度生成, 虚化突兀): ≈+0.8 per doubling of those positives — the only measured lever.
2. Unlabelled frames from the same production pipelines as the labelled set: self-training recovered 25–32 % of the label value on
   such frames (`ssl`); frames from other sources (KVQ) gave nothing.
3. Nothing else: ten training-recipe changes, three external knowledge sources and five decoding rules are all null inside the ensemble.

## 9. Synthetic-data programme (SynthLab) — outcome, 2026-11-12 07:10 UTC

Full record: `docs/SYNTHLAB.md` (§7–§13) and `results/r55/synthlab/LEDGER.md`.

| Work | Result on real labels | Status |
|---|---|---|
| Behavioural probe (240 held-out clean frames × 31 controlled conditions) | blur, codec loss and sharpening are detected in order; two blind spots (resolution loss, edge-preserving smoothing) are not what annotators label | no recipe |
| Benign-processing invariance: extra rows, in-place augmentation, in-place twins | in distribution within training noise; unseen-pipeline folds worse by +0.0017 to +0.0050 mean RPS | closed |
| 过度生成 from public restoration models (Real-ESRGAN x4plus / x2plus, BSRGAN, compact; matched bicubic negatives) | recognised by real-trained models on 24 % of positives vs 3 % of negatives; training on it: 过度生成 RPS −0.00004, no change in detection | closed |
| Restoration models as an instrument (33 idempotence features) | pool + features worse than pool (+0.0006 mean RPS) | closed |
| Gate method | control trained with three seeds; seed 2026 is a favourable draw (过度生成 F1 0.463 vs 0.387 / 0.377), verdicts now use the three-seed mean and a training-noise guard; one leak found and voided | in force |

The deliverable is unchanged: `primary` (nested out-of-fold 51.45; engine 19.3 GB, 5.5 s per image), with `primary_v2`
as the option that is robust to 花屏 from unseen sources at no measurable cost or gain in score.

**Online reference for `primary` (reported by the owner, 2026-11-12):** 51.78 = positive 15.44 + P0 16.34 + clean 20.00
(file assumed to be `results/final/pred_nriqa_dataset_test_r55_primary.json`).

| | Positive / 25 | P0 / 20 | Clean / 20 | Total |
|---|---|---|---|---|
| `primary`, offline nested out-of-fold | 15.53 | 16.31 | 19.61 | 51.45 |
| `primary`, online practice set | 15.44 | 16.34 | 20.00 | 51.78 |
| `s1`, offline nested out-of-fold | — | — | — | 51.04 |
| `s1`, online practice set | 15.08 | 16.15 | 20.00 | 51.23 |

The two defect components agree with the offline estimate to within 0.1; the whole offline-to-online difference is the
clean component, which is 20.00 on every file uploaded to this set. The step from `s1` to `primary` is +0.41 offline and
+0.55 online. Offline selection therefore transfers, and nothing was tuned on the practice set. Pass/fail conditions:
total > 40, positive ≥ 15 (margin 0.44), P0 ≥ 13, clean ≥ 12 — all met.


## 10. Further exploration after the online reference (owner: "keep exploring"), 2026-11-12 08:30–09:20 UTC

All checks are out-of-fold on the 3,573 labelled images, CPU only except the engine bench; nothing uses the practice
set. Scripts, contact sheets and the index-to-file table are in `results/r55/audit_2026-11-12/`.

**10.1 Visual audit of the ensemble's confident grading errors (new kind of evidence: looking at the frames).**
过度生成 confusion (rows = human label none / P2 / P1 / P0, columns = pooled argmax):
`[2828 39 14 8] [141 58 63 14] [45 32 88 53] [12 14 48 116]`.
- Correct P0 calls are frames whose whole surface is "painted" (swirled fur, brush-stroke texture).
- Only 12 of 190 P0 frames are called clean. The ones viewed show small local inventions on an otherwise normal
  frame: tiny caption text turned into pseudo-glyphs (#1943), squiggles in a plain sky (#1800, #1802), altered
  faces (#3269, #222).
- Slight (P2) 过度生成 is called clean for 141 of 276 frames. The ones viewed have no common cue: a dithered arm
  (#3245), fused fingers (#865), ripples in a sky (#2748), an over-sharpened old film (#1634).
- False alarms are often frames that show the same defect under another label: 25 of the 86 frames called
  过度生成 without the label carry 虚化突兀 P0 (e.g. #279, #1751, both swirled like the P0 hits). The two labels
  are used as mutually exclusive (7 frames carry both, out of 684 + 225). Frames with destroyed faces and garbled
  captions (#287) or white edge halos (#1917) carry neither 过度生成 nor 过锐.
- Reading: a visible share of the "errors" at the confident end are label-convention choices, not model blindness.
  This supports, and still does not measure, the label-noise explanation of §7.

**10.2 Tests that followed from the audit.**

| Idea | Test | Result | Decision |
|---|---|---|---|
| Local evidence: look at more native-resolution regions and keep the most suspicious one | existing 8 random crop-set passes of two pe towers (`tta_r50j96.npz`): spread of P(过度生成 present) across passes; max over passes vs mean | spread 0.008 / 0.013 (0.03 / 0.04 on P2 frames); detection AUC +0.0006 / +0.0014, none\|P2 AUC +0.001 / +0.002, grade AUCs unchanged | no tile model; the towers' answer does not depend on which regions they see |
| 过度生成 and 虚化突兀 are exclusive: when both are decoded keep the more probable one | nested decoder output + rule, paired bootstrap over videos | 33 frames decoded as both (human: 22 / 10 / 1 both); total −0.02, CI90 [−0.12, +0.08] | closed; the pool cannot tell which label the annotator chose |
| Cross-defect second stage | boosting on all 24 pooled probabilities (control arm of `probe_eval.py`) vs the raw pool | worse on all six defects (过度生成 RPS 0.0427 vs 0.0405) | closed |
| Pooling rule | arithmetic (current), geometric, power 0.5 / 2, cumulative log-odds, half-half | arithmetic is best: geometric +0.0008, power 0.5 +0.00009, power 2 +0.0003, log-odds +0.0009, half-half +0.0002 mean RPS, every CI90 above zero | keep arithmetic |

**10.3 Provenance of the labelled set.** By file-name pattern (used for analysis only): 1,423 timestamp frames
(the only large source of clean frames, 419), 1,308 frames tagged with an internal pipeline, 277 from the
supplement batch named by defect, 488 other, 77 stock photos. Nothing is a public dataset, so "more frames from
the same pipelines" can only come from the owner. The statement confirms that the 1,902 + 1,671 images in use
are all the labelled data released.

**10.4 Not tested, with reason.** The models the statement cites (DeQA-Score, Q-Insight, VisualQuality-R1,
FGResQ) output a quality score or a distortion description. E1 already showed that 19 public quality scores,
informative alone (detection AUC 0.85–0.96), add nothing to the pool, and the pool already contains a fine-tuned
8B VLM. A higher-resolution whole-frame view is already what the V16 and VLM members see, with the same
过度生成 skill (0.70–0.75). Neither was run.

**10.5 Image resolution as an input — a shortcut, not adopted.** Six size features (height, width, pixels, aspect)
added to the second-stage model (`logs/r55/probes/resolution_eval.log`): 虚化突兀 RPS −0.0008 (CI90 [−0.0014,
−0.0003]), 过锐 adjacent-grade AUC +0.0065 (CI90 [+0.003, +0.010], RPS n.s.), other defects nothing; the second
stage with the features is still worse than the raw pool (0.0155 vs 0.0150 on 虚化突兀). The 虚化突兀 effect is a
collection artefact: 79 of the 225 positives come from one supplement batch with two unusual sizes (941×1672:
60 of 60 positive; 1086×1448: 19 of 19). A size rule would not generalise and the ensemble already finds those
frames (recall 0.97). What it shows about the deliverable: 虚化突兀 recall is 0.97 on that batch and **0.69 on the
other 146 positives** (detection AUC 0.989 overall, 0.984 without the batch), so the 虚化突兀 score on the final
set depends on how much of it comes from that batch.

**10.6 `primary_v2` completed** (花屏-robust pe tower in place of the pe tower; same expected score).
Decoder fitted on pooled out-of-fold probabilities: argmax 51.16, nested 51.31 (`results/r55/final/primary_v2_decoder.pkl`).
Engine check, 40 images, batch 1, bf16 backbones (`logs/r55/final/bench_primary_v2_half.log`): 19.25 GB resident,
20.42 GB peak, 2.48 s mean / 6.38 s p95 per image on an otherwise idle H200; reload sanity passed for all six members.
Predictions for the practice template are written to `results/final/pred_nriqa_dataset_test_r55_primary_v2.json`
(fresh inference; not uploaded).

**10.7 Synthetic 虚化突兀 with a real subject mask (`bkseg1`) — closed.** Pre-registered in `docs/SYNTHLAB.md` §14
because 虚化突兀 P0 is the most valuable label and recall is 0.69 outside one easy batch. 1,800 frames from 900
labelled seeds (Mask R-CNN subject mask, background-only blur vs the same blur on the whole frame). Fidelity is
weak (real-trained models call 12.1 % of positives, 8.3 % of the matched negatives). Screen, two seeds against
three control seeds: 虚化突兀 Brier −0.00046 on standard folds (CI90 [−0.0043, +0.0032]), +0.0041 on the fold that
holds out the family with 142 positives (CI90 [−0.0069, +0.0158]); detection AUC unchanged or lower; 过锐 and
过度生成 worse (the built frames inherited seed labels the blur invalidates — a flaw of the recipe, irrelevant to
the null on the target). No corrected variant was run: the rule fixed beforehand closes a null screen.

**10.8 Averaging training seeds of a member — closed without training.** With the three control seeds of the pe
tower on folds 0–1 (`results/r55/audit_2026-11-12/seed_bag.py`, `logs/r55/probes/seed_bag.log`): the tower alone
improves by 5.7 % RPS when three seeds are averaged (0.0512 → 0.0483), the tier-1 family by 0.7 %, the primary
ensemble by 0.09 % (0.04128 → 0.04125). Doing this for all three tier-1 members would cost about 6 GB more memory
on the 24 GB card and buy roughly 0.3 % of RPS. The ensemble has already averaged away the training noise of its
members; what remains is error the members share.

## 11. Third exploration round (owner: "no more data on my side, explore by yourself, other ways allowed"), 2026-11-12 11:00–17:00 UTC

Target of the round: find anything that moves the objective score towards 55 without practice-set tuning. All numbers
are out of fold on the 3,573 labelled images; scripts and logs: `results/r55/explore3/`, `logs/r55/explore3/`,
`code/r55/{eval_batch_study,eval_sib_study,build_sib_split,probe_iqa_bank2}.py`. Nothing reads practice-set labels;
practice-set **file names** were read once (11.2) and are used for analysis only.

### 11.1 Summary

| Question | Test | Result | Decision |
|---|---|---|---|
| Does the supplement batch help on first-batch images? | pe tower trained on both batches / first batch only / supplement only (`origonly`, `supponly`, 8 folds) | **No.** On first-batch validation images: detection AUC 0.9614 vs 0.9632 (first batch only), adjacent-grade AUC 0.8045 vs 0.8010, decoded 49.53 vs 49.71 (+0.18, CI90 [−0.54, +0.88]). The reverse holds too (supplement images: both 0.05637, supplement only 0.05614 RPS) | the two releases are separate domains; for a first-batch-like test set the effective labelled set is ≈1,900 images |
| Is the practice set video-disjoint from the training data? | leading video id of the file names | **No.** 95 of 150 practice frames come from a video that has 1–6 labelled training frames (126 of 129 such training frames are in the first batch); no practice video has two practice frames | the standard grouped folds describe unseen videos; see 11.3 |
| Does a tower predict a frame better when its siblings were trained on? | sibling-exposed folds (`build_sib_split.py`, presets `sibxA` / `sibxB`), 620 held-out frames of 310 multi-frame videos | RPS −10.0 % (CI90 [−0.0066, −0.0025]); 过度生成 −22.5 %, 纹理损失 −9.1 %; single-frame videos (control) −1.1 % (n.s.). Decoded with a nested decoder: 49.71 → 50.18 (argmax 48.76 → 50.50) | already contained in the full-data models; no change to the deliverable |
| Do the siblings' human labels add more when given explicitly? | nested logistic combiner on pooled probabilities + sibling labels / residuals | on the pool that never saw the video: RPS −4.4 % on sibling frames, decoded −0.3 (CI90 [−1.0, +0.4]); on top of a tower that trained on the siblings: −1.0 % (n.s.) | no look-up component (it would also need the file name as a key) |
| Would a decoder fitted for the exposed condition differ? | pe tower, decoder fitted on exposed probabilities (sibling frames weighted to 63 %) vs the grouped decoder | +0.07 (CI90 [−0.72, +0.82]) | keep the decoder |
| Per-batch decoders | decoder fitted on first-batch out-of-fold rows only | 52.21 vs 52.30 with the pooled decoder (−0.11, CI90 [−0.40, +0.16]) | keep one decoder |
| File-name side facts (pipeline tag, batch, defect keyword) as inputs — analysis only | nested logistic regression on pooled log-probabilities + one-hot facts | RPS −2.7 % (CI excludes 0), decoded 51.21 → 51.50 (within noise) | a shortcut worth ≤ 0.3 points; not used |
| Linear cross-defect second stage | logistic regression on all 24 pooled log-probabilities, three regularisation strengths | worse (+0.0031 RPS, CI above 0) | closed (confirms the boosting result of §10.2) |
| Decoder head-room | Decoder55 fitted in-sample on all rows | 51.87 (nested 51.45, argmax 51.28) | at most ≈0.4 is lost to threshold estimation; decoding is not a lever |
| Annotation-session effects | residual autocorrelation along spreadsheet rows, different videos only | lag-1 correlation 0.05–0.14; 50-row blocks explain 5–8 % of residual variance (random blocks 2 %) | small, and the row order follows the file names (same pipelines adjacent) |
| Label repeatability from near-identical frames | 48×48 thumbnail correlation ≥ 0.95 | 13 pairs in total; cannot be measured | — |
| Public quality models that E1 could not load, incl. FGResQ (cited in the statement), A-FINE, TOPIQ ×3, UNIQUE, WaDIQaM, CNNIQA + 16 principal components of the FGResQ feature | `probe_iqa_bank2.py` → `probe_eval.py`, 48 features | pool + bank − pool: mean RPS +0.00013; 过度生成 +0.0018 (worse, CI above 0); no defect improves | closed. CLIP-IQA+, LIQE, QualiCLIP need the external `clip` package; installing it from GitHub was not permitted in this session, so they remain untested |
| Ensemble members on first-batch images | leave-one-out / double weight, first-batch rows only | every member contributes (dropping V16 +0.0011 RPS); a first-batch-only pe tower added as a fourth tier-1 member: −0.7 % RPS on first-batch images (same-recipe control −0.2 %), +1.4 % on supplement images | a small option (≈ +0.1 on first-batch-like data), not adopted |

### 11.2 The practice set and the first batch

- File-name patterns of the 150 practice images are those of the first batch (time-stamp frames, `v7L3HiV2`, …); none
  of the supplement's patterns occurs. 95 practice frames share their video id with first-batch training frames.
- Out-of-fold score of the primary ensemble by release batch: first batch 52.30 (15.93 / 16.76 / 19.61), supplement
  51.09 (15.15 / 15.94 / 20.00). The online reference for `primary` (51.78 = 15.44 + 16.34 + 20.00) lies inside the
  spread of 140-image draws from either.
- 654-image draws from the first batch, nested decoder: **52.25 ± 0.56** (5 %–95 %: 51.3–53.1); P(positive < 15) =
  0.1 % (it was 5.3 % on draws from all labelled data, the figure used in §7); P(total ≥ 54) = 0.2 %; P(total ≥ 55) = 0.
  These are for videos the models have not seen; the full-data models add the usual ≈0.4 and, if the final set shares
  videos with the training data as the practice set does, a further few tenths (11.3).

### 11.3 What sibling exposure is worth

A second frame of the same video is a different picture (median thumbnail correlation to the nearest sibling 0.27 in
the training data, 0.30 between practice frames and their training siblings). Human labels of two frames of one video
agree on 76 % / 65 % / 77 % / 86 % of 模糊 / 纹理损失 / 过锐 / 过度生成 grades and on all six defects for 44 % of the
pairs; the ensemble's prediction for a frame agrees with the sibling's human label at the same rate (75 / 61 / 78 /
84 %), and with the frame's own label more often (80 / 77 / 80 / 85 %). So a sibling's label is a weaker predictor than
the ensemble, which is why copying it scores 46.4 against 52.0 and why the look-up adds nothing after decoding. What a
tower gains by training on the sibling (RPS −10 %) is real, comes free with the full-data models, and cannot be
enlarged without more labelled frames of the same videos.

### 11.4 Where 55 stands

| Evidence | What it says |
|---|---|
| First-batch-like 654-image draws | 52.25 ± 0.56; 55 is five standard deviations away |
| Label saturation (§2 item 4) | 模糊, 纹理损失, 过锐 gain +0.003 to +0.017 mean F1 per doubling of the labels; only 过度生成, 虚化突兀 and 花屏 are data-limited |
| Even the best-learned defect | 纹理损失 (2,803 positives, saturated) reaches F1 0.65 / 0.65 / 0.78. If every graded defect reached those values the total would be ≈54.4, and equal F1 is harder to reach for rarer labels |
| Supplement batch | 1,671 extra labelled frames from other pipelines add nothing on first-batch images: only labels from the same sources count |
| Twelve further checks of this round | nothing beyond noise |

With the released labels and image-only models the reachable range is about 52–53.5 on a first-batch-like final set.
No experiment of the three rounds points to a route to 55. The levers that remain are not modelling:

1. **Labelled frames from the same sources as the first batch** (ideally other frames of the same videos). The only
   lever with measured multi-point value; frames from other collections are worth nothing (supplement, KVQ).
2. **The three scored submissions on the final set**, which the organisers allow to be used for tuning ("可根据反馈微调模型").
   AGENTS.md rule 2 forbids it. A pre-declared choice between two operating points (current decoder / defect shift)
   is the most that three readings can support, ≈ ±0.3.
3. **Unlabelled test frames as extra in-domain data** (self-training): they come from first-batch videos, so the
   mechanism of `ssl` applies; estimated ≈ +0.1 to +0.3 (not measured), and it uses test images, which rule 2 also excludes.
   The same mechanism without test images: unlabelled frames exported from the first-batch videos (the video ids are
   in the file names). No annotation is needed; `ssl` recovered 25–32 % of the label value on such frames.
4. **The 35 non-objective points** (deployability 15, cost 20) and the optional 5 for explanations, which separate
   teams by far more than what is left in the objective score.

Open question worth asking the organisers: whether the final 654 images, like the practice images, come from videos
that also appear in the training data. It changes the expected score by a few tenths and nothing in the model.

### 11.5 Replication on a second seed (2026-11-12 14:30 UTC)

Both headline findings were repeated with seed 2027 (same folds, same recipe; logs `logs/r55/explore3/*_s2027.log`).

| Finding | Seed 2026 | Seed 2027 |
|---|---|---|
| First-batch-only pe tower vs both-batch tower, first-batch validation images: mean RPS difference | +0.00055, CI90 [−0.00077, +0.00206] | +0.00038, CI90 [−0.00102, +0.00173] |
| — mean detection AUC (both → first batch only) | 0.9614 → 0.9632 (+0.0019, CI90 [−0.0007, +0.0044]) | 0.9621 → 0.9631 (+0.0010, CI90 [−0.0011, +0.0034]) |
| — mean adjacent-grade AUC | 0.8045 → 0.8010 | 0.8021 → 0.7968 (−0.0053, CI90 [−0.0118, +0.0025]) |
| — decoded, decoder nested on first-batch rows | 49.53 → 49.71 (+0.18, CI90 [−0.54, +0.88]) | 48.92 → 48.98 (+0.05, CI90 [−0.72, +0.65]) |
| — same towers on supplement images: mean RPS | 0.05637 → 0.07795 | 0.05688 → 0.07784 |
| Sibling-exposed tower vs grouped tower, held-out frames of multi-frame videos: mean RPS | −10.0 %, CI90 [−0.0066, −0.0025] (variants A + B, 620 frames) | −8.1 %, CI90 [−0.0066, −0.0007] (variant A, 310 frames) |
| — frames of single-frame videos (control) | −1.1 % (n.s.) | −1.0 % (n.s.) |
| — argmax score on the held-out frames (variant A) | 48.30 → 50.00 | 48.26 → 49.75 |

Both hold. One per-defect detail differs between the seeds and is therefore not a finding: with seed 2027 the
first-batch-only tower is worse on 过度生成 (+0.0047 RPS, CI above 0) and better on 模糊 (−0.0025, borderline); with
seed 2026 it was 过锐 and 模糊 that moved.

Tooling fixed on the way: `queue_run.py` named its job lock and job log by preset / seed / fold only, so two instances
training the same preset for different backbones collided (one crashed). Locks and logs of non-default backbones now
carry the backbone name, and a lock that disappears between two checks no longer raises.

### 11.6 First-batch-only twins of all three tier-1 towers (2026-11-12 15:00 UTC)

`origonly:2026` was trained for `cradio_v4_h` and `siglip2_so400m_naflex` as well (8 folds each) and the three twins
were put into the primary ensemble (`results/r55/explore3/batch3.py`, log `logs/r55/explore3/twins_pool.log`). All
numbers are on first-batch out-of-fold rows; the decoder is nested over folds and fitted on first-batch rows.

| Ensemble | RPS vs primary (CI90) | Decoded | Decoded vs primary (CI90) | RPS on supplement rows |
|---|---|---|---|---|
| primary | 0.03733 | 52.21 | — | — |
| tier-1 = primary towers + 3 twins | −0.5 % [−0.00034, −0.00005] | 52.02 | −0.19 [−0.50, +0.18] | +0.00151 (worse) |
| tier-1 = primary towers + 3 both-batch seed-2 towers (control) | +0.0 % | 52.18 | — | −0.00006 |
| tier-1 = twins only | +0.2 % (n.s.) | 51.69 | −0.52 [−0.89, −0.08] | +0.00427 (worse) |
| twins as a family of their own | +0.1 % (n.s.) | 51.92 | −0.29 [−0.65, +0.06] | +0.00239 (worse) |

Decision: **no twins.** The small RPS gain of adding them does not survive decoding, and they cost accuracy on
supplement-like images. The option noted in 11.1 (≈ +0.1) is withdrawn.

Caveat on the headline of 11.1 ("the supplement adds nothing on first-batch images"): it was measured with a
same-recipe comparison for pe only (two seeds). For the other two backbones the first-batch-only twin is clearly worse
on first-batch rows than the deployed both-batch member (cradio 0.04951 vs 0.04648, siglip 0.05261 vs 0.04896 mean
RPS), and replacing all three tier-1 towers by twins costs 0.5 decoded. That comparison mixes two recipes (the
deployed members select the best epoch, the twins use the fixed last epoch), so same-recipe both-batch towers
(`fixed:2026`) of both backbones are being trained as the control; result in 11.7.

### 11.7 Same-recipe control for the other two backbones (2026-11-13 04:00 UTC) — closes the caveat of 11.6

`fixed:2026` (both batches, same recipe as the twins) was trained for `cradio_v4_h` and `siglip2_so400m_naflex`
(logs `logs/r55/explore3/batch_study_s2026_<backbone>.log`, `batch_auc_s2026_<backbone>.log`, `twins_pool_v2.log`).
First-batch validation images, both batches → first batch only:

| Tower | mean RPS (CI90 of the difference) | detection AUC | adjacent-grade AUC | decoded (CI90) |
|---|---|---|---|---|
| pe_core_large, seed 2026 | 0.04630 → 0.04685, +1.2 % [−0.00077, +0.00206] | 0.9614 → 0.9632 | 0.8045 → 0.8010 | +0.18 [−0.54, +0.88] |
| pe_core_large, seed 2027 | 0.04728 → 0.04766, +0.8 % [−0.00102, +0.00173] | 0.9621 → 0.9631 | 0.8021 → 0.7968 | +0.05 [−0.72, +0.65] |
| cradio_v4_h | 0.04826 → 0.04951, +2.6 % [−0.00013, +0.00259] | 0.9603 → 0.9603 | 0.8031 → 0.7891 (−0.0140, CI90 [−0.0207, −0.0065]) | −0.61 [−1.36, +0.16] |
| siglip2_so400m_naflex | 0.05059 → 0.05261, +4.0 % [+0.00047, +0.00347] | 0.9581 → 0.9548 (−0.0033, CI90 [−0.0063, −0.0004]) | 0.7954 → 0.7810 (−0.0144, CI90 [−0.0217, −0.0065]) | −0.26 [−0.97, +0.42] |

Ensemble, tier-1 family made of the three same-recipe towers (v16 and the two phoenix members unchanged):

| tier-1 family | first-batch RPS | decoded |
|---|---|---|
| three both-batch towers (`fixed`) | 0.03753 | 51.57 |
| three first-batch-only towers | 0.03742 (−0.3 %, CI90 [−0.00042, +0.00017]) | 51.69 (+0.12, CI90 [−0.23, +0.54]) |

Corrected reading of the 11.1 headline:

- **Single towers:** the supplement gives a small grading benefit on first-batch images (adjacent-grade AUC lower
  without it for all four runs: −0.004, −0.005, −0.014, −0.014; outside noise for cradio and siglip) and no detection
  benefit. "Adds nothing" was true for pe and too strong for the other two backbones.
- **Ensemble:** nothing measurable. With or without the supplement in all three tier-1 towers the pooled RPS and the
  decoded score on first-batch images are the same within noise. The practical statement stands: for a first-batch-like
  test set 1,671 labelled frames from other pipelines buy at most a few per cent of single-tower RPS and nothing after
  ensembling; only labels from the same sources have shown multi-point value.
- The 6–8 % gap seen in 11.6 between the twins and the deployed members was the recipe, not the supplement: the
  same-recipe both-batch family also sits 0.6 below the primary (51.57 vs 52.21). The deployed tier-1 members keep the
  best validation epoch (§ review of 2026-11-11: up to ≈0.5 of selection optimism per tower), so part of that 0.6 is
  optimism of the out-of-fold figure and part is a real recipe difference; the two cannot be separated out of fold.
  The honest centre for a first-batch-like final set is therefore 51.6–52.3 (the practice-set reading of `primary`,
  51.78, lies inside), with ±0.6 sampling spread on 654 images.

Decision: deliverable unchanged (`primary` / `primary_v2`); no twins; the supplement stays in the training data (it
costs nothing on first-batch images and is needed if the final set contains supplement-like images).

### 11.8 Owner: "improve the score by any legal means" — what is left (2026-11-13)

Checks made in answer (CPU, out of fold; `results/r55/explore3/best_of_3.py`, `logs/r55/explore3/best_of_3.log`):

| Question | Result |
|---|---|
| `primary` vs `primary_v2` vs `fallback` on first-batch rows (decoder nested, all rows) | 52.30 / 51.76 / 52.11 (all rows: 51.45 / 51.31 / 51.35). On 654-image first-batch draws `primary_v2` is −0.55 (sd 0.40; better than `primary` in 8 % of draws), `fallback` −0.20 (sd 0.34; better in 29 %). Part of the `primary_v2` gap is the recipe artefact of 11.7 (its pe tower uses the fixed last epoch, the one it replaces the best validation epoch). **For score, ship `primary`**; `primary_v2` only buys robustness to 花屏 from unseen sources. This corrects the "same expected score" wording of §8 / §10.6. |
| What are the three scored final-set submissions worth with the candidates we have? | best of {`primary`, a second decoder or `fallback`, a third}: **+0.07 to +0.13** expected (90th percentile +0.3 to +0.4). The earlier guess of ±0.3 was too high: every alternative is slightly worse on average, so the maximum rarely beats `primary`. |
| Subject-aware views (the statement ties grades to the main subject) | not new: the tier-1 towers already take four foreground-guided native crops and the VLM a face / skin region (`dual_engine_preprocess.py` §3.4). |
| A further model family | the only modelling step that ever helped was a new family (+0.41 when the phoenix family joined). The engine is at 20.4 of 24 GB with bf16 weights and a 4-bit language model, so a fourth family does not fit a single 4090. |

Cost-score observation (statement, 执行成本 20 points): a system that runs on one 4090 without quantisation scores
0.7–1.0, one that needs "量化、裁剪、分批等轻量优化" 0.3–0.6. `primary` needs the 4-bit language model (20.4 GB peak,
33.7 GB without `--half`); `fallback` (no VLM member) peaks at 14.8 GB, 3.7 s per image, and costs 0.1–0.2 of
objective score (within noise). Which tier the jury assigns is its call, but the difference between the tiers
(2–14 points) is larger than anything left in the objective score. Owner's decision.

No legal step with evidence for more than ≈ +0.1 objective is left. The earlier online 52.89 / 53.03 were the
21-model file (52.43 online, ≈50.5 strict out of fold) with labels edited against practice-set feedback; that gain
does not exist on the final set.

**11.8 addendum — what the VLM member is worth** (`results/r55/explore3/vlm_value.py`, `logs/r55/explore3/vlm_value.log`).
Alone, the fine-tuned Qwen3-VL-8B (p1) is as good as the best towers (mean RPS 0.0480; v16 0.0473, p3 0.0473, pe
0.0499; adjacent-grade AUC 0.815). Inside the ensemble it is redundant with p3:

| Left out | RPS vs primary (CI90) | Decoded, all rows (CI90) | Decoded, first-batch rows (CI90) |
|---|---|---|---|
| VLM p1 (p3 keeps the family weight = `fallback`) | +0.7 % [+0.00015, +0.00040] | −0.11 [−0.37, +0.15] | −0.19 [−0.49, +0.11] |
| VLM p1 (p3 at weight 0.5) | +0.8 % | +0.02 [−0.19, +0.23] | +0.13 [−0.15, +0.42] |
| p3 | +0.7 % | −0.35 [−0.56, −0.09] | −0.53 [−0.84, −0.24] |
| both phoenix members | +2.5 % | −0.16 [−0.41, +0.14] | −0.55 [−0.93, −0.22] |
| v16 | +2.1 % | −0.65 [−1.00, −0.28] | −1.43 [−2.00, −0.90] |
| tier-1 (all three) | +2.6 % | −0.60 [−0.92, −0.30] | −0.95 [−1.36, −0.49] |

The VLM adds a small, real amount of probability quality (RPS) and nothing measurable after decoding, for 5.6 GB, the
only quantised component and about a third of the inference time. Recommendation to the owner: ship `fallback`
(14.8 GB peak, 3.7 s per image, no quantisation); the choice of p3's weight between 0.5 and 1 is inside the noise and
stays at the pre-declared family-balanced value.

### 11.9 New decision rules for `fallback` (2026-11-13) — from the research report `reports/Breaking the 52 point plateau.md`

`code/r55/decoder_eu.py` (rules `eu`, `vote`; description in the file header), fitted on the pooled out-of-fold
probabilities of `fallback`; `final_engine.py --decoder {d55,eu,vote}`. Nested over the canonical folds, difference to
the shipped Decoder55 (four starts), CI90 by video bootstrap (`logs/r55/final/fit_decoder_eu_fallback.log`):

| Rule | All rows | First-batch rows | Components on first-batch rows (positive / P0 / clean) |
|---|---|---|---|
| Decoder55 | 51.28 | 52.12 | 15.88 / 16.63 / 19.61 |
| `eu` (λ = 0.25 on every fold) | 51.69, +0.40 [+0.14, +0.72] | 52.22, +0.10 [−0.29, +0.53] | 15.89 / 16.48 / 19.84 |
| `vote` | 51.52, +0.24 [+0.08, +0.42] | 52.30, +0.18 [−0.01, +0.39] | 15.93 / 16.64 / 19.73 |

Over four nesting partitions (research notes): `eu` +0.52 / +0.29, `vote` +0.26 / +0.28. `eu` has no in-sample
optimism (51.70 fitted on all rows vs 51.69 nested; Decoder55 51.84 vs 51.28). Its gain is mostly clean score; on the
practice set the clean score has been 20.00 on every upload, so `eu` cannot show its gain there. Recommendation:
`vote` for a first-batch-like final set (never worse on any partition), `eu` if the final set is mixed.

Practice-set files (owner asked for them to upload as a reference): `results/final/pred_nriqa_dataset_test_r55_fallback_eu.json`
(55 of 150 images differ from the Decoder55 file) and `…_fallback_vote.json` (23 differ). They were decoded from the
engine's saved probabilities of 2026-11-11 (same full-fit weights; Decoder55 on those probabilities reproduces the
JSON of that run exactly) because another user's job held all eight GPUs; a fresh inference pass is queued
(`results/final/fresh_20261113/`, `results/r55/explore4/compare_fresh.py`) and replaces them if anything differs.
A single practice reading (±1.6) cannot confirm a +0.2 change and must not decide between the rules (AGENTS.md rule 2).

**11.9 addendum (2026-11-13 06:30 UTC).**
- Fresh inference pass for `fallback` on the practice images (`results/final/fresh_20261113/`,
  `logs/r55/final/fresh_fallback_test150_20261113.log`): pooled probabilities identical to the saved ones (max |Δp|
  0.00000), 0 of 150 images differ for `d55`, `eu` and `vote`. The delivered `eu` / `vote` files are therefore fresh
  inference output of the current weights (AGENTS.md rule 1). The owner uploaded the `eu` file and reported it as the
  highest practice reading so far (value not recorded here).
- `eu` with a finer λ grid (0.05 … 2): 51.66 / 52.20, no better than the five-value grid (51.69 / 52.22); kept.
- Same rules on the `primary` pool (with the VLM), nested (`logs/r55/final/decoder_eu_primary_report.log`):
  Decoder55 51.52 / 52.43; `eu` 51.89 / 52.46 (+0.37 [+0.07, +0.71] / +0.03 [−0.37, +0.48]); `vote` 51.72 / 52.70
  (+0.20 [+0.03, +0.38] / +0.27 [+0.05, +0.50]). With either rule `primary` is about 0.2–0.4 above `fallback`
  (not significant); the best out-of-fold cell is `primary` + `vote` on first-batch rows, 52.70.
- Roadmap item 3 started: preset `hp` (`train_testbed.py --input-hp 2.0 --hp-gain 2.0`, high-pass residual input,
  pe_core_large) queued for folds 0–1 (`logs/r55/explore4/queue_hp_screen.log`); it starts when another user's job
  releases a GPU. Gate: ΔRPS inside the pool ≤ −0.0003 with CI90 excluding 0 against the same-recipe control towers
  (`fixed` seeds 2026–2028) before any 8-fold run.

### 11.10 Practice reading of `fallback` + `eu`, and two corrections (2026-11-13 06:50 UTC)

- **Owner's upload of `…_fallback_eu.json`: 52.37 = positive 15.90 + P0 16.47 + clean 20.00** (the highest reading so
  far; `primary` + Decoder55 read 51.78 = 15.44 + 16.34 + 20.00). Out of fold on first-batch rows the same system has
  positive 15.89, P0 16.48, clean 19.84. The agreement is closer than one 140-image reading warrants (the `primary`
  reading was 1.1 below its own first-batch figure), so it is a reference, not a measurement of the decoder's gain.
- **Correction 1 — the VLM member is worth something with the multi-start / new rules.** `primary` minus `fallback`,
  nested, paired video bootstrap (`results/r55/explore4/pool_vs_pool.py`, `logs/r55/explore4/pool_vs_pool.log`):

  | Rule | All rows | First-batch rows |
  |---|---|---|
  | Decoder55 (four starts) | +0.23 [−0.02, +0.48], P 0.93 | +0.31 [−0.04, +0.64], P 0.93 |
  | `eu` | +0.20 [+0.00, +0.40], P 0.95 | +0.25 [−0.03, +0.52], P 0.93 |
  | `vote` | +0.20 [−0.04, +0.40], P 0.92 | +0.40 [+0.07, +0.71], P 0.98 |

  The single-start comparison of 11.8 (−0.11 / −0.19, "nothing measurable") understated it; across the three rules the
  VLM adds about +0.2 on all rows and +0.25 to +0.4 on first-batch rows.
  Against the uploaded system (`fallback` + `eu`): `primary` + `eu` +0.20 [+0.00, +0.40] / +0.25 [−0.03, +0.52];
  `primary` + `vote` +0.03 [−0.24, +0.27] / +0.49 [+0.09, +0.86].
- **Correction 2 — quantisation does not lower the cost tier.** The newer statement (file "(2)", 2026-11-05) says for
  执行成本: tier ① (0.7–1.0) = runs stably on one 4090 with resource head-room, "是否采用量化、蒸馏、裁剪、串行或分批执行，不作为
  降档依据"; tier ② (0.3–0.6) = head-room insufficient / throughput limited. The argument of 11.8 came from the older
  file "(1)". What separates `primary` (20.4 GB peak, 5.5 s) from `fallback` (14.8 GB, 3.7 s) is head-room on 24 GB,
  not the 4-bit language model.
- Recommendation: `primary` + `eu` (rule chosen on all-rows out-of-fold evidence, as before). Practice files
  `results/final/pred_nriqa_dataset_test_r55_primary_{eu,vote}.json` decoded from the saved probabilities of
  2026-11-11 (integrity check passed); fresh pass queued (`logs/r55/final/fresh_primary_test150_20261113.log`).
  `primary_eu` differs from the uploaded `fallback_eu` on 12 of 150 images, `primary_vote` on 41.

**11.10 addendum (2026-11-13 07:00 UTC).** Fresh inference pass for `primary` on the practice images
(`logs/r55/final/fresh_primary_test150_20261113.log`): pooled probabilities identical to the saved ones, 0 of 150
images differ for `d55`, `eu`, `vote`. `results/final/pred_nriqa_dataset_test_r55_primary_{eu,vote}.json` are fresh
inference output of the current weights.

### 11.11 Owner's question: a closed-source VLM or a very large open VLM with prompting, as teacher or judge? (2026-11-13)

Evidence already in the repository (`results/decision_model_screen.md`, 2026-10-26; first-batch images, average
precision over the 14 labels / over the six P0 labels), next to today's out-of-fold figures on the same 1,902 images:

| System | mAP14 | mAP_P0 | 过锐 | 过度生成 | 虚化突兀 |
|---|---|---|---|---|---|
| Zero-shot Qwen3.8-27B, rubric prompt, option log-probabilities | 0.269 | 0.356 | 0.129 | 0.072 | 0.077 |
| Other zero-shot 8–27B models (five) | 0.21–0.27 | 0.26–0.39 | 0.11–0.15 | 0.06–0.10 | 0.06–0.10 |
| Linear probe on a frozen 400M encoder | 0.489 | 0.737 | 0.585 | 0.343 | 0.733 |
| One fine-tuned tower (pe_core_large), out of fold | 0.623 | 0.833 | 0.564 | 0.403 | 0.858 |
| Fine-tuned Qwen3-VL-8B (p1), out of fold | 0.629 | 0.835 | 0.586 | 0.384 | 0.846 |
| `primary` ensemble, out of fold | 0.672 | 0.879 | 0.628 | 0.468 | 0.907 |

(Per-defect columns of the 2026-10-26 screen may be defined slightly differently from the mean AP over grades used for
the last three rows; the mAP14 / mAP_P0 columns are the same definition.) Raising the input from 768² to 1024² pixels
changed nothing in that screen. Not tested there: few-shot exemplars, pairwise comparison, closed-source models.
Web research on the question: workflow run `wf_9fde1805-949` (report to be summarised here).

### 11.12 High-pass residual tower (`hp`) — two-fold screen, closed (2026-11-13 07:20 UTC)

`results/r55/explore4/hp_screen.py`, `logs/r55/explore4/hp_screen.log`; 872 validation images of folds 0–1; controls =
same-recipe RGB towers (`fixed`, seeds 2026–2028).

| | `hp` | controls |
|---|---|---|
| Single tower: mean RPS / detection AUC / adjacent-grade AUC | 0.05695 / 0.947 / 0.816 | 0.0509–0.0514 / 0.953–0.956 / 0.821–0.827 |
| Correlation of per-image errors with an RGB control tower | 0.32–0.38 | 0.60–0.66 (controls among themselves) |
| Added as a 4th tier-1 member: change of pooled RPS | +0.1 % [−0.00011, +0.00020] | −0.1 to −0.3 % |
| Added as a family of its own | +1.9 % (worse, CI above 0) | +0.3 to +0.8 % |

`hp` minus the mean of the controls: +0.00013 [−0.00002, +0.00029] as a tier-1 member. Its errors are indeed less
correlated with the RGB towers, but it is 11 % weaker alone and the pool gains nothing; worst on 虚化突兀 (which needs
the image content). Gate (ΔRPS ≤ −0.0003, CI excluding 0) failed → no 8-fold run. Roadmap item 3 is closed for the
high-pass variant; the restoration-pretrained-encoder variant stays untested.

### 11.13 Large VLM as judge or in-context grader: local 27B pilot, published evidence, HTTP harness (2026-11-13 08:10 UTC)

**Local pilot** (`code/decision_eval/run_vllm_judge_pilot.py`, `eval_vllm_judge_pilot.py`; Qwen3.8-27B, bf16, 768² per
image, option log-probabilities; `logs/r55/explore4/vlm_pilot_eval.log`). First-batch frames, different videos.

| Test | Qwen3.8-27B | Ensemble, out of fold, same frames |
|---|---|---|
| Pairwise "which frame is worse", grades one step apart, 1,800 pairs, both orders | 0.718 [0.700, 0.735] | 0.857 [0.844, 0.871] |
| — 过度生成 P0 vs P1 / P1 vs P2 / P2 vs none | 0.633 / 0.590 / 0.613 | 0.700 / 0.793 / 0.913 |
| — 过锐 | 0.747 / 0.687 / 0.733 | 0.847 / 0.813 / 0.947 |
| — 纹理损失 | 0.723 / 0.627 / 0.853 | 0.933 / 0.827 / 0.913 |
| — 模糊 | 0.960 / 0.667 / 0.777 | 0.967 / 0.760 / 0.873 |
| Grade question, balanced sample (200 per defect), order AUC: zero-shot → four labelled exemplars | 模糊 0.831 → 0.823, 纹理损失 0.798 → 0.802, 过锐 0.680 → 0.671, 过度生成 0.670 → 0.670 | 0.940, 0.943, 0.920, 0.884 |

The 27B model is below the ensemble on every cell (−0.140 overall, CI90 [−0.159, −0.121]); on pairs the ensemble gets
wrong it is right 27–58 % of the time (no usable complement); four labelled exemplars change nothing. Closed for open
models of this size, as judge, in-context grader and teacher.

**Published evidence on which model could do better** (read 2026-11-13):
- DistortBench (ICML 2026, arXiv 2604.19966; type + severity of 27 classic distortions, four-choice): best Qwen3.5-27B
  61.9 %, GPT-5.4 51.6 % (the only proprietary model), human majority 65.7 %; thinking variants are worse.
- Q-Bench-Portrait (arXiv 2601.18346): Qwen3-VL-32B 64.0 %, Grok-4 62.5 %, Claude Sonnet 4.5 60.4 %, GPT-5.2 56.7 %,
  Gemini 2.5 Pro 50.8 %. Closed models do not lead on classic low-level perception.
- SalArt-VQA (NeurIPS 2026, arXiv 2606.12671; artifacts in AI-generated images): detection recall / false alarms on
  real images: Gemini 3.1 Pro 99.4 % / 49.7 %, Gemini 3.1 Flash Lite 83.4 % / 6.7 %, Gemma-4-31B 86.3 % / 31.7 %,
  Claude 4.6 Opus 47.6 % / 0 %, Qwen3.5-397B 33.3 % / 3.4 %, GPT-5.4 4.6 % / 0 %.
- Hallucination Score (arXiv 2507.14367): GPT-4o given low-resolution, restored and ground-truth images rates
  super-resolution hallucination about as consistently as humans agree with each other; it needs the reference image.
- VQualA 2025 quality-comparison challenge (arXiv 2509.09190): fine-tuned entries 0.757 vs untuned Qwen2.5-VL-72B
  0.755; pairs are the hardest group (≤ 0.69).

Reading: for 模糊 / 纹理损失 / 过锐 no closed model is ahead of the open family already tested here. For generation
artifacts (closest to 过度生成, our largest loss, 4.3 points) the Gemini 3.1 family is far more sensitive than any other
model, Claude 4.6 Opus is the precise one. Candidates to test through an owner-provided HTTP interface, in order:
Gemini 3.1 Pro (or its current successor), Gemini 3.1 Flash Lite, Claude 4.6 Opus (or successor).

**Harness ready, nothing sent:** `code/decision_eval/run_http_judge_pilot.py` (OpenAI-compatible chat completions;
same pairs / queries as the local pilot; requires `--url --model --name --yes-send-frames`; key from an environment
variable). Default pilot: 720 pairs + 400 queries = 2,240 requests. Go / no-go: pairwise accuracy on 过度生成 above the
ensemble's (0.70 / 0.79 / 0.91) or clearly above 0.6 on the pairs the ensemble gets wrong. A model behind an API can
only be used offline (scores distilled into a tower as an auxiliary target, label adjudication with owner approval):
the newer statement puts a system that needs a paid inference service outside cost tier ①.

**11.13 addendum — verified web research (workflow run `wf_9fde1805-949`, 18 sources, 24 of 25 claims confirmed by
3-vote checks; full findings in `research_notes/vlm_judge_web_research_2026-11-13.json`).**
- No published result shows a closed-source model beating the Qwen 27B family at distortion type + severity
  (DistortBench; GS-IQA severity accuracy, from verifier notes only: Qwen3-VL-235B 0.677, Gemini-3-Pro 0.563, GPT-5
  0.400, Gemini-2.5-Pro 0.363). Coverage of current frontier models is thin: absence of evidence, not proof.
- Severity at adjacent / middle grades is the weakest skill of prompted VLMs; their exact-grade error is about one
  human annotator's (rubric-prompted GPT-4o on super-resolution hallucination: Spearman 0.56 with the human mean vs
  0.54 between humans, and only with the ground-truth image; 0.47 without it).
- Where in-domain labels exist, fine-tuned 7–8B VLMs and small encoders beat prompted closed models by wide margins
  (e.g. Qwen2.5-VL-7B fine-tuned 0.70 vs GPT-4o 0.56 on the same hallucination task). The ensemble already contains a
  fine-tuned 8B VLM.
- Teacher distillation has evidence only where no human labels existed; a pairwise judge is the best-supported
  prompted mode (GPT-4V and humans both compare better than they grade), ideally against anchor frames of known grade.
- The SalArt-VQA numbers that single out Gemini 3.1 for generation artifacts were read from one summary page and are
  not part of the verified set; lower confidence than the rest.

**11.13 addendum — owner-provided endpoint (2026-11-13).** The owner supplied an internal gateway endpoint
(`http://wanqing-prt.internal/api/gateway/v1/endpoints`, model id `ep-2bbmni-1794561470122917386`; the key is not
recorded anywhere in the repository). One text request succeeded: the endpoint serves `gemini-3.1-pro-preview`
(a reasoning model: 61 reasoning tokens for a one-line question, so requests need a token budget of a few thousand).
Every later request, text or image, returned HTTP 402: the project's quota for `gemini-3.1-pro-preview-external` is
used up and has to be requested on the gateway's quota page. Six pairwise smoke-test requests (six frames) reached the
gateway and were rejected there in 0.1 s; no frame was answered by the model. The pilot is unchanged and ready:
`run_http_judge_pilot.py --no-proxy --max-tokens 4096 --defects 过度生成 --parts pairs` (360 requests) as the first step.

**11.13 addendum — Gemini 4 Argon, 过度生成 pairwise pilot (2026-11-13).** Second owner-provided gateway endpoint
(model id `ep-ke9ljw-1794575112558413277`, replies as `gemini-4-argon`; no log-probabilities, answer letter only;
key not recorded). 180 one-grade-apart 过度生成 pairs × 2 orders = 360 requests, all valid, 837 s, 0.56M prompt tokens
(`results/r55/explore4/vlm_pilot/pairs_gemini4argon.jsonl`).

| Contrast | Gemini 4 Argon | Qwen3.8-27B, same pairs | Ensemble, out of fold | Gemini on the pairs the ensemble gets wrong |
|---|---|---|---|---|
| P0 vs P1 | 0.700 [0.617, 0.783] | 0.633 | 0.667 | 0.75 (n = 20) |
| P1 vs P2 | 0.525 | 0.525 | 0.800 | 0.54 (n = 12) |
| P2 vs none | 0.525 | 0.592 | 0.900 | 0.42 (n = 6) |
| all | 0.583 | — | 0.789 | −0.206 [−0.281, −0.136] vs ensemble |

Gate failed overall: chance on the two mild contrasts, order-inconsistent on 38–45 % of those pairs. One signal: P0 vs
P1 (the costliest confusion) is level with the ensemble and right on 15 of the 20 pairs the ensemble gets wrong — too
few to act on; a P0-vs-P1-only confirmation (the 150 pairs of the local pilot, 300 requests) would decide it.

**11.13 addendum — prompt study for the 过度生成 pairwise judge (Gemini 4 Argon, owner-approved, 2026-11-13).**
`code/decision_eval/run_http_prompt_variants.py`, `eval_prompt_variants.py`; logs `logs/r55/explore4/prompt_variants_*.log`.
Development pairs (67, disjoint from the held-out pairs and from the four anchor frames), accuracy P0-vs-P1 / P1-vs-P2 /
P2-vs-none / mean: v1 (definition rewritten from the annotators' convention) 0.361 / 0.800 / 0.662 / 0.608; v2 (+ four
labelled anchors) 0.528 / 0.600 / 0.485 / 0.538; v3 (+ native-resolution crop) 0.333 / 0.500 / 0.544 / 0.459; v4
(describe, then answer) 0.417 / 0.667 / 0.632 / 0.572; ensemble out of fold 0.556 / 0.733 / 0.941 / 0.743. Picked v1.
Held-out 180 pairs, once: original prompt 0.700 / 0.525 / 0.525 / 0.583; v1 0.508 / 0.558 / 0.700 / 0.589 (14 invalid
answers, order-consistent 0.54); ensemble 0.667 / 0.800 / 0.900 / 0.789. The first run's P0-vs-P1 signal (0.70) was
chance variation (0.36 / 0.51 on fresh pairs). **Closed: a better prompt does not make this judge useful for
过度生成** (Gemini 4 Argon, definition, anchors, crops, reasoning); the remaining closed-model options are other models.

### 11.14 Native-resolution tile towers (owner: "all images trained at their native resolution"), two-fold screen (2026-11-13)

`code/r55/native_tiles.py` + additive `native_tiles` branches in `qingyu/train_tier1_5backbones_8fold.py`;
`train_testbed.py --native-tiles {tiles,tiles+global} --tile-cap 16 --batch-size 6`; presets `nt`, `ntg`;
`results/r55/explore4/nt_screen.py`, `logs/r55/explore4/nt_screen.log`. Every frame is cut into 448×448 tiles at native
resolution (every pixel covered; 720p 6 tiles, 1080p 15, 4K 45), shared pe_core_large backbone, attention / max / mean
pooling over tiles; training uses ≤ 16 random tiles per frame, inference all. `nt` = tiles only; `ntg` = tiles + the
whole frame squashed to 448×448 (a downscaled extra view). 872 validation images of folds 0–1, controls = same-recipe
RGB towers `fixed` seeds 2026–2028 (batch 12, not 6).

| | `nt` | `ntg` | controls |
|---|---|---|---|
| Single tower: mean RPS | 0.05138 | **0.04818** | 0.0509–0.0514 |
| Detection AUC / adjacent-grade AUC | 0.957 / 0.810 | 0.958 / **0.832** | 0.953–0.956 / 0.821–0.827 |
| Per-image error correlation with a control tower | 0.42–0.46 | 0.44–0.50 | 0.60–0.66 among themselves |
| Pooled RPS, added as 4th tier-1 member, minus control mean | +0.00002 [−0.00011, +0.00015] | −0.00015 [−0.00027, −0.00003] | — |
| Pooled RPS, as a family of its own, minus control mean | −0.00001 [−0.00038, +0.00038] | **−0.00053 [−0.00089, −0.00017]** | — |
| Same, per defect (family of its own) | 过度生成 −0.0008, others ≥ 0 | 过度生成 −0.0012, 虚化突兀 −0.0013, 过锐 −0.0008 | — |

Reading: tiles alone (every pixel at native resolution, nothing resized) are exactly as good as the deployed
5-view tower, with less correlated errors, but add nothing to the pool. Tiles plus a shrunk whole-frame view are 6 %
better alone and pass the gate (family of its own: ≤ −0.0003 with CI excluding 0) mainly through 过度生成, 虚化突兀 and
过锐. So local coverage is not the missing evidence on its own; coverage plus global context is. Two folds only; batch
size 6 vs 12 and the train/inference tile-count mismatch are uncontrolled. `ntg` goes to 8 folds
(`ntg:2026:2-7`, `logs/r55/explore4/queue_ntg_rest.log`).

**11.14 addendum — 8-fold result for `ntg` (2026-11-13/14; `results/r55/explore4/ntg_8fold.py`, `logs/r55/explore4/ntg_8fold.log`).**
All 3,573 images; controls = same-recipe RGB towers `fixed` seeds 2026 and 2027 (seed 2028 has two folds only).
Single tower: mean RPS 0.04978 vs 0.05101 / 0.05177 (−2.4 % / −3.8 %); detection AUC 0.955 vs 0.954; adjacent-grade AUC
0.814 vs 0.812 / 0.810. Inside the primary pool (pooled RPS, minus the control mean; decoded = nested `eu`, all rows /
first-batch rows; reference `primary + eu` 51.89 / 52.46):

| `ntg` used as | pooled RPS vs controls | decoded vs controls, all / first batch | decoded vs `primary + eu`, all rows |
|---|---|---|---|
| 4th tier-1 member | −0.00011 [−0.00017, −0.00005] | −0.12 [−0.26, −0.00] / −0.24 [−0.41, −0.05] | −0.17 [−0.33, −0.02] |
| a family of its own (weight 1) | −0.00036 [−0.00055, −0.00019] | +0.01 [−0.19, +0.25] / +0.08 [−0.23, +0.36] | −0.09 [−0.37, +0.19] |
| replacing pe | −0.00012 [−0.00021, −0.00005] | −0.08 [−0.20, +0.06] / −0.07 [−0.24, +0.12] | −0.27 [−0.43, −0.11] |

The two-fold screen overstated the gain (family of its own −0.00053 → −0.00036). The probability-quality gain is real and
sits in 虚化突兀 (−0.0011) and 过度生成 (−0.0005), but it does **not** reach the decoded score: no variant beats
`primary + eu`. **Closed; not added to the deliverable.** Tiles only (`nt`) was no better than the deployed tower in the
two-fold screen, so "every pixel at native resolution" is not the missing ingredient either.
