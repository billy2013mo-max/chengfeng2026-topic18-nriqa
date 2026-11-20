# Error analysis of the `fallback` ensemble: where the points are lost (OOF, Doc2_Strict)

Setup: fallback = cradio_v4_h + pe_core_large + siglip2_so400m_naflex (tier-1 family) + v16_full8 + phoenix p3, family-balanced mean (`code/r55/final_engine.py` `oof_pool(SETS["fallback"])`). Decoder55 is nested over the 8 video-grouped folds with objective `doc2_strict_total`. N = 3,573 labelled images, all out of fold. No test set, no legacy JSON, no GPU. The reproduced reference matches the brief: **nested 51.347** (pos 15.49 / p0 16.25 / clean 19.61); first batch 52.110; argmax 51.10; decoder fitted in-sample 51.85.
Sources (all local):
- [S1] `results/r55/explore4/dump_fallback.py` → `logs/r55/explore4/dump_fallback.log`
- [S2] `results/r55/explore4/error_analysis.py` → `logs/r55/explore4/error_analysis.log`, `results/r55/explore4/error_analysis.json`
- [S3] `results/r55/explore4/error_followup.py` → `logs/r55/explore4/error_followup.log`

Unless stated otherwise, CIs are 95 % cluster bootstraps over `group_id` (300–1000 resamples), with the nested predictions held fixed.

## Q1. Per component and per label: where the 13.65 lost points are

### Takeaway
S_pos loses 9.51 of the 13.65 points, S_p0 loses 3.75 and clean only 0.39. The P2/P1 positive labels alone lose 7.3 points (54 % of the gap). By defect, **过度生成 (4.31 pts) > 过锐 (3.02) > 模糊 (2.71) > 纹理损失 (2.19) ≫ 虚化突兀 (0.93) > 花屏 (0.10)**.

### Cited Findings
- Component losses: S_pos 9.51/25, S_p0 3.75/20, clean 0.39/20, total 13.65/65 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- S_p0 sub-terms: macro P0 recall 0.843 (costs 10·(1−R) = 1.57), F1 0.796 (costs 1.23), precision 0.762 (costs 0.95). The decoder trades precision for recall, as the 0.5 weight on recall rewards — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- Per-label accounting. Loss = 25/14·(1−F1), plus the 20/6·(0.5(1−R)+0.3(1−F1)+0.2(1−P)) term for P0 labels. Each label's loss has a cluster-bootstrap CI [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log):

| label | n_gold | n_pred | P | R | F1 | loss (CI) |
|---|---|---|---|---|---|---|
| 过度生成_P0 | 190 | 303 | 0.495 | 0.789 | 0.609 | **1.78** [1.53, 2.02] |
| 过度生成_P1 | 218 | 92 | 0.402 | 0.170 | 0.239 | **1.36** [1.25, 1.47] |
| 过锐_P0 | 225 | 217 | 0.770 | 0.742 | 0.756 | **1.26** [1.05, 1.51] |
| 过度生成_P2 | 276 | 204 | 0.402 | 0.297 | 0.342 | 1.18 [1.10, 1.27] |
| 纹理损失_P0 | 762 | 750 | 0.812 | 0.799 | 0.806 | 1.00 [0.88, 1.11] |
| 过锐_P1 | 328 | 255 | 0.533 | 0.415 | 0.467 | 0.95 [0.87, 1.04] |
| 模糊_P1 | 499 | 371 | 0.563 | 0.419 | 0.480 | 0.93 [0.86, 1.01] |
| 虚化突兀_P0 | 225 | 259 | 0.753 | 0.867 | 0.806 | 0.93 [0.74, 1.13] |
| 模糊_P2 | 649 | 623 | 0.498 | 0.478 | 0.487 | 0.92 [0.86, 0.99] |
| 模糊_P0 | 563 | 639 | 0.770 | 0.874 | 0.819 | 0.87 [0.75, 0.99] |
| 过锐_P2 | 730 | 761 | 0.540 | 0.563 | 0.551 | 0.80 [0.74, 0.86] |
| 纹理损失_P2 | 1003 | 843 | 0.720 | 0.605 | 0.658 | 0.61 [0.57, 0.65] |
| 纹理损失_P1 | 1038 | 1179 | 0.635 | 0.722 | 0.676 | 0.58 [0.54, 0.62] |
| 花屏_P0 | 213 | 216 | 0.972 | 0.986 | 0.979 | 0.10 [0.04, 0.17] |

- Oracle substitution at the prediction level. Gain if the label were predicted perfectly, i.e. rows with gold = g get g and rows falsely predicted g get their gold. Cross-label effects are included, so gains are not additive. Results: 过度生成_P0 +2.81, 过度生成_P1 +2.50, 过度生成_P2 +2.17, 过锐_P1 +2.12, 模糊_P1 +1.89, 纹理损失_P1 +1.88, 过锐_P2 +1.67, 模糊_P2 +1.61, 过锐_P0 +1.54, 模糊_P0 +1.30, 纹理损失_P0 +1.22, 纹理损失_P2 +1.20, 虚化突兀_P0 +0.93, 花屏_P0 +0.10, all clean rows perfect +0.40 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- Whole defect perfect: 过度生成 +4.31 (to 55.66), 过锐 +3.02, 模糊 +2.75, 纹理损失 +2.43, 虚化突兀 +0.93, 花屏 +0.10 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- 过度生成_P1 is badly under-predicted: 92 predicted vs 218 gold, recall 0.17. 过度生成_P0 is over-predicted: 303 vs 190, precision 0.495 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)

### Inferences
- 过度生成 is the single largest target: 4.3 points with a CI clear of every other label. Its P0 precision problem and its P1 recall problem are one confusion, P1 graded as P0 (see Q2).
- Under the metric, a rare label weighs as much as a frequent one: 过度生成_P1 (218 gold) carries 2.3× the loss of 纹理损失_P1 (1,038 gold). Gains per corrected cell are therefore largest on the rare 过度生成/过锐 grades.
- 花屏 is solved (0.10). The clean score is at 98 % of its ceiling. Neither is worth roadmap effort on OOF.

### Gaps
- Per-label metric losses are exact accounting, but the oracle substitutions interact (fixing one label changes another label's FP/FN), so they cannot be summed into a budget.

## Q2. Detection errors vs grading errors per graded defect

### Takeaway
**Grading dominates for every graded defect.** Perfect grading of the detected defects would add +9.32 (to 60.67), while perfect detection with model grading adds only +3.54 (to 54.89). Almost all grading errors are off by one grade (92–98 %).

### Cited Findings
- Confusion matrices (rows = gold none/P2/P1/P0, cols = pred) — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log):
  - 模糊 `[[1679,157,13,13],[222,310,96,21],[35,142,209,113],[4,14,53,492]]`
  - 纹理损失 `[[643,80,32,15],[131,607,254,11],[22,152,749,115],[5,4,144,609]]`
  - 过锐 `[[2061,211,16,2],[252,411,62,5],[24,125,136,43],[3,14,41,167]]`
  - 过度生成 `[[2809,56,5,19],[124,82,35,35],[33,49,37,99],[8,17,15,150]]`
  - 花屏 none→P0 6, P0→none 3
  - 虚化突兀 none→P0 64, P0→none 30
- Detection (defect vs none) and grading among rows detected by both gold and prediction [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log):

| defect | det TP / FP / FN | det F1 | grading errors (off-by-1 / off-by-2) | grade acc given detected | over / under-graded |
|---|---|---|---|---|---|
| 模糊 | 1450 / 183 / 261 | 0.867 | 439 (404 / 35) | 0.697 | 230 / 209 |
| 纹理损失 | 2645 / 127 / 158 | 0.949 | 680 (665 / 15) | 0.743 | 380 / 300 |
| 过锐 | 1004 / 229 / 279 | 0.798 | 290 (271 / 19) | 0.711 | 110 / 180 |
| 过度生成 | 519 / 80 / 165 | 0.809 | 250 (198 / 52) | **0.518** | 169 / 81 |

- Headroom oracles (gain over nested, cluster CI) [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log):
  - (a) Decoder fitted in-sample: 51.85, +0.50 [0.29, 0.72].
  - (b) Perfect detection + model grading: 54.89, +3.54 [3.20, 3.94]. Model grading uses each fold's own decoder biases restricted to P2/P1/P0.
  - (c) Model detection + perfect grading: 60.67, +9.32 [8.86, 9.88].
- Per-defect (b)/(c) gains: 模糊 +0.60 / +2.01, 纹理损失 +0.48 / +1.89, 过锐 +0.57 / +2.27, 过度生成 +0.75 / +3.16. For the P0-only defects, perfect detection gives 虚化突兀 +0.93 and 花屏 +0.10 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- The most costly single confusion cells, as the gain from fixing only that cell [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log):

| defect | gold → pred | cells | gain from fixing |
|---|---|---|---|
| 过度生成 | P1 → P0 | 99 | **+1.35** |
| 纹理损失 | P0 → P1 | 144 | +0.71 |
| 过锐 | P0 → P1 | 41 | +0.69 |
| 模糊 | P1 → P0 | 113 | +0.66 |
| 过度生成 | P2 → none | 124 | +0.61 |
| 过锐 | P1 → P2 | 125 | +0.57 |
| 过锐 | P1 → P0 | 43 | +0.54 |
| 虚化突兀 | none → P0 | 64 | +0.51 |
| 过度生成 | P1 → P2 | 49 | +0.50 |
| 虚化突兀 | P0 → none | 30 | +0.43 |

  Per corrected cell, the rare-grade 过度生成/过锐 cells are worth 1.3–1.9 points per 100 cells, against 0.15–0.5 for 纹理损失/模糊 cells.

### Inferences
- The plateau is an ordinal-grading problem, not a detection problem. The P1↔P0 boundary is the most expensive because it feeds both S_pos and S_p0. Losses that target ordinal calibration (e.g. CORN/cumulative-link heads, grade-boundary-specific features or crops) are where the headroom is. Better defect-vs-none detection is worth at most about 3.5 points.
- 过度生成 grading is close to chance among detected rows (0.518) and skews to over-grading (169 vs 81). That fits the decoder pushing P0 recall (pred P0 303 vs gold 190): the P0 recall floor/bias for 过度生成 may be too aggressive. The nested decoder already optimises that trade-off, though, so the fix probably has to come from better P1/P0 probabilities rather than from thresholds.
- Decoding headroom is small: an optimistic in-sample decoder adds only +0.50. More decoder search is not a route past 52.

### Gaps
- Oracle (b) depends on how model grading is defined for an oracle-detected row. Here it is the argmax of the fold decoder's biased logits over P2/P1/P0; other definitions would shift (b) somewhat.

## Q3. Clean images

### Takeaway
The clean score is 19.61/20, so only 0.39 points of headroom. 10 of the 513 clean images get any defect: 9 are P2-level, mostly 纹理损失, and 1 is P0. The gate's other cost, 88 defective images predicted all-clean, is worth +0.29.

### Cited Findings
- 503/513 clean images are predicted all-clean. 10 get some defect; their worst-grade histogram is {P2: 9, P0: 1}. Only 1 gets any P0, and it is flagged on both 模糊 and 纹理损失. By dimension: 纹理损失 9 (6 with no other defect), 模糊 2, 过锐 2, none on 花屏/过度生成/虚化突兀. Each wrong clean image costs 0.039 points — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- 591 rows are predicted all-clean. 88 of them are gold-defective, with gold worst grade P2 73, P1 13, P0 2, mostly mild 纹理损失 (68 rows) and 模糊/过锐 P2. Making those 88 rows perfect would give +0.29 — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- The clean oracle (all clean rows perfect) gives +0.40 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)

### Inferences
- The clean term is saturated on OOF. On a 654-image final set (about 94 clean) one false P0 matters much more under Doc1's cliff, but under Doc2_Strict each error is linear and small. No roadmap item should target clean images unless the online metric is Doc1.

### Gaps
- None material.

## Q4. First batch vs supplement (52.11 vs 51.35 overall)

### Takeaway
On first-batch rows the ensemble scores higher in both S_pos (+0.35 vs all) and S_p0 (+0.41 vs all). The 0.76 gap is not a prevalence artefact: prevalence works *against* the first batch (−0.52 in S_pos), and the per-label TPR/FPR accuracy gap is +0.87. The first batch contains an easy subpopulation (first-batch timestamp frames, stock JPGs). Per-row error rates on defective images are statistically indistinguishable between the batches.

### Cited Findings
- Scores: all 51.35 (pos 15.49, p0 16.25, clean 19.61); first batch 52.11 (15.84, 16.66, 19.61); supplement 51.05 (15.13, 15.91, 20.00). Cluster CIs: first batch [51.22, 52.90], supplement [50.29, 51.69] — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- First batch − supplement: total CI [−0.01, 2.21] (borderline), S_pos +0.70 [0.11, 1.33], S_p0 +0.75 [0.17, 1.34] — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- Clean convention artefact: all 513 clean images are in the first batch, so a supplement-only score gets clean = 20 for free. With the 513 clean images attached, the supplement scores 50.64. The first batch without its clean images scores 52.53. Like for like, the batch gap is therefore about 1.5–1.9 points — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- S_pos decomposition (first batch − all = +0.35). The prevalence part is −0.52: F1 recomputed with all-row TPR/FPR at first-batch prevalence, which is lower for almost every label (e.g. 模糊_P2 0.127 vs 0.244 in the supplement). The accuracy part is +0.87. Largest accuracy gaps: 模糊_P0 F1 0.889 vs 0.755, 过锐_P2 0.609 vs 0.505 (TPR 0.728 vs 0.461), 纹理损失_P0 0.852 vs 0.769, 过度生成_P1 0.282 vs 0.195. FPR on P0 labels is about 3× higher in the supplement (模糊_P0 0.077 vs 0.025, 纹理损失_P0 0.080 vs 0.026, 过度生成_P0 0.066 vs 0.028) — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- P0 macro R/P/F1: first batch 0.853 / 0.799 / 0.823, supplement 0.835 / 0.734 / 0.772. The S_p0 gap comes mainly from precision — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- Wrong dimensions per defective row: first batch 1.037 [0.97, 1.10], supplement 1.071 [1.02, 1.12], with similar label load (gold defective dims per row 2.22 vs 2.30, severity sum 4.36 vs 4.37) — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- Same tag, different batch: timestamp frames are easy in the first batch (0.45 wrong dims per row, 1.93 defective dims) and hard in the supplement (1.43, 2.91). v8L3HiV6 and other_untagged are actually *better* in the supplement (1.16 vs 1.44; 0.63 vs 1.15) — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- Supplement composition: v8SlHiV4 489, v8L3HiV6 390, cn_defectword 277, timestamp 276, other_untagged 112, v8MdHiV4 84, v8L3HiVE 43. v8SlHiV4 has only 19 first-batch rows and cn_defectword none — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- By tag, oracle gain per 100 rows (all rows of the tag perfect) and wrong dims per row: v7L3HiV2 0.86 / 1.51 (170 rows, all first batch), v8L3HiV9 0.73 / 1.20, v8L3HiVE 0.62 / 0.95, v8L3HiV6 0.52 / 1.23, cn_defectword 0.46 / 1.18, v8L3HiV1 0.45 / 1.27, other_untagged 0.40 / 0.99, timestamp 0.29 / 0.55, v8SlHiV4 0.29 / 0.89, stock_jpg about 0 (76/77 clean). Total gain by tag: timestamp +3.07 (1,066 rows), v8L3HiV6 +2.68, other_untagged +1.96, v7L3HiV2 +1.47, v8SlHiV4 +1.46, cn_defectword +1.28 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)

### Inferences
- The first batch's advantage looks like *case mix*: easy timestamp frames, plus defects in the supplement that sit on more boundary-heavy grade combinations, which raise P0 false positives from neighbouring grades. It does not look like a per-image model deficit. Because prevalence works against the first batch, a test set that looks like the first batch is not penalised by its lower prevalence; it is helped by easier images.
- Older v7 pipeline frames (v7L3HiV2) are the costliest per row. They are 170 first-batch rows, and practice/final frames come largely from first-batch videos (see memory note explore3). That makes them a sensible hard-example target.

### Gaps
- Tags are derived from file names (the `build_tag_split.py` regex). The source of untagged and timestamp frames is unknown, so per-source conclusions for those groups are coarse.
- Per-tag sub-scores have different label prevalences and are not comparable across tags. Wrong dims per row and oracle gain per 100 rows are used instead, without CIs.

## Q5. Error concentration and confident errors (label noise)

### Takeaway
Errors cluster moderately within videos: a frame is 1.5–3.4× more likely to be wrong when its sibling is wrong, and per-video error counts are over-dispersed by about 1.4×. But 2,826 of the 3,136 videos have a single frame, so "hard videos" are effectively "hard images". About 10–15 % of each graded defect's errors are confident (predicted-grade probability > 0.8). Sibling frames disagree with each other more than the model disagrees with gold, so label inconsistency puts a hard ceiling on grading accuracy near the model's current level.

### Cited Findings
- 3,136 videos/sources: 2,826 single-frame, 231 two-frame, 59 three-frame, plus a few larger (one with 14 frames, two with 8) — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- Wrong cells: 3,244 in total. 41 % of videos have none; the top 10 % of videos hold 34.9 % of the wrong cells and the top 25 % hold 64 %. Variance of per-video wrong-cell counts is 4,472 against about 3,244 under independent rows, a 1.38× over-dispersion. Fixing the 100 / 400 most costly videos (4.5 % / 14.8 % of rows) would gain +3.10 / +7.78, against +0.45 / +1.84 for random videos. That selection is post hoc, so it only shows that the loss is spread over hundreds of images, not a few — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- Sibling error correlation, P(sibling wrong | this frame wrong) against the base error rate: 模糊 0.36 vs 0.25, 纹理损失 0.42 vs 0.27, 过锐 0.46 vs 0.22, 过度生成 0.47 vs 0.14, 虚化突兀 0.20 vs 0.03 — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- Confident errors (predicted-grade pooled probability > 0.8) out of all errors: 模糊 144/883, 纹理损失 111/965, 过锐 125/798, 过度生成 77/495, 虚化突兀 29/94, 花屏 3/9. Confident missed detections (P(none) > 0.8 but gold defective): 模糊 115, 过锐 116, 过度生成 73. Confident false detections (P(none) < 0.2 on gold none): 纹理损失 90, 模糊 65, 过锐 61 — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- Sibling label agreement (310 multi-frame videos, 700 pairs): all six dimensions agree 0.443. Per dimension: 0.764 / 0.646 / 0.767 / 1.0 / 0.857 / 0.984; at detection level: 0.863 / 0.903 / 0.856 / 1.0 / 0.95 / 0.984. When both frames are gold-defective, the *grade* agrees only 0.60 (模糊), 0.61 (纹理损失), 0.57 (过锐) and 0.50 (过度生成). The model's grade accuracy given detection is 0.70 / 0.74 / 0.71 / 0.52 — [S3](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_followup.log)
- Sibling-gold predictor, labelling each multi-frame row with a random sibling's gold (20 draws): Doc2S 46.68 (pos 13.65, p0 13.92) on those 664 rows, against the model's 52.33 (pos 15.81, p0 16.70) — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)
- Of 86 confident errors on multi-frame videos, 45 (52 %) have a sibling frame whose gold equals the model's prediction. The rate over all 575 errors on multi-frame videos is 44.5 % — [S2](file:///wanqing-develop/luowenjing/18/logs/r55/explore4/error_analysis.log)

### Inferences
- Either labels are noisy at the grade level or grades genuinely vary between frames of one video. In both cases the grade signal per image is weak: a "perfect video-level" predictor scores *below* the model. The 9-point "perfect grading" headroom (Q2) is therefore largely not attainable. A rough reading is that much of the residual P2/P1 confusion is at or near the annotation-consistency floor, especially for 过度生成, where siblings agree on grade only 50 % of the time and the model reaches 52 %.
- Confident errors whose sibling carries the predicted label (52 %, slightly above 44.5 % for all errors) are candidates for relabelling or down-weighting. Only about 86 cells are involved on multi-frame videos, so a relabelling audit is cheap. Single-frame videos cannot be checked this way.
- Hard-example mining by video offers little leverage beyond per-image reweighting, because 90 % of sources are single frames.

### Gaps
- True annotator noise cannot be separated from genuine frame-to-frame change without repeated annotation of the *same* image. No such data was found in the repo.
- The large videos (14 and 8 frames) contribute 147 of the 700 pairs and may bias the agreement rates. The rates were not re-weighted per video.
