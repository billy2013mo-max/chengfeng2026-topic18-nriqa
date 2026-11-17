# Decoding, calibration, structured constraints and ensemble blending (probabilities to grades) for Doc2_Strict

Scope: how much objective score is left in the step from the pooled (N,6,4) probabilities to the six discrete grades. All numbers below are out of fold. Every threshold, temperature, weight and statistic for fold f was fitted on the other seven folds only. Deltas are paired **video-cluster bootstrap** (1000 draws, clusters from the video-id regex in `code/r55/eval_sib_study.py`), reported as point [CI90] and P(delta>0). The pool is `final_engine.oof_pool(SETS["fallback"])` (cradio, pe, sig, v16, p3, family-balanced arithmetic mean), N=3573 rows, 3136 videos, 1902 first-batch rows (`orig`), and 513 clean images, all of them in `orig`. Scripts are in `results/r55/explore4/decoding/` and logs in `logs/r55/explore4/decoding/`. Nothing reads test_150, its distribution or legacy JSONs. Another agent was also writing files in `results/r55/explore4/` (its `dump_fallback.py` / `fallback_pool.npz` overwrote mine), so all my files sit in the `decoding/` subfolder with the `dec_` prefix.

Correction to the brief: the official Clean term in the statement is **not** max(0, 1 − FPR/τ). It is "fraction of fully-clean images predicted clean on all six dimensions" (any defect at any grade counts as an error). That is what `doc2_strict_total` (`doc2_Clean2`) implements ([statement, appendix table S_clean](documents/【2026乘风挑战赛】命题18：晴羽观感体系·画质BadCase的无参检测模型%20(2).md)). The statement also says Precision is 0 when there are no predicted positives, which the scorer matches.

Reproduced baselines ([check.log](logs/r55/explore4/decoding/check.log)). The fast scorer `dec_lib.fscore` is asserted equal to `common55.score` on 20 random predictions.

| decoder (fallback pool) | all rows | first batch |
|---|---|---|
| argmax | 51.10 | 51.60 |
| Decoder55 single start, nested | 51.35 | 52.11 |
| **Decoder55 4 starts, nested (shipped = reference below)** | **51.28** | **52.12** |
| Decoder55 4 starts, fitted on all rows (in-sample, optimistic) | 51.84 | 52.58 |

The gap between in-sample and nested is 0.56 points, which is the optimism of the 21-parameter search.

## Q1. What does Decoder55 implement, and is it the metric-optimal decision rule?

### Takeaway
Decoder55 is an argmax over log p + per-(dimension, grade) log-biases. On top of that it applies a per-dimension P0 floor and a global clean gate on ∏_d P(无问题), with all 21 parameters chosen by coordinate ascent on the empirical Doc2_Strict. This is "threshold tuning on validation data". The literature says it is consistent in principle but prone to overfitting, and coordinate-wise search is not guaranteed to find the optimum for coupled F-measures. It is not the Bayes rule for this metric. The plug-in rule for F1 says to predict a label when its calibrated probability exceeds a statistic-dependent threshold (F*/2 for F1). For an ordinal grade the natural rule thresholds cumulative probabilities, not the argmax of the grade mass.

### Cited Findings
- Decoder55 decision rule: `pred[d] = argmax_k log p[d,k] + b[d,k]` (无问题 bias fixed at 0). Then `p[d,P0] >= f0[d]` forces P0, and `prod_d p[d,无问题] >= t_clean` forces all six to 无问题. Biases are searched on a grid from −2 to 2 in steps of 0.25, floors on 9 values and the gate on 12 values, for 3 sweeps, best of 4 starting shifts (0, 0.3, 0.6, 0.9) — [decoder55.py](code/next55/decoder55.py), [decoder_select.py](code/r55/decoder_select.py)
- For a calibrated probabilistic classifier, the F1-optimal threshold is half the optimal F1 score. An uninformative classifier should predict everything positive, and the paper says macro-averaged F1 can hide this extreme thresholding behaviour — [Lipton, Elkan & Narayanaswamy 2014](https://arxiv.org/pdf/1402.1892)
- For metrics that are ratios of linear functions of the confusion counts (accuracy, F-measure, Jaccard), the Bayes-optimal binary classifier thresholds the conditional probability at a metric-dependent value. Plug-in estimation plus threshold selection is consistent — [Koyejo et al., NIPS 2014](https://papers.neurips.cc/paper_files/paper/2014/hash/98053046e0dce5c7d946c67b96a85e18-Abstract.html)
- Multiclass complex measures are handled as optimisation over feasible confusion matrices: Frank–Wolfe for concave measures, bisection for ratio-of-linear measures (e.g. micro-F1). These algorithms are provably consistent — [Narasimhan et al., ICML 2015](https://proceedings.mlr.press/v37/narasimhanb15.html). Extended to metrics with constraints in [Narasimhan et al., JMLR 2024](https://jmlr.org/papers/v25/22-1137.html)
- F-measure maximisation reduces to a series of cost-sensitive problems with unknown costs. The journal version states that thresholding cost-insensitive scores can be sub-optimal — [Puthiya Parambath et al., NIPS 2014](https://proceedings.neurips.cc/paper/2014/hash/678a1491514b7f1006d605e9161946b1-Abstract.html); [arXiv 1505.00199](https://arxiv.org/pdf/1505.00199v4.pdf)
- An exact expected-F maximiser needs only m²+1 parameters of the joint distribution (versus Jansche's O(m⁴) procedure) — [Dembczyński et al., NIPS 2011](https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization). A plug-in rule built on multinomial regressions is consistent, whereas structured-SVM approaches are not — [Dembczyński et al., ICML 2013](https://proceedings.mlr.press/v28/dembczynski13.html). Surrogate losses for F can have high worst-case regret — [Waegeman et al., JMLR 2014 / arXiv 1310.4849](https://web3.arxiv.org/pdf/1310.4849)
- Tuning a threshold directly to maximise the F-measure "has been observed to overfit easily". Per-label thresholds are an effective macro-F method, and the FBR heuristic is offered as a fix — [Fan & Lin, threshold selection for multi-label classification](https://www.csie.ntu.edu.tw/~cjlin/papers/thresholding/smooth_acm.pdf). For the F of macro precision/recall a coordinate-wise optimum need not be the maximum (Berger & Guda, as summarised in search results; I did not open the full text)
- Ordinal decision theory: under 0–1 loss the Bayes prediction is the mode (argmax), under absolute-error loss it is the median, and the median depends only on the order of the labels — [Dembczyński/Kotłowski slides, PL-09](https://ke-tud.github.io/events/PL-09/04-Dembczynski-Slides.pdf); [Kotłowski, ordinal LNAI](https://www.cs.put.poznan.pl/wkotlowski/research/LNAI-ordinal.pdf)

### Inferences
- Doc2_Strict is a sum of 14 one-vs-rest F1s, six P0 recall/precision/F1 terms and a row-level clean term. Every term is a ratio of corpus counts, so the "joint optimisation across labels sharing denominators" is already done by Decoder55's empirical search. Its weakness is variance (21 thresholds on step functions), not a missing joint term. The 0.56 in-sample−nested gap shows this.
- The argmax-plus-bias parametrisation puts the boundary in log-odds between *pairs* of grades. Ordinal grades split mass between adjacent grades (P1/P0), so a row can have high P(≥P2) and still have 无问题 as its argmax. Thresholds on cumulative probabilities (the median family) fit graded labels more naturally. Q2 tests this.

### Gaps
- No paper treats this exact composite: macro-F1 + recall-weighted P0 score + an all-six-correct clean term. The marginal-gain rule in Q2 is my first-order derivation, not a published algorithm.
- I did not run a Frank–Wolfe or bisection confusion-matrix optimiser (Narasimhan et al.). The metric is not concave and not a single ratio, so its guarantees would not apply directly.

## Q2. Expected-score-maximising decoder variants (nested OOF)

### Takeaway
Two rules beat the shipped decoder on every nesting partition. The best is **a near-parameter-free expected-utility (EU) plug-in decoder** with one fitted clean-penalty scale: **+0.52 all rows, +0.29 first batch** on average over four nesting partitions. Its first-batch CI90 includes 0 on two of the four partitions. A **per-cell majority vote of shipped/ordinal/EU** is the steadiest choice: +0.26 all rows, +0.28 first batch, P(delta>0) ≥ 0.93 everywhere, narrow CIs. **Ordinal cumulative thresholds** gave +0.26 / +0.25 on the fallback pool but did not replicate on the primary pools. Realistic decoding headroom over the shipped rule is about +0.2–0.4 on first-batch-like data.

### Cited Findings
All results come from local runs: [decoders.log](logs/r55/explore4/decoding/decoders.log), [robust.log](logs/r55/explore4/decoding/robust.log), [eu_lambda.log](logs/r55/explore4/decoding/eu_lambda.log). The scripts are [dec_exp_decoders.py](results/r55/explore4/decoding/dec_exp_decoders.py), [dec_robust.py](results/r55/explore4/decoding/dec_robust.py) and [dec_eu_lambda.py](results/r55/explore4/decoding/dec_eu_lambda.py).

Rule definitions:
- **Ordinal**: grade = highest k with P(grade ≥ k) ≥ t[d,k], for k = P2, P1, P0. Binary dimensions use P(P0) ≥ t. It adds the same clean gate as Decoder55. Fifteen thresholds are fitted by coordinate ascent, best of 4 starts.
- **EU plug-in**: first-order marginal gain of predicting label l with calibrated probability p is

  w_F(2p − F)/D + w_R·p/n_pos + w_P(p − P)/n_pred,  where D = n_pred + n_pos

  - The F-part is Lipton's F*/2 rule.
  - w_F = 25/14 (+ 20·0.3/6 for P0), w_R = 20·0.5/6 for P0, w_P = 20·0.2/6 for P0.
  - Per dimension, predict the grade with the largest positive gain.
  - A row is set fully clean if its summed gain < λ·q_clean·20/n_clean, with q_clean = ∏_d P(无问题).
  - F, P, n_pos, n_pred and n_clean are fixed-point statistics from the **training folds**, scaled to the size of the decoded set. Every term scales as 1/N, so the decision is per-image and size-invariant, and it works with the one-image-at-a-time engine.
  - No search, except λ ∈ {0.25, 0.5, 1, 2, 4} in the "fitted λ" variant.

Fallback pool, canonical folds, delta vs shipped:

| variant | all rows | first batch |
|---|---|---|
| Decoder55 single start | +0.06 [−0.09, +0.21] | −0.01 [−0.21, +0.19] |
| argmax | −0.18 [−0.53, +0.16] | −0.52 [−1.04, −0.01] |
| ordinal cumulative thresholds | **+0.31 [+0.11, +0.53]**, P 0.99 | +0.29 [−0.01, +0.58], P 0.94 |
| ordinal, finer grid (0.025) | +0.33 [+0.09, +0.57] | +0.29 [−0.05, +0.64] |
| EU plug-in, no clean term | −0.34 [−0.72, +0.04] | −0.61 [−1.07, −0.16] |
| EU plug-in (λ=1, zero search) | +0.26 [−0.07, +0.61] | +0.03 [−0.41, +0.48] |
| EU + nested per-dim temperature | +0.22 [−0.11, +0.58] | +0.07 [−0.38, +0.53] |
| **EU + fitted λ** | **+0.40 [+0.10, +0.72]**, P 0.99 | +0.10 [−0.33, +0.52] |
| EU, transductive statistics (see compliance note) | +0.32 [−0.00, +0.66] | +0.09 [−0.34, +0.55] |
| vote(shipped, ordinal, EU) | +0.24 [+0.08, +0.42], P 0.99 | +0.18 [−0.02, +0.39], P 0.93 |

Same comparison on three extra random 8-way nesting partitions (video/group components kept together):

| partition | shipped (all / first) | ordinal Δ (all / first) | EU+λ Δ (all / first) | vote Δ (all / first) |
|---|---|---|---|---|
| canonical | 51.28 / 52.12 | +0.31 / +0.29 | +0.40 / +0.10 | +0.24 / +0.18 |
| random0 | 51.15 / 51.92 | +0.35 / +0.33 | +0.60 / +0.43 | +0.26 / +0.30 |
| random1 | 51.07 / 51.81 | +0.29 / +0.29 | +0.60 / +0.43 | +0.33 / +0.34 |
| random2 | 51.30 / 52.07 | +0.10 / +0.09 | +0.47 / +0.19 | +0.22 / +0.31 |
| **mean** | **51.20 / 51.98** | **+0.26 / +0.25** | **+0.52 / +0.29** | **+0.26 / +0.28** |

- The shipped decoder's own nested score moves by ±0.1–0.2 across partitions. This is partition noise from re-fitting 21 thresholds.
- On the vote rows, P(delta>0) is ≥ 0.93 in all eight cells and the CI90 lower bound is > 0 in seven of eight.
- For EU+λ, the all-rows CI90 lower bound is > 0 on all four partitions. The first-batch lower bound is > 0 on two (random0 [+0.03, +0.84] and random1 [+0.06, +0.86]); canonical [−0.33, +0.52] and random2 [−0.23, +0.66] cross 0.

Replication on other probability pools (canonical folds, delta vs shipped; EU here means λ = 1):

| pool | ordinal (all / first) | EU (all / first) |
|---|---|---|
| member cradio | +0.45 / +0.63 | +0.86 / +1.00 |
| member pe | +0.14 / +0.33 | +0.86 / +1.13 |
| member sig | +0.22 / +0.14 | +0.92 / +0.57 |
| member v16 | −0.04 / −0.03 | +0.49 / +0.66 |
| member p3 | −0.02 / +0.14 | **−0.54 / −0.49** |
| primary set | +0.10 / +0.08 | +0.29 / −0.04 |
| primary_v2 set | −0.15 / −0.04 | +0.25 / +0.32 |

- Component breakdown: EU wins mostly on the Clean term (19.96 vs 19.61, i.e. ~9 fewer clean images flagged) and gives back some positive-F1. Ordinal improves positive-F1 (15.61 vs 15.47) and P0.

### Inferences
- The searched decoder's in-sample optimism (0.56) is about the same size as the gain from the EU rule, which has almost no free parameters. This fits the literature warning that F-threshold tuning overfits. A rule derived from the metric plus calibrated probabilities transfers better than 21 searched thresholds.
- EU depends on calibration. It helps strongly on well-pooled or individual tier-1/v16 members. It fails on p3 alone, which loses 1 point of positive-F1 even though p3 has the lowest member ECE (0.022). Top-label ECE evidently does not capture the per-label calibration EU needs. Treat EU as pool-specific and re-validate it on whatever pool ships.
- The steadiest candidate is the vote, because it averages three different rules' errors. The highest mean is EU+λ. Both are deployable per image (each needs only training-fold statistics) and comply with AGENTS.md rule 2.
- Compliance: the "transductive" EU re-estimates F/P/n_pos from the decoded set's own probabilities. That uses the test set's (unlabelled) distribution, which AGENTS.md rule 2 forbids ("不得使用测试集分布信息做后验校准"). It also gave nothing beyond the fitted-λ version. **Do not use it.**
- Scale: the bootstrap is over 1902 first-batch rows. On the 654-image final set the score SD is about √(1902/654) ≈ 1.7× larger, so a +0.3 gain will not reliably show up in a single final reading.

### Gaps
- I did not test EU+λ and the vote on the primary / primary_v2 pools across partitions. Only λ = 1 EU and ordinal were tested there.
- I did not log the λ values chosen per fold.
- The first-order (linearised) gain ignores the change in F when many predictions flip at once. An exact expected-F decoder (Dembczyński 2011) would need the joint label distribution and was not attempted.

## Q3. Temperature / vector scaling, geometric vs arithmetic pooling, rank averaging, per-defect learned weights

### Takeaway
The family-balanced arithmetic pool is already very well calibrated (top-label ECE 0.009, NLL 0.365). None of the calibration or blending variants improves the nested decoded score. Calibrate-then-pool, geometric pooling and rank averaging are significantly *worse* on first-batch rows (−0.4 to −0.8). Per-dimension NLL-learned member weights are null (+0.07 [−0.15, +0.31] all rows, +0.01 first batch). **There is no headroom in pooling or calibration.**

### Cited Findings
- Local nested results ([pool.log](logs/r55/explore4/decoding/pool.log), [dec_exp_pool.py](results/r55/explore4/decoding/dec_exp_pool.py)). Each pool is decoded by the shipped Decoder55 (4 starts). Columns are held-out NLL / ECE, then Δ vs shipped pool for all rows and first batch:

| pool | NLL / ECE | Δ all rows | Δ first batch |
|---|---|---|---|
| individual members | 0.40–0.55 / 0.022–0.075 | — | — |
| arith family-balanced (shipped) | 0.365 / 0.009 | 0 | 0 |
| arith equal member weights | 0.370 / 0.017 | −0.02 [−0.36, +0.36] | −0.31 [−0.75, +0.12] |
| geometric (log-linear) | 0.383 / 0.037 | −0.09 | **−0.48 [−0.84, −0.10]** |
| per-member per-dim temperature, then arith (calibrate-then-pool) | 0.372 / 0.030 | −0.19 [−0.41, +0.02] | **−0.46 [−0.85, −0.12]** |
| pooled per-dim temperature (pool-then-calibrate) | 0.364 / 0.004 | −0.03 [−0.22, +0.15] | −0.19 [−0.44, +0.04] |
| geometric + pooled temperature | 0.366 / 0.006 | −0.09 | −0.41 [−0.77, −0.03] |
| per-dim NLL weights, shrink 0.5 to family prior | 0.365 / 0.010 | +0.04 | −0.24 |
| per-dim NLL weights, no shrink | 0.365 / 0.011 | +0.07 [−0.15, +0.31] | +0.01 [−0.30, +0.33] |
| rank average (quantile-mapped back) | 0.383 / 0.010 | −0.14 | **−0.83 [−1.23, −0.46]** |

- A temperature on top of the EU rule was also null (+0.22 vs +0.26 without; Q2).
- Temperature scaling (one scalar) is often the most effective post-hoc calibrator for modern networks — [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)
- For ensembles, pool-then-calibrate beats calibrate-then-pool on ECE, NLL and Brier — [Rahaman & Thiery, NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/file/a70dc40477bc2adceef4d2c90f47eb82-Paper.pdf). Individually calibrated members do not guarantee a calibrated ensemble and can make it under-confident — Wu & Gales 2021, as summarised in the same search; I did not open the full text
- Dirichlet / vector-scaling calibration (log-probs, then a linear layer, then softmax) generalises temperature scaling to classwise miscalibration — [Kull et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)

### Inferences
- The local data repeats the Rahaman & Thiery ordering: per-member temperature before pooling is the worst calibration choice, and calibrating after pooling is harmless. The pool barely needs calibrating, because the arithmetic mean of over-confident members is already close to calibrated. Decoder55's log-biases also absorb any monotone per-dimension miscalibration, so calibration cannot add decoded score.
- I did not run full Dirichlet/vector scaling. Pooled temperature already brings ECE to 0.004 with no score gain, and the decoder biases are a superset of per-class vector-scaling offsets, so a gain is unlikely.

### Gaps
- Dirichlet (full-matrix) calibration was not run.
- I did not try stacking with a learned meta-model on member probabilities. Earlier project work found logistic stacking worth +0.3 only with file-name features, a shortcut that was not used.

## Q4. Class-prior adjustment toward first-batch priors, and fitting the decoder on first-batch rows

### Takeaway
Neither helps measurably on first-batch rows. Fitting the decoder on first-batch rows only gives +0.08 [−0.22, +0.37]. Duplicating first-batch rows gives −0.16 [−0.38, +0.05]. Saerens-style prior shift with training-fold priors gives −0.15 [−0.40, +0.13]. The prior differences between batches are absorbed by the decoder's biases, and batch-specific fitting only adds variance. This agrees with the earlier per-batch-decoder null (−0.11).

### Cited Findings
- Local, nested ([decoders.log](logs/r55/explore4/decoding/decoders.log)), delta vs shipped decoder:

| variant | all rows | first batch |
|---|---|---|
| decoder fitted on first-batch training rows only | +0.09 [−0.11, +0.33] | +0.08 [−0.22, +0.37] |
| first-batch rows ×2 in the fit | −0.05 | −0.16 [−0.38, +0.05] |
| prior shift p·π_orig/π_all (priors from training-fold gold), then shipped decoder | +0.02 | −0.15 [−0.40, +0.13] |

- First-batch versus all-rows label priors differ substantially. Examples: 纹理损失 无问题 35% vs 22%; 模糊 P0 14.5% vs 15.8%; and all 513 clean images are first-batch ([dec_pool.npz statistics](results/r55/explore4/decoding/dec_dump_pool.py)).
- The label-shift EM correction adjusts posteriors to new priors without refitting. It always improved accuracy when priors actually changed in the original experiments — [Saerens, Latinne & Decaestecker, Neural Computation 2002](https://mlanthology.org/neco/2002/saerens2002neco-adjusting)

### Inferences
- Prior correction is aimed at accuracy (0–1 loss). Here the decision layer is refitted on the metric anyway, so a prior shift applied after fitting only moves already-optimised thresholds. Estimating new priors by EM on the test set would be test-distribution calibration, which rule 2 forbids, so it was not run.
- Fitting on first-batch rows was the one positive point estimate (+0.08). It is well inside noise and halves the fit data, so it is not recommended.

### Gaps
- Whether the 654-image final set resembles the first batch is unknown. Memory notes that 95/150 practice frames come from first-batch videos.

## Q5. Structured constraints across defects and clean-image gating

### Takeaway
Some real exclusivity exists in the labels, but enforcing it does not help. Clean-image gating with a learned clean detector is also null, because the Clean term is already 19.61/20 under the shipped decoder. The EU rule's clean term (Q2) is the only clean-side mechanism that gains (+0.35 Clean). It does so by trading the gate for a per-row comparison of expected defect gain against expected clean loss.

### Cited Findings
- Gold co-occurrence, observed vs expected under independence ([co-occurrence run](results/r55/explore4/decoding/dec_exp_decoders.py)):

| pair | observed | expected | ratio | shipped decoder predicts both | of those, gold has both |
|---|---|---|---|---|---|
| 过锐 & 花屏 | 5 | 76.5 | 0.07 | — | — |
| 过度生成 & 虚化突兀 | 7 | 43.1 | 0.16 | 53 rows | 1 |
| 花屏 & 过度生成 | 0 | 40.8 | 0 | — | — |
| 模糊 P0 & 过锐 P0 | 0 | 35.5 | 0 | — | — |
| 模糊 & 过锐 | 418 | 614 | 0.68 | 344 rows | 149 |
| 模糊 P0 & 纹理损失 P0 (positive correlation) | 551 | 120 | — | — | — |

  - 模糊 & 过锐 is the only pair that is anti-correlated but still common.
  - "Forbidden" combinations found on all rows (never seen, ≥ 8 expected): (模糊P0, 纹理P2), (模糊P0, 过锐P0), (过锐P0, 花屏P0), (花屏P0, 过度生成P2/P1/P0), (花屏P0, 虚化P0), (过度生成P0, 虚化P0).
- Nested results ([decoders.log](logs/r55/explore4/decoding/decoders.log)):

| variant | all rows | first batch |
|---|---|---|
| pair constraints fitted inside Decoder55 (keep the more confident defect) | −0.08 [−0.22, +0.06] | −0.17 [−0.39, +0.03] |
| pair constraints applied post hoc | −0.00 | −0.05 |
| learned logistic clean detector as the gate (inner cross-fitted) | −0.02 [−0.12, +0.08] | −0.07 [−0.21, +0.05] |
| EU rule without its clean term | −0.34 | −0.61 |

  - The last row shows that a clean-aware term is essential. A plain F/recall plug-in floods clean images.

### Inferences
- The visible conflict (过度生成 + 虚化突兀 predicted together on 53 rows, 52 wrong) involves rows where *both* labels are uncertain. Dropping the less confident one mostly converts one false positive into a false negative of similar cost, which is why the constraints are flat.
- The clean term is close to saturated (98% of clean images correct), so gating has ≤0.4 points of ceiling and costs recall elsewhere. EU recovers part of that ceiling because its clean decision weighs each row's defect evidence against q_clean, rather than applying one global threshold on ∏ P(无问题).

### Gaps
- A full joint decoder over the 6-dimensional label vector (e.g. pairwise CRF with co-occurrence potentials learned from gold) was not tried. The flat constraint results suggest a small ceiling.

## Q6. Overall: how much objective score is left in decoding, and what survives nesting?

### Takeaway
About **+0.25 to +0.5 points** (all rows) and **+0.25 to +0.3** (first-batch rows) are available from the decision rule. Nothing measurable is left in calibration, pooling, priors or cross-defect constraints. Two candidates are worth considering for shipping, both compliant, per-image and fitted only on OOF:
- **(a) EU plug-in with a fitted clean λ**: mean +0.52 all rows / +0.29 first batch over four nesting partitions. It depends on calibration and failed on p3 alone.
- **(b) per-cell majority vote of shipped Decoder55, ordinal-threshold decoder and EU**: +0.26 / +0.28, the steadiest, with CI90 lower bound > 0 in 7 of 8 partition×subset cells.

Before shipping either, re-validate it on the exact shipping pool (primary vs fallback). Ordinal replicated poorly on the primary pools.

### Cited Findings
- Summary of measured nested deltas vs shipped (local logs listed in Q2–Q5):

| family | variants | result |
|---|---|---|
| decision rule | EU+λ, vote, ordinal | positive (Q2 tables) |
| calibration / pooling | 8 variants | −0.83 … +0.07 first batch (Q3) |
| prior / first-batch | 3 variants | −0.16 … +0.08 (Q4) |
| constraints / gating | 3 variants | −0.17 … −0.02 (Q5) |
| transductive EU | — | no better, and non-compliant (Q2) |

- Threshold tuning for F-measures overfits easily — [Fan & Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/thresholding/smooth_acm.pdf). Shipped-decoder optimism here is 0.56 points (in-sample 51.84 vs nested 51.28) ([check.log](logs/r55/explore4/decoding/check.log))

### Inferences
- Decoding cannot break the plateau by itself. +0.3 on a ~52 baseline is real but likely smaller than the 654-image sampling SD of a single final reading (I did not re-measure that SD here; `results/r55/explore3/best_of_3.py` computes it). The remaining gap to 55 must come from the probabilities, i.e. the models and data.
- If one change is adopted, the vote is the lower-risk choice. It never lost on any partition or subset and its CI is roughly half the width of EU's.

### Gaps
- Not checked: whether EU+λ or the vote survives on the `primary`/`primary_v2` pools across partitions, and its effect on the Doc1 objective. Both are easy follow-ups with `dec_robust.py` / `dec_eu_lambda.py`.
