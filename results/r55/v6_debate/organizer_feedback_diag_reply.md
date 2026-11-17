# Diagnosis of `resident_v6` (644-image feedback) and a ranked plan

The OOF numbers you posted show the decoder is already close to the ceiling the current probabilities allow for 过度生成 and 过锐. Most of the remaining loss is upstream: weak severity ranking plus noisy grade labels. 虚化突兀 is different: it is near-separable in-distribution, and the 644 result points to a distribution shift. Nothing below has been run in this session. It is a diagnosis from your numbers plus the project notes, with the cheap checks to run first.

## 1. 虚化突兀: 0.98 out of fold but flagged on 644

**What the numbers already show:**
- **Predicted rate.** The model predicted 51 positives on 644 images (7.9%). At the ORIG prevalence of 5.7% there would be about 37 true positives. Even if all 37 were caught, F1 would be at most 2·37/(51+37) ≈ **0.84**, and each lost point of F1 costs about 5.1 objective points (it counts in both S_pos and S_p0). Treat this only as a hint about the failure mode. Do not use it to pick a threshold (AGENTS rule 2).
- **Clean = 20/20 proves nothing here.** False positives on images that have some other defect are invisible to the clean score.
- **0.98 on `tag_fold` is suspicious, not reassuring.** A defect that is near-perfectly separable on 109 positives usually means the model found a shortcut. Likely candidates are source device or app, resolution or aspect ratio, compression, or all positives coming from a few videos. A wide empty gap between positive and negative scores also means any threshold inside the gap scores 0.98 out of fold. The EU decoder, rewarding 虚化突兀 at about 5.1 points per unit of F1, will tend to place the threshold low in that gap, so negatives from a new source will cross it.

**Checks to run, in order (all without 644 labels):**
1. **Read the statement file "(2)"** for how 虚化突兀 is graded on the final set. If it has P1/P2 levels there and we only output P0, that alone would explain the flag.
2. **Shortcut probe.** Fit a small gradient-boosted model on metadata only (resolution, aspect ratio, file size, JPEG quantisation tables, file-name tokens, video id) to predict 虚化突兀. AUC ≥ 0.9 means a shortcut.
3. **Video-disjoint OOF.** Re-split by video cluster plus a perceptual-hash cluster, and for 虚化突兀 report AP and where the decoder threshold sits relative to the negative-score distribution.
4. **External false-positive rate.** Run the engine on natural shallow-depth-of-field photos (portraits, optical bokeh) from public sets. They should be about 0% positive.
5. **Synthetic recall.** Use `build_bokeh_seg.py` / `bkseg1` to apply mask-based blur with hard edges to clean training frames. Low recall means the model learned the source, not the defect.
6. **Attribution.** Run `defect_attribution_map.py` and check whether the evidence sits on subject boundaries.

**Fixes, each gated on keeping OOF ≥ 0.97 and on improving checks 4–5:**
- **No retrain:** put the threshold at the logit midpoint of the out-of-fold gap. This is chosen on va only and costs nothing on va.
- **Retrain:** add natural shallow-depth-of-field images as negatives for 虚化突兀 only (mask the loss on the other dimensions), add synthetic positives, and randomise resolution, compression and aspect ratio to break the shortcut.

## 2. 过锐 and 过度生成

**Decoder (Track A) is near its ceiling.** Current F1 is already about equal to the class AP:

| class | F1 | AP |
|---|---|---|
| 过锐 P0 / P1 / P2 | 0.563 / 0.363 / 0.438 | 0.61 / 0.36 / 0.41 |
| 过度生成 P0 / P1 / P2 | 0.757 / 0.458 / 0.516 | 0.83 / 0.47 / 0.53 |

Max-F1 is normally at or above AP for weak classes, so the decoder is already getting what the ranking allows. Your λ sweep agrees, and the notes record a total decoder head-room of ≤ 0.4 in-sample versus nested.

**Why P1/P2 lose to P0.** A unit of P0 F1 is worth 1.79 + 3.33 = 5.12 points, against 1.79 for P1 or P2. The decoder therefore rationally moves borderline P1 to P0 (25 such rows for 过锐, 32 for 过度生成), and argmax→EU on 过锐 trades P2 (0.51→0.44) for P0 (0.34→0.56). This is correct under the current scoring and only becomes wrong if the organizers add a per-dimension floor.

**过度生成: the 147 无→P2 rows are the expected cost of maximising F1, not a bug.** The F1-optimal threshold is about F1\*/2 ≈ 0.26, so P2 is over-predicted 368 vs 279. Two checks:
- How many of the 147 are on images that have other defects? Many would mean defects bleeding into each other in the features.
- Sibling-label agreement for 过度生成: from the notes, siblings of a P1 frame are labelled P2 33% / P1 34% / P0 23%.

**过锐 is a mix of label noise, batch drift and representation:**
- **Noise.** The true-P1 row is almost flat (17 / 27 / 35 / 22% across 无/P2/P1/P0), and same-video frames get the same grade only 44–61% of the time. That puts a hard ceiling on P1/P2.
- **Batch drift.** On `supp`, P0 is predicted 180 times against 100 true and P1/P2 recall is about 0.2. The severity scale differs by batch for this one dimension. Check the per-batch confusion for 过锐 specifically; earlier per-batch decoder tests were pooled over all dimensions.
- **Representation.** Detection AP is 0.869, the lowest of all six, and the tile/downsampled members are the weakest on it. Before training anything, re-score the existing `hp` and `ntg` OOF on 过锐 AP alone. Both were closed on pooled RPS, which can hide a gain on one dimension, so this costs zero GPU. Then run a CPU probe of halo/overshoot features at native resolution on edge pixels (halo width, overshoot amplitude, ringing) added to the stacker. Gate: 过锐 AP +0.02 over 3 control seeds. Check the LEDGER first that this wasn't among the "11 CPU checks".

## 3. Ranked plan

| # | Action | Cost | Expected gain / risk |
|---|---|---|---|
| 0 | Get from the organizers the exact per-dimension baseline metric and floor | none | Decides the decoder objective; a floor makes constrained decoding legitimate (still fit on va only) |
| 1 | 虚化突兀 checks 1–6 | CPU + 1 GPU-hour | Largest per-dimension swing: 0.84→0.95 is about +0.5 points |
| 2 | 虚化突兀 midpoint threshold, then hard-negative/synthetic retrain | 8×H200 for about half a day | Low risk if both gates hold |
| 3 | Re-score `hp`/`ntg` on 过锐 alone, plus the halo feature probe | ≈0 GPU | Decides whether a native-resolution high-pass specialist is worth building |
| 4 | 过锐 native-resolution high-pass specialist, only if step 3 passes | 8-fold, 8×H200 | +0.1–0.3; the label-noise ceiling limits it |
| 5 | Per-dimension constrained decoder, only if a floor is confirmed | CPU | Trades Doc2 for floor safety |
| — | P1/P2 loss changes (`p0_sample_boost`, EMD, ordinal) | — | Skip: radj/rsame harmed, soft targets and losses already closed |
| — | Synthetic 过度生成 data / zero-shot VLM help | — | Skip: SynthLab null, ogk teachers see nothing (P 0.001), VLMs near chance on 过度生成 |

**Gating for every Track B run:** full 8-fold, video-grouped folds, at least 3 control seeds, CI90 on decoded Doc2 and on the target dimension's F1. Two-fold screens overstated gains by about 30% before. Then a fresh full-fit and fresh inference (AGENTS rule 1).

**Expectations:** realistically about +0.3–0.8 objective in total, mostly from 虚化突兀. 过度生成 P1/P2 is limited mainly by label consistency. The one lever left there is more labelled frames from the same videos, and that depends on the owner, not on modelling.
