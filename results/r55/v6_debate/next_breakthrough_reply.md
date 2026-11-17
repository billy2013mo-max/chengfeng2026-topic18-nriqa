Short answer: nothing I can see will clear your gate. Three cheap checks are worth running, and after them there is one GPU bet. Everything else on your list has already been measured as null or harmful. This comes from reading the ledger, the earlier debate replies and the bench file; I ran nothing new. I tried to save it to `results/r55/v6_debate/next_breakthrough_reply.md`, but that write wasn't approved.

## 0. Corrections to the premise
1. **Your two gates describe different sizes of gain.**
   - Adding `p1_qy` moved `RPS_ORIG` by −0.00032 but mean4 ORIG Doc2S by only +0.099 (prune round 1). So `dRPS_ORIG ≤ −0.00035` is worth about **+0.1 Doc2S**, not +0.35..0.80.
   - A real +0.35 needs about **−0.0011 RPS_ORIG** (−3 %). No single member has done that since `resident_v3`; `p12_pe`, the last recipe that passed, gave −0.00021.
   - +0.35 is also within the ±0.3 rerun noise of the decoded score, and a 654-image draw has sd ≈ 0.56. A gain you could see on the final set would have to be about +0.6.
2. **The parameter count looks too low.** The measured sizes (V16 3.3B, p1 8.77B, PE/C-RADIO 0.33–0.68B each) add up to about 13.6B, not 9.656B. The bench probably counts 4-bit tensors at half size, because they pack 2 weights per byte. The 30B cap still isn't binding, with about 16B of room rather than 20B.
3. **Memory is what limits you, not parameters.** The `nvidia-smi` peak of 21.16 GiB already breaks the ≤20 GiB safety gate proposed in prune round 1, leaving about 2.3–2.8 GiB on a real 4090. The room is in latency: 1.98 s against a 15 s budget. Adding a member means dropping one or loading members one at a time.
4. **The decoder has no room left.**
   - In-sample 51.711 vs nested 51.693, and decoder v2 was null on honest OOF.
   - FP=220 on `模糊_P2` is the decoder deliberately trading misses for false alarms under Doc2_Strict. It isn't a miscalibration you can recover.
5. **The ceiling is the labels.** 过度生成 grade AUC is about 0.74–0.77, and two frames of the same video get the same human grade only 44–61 % of the time.

## 1. Ranked directions

**R1 — Free CPU check: average two `p1_qy` seeds.** s2027 already exists on 6 of 8 folds.
- **Test:** average the two seeds' probabilities on those folds and refit with the decoder settings frozen.
- **Expected:** ΔRPS_ORIG −0.00003…−0.00012. It will probably miss the gate.
- **Shipping it:** two copies don't fit in memory, so it would ship as one merged model (weight soup). That only works if the merged model matches the averaged probabilities closely (TV < 0.01).
- **Go/no-go:** CI90 < 0 on the 6 folds, then train the 2 missing folds. If not, stop adding seeds for `p1_qy`.

**R2 — Free CPU check: does per-dimension decoder tuning have any ceiling?**
- **Test:** fit `(lam_cl, p0_scale)` separately for each of the 6 dimensions (12 parameters), plus a per-dimension P2/P1 boundary offset, in sample.
- **Go/no-go:** if the in-sample gain is under +0.30, a nested gain can't beat noise, so close the decoder for good.
- **Expected:** +0.05–0.15 in sample, about 0 nested. It's roughly 30 minutes of CPU and settles the decoder question.

**R3 — The one GPU bet: `p1_qy` v2.**
- **Change:** replace the two 384 crops with the `p12` semantic regions (skin/face/hand, text/strokes, matte/bokeh), and raise the main view from 784 to about 1024 on the long side.
- **Why this one:** semantic regions were the only recipe since v3 that passed on PE. 过度生成 and 过锐 grading depends on fine texture in those regions, and the VLM is the only member whose views haven't been through this kind of change yet.
- **Cost:** about +0.5–1.0 GB memory (stop if the `nvidia-smi` peak goes over 22.5 GiB) and about +0.3 s per image. Training 8 folds of an 8B model is the most expensive item here.
- **Expected:** ΔRPS_ORIG −0.0001…−0.0003, Doc2S +0.03…+0.10.
- **Gates:**
  - A 2-fold screen against both existing seeds, s2026 and s2027. The effect must be more than twice the gap between those two seeds, because 2-fold screens have overstated gains by about 30 %.
  - Then all 8 folds, judged as a replacement for `p1_qy`, not an addition: CI90 < 0, and nested Doc2S ≥ 0 on 4 of 4 partitions.
  - Then fresh full-fit inference, N=1 == N=150, and the bench gate.

**R4 — Optional, only if R3 passes:** an ordinal grade head for 过度生成 only, trained with smoothed neighbour-grade targets. On PE towers every boundary loss was null or harmful (nT, radj, rsame, rank ladders), so expect ≤ +0.05.

## 2. Don't spend time on (all already measured)
- **New or orthogonal backbones, or replacing a PE variant:** DINOv3, ConvNeXt, SigLIP-giant, PE-G14, C-RADIO native tiles, InternViT-6B on top of v5 + p1 (zero gain), high-pass tower, twin towers. Keep `peg`: its 花屏 robustness on unseen sources doesn't show up in OOF.
- **Synthetic data for grades:** ogsr1, the FLUX ladder, ogk, bkseg1, benign invariance. Synthetic frames can show that a defect is present, not how severe it is. The glitch data is the one exception, and it's already in `peg`.
- **Tower loss changes:** ranking losses, soft labels, noise transition, clean boosting, random crops, other KD variants.
- **Decoder work beyond R2:** per-batch decoders, a cross-defect stage, mutual exclusion, max or geometric pooling, temperature.
- **Others:** sibling or kNN look-ups, file-name or image-size features, TTA, public IQA metric banks, zero-shot VLM judges, KVQ self-training.
- **Anything chosen by a practice upload** (rule 2).

## 3. Expected outcome
- **Best case**, if R1 and R3 both pass: about +0.1–0.15 Doc2S on orig rows, still below what one final-set draw can show.
- **Expected:** about +0.05.

The large measured effects left are on the data side: about +1 Doc2S per +25 % more labels, and −10 % RPS when a model has seen other frames of the same video. Using them would need more labelled first-batch frames or re-adjudicated 过度生成 P1/P0 labels, and that's the owner's call. Without new data, I'd run R1–R3, then freeze `resident_v6_core`.
