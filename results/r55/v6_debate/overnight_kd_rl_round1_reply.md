I read the KD code in `code/r55/p12_routed_head.py` (L355–417) and `build_clean_kd_teacher.py`, and checked them against what earlier rounds found. I did not run anything new, because the H200s are busy with the P0 probe. Everything below is argument. The experiments I propose are at the end.

---

## 0. Problems with the P0 gate itself (fix these before reading its result)

1. **Passing the gate does not mean KD will pay off.** The gate measures the teacher added as an ensemble member. A KD student typically recovers only about 30–60% of the gap between teacher and student. If the teacher just clears −0.00020, the distilled 4090 student would gain about −0.0001, which can't be told apart from noise on 2 folds. I'd add a second, stricter threshold before spending GPU time on KD, roughly **dRPS_ORIG ≤ −0.0005**.
2. **The control has to add a member too.** Adding any model and re-weighting lowers RPS a little through variance reduction alone. The honest comparison is "+teacher" versus "+one more p1_qy seed". R1 already showed that +seed gives nothing, so this costs little, but report both numbers side by side. Fix the teacher's weight in advance (e.g. family-balanced, 1/(n_families+1)) or choose it nested. Don't pick it after seeing the folds.
3. **The baseline must use the honest V16 predictions.** The current ensemble needs to use the out-of-fold predictions from `v16_qy`, not the frame-leaky V16. A leaky baseline looks stronger than it is, so it would hide some of the teacher's gain.
4. **Rules question to raise with you:** the statement says ≤30B parameters if trained. Neither the 32B nor the 38B model can ship, so KD is the only way either can reach the deliverable. Whether a >30B model may be used as a training-only teacher is your call, and it should be on record.

---

## 1. Unconstrained ceiling and the noise floor

### Realistic upper bound

The evidence points to saturation from limited data, not limited capacity:
- 9 models (`resident_v6`) score the same as 6 (`core`).
- Second seeds add nothing.
- PE-Core-G was no better than PE-Core-L.
- R1 (8B, 2 seeds) failed its gate.
- The 21-model ensemble scored only 50.46 under the strict protocol.
- Halving the labels costs about 2 Doc2S, so data is the steep axis.
- Visual audits show that many confident "errors" are label-convention choices (for example, the same swirl labelled 过度生成 in one place and 虚化突兀 in another).

My estimate if every limit is lifted:

| | Today | Unconstrained estimate |
|---|---|---|
| RPS_orig | 0.04231 | **0.0410–0.0418** (−1% to −3%) |
| Doc2S_orig | 51.87 | **≈52.2–52.7** (+0.3 to +0.8), most likely at the low end |

Most of that comes only if the VLM teachers bring a truly new kind of error (a different family), not from stacking more models. I'd put the chance of passing the P0 gate at roughly 30–40%. Non-linear stacking over 9 members × 24 probabilities on 3.1k rows will overfit unless it is nested. Earlier pooling and stacking variants were all null, so I'd expect ≤ +0.1 from that alone.

There is a cheap way to measure this tonight on CPU: a nested stack of every honest out-of-fold prediction we already have (resident_v6 9 models + pe_core_g14). That gives the ceiling without the 32B/38B. Adding the P0 teacher's out-of-fold predictions later gives the full figure.

### Estimating the noise floor

- **Teacher–teacher disagreement is the wrong estimator.** It measures variance between models, which shrinks as models are added. Two teachers fine-tuned on the same labels also share the same label biases, so their agreement says nothing about whether the labels are right.
- **Teacher–gold residuals mix model error and label noise**, and you can't separate the two from one model.
- **Out-of-fold calibration curves don't identify the floor either.** The Murphy split is RPS = reliability − resolution + uncertainty. The irreducible part is the resolution the true posterior would reach, and that is not observable from one label per image.

Three estimators that do work, best first:

1. **Near-duplicate sibling pairs (CPU, about an hour).** Within each video, find frame pairs that are nearly the same image (DINO cosine > 0.98 or SSIM > 0.95), from labelled training rows only.
   - If two labels are independent draws from p(y|x), then E[RPS(δ_y1, y2)] = 2 · E[Σ_k F_k(1−F_k)]. That is twice the RPS of a perfect Bayes predictor.
   - So **Bayes RPS ≈ ½ × RPS of the sibling pairs.** It is an overestimate if the content really differs between the frames, and an underestimate if annotators copied labels across a video.
   - Report it alongside the known sibling transition rate (siblings of 过度生成 P1 frames are only 34% P1). If ½·sibling-RPS comes out near 0.040, we are already at the floor.
2. **Learning-curve asymptote (GPU, cheap towers).** Fit Doc2S and RPS at 25/50/75/100% of the labels, grouped by video, to `a − b·n^−c`. The gap between the fitted `a` and today's score is the most anyone can get from more data. The gap between towers and the ensemble at each n is the most anyone can get from more capacity.
3. **Blind re-labelling of about 200 rows from folds 0 and 1.** This needs a person, so it's your decision. It is the only direct measurement and is worth more than any model run.

---

## 2. KD design

What I found in the current code:
- The KD term uses `T=1.25` on both sides with T² scaling.
- It **never touches `clean_logit`**.
- It is **not class-weighted**, while the gold KL is (×1.35 on clean images).
- For the P0-only dimensions it renormalises the teacher over grades {0, 3} (L413). If the teacher outputs all 4 grades there, the mass on grades 1–2 is silently dropped and P(3) is inflated. Check what the teacher npz contains.

### a) Ordinal / CDF KD: add it, don't replace KL

- For one dimension, E_{y~q}[RPS(p, y)] = Σ_k (P_k − Q_k)² + const. So **squared CDF distance is exactly the KD loss that matches the RPS metric.**
- KL is also proper, so with unlimited capacity both losses reach p = q. They differ only when the student can't fit the teacher exactly.
  - CDF-L2 has weak gradients in the tails.
  - The routed gate decoder makes its P0 and Clean decisions on tail probabilities, P(y=3) and P(all dims = 0), which is exactly where KL's log penalty is strongest.
- **Recommendation:** `L_kd = KL(T=1) + β·Σ_k (P_k − Q_k)²` with β ≈ 1.
- **Set T to 1.** With 4 ordinal classes and an already calibrated teacher there is little hidden information for a higher temperature to expose. T > 1 mainly flattens the tails the decoder relies on.
- **Add clean-gate binary KD:** `BCE(clean_logit, teacher P(clean))`. Take P(clean) from the teacher's own clean head, not from the product of the six marginals, because the dimensions are correlated. Expect a small effect: decoder v2's learned P(clean) was null.

### b) Cross-fitting: halfA/halfB cripples the teacher

- halfA/halfB trains each teacher on 0.5 × 7/8 ≈ **44% of all data**. By our own scaling, half the labels costs about 2 Doc2S. A teacher that weak is likely **worse than the 100% ensemble it is meant to teach**. That would explain why the cross-fitted soft labels (`xd`) helped single towers but did nothing inside the ensemble.
- Two different leaks need handling:
  - **Self-target leak:** a teacher trained on row i just copies its label back, so the KD signal is useless.
  - **Evaluation leak:** a teacher that saw the outer validation fold makes the evaluation dishonest.
- Coverage of the options:

| Protocol | Share of all data per teacher | Teacher runs |
|---|---|---|
| halfA/halfB | 44% | 2 per outer fold |
| Inner 4-fold | 66% | 4 per outer fold |
| Leave-two-folds-out (exclude outer k and the row's own fold j) | **75%** | 13 per family for folds 0 and 1 (pair {0,1} shared) |

- **Deployment needs none of this.** For the full-fit student, the targets are simply the standard 8-fold out-of-fold predictions (87.5% coverage, no leak). The 4-fold, 5-fold or leave-two-out machinery is needed only to *evaluate* KD honestly.
- **Cost-aware plan:** use the 4-fold inner scheme for 1 outer fold × 1 teacher family as the KD feasibility test. Use leave-two-out only if that passes.

### c) Selective KD: don't gate on disagreement with gold

- Masking rows where the teacher and gold differ by more than one grade means selecting on the label. Those are exactly the rows where either the label is noisy (the teacher is the useful signal) or the teacher is wrong. You can't tell which without more information, and the gold loss already covers them.
- Down-weighting high-entropy rows is the opposite of what RPS needs. Those soft targets carry the ambiguity that calibration is about.
- **If anything:** use a **per-dimension λ_d** (6 parameters) set from the teacher-vs-student RPS gap measured on inner folds, never on the outer validation fold. R2 failed with 18 parameters, so keep this one coarse.

### d) Feature KD versus logit-only KD: logit/CDF only

- Qwen3-VL-8B and 32B have different hidden widths (4096 vs 5120). Feature KD would need a learned projector trained on 3.1k images, which is weak supervision. Pulling the student's DoRA adapters toward reconstructing teacher features also competes with the readout objective.
- **One exception worth noting:** InternVL3.5-38B uses an InternViT-6B vision encoder, the same family as v16, with matching widths, so alignment is mechanically easy.
  - It only helps if the QLoRA run actually tuned the vision tower. If LoRA was on the language model only, the teacher's ViT features are the pretrained ones and carry no task information. Check the config before discussing this further.
- **Where KD can actually add information** is on **unlabelled in-domain frames**: the extra frames from training videos and the external frames you allowed. Pseudo-labels already recovered 25–32% of the value of real labels. On the labelled rows, a teacher trained on those same labels mostly adds smoothing. Put the KD budget there.

---

## 3. RL viability

### a) RL at the decision layer: redundant, and it loses information

- **RPS:** REINFORCE on a sampled grade with reward −RPS(δ_a, y) optimises Σ_a p(a)·RPS(δ_a, y). That is linear in p, so its optimum is a vertex: **it collapses the predicted distribution to a single grade and destroys the calibration RPS rewards.**
  - If the policy instead outputs the distribution itself, reward −RPS(p, y) is differentiable. That is just supervised learning, which `loss_rps` (L375–378) already does.
- **Doc2:** with calibrated posteriors, the expected-reward-maximising act per image is the Bayes act, which the EU decoder already computes.
  - RL could only help by implicitly recalibrating. Decoder v2's calibration was null.
  - Or through the non-decomposable corpus-level part of the metric, which the coordinate decoder already handles with a few parameters.
- **Signal per image:** RL gets about one scalar reward per image, versus about 6 × log₂4 bits of supervised signal, at higher variance, on 3.1k images.
- **DPO on boundary pairs** is a log-ratio margin loss, which the existing `bmargin` term already provides.
- **Detectability:** nested-decoder Doc2S moves ±0.3 between reruns of one recipe, so any gain RL could plausibly deliver is smaller than the noise.

### b) Visual CoT with RLVR/GRPO on a 32B model: not viable for this score

RLVR works when the base model already has the ability and the reward is clean, i.e. pass@k is well above pass@1. Here:
- The taxonomy is an annotator convention. Earlier teacher probes recognised almost none of the generated defects (ogk: P ≈ 0.001).
- The reward carries the label noise.
- The defects are low-level and perceptual. Text reasoning doesn't add visual resolution, because the visual tokens are fixed.

On calibration: you can still read grade-token probabilities after each CoT and average over k samples. But GRPO is known to collapse entropy, so you would need per-dimension temperature recalibration on the validation set.

The result also can't be deployed: it is over 30B and CoT breaks the 15 s per image limit. It would have to be distilled again, adding a second lossy step.

**Verdict:** if the supervised 32B fails P0, CoT-RL is a long shot not worth funding. If it passes, CoT-RL is at best a better teacher, and the cheapest version of that question is already answered by P0.

### c) Falsifiable experiments for tonight (≤2 h each)

**E1. Decision-layer RL versus EU (CPU only, about 30 min, no conflict with the probe).**
- Input: the honest 8-fold out-of-fold features from `resident_v6_core` (6×4 probabilities plus P(clean)).
- Policy: a small network that outputs decision logits.
- Training: REINFORCE/GRPO with a group baseline and reward = per-image Doc2 contribution, nested exactly like `routed_gate_eu`.
- Second arm: the same network trained on a differentiable expected-Doc2 surrogate.
- Pre-registered pass: RL beats nested `routed_gate_eu` by **≥ +0.3 Doc2S_orig with 10k video-cluster bootstrap CI90 > 0**.
- Prediction: RL ≈ surrogate ≤ EU, which would confirm the redundancy.

**E2. Necessary condition for CoT-RL ("headroom" test).**
- Setup: on 300 fold-0 validation images, sample 8 inspection CoTs per image from Qwen3-VL-8B with a taxonomy prompt plus few-shot examples taken from training rows, and read the 6×4 grade-token probabilities after each CoT.
- Measures:
  1. How much the grade distribution varies across CoTs.
  2. Oracle best-of-8 RPS versus the RPS of the averaged distribution.
- Kill criterion: if the oracle best-of-8 RPS beats the mean by less than 10%, or a variance-versus-correctness AUC is below 0.6, GRPO has nothing to amplify and RL is closed.
- Resources: the H200s show about 55 GB free each, so a vLLM 8B instance at low memory use could share a GPU but would slow the probe. I'll only launch it if you approve; otherwise it runs after P0 finishes.

---

## Recommended order

1. **Now, CPU:** E1, the sibling-pair noise floor, and the nested stack of everything we already have.
2. **When P0 lands:** judge it against the "+seed" control, then apply the stricter −0.0005 threshold before doing any KD.
3. **Only if it passes:** a 4-fold inner-cross-fit teacher on 1 outer fold, KD loss = KL(T=1) + CDF-L2 + clean BCE, applied to labelled rows plus unlabelled in-domain frames.
4. **E2 only if E1 surprises us.**

Shall I start the CPU experiments (E1, sibling noise floor, nested stack) now?
