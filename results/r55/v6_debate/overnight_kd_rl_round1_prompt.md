# Overnight Deep Technical Debate (Round 1): Unconstrained Ceiling, KD Design, and RL Viability

We are running an overnight `/goal` investigation on Topic 18 (NR-IQA, 6 dims x 4 ordinal grades `{0,1,2,3}`, ~3,573 labelled images across video-grouped 8-fold CV, evaluated by `RPS`, `mAP14`, and `Doc2_Strict` = `Pos_orig + P0_orig + Cln_orig`).

## Current State & Running Experiments
1. **Shipped 4090 Baseline (`resident_v6_core`, 6 models)**:
   - Honest 8-fold nested `routed_gate_eu`: `Doc2S all=51.597, orig=51.873 (Pos_orig=15.81, P0_orig=16.30, Cln_orig=19.77)`, `RPS=0.04231`, `mAP14=0.6650`.
   - `resident_v6` (9 models) is ~identical (`Doc2S ~51.6`).
   - `R1` (2-seed `p1_qy` 8B) and `R2` (18-param per-dim EU) both failed their pre-registered gates (`NO-GO`).
2. **Currently Running on 8xH200 (P0 Mega-Teacher Ceiling Probe on Folds 0 & 1, 2 seeds `2026` & `2027`)**:
   - GPUs 0–3: `Qwen3-VL-32B-Instruct` (bf16 DoRA `r=24`, `bs=4, ga=2`, ~85.4 GB VRAM, ~5.3h ETA).
   - GPUs 4–7: `InternVL3_5-38B-HF` (4-bit NF4 QLoRA + standard LoRA `r=24`, `bs=4, ga=2`, ~80 GB VRAM).
   - Pre-registered P0 gate: `add-teacher dRPS_ORIG <= -0.00020` with 10,000-draw paired video-cluster bootstrap `CI90 < 0`.

## Questions to Argue Rigorously (Be Adversarial & Specific)
Please challenge our assumptions and debate every technical detail of these 3 questions:

### 1. Unconstrained Offline Ceiling & Irreducible Label Noise
- If we lift ALL 24GB VRAM / 15s latency / 6-model limits (e.g. combining 32B + 38B VLM teachers + `pe_core_g14` 1.88B + `resident_v6` 9m + non-linear stacking), what is the **realistic upper bound** on `RPS_orig` and `Doc2S_orig` on ~3,573 images?
- How should we empirically estimate the **Bayes / annotator noise floor** on Folds 0 & 1 (e.g. via teacher-teacher disagreement vs teacher-gold residuals, or out-of-fold calibration curves) so we don't chase noise?

### 2. Knowledge Distillation (KD) Technical Details (If P0 Passes)
Look at our current `p12_routed_head.py` KD implementation:
- It uses `F.softmax(log(tp)/T)` + `F.kl_div` (categorical KL at `T`, `lambda_kd=0.35`).
- Debate these 4 specific design choices for distilling `32B + 38B` teachers into 4090 students (`p1_qy` 8B in `vlm_c233397_qy8.py` and `v16` 6B):
  a) **Ordinal-aware KD loss**: Should we replace or augment categorical KL with **CDF / Wasserstein-1 (squared CDF difference, i.e., soft-RPS KD)** + **clean-gate binary KD**, given our downstream metric is `RPS` + `TrueGateConstrainedEU`?
  b) **Cross-fitting leak-free teacher protocol**: `build_clean_kd_teacher.py` used `halfA/halfB` (50% data per teacher). For a 32B/38B teacher on 3,139 rows/fold, does 50% data (~1,570 rows) cripple the teacher, and should we instead use **K-fold inner cross-fitting** (e.g. 4-fold or 5-fold inner CV = 75–80% data per teacher) or leave-one-fold-out?
  c) **Selective / Disagreement-Gated KD**: Should students distill uniformly on all samples, or downweight/mask samples where teacher entropy is high or teacher disagrees with gold by >1 grade?
  d) **Feature/Representation KD vs Logit-only KD**: Given `p1_qy` (Qwen3-VL-8B) shares the same family as `Qwen3-VL-32B`, is logit/CDF KD alone optimal, or is there any value in penultimate hidden-state alignment?

### 3. RL (GRPO / REINFORCE / DPO / Policy Gradient) Viability
The user explicitly wants us to determine whether **RL** is viable here (either with or without VRAM limits):
- Given that `RoutedGateEUDecoder` already computes the **Bayes-optimal decision rule** for `Doc2_Strict` (`Pos`, `P0`, `Clean2`) from calibrated marginal probabilities `P(y_d | x)`:
  a) Can end-to-end RL (e.g. REINFORCE / GRPO directly optimizing `Doc2_Strict` reward over `{0,1,2,3}^6` actions, or DPO on boundary pairs) beat Bayes-optimal EU decoding of calibrated probabilities, or is it mathematically redundant / strictly higher-variance on 3.1k images?
  b) What about **RLVR / GRPO with Visual CoT** on a 32B VLM (generating inspection reasoning traces before outputting the 6x4 grades, rewarded by `-RPS` or `Doc2_Strict`)? Could CoT-RL unlock higher unconstrained ceiling, or does discrete token generation destroy probability calibration needed for `RPS` and `EU`?
  c) What concrete **falsifiable mini-experiment** (<= 2 hours on H200) could we run tonight if we want to empirically test RL's viability rather than just theorizing?
