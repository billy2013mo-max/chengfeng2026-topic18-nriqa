# Overnight Technical Debate (Round 2): Interim P0 Results (`Qwen3-VL-32B`) + CPU Results + GPU 0–3 Allocation

Here are the exact empirical results from tonight so far:

1. **E1 Decision-Layer RL Falsification (`eval_e1_rl_vs_eu.py`, 8-fold nested CV on `resident_v6_core`)**:
   - `RoutedGateEU (baseline)`: `Doc2S_all=51.585 | Doc2S_orig=51.580 (Pos=15.66, P0=16.16, Cln=19.76)`
   - `E1_Surrogate_Doc2`: `Doc2S_orig=34.914`
   - `E1_GRPO_REINFORCE`: `Doc2S_orig=10.474 (Clean2=0.00)`
   - Confirmed: decision-layer RL is strictly worse and collapses on corpus-level F1 / Clean2.

2. **Sibling-Pair Noise Floor (`eval_sibling_noise_floor.py`, 1,243 same-video orig pairs)**:
   - `all_siblings (n=1243)`: `pair_RPS=0.03991`, `est_Bayes_RPS=0.01996`, `clean_agree=0.983`
   - `cos>=0.90 (n=20)`: `est_Bayes_RPS=0.01250`; `cos>=0.95 (n=12)`: `0.01389`.
   - Note: `clean_agree=1.000` suggests annotators may label same-video frames consistently, so `0.013-0.020` is a lower bound on Bayes RPS.

3. **Unconstrained Existing-OOF Ceiling (`eval_unconstrained_ceiling.py`, 8-fold nested CV)**:
   - `resident_v6_core (6m)`: `RPS_orig=0.04207, Doc2S_orig=51.580`
   - `resident_v6 (9m)`: `RPS_orig=0.04200, Doc2S_orig=51.662`
   - `v6 + p12_g14_kd (10m)`: `RPS_orig=0.04197, Doc2S_orig=51.584`
   - `v6 + p12_g14_kd + p12_pe_kd (11m)`: `RPS_orig=0.04197, Doc2S_orig=51.664`

4. **P0 Interim Gate Results on Folds 0 & 1 (`N_all=872, N_orig=456`, `Qwen3-VL-32B` finished; `InternVL3_5-38B` still running on GPUs 4–7, ~4h left)**:
   - Solo OOF on Folds 0 & 1:
     - `p1_qy_8b (s2026)`: `RPS_all=0.04769 | RPS_orig=0.04134 | mAP14=0.6327`
     - `ctrl_p1_8b_s2027`: `RPS_all=0.04694 | RPS_orig=0.04129 | mAP14=0.6516`
     - `v16_6b`: `RPS_all=0.04743 | RPS_orig=0.04060 | mAP14=0.6394`
     - `qwen32b_s2026`: `RPS_all=0.04809 | RPS_orig=0.04044 | mAP14=0.6370`
     - `qwen32b_s2027`: `RPS_all=0.04730 | RPS_orig=0.04081 | mAP14=0.6437`
     - `qwen32b_2seed`: `RPS_all=0.04627 | RPS_orig=0.03931 | mAP14=0.6517`
   - Ensemble on Folds 0 & 1 (`resident_v6_core` decoder fitted on Folds 2..7: `RPS_all=0.04140, RPS_orig=0.03673, Doc2S=52.208 | 52.357`):
     - `[ADD ctrl_p1_8b_s2027 a=0.15]`: `dRPS_all=-0.00003 [-0.00023,+0.00016] | dRPS_orig=-0.00004 [-0.00030,+0.00022]` (FAIL)
     - `[ADD qwen32b_s2026    a=0.15]`: `dRPS_all=-0.00007 [-0.00030,+0.00015] | dRPS_orig=-0.00029 [-0.00057,-0.00002]` (**PASS -0.00020**)
     - `[ADD qwen32b_2seed    a=0.15]`: `dRPS_all=-0.00010 [-0.00029,+0.00010] | dRPS_orig=-0.00025 [-0.00050,-0.00001]` (**PASS -0.00020**, FAIL `-0.00050` KD-ROI)
     - `[ADD qwen32b_2seed    a=0.25]`: `dRPS_all=+0.00000 [-0.00032,+0.00034] | dRPS_orig=-0.00029 [-0.00070,+0.00011]`
     - `[REP qwen32b_2seed    0.18 ]`: `dRPS_all=-0.00012 [-0.00034,+0.00010] | dRPS_orig=-0.00017 [-0.00047,+0.00012]`

## Debate Questions for GPUs 0–3 Right Now (while GPUs 4–7 finish InternVL3_5-38B in ~4h)
1. **Interpretation of `Qwen3-VL-32B` P0 result**:
   - Solo `qwen32b_2seed` beats `p1_qy_8b` by `-0.00203` `RPS_orig` (`0.03931` vs `0.04134`) and passes `-0.00020` (`-0.00025`, CI90 `[-0.00050, -0.00001]`), 6x larger than `+seed` control (`-0.00004`).
   - Yet it misses your `-0.00050` KD-ROI threshold, and `dRPS_all` is only `-0.00010`. Why did `qwen32b` improve `orig` (`-0.00025`) much more than `all` (`-0.00010`)?
   - Does `qwen32b_2seed` justify KD, or should we wait for `InternVL3_5-38B` + `dual_32b_38b`?
2. **Best use of GPUs 0–3 for the next 4–5 hours**:
   - **Choice 1**: Run 4-fold inner cross-fitting of `Qwen3-VL-32B` on Fold 0 (all 4 inner folds on GPUs 0–3 in parallel, ~4h) so we can immediately test whether `KL(T=1) + CDF-L2` KD into `p1_qy` 8B recovers any of that `-0.00203` solo gap!
   - **Choice 2**: Run Folds 2, 3, 4, 5 of `Qwen3-VL-32B` (`s2026`, 4 GPUs in parallel, ~5.3h) so we can measure the true unconstrained `Doc2S` & `RPS` ceiling across 6–8 folds (not just F0–F1).
   - **Choice 3**: Run E2 (CoT-RL headroom test on 300 Fold-0 val images with `Qwen3-VL-8B` / `32B` on 1 GPU) + Choice 1 or 2 on the other GPUs.
Argue which choice maximizes scientific certainty tonight for the user's `/goal`!
