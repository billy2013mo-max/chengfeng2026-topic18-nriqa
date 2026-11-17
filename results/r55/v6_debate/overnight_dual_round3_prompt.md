# Overnight Technical Debate (Round 3): Dual-Teacher (`32B + 38B`) P0 Results & Next 4.5h GPU Plan

`InternVL3_5-38B-HF` (Folds 0 & 1, 2 seeds) just finished, and we evaluated **ALL** teachers on Folds 0 & 1 (`N_all=872, N_orig=456`) against `resident_v6_core` (decoder fitted on Folds 2–7):

## Exact Results (`eval_p0_teacher_gate.py`)
### Solo OOF (Folds 0 & 1)
- `p1_qy_8b (s2026)`: `RPS_all=0.04769 | RPS_orig=0.04134 | mAP14=0.6327`
- `v16_6b`: `RPS_all=0.04743 | RPS_orig=0.04060 | mAP14=0.6394`
- `qwen32b_2seed`: `RPS_all=0.04627 | RPS_orig=0.03931 | mAP14=0.6517`
- `intern38b_2seed`: `RPS_all=0.04814 | RPS_orig=0.04062 | mAP14=0.6434`
- **`dual_32b_38b` (50/50 blend)**: **`RPS_all=0.04542 | RPS_orig=0.03831 | mAP14=0.6591`** (`-0.00303` RPS_orig vs `p1_qy_8b`, `+2.64%` mAP14!)

### Ensemble on `resident_v6_core` (`RPS_all=0.04140, RPS_orig=0.03673, Doc2S=52.208 | 52.357`)
- `[ADD ctrl_p1_8b_s2027 a=0.15]`: `dRPS_orig=-0.00004 [-0.00030,+0.00022]` (`P020=False`)
- `[ADD qwen32b_2seed    a=0.15]`: `dRPS_orig=-0.00025 [-0.00050,-0.00001]` (`P020=True, PKD=False`)
- `[ADD intern38b_2seed  a=0.15]`: `dRPS_orig=-0.00033 [-0.00059,-0.00009]` (`P020=True, PKD=False`)
- `[ADD dual_32b_38b     a=0.15]`: `dRPS_orig=-0.00033 [-0.00054,-0.00012]` (`P020=True, PKD=False`)
- **`[ADD dual_32b_38b     a=0.25]`**: `dRPS_orig=-0.00044 [-0.00079,-0.00010]` (`P020=True`, right near `-0.00050`! `dRPS_all=-0.00009 [-0.00037,+0.00020]`)
- `[REP dual_32b_38b     0.18 ]`: `dRPS_orig=-0.00025 [-0.00054,+0.00004]`

## Current GPU Schedule (~4.5h window on GPUs 0–3, 5, 6 while Folds 6–7 of 32B finish on GPUs 4 & 7)
- GPUs 4 & 7: `Qwen3-VL-32B` Folds 6 & 7 (`s2026`, ~5.2h left).
- GPUs 0–3: `Qwen3-VL-32B` Folds 2–5 finish in ~40 min.
- GPUs 5 & 6: `p1_qy` 8B control `s2028` F0 & F1 finish in ~25 min.

## Questions for Round 3 Debate
1. **Unconstrained Ceiling & Teacher Diversity**: Notice `dual_32b_38b` (`0.03831`) beats BOTH `qwen32b_2seed` (`0.03931`) and `intern38b_2seed` (`0.04062`) by `-0.00100` solo, AND `intern38b` + `qwen32b` reaches `dRPS_orig = -0.00044` (`CI90 [-0.00079, -0.00010]`) on top of `resident_v6_core`! Does this justify:
   - **Plan A**: Spend the 4.5h window on GPUs 0–3 to run **4-fold inner cross-fitting of `Qwen3-VL-32B` on Fold 0** + train a **KD `p1_qy` 8B student (`KL(T=1) + CDF-L2`)** on Fold 0, so we have a *direct empirical KD measurement* (rather than extrapolating)?
   - **Plan B**: Spend the 4.5h window on GPUs 0–3 to train **`InternVL3_5-38B-HF` Folds 2–5** (or Folds 2, 3, 4, 5 — takes ~9h, won't finish by morning) vs **`Qwen3-VL-32B` + `InternVL3_5-38B` E2 CoT-RL headroom test** on 1 GPU + **Plan A** on 4 GPUs?
   Wait: we have **6 GPUs** freeing up soon (0–3 + 5–6)! So we can run **Plan A (4-fold inner KD test on Fold 0)** on GPUs 0–3 AND **E2 CoT-RL headroom test** on GPU 5!
2. Challenge Plan A's exact KD design: what `lambda_kd`, `beta_cdf`, and `T` should we use in `vlm_c233397_qy8.py` so we don't sabotage the KD test with bad hyperparameters?
