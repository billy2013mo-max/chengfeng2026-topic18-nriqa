# Debate: Next Above-Noise Breakthroughs on Top of Converged `resident_v6_core`

## 1. Where We Stand (`resident_v6_core`, 6 members)
We successfully pruned 9-member `resident_v6` down to 6-member `resident_v6_core` (`["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy"]`):
- **Offline Nested OOF (`Doc2_Strict`, 3,573 real images)**:
  - `all = 51.597`, `orig = 51.873` (`Pos_orig = 15.81 / 25`, `P0_orig = 16.30 / 20`, `Cln_orig = 19.77 / 20`)
  - `routed_eu`: `RPS_all = 0.04231`, `mAP14 = 0.6650`; pooled `routed_gate_eu` `RPS_all = 0.04192`, `RPS_orig = 0.03760`, `mAP14_orig = 0.6661`
- **Hardware Headroom (single GPU, `--half`)**:
  - Active params: **9.656B** (`<= 30B` limit -> **20.3B headroom**)
  - Resident VRAM: **15.77 GB** | Infer peak: **17.74 GB** | Load peak: **18.83 GB** | `nvidia-smi` peak: **21.16 GiB** (`<= 24 GiB`)
  - Latency: **1.98s/img** H200 (`2.21s/img` on `test_150`) -> well under **15s/img** 4090 budget!

## 2. Exact Error & Headroom Breakdown (`orig` 1,902 real images)
Per-member standalone (`RPS_all | RPS_orig | mAP14_orig`):
- `p1_qy` (Qwen3-VL-8B DoRA `r=24`, 4-bit NF4, 3-view `784 + 2x384`, 1 seed `s2026`): `0.04761 | 0.04223 | 0.6286` (#1 single model by far on `orig`!)
- `p12_pe` (PE-Core-L + semantic ROI + KD): `0.04756 | 0.04363 | 0.6236`
- `v16` (InternViT-6B + multi-backbone microscope): `0.04905 | 0.04397 | 0.6206`
- `cradio_v3` (C-RADIO-v4-H): `0.04891 | 0.04475 | 0.6201`
- `ntg_pe` (PE-Core-L native tile+global): `0.04978 | 0.04542 | 0.6296`
- `peg` (PE-Core-L + simulated glitch): `0.05148 | 0.04688 | 0.6162`

Per-label F1 (`routed_gate_eu` on `orig`):
- **Already near ceiling**: `花屏_P0` **0.977**, `模糊_P0` **0.903**, `虚化突兀_P0` **0.829**, `纹理损失_P0` **0.827**
- **Massive missing points (`9.19` pts missing in `S_pos`, `3.70` pts missing in `S_p0`)**:
  1. `过度生成` (`P1` **0.370** [TP=40, FP=63, FN=73], `P2` **0.437** [TP=71, FP=105, FN=78], `P0` **0.541** [TP=49, FP=42, FN=41]) — weakest dimension by far, hurts BOTH `S_pos` (3 of 14 labels!) AND `S_p0` (1 of 6 P0 labels!)
  2. `模糊_P2` **0.431** (FP=220, FN=115), `模糊_P1` **0.547**
  3. `过锐_P1` **0.484**, `过锐_P2` **0.537** (FP=198), `过锐_P0` **0.755**
  4. `纹理损失_P1` **0.597**, `纹理损失_P2` **0.621**

## 3. Question for Debate
Given `test_150` has `±0.2–0.4` noise, we only want to pursue directions that can realistically deliver **above-noise gains** (e.g. `dRPS_ORIG <= -0.00035`, `CI90 < 0`, `>= +0.35..+0.80` on 3,573-image nested OOF `Doc2S`, and strong transfer to the final 654-image blind test), while respecting:
- Single 24 GB GPU (`<= 24 GiB` `nvidia-smi` peak, `<= 15s/img` on 4090, `<= 30B` params, `N=1 == N=150` batch invariance)
- Zero test-set tuning, zero legacy JSONs, 8-fold video-grouped OOF gating.

Please analyze and propose **3–5 concrete, ranked, high-ROI directions** on top of `resident_v6_core` (and explicitly flag what NOT to waste time on):
1. **Model/Representation Upgrades** (e.g., upgrading `p1_qy` — 2nd VLM / Qwen3-VL-8B with native-res / semantic ROI / multi-task or ordinal/defect-focused loss, or replacing a weaker PE variant with a genuinely stronger orthogonal backbone, or `过度生成` semantic specialist).
2. **Decoder / Calibration Upgrades** (note: `TrueGateConstrainedEU` currently optimizes a single global `(lam_cl, p0_scale)` across all 6 dimensions and uses 1-D `DirDiag` + L2-shrunk routing; is there genuine above-noise headroom in ordinal boundary calibration or per-dimension utility/coupling, or is decoder largely saturated?).
3. **Training Objective / Data Signal Upgrades** (specifically targeting `过度生成` + `P2/P1` boundary precision where `FP=195..220`!).

Be brutally honest about expected ΔRPS / ΔDoc2S, VRAM/latency cost, and concrete go/no-go gates for each!
