# Debate: High-Compute Breakthroughs Beyond `resident_v6_core` (8×H200 Training Available)

## 1. Status Update (R1 & R2 Closed)
We just ran both R1 and R2 on CPU:
- **R1 (`p1_qy` 2-seed `s2026+s2027` on 6 folds, 2,682 imgs)**: solo RPS improved (`0.04735 -> 0.04597`), but inside `resident_v6_core` (frozen decoder) `dRPS_ALL = -0.00005 [-0.00011, +0.00001]`, `dRPS_ORIG = +0.00000 [-0.00008, +0.00009]`, `Doc2S -0.067 | -0.076`. **R1 CLOSED.**
- **R2 (18-param per-dim EU in full-nested 8-fold CV)**: `BASE_NESTED all=51.597 / orig=51.873` vs `PERDIM_NESTED all=51.558 (-0.039) / orig=51.986 (+0.113, Pos dropped 15.81 -> 15.75)`. **R2 CLOSED.**

## 2. User's Explicit Mandate & Available Heavy Compute
The user explicitly wants to **invest serious GPU compute (we have 8×H200 141GB GPUs)** to push the objective score higher rather than settling for `resident_v6_core`, **within the deployment rules**:
- Deployment: single 24 GB GPU (RTX 4090), `N=1 == N=150`, `<= 15s/img` at `N=1` (meaning weights must stay **resident** in `<= 24 GB` VRAM, no per-image disk reload), `<= 30B` total active params.
- Current `resident_v6_core`: `15.77 GB` resident, `17.74 GB` infer peak, `21.16 GiB` `nvidia-smi` peak, `1.98s/img`.
- **Untapped local giant models in `models/`** (never trained 8-fold OOF with `c233397` DoRA + 63-avg+1-summary + SWA):
  1. `JEV-27B-VL` (`Qwen3_5ForConditionalGeneration`, 27B!)
  2. `JEV-Gemma4-26B-A4B` (`Gemma4ForConditionalGeneration`, 26B MoE, 4B active!)
  3. `nemotron-omni` (`NemotronH_Nano_Omni_Reasoning_V3`)
  4. `qwen3vl-8b` (currently only trained with `r=24` DoRA, 3-view `784 + 2x384`, 4 epochs, frozen base LLM/vision except LoRA)

## 3. Question for Debate
If we are willing to spend **many H200 GPU-hours** on training/distillation/architecture search, what are the **highest-probability above-noise paths** to push `resident_v6_core` (`orig 51.873 / all 51.597`, `mAP14 0.665`, `RPS_orig 0.03760`) meaningfully higher?

Please evaluate and rank concretely:
1. **R3 (`p1_qy v2` on `qwen3vl-8b`)**: semantic ROIs + 1024px + full-vision-tower unfreeze or higher LoRA rank (`r=64`?) + 7-of-8 nested KD from a 27B/26B mega-teacher?
2. **Replacing `p1_qy` (8B) with `JEV-27B-VL` or `JEV-Gemma4-26B-A4B` (4-bit NF4)**:
   - Can `JEV-Gemma4-26B-A4B` or `JEV-27B-VL` in 4-bit NF4 fit in `<= 22 GiB` `nvidia-smi` alongside the 5 vision towers (`~10 GB` resident -> leaves `~11-12 GB` for VLM! Wait: 27B 4-bit is ~14-15 GB, so `JEV-27B-VL` + 5 vision towers would exceed 24 GB unless we prune 1-2 vision towers or distill `27B -> 8B/vision`!).
   - What about **8×H200 Mega-Teacher Ensemble -> 7-of-8 Nested Cross-Fitted Knowledge Distillation (KD)** into ALL resident 4090 students (`p1_qy`, `v16`, `cradio_v3`, `ntg_pe`, `p12_pe`), just like `p12_pe` KD gave `-0.00030` RPS? Notice **KD adds ZERO inference VRAM and ZERO inference latency**!
3. **Training `v16` (`InternViT-6B`) or `cradio_v3` with better data/resolution/KD** or **multi-task auxiliary heads** (e.g., training with fine-grained spatial/region/pairwise-rank signals on 8×H200)?
4. **Is there any realistic ceiling** on how much even 27B+26B+KD can gain given the label-noise ceiling (`过度生成` inter-frame human agreement 44-61%), and what is the **single best 2-fold pilot** to run FIRST on 8×H200 right now?
