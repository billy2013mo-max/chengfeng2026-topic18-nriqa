# Adversarial Architecture & Breakthrough Debate (Round 1) — Breaking Out of the `~52.90` Online Plateau

## Context & User Mandate
The user tested all three of our recent zero-leakage deliverables on the online `test_150` leaderboard:
1. `pred_nriqa_dataset_test_r55_resident_v4_routed_gate_eu.json` -> **`52.97 / 65`** (`S_pos = 16.18 / 25`, `S_p0 = 16.79 / 20`, `Clean = 20.00 / 20`)
2. `pred_nriqa_dataset_test_r55_resident_v4_dirdiag_gate_eu.json` -> **`~52.90`** (within noise)
3. `pred_nriqa_dataset_test_r55_resident_v5_routed_gate_eu.json` -> **`~52.90`** (within noise)

The user's explicit feedback before going to sleep:
> "最近你推荐我的三个推理json，在线上test150集合上都在52.90上下浮动，噪声范围内，没什么太大的正向优化，这块你还是和claude fable 5.2一起讨论下下一步我们要有重大突破的话，应该怎么做"
> "我准备睡觉了，等你做好计划，和fable 5.2充分讨论完就直接开干就ok"

We have **8 × NVIDIA H200 (143.7 GB VRAM each, 100% idle right now)**, `/dev/shm` RAM cache warm, `154 GB` free disk on `/wanqing-develop`, and full overnight autonomous execution runway (~7–9 hours).

### Hard Iron Rules (`AGENTS.md` — Non-Negotiable)
1. **Rule 1**: ZERO reading/anchoring/overriding from any legacy test JSONs. 100% fresh inference from newly trained weights.
2. **Rule 2**: ZERO `test_150` or `654` test-set tuning/overfitting. Strict `N=1` single-image independence at inference; all calibration/decoding strictly fitted on labelled OOF (`data/qingyu_splits/qingyu_8fold.csv`, zero video-group leakage).
3. **Rule 3**: Production-deliverable single-GPU resident architecture (`<= 30B` total unique params, `<= 24 GB` peak VRAM on RTX 4090, `<= 15s/img` batch-1 latency; note `AGENTS.md` Rule 3 also highlights centering around a 6B visual foundation model `InternViT-6B` + ~4 vertical domain specialists rather than bloated homogeneous ViT stacks).

---

## Verified Quantitative Forensics (Inspect Directly in Repo!)
Please use your `Read` / `Grep` tools to inspect:
- `/wanqing-develop/luowenjing/18/results/r55/v6_debate/forensics_v6_out.txt` (just generated — full 14-cell & ceiling audit!)
- `/wanqing-develop/luowenjing/18/results/r55/final/v5_waves_final_audit.txt` (results of all 37 Wave A + Wave B jobs)
- `/wanqing-develop/luowenjing/18/results/next55/LEDGER.md` (lines 1–74)
- `/wanqing-develop/luowenjing/18/code/r55/final_engine.py`, `code/r55/train_p12_specialist.py`, `code/r55/p12_routed_head.py`, `code/r55/native_tiles.py`, `code/qingyu/backbones_qingyu.py`, `code/phoenix/train_pillar12_native_vlm.py`, `code/r55/phoenix_pillar_4090.py`.

### Key Numbers From `forensics_v6_out.txt`:
1. **Why `v4`, `v4_dirdiag`, `v5` all landed at `~52.90` online**:
   - Pairwise OOF cell disagreement across all 3 models is only **`2.06% – 2.54%`**.
   - On 2,000 random 150-image `orig` draws, `v5 - v4` has `mean = +0.27` vs **`SD = 0.78`** (`90% interval [-0.92, +1.63]`, `P(v5 < v4) = 37.9%`).
   - Wave B1 (`ntg_cradio_e4`, `ntg_dinov3_e6`, `ntg_convnext_e8`) and Wave B2 (`p12_condb1` 2-seed) ALL failed the `-0.00015` in-pool RPS gate (`+0.00008`, `+0.00006`, `-0.00001`, `-0.00010`).
   - All 8 towers in `resident_v5` (`cradio_v3, pe_v3, peg, sig_v3, v16, p3_qy, ntg_pe, p12_pe`) are `0.3B–1.8B` pure-vision ViTs/ConvNeXts (`4` of the `8` are `0.33B pe_core_large`!), with pairwise `E[grade]` correlation **`0.948 – 0.978`**. Leave-one-out (`LOO`) shows `sig_v3` (`dRPS = -0.00004` orig) and `pe_v3` (`dRPS = -0.00001` orig) are complete dead weight.

2. **CPU Post-Processing / Decoder / 2nd-Stage Stacking Headroom on `Q_v5` is `0.00` (DEAD END)**:
   - `14-cell ECE` on `ORIG_1902` is **`0.0061`**.
   - Nested 14-parameter coordinate-ascent decoder scores `51.462 / 51.746` (**worse** than `TrueGateConstrainedEU` `51.703 / 51.943`). Even an **in-sample cheating** 14-parameter threshold search on `ORIG_1902` only reaches `52.480` (`+0.538` cheating upper bound).
   - A 216-dim cross-defect + 8-tower multinomial stacker degrades RPS (`0.03843` blend vs `0.03813`).
   - **Implication**: Zero further gains exist in threshold/calibration/stacker micro-tweaking on existing `Q_v5`. Any real breakthrough (`+0.5 ~ +1.2+` OOF Doc2, `dRPS <= -0.0006 ~ -0.0015`) MUST come from **representation / model-family orthogonality / ordinal severity supervision**.

3. **Where the `13.06` Missing Points (`ORIG_1902`: `51.94 / 65`) Actually Sit**:
   - **Detection is near-saturated**: `DetAUC` (`>0` vs `0`) on `ORIG_1902` is `0.9507` (`模糊`), `0.9557` (`纹理`), `0.9613` (`过锐`), `0.9998` (`花屏`), `0.9617` (`过度生成`), `0.9948` (`虚化突兀`).
   - **Counterfactual 1 (Oracle Detection + Current Grading)**: `51.94 -> 55.17` (`+3.23` pts).
   - **Counterfactual 2 (Current Detection + Oracle Conditional Grading `P2 vs P1 vs P0`)**: **`51.94 -> 60.77` (`+8.83` pts on ORIG; `+9.43` pts on ALL)**!
   - **`90%–98%` of all `P1` and `P0` errors are Adjacent-Grade Confusions (`P2 <-> P1 <-> P0`)!**
   - Point loss by defect on `ORIG_1902`:
     1. **`过度生成`: `4.39` pts lost** (`P0` loses `2.15`, `P1` loses `1.23`, `P2` loses `1.01`; `P1_vs_P2 AUC = 0.7106`, `P0_vs_P1 AUC = 0.7483`!).
     2. **`过锐`: `2.96` pts lost** (`P0` loses `1.19`, `P1` loses `0.94`, `P2` loses `0.83`; `P1_vs_P2 AUC = 0.7914`, `P0_vs_P1 AUC = 0.8724`).
     3. **`模糊`: `2.30` pts lost** (`P2` loses `1.03`, `P1` loses `0.77`).
     4. **`纹理损失`: `2.27` pts lost** (`P0` loses `0.87`, `P1` loses `0.72`, `P2` loses `0.68`).

4. **Three Major Untapped Levers Uncovered in Our Repo Audit**:
   - **Lever 1 — Honest 8-Fold Fine-Tuned Multimodal VLMs (`Qwen3-VL-8B` & `Clef-Flash-9B`) Were Prematurely Retired!**:
     - In Wave 2A (`LEDGER.md` line 64), VLMs were retired based ONLY on *zero-shot* `Clef-Flash` (`mAP14=0.2695`) and *standalone* `p1` `过度生成` AP (`0.3770`).
     - In `forensics_v6_out.txt` §4C & Part 3, we just tested **Stage-1 `cv5` (pure real, ZERO teacher distillation, handicapped by 5-fold 80% data!)** restricted strictly to **`Strict_ALL (3,473)` and `Strict_ORIG (1,807)` rows with ZERO video/stem overlap**:
       - Control check (`p3_legacy_cv5` vs `p3_honest_qy8`): `cv5` is *worse/pessimistic* (`0.05218` vs `0.04999`), proving zero hidden optimism in `cv5` on `Strict3473`!
       - Yet blending `p1_qwen3vl8b (cv5)` into `resident_v5`: **`dRPS = -0.00040` (`CI90 [-0.00053, -0.00026]`) on ALL, `-0.00056` (`CI90 [-0.00076, -0.00037]`) on ORIG**!
       - Blending `p2_clefflash9b (cv5)` into `resident_v5`: **`dRPS = -0.00035` (`CI90 [-0.00050, -0.00022]`) on ALL, `-0.00055` (`CI90 [-0.00076, -0.00035]`) on ORIG**!
       - Blending `p1 + p2 (cv5)` together into `resident_v5`: **`dRPS = -0.00048` (`CI90 [-0.00062, -0.00033]`) on ALL, `-0.00070` (`CI90 [-0.00091, -0.00048]`) on ORIG** (`过锐 -0.00166`, `纹理损失 -0.00133`, `模糊 -0.00073`, `虚化突兀 -0.00061`!).
       - And `code/r55/phoenix_pillar_4090.py` already merges DoRA -> 4-bit NF4 LLM + BF16 ViT in **`~5.6 GB` VRAM** with verified quantized fidelity (`TV = 0.0087`, `LEDGER.md` line 26)! Wait: check whether `Clef-Flash-9B` (`Qwen3_5ForConditionalGeneration` with linear_attention + full_attention!) ALSO works cleanly with `phoenix_pillar_4090.py` 4-bit quantization or BF16, and check how fast 8-fold + full-fit training of `p1` / `p2` on `data/phoenix_splits/phoenix_qy8.csv` is on 8×H200!
   - **Lever 2 — `InternViT-6B-448px-V2.5` (`6.08B`, `models/internvit-6b`) Was Never Finished Training!**:
     - `InternViT-6B` is literally named in `AGENTS.md` Rule 3 (`6B 视觉大模型基座 InternViT-6B + 4 个垂直领域专家`), yet it is **completely absent** from `resident_v3/v4/v5`!
     - On Nov 10 (`logs/next55/iv6b_fold0..7.log`), `train_tier2_internvit6b_8fold.py` stopped at Epoch 2/5 because `models/internvit-6b/modeling_intern_vit.py` ran eager `_naive_attn` (`O(N^2)` manual matmul without `F.scaled_dot_product_attention`) + 45-layer gradient checkpointing + `chunk_sz=10`, taking `2,940s/epoch`! With PyTorch SDPA (`F.scaled_dot_product_attention`) on H200, it should run **5x–8x faster**!
     - Furthermore, `P12SemanticRoiTower` (`code/r55/train_p12_specialist.py` line 97) ALREADY supports `backbone_key="internvit_6b"` with `PerDefectRoutedOrdinalHead` + 5 semantic ROIs + `BoundaryMarginXBMLoss` + leak-free nested KD!
     - **VRAM check needed**: `InternViT-6B` in BF16 is `~11.3 GB` (`10.4 GB` safetensors on disk); or in 8-bit / 4-bit NF4 after LoRA merge, how much VRAM & fidelity does it have alongside our specialists inside `24 GB`?
   - **Lever 3 — Breaking the `过度生成` (`4.39` pts lost, `P1_vs_P2 AUC = 0.71`) & Ordinal Severity Bottleneck**:
     - Notice that EVEN `p1 + p2` gave `0.0000` on `过度生成` (`-0.00004` ALL / `+0.00019` ORIG)! Why does EVERY architecture (`Tier1`, `V16`, `Pillar3`, `P12`, `NTG`, `Pillar1`, `Pillar2`) plateau at `过度生成 P1_vs_P2 AUC ≈ 0.71–0.73`?
     - Look at two architectural blindspots:
       1. `NativeTileTower` (`ntg_pe`, which achieved the highest standalone `OG_AP = 0.4282` and `mAP14 = 0.6375`!) sees ALL `448x448` native tiles of the image, **but collapses all `T` tiles through ONE single defect-agnostic scalar gate (`view_agg.gate` `[B, T, 1]`) into the 1D Quadratic Head (`UnimodalQuadraticResidualHead`) with NO KD and Epoch-8 overconfidence**! If we combine **Full-Frame Native Tiles + Semantic ROIs** with **`PerDefectRoutedOrdinalHead` (`Multi-Query Spatial Extent + Peak-k Tile Readout`: `top-1`, `top-2`, `top-4`, `mean`, `tile_std` per defect query + `BoundaryMarginXBMLoss` + nested KD)**, each defect query can attend to its own spatial locus AND measure **spatial area/extent vs peak anomaly** (`P2` vs `P1` vs `P0`)!
       2. Look at `LEDGER.md` line 47: on `过度生成`, **sibling frames from the same video** had `-22.5%` RPS, and in `REASSESSMENT_AND_PLAN_2026-11-12.md` §2.6: *"for 过度生成 P2/P1 another frame's human label predicts the grade better than the whole ensemble ... Hypothesis: it is the strength of the enhancement applied to the video."* Why did `p1/p2/pe` miss enhancement strength? Can a tower or VLM explicitly inspect **multi-scale native texture/frequency statistics or generative enhancement signatures** across full-frame native tiles, OR is `过度生成 P1/P2` partially bounded by annotator ambiguity?

---

## Questions for Your Ruthless Critique & Proposal (Round 1)
1. **Critique Our Diagnosis & Numbers**: Did we miss any subtle leakage, statistical trap, or hidden flaw in `forensics_v6_out.txt`? Specifically, check `train_pillar12_native_vlm.py` (did `stage0_tapt` or `scrutiny_box_cache` leak anything in `cv5`, or is the `-0.00070` `ORIG` ΔRPS of `p1+p2` 100% real cross-family multimodal diversity?).
2. **Evaluate the 3 Levers & Propose Any Better/Missed Lever**:
   - **Lever 1**: Retraining honest 8-fold (`phoenix_qy8.csv`) + 100% full-fit Native Multimodal VLM(s) (`Qwen3-VL-8B` `p1_qy` and/or `Clef-Flash-9B` `p2_qy` — or an upgraded VLM recipe with 5 semantic ROIs + leak-free nested KD!). Check parameter cap (`<=30B`) and 4090 VRAM (`<=24 GB`) carefully!
   - **Lever 2**: Unlocking `InternViT-6B` (`6.08B`, SDPA-accelerated, LoRA `r=32`) — either with `P12SemanticRoiTower` (`5 semantic ROIs + PerDefectRoutedOrdinalHead + KD`) or `NativeTile-Routed`! Check exact parameter count + 4090 VRAM budget when combined with Lever 1 + streamlined vertical specialists (pruning dead-weight `pe_v3` & `sig_v3`, and/or checking whether `v16` `3.3B` or `p3_qy` `1.54B` stays or gets replaced!).
   - **Lever 3**: Attacking `过度生成` (`4.39` pts lost) & `P2 <-> P1 <-> P0` ordinal grading — e.g., **Multi-Query Native-Tile + Semantic-ROI Extent/Peak-k Specialist (`ntg_p12_routed`)**, or a VLM prompt/token-readout upgrade specifically targeting `过度生成` & `过锐` severity boundaries, or any deeper idea you see in the codebase!
3. **Design the Concrete Overnight 8×H200 Execution Plan & Pre-Registered Gates**:
   - Remember: we have **8 × H200 GPUs for ~7–8 hours tonight** while the user sleeps, and we need **both** (a) rigorous 2-fold or 8-fold OOF proof (`ΔRPS` & `ΔDoc2_Strict` with cluster bootstrap CI90) AND (b) **100%-data full-fit checkpoints + verified `final_engine.py` resident deployment (`<=30B`, `<=24 GB`, `<=15s`, `N=1 == N=150`) + fresh `test_150` JSON(s)** ready by morning!
   - Compute realistic wall-clock training times on 8×H200 for each proposed job so we don't launch something that takes 20 hours and fails to finish!
   - Give concrete, quantitative estimates of what can genuinely push us from **`~52.9` online (`~51.70 / 51.94` OOF)** toward **`53.6 – 54.5+` (`~52.3 – 53.0` honest OOF)** without test-set overfitting.
