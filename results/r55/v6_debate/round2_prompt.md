# Round 2 Convergence — Empirical Screen Results Are In (All Probes Completed on 8×H200)

Thank you — Round 1 was razor-sharp. Before replying, we executed **every single empirical probe and screen** you asked for on our 8×H200 GPUs (`/wanqing-develop/luowenjing/18/results/r55/v6_debate/{probe_vlm.log,probe_iv6b_v2.log,r1_empirical_checks.txt}`). Here are the hard numbers:

## 1. Empirical Results on Your Round-1 Checks

1. **`InternViT-6B` Epoch-2 8-Fold OOF Screen (COMPLETED across all 8 folds, `iv6b_ep2_8fold_oof.npz`)**:
   - Standalone `IV6B Ep2`: `RPS_ALL = 0.05423`, `RPS_ORIG = 0.04991`, `mAP14 = 0.5724`, `OG_AP = 0.3481`.
   - Blend `v5 + 0.12 * IV6B_Ep2`: **`dRPS_ALL = +0.00003 [-0.00007, +0.00014]`**, `dRPS_ORIG = -0.00012 [-0.00025, +0.00001]`.
   - Inside `RoutedGateEUDecoder`: `v5_pruned + p1_cv5` (`51.513` ORIG) -> `+ iv6b_ep2` (`51.304` ORIG, **-0.209**).
   - Also tested `IV6B` NF4 4-bit quantization (`probe_iv6b_v2.log`): `z_norm_cos = 0.5916`, `TV = 0.1264` (catastrophic degradation when 4-bit quantizing a vision encoder, unlike LLM layers!).
   - **Agreement**: **Lever 2 (`InternViT-6B`) is CLOSED / DROPPED** by your pre-registered screen gate (`+0.00003 > -0.00003`). Lever 3 (`ntg_p12_routed`) is also **CLOSED**.

2. **Phoenix 5-Fold Hygiene & Sibling-Label Noise Ceiling (`r1_empirical_checks.txt`)**:
   - `phoenix_5fold.csv` has `11 / 3,049` multi-fold `canon_group`s (`100 / 3,573` rows); `3,473` strict zero-sibling rows (`1,807` `orig`). And `data/phoenix_splits/phoenix_qy8.csv` (`qingyu_8fold.csv` exact fold assignment!) already exists on disk.
   - On multi-frame videos (`N = 844`), predicting a random sibling frame's gold label scores **`Doc2_Strict = 47.26`** (`Pos = 13.83, P0 = 14.07, Clean = 19.36`) vs `resident_v5 = 52.14` (`+4.88` pts above single-annotator agreement!). Your diagnosis that most of the `+8.83` "oracle grading" gap is annotator boundary noise is **100% confirmed**.

3. **Exact Commit & Recipe Forensics (`c233397` vs `ef417c4` vs `0a8775b` `HEAD`)**:
   - `c233397` (`Nov 11 06:14 UTC`): trained `pillar1_cv5` (`08:22 UTC`, `RPS=0.05003`) & `pillar2_cv5` (`09:59 UTC`, `RPS=0.05226`). Used `r=24`, 32-avg-token readout, `sigma_anchor=0.25, sigma_mid=0.46`, NO `oversample_rare_real_p1_p0`, 1x val (no HFlip).
   - `ef417c4` (`Nov 11 10:22 UTC`): upgraded readout to **View-Separated Peak-Anomaly (`20 struct + 42 unaveraged peak tokens`) + 2x HFlip Val TTA**, STILL **NO `oversample_rare_real_p1_p0`** and **NO `grade_w=[1.0,1.25,1.75,1.65]` distortion**! Trained `pillar1_stage2-rl2r` (`12:11 UTC`, solo `RPS = 0.04930`, blend `dRPS_ORIG = -0.00056`) and `pillar1_fulldata_swa.pt`.
   - `0a8775b` (`Nov 11 12:54 UTC`, `HEAD`): added `oversample_rare_real_p1_p0` + `grade_w=[1.0, 1.25, 1.75, 1.65]` + `sigma_mid=0.22` — **NEVER OOF-tested on VLM**! Your catch saved us from running `0a8775b` blind!

4. **VLM 4-Bit Fidelity & Crucial Resident VRAM Discovery (`probe_vlm.log` + `resident_v5_bench.json`)**:
   - `pillar1` (`Qwen3-VL-8B`, `8.23B`): `4090_q4_resident = 5.65 GB`, `fwd_peak = 6.16 GB`, `2x_TTA = 0.201s` (views `0.28s`), `mean_TV = 0.0089`, `argmax_agree = 100.0%`.
   - `pillar2` (`Clef-Flash-9B`, `8.46B`): `4090_q4_resident = 6.15 GB`, `fwd_peak = 6.61 GB`, `2x_TTA = 0.241s`, `mean_TV = 0.0088`, `argmax_agree = 97.5%`.
   - **CRITICAL PRUNING & VRAM FINDING**:
     - When we tested `RoutedGateEUDecoder` on `Strict_ORIG (1,807)`:
       - `resident_v5 (8m)` = `51.960` (`RPS_O = 0.04008, mAP = 0.6637`)
       - `v5 - pe_v3 - sig_v3 (6m)` = **`51.479` (`-0.481` Doc2_Strict!)** because dropping `pe_v3` caused `peg` (synthetic glitch tower) to jump 3x in Tier-1 prior weight, and dropping `sig_v3` shifted `TrueGateConstrainedEU` across its discrete `(lam_cl, p0_scale)` feasibility boundary (`51.713`).
       - **Keeping ALL 8 members (`v5 + p1_cv5`, 9m)**: **`Nested Doc2S ALL = 51.753 (+0.057), ORIG = 52.058 (+0.098)`**, `RPS_O = 0.03970 (-0.00038)`, `mAP14 = 0.6653 (+0.0016)`, and `RoutedGateEU` assigns `p1_cv5` **#1 weight on `模糊` (`0.194`) and #1 weight on `过锐` (`0.182`)**!
     - **Why ALL 9 members (`v5 + p1`) DO fit inside `< 21.5 GB` peak VRAM**:
       In `resident_v5_bench.json`, `p3_qy` was `3.96 GB` because `final_engine.py:343` forgot to cast `p3.vision_tower` (`convnextv2_base` + `cradio`) to `bfloat16` when `--half` is set (every other member was cast to `bfloat16`)! Casting `p3.vision_tower` to `bfloat16` (already runs under `bf16` autocast) frees **~1.85 GB** -> `v5 (all 8m)` drops from `15.54 GB` to **`~13.7 GB` resident**, so **`v5 (all 8m) + p1 (5.65 GB)` = `~19.35 GB` resident, `~21.3 GB` peak (`<= 24.0 GB`!), `15.37B` params (`<= 30.0B`!)** — ZERO pruning damage needed!
     - Also noticed in `v5 + p1_cv5 (9m)`: `TrueGateConstrainedEU`'s `lams_cl = (1.75, 2.00, 2.25, 2.50, 2.75)` hit the `2.75` upper grid boundary on several folds because `fit_nested_routed_pool` fit `T = 1.30` on `p1_cv5`, slightly lowering `prod_d Q[:, d, 0]` on Clean frames (`0.896 -> 0.804`). Extending `lams_cl` slightly (e.g. `(1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)` strictly inside nested CV!) or capping VLM temperature prevents artificial boundary clipping.

---

## 2. Three Concrete Questions to Lock Final Execution Plan (Round 2 = Final Lock)

We have ~5.5 hours and **8 idle H200s** right now. Please give crisp, decisive answers on these 3 design points so we execute immediately:

1. **Exact `p1_qy` (`Qwen3-VL-8B` on `phoenix_qy8.csv`, 8 folds on GPUs 0–7) Recipe**:
   - Between **`c233397`** (`r=24`, 32-avg-token readout, `sigma_anchor=0.25, sigma_mid=0.46`, no oversampling, **+ 2x HFlip Val TTA added**) vs **`ef417c4`** (`r=32`, View-Separated 20-struct + 42-peak-anomaly readout, same `c233397` loss without oversampling/grade_w, + 2x HFlip Val TTA; zero `init_batch_shift` or fold-local `tr_idx`-only `init_batch_shift`, `teacher=None`), which exact recipe should Wave 1 (`00:35–02:30`) run across the 8 folds?
   - Also: for the 100%-data full-fit (`p1_qy_fullfit`), should we run **exact schedule-matched 4-epoch single cosine run (SWA of epochs 3 & 4, identical to each CV fold!)** instead of `HEAD`'s 3+2+2 branch recipe so full-fit probability calibration matches OOF 1:1?

2. **Best Use of Wave 2 (`02:30–04:30`, 7 Free GPUs While GPU 7 Runs `p1_qy` Full-Fit)**:
   Instead of leaving GPUs 0–6 idle while GPU 7 runs `p1_qy` full-fit, what is the highest-EV second shot on `phoenix_qy8.csv` (we can run 8 folds across GPUs 0–6 + 1 sequential fold, or after `p1_qy` full-fit):
   - **Option 2A (`p1_sem4_qy`)**: Same `Qwen3-VL-8B` recipe, but replacing the 2 generic Wavelet+CoC crops (`OG_AP = 0.377`) with **Semantic+Wavelet 1:1 crops** (`[Global_Native_784, roi_subtitle_ocr_384, roi_face_anat_384, WaveletCoC_sharp_halo_384]`) so `Qwen3-VL-8B`'s native Chinese OCR & anatomy priors actually see bottom subtitles and faces/hands (`semantic_roi_meta.py` already boosted `pe_core_large` `OG_AP` from `0.382 -> 0.422`), gated head-to-head against Wave-1 `p1_qy` on honest 8-fold OOF!
   - **Option 2B (`p2_qy`, `Clef-Flash-9B` on `phoenix_qy8.csv`)**: Honest 8-fold `Clef-Flash-9B` (`mean_TV = 0.0088`, `6.15 GB`) as backup / head-to-head challenger (and check whether `p1_qy` or `p2_qy` wins in-pool on `v5`, or if replacing `v16` `6.73 GB` with `p2_qy` `6.15 GB` ever wins — likely not since `v16` LOO is `+0.00060`, so `p1_qy` vs `p2_qy` would be 1-slot winner-take-all).
   - **Option 2C**: Seed-2027 2-seed adapter/prediction soup or SWA check on `p1_qy`.
   Which ONE should Wave 2 run?

3. **Decoder & Gate Lock for `resident_v6` (`v5_8m + winning_VLM`)**:
   - Confirm keeping **all 8 `resident_v5` members** (with `p3_qy` vision tower cast to `bfloat16`, verified `TV < 0.002` before locking) + **1 winning VLM (`p1_qy` or `p1_sem4_qy` / `p2_qy`)** = 9 members (`~19.35 GB` resident, `~21.3 GB` peak, `15.37B` params).
   - Should we keep `RoutedGateEUDecoder` **100% unchanged** (`lams_cl = (1.75..2.75)`) to avoid any post-hoc decoder degree of freedom, OR pre-register `lams_cl = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)` strictly inside nested CV before Wave 1 finishes?

Keep it tight and concrete — as soon as you reply, we lock the spec and fire all 8 H200s!
