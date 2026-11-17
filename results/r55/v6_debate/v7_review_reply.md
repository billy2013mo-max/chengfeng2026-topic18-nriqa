I'd hold submission #2. I found a real bug that changes how the `resident_v7` gains should be explained.

## 0. Bug: the `bkseg1_alpha` blend writes into 花屏, not 虚化突兀

The canonical order is `DIMS = ["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]` (`code/qingyu/config_and_paths.py:56`), so index 3 is **花屏** and 虚化突兀 is index **5**. The blend at `code/r55/routed_eu_decoder.py:182-183` and `:572-573` uses `Q[:, 3, :]`. It looks copied from the `peg` 花屏 rescue just above it, where index 3 is correct.

The router weights in `results/r55/final/resident_v7_decoder_routed_eu.json` tell the same story. Here is what `pe_bkseg1` gets in each dimension:

| 模糊 | 纹损 | 过锐 | 花屏 | 过度生成 | 虚化突兀 |
|---|---|---|---|---|---|
| 0.122 | 0.081 | 0.105 | 0.026 | 0.067 | **0.000** |

The 0.1032 you quoted is `p1_qy`'s weight on index 3, not `pe_bkseg1`'s.

What this means:
- In the shipped decoder, `pe_bkseg1` contributes **nothing** to 虚化突兀. The router gave it zero weight there, and the alpha blend went into 花屏 at about 42%.
- So the +0.08 OOF gain and the 53.92 come from `ntg_convnext` plus `pe_bkseg1` acting as an extra `pe_core_large` seed in 模糊/纹损/**过锐**/**过度生成**. Those are the two dims where LEDGER row 25 says bkseg1's labels are contaminated.
- The "OOD cutout 0.218 → 0.527" check needs re-checking. If it read index 3, it measured 花屏 probability on bokeh cutouts. Your 99.2% recall / 0.8% FPR numbers are for the tower alone, so they don't show that the ensemble uses it.
- 花屏 had no changed cells on `test_150`, so the bug apparently did no visible harm there. That doesn't make it correct.

**Fix and re-evaluate on nested OOF only.** `test_150` can't be used to pick the variant (AGENTS.md rule 2). Compare three variants:
- **(a)** `bkseg1_alpha = 0`, i.e. router only (what effectively ships now, minus the 花屏 contamination).
- **(b)** the blend on index 5, with alpha chosen inside the nested folds rather than fixed at 0.40.
- **(c)** `pe_bkseg1` masked to dim 5 only: zero its W columns for dims 0–4, then refit W.

Pick by nested Doc2_Strict / RPS with CI90 on the standard and shift folds. Per the SynthLab rule, compare against the mean of 3 control seeds, because (a) is mostly a seed-averaging effect.

## 1. Fitting a 24 GB 4090

- **Tile chunk 64 → 16** in `native_tiles.py:49`. Make it a parameter, since training uses the same path. This removes most of the +3.7 GB spike and should leave outputs unchanged apart from bf16 noise; check max-abs diff on the 150 images.
- **The 55.55 GiB NVML figure is mostly the caching allocator** on a large card, not real need. To test honestly on the H200, call `torch.cuda.set_per_process_memory_fraction(21/140.4)` (or the 24 GiB equivalent) before loading. Run the full `test_150` pass plus synthetic 1080p and 4K frames (45 tiles is the worst case). Log NVML and allocated per member; the hooks at `final_engine.py:432/705` already exist. Passing under that cap is the compliance proof.
- **Budget:** 20.7 GB resident + about 1 GB peak with chunk 16 + about 0.6–1 GB CUDA context ≈ 22.5 GB. That fits 24 GB but **not** your 21 GiB target. Reaching 21 GiB means dropping about 1.5–2 GB of members.
- **Choose what to drop by nested-OOF leave-one-out ablation, not `test_150`.** The weakest by router weight are `pe_v3` (≈0.02–0.05), `sig_v3` (≈0.01–0.05) and `cradio_v3` (≈0.02–0.09). Dropping `pe_v3` is also cheap in principle, because `pe_bkseg1` is the same architecture and takes over its seed-averaging role. That gives the `resident_v6_core`-style pattern you suggested. `peg` must stay: it is the only 花屏 rescue.

## 2. Your three remaining questions

- **All 6 dims or 虚化 only:** decide by variants (a)/(b)/(c) above on OOF. A priori I lean toward (c) plus a retrain (next bullet), because the dims where `pe_bkseg1` carries weight are exactly the contaminated ones. But what currently helps is the seed-averaging, so measure rather than assume.
- **Highest-ROI zero-overfit fix for 过锐/过度生成:** retrain bkseg1 with the 过锐 and 过度生成 losses ignored (ignore-index) on synthetic frames. That removes the documented cause of the +0.0077 / +0.0029 RPS harm and doesn't need new data. Beyond that, past memory and LEDGER show every other 过锐/过度生成 recipe (rank aux, P3–P6, decoder v2) came out null or harmful, so I don't have another honest candidate. Native resolution (`ntg_convnext`) is the one lever that has moved 过锐 P0.
- **Fast natural shallow-DoF false-positive check (no test data):**
  1. On ORIG OOF, take frames with gold 虚化突兀 = 0 and compare the rate of P(虚化) > 0.5 for `pe_bkseg1` vs `pe_v3`, per fold, with a bootstrap CI. Do the same for the full decoder before and after the index fix, and count clean false alarms (Clean2).
  2. Repeat on the s16/s17 blind sets.
  3. Optionally build a small set of external natural-bokeh frames (allowed per the R55 decisions) as negatives, report the FP rate only, and don't tune on it.

Order before submission #2: index fix → (a)/(b)/(c) nested OOF → chunk 16 plus the capped-memory dress rehearsal → then decide. I haven't changed any code; say the word and I'll apply the index fix and the chunk parameter.
