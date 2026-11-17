# Alignment Review with `claude` (`fable 5.2`): `resident_v7`, `ntg_convnext+rank` Screen, & Pre-Submission Checks

## 1. What We Built & Verified (`resident_v7`)
We implemented `resident_v7 = resident_v6 + pe_bkseg1 + ntg_convnext` (full 8-fold OOF + full-fit):
1. **`pe_bkseg1` (`r55_synth-bkseg1_s2026`, `pe_core_large`)**:
   - On held-out `bkseg1` synthetic hard-edge cutout bokeh (`seed_fold == 0`, 120 pos + 120 matched full-frame-blur neg):
     - Baseline `pe_core_large`: `Pos Recall = 8.3%`, `Neg FPR = 10.8%` (`Pos mean = 0.1298`).
     - `pe_bkseg1` (`th=0.50`): **`Pos Recall = 99.2%`**, **`Neg FPR = 0.8%`** (`Pos mean = 0.9904`).
   - Learned OOF router `W` only gave `pe_bkseg1` `0.1032` weight on `虚化突兀` (because OOF is already saturated at `0.977`). With `bkseg1_alpha = 0.40` (effective weight `~0.46` on `虚化突兀`, `dim=3`), `ORIG` `虚化` OOF F1 stays `0.9720` (`Clean2 = 19.805`) while OOD cutout bokeh crosses `0.50` (`0.218 -> 0.527`).
2. **`ntg_convnext` (`r55_ntg_convnext_e8_s2026`, `convnextv2_base` + `native_tiles` `tiles+global`)**:
   - `resident_v7` 8-fold nested OOF (`ORIG 1902`):
     - `Doc2_Strict`: `52.034 -> 52.114 (+0.080)` | `S_pos`: `15.848 -> 15.936` | `Clean2`: `19.805`
     - `Mean P0/P1/P2 F1`: `0.8118/0.4970/0.5040 -> 0.8134/0.5040/0.5070`
     - `过锐 P0`: `0.563 -> 0.586 (+0.023)` | `过度生成 P1/P2`: `0.458/0.516 -> 0.473/0.535`
3. **Fresh `test_150` Inference (we did NOT touch the 644 final eval set)**:
   - User reported official `test_150` score for `resident_v7`: **`53.92 / 65`** (`S_pos=16.81/25, S_p0=17.11/20, Clean=20.00/20`, **`+0.65`** over `53.27`!).
   - On `test_150`, `resident_v7` vs `resident_v6` changed `15 / 900 (1.7%)` cells across `12 / 150` images (`虚化`: 1 swap `P0<->无`; `纹理损失`: 6; `模糊`: 5; `过度生成`: 1; `过锐`: 1).

## 2. `ntg_convnext + --aux-kinds rank` Folds 0+1 Screen (REJECTED)
We tested adding `--aux-kinds rank` (using the 1,389 `过锐` + `模糊/纹损` rank tuples in `manifest.csv`) to `ntg_convnext` on Folds 0+1 (`ORIG`):
- `ntg_convnext` (no rank): `RPS_ALL=0.05361`, `RPS_ORIG=0.04872`, `过锐_any=0.7463 (0.558/0.263/0.291)`
- `ntg_convnext + rank`: `RPS_ALL=0.05426`, `RPS_ORIG=0.05074` (`+0.00202` worse), `过锐_any=0.7333` (`-0.0130` worse)
So we **closed** `ntg_convnext + rank`.

## 3. Critical Questions for `claude` (`fable 5.2`) Before Using Submission #2 on 644
1. **4090 VRAM Compliance (`<= 24 GiB` NVML)**:
   - `resident_v7` (`11` members resident) hit **`20.70 GB` resident**, **`24.40 GB` torch peak** (`ntg_convnext` 64-tile chunk spike `+3.7 GB`), and **`55.55 GiB` NVML peak** on H200.
   - How should we make `resident_v7` strictly `<= 21 GiB` NVML on a 24 GB RTX 4090 without losing `test_150`/`OOF` gains? (e.g., `native_tiles.py` tile chunk `64 -> 16`, plus `resident_v7_core` or replacing `pe_v3` with `pe_bkseg1` / dropping `sig_v3` & `p3_qy` like `resident_v6_core`?)
2. **`bkseg1` Cross-Dimension Masking & Natural Shallow-DoF Check**:
   - `LEDGER.md` row 25 noted `bkseg1` kept seed human labels for `过锐/过度生成` despite blur. Should we keep `pe_bkseg1` participating in all 6 dims (which gave `+0.08` on `ORIG` and `53.92` on `test_150`) or restrict `pe_bkseg1` to `虚化突兀` only?
   - Before spending submission #2 of 3 on 644, what **fast check** should we run on `pe_bkseg1` / `resident_v7` for **natural shallow-DoF false positives** and **`过锐`/`过度生成`**, and is there any remaining high-ROI, zero-overfit improvement for `过锐` or `过度生成`?
