# Convergence, Pruning & Breakthrough Review (for `claude-fable-5-2`)

## Context & User Mandate
We have delivered `resident_v6` (9 models, 11.821B params, 19.96 GiB res / 21.92 GiB peak, `p0 ORIG = 52.034`, `mean4 ORIG = 51.986`, `Tag ORIG = 51.747`, `N=1 == N=150` exact 0/900 diff).
Now the user asks us to **review & prune/converge the entire pipeline** so it is cleaner, simpler, and easier for the competition organizers to reproduce and iterate on, **without hurting (and potentially improving!) the score**, and to **discuss and execute** the agreed plan together.

## Hard Empirical Data Just Measured on CPU (Exact Honest 8-Fold OOF + 4-Partition + TagSplit)

### 1. Leave-One-Out (LOO) Ablation from `FULL_v6 (9m)`
- `FULL_v6 (9m)`: `2.83s | 20.61G | RPS_ALL=0.04197 ORIG=0.03781 mAP=0.6644 | Doc2_ALL=51.693 ORIG=52.034`
- **3 Indispensable Pillars (removing ANY hurts RPS_ORIG with 90% CI excluding 0):**
  - `drop_v16`: `dRPS_ORIG = +0.00035 [+0.00020, +0.00051]`, `Doc2_ORIG -> 51.514 (-0.520)`
  - `drop_p1_qy`: `dRPS_ORIG = +0.00032 [+0.00018, +0.00046]`, `RPS_ALL -> 0.04217`
  - `drop_ntg_pe`: `dRPS_ORIG = +0.00028 [+0.00012, +0.00043]`, `mAP14 -> 0.6625`
- **Noise / Redundant Member:**
  - `drop_sig_v3`: `dRPS_ORIG = -0.00002 [-0.00007, +0.00002]`, `mAP=0.6644`, `Doc2_ALL -> 51.712 (+0.019)`, `Doc2_ORIG -> 52.110 (+0.076)`!
- **Borderline / Low-impact members in LOO:**
  - `drop_pe_v3`: `dRPS_ORIG = +0.00001 [-0.00003, +0.00004]`
  - `drop_peg`: `dRPS_ORIG = +0.00003 [-0.00013, +0.00018]` (note: `peg` provides unseen-source 花屏 rescue `98.6%` recall @ `100%` prec!)
  - `drop_p12_pe`: `dRPS_ORIG = +0.00001 [-0.00013, +0.00014]`
  - `drop_cradio_v3`: `dRPS_ORIG = +0.00004 [-0.00002, +0.00011]`
  - `drop_p3_qy`: `dRPS_ORIG = -0.00005 [-0.00017, +0.00006]` (saves `2.75 GiB` + `0.62s` + 4-backbone physics tower!)

### 2. Multi-Partition (4×8-fold) + TagSplit + Hardware Matrix (`9m` down to `3m`)
```
v6_full_9m       | 9m 11.82B 20.61G 2.83s | RPS=0.04197 ORIG=0.03781 (ref)                         | mAP=0.6644 | mean4 ALL=51.635 ORIG=51.986 Cln=19.795 | p0 ORIG=52.034 | Tag ORIG=51.747 | P<15=0.41%
v6_nosig_8m      | 8m 11.37B 19.71G 2.70s | RPS=0.04197 ORIG=0.03779 (d=-0.00002 [-0.00007,+0.00002])| mAP=0.6644 | mean4 ALL=51.676 ORIG=52.050 Cln=19.795 | p0 ORIG=52.110 | Tag ORIG=51.837 | P<15=0.35%
v6_pegonly_7m    | 7m 11.04B 19.06G 2.61s | RPS=0.04199 ORIG=0.03780 (d=-0.00001 [-0.00008,+0.00004])| mAP=0.6642 | mean4 ALL=51.588 ORIG=51.909 Cln=19.776 | p0 ORIG=51.934 | Tag ORIG=51.733 | P<15=0.59%
v6_core_nop12_6m | 6m 10.71B 18.41G 2.42s | RPS=0.04206 ORIG=0.03781 (d=+0.00000 [-0.00017,+0.00017])| mAP=0.6638 | mean4 ALL=51.565 ORIG=51.761 Cln=19.776 | p0 ORIG=51.837 | Tag ORIG=51.625 | P<15=1.07%
v6_core_nop3_6m  | 6m  9.77B 16.31G 1.98s | RPS=0.04204 ORIG=0.03774 (d=-0.00006 [-0.00020,+0.00007])| mAP=0.6640 | mean4 ALL=51.535 ORIG=51.837 Cln=19.756 | p0 ORIG=51.873 | Tag ORIG=51.912 | P<15=0.60%
v6_slim_5m       | 5m  9.09B 14.93G 1.80s | RPS=0.04219 ORIG=0.03781 (d=+0.00000 [-0.00016,+0.00016])| mAP=0.6617 | mean4 ALL=51.513 ORIG=51.955 Cln=19.776 | p0 ORIG=51.992 | Tag ORIG=51.935 | P<15=0.57%
v6_ultra_4m      | 4m  8.76B 14.28G 1.61s | RPS=0.04233 ORIG=0.03787 (d=+0.00007 [-0.00017,+0.00032])| mAP=0.6601 | mean4 ALL=51.599 ORIG=51.946 Cln=19.825 | p0 ORIG=51.976 | Tag ORIG=51.732 | P<15=0.86%
v6_3fam_3m       | 3m  8.43B 13.63G 1.52s | RPS=0.04252 ORIG=0.03800 (d=+0.00019 [-0.00017,+0.00056])| mAP=0.6597 | mean4 ALL=51.290 ORIG=51.523 Cln=19.630 | p0 ORIG=51.537 | Tag ORIG=51.379 | P<15=2.47%
```

## Questions for `claude-fable-5-2`
1. **Which converged architecture should we promote and ship?**
   - **Option A (`v6_nosig_8m`)**: Strictly dominates `9m` on EVERY metric (`mean4 ORIG +0.064 -> 52.050`, `Tag ORIG +0.090 -> 51.837`, `P<15 = 0.35%`), removes `SigLIP2-NaFlex`, zero score loss.
   - **Option B (`v6_slim_5m = [peg, v16, ntg_pe, p12_pe, p1_qy]`)**: Cuts **9 -> 5 models**, **-5.7 GiB VRAM** (`14.93 GiB`), **-36% latency** (`1.80s` H200 / `~3.8s` 4090), eliminates `p3_qy` (4-backbone physics tower), `cradio_v3`, `pe_v3`, `sig_v3`, keeps `RPS_ORIG = 0.03781` identical AND achieves the **HIGHEST `TagSplit ORIG = 51.935` (`+0.188` over `9m`!)**, `p0 ORIG = 51.992`, `P<15 = 0.57%`.
   - **Option C**: Ship **BOTH** cleanly (`resident_v6_nosig` as the max-score 8m primary, AND `resident_v6_slim` as the 5m converged fast/reproducible architecture)? Or pick ONE clear winner? Give your strict recommendation!
2. **Can we also clean up unused/dead code paths in `final_engine.py` / `routed_eu_decoder.py` without breaking backwards compatibility?** (e.g. legacy `peg`-only vs `pe_v3+peg` branch, unused `d55`/`eu`/`vote` clutter, or keep them cleanly isolated).
3. **Any remaining CPU-only zero-leakage decoder improvement** on top of the converged set (without overfitting)?
Keep your reply concise, sharp, and actionable (~40 lines).
