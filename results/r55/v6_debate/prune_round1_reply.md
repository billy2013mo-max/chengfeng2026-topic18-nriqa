**Verdict:** ship one set, `v6_core_nop3_6m`, which by its parameter count is `[cradio_v3, peg, v16, ntg_pe, p12_pe, p1_qy]` (please confirm). Not A, B or C. Describe it as "equivalent and simpler", not "better": nothing in the matrix shows a score gain.

I could not rerun anything, because Python execution was not permitted in this session. The 9m row matches `resident_v6_decoder_routed_gate_eu.json` and `v6_gate_report.json`; the LOO and the 8-row matrix have no script or log on disk.

## 1. Which set

The decoded differences are noise and cannot rank the sets:
- **Exchange rate:** adding `p1_qy` moved routed RPS_ORIG by −0.00032 and mean4 ORIG by only +0.099. The prune deltas are at most 0.00007, worth about 0.03 Doc2.
- **Spread:** 9m alone ranges 51.91–52.03 ORIG across its four partitions; 9m down to 4m span 51.76–52.05. "A dominates by +0.064" and "B has the highest TagSplit" are the maximum of eight noisy numbers.
- **TagSplit:** it refits only the decoder on tag folds. Member OOFs still come from the video folds, so it says nothing about unseen pipelines.

What the probabilities show:

| Set | ΔRPS ALL | ΔRPS ORIG | mAP14 | Memory (your column) | Verdict |
|---|---|---|---|---|---|
| A `nosig_8m` | 0.00000 | −0.00002 | 0.6644 | 19.71 G | Safe, but keeps `p3_qy`, `pe_v3` and a ~21 GiB peak |
| `core_nop3_6m` | +0.00007 | −0.00006 | 0.6640 | 16.31 G | **Ship** |
| B `slim_5m` | +0.00022 | 0.00000 | 0.6617 | 14.93 G | First step with a visible cost |

- **Why 6m:** `sig_v3`, `pe_v3` and `p3_qy` each have LOO ≤ +0.00001 on ORIG. Removing all three saves 4.3 GiB, 0.85 s, 2.05B parameters and the whole Pillar-3 training stack.
- **Why not B:** dropping `cradio_v3` is where a cost appears: mAP −0.0023, and supplement-row RPS about +1% against +0.5% for 6m (derived from your ALL/ORIG figures). It also leaves three of five members on PE-Core-L. C-RADIO costs 1.37 GiB and 0.18 s.
- **Why not C:** two deliverables double what organisers must reproduce and invite choosing by practice score (rule 2).
- **Keep `peg`:** OOF cannot measure its 花屏 rescue, since 209 of 211 positives come from one source.

Decision rule: the smallest set with point ΔRPS ≤ +0.00010 and CI90 upper bound ≤ +0.00030 (one pillar) on both ALL and ORIG. I wrote it after seeing the matrix, so it is a rule, not a test. The ALL-row CIs are missing from the matrix.

## 2. Cleanup

- **Pickles:** do not move or rename pickled classes (`RoutedGateEUDecoder`, `TrueGateConstrainedEU`, `RoutedEUDecoder`, `DecoderEU`, the video aggregator). Add a JSON export and loader for the shipped decoder; the current JSON lacks `eu.st`.
- **Isolate, don't delete:** move the older sets and their branches (`p1`, legacy `p3`, `p12_g14`, the `OOF_KEY` bank, `d55`/`eu`/`vote`, `RoutedEUDecoder` fitting) behind a legacy module. Keep `resident_v6` as a frozen reference.
- **One prior builder:** the family map exists three times (`final_engine.FAMILY`, the fallback in `RoutedGateEUDecoder.__init__`, `eval_v6_gates.build_proto_cfg`). The shipped pickle comes from the first and the reported gates from the third.
- **Fix while there:**
  - `final_engine.py:816` requires the unused d55 pickle.
  - `final_engine.py:461` turns on `v16_chunk=2` only when `p1_qy` is present; make it explicit per set.
  - The saved `pooled` array is the fixed-prior pool, not the routed probabilities that were decoded.
  - With `pe_v3` gone, `pool()` falls into the old unthresholded 花屏 max rule while the routed decoder uses ≥ 0.90.
  - Docstrings say 0.80 for that threshold; the code uses 0.90.
- **Reproducibility gap in `p12_pe`:** `kd_provenance_audit.json` shows the committed builder reproduces the CV teacher (MAE 6e-7) but not the full-fit teacher the shipped checkpoint was trained on (MAE 0.0234). Nothing in the repo builds `fullfit_halfAB_kd_teacher.npz`. Either retrain the `p12_pe` full-fit on `clean_honest_kd_teacher_fullfit.npz`, or first check on CPU whether the no-KD OOF (`r55_p12_e5_s2026`) is equivalent in the pool, which would remove the 32-run teacher chain.

## 3. Decoder

No. In-sample 51.711 against nested 51.693 leaves nothing to harvest, and decoder v2 was null. λ sits on the grid edge in 4–5 of 8 folds on every partition, which the round-2 guard said to stop at. Refit the new set with the same frozen hyper-parameters and grid.

## Execution order

1. Commit the prune script and its output, adding paired video-cluster CI90 for ΔRPS on ALL and for decoded ΔDoc2 (reported, not gated).
2. Apply the rule, add the set, run `--fit-decoder`.
3. Run `--bench 40 --half` for real; the pruned rows have no bench file, and the 9m "20.61G" matches neither the measured 19.96 resident nor the 21.92 peak. Gates:
   - Peak reserved memory plus `nvidia-smi` ≤ 20 GiB. `max_memory_allocated` excludes allocator slack and the CUDA context, and 9m cleared its 22.0 gate by only 0.08 GiB.
   - Per-member probabilities identical to `resident_v6_bench_probs_half.npz` (same 40 labelled images).
   - N=1 equals N=150.
4. Run fresh inference for the new JSON (rule 1: no re-decoding of the saved 9m probabilities). No practice upload to choose between sets.
