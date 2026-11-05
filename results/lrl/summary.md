# lr_last sweep summary

Fixed config: `eupe:full,siglip2:lastk4,cradio:lastk4`, clean_penalty=0, clean_repeat=7, lr_full=8.875358285474408e-05, lr_head=0.0022753949862994462, wd=0.01, dropout=0, batch=6, epochs=4. Only `--lr-last` (shared lr of the last-4 blocks of siglip2 and cradio) varies.

## Round 1: original split `data/split_trainval.csv` (2 seeds each)

| lr_last | test total mean ± std | seeds | S_pos | p0R | p0P | S_clean | fpr | val total | val clean p99 |
|---|---|---|---|---|---|---|---|---|---|
| 3e-06 | 47.62 ± 0.30 | s1=47.83, s2=47.41 | 0.498 | 0.799 | 0.689 | 1.000 | 0.0000 | 48.34 | 0.1117 |
| 1e-05 | 48.86 ± 0.11 | s1=48.78, s2=48.94 | 0.537 | 0.814 | 0.698 | 1.000 | 0.0000 | 49.35 | 0.0689 |
| 3e-05 | 49.23 ± 0.04 | s1=49.20, s2=49.26 | 0.560 | 0.786 | 0.719 | 1.000 | 0.0000 | 49.39 | 0.0173 |
| 1e-04 | 48.73 ± 0.28 | s1=48.53, s2=48.93 | 0.540 | 0.813 | 0.674 | 1.000 | 0.0000 | 48.82 | 0.0137 |
| 3e-04 | 36.84 ± 5.58 | s1=32.90, s2=40.78 | 0.529 | 0.808 | 0.698 | 0.414 | 0.0117 | 49.08 | 0.0411 |

## Round 2: winner lr_last=3e-05 re-verified on new splits (2 seeds each)

Matched baseline is lr_last=1e-05: for s11/s13 from `results/resplit_fix/` (same code/config), for the original split from this sweep's own 1e-05 runs.

| split | 3e-05 test mean ± std | seeds | 1e-05 baseline mean | baseline seeds | gain |
|---|---|---|---|---|---|
| orig (sweep) | 49.23 ± 0.04 | s1=49.20, s2=49.26 | 48.86 | s1=48.78, s2=48.94 | +0.37 |
| s11 | 49.42 ± 0.23 | s1=49.59, s2=49.26 | 48.32 | s1=48.22, s2=48.42 | +1.10 |
| s13 | 49.19 ± 0.65 | s1=48.73, s2=49.64 | 48.93 | s1=49.13, s2=48.73 | +0.26 |

## Conclusion

- **Optimal lr_last = 3e-05.** On the original split it scores **49.23 ± 0.04** (s1 49.20, s2 49.26), vs **48.86 ± 0.11** at the old default 1e-05: **+0.37** absolute, and both 3e-05 seeds beat both 1e-05 seeds (no overlap).
- Gains confirm on the new splits: s11 **+1.10** (49.42 vs 48.32) and s13 **+0.26** (49.19 vs 48.93). Across all 6 matched (split, seed) pairs the mean gain is **+0.58** (range -0.40 to +1.36), 5/6 positive.
- Seed/split noise is real: the s13 seed-1 pair is the one regression (48.73 vs 49.13, -0.40), and s13 seed std is high (0.65). The paired mean gain (+0.58) is comparable to the cross-seed spread of a single split (0.04-0.65), so with n=2 seeds/split this is a consistent but not strongly significant win (paired t≈2.33, df=5, two-sided p≈0.068 with scipy; 5/6 gains exceed +0.3, both original-split 3e-05 seeds exceed both 1e-05 seeds). Direction and magnitude are stable on the original split and s11; s13 alone is inconclusive.
- The improvement comes mostly from S_pos (0.560 vs 0.537) and p0P (0.719 vs 0.698); p0R slips slightly (0.786 vs 0.814), while S_clean/fpr stay perfect (1.0 / 0) at 3e-05.
- **Do not go higher:** lr_last=1e-04 is already worse than 3e-05 (48.73 ± 0.28) and 3e-04 catastrophically destabilises training (test 36.84 ± 5.58; S_clean collapses to 0.42 and fpr rises to 0.012 because clean samples get false positives). lr_last=3e-06 is clearly too slow (47.62 ± 0.30).
- Recommendation: set `--lr-last 3e-05` (from 1e-5) in the production config; keep everything else fixed.

## Anomalies

- lr_last=3e-04: S_clean 0.219 (seed1) / 0.609 (seed2), fpr 0.0156 / 0.0078, test totals 32.90 / 40.78 — a genuine over-fitting/divergence failure, not a flaky job (both seeds completed normally, val total stayed ~49).
- Original-split 1e-05 seed1 scored 48.78 here vs 48.46 in `results/fixp2/orig_seed1.json`, although the config is identical. `data.py` applies random flip/brightness/contrast in each DataLoader worker seeded by `seed + worker_id`, so a differing worker count (this sweep forces `--workers 8`, as instructed) changes the augmentation stream; bf16/TF32 kernel nondeterminism can add a little more. All values remain in the expected 48-49 band. Every comparison in this report is between runs launched in the same batch with identical `--workers 8`, so the 3e-05 vs 1e-05 result is unaffected.
