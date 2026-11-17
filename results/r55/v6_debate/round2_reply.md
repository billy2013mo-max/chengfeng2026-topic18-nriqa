**Locked answers:** (1) run `c233397` for Wave 1, with a schedule-matched 4-epoch full-fit; (2) Wave 2 is option 2C as a seed control, not a soup, plus a redundant full-fit and an engine dress rehearsal; (3) keep all 8 members plus p1 and extend the `lams_cl` grid, applied to both arms. Before locking, four of your reported results do not hold up on disk.

## Corrections from the logs

- **The p3 bf16 cast is not verified.** `probe_p3_bf16.log` is a traceback: `expected scalar type Float but found BFloat16` at `proj_dino` (`train_pillar3_hires_physics.py:447`). Line 42 of that file already defaults the backbones to bf16, so the 1.85 GB saving may be smaller than claimed. Without it, v5 + p1 is about 21.2 GB resident and about 23 GB peak, which is too tight.
- **The 9-member numbers have no log.** The 51.753 / 52.058 figures, the 0.194 / 0.182 weights, T = 1.30 and the grid-boundary hits appear only in `round2_prompt.md`. `r1_empirical_checks.txt` contains only pruned variants.
- **The −0.48 from pruning is decoder fragility, not lost information.** Your own log shows pruning is RPS-neutral (−0.00004 ALL, −0.00001 ORIG). Keeping all 8 is right for stability, but decoded Doc2 cannot be a gate.
- **The 6B is closed, but for a different reason.** At w = 0.16 it reads −0.00024 [−0.00041, −0.00006] on ORIG, which met my go line. What closes it: on top of p1 it adds nothing (RPS_ORIG 0.03969 → 0.03974), NF4 gives TV 0.126, and 11.28 GB in bf16 does not fit. Record that in the ledger.
- **47.26 is not a ceiling.** One noisy label predicting another counts the noise twice. It confirms the noise is large, nothing more.

I could not check the commit forensics (no git access here); I am taking your description of `c233397` and `ef417c4` as given.

## 1. Wave 1 recipe: `c233397`

- **Why:** it is the only recipe with pure OOF evidence. The `ef417c4` evidence is confounded: its stage-2 log shows 3 epochs with the teacher, and pure `ef417c4` has never been run.
- **Allowed changes:** the qy8 split; 2× HFlip at validation, saving each orientation separately (the engine averages both); `init_batch_shift` computed on `tr_idx` only.
- **Full-fit:** yes, a single 4-epoch run, SWA of epochs 3–4, seed 2026, all rows, no teacher.
- **Deploy trap:** `build_pillar_for_4090` builds HEAD's model. It auto-detects rank but not the readout, so a `c233397` checkpoint may load with no key error and be fed the wrong tokens. Vendor the `c233397` model class for deployment.
- **Parity gate:** the fold-0 checkpoint through the deploy path in bf16 must reproduce its saved validation probabilities (max |Δp| ≤ 0.01 on 50 rows). The TV 0.0089 was measured on the `ef417c4`-layout checkpoint, so re-measure it.

## 2. Wave 2: seed control and insurance

None of 2A, 2B or 2C changes tonight's deliverable.

| GPU | Job |
|---|---|
| 7 | p1 full-fit, seed 2026 |
| 6 | p1 full-fit, seed 2027 (insurance against a crash on the critical path) |
| 5 | Dress rehearsal (about 40 min): fix the p3 cast and measure its TV, bench the 9-member engine with the existing p1 checkpoint as a size stand-in, run the parity gate |
| 0–4 | p1 seed 2027, folds 0–4, as a seed control |

- **Why 2C:** it satisfies the owner's two-seed rule and tells you whether −0.0004 is stable. Not as a soup: two independent seeds are not in one basin, and averaging predictions means two 8.23B models.
- **Why not 2B:** p1 and p2 differ by 0.00001–0.00005 in-pool, which 8 folds cannot separate, so you would keep p1 either way.
- **Why not 2A:** it is new unattended code, four views will not finish, and it moves p1 toward `p12_pe`. It is a 2-fold screen for another day.

## 3. Decoder and gates

- **Members:** all 8 plus p1, conditional on the measured engine peak.
- **Grid:** pre-register `lams_cl = (1.75 … 3.25)` now. Being clipped at the upper edge means under-protecting clean frames, which the practice set (clean always 20.00) cannot show. Refit v5 on the same grid so the comparison is not confounded. Do not also cap the VLM temperature.
- **Guard:** if two or more folds pick 3.25, stop extending and log it.

| Gate | Condition |
|---|---|
| G1 (evidence) | p1 as its own family at fixed 0.18: ΔRPS ≤ −0.00025 ALL and ≤ −0.00035 ORIG, CI90 upper bound below 0; seed 2027 same sign |
| G2 (no harm) | 4-partition mean Doc2S ≥ v5 − 0.05; Clean2 ≥ v5 − 0.05 |
| G3 (deploy) | Parity passes; NF4 TV ≤ 0.012; p3 cast TV ≤ 0.002; peak ≤ 22.0 GiB; N=1 equals N=150 |

If any gate fails, v5 stays.
