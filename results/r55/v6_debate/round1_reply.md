**Verdict:** take Lever 1 with one VLM (p1) only, screen Lever 2 before spending on it, and do not run Lever 3. The honest expectation is about +0.2–0.35 OOF (ORIG 51.94 → ~52.2–52.3), not 53.6–54.5; no number in the repo supports that range. I only have read access in this session, so nothing is launched and `round1_reply.md` is still empty.

## 1. What the diagnosis gets wrong

- **"+8.83 from oracle grading" is mostly label noise, not headroom.** Sibling frames get the same grade only 50–61% of the time; for 过度生成 P1 the sibling's label is P2 33% / P1 34% / P0 23%. Every architecture plateauing at AUC 0.71–0.73 and p1+p2 adding 0.0000 there fits a noise floor, not an architectural blind spot.
- **−0.00070 is the best of several looks.** That is ORIG, both VLMs, 22% weight. The deployable case (one VLM) is −0.00040 ALL / −0.00056 ORIG. Using the ledger's own pairs (s1→primary: −0.00106 RPS for +0.41; halfA: +0.0036 for −2.07), that is about +0.15–0.35 Doc2.
- **Lever 1 is a rediscovery.** p1 in-pool was already measured at +0.2 all / +0.25–0.4 first-batch (2026-11-13). The Wave 2A retirement used the wrong criterion, so reinstating it is right, but the size is known.
- **The SDPA "5–8× faster" claim is arithmetic-wrong.** At 1,025 tokens × 3,200 width, attention matmuls are about 5% of forward FLOPs (N/6d), so SDPA can save at most ~5% of compute. The real waste is the per-epoch prototype-bank refresh plus validation TTA, about 30% of each 2,920 s epoch.
- **InternViT-6B is not obviously strong.** Its epoch-2 fold scores average 48.1 Doc2, with the decoder fitted on the same validation fold. PE-L (0.33B) scores 49.6–49.8 argmax, and PE-G, SigLIP2-giant, and the three native-tile backbones all failed in-pool.
- **The p3 "control check" proves nothing about p1/p2.** It compares a different model, recipe and data fraction. The valid evidence is the strict-row filter plus replication across two VLMs.

## 2. Leak audit of p1/p2 `cv5`

- **Scrutiny boxes:** clean. They come from wavelet maps and a Depth-Anything probe, with no labels.
- **Stage-0 TAPT:** the trainer never loads a stage-0 checkpoint, and `run_phoenix_pipeline.sh:220` passes no `--model-path`. I did not open `phoenix/config.py` to confirm the base path resolves to `models/qwen3vl-8b`.
- **`init_batch_shift`:** 24 label statistics computed on all rows including validation folds. It is immaterial (disabled at inference), but should be zeroed or computed per fold in the rerun.
- **Fold hygiene:** unresolved. Ledger line 68 says 370 leaked video groups for the 5-fold; the forensics finds 100 leaked rows. Retraining on `phoenix_qy8.csv` removes the question.
- **Recipe drift (the real trap):** `pillar1_cv5_fold0.log` shows train=2858 and 715 steps per epoch, meaning no oversampling. HEAD always applies `oversample_rare_real_p1_p0` and has a newer readout. Rerunning HEAD is not the recipe the −0.0004 was measured on.
- **Existing full-fit:** `pillar1_fulldata_swa.pt` was trained with the 3-pillar teacher and a 3+2+2 epoch schedule, so it is not schedule-matched to `cv5`.

My read: the gain is real (4–7× the in-pool seed noise, replicated on two VLMs), but it is not yet shown to survive the routed decoder, 4-bit deployment, or the HEAD recipe.

## 3. Lever verdicts

| Lever | Verdict | Expected | Reason |
|---|---|---|---|
| 1: p1 (Qwen3-VL-8B) | Go | +0.15–0.35 Doc2 | Only family to clear an in-pool gate with margin; 4-bit path verified |
| 1: p2 (Clef-Flash) | Backup only | +0.00008 / +0.00014 on top of p1 | Below the −0.00015 gate; 4-bit fidelity never verified; cannot fit alongside p1 |
| 1: nested KD into the VLM | No | 0 | Stage-2 KD improved solo RPS but left blend ΔRPS at −0.00040; KD removes the decorrelation that is its value |
| 2: InternViT-6B | Screen first | my guess: 20–25% to pass | Eight epoch-≤2 fold checkpoints exist; inference-only OOF costs about 12 min per fold |
| 3: `ntg_p12_routed` | No | under 10% | It is the Defect-Zoom tower that Step 0a closed by pre-registered stop rule (ΔAUC +0.0002…+0.0051 vs +0.030) |

Budget, using ledger figures:

| Composition | Params | Resident / peak | Fits 24 GB |
|---|---|---|---|
| v5 + p1 (NF4) | 15.9B | ~21.2 / ~23.1 GiB | Too tight |
| v5 − `pe_v3` − `sig_v3` + p1 | 15.1B | ~19.6 / ~21.5 GiB (est.) | Yes |
| v5 + 6B (bf16) | 13.2B | ~27.5 / ~29.5 GiB | No |
| v5 + p1 + p2 + 6B | 31.5B | n/a | Over the 30B cap |

The 6B and p1 compete for the same ~6 GiB. They can only coexist if the 6B replaces `v16`, and a quantized vision tower has never been fidelity-tested here.

## 4. Overnight plan

| Time | GPUs | Job |
|---|---|---|
| 0:00 | CPU | Count groups spanning folds in `phoenix_5fold.csv`; score sibling-label-as-prediction F1 per cell (the noise ceiling) |
| 0:10–2:20 | 0–7 | p1 on `phoenix_qy8.csv`, 8 folds, the old `cv5` recipe recovered from git (4 epochs, no teacher); about 2.1 h per fold |
| during wave 1 | 0–7, shared | 6B screen: inference from the eight `fold_k_best.pt`, in-pool ΔRPS |
| 2:30–4:50 | 7 | p1 full-fit, schedule-matched (4 epochs × ~2,090 s) |
| 2:30–4:30 | 0–6 | If the 6B screen passes: 6B fixed 3 epochs without per-epoch validation (~1.9 h per fold), verdict only. Otherwise: p1 seed 2027 on folds 0–1, plus p2 on qy8 as backup |
| 4:50–6:00 | 0 | Decoder refit, engine build, fidelity check, fresh `test_150` inference; then stop all processes |

Pre-registered gates:

- **G1 (evidence):** p1 as its own family at fixed 0.18 on v5: ΔRPS ≤ −0.00025 ALL and ≤ −0.00035 ORIG, CI90 upper bound below 0, no defect harmed. Also standalone RPS ≤ 0.0505, as a recipe-drift check.
- **G2 (pruning):** dropping `pe_v3` and `sig_v3` costs ≤ +0.00005.
- **G3 (deploy):** merged NF4 vs bf16 TV ≤ 0.012; peak ≤ 22 GiB; N=1 equals N=150; full-fit beats OOF.
- **6B screen:** go at ≤ −0.00010 with CI below 0; no-go above −0.00003.
- **Stop rule:** if G1 fails, v5 stays and nothing new ships.

## 5. What to expect in the morning

If G1–G3 pass, there is one new JSON. The paired 150-image SD is about 0.78, so a true +0.3 shows as an improvement online only about 65% of the time, and reads ≥ 53.6 about one time in three, mostly by luck. The practice board cannot confirm or refute this change; the OOF ΔRPS with CI is the evidence.

I did not check current GPU state, file timestamps, or 4090 latency (it has only ever been measured on H200).
