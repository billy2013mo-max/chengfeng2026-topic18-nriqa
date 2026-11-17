# 命题18 — technical review, evidence, and executable plan (2026-11-11, revision 2)

Guiding principle: every decision serves real task performance and generalisation. `test150` is one evaluation
set and was not used for any decision here. All numbers are nested out-of-fold (OOF) on the 3,573 labelled images
with the `Doc2_Strict` metric unless stated. "Confirmed" = measured or read in code today; "hypothesis" = not yet
measured. Results are appended in §9. A second session (18-ab) works in this repo: it owns the same-script control,
stack-level gates, the `s1` full-fit and the parameter/latency measurement; this document owns the training recipe.

## 0. Bottom line

1. The system is **information-limited, not architecture-limited**. Six unrelated designs (three single-backbone
   towers, a 5-backbone 4.5B tower, three VLM pillars) all land at 48.3–50.5 alone and ≈51 together.
2. The loss is **grade confusion between neighbours**, concentrated in P1/P2 of 过度生成, 过锐 and 模糊. Detection
   is solved (AUC 0.95); adjacent-grade AUC is 0.70–0.89.
3. Human labels are inconsistent at that level, but the model is **not at the consistency ceiling** for P1/P2: on
   the 844 sibling frames a pessimistic bound gives macro-F1 0.73 against the model's 0.63.
4. Two **evaluation defects** inflate or blur what we measure: checkpoints are chosen on the validation fold, and
   `S_clean` mostly measures "unprocessed source vs processed frame", not absence of defects.
5. A concrete **real-world weakness**: clean frames that went through a processing pipeline are flagged 32 % of the
   time by the ensemble (39–48 % by single towers), against 1.5–4 % for other clean images.
6. The plan therefore buys *information and regularisation* (in-domain unlabelled frames, cross-fitted soft labels,
   random native crops, rebalancing the processed-clean subgroup) and fixes the measurement first. I do not expect a
   step change: the evidence supports gains of tenths of a point per item, not points.

## 1. What was inspected

| Area | Files |
|---|---|
| Model, loss, trainer | `code/qingyu/backbones_qingyu.py`, `unimodal_xbm_losses.py`, `train_tier1_5backbones_8fold.py`, `code/r55/train_testbed.py` |
| Decoding, stacking, gates | `code/next55/decoder55.py`, `stack55.py`, `common55.py`, `code/r55/gate_r55.py` |
| Data pipeline | `code/qingyu/dual_engine_preprocess.py` (guided crops, physics vector), `data/qingyu_splits/`, `code/r55/build_*`, `ingest_external_pool.py`, `prep_aux.py`, `certify_clean.py` |
| Deliverable path | `code/next55/launch_fullfit.py`, `make_fullfit_splits.py`, `infer_tier1_ckpt.py` |
| Other lines | `code/finetune/train_v16_microscope_specialist.py` + launcher, `code/phoenix/*`, `run_phoenix_pipeline.sh` |
| Results | `results/next55/LEDGER.md`, tower OOF files, Phoenix OOF files, fold logs |
| Rules | `documents/…命题18 (2).md`: ≤30B parameters for trained solutions, single-4090 reference, ≤15 s/image, 654-image final set, practice set "not a tuning target". External data and pretrained models are neither permitted nor forbidden in the text. Per the owner's instruction these facts are recorded but do not restrict the research space. `AGENTS.md` rules 1–2 (no legacy JSONs, no test-set tuning) stay binding. |

## 2. Confirmed measurements

### 2.1 Where the points are lost (`s1`, 8 towers, 51.04 = S_pos 15.57/25 + S_p0 16.10/20 + S_clean 19.38/20)

| Dimension | F1 P2 / P1 / P0 | exact grade given positive | within one grade | false alarms on grade 0 |
|---|---|---|---|---|
| 模糊 | 0.45 / 0.52 / 0.82 | 0.58 | 0.96 | 0.09 |
| 纹理损失 | 0.65 / 0.66 / 0.80 | 0.69 | 0.99 | 0.17 |
| 过锐 | 0.54 / 0.48 / 0.74 | 0.56 | 0.96 | 0.12 |
| 过度生成 | 0.35 / 0.30 / 0.62 | 0.41 | 0.88 | 0.03 |
| 花屏 (P0 only) | – / – / 0.98 | 0.98 | – | 0.00 |
| 虚化突兀 (P0 only) | – / – / 0.82 | 0.86 | – | 0.02 |

One channel is worth 25/14 = 1.79 points per unit of F1. If 过度生成 reached the level of 纹理损失 the score would
rise by ≈2.0 (1.5 in S_pos, 0.5 in S_p0); it is the single most valuable dimension. Adjacent-grade AUC of one
PE-Core tower: 模糊 0.81/0.89, 纹理损失 0.84/0.89, 过锐 0.79/0.86, 过度生成 0.71/0.70 (P2|P1, P1|P0).

### 2.2 Architecture is not the bottleneck

| Model | Params | Score |
|---|---|---|
| SigLIP2-so400m / C-RADIOv4-H / PE-Core-L towers (2 seeds each) | 0.4–0.7B | 48.3–49.6 |
| Phoenix pillars 1/2/3 after stage 2 (VLM-style, 5-fold) | — | 48.9 / 49.3 / 49.8 |
| V16 (PE-Core-G + DINOv3-L + C-RADIO + EVA02 + ConvNeXt fused in one tower, synthetic 过度生成/虚化 with class masks) | 4.5B | 50.3 / 50.5 |
| `s1` = 6 Tier-1 towers + 2 V16 | | 51.04 |
| `s1` + 3 Phoenix pillars (18-ab) | | 51.39 (+0.36, CI90 [+0.07, +0.64]) |

Error indicators of different towers correlate 0.58–0.71; 5.9 % of all image×dimension cells are wrong for all seven
towers examined, the ensemble is wrong on 15.5 %. Doubling towers gave +0.46, three more pillars +0.36: diminishing.

### 2.3 Label consistency (844 images in 320 multi-frame videos)

Using a sibling frame's human label as the prediction scores 46.93; the ensemble scores 51.88 on the same images.
If sibling differences were pure annotation noise, a perfect model would reach about √(sibling F1) per channel:

| | sibling F1 | √ bound | model |
|---|---|---|---|
| macro over 14 channels | 0.545 | 0.728 | 0.634 |
| 过锐 P1 | 0.40 | 0.63 | 0.43 |
| 过度生成 P1 / P2 | 0.27 / 0.41 | 0.52 / 0.64 | 0.30 / 0.46 |
| 模糊 P2 | 0.37 | 0.60 | 0.43 |
| 模糊 P0 / 纹理损失 P0 | 0.63 / 0.64 | 0.80 / 0.80 | 0.83 / 0.84 |

The bound is pessimistic (siblings are different frames, 50 s to 400 s apart), and the model already exceeds it on
P0. So the mid grades have real headroom; an OOF of 55 would need the model at or above this bound on every channel.

### 2.4 Training dynamics and selection

| epoch | 1 | 3 | 5 | 6 | 8 |
|---|---|---|---|---|---|
| training loss | 1.00 | 0.47 | 0.26 | 0.17 | 0.06 |
| validation ECE | 0.059 | 0.048 | 0.071 | 0.086 | 0.096 |
| validation score (in-fold decoder) | 44.9 | 48.4 | 49.0 | 49.3 | 49.5 |

- Every image is shown through the same five views each epoch: a 448×448 squash of the whole frame plus four fixed
  guided 448-px native crops; only h-flip and ±5 % brightness/contrast vary (`train_tier1…:122-180`).
- The checkpoint of each fold is chosen on that fold's validation data (`train_tier1…:439-471`,
  `train_testbed.py`). Mean best-epoch score 49.98 vs last-epoch 49.47: up to ≈0.5 of selection optimism per tower.
- The full-fit launcher reuses that rule with a validation set that is a subset of the training set
  (`make_fullfit_splits.py`), so a shipped model would be selected on memorised data.

### 2.5 `S_clean` measures the source, not the absence of defects

| Clean images (513) | n | flagged by `s1` |
|---|---|---|
| untagged frames with a timestamp name (unprocessed sources) | 388 | 1.5 % |
| stock photographs (`forest_000044.jpg` …) | 76 | 0 % |
| other | 18 | 0 % |
| **frames carrying a processing-pipeline tag** | **31** | **32 %** (single towers 39–48 %) |

Clean images have a median short side of 1080 px and 5 % are 720p; defective images are 61 % 720p. Among frames
with a pipeline tag 98.1 % have a defect and 92.8 % have 纹理损失. The classifier has learned that a processed look
implies a defect. In production most inspected frames are processed and presumably fine, so the false-alarm rate
there is likely far above what S_clean = 19.4 suggests. n = 31 is small (95 % interval roughly 17–51 %), but the
contrast with 1.5 % is not noise.

### 2.6 The gates are noisy; a probability-level gate is not

`code/r55/gate_plus.py` (new) compares probabilities with ranked probability score, NLL and adjacent-grade AUC under
a cluster bootstrap over videos. On it the `og` experiment is a null (ΔRPS +0.0001, ΔAUC +0.0004), not the −0.40
loss the thresholded score reported. The seed-2026 baseline (original trainer) and a seed-2028 run (testbed trainer)
differ in NLL by +0.037 (CI excludes 0): the two trainers are not calibration-equivalent, so the failed screens were
compared against a slightly different recipe.

## 3. Failure modes

| ID | Failure mode | Evidence | Status |
|---|---|---|---|
| FM1 | Too little label information for mid-grade boundaries | §2.2 saturation across designs; §2.3 headroom; old learning curve 47.4/48.3/49.4 at 25/50/75 % | confirmed pattern; slope under the current protocol to be measured (T2) |
| FM2 | Memorisation of fixed views | §2.4 | confirmed |
| FM3 | Optimistic and noisy measurement | §2.4 selection, §2.6 | confirmed |
| FM4 | Source/processing shortcut for "clean" | §2.5 | confirmed (small n) |
| FM5 | 过度生成 grading far behind other dimensions | §2.1 AUC 0.70 vs 0.79–0.89 | confirmed; cause unknown (label noise vs missing local evidence) |
| FM6 | No shippable model | no full-fit, no timing | confirmed; 18-ab is running the `s1` full-fit |

## 4. Research synthesis, organised by the decision it informs

**Random native crops (for FM2).** Quality models are normally trained on random native-resolution patches and
tested by averaging many of them, because resizing removes the evidence: [MUSIQ](https://arxiv.org/abs/2108.05997)
works on native resolution; [FAST-VQA](https://arxiv.org/abs/2207.02595) samples raw-resolution fragments; HyperIQA
averages 25 random patches at test time (reported in the [UHD-IQA benchmark](https://arxiv.org/pdf/2406.17472)). The
MANIQA crop count could not be verified from primary text and is not relied on. Transfer: direct — our crops are
already native-pixel, only their positions are frozen. Unknown: guided crops may carry more signal than random ones,
so the experiment mixes both (50 % random, otherwise ±96 px jitter).

**Unlabelled in-domain data and distillation (for FM1).** [Noisy Student](https://arxiv.org/abs/1911.04252): an
un-noised teacher labels unlabelled images, a noised student learns from labels plus soft pseudo-labels; soft labels
are the safer choice for out-of-domain data; teacher noise hurts. An NR-IQA instance placed third at NTIRE 2022
([Conde et al.](https://arxiv.org/abs/2204.12819)). [Allen-Zhu & Li](https://arxiv.org/abs/2012.09816) explain why
a student trained on an ensemble's outputs can inherit the ensemble's gain. Local evidence: Phoenix stage 2
(teacher OOF distillation) moved pillars 2 and 3 by +0.6 and +0.8 and pillar 1 by +0.03, but its teacher was a 5-fold
OOF, which leaks validation labels at second order. Conflict: `rank`, `og` and `tag` auxiliary data did nothing; those
were synthetic or out-of-domain, which is exactly where the literature expects little. Unresolved: how much an
in-domain unlabelled frame is worth here — measured directly by T2 before any external data is used.

**An in-domain pool exists.** [KVQ](https://arxiv.org/abs/2402.07220) (Kuaishou, CVPR 2024): 600 user uploads and
3,600 versions produced by the platform's pre-processing, transcoding and enhancement, with MOS from professional
raters; the 2,926 training videos are public and are now on disk. The NTIRE 2024 challenge on it
([report](https://arxiv.org/abs/2404.11313)) was won by ensembles with external pretraining (top teams averaged 7–10
models and used LSVQ/Q-Align/in-house data); no team reported pseudo-labelling. The NTIRE 2025 follow-up
([KwaiSR](https://arxiv.org/abs/2504.15003)) describes the same defects we grade: face deformation and "AIGC-style
textures" from generative enhancement. KVQ has no grade labels in our taxonomy; it is an unlabelled pool plus MOS.

**Label noise (for FM1).** Methods that soften or correct ordinal labels
([ORDAC](https://arxiv.org/abs/2509.02351), [Self-Relaxed Joint Training](https://arxiv.org/abs/2410.21885)) and
[early-learning regularisation](https://arxiv.org/abs/2007.00151) report large gains on synthetic noise and weak ones
on real noise (NeurIPS reviewers' reading of ELR). The trainer already uses soft ordinal targets and a ranking loss.
[DistortBench](https://arxiv.org/abs/2604.19966) shows humans drop from 83.6 % (type) to 69.9 % (severity), which is
the same shape as §2.3. Conclusion: cross-fitted soft labels (T3) are the one variant worth a controlled test;
further loss engineering is not.

**Subgroup shortcut (for FM4).** [Idrissi et al.](https://arxiv.org/abs/2110.14503): plain group balancing by
re-sampling matches specialised methods for worst-group accuracy. Transfer: the minority group is "clean and
processed" (31 images); repeating it is free, and KVQ supplies more candidates (high-MOS processed frames).

**Robust fine-tuning.** [LP-FT](https://arxiv.org/abs/2202.10054) and model soups
([Wortsman et al.](https://arxiv.org/abs/2203.05482)) argue for keeping pretrained features and for weight
averaging from a shared initialisation. The trainer already freezes the lower two thirds of each backbone; soups
would need a shared head initialisation across folds, which the current seeds do not give. Parked (§6).

## 5. Proposals

Common protocol for every experiment: PE-Core-L, seed 2026 first, 8 folds, `--select fixed` (weights = mean of the
last two epochs, no validation-based choice), compared with the `fixed` control of the same seed. Decision metrics:
(a) `gate_r55.py` Doc2_Strict with paired bootstrap; (b) `gate_plus.py` RPS, NLL, adjacent-grade AUC, and the
flag rate on clean+processed images. A one-seed screen can **reject** (Doc2S Δ < 0 and no probability-level gain) or
**earn a second seed**. **Accept** needs two seeds and either Doc2S Δ ≥ +0.30 with P(Δ ≤ 0) ≤ 0.20, or RPS and
grade-AUC both better with P(no gain) ≤ 0.05 and Doc2S Δ ≥ 0; S_p0 and S_clean may not fall by more than 0.10.

### P1 — honest protocol (`fixed`) and the probability-level gate
- Failure mode: FM3. Evidence §2.4, §2.6.
- Mechanism: none on the model; it makes every later comparison valid and makes OOF describe the model that ships.
- Implementation: done. `train_testbed.py --select fixed`; `code/r55/gate_plus.py`; preset `fixed`.
- Control/ablation: `fixed` vs `ctrl` (same script, validation-chosen; 18-ab) quantifies the optimism.
- Success: `fixed` within 0.5 of `ctrl`. If it is more than 0.5 lower, the last-two-epoch rule is poor → test a
  6-epoch schedule (`--epochs 6`) as the fixed rule and re-baseline.
- Cost: 8 fold-jobs. Risk: none.

### P2 — random native crops in training, multi-crop averaging at inference
- Failure mode: FM2. Evidence §2.4.
- Mechanism: each epoch shows different native-pixel evidence, so the tower cannot memorise five views per image;
  averaging crop sets at inference reduces variance at the grade boundary.
- Implementation: done. `CropJitterStore` in `train_testbed.py` (`--crop-rand 0.5 --crop-jitter 96`, training only);
  preset `rcrop`; `code/r55/eval_tta.py` re-predicts each fold's validation rows with K random crop sets.
- Controls: `rcrop` vs `fixed`; TTA on a baseline tower (`qy_s2028`, running) and on `rcrop` towers; flip-rate of the
  argmax between two random passes as a label-free stability measure.
- Success / next action: accept rule above → second seed, then C-RADIO. If training gain is null but TTA gains
  ≥ +0.15 with K ≤ 4 → keep TTA only. If both null → closed; guided crops already carry the signal.
- Cost: 8 fold-jobs + inference. Risk: random crops may land on flat regions (mitigated by keeping 50 % guided).

### P3 — cross-fitted soft labels on the labelled data (`xd`)
- Failure mode: FM1 (noisy grades). Evidence §2.3; Phoenix stage 2.
- Mechanism: a teacher that never saw an image gives a smoother estimate of its grade distribution than one human
  label; training on label + soft label reduces fitting of annotation noise.
- Implementation: done. For each fold the training pool is split by video into halves A and B (fixed rule). Teachers
  are trained on A and on B (2 seeds each, `--half A|B --save-pool`); a row of A is labelled only by B-teachers and
  vice versa, so no teacher has seen the row or the validation fold. Student: all labels + `--lambda-kd-lab 1.0`.
- Controls: `xd` vs `fixed`. Known weakness: half-data teachers are weaker than the student.
- Success / next action: accept rule → second seed. If null → the half-data teacher is the suspect; one retry with
  7-way cross-fitting inside the pool (teachers on 6/7 of the pool; 56 fold-jobs) only if T2 shows teachers help.
- Cost: 32 half-jobs (shared with P4) + 8 fold-jobs.

### P4 — in-domain unlabelled frames (`ssl` test, then KVQ self-training)
- Failure mode: FM1. Evidence §2.2, old learning curve.
- Mechanism: more in-domain inputs with ensemble-quality soft labels regularise the grade boundaries.
- Step 1, value test with no external data (`ssl`): train on half A with labels and on half B *without* labels, using
  the A-teachers' soft labels (12 unlabelled images per step). Compare with `halfA` (labels only) and `fixed` (all
  labels). The fraction of the halfA→fixed gap that `ssl` recovers is the worth of one in-domain unlabelled frame
  relative to one label; `halfA` vs `fixed` is the learning-curve slope under the current protocol.
- Step 2 (only if step 1 recovers ≥ 25 % of the gap, or ΔRPS is significant): KVQ. `code/r55/ingest_kvq.py` (running)
  writes two native frames per video to `data/r55_aux/images/r55kvq_*`, de-duplicated against labelled and practice
  images, in a separate manifest. `prep_aux.py --manifest data/r55_aux/manifest_kvq.csv --suffix _kvq --tag r55kvq`
  adds physics meta and fold-teacher labels (teacher for fold k = fold-k models of six towers). Preset `kvq`.
- Controls: `kvq` vs `fixed`; `ext` (KonIQ + Inter4K, already prepared) as the out-of-domain contrast.
- Success / next action: accept rule → add to combo. If `ssl` recovers < 10 % → self-training is closed for every
  source and `kvq`/`ext` are not run.
- Cost: step 1 = 16 half-jobs + 8 jobs (≈1.6× time each); step 2 = ≈1 h of prep on 2 GPUs + 8 jobs.
- Risk: KVQ frames differ in processing mix from the labelled set; soft labels limit the damage.

### P5 — processed-clean robustness (`boost`, `kvqclean`)
- Failure mode: FM4. Evidence §2.5.
- Mechanism: the model needs counter-examples to "processed ⇒ defective".
- Implementation: (a) `--boost-proc-clean 8` repeats the ≈27 clean+processed training rows of each fold (the tag
  selects rows to repeat; it is never an input) — done, preset `boost`. (b) KVQ frames with MOS in the top decile
  that also pass the fold-teacher certificate become low-weight clean negatives (`certify_clean.py` logic on the KVQ
  teacher file, written to `data/r55_aux/kvq_clean.csv`; preset `kvqclean`, loss weight 0.1) — to do after KVQ prep.
- Metric: flag rate on the 31 clean+processed OOF images (each image = 3.2 points, so only a drop of ≥ 10 points is
  meaningful), plus the flag rate on held-out high-MOS KVQ frames as a label-free second reading; S_p0 and grade AUC
  must not fall, and the 纹理损失 P2 recall is watched because "slight texture loss" is the nearest label.
- Success / next action: flag rate on clean+processed falls by ≥ 10 points with Doc2S Δ ≥ −0.10 → keep in the combo
  even without a score gain (it is a generalisation fix). If S_pos falls by > 0.2 → reduce the repeat factor to 4 once.
- Cost: 8 fold-jobs each.

### P6 — shift test (reporting, not selection)
- Question: does OOF overstate performance on unseen processing pipelines?
- Implementation: build `data/r55_tagsplit/` where folds are whole pipeline-tag families (v8L3HiV6, v8SlHiV4,
  v7L3HiV2, all other tags, untagged-1…4), train `fixed`, compare per-fold scores and RPS with the video-grouped
  folds. To do; 8 fold-jobs. Outcome is reported next to every headline number; it does not gate experiments.

### P7 — deliverable (18-ab leads; recipe input from here)
- `s1` full-fit is being run by 18-ab as the safety net. If P2–P5 produce an accepted recipe, the shipped model is the
  fold models themselves (shared frozen trunk, 0.13–0.19B trainable per fold, ≈8–9B for 3 backbones × 16 folds/seeds),
  which is exactly the model the OOF numbers describe.

## 6. Considered and not pursued

| Direction | Reason |
|---|---|
| More seeds/towers, bigger backbones | saturation in §2.2; PE-Core-G gave +0.3–0.5 in 9/10 old runs and one collapse; no new information |
| Synthetic degradation ladders, tag head | measured nulls (`rank`, `og`, `tag`); QualiCLIP-style pretraining is the same mechanism |
| VLM prompting / RL fine-tuning (Q-Insight) | organisers' own 9B trials reached 57–59 % accuracy; DistortBench best 61.9 %; Phoenix pillars add +0.36 to the stack at high cost |
| Relabelling training data ourselves | the score is agreement with the organisers' raters; our rubric would differ |
| Weight soups | need shared head initialisation; revisit only for cost, not quality |
| Dense tiling of the whole frame | not rejected: run it if multi-crop TTA (P2) improves 过度生成 or 虚化突兀 specifically, which would show that local coverage is the missing evidence |

## 7. Phased plan

GPU budget: 6 GPUs for this line (0–5; 6–7 belong to 18-ab), two fold-jobs per GPU, ≈50 min per full fold-job,
≈28 min per half-job. Disk: ≈5 GB per tower run; stop below 80 GB free.

| Phase | Task | Command | Depends on | Deliverable | Decision |
|---|---|---|---|---|---|
| 1a | `fixed`, `rcrop` (seed 2026) | queue: `fixed:2026:0-7 rcrop:2026:0-7` | — | two tower dirs | D1: P1 success rule; D2: P2 rule |
| 1b | TTA on a baseline tower | `python code/r55/eval_tta.py --tower-dir results/next55/towers/qy_s2028/pe_core_large --k 8` then `--score` | — | `tta_r50j96.npz` | D2 |
| 1c | half-pool teachers | queue: `halfA:2026 halfA:2027 halfB:2026 halfB:2027` | — | 4 tower dirs with pool predictions | learning-curve point |
| 1d | `ssl`, `xd` | queue: `ssl:2026:0-7 xd:2026:0-7` | 1c | two tower dirs | D3: P3 rule; D4: P4 step-1 rule |
| 1e | KVQ ingest | `python code/r55/ingest_kvq.py` | download (done) | `manifest_kvq.csv`, frames | — |
| 2a | KVQ prep | `CUDA_VISIBLE_DEVICES=0,1 python code/r55/prep_aux.py --gpus 2 --manifest data/r55_aux/manifest_kvq.csv --suffix _kvq --tag r55kvq` | 1e, GPUs 0–1 free | meta + teacher files | — |
| 2b | `boost` | queue: `boost:2026:0-7` | 1a | tower dir | D5: P5 rule |
| 2c | `kvq` (if D4 positive), `kvqclean` | queue | 2a, D4 | tower dirs | accept rule |
| 2d | shift test | build tag-grouped split, `fixed` on it | 1a | report | none |
| 3a | second seed (2027) for every screen that earned it | queue | D1–D5 | | accept / reject |
| 3b | combo of accepted flags, 2 seeds | `R55_COMBO_FLAGS=… run_r55.sh exp combo` | 3a | | combo must beat the best single by ≥ +0.10 |
| 3c | propagate to C-RADIOv4-H (then SigLIP2) | `BACKBONE=cradio_v4_h …` | 3b | | same accept rule per backbone |
| 4 | stack gate vs `s1`, fold-bagged engine, timing, final-set inference twice | 18-ab + this line | 3c | shippable model | accept `r1` if Δ ≥ +0.15, P ≤ 0.20 |

Queue command in use (resumable, lock files in `logs/r55/locks`):
`python3 -u code/r55/queue_run.py --gpus 2 3 4 5 --jobs fixed:2026:2-7 rcrop:2026:2-7 halfA:2026:0-7 halfA:2027:0-7 halfB:2026:0-7 halfB:2027:0-7 ssl:2026:0-7 xd:2026:0-7`

Stop rules: one documented retry per idea, never edit gate thresholds or leak filters to pass, nothing selected on
`test150`, a recipe that passes on PE-Core but fails on C-RADIO ships on PE-Core only.

## 8. Evaluation protocol

- Primary: nested OOF `Doc2_Strict` (decoder fitted on the other folds), paired bootstrap.
- Secondary (lower variance): RPS, NLL, adjacent-grade AUC, detection AUC; cluster bootstrap by video.
- Subgroups reported with every accepted result: first batch / supplement; clean+processed flag rate; per-dimension
  exact-grade accuracy on positives; images with ≥ 3 positive dimensions (hard cases: exact accuracy 0.43–0.47 on
  模糊/过锐/过度生成 when ≥ 4 dimensions are positive).
- Robustness: argmax flip rate between random crop sets; tag-grouped folds (P6); two seeds for every accept.
- Not used: `test150`, any test template, legacy submission JSONs, file names as model input.

## 9. Results log

| Time (UTC) | Item | Result | Decision |
|---|---|---|---|
| 11-11 15:45 | `og` screen | 49.20 vs 49.60 (−0.40, CI90 [−0.86, +0.07]); gate_plus: ΔRPS +0.0001, ΔAUC-grade +0.0004 | null; closed |
| 11-11 16:15 | Phoenix cv5 + stage 2 | pillars 48.9 / 49.3 / 49.8; `s1` + 3 pillars 51.39 (18-ab) | stack candidate for 18-ab's gate |
| 11-11 16:50 | same-script control `ctrl` (18-ab) | 49.67 vs 49.60 original trainer (+0.08, CI90 [−0.30, +0.49]); NLL +0.017 (testbed slightly worse calibrated); clean+processed flagged 55 % (n=31) | `r55_ctrl_s2026` is the seed-2026 reference. Against it: `rank25` −0.98, `tag` −1.28, `og` −0.47 [−0.96, +0.04] |
| 11-11 17:20 | P2 inference: crop averaging on a baseline tower (`qy_s2028`, 8 random crop sets) | Doc2S 49.80 → 49.68 (×4) / 50.02 (×8), no trend; RPS −0.0008 (×4) / −0.0009 (×8), CI90 excludes 0; NLL 0.558 → 0.534; grade AUC +0.001; 过度生成 grade AUC unchanged (0.72 / 0.70); argmax flip rate between two crop sets 3–5 % | Crop averaging only improves calibration, which an ensemble already provides: not worth 4–8× inference on this tower. Re-test on `rcrop` towers. Crop coverage is not the missing evidence for 过度生成, so dense tiling stays parked |
| 11-11 16:55 | KVQ pool | 5,657 native frames from 2,888 videos in `data/r55_aux/manifest_kvq.csv`; 583 frames from videos with MOS ≥ 3.9; meta in `results/r55/dual_engine_meta_aux_kvq.npz` | teacher labels pending (GPUs 0–1) |
| 11-11 18:40 | P1 `fixed` (last epoch, no validation-based checkpoint choice) vs `ctrl`, seed 2026, 8 folds | Doc2S 49.81 vs 49.67 (+0.14, CI90 [−0.30, +0.56]); argmax 48.74 vs 48.94; RPS +0.0006, NLL +0.045 (last epoch is more over-confident), grade AUC −0.0003 | Choosing the checkpoint on the validation fold does **not** inflate the OOF score; reported OOF numbers stand. `fixed` is the base of every recipe experiment and is valid for full-fit (no selection needed) |
| 11-11 18:40 | P2 `rcrop` (random native crops, p=0.5, jitter 96) vs `fixed`, seed 2026 | single tower: argmax 49.20 vs 48.74 (+0.46), nested Doc2S 49.27 vs 49.81 (−0.54, CI90 [−0.98, −0.12]); RPS −0.0008 (P=0.02), NLL −0.032, detection AUC +0.0016 (P=0.01), grade AUC +0.0014 (P=0.24); clean+processed flagged 48 % vs 58 % (n=31). Inside `s1` (swap for `qy_pe_core_large`): RPS 0.04300 vs 0.04303 (`fixed`) vs 0.04300 (`ctrl`) | **Rejected, no second seed.** A small single-tower calibration/detection gain that disappears in the ensemble. The opposite signs of argmax and nested Doc2S are decoder noise (next row) |
| 11-11 18:50 | Noise of the nested-decoder score at stack level | swapping one of eight `s1` members for a same-recipe rerun (`ctrl`): 51.04 → 51.36 (+0.32) with RPS unchanged (0.04298 → 0.04300). Decoder gain over argmax: −0.23 … +1.07 across towers and stacks of equal probability quality | Stack decisions below ≈0.4 Doc2S are not evidence. Stacks are judged on RPS / NLL / grade AUC (`code/r55/stack_plus.py`); sent to 18-ab |
| 11-11 19:00 | Stack composition on probabilities (`stack_plus.py --fit --decode`, Phoenix stage-2 OOF) | `s1` RPS 0.04298, argmax 50.68. + 3 Phoenix pillars, equal weights: 0.04203 (−0.00096, CI90 [−0.00112, −0.00080]), NLL 0.379 → 0.363, grade AUC 0.8346 → 0.8368, argmax 51.13; same without the 100 fold-straddling images. Leave-one-out in `s1`: dropping a SigLIP2 tower *improves* RPS (−0.0002), dropping a V16 costs +0.00065. **Family-balanced weights** (tier-1 / V16 / Phoenix each sum to 1, nothing fitted): 0.04172 (−0.00125 vs `s1`), argmax 51.37. **One seed each + 3 pillars, balanced (7 models): 0.04186, argmax 51.33**; 6 models (no pillar 2) 0.04192; 5 models (pillar 3 only) 0.04220. Nested fit of family temperatures + weights adds nothing (ΔRPS 0.00000, T = 1.01–1.07) | Three effects are each far outside their CI: Phoenix adds real information; equal weight per model over-weights the weakest family; second seeds are redundant (+0.00014). Proposed deliverable to 18-ab: one seed per model, family-balanced weights, which also brings all three pillars under 30B. Ordering among balanced variants is within noise; compared post hoc |
| 11-11 18:44 | Phoenix pipeline (PID 3277617) | exited after writing `checkpoints/phoenix/pillar{1,2,3}_fulldata_swa.pt`; step 5 (single-4090 inference) wrote no output file | not rerun by this session; reported to the owner |
| 11-11 19:05 | Learning-curve point: `halfA` (half of each training pool) vs `fixed`, seed 2026 | argmax 46.69 vs 48.74 (−2.05), nested Doc2S 47.74 vs 49.81; RPS +0.0036 (CI90 [+0.0025, +0.0044]); grade AUC −0.013; detection AUC −0.006; other-clean images flagged 5.8 % vs 2.7 % | The model is still **data-limited**: the last doubling of labelled data was worth about 2 points. Second seed running; `ssl` measures how much of this gap unlabelled in-domain frames recover |
| 11-11 19:30 | Phoenix single-4090 inference (`deploy_single_4090_engine.py`) and pillar-1 fidelity (`code/r55/phoenix_deploy_fidelity.py`, 120 unseen KVQ frames, reference = bf16 + DoRA as trained) | Step 5 of the pipeline crashed: `bitsandbytes` missing. With it, the checkpoint does not load (8 trained merger tensors vs 4-bit layers). Forced to load, the engine's configuration (4-bit at load + plain LoRA, 368 DoRA magnitudes dropped) changes the grade of **73 %** of images (mean TV 0.160). Correct order — merge adapters, then quantise the language model in place: 5.6 GB, mean TV 0.0087, 11.7 % of images change a grade (bf16 merge alone: 3.3 %; DoRA dropped in bf16: 13.3 %) | The Phoenix engine has never run with real backbones; its latency/VRAM figures are unverified. Pillars 1–2 are shippable only through `code/r55/phoenix_pillar_4090.py` (merge → 4-bit). Pillar 2 is not worth its ≈5.6 GB (6 models 0.04192 vs 7 models 0.04186). Recommended to 18-ab: radio + pe + sig (one seed) + one V16 + p1 (merged 4-bit) + p3, family-balanced (≈14.6B, argmax 51.28, RPS 0.04192); fallback without p1 (≈5.9B, 51.10, 0.04220). `bitsandbytes` installed only in a scratch directory, not in the shared environment |
| 11-11 19:40 | Learning curve, second seed; scale of the quantisation error | `halfA` seed 2027: argmax 46.96 (seed 2026: 46.69) vs `fixed` 48.74; two-seed `halfA` vs `fixed`: RPS +0.0017 (CI90 [+0.0008, +0.0024]), grade AUC −0.009. Two seeds of one recipe differ by mean TV 0.06–0.08 and change a grade on 30–40 % of images (pe, cradio, V16 pairs); merge-then-4-bit differs from its bf16 original by TV 0.009 | Data-limited conclusion confirmed on two seeds (−1.9 argmax for half the data). Correct 4-bit quantisation is about 8× smaller than seed-to-seed variation. `code/r55/phoenix_pillar_4090.py` self-check reproduces the measured outputs exactly (5.7 GB, 1.7 s per image for pillar 1 alone on a shared H200) |
| 11-11 20:15 | P2 inference on the `rcrop` tower: 8 random crop sets (`eval_tta.py`) | guided crops 49.27; random ×1 / ×2 / ×4 / ×8: 49.92 / 49.87 / 49.73 / 49.88; guided + random: 49.7–50.0; flip rate between two crop sets 2–3 % | A model trained on random crops prefers random crops at test time (+0.5), which only brings it back to the control (`fixed` 49.81). P2 closed: neither random-crop training nor crop averaging is adopted |
| 11-11 20:10 | P6 shift test, first two families (`eval_shift.py`, same images, `shift` = family never seen in training vs `fixed` = other videos of the family seen) | tag v8L3HiV6 (n=517): RPS 0.0720 → 0.0798 (+11 %), grade AUC 0.792 → 0.761; tag v8SlHiV4 (n=509): RPS 0.0420 → 0.0512 (+22 %), grade AUC 0.853 → 0.838; pooled ΔRPS +0.0085, CI90 [+0.0065, +0.0105] | An unseen processing pipeline costs more than halving the training data (ΔRPS +0.0036). Remaining five families running; reporting only |
| 11-11 20:45 | P7 deliverable engine (`code/r55/final_engine.py`): one process, one GPU, batch 1, every model resident, all preprocessing on the fly, 2× flip per member, family-balanced pooling, decoder fitted on pooled OOF. 40 labelled images, shared H200 (two training jobs on the same GPU) | **primary** (cradio + pe + siglip2 + v16_full8 + p1 merged 4-bit + p3): 19.3 GB resident, 20.4 GB peak, 5.5 s mean / 9.7 s p95 / 10.4 s max per image; OOF argmax 51.28, nested decoder 51.45. **fallback** (no p1): 13.6 GB resident, 14.8 GB peak, 3.7 s mean / 6.6 s p95; OOF 51.10 / 51.35. Storing Tier-1 and V16 backbones in bf16 is required (fp32: 27.6 GB resident, 33.7 GB peak) and changes no grade (TV ≤ 0.0013). Every full-fit member reloads correctly (in-sample grade accuracy 0.89–1.00 vs 0.83–0.87 out of fold; pooled 0.971 vs 0.850) | Both sets fit a 24 GB card and the 15 s budget on this hardware; latency on a real 4090 is not measured. Decoders in `results/r55/final/`. `bitsandbytes` is vendored in `third_party/pydeps` (git-excluded), not installed in the shared environment |
| 11-11 21:25 | P4 `ssl`: labelled half A + the other half with labels hidden and soft labels from two half-A teachers (no leak), seed 2026, 8 folds | argmax `halfA` 46.69 → `ssl` 47.34 → `fixed` (all labels) 48.74; RPS 0.05460 → 0.05365 → 0.05101; grade AUC 0.7982 → 0.8016 → 0.8118; detection AUC 0.9479 → 0.9500 → 0.9541. `ssl` vs `halfA`: ΔRPS −0.0010 (CI90 [−0.0016, −0.0003], P(no gain) = 0.01), ΔNLL −0.010, Δgrade AUC +0.0034 (P = 0.04), Δdetection AUC +0.0020 | Pseudo-labelled in-domain frames recover **25–32 %** of the value of real labels on the same images (pre-registered bar: ≥25 %). KVQ self-training (`kvq`, `kvqcert`) launched on GPUs 2–5 |
| 11-11 21:20 | KVQ teacher labels (6 Tier-1 towers × 8 fold models, 5,657 frames) and an independent check against KVQ's human MOS | Spearman between the teachers' predicted severity (max over dims of the expected grade) and video MOS: **−0.70**; P(all clean) vs MOS: +0.69; per dim 模糊 −0.62, 纹理损失 −0.70, 花屏 −0.46. Share of frames called clean rises with MOS: 0 % (MOS < 2.5), 1 %, 12 %, 43 % (MOS ≥ 3.9). 583 high-MOS candidates, of which 144–203 per fold pass that fold's teacher certificate (P(clean) ≥ 0.85 in all dims) | The grades track human quality on a public set the models never saw — an external, label-leak-free generalisation benchmark. The final engine is being scored on 800 KVQ videos the same way (`results/r55/kvq_val/`) |
| 11-11 21:35 | P6 shift test complete (`eval_shift.py`, 7 source families, pe_core_large seed 2026; every image predicted by a model that never saw its family vs the usual video-grouped OOF of the same recipe) | **All images: argmax Doc2S 43.87 vs 48.74 (−4.9); RPS 0.0510 → 0.0657; clean images flagged 6.0 % → 12.1 %.** Three tagged pipelines (n=1,664): ΔRPS +0.0065 (+10 %, CI90 [+0.0049, +0.0080]), grade AUC 0.802 → 0.785, detection AUC 0.922 → 0.904. Supplement batch (n=280): no change. "Other untagged" (n=488): RPS 0.061 → 0.138 — this family holds 209 of the 211 花屏 positives, so holding it out removes the class (花屏 detection AUC 0.995 → 0.885); label shift, not pipeline shift. Control (one family split by video, half of it seen instead of 7/8): ΔRPS +0.0033, clean flagged 2.6 % → 9 % | The OOF score describes sources the model has seen. On an unseen pipeline expect roughly a 10 % worse RPS (about 1.8× the cost of halving the labels); 花屏 on any source other than the one it was collected from is unverified. Follow-up running: `shiftkvq` (KVQ self-training with shift-fold teachers) and `shiftrc` |
| 11-11 22:10 | P3 `xd`: all labels + cross-fitted soft labels (a row of half A is soft-labelled only by teachers trained on half B and vice versa), seed 2026, vs `fixed` | single tower: RPS 0.0510 → 0.0468 (−8 %, CI90 [−0.0048, −0.0036]), NLL 0.583 → 0.429, detection AUC +0.0041, grade AUC +0.0032 (P = 0.04), argmax 48.62 vs 48.74. A 0.33B tower now matches the 3.3B V16 on RPS (0.0473). Inside the recommended six-model ensemble (pe → xd-pe): RPS 0.04192 → 0.04201 (CI90 [+0.00002, +0.00015], slightly worse); inside `s1`: 0.04300 vs 0.04303 | **Not adopted for the ensemble** (the soft-label member is better alone but less complementary), not extended to the other backbones. Keep as the recipe if a single small model must be shipped. It also shows that fitting noisy hard labels is what ruins single-tower calibration, not what limits accuracy |
| 11-11 22:20 | External check of the final engine: 800 KVQ videos (one frame each), Spearman between predicted severity and human MOS (`kvq_srcc.py`; statistic fixed beforehand, nothing trained or tuned on KVQ scores) | engine pooled (primary) **0.741** (CI90 [0.712, 0.768]); without p1 0.737 (difference +0.004, CI90 [−0.000, +0.008]); single full-fit members: V16 0.743, p1 0.711, p3 0.699, cradio / pe / siglip2 0.696–0.697; 48 Tier-1 fold models together 0.726 | On data from another source with independent human labels, the ensemble is clearly better than any Tier-1 tower (+0.045) and than the Tier-1 teachers; V16 alone is as good as the pool on this measure; p1 adds a small, borderline amount, consistent with OOF. MOS is not the competition label, so this ranks models, it does not estimate the score |
| 11-11 22:55 | P5 `boost`: the ≈29 clean+processed training rows of each fold repeated ×8, seed 2026, vs `fixed` | clean+processed flagged 58 % → 52 % (2 of 31 images), other clean 2.7 % → 3.3 %; RPS +0.0004 (n.s.), NLL +0.009, grade AUC −0.0040 (CI90 [−0.0071, −0.0009]); argmax 48.68 vs 48.74 | **Rejected.** Thirty examples repeated do not teach "processed but clean"; grading gets slightly worse. The subgroup needs new examples (the `kvqcert` test), not reweighting |
| 11-11 23:30 | P4 `kvq`: self-training on 5,657 KVQ frames with the 6-tower fold teachers (8 aux frames per step), seed 2026, vs `fixed`, standard folds | argmax 48.76 vs 48.74; RPS −0.0002 (CI90 [−0.0009, +0.0005]); NLL −0.015 (CI excludes 0); grade AUC +0.0035 (P = 0.03; 过度生成 P2\|P1 0.704 → 0.721, 过锐 P2\|P1 0.800 → 0.810); detection AUC +0.0004; clean+processed flagged 58 % → 52 % | Small gain in grading skill and calibration, none in RPS or score: **does not meet the acceptance rule in distribution**. KVQ frames are a different source from the labelled set, so the in-distribution folds are the wrong place to see their value; the decision rests on `shiftkvq` |
| 11-12 00:30 | P4/P6 `shiftkvq` (shift folds + KVQ self-training, teacher of fold f = the shift model of fold f) and `shiftrc` (shift folds + random crops), vs plain `shift` | Three tagged pipelines: ΔRPS to the seen-family reference +0.0065 (`shift`), +0.0070 (`shiftkvq`), +0.0062 (`shiftrc`); `shiftkvq` − `shift` +0.0005 (CI90 [−0.0006, +0.0016]). All images: `shiftkvq` − `shift` +0.0033 (worse, CI excludes 0), because the family that carries 花屏 gets worse still (RPS 0.138 → 0.161: the student copies a teacher that never saw the class). `shiftrc`: all-image RPS 0.0643 vs 0.0657, detection AUC 0.906 vs 0.878, argmax 44.47 vs 43.87; no change on the tagged pipelines | **KVQ self-training rejected**: no gain in distribution beyond calibration, none on unseen pipelines, harmful where the teacher is blind. Pseudo-labels cannot add what the teacher does not know. Random crops help a little under shift but not on unseen pipelines; not adopted. The unseen-pipeline gap stays open and needs labelled examples from more pipelines |
| 11-12 00:30 | Why earlier practice-set uploads scored 52.89 / 53.03 (investigation for the owner: `build_phase1_breakthrough_submission.py`, `SUBMISSION_LEDGER.md`, earlier session transcript, file diffs) | The 21-model weighted average scored 52.43; 52.89 is that file with 47 of 900 labels changed (second ensemble matched to the first file's test-set distribution, hand-set thresholds, the first file's 15 clean images locked); 53.03 is 23 labels from 52.89. ≥10 scored uploads, keep-the-best with rollback (50.26, 52.70, 52.1, 51.80 were discarded). Strict held-out score of that model family: 50.46 single, ≈51 ensembles. Simulation on OOF: best of 9 similar submissions on 140 images reads +1.0–1.5 above their real level. Clean component: 20.00 online on every upload vs 19.4 offline (+0.6, a property of the practice set). Legacy files call 344–350 positive labels; calibrated probabilities expect ≈306; the nested decoder gives 296 | No label leakage (0 duplicate images), but adaptive selection on the practice set: about 51 real + 0.6 easy clean images + ≈1.2 selection. Not a better ensemble method. Expected practice score of an untuned fresh ensemble is therefore ≈52, not 53 |
| 11-12 00:35 | P5 `kvqcert`: `kvq` + clean negatives = high-MOS KVQ frames that this fold's own teacher also calls clean (≈176 of 583 candidates per fold), vs `fixed` | clean+processed flagged 58 % → 45 % (4 of 31 images), other clean 2.7 % → 2.5 %; argmax 49.09 vs 48.74; RPS −0.0002 (n.s.); NLL −0.014; grade AUC +0.0009 (n.s.) | Right direction on the processed-clean weakness and no harm, but 31 images cannot carry a decision and the acceptance rule is not met: **not adopted, no second seed**. It is the only idea that moved that subgroup; worth repeating once there are more labelled processed-clean frames to measure it on |

## 10. Outcome of this round (2026-11-12 00:40 UTC)

**Recommended deliverable** (`code/r55/final_engine.py`, decoders in `results/r55/final/`):
cradio_v4_h + pe_core_large + siglip2 (one seed, full-fit) + v16_full8 + Phoenix pillar 1 (adapters merged, language model
4-bit via `phoenix_pillar_4090.py`) + pillar 3, family-balanced mean, Decoder55 fitted on pooled OOF.
OOF argmax 51.28 / nested decoder 51.45 (`s1`: 50.68 / 51.04), RPS 0.04192 (`s1` 0.04298); ≈14.9B parameters;
19.3 GB resident, 20.4 GB peak, 5.5 s per image at batch 1 on a shared H200 (bf16 backbones required).
Fallback without pillar 1: 51.10 / 51.35, ≈6.2B, 13.6 GB, 3.7 s. External check on 800 unseen KVQ videos: Spearman 0.741
with human MOS (single Tier-1 towers 0.70). Local practice-set predictions exist for both sets; nothing was uploaded.

| Item | Verdict | One-line reason |
|---|---|---|
| Cross-family ensemble (add Phoenix) | **adopted** | RPS −0.0010 vs `s1`, CI excludes 0, holds without fold-straddling videos |
| Family-balanced weights, one seed per model | **adopted** | −0.0003 on top; second seeds worth 0.00014; fitted weights/temperatures add nothing |
| bf16 backbones, merge-then-4-bit pillar 1 | **adopted** (deployment) | fits 24 GB; no grade changes (bf16) / TV 0.009 (4-bit), ≈8× below seed-to-seed variation |
| `fixed` protocol (no validation-based checkpoint choice) | adopted as base | equals the control; OOF numbers were not inflated |
| Probability-level gates (`gate_plus.py`, `stack_plus.py`) | adopted | nested-decoder Doc2S moves ±0.3 between reruns of one recipe |
| Random-crop training, crop averaging | rejected | single-tower calibration only; nothing in the ensemble |
| Cross-fitted soft labels (`xd`) | not adopted for the ensemble | single tower RPS −8 %, ensemble unchanged; keep for a single-model deployment |
| KVQ self-training (`kvq`, `shiftkvq`) | rejected | no RPS gain in distribution, none on unseen pipelines, worse where the teacher lacks a class |
| Clean boosting (`boost`) | rejected | 30 repeated rows do not teach the subgroup |
| Certified-clean KVQ negatives (`kvqcert`) | not adopted | right direction (58 % → 45 % on 31 images), not conclusive |
| Ranking ladders, tag head, generated 过度生成 ladder | rejected (earlier) | see rows above |

**What limits the system now**
1. Labelled data: half the labels cost ≈2 points; pseudo-labels recover only 25–32 % of real labels in distribution.
2. Unseen processing pipelines: RPS ≈10 % worse than the OOF figure suggests; clean images flagged twice as often.
3. 花屏: 209 of 211 positives come from one source, so the 0.98 F1 says nothing about other sources.
4. Processed-but-clean frames: flagged 32 % of the time by the ensemble; only 31 labelled examples exist.

**Next steps, in order of expected value**
1. Label more frames, chosen for coverage rather than volume: (a) frames from processing pipelines that are absent or
   rare in the training set, (b) processed-but-clean frames, (c) 花屏 from other sources. A few hundred targeted
   labels address limits 2–4 directly; KVQ frames with high teacher disagreement are ready-made candidates.
2. Ship the primary set if a real 4090 confirms latency and memory, otherwise the fallback. Install `bitsandbytes`
   in the delivery environment (or keep `third_party/pydeps`).
3. Do not spend further GPU time on single-tower recipe changes: five independent ones were null inside the ensemble.
4. Treat the practice-set score as a noisy check (sd ≈1.6 on 140 images). An untuned fresh ensemble should land near
   52; the earlier 52.89 / 53.03 contained ≈1.2 points of selection on that set.

### Addendum 2026-11-12 — the `s1` reference upload (owner's result)

`results/final/pred_nriqa_dataset_test_next55_s1.json` scored **51.23** online: positive defects 15.08 / 25, P0 16.15 / 20,
clean 20.00 / 20.

| Component | Online | Offline, all OOF | Offline, random 140-image subsets (mean ± sd) | Reading |
|---|---|---|---|---|
| Total | 51.23 | 51.04 | 50.64 ± 1.83 | 60th percentile of the offline subsets: no systematic online offset |
| Positive defects | 15.08 | 15.57 | 15.34 ± 0.81 | consistent |
| P0 | 16.15 | 16.10 | 16.02 ± 0.87 | consistent |
| Clean | 20.00 | 19.38 | 19.28 ± 0.99 | the practice set's clean images are easy (20.00 on every upload so far) |

1. The offline protocol predicts the online score of an untuned model. The earlier belief in a "+2 online offset" is refuted.
2. 52.89 − 51.23 = 1.66, of which +1.43 is the positive-defect component. Offline, calling more defect labels raises
   that component and lowers the clean component by about the same amount (argmax decoding of the primary pool:
   1.82 → 2.05 labels per image gives S_pos 15.67 → 16.02, S_clean 19.61 → 19.03, total 51.28 → 51.19). On the practice
   set the clean component stays at 20.00 whatever is submitted, so over-calling is free there; the earlier files call
   2.29–2.33 labels per image against 1.95 for `s1`. That accounts for roughly +0.4 of the +1.4; the remainder is the
   selection effect estimated in §9.
3. Expected practice-set score of the new ensembles: the primary set beats `s1` by +0.48 ± 1.14 on the same 140 images
   offline, but +0.27 of that is the clean component, which cannot rise above 20.00 there. Realistic expectation
   ≈51.5–51.7 (10–90 % range about 50.4–53.2); it would read below `s1` in about one upload out of three.
4. Not done, and not recommended without the owner's decision: moving the decoder to a more defect-prone operating
   point because the practice set does not punish it. Offline it is neutral; it is a statement about how costly false
   alarms on clean frames are in the target data, which only the practice-set feedback currently supports.
