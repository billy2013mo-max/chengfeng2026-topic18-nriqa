# SynthLab — synthetic-data pipeline with built-in evaluation (命题18)

Purpose: iterate on synthetic data without repeating the failures of the earlier rounds. A recipe is only as good as the
gates it passes; every gate compares against a control trained with the same code, folds and seed, on real labelled
images the model has not seen, and every result is appended to `results/r55/synthlab/LEDGER.md`.

## 1. What the evidence says synthetic data can and cannot do here

| Finding | Evidence | Consequence for recipes |
|---|---|---|
| The models already respond strongly and monotonically to blur, denoising and unsharp-mask ladders | Real-trained fold teachers on the earlier ladders (never saw the seeds): Gaussian blur level 1→3 P(模糊) 0.44 → 0.72 → 0.98; USM P(过锐) 0.48 → 0.61 → 0.80; NLM P(纹理损失) 0.82 → 0.93 | More data of these kinds teaches nothing (the ranking ladders failed for this reason). Do not build them again |
| The FLUX re-render ladder is not seen as 过度生成 by models that detect the real defect at AUC 0.96 | Teacher P(过度生成) 0.16 → 0.24 → 0.25 over its three levels; never called P0 | That ladder was a different phenomenon; labelling it 过度生成 taught the wrong thing (−0.40, P1 F1 −0.17). A recipe must pass the fidelity gate before it is trained on |
| A defect that is produced by a known process can be simulated exactly, and then it works | 花屏 from corrupted bitstreams: detection on a source family never seen in training 0.885 → 0.9999 AUC | Prefer physically exact simulation |
| Without matched negatives the model learns the generator, not the defect | 花屏 v1 (no negatives): 7 false alarms on blurry / screen-recording frames; v2 (each corrupted frame paired with the same window decoded cleanly): 1 | Matched negatives are mandatory |
| Grades cannot be assigned by construction | Annotators' P2/P1/P0 boundaries are not a function of operator strength; errors are 78–92 % one grade off | Recipes assert presence / absence only (`level 9` / `level 0`), or P0 for the two P0-only defects |
| Where the points are lost | P2 labels: mostly the clean ↔ P2 boundary (模糊 242 of 367 misses called clean; 过锐 256 of 349; 过度生成 124 of 198). Weakest separations: 过度生成 P2\|P1 0.74 and P1\|P0 0.74; clean\|P2 0.885–0.93 | The largest deficit is grading and slight-level detection of 过度生成; it needs a faithful generator of that defect |
| Clean frames with a pipeline tag are flagged far more often | 6 of 31 (19 %) vs 4 of 482 (0.8 %) | Invariance to benign processing is a legitimate target (`benign1`) |

## 2. Stages and commands

| Stage | Question | Command | Output |
|---|---|---|---|
| 0 Probe | Which sensitivities are missing? (controlled stimuli on held-out clean frames, no training) | `synth_probe.py --build`, `--run --fold f --gpu g`, `--report` | `results/r55/synthlab/probe/*.md` |
| 1 Build | Make the recipe's frames, labels by construction, fold bookkeeping, image metadata | `synth_build.py --recipe R` | `data/r55_aux/manifest_R[_shift].csv`, `results/r55/aux_teacher_R[_shift].npz`, `results/r55/dual_engine_meta_aux_R.npz` |
| 2 Fidelity | Do models trained on real data see the intended defect in the positives and not in the matched negatives? | fold teachers on the frames (`prep_aux.py --steps teachers merge --manifest … --suffix _R`) | response by level; a recipe whose positives are not recognised is a different phenomenon unless the real data lacks the case (unseen source) |
| 3 Screen | Does it help or hurt on real labels? two standard folds + the unseen-source folds | `queue_run.py --jobs synth-R:2026:0-1 shiftsynth-R:2026:0-2`, then `synth_eval.py --recipe R --targets … --stage screen` | PASS / NULL / HARM in the ledger |
| 4 Confirm | Same on all eight folds | `queue_run.py --jobs synth-R:2026:2-7`, `synth_eval.py --stage confirm` | ledger |
| 5 Pool | Does the ensemble improve when the recipe tower replaces `pe`? | `synth_eval.py --recipe R --stage pool` | ledger; adoption needs this gate |

Controls: `r55_fixed_s2026` (standard folds) and `r55_shift_s2026` (source-held-out folds): same trainer, same seed, no
synthetic rows. Statistics: per-defect RPS with a paired bootstrap over videos; a difference counts only if its 90 %
interval excludes zero. Clean-image false alarms are reported separately for tagged and untagged frames.

## 3. Rules for recipes (`code/r55/synth_recipes.py`)

1. Matched negatives: everything done to a positive except the defect itself is also done to its negative.
2. One file path and format for every variant; parameters drawn per frame from a range.
3. Labels by construction only: `level 0` absent, `level 9` present with unknown grade (the trainer then teaches
   presence: the loss is −log(1 − P(clean)) and says nothing about the grade), `level 3` P0 for 花屏 / 虚化突兀.
4. A defect whose label after the operation is not known is masked (`mask`), never guessed.
5. A synthetic frame inherits the fold of its seed; the trainer drops frames whose seed is in the validation fold (and,
   on the shift folds, in the held-out source family).
6. No model prediction enters the build. No test image is used anywhere.

## 4. Trainer support (default off)

`train_testbed.py --aux-kinds synth`: rows come from the recipe's manifest; `aux_teacher_R.npz` carries the
construction labels (`probs_folds`), the per-defect supervision mask (`mask`) and the presence-only flag (`det`).
The synthetic batch is a second forward/backward per step (`--aux-singles`, weight `--lambda-aux-kd`).

## 5. Ledger

`results/r55/synthlab/LEDGER.md` (append-only) holds every gate of every recipe; per-recipe JSON in
`results/r55/synthlab/<R>/`. Earlier synthetic runs that predate SynthLab are in `results/next55/LEDGER.md`
(`rank`, `rank25`, `og`, `glitch`, `glitch2`).

## 6. Safeguards added after the first iteration

- **Seed-in-training guard (trainer).** A synthetic frame is used only if its seed is a training image of the fold being
  trained, regardless of the fold column in the manifest. The first `benign1` screen on the source-held-out folds read the
  wrong manifest: processed copies of held-out images, with labels, were in training, and the unseen-source RPS "improved"
  by 0.024. The result was voided in the ledger and rerun. Any gain of that size should be treated as a leak until
  proven otherwise.
- **Trainer-side recipes.** Augmentations that change training frames in place (no extra rows) are recipes too:
  `baug` / `shiftbaug` (`--aug-benign`), evaluated by the same `synth_eval.py`.

## 7. Iteration log (details in `results/r55/synthlab/LEDGER.md`)

| Recipe | What it is | Fidelity | Screen (2 standard folds / 3 unseen-pipeline folds) | Status |
|---|---|---|---|---|
| `glitch2` (pre-SynthLab) | simulated 花屏 from corrupted bitstreams + matched clean decodes | real-trained models miss it on an unseen source (recall 0.36) | unseen source: AUC 0.885 → 0.9999; in distribution: no change; in the ensemble: total −0.14 (n.s.) | robustness option (`primary_v2`), not adopted for score |
| `benign1` (extra rows) | 1,600 labelled frames through a mild transcode / tone change, added as extra rows with their human labels | predictions already move little (TV 0.031; clean flagged 5.7 % → 7.0 %) | in distribution **HARM**: mean RPS +0.0021 (CI90 [+0.0006, +0.0037]), 过度生成 F1 0.46 → 0.39; unseen sources (clean rerun) **HARM**: +0.0049 (CI90 [+0.0035, +0.0065]) | closed: extra hard-labelled copies of training images hurt (as `boost` did) |
| `baug` | mild JPEG / tone change applied in place (p = 0.5), no extra rows | — | in distribution: mean −0.0001 (null), 模糊 −0.0040, 过度生成 +0.0027; unseen sources **HARM**: mean +0.0022 (CI90 [+0.0008, +0.0034]), 虚化突兀 +0.0084 | closed: re-quantisation is not label-preserving for 过度生成 / 虚化突兀 |
| `tw-benign1` | the processed twin (mild x264 / JPEG / tone) replaces its seed in place (p = 0.5) | as `benign1` | in distribution: mean −0.0007 (null), 虚化突兀 −0.0027; unseen sources **HARM**: mean +0.0055 (CI90 [+0.0041, +0.0070]), 虚化突兀 F1 0.79 → 0.62 | closed |
| blur / denoise / USM ladders | — | already recognised with certainty and in order by real-trained models | not built | closed by the fidelity gate |
| FLUX 过度生成 ladder (`og`) | — | not recognised (P 0.16–0.25) | earlier: −0.40 | closed; a faithful generator is the open need |

## 8. Behavioural probe, first run (`results/r55/synthlab/probe/tier1_baseline.md`)

240 clean held-out frames × 31 conditions, mean of the three Tier-1 towers, each frame scored only by fold models that never saw it.
Untouched frames: any defect called on 5.0 %.

| Finding | Numbers | What was done with it |
|---|---|---|
| Optical blur, motion blur, codec loss, unsharp masking and Laplacian sharpening are detected in order, with ≥ 95 % monotone ladders | Gaussian σ 0.7 / 1.2 / 2.0 / 3.5 → called 模糊 8 / 27 / 77 / 97 %; x264 crf 30 / 37 / 44 → 纹理损失 57 / 93 / 97 %; USM → 过锐 6 / 31 / 70 / 92 % | no recipe: nothing to teach |
| **Resolution loss is almost invisible to the models** | down-scale to 0.75 / 0.5 / 0.33 and bicubic up-scale → called 模糊 3 / 3 / 10 % | **real-data relevance check**: on the labelled frames, a fine-detail measure does not separate the slight-blur frames the ensemble misses from truly clean ones (AUC 0.57; texture loss 0.48). The blind spot is real but is not what annotators label, so teaching it would create false alarms. Not built |
| Edge-preserving smoothing is under-detected | bilateral (σ_color 100) → 纹理损失 45 %, NLM h = 16 → 75 %, against 93 % for x264 crf 37 | same relevance check negative (flat-area texture measure AUC 0.49–0.55). Not built |
| **A harmless transcode creates false alarms** | x264 crf 18: any defect called 9.6 % vs 5.0 % untouched (纹理损失 +0.042); JPEG 93, 0.9× resize and ±7 % tone: no effect | target of `tw-benign1` (the transcoded twin replaces the frame in place with p = 0.5) |

Stage added to the loop: **relevance check** — a blind spot found by the probe is only worth a recipe if, on real labelled
frames, a measurement of the same property separates the frames the ensemble gets wrong from the ones it gets right.

Label-preserving recipes are now used **in place** (`--aug-twins R`, presets `tw-R` / `shifttw-R`): the twin replaces its
seed with probability p, the row and its label are not repeated. Construction-label recipes (presence / absence of a
defect) stay extra rows (`synth-R`).

## 9. Training noise and the verdict rule (changed 2026-11-12 06:50 UTC)

The control was trained with three seeds (2026, 2027, 2028) on the screening folds (`synth_eval.py --stage aa`,
`results/r55/synthlab/noise_floor.json`).

| Scope | Mean RPS of the three control seeds | 过度生成 RPS / label F1 per seed | argmax Doc2S per seed |
|---|---|---|---|
| standard folds 0–1 (872 images) | 0.05085 / 0.05144 / 0.05130 | 0.0464 / 0.0490 / 0.0487 · 0.463 / 0.387 / 0.377 | 49.37 / 48.32 / 48.69 |
| source-held-out folds 0–2 (1,664 images) | 0.06905 / 0.06981 / 0.06978 | 0.0680 / 0.0698 / 0.0709 · 0.325 / 0.384 / 0.345 | 33.54 / 36.85 / 32.21 |

Seed 2026, the only control used at first, is a favourable draw (best of three on both sets of folds, by a wide margin
on 过度生成). Every recipe therefore looked harmful on 过度生成 against it. **Rule now implemented in `synth_eval.py`:**

1. The reference is the mean of all control seeds (`--ctrl-seeds`, default 2026 2027 2028); a recipe may pool several
   seeds (`--seeds`).
2. An effect counts only if the paired bootstrap interval over videos excludes zero **and** it exceeds twice the
   training-noise standard error, sd(control seeds) · sqrt(1/n_control + 1/n_recipe).
3. Harm on the source-held-out folds counts like harm in distribution.
4. The argmax score of a two- or three-fold screen moves by one to four points between identical trainings and is
   never used for a verdict.

## 10. Results against the three-seed control (valid verdicts)

| Recipe | In distribution: mean RPS (control 0.05119; 2×SE 0.0007) | Unseen pipelines: mean RPS (control 0.06954; 2×SE 0.0010) | 过度生成 RPS in distribution (control 0.0480) | Verdict |
|---|---|---|---|---|
| `benign1` extra rows | +0.0018 (CI90 [+0.0006, +0.0030]) | +0.0044 ([+0.0034, +0.0057]) | +0.0031 (within noise) | HARM |
| `baug` in place | −0.0005 (n.s.); 模糊 −0.0040 | +0.0017 ([+0.0006, +0.0027]); 虚化突兀 +0.0070 | +0.0011 (n.s.) | HARM (unseen pipelines) |
| `tw-benign1` in place | −0.0010 ([−0.0022, +0.0001]) | +0.0050 ([+0.0038, +0.0063]); 虚化突兀 +0.0220 | −0.0018 (n.s.) | HARM (unseen pipelines) |
| `ogsr1` restoration models | +0.0008 (n.s.); 过锐 +0.0035 | +0.0012 ([+0.0001, +0.0023]) | −0.00004 (CI90 [−0.0030, +0.0025]) | no gain on the target; closed |

Withdrawn: the earlier reading that mild re-compression specifically damages 过度生成 (F1 0.46 → 0.39–0.41). That drop was
the control seed's luck; against three control seeds 过度生成 does not move for any of these recipes. What does stand is
that all three label-preserving variants make the unseen-pipeline folds worse, mostly through 虚化突兀.

## 11. Recipe `ogsr1`: 过度生成 produced by real restoration models (`code/r55/build_overgen_sr.py`)

900 labelled frames without 过度生成 or 花屏 are down-scaled 2–4× (half of them also JPEG-compressed, quality 60–91) and
restored to their original size by one of four public GAN super-resolution models (Real-ESRGAN x4plus / x2plus, BSRGAN,
realesr-general-x4v3; weights in `data/r55_aux/raw/sr_weights`, loaded as tensors only, architectures from `spandrel` in
`third_party/pydeps_sr`, run with `python3 -I`). Positive = restored frame (过度生成 present, grade unknown); matched
negative = the same degraded frame brought back with bicubic interpolation (过度生成 absent). 模糊 / 纹理损失 / 过锐 are
masked on both; 花屏 and 虚化突兀 keep the seed's labels.

| Gate | Result |
|---|---|
| Visual check | restored frames show the painterly, invented-texture look of real 过度生成 where the content has texture; on smooth content (faces of animation, sky, fabric) nothing is invented and the frame equals its seed |
| Fidelity | real-trained models call 过度生成 on 23.7 % of the restored frames, 3.1 % of the bicubic twins, 4.0 % of the seeds. By model: x4plus 32 %, BSRGAN 26 %, compact 19 %, x2plus 16 %. JPEG before restoration 29 % vs 18 %. Restored frames also raise 过锐 (0.34 → 0.58) |
| Can construction tell which frames are recognised? | no: five pixel measures of invented detail (added high-frequency / Laplacian energy, difference to the twin) have AUC ≤ 0.57 |
| Screen | 过度生成 RPS −0.00004 in distribution, +0.0007 on unseen pipelines; detection AUC 0.938 → 0.946 (control seeds range 0.932–0.946) and 0.879 → 0.871; 过锐 +0.0035 in distribution |

Reading: the generator is a fair imitation of part of the labelled defect, and learning it changes nothing on real
labels. Detection of real 过度生成 is not what limits the score (ensemble detection AUC 0.96); the loss is at the grade
boundaries (P2 | P1 0.74, P1 | P0 0.74) and on slight cases, and a generator can assert presence but not an annotator's
grade. Real 过度生成 frames are also mostly weak sources (纹理损失 co-labelled on 88–96 %, 模糊 on 53–78 %), a context the
labelled set already covers.

## 12. E4: restoration models as an instrument (`code/r55/probe_sr_idem.py`)

For each labelled image and three restoration models: how much better the model reproduces the frame from its
down-scaled copy than bicubic interpolation does (error ratio, high-pass correlation and energy, spread over an 8 × 8
grid; 33 features). Tested with `probe_eval.py` against the pooled out-of-fold probabilities: pool + features is worse
than pool alone (mean RPS +0.0006; 过度生成 +0.0009, CI90 [+0.0002, +0.0017]); the features alone reach detection AUC
0.80 on 过度生成 and 0.62 / 0.58 on its grade boundaries. Closed.

## 13. Where the synthetic direction stands

| Target | Outcome | Open? |
|---|---|---|
| 花屏 on unseen sources | solved by exact simulation with matched negatives (AUC 0.885 → 0.9999); score-neutral in distribution | available as `primary_v2` |
| 模糊 / 纹理损失 / 过锐 | models already respond in order to every simulable degradation; the two blind spots found (resolution loss, edge-preserving smoothing) are not what annotators label | closed |
| Invariance to benign processing | three variants; none helps in distribution beyond noise, all hurt unseen pipelines | closed |
| 过度生成 | FLUX re-render (not the phenomenon), GAN restoration (partly the phenomenon, no gain), restoration-idempotence features (no information) | closed unless graded examples exist |
| 虚化突兀 | bokeh simulation unfaithful | closed |

What remains limiting is the number and consistency of human labels at the grade boundaries, which synthetic frames
cannot supply.

## 14. Recipe `bkseg1` — 虚化突兀 on labelled frames with a real subject mask (2026-11-12, after §13)

**Why this is not a repeat of §13.** §13 closed the recipes whose label is "present, grade unknown" on a graded
defect. 虚化突兀 has no grades (P0 / none), so a frame built by construction carries the complete label, as for 花屏.
New evidence (`docs/REASSESSMENT_AND_PLAN_2026-11-12.md` §10.5): the label is worth more than any other (it counts
in the positive and in the P0 score), 79 of its 225 positives come from one batch, and outside that batch the
ensemble finds 69 %. The labelled positives viewed are a sharp subject cut out against a blurred background.
The first simulation (`build_bokeh.py`) was stopped for fidelity: a depth split of KVQ frames rarely isolated a
subject and the blur removed the captions.

**Recipe** (`code/r55/build_bokeh_seg.py`, run with `python3 -I`): seeds are labelled frames without 虚化突兀 / 花屏
in which Mask R-CNN (COCO weights from the PyTorch model site, tensors only) finds a person or animal covering
6–60 % of the frame. Positive: background-only blur (Gaussian or lens kernel) composited through a hard mask
(exact, grown or shrunk by 2–8 px), subject sharpened in 40 % of the frames, captions kept. Matched negative: the
same blur on the whole frame, captions kept. 模糊 / 纹理损失 unsupervised on both, 过锐 unsupervised when the subject
was sharpened; other defects keep the seed's human labels.

**Decision rule, fixed before any training result:**
1. Fidelity: fold models that never saw the seed must call 虚化突兀 clearly more often on positives than on the
   negatives and seeds. (If positives are already called with certainty the recipe carries no information.)
2. Screen, two recipe seeds (2026, 2027) against the three control seeds, `synth_eval.py`: target defect 5.
   Pass = 虚化突兀 Brier lower with CI90 excluding zero and beyond the training-noise guard on the standard
   folds 0–1 **or** on shift fold 0 (the fold that holds out the family with 142 of the 225 positives), with no
   significant harm to mean RPS on either split and no rise in clean false alarms above 1.5 per 100.
3. Only then: remaining folds, pool gate (ensemble RPS), full-data model, decoder refit, engine check.
A null screen closes the recipe; no second variant is tuned on the screening folds.

**Result (2026-11-12 10:10 UTC) — closed.**
1. Fidelity (`logs/r55/synthlab/fidelity_bkseg1.log`): fold models that never saw the seed call 虚化突兀 on 12.1 % of
   the positives, 8.3 % of the matched negatives and 0.9 % of the seeds. Rule 1 is not met: whole-frame blur with
   sharp captions triggers the label almost as often as the cut-out subject. (Lens kernel 17 %, Gaussian 7 %; mask
   variant, subject sharpening and subject size make little difference.) The same models find 69 % of the real
   positives outside the easy batch, so what they learned from the labels is not "sharp subject, blurred
   background" as I simulated it.
2. Screen (`logs/r55/synthlab/eval_synth-bkseg1_screen.log`, per fold `eval_synth-bkseg1_perfold.log`), two recipe
   seeds against three control seeds:

   | Split | Positives | 虚化突兀 Brier, control → recipe | Difference (CI90) | Detection AUC |
   |---|---|---|---|---|
   | standard folds 0–1 | 56 | 0.01809 → 0.01763 | −0.00046 [−0.0043, +0.0032] | 0.9852 → 0.9842 |
   | shift fold 0 | 142 | 0.07967 → 0.08376 | +0.0041 [−0.0069, +0.0158] | 0.9512 → 0.9482 |
   | shift fold 2 | 67 | 0.03941 → 0.03555 | −0.0039 [−0.0091, +0.0017] | 0.9689 → 0.9735 |

   No pre-registered test passes. Recall at 0.5 rises (0.69 → 0.75 in distribution) with false alarms 2.3 → 6.0:
   a threshold shift, not better ranking. Other defects get worse: 过锐 +0.0077 and 过度生成 +0.0029 in distribution,
   mean RPS +0.0015 / +0.0018 (both CI90 above zero and beyond the noise guard).
3. A flaw of the recipe, found when looking for the cause of the side harm: the built frames keep the seed's
   human labels for 过锐 (382 of 900 seeds positive) and 过度生成 (215), although the blur removes those defects from
   the whole frame (negatives) or from the background (positives). That is the likely cause of the harm on those
   two defects. It does not affect the target, which is null on every test, so by the rule fixed above the recipe
   is closed and no corrected variant is run on the screening folds. Lesson for any later recipe: an operation
   that changes the visibility of another defect must leave that defect unsupervised.
