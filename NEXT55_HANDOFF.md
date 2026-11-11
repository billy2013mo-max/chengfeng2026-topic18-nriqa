# NEXT55 Handoff — NR-IQA 命题18: path from ~53 toward 55+

> Written 2026-11-10 after a full audit of this repo.
> **Executing agent: follow §6 step by step, in order. Do not skip a gate. Do not improvise new architectures.**
> Every number in §3–§5 was reproduced in this session with `code/next55/*` on the canonical 8-fold split
> (`data/qingyu_splits/qingyu_8fold.csv`, N=3573, 513 clean, group-constrained).

---

## 0. TL;DR

| | value |
|---|---|
| Best online score so far | **53.03** (`phaseB_overgen8`, 2026-11-08). The V18 retrains scored 52.1 / 51.80. |
| Best strict OOF today (6-tower mean, nested decoder) | Doc1 **51.13** · Doc2_Strict **50.51** · **Doc1@654 51.16** |
| Same ensemble *without* clean-cliff accidents (pos+p0+20) | **≈ 53.1–53.5**, which matches the online 53.03 |
| What 55 needs | Doc1 **S_pos + S_p0 ≈ 35.0** (today 33.1–33.5) **and zero P0 on clean images** in the final 654 |
| Realistic outcome of this plan | +0.4 … +1.0 → **online-equivalent ≈ 53.8–54.5**. 55 is a stretch (≈15%). Report honestly. |
| Most important new findings | (1) The website almost certainly scores **Doc1**, so the clean *cliff* is the #1 risk on 654 images. (2) The current 8-fold ensemble **is not shippable** (exceeds the 30B cap; V16 had no weights), hence the full-fit refit in §6. |

---

## 1. Hard rules (read before touching anything)

1. **AGENTS.md rules 1–2 stay in force.** Never read old submission JSONs as a base or anchor. Never tune on test150 / final-set predictions, distributions or online feedback. All thresholds come from OOF (the nested decoder in `code/next55/decoder55.py`).
2. **Owner approval required** before any online submission (practice site or the 3 early-evaluation slots). This includes the metric probes in §2.3 and the A/B in Step 8.
3. **Kill finished inference processes** and check with `nvidia-smi` (see memory `stop-idle-inference-processes`).
4. **Disk:** `/wanqing-develop` is **97% full (≈376 GB free)**. Each Tier-1 fold ckpt is 0.2–0.75 GB. V16 ckpts now save trainable params only (<~2 GB/fold). Run `df -h /wanqing-develop` before every GPU step and stop if free space is < 80 GB.
5. **Never `git add` results/, checkpoints, `.pt` or `.npz`.** Commit only code and docs, and only when the owner asks.
6. AGENTS.md rule 3 ("6B + 4 specialists") conflicts with the 5-backbone qingyu direction the owner launched on 11-10. This plan keeps InternViT-6B as an *optional* tower (Step 4) and lets the OOF gate decide. **Flag this to the owner. Do not resolve it yourself.**

---

## 2. The metric — what actually matters

### 2.1 Two formulas live in this repo
`code/qingyu/config_and_paths.py::evaluate_qingyu_dual_metrics` computes both.

| | S_pos (25) | S_p0 (20) | S_clean (20) |
|---|---|---|---|
| **Doc1** (older doc, `utils.competition_score`) | mean over 6 *classes* of class-mean F1 | 0.6·R + 0.4·F1 (macro over 6 P0) | **cliff**: `max(0, 1 − FPR/0.02)`, FPR = fraction of clean images with *any P0* |
| **Doc2_Strict** (current spec, appendix 2) | mean over 14 labels | 0.5·R + 0.3·F1 + 0.2·P | fraction of clean images predicted all-six 无问题 |

### 2.2 Evidence that the website uses Doc1
- Online S_pos was 16.5–16.7 on every submission. On OOF, Doc1 S_pos is 16.7–17.0 and Doc2 S_pos is 15.1–15.4. Only Doc1 matches.
- Online S_clean was 20.00 on all ~8 submissions. Under the Doc1 cliff (OOF rate of P0-on-clean ≈ 0.2%) the chance per submission is ≈97%. Under Doc2 strict (OOF Clean2 ≈ 0.97) it is ≈61% per submission, ≈2% for eight in a row.
- Doc1 pos+p0 on OOF (33.1–33.5) + 20 gives 53.1–53.5. The best online score is 53.03.

**Consequence.** On the 654-image final set (≈94 clean images), **one** P0 on a clean image costs ≈10.6 points and two cost all 20. The pooled 3573 OOF hides this: one flag there costs only ≈2. `common55.score()` therefore adds **`doc1_final`** = Doc1 S_pos + S_p0 + E[cliff on a 654-size set]. **`doc1_final` is the default objective for every decoder in this plan.** Doc2_Strict is always printed as a guard.

### 2.3 Optional: make it certain (needs owner OK, costs 2 practice submissions)
```bash
python code/next55/make_metric_probes.py --write      # -> results/next55/probes/probe_{A,B}.json
# owner uploads both to the practice site and reads the three numbers back:
python code/next55/make_metric_probes.py --A-clean <S_clean of A> --B-pos <S_pos of B> --B-p0 <S_p0 of B>
```
The probes are constant predictions, so no per-image information is learned. If the result says **Doc2**, re-run every `stack55.py` call with `--objective doc2_strict_total`. Nothing else changes.

---

## 3. Stage-by-stage review (history → score → bottleneck)

| Stage (commit) | What it was | Online | Strict / OOF | Bottleneck / verdict |
|---|---|---:|---:|---|
| V9–V12 (`9d71990` → `2f18b32`) | multi-backbone specialists, ABMIL+CORN, InternViT-6B ABMIL | 52.89 (Phase-1) | blind s16/17 Doc1 ≈ 52.1–52.9 | Plateau ~50 Doc2. 6B adds ≈0 on top of the ensemble (§4.4). |
| Phase B + overgen8 (`2f18b32`, 11-08) | V12 physics + SA-AOR + 8 OverGen corrections | **53.03** (best) | — | Included hand corrections on test150, so it is not reproducible from the model alone. |
| QualiRAG / d1a437a consensus | rule-based edits on top of Phase-1 | 52.70 | +0.52 blind (claimed) | Online regressed. Post-hoc edits do not generalize. |
| V17 Sim2Real (synthetic quads) | synthetic AIGC/ISP data | — | worse on real | Domain shift. Synthetic data rejected. |
| V18 / V18.1 (`1ee7d70`, `2a78ecb`) | 4 specialists + 6B + KD, full-data | 52.1 / **51.80** | — | Under-predicted positives (329 cells), P0 dilution. |
| PRISM v1–v3.1 (`5dda244` → `4ca1a2b`) | pairwise reference-anchored, VLM8B tower | not submitted | quad s16/17 Doc1 48.6 / 49.9; vlm8b s16 43.1 (clean collapse) / s17 51.0 | Unstable. Half-data training. Abandoned. |
| Qingyu Stage-1 (`f2876e2`, 11-10) | zero-retrain stack of legacy towers | not submitted | s16–s20 Doc1 52.41 / Doc2 50.00; OOF Doc2 50.84 | Honest number ≈ 50.8 Doc2. |
| Qingyu Stage-2 Tier-1 (11-10) | 5 backbones × 8 folds | — | OOF Doc1/Doc2: cradio 51.70/49.50, pe 50.39/49.70, siglip 51.30/48.73, convnext 49.96/47.41, dinov3 48.26/47.26 | Good, reusable OOF + ckpts. |
| Qingyu Stage-2 Tier-2 (6B) | InternViT-6B 8-fold | — | **crashed (CUDA OOM)** | `gradient_checkpointing=False` with B·V = 30 crops. **Fixed in this session** (29.5 GiB peak). |
| Stage-3/4 (VLM KD, soup, deploy) | never ran | — | — | Untested code. Treat as such. |

---

## 4. Current bottlenecks — quantified (all nested, objective `doc1_final`)

### 4.1 Towers available today (full 8-fold OOF)
Reproduce with `bash run_next55.sh diag`.

| tower | Doc1 | Doc2_Strict | Doc1@654 | note |
|---|---:|---:|---:|---|
| v16_full8 (V16 AllGrade Microscope, 5-backbone, synth OG/BK) | 50.67 | 49.70 | 50.70 | best single tower; **had no saved weights** (fixed: `--save-ckpt`) |
| qy_pe_core_large | 50.55 | 49.30 | 50.58 | carries the ensemble (LOO −0.37) |
| qy_cradio_v4_h | 50.02 | 48.47 | 50.05 | carries the ensemble (LOO −0.28) |
| qy_convnextv2_base | 50.38 | 46.81 | 50.38 | LOO +0.06 (slightly harmful) |
| qy_dinov3_vitl16 | 48.39 | 46.64 | 48.42 | LOO +0.14 (harmful) |
| qy_siglip2_so400m_naflex | 47.56 | 47.91 | 47.69 | 2 P0-on-clean (cliff) |
| **mean of all 6** | **51.13** | **50.51** | **51.16** | 1 P0 on a clean image, probably a label error: `200591347322_v8L3HiV9_4.53s.png` |
| pe + cradio + V16×2 | 51.55 | 51.08 | — | V16 weight 2 ≈ a 2nd V16 seed → motivates Step 2 |

(LOO = change when the tower is removed from the 6-mean, measured with the `blend` objective. The last row is from the same session.)

### 4.2 Where the points go (6-tower mean, per-channel F1)
Doc1 S_pos is class-averaged, so each class is worth 25/6 = 4.17 points.

| class | P2 | P1 | P0 | class-mean F1 | Doc1 points lost |
|---|---:|---:|---:|---:|---:|
| **过度生成** | 0.28 | 0.28 | 0.60 | 0.38 | **2.57** |
| 过锐 | 0.52 | 0.43 | 0.75 | 0.57 | 1.81 |
| 模糊 | 0.44 | 0.51 | 0.83 | 0.59 | 1.70 |
| 纹理损失 | 0.67 | 0.65 | 0.80 | 0.71 | 1.23 |
| 虚化突兀 | — | — | 0.81 | 0.81 | 0.79 |
| 花屏 | — | — | 0.98 | 0.98 | 0.10 |

The **P1/P2 grade boundary** is the dominant loss, and 过度生成 is the biggest single sink. S_p0 ≈ 16.4/20 (R 0.84, P 0.75).

### 4.3 Dead ends verified in this session (do NOT retry)
- **Learned stacking** (per-dimension multinomial LR on all tower log-probs, cross-dimension features, strictly nested): C = 0.005 / 0.05 / 0.5 gives **−0.11 / −0.32 / −1.12** vs plain mean, winning 0–3 of 8 folds. The fusion layer has no headroom. Use `mean`.
- **Routing 过度生成 to the VLM** (legacy Qwen3-VL-8B tower): the VLM is *worse* on 过度生成 (class F1 0.333 vs 0.398 for the ensemble). Routing gives −0.03 to −0.19.
- **Legacy InternViT-6B towers** on top of the 8-fold ensemble (2891-image common subset): +0.08 Doc2 (v11) / −0.42 (v12fix). The legacy VLM gives +0.21. These were trained on smaller splits, so an 8-fold retrain may do somewhat better (Steps 4–5 are gated).
- From history (memory files): 16-dim per-class threshold search, PE-Core-G (clean collapse), DINOv3, synthetic Sim2Real, post-hoc rule edits, 69-model caches, and anchoring to legacy JSONs.

### 4.4 Calibration trap (fixed in tooling)
The decoder is fitted on **single-model** OOF probabilities. Averaging 8 fold models on test is smoother and under-predicts rare grades: on test150, mean_prob gives 过度生成 P2/P1 = 7/1, while per-fold decode + median gives 11/4 (train prevalence ≈ 7.7%/6.1%).

`build_submission55.py --fold-agg median_vote` (default) decodes each fold-k ensemble in the OOF regime and takes the per-cell ordinal median. With the **full-fit** model (one model per tower) this issue disappears, because each tower is again a single model.

---

## 5. Gap math to 55 (be honest in every report)

Online ≈ Doc1 (S_pos + S_p0) + 20, if the cliff is not hit. Today that is 33.1–33.5, so 53.1–53.5. Expected increments from this plan (from OOF evidence above, not promises):

| lever | expected Δ (pos+p0) | evidence |
|---|---:|---|
| 2nd seed of V16 (Step 2) | +0.2 … +0.4 | V16×2 weighting gave +0.28 Doc2 / +0.13 Doc1 |
| 2nd seed of pe/cradio/siglip (Step 1) | +0.1 … +0.3 | variance reduction; LOO shows pe/cradio matter most |
| drop dinov3/convnext | +0.05 … +0.15 | LOO |
| InternViT-6B 8-fold (Step 4) | 0 … +0.3 | 6B is strongest on 过锐 (0.55 vs 0.50 per tower) |
| Qwen3-VL-8B 8-fold (Step 5) | 0 … +0.3 | legacy VLM +0.21 on subset |
| **total (not additive, expect ~60%)** | **+0.4 … +1.0** | → **≈53.8–54.5** online-equivalent |

Reaching 55 needs ≈+1.5. That most likely requires a *new* signal for 过度生成 / P1-P2 boundaries, for example label-noise-robust ordinal training or a stronger semantic tower. The VLM tower is the only one scheduled. If after Step 6 the frozen stack's `doc1_final` + clean-cliff headroom is < 54.0, **say so plainly to the owner. Do not "close the gap" with post-hoc edits.**

---

## 6. Execution plan (strict order)

Conventions:
- Run everything from the repo root `/wanqing-develop/luowenjing/18`.
- Every step has **CMD → expected time → PASS gate → on FAIL**.
- Log every result in `results/next55/LEDGER.md` (create it). Columns: date, step, tag, towers, Doc1, Doc2S, Doc1@654, Δ vs previous accepted, decision.

### Step 0 — sanity (CPU + 1 GPU, ~10 min)
```bash
cd /wanqing-develop/luowenjing/18 && nvidia-smi && df -h /wanqing-develop
bash run_next55.sh diag          # must reproduce §4.1: 6-tower mean Doc1 ≈51.13, Doc2S ≈50.51
bash run_next55.sh smoke_iv6b    # must print "[smoke-iv6b] PASS" (peak ≈29.5 GiB, LoRA_B 180/180)
```
**PASS:** numbers within ±0.05 of §4.1 and the smoke prints PASS. **FAIL:** stop and report. Something changed in `results/qingyu` or the split.
Accepted reference = stack tag **`s0`** (6 towers).

### Step 1 — Tier-1 seed bag, seed 2027 (8 GPU, ≈2 h)
```bash
nohup bash run_next55.sh tier1_seedbag 2027 pe_core_large cradio_v4_h siglip2_so400m_naflex > logs/next55/step1.log 2>&1 &
```
Outputs go to `results/next55/towers/qy_s2027/<bb>/fold_{0..7}_{best.pt,probs.npz}` and are auto-discovered as `qy_s2027_<bb>`.
**Per-tower PASS:** `python code/next55/diag_oof.py --only qy_s2027_pe_core_large --no-combos` gives Doc2S within ±0.5 of the seed-2026 tower (§4.1). A large drop means a broken run: delete that dir and re-run that backbone once.

### Step 2 — V16 seed bag, seed 700 (8 GPU, ≈1.6 h; can run after Step 1 or on another node)
```bash
nohup bash run_next55.sh v16_seedbag 700 > logs/next55/step2.log 2>&1 &
# after ~30 min check fold 0 is training:  grep "ep=" results/next55/towers/v16_s700/V16_Full_F0_AllGrade_Microscope.log
```
**PASS:** 8 × `*_probs.npz` **and** 8 × `*_swa.pt` exist, and the `DONE` lines show Doc2 ≈ 49–51.
**Reload gate (mandatory, new code path):**
```bash
CUDA_VISIBLE_DEVICES=0 python code/next55/infer_v16_ckpt.py --tower-dir results/next55/towers/v16_s700 --fold 0 \
   --template results/v2_full/template_150.json --tag t150chk --verify-va
# must print "[verify] fold 0: max|Δ va_p0| = <2e-2" ; then: rm results/next55/towers/v16_s700/ext_t150chk_F0.npz
```
**FAIL** (assert): the checkpoint misses tensors. Inspect `missing` keys in `infer_v16_ckpt.py`. The likely cause is a frozen weight that `build_v9_backbone` does not restore. Fix that before going further.

### Step 3 — stack gate #1 (CPU, ~5 min)
```bash
bash run_next55.sh stack s1 qy_pe_core_large qy_cradio_v4_h qy_siglip2_so400m_naflex \
     qy_s2027_pe_core_large qy_s2027_cradio_v4_h qy_s2027_siglip2_so400m_naflex v16_full8 v16_s700
# also try the "no-weak-tower" variant and the "everything" variant:
bash run_next55.sh stack s1_all          # every discovered tower
```
Read `results/next55/stack/<tag>/report.json → cands.mean.bootstrap_vs_ref.doc1_final` (the ref is `s0`'s 6 towers).
**ACCEPT a tag if:**
- Δ`doc1_final` ≥ **+0.15**, and
- bootstrap `p_le0` ≤ **0.20**, and
- Δ`doc2_strict_total` ≥ **−0.10**, and
- `n_clean_p0_flag` (printed as `P0-on-clean=`) does not increase.

Pick the accepted tag with the highest `doc1_final`. Among near-ties (within 0.05) prefer **fewer towers / fewer params**. Expect ≈ +0.3 … +0.6.

### Step 4 — InternViT-6B 8-fold (optional, 8 GPU, time-boxed)
Only start once Steps 1–3 are done. Time-box: **if fold 0 has not finished epoch 1 within 3 h, kill it and skip this step.**
```bash
nohup bash run_next55.sh iv6b_8fold > logs/next55/step4.log 2>&1 &
python code/qingyu/train_tier2_internvit6b_8fold.py --aggregate-8fold --out-dir results/next55/towers/iv6b   # summary line
bash run_next55.sh stack s2 <accepted towers from Step 3> iv6b_internvit_6b
```
**ACCEPT** with the Step-3 rule (Δ vs the accepted tag; pass `--ref-towers <accepted list>` via `python code/next55/stack55.py --tag s2 --towers ... --ref-towers ...`). Otherwise drop it. It costs 6B params and 4090 VRAM.

### Step 5 — Qwen3-VL-8B 8-fold (optional, never run before → smoke first)
```bash
bash run_next55.sh vlm_smoke     # fold 0 only, hard timeout 3 h
```
**Smoke PASS:** `results/next55/towers/vlm/qwen3_vl_8b/fold_0_probs.npz` exists, and the summary shows Doc1 ≥ 47 and Clean2 ≥ 0.9 on fold 0. If it OOMs, retry once with `--batch-size 2 --grad-accum 4` (edit the `vlm_smoke` line). Otherwise skip Step 5.
If PASS: `nohup bash run_next55.sh vlm_8fold > logs/next55/step5.log 2>&1 &`, then a stack gate exactly as in Step 4 (tower key `vlm_qwen3_vl_8b`).

### Step 6 — second seeds (only if Steps 1–2 each helped)
If Step 3 showed both the Tier-1 seed bag and V16 seed 700 helped (check by removing each group, e.g. a `stack` run without the `qy_s2027_*` towers):
```bash
bash run_next55.sh v16_seedbag 800
bash run_next55.sh tier1_seedbag 2028 pe_core_large cradio_v4_h
bash run_next55.sh stack s3 <accepted towers> v16_s800 qy_s2028_pe_core_large qy_s2028_cradio_v4_h
```
Same acceptance rule. **Stop adding seeds** when Δ < +0.10.

### Step 7 — freeze the configuration
- Final tag = the best accepted tag. Record it in LEDGER.md as **FROZEN**. After this point, no further OOF-driven changes.
- **Param budget check (≤ 30B, rules §2):** sum the full-fit params of the frozen towers.
  - Tier-1: pe 0.35B, cradio 0.67B, siglip 0.43B, convnext 0.18B, dinov3 0.33B (per seed).
  - V16 ≈ 3.3–4.2B per seed (5 backbones). The exact value is printed by `infer_v16_ckpt.py` as `params=`.
  - InternViT-6B 6.1B; Qwen3-VL-8B 8.1B.
  - Must be ≤ 30B, and preferably ≤ 20B for the 4090 S_cost score.
  If over budget, drop the lowest-LOO towers and re-run `stack` with the reduced list. That becomes the new frozen tag.

### Step 8 — test150 A/B (owner approval; at most 2 practice uploads)
```bash
bash run_next55.sh t150 <FROZEN>                                   # median_vote (default)
python code/next55/build_submission55.py --stack-tag <FROZEN> --target t150 --deploy 8fold --fold-agg mean_prob \
   --template results/v2_full/template_150.json --out results/final/pred_nriqa_dataset_test_next55_<FROZEN>_meanprob.json
```
Both must print `[Submission Hard Gate: PASS]`. The owner decides whether to upload. Record the online scores in `docs/SUBMISSION_LEDGER.md`. **Do not tune anything on the returned numbers.** The only allowed decision is the binary fold-agg choice, and it matters only for 8-fold deploys (the shipped full-fit model is unaffected).

### Step 9 — full-fit refit = the shippable model (8 GPU, ≈2–4 h)
```bash
python code/next55/launch_fullfit.py --stack-tag <FROZEN> --dry-run     # review the plan
bash run_next55.sh fullfit <FROZEN>                                      # writes results/next55/fullfit/...
```
- One model per frozen tower, trained on 100% of the data with the same seed and hyper-parameters as its OOF run.
- The decoder stays the OOF-fitted one inside `final.pkl`. Do not refit it.

**PASS:** every planned job prints `DONE`. Then check the test150 build from the shippable model:
```bash
python code/next55/build_submission55.py --stack-tag <FROZEN> --target t150 --deploy fullfit \
   --template results/v2_full/template_150.json --out results/next55/pred_t150_fullfit_<FROZEN>.json
```
Must pass the hard gate. Its per-dimension counts should be in the same range as Step 8 (±30%). Wildly different counts mean a broken refit. Investigate; do not "fix" with thresholds.

### Step 10 — final set (when the organiser delivers the 654-image set)
```bash
python code/next55/prep_final_set.py --template <final.json> --img-dir <final_img_dir>   # prints 4 export lines
export FINAL_TEMPLATE=... FINAL_IMG_DIR=... NEXT55_EXTRA_IMG_DIR=... NEXT55_DUAL_NPZ=...    # paste them
FINAL_TAG=final654 bash run_next55.sh final <FROZEN>
```
This step:
1. Runs inference with every full-fit tower, checking that each checkpoint reload reproduces its stored val probabilities.
2. Builds `results/final/pred_final654_next55_<FROZEN>.json`.
3. Runs the official hard gate.

Hand that file plus `results/next55/stack/<FROZEN>/report.json` to the owner. **Run the `final` step twice** and `diff` the JSONs to confirm determinism (required by the reproduction audit).

---

## 7. Stop / escalate rules
- Any gate FAIL that a single documented retry does not fix → stop, write what happened in LEDGER.md, and ping the owner.
- Never edit `decoder55.py` grids, `TAU`, `FINAL_N` or the acceptance thresholds to make a gate pass.
- If the probe (§2.3) says Doc2: switch `--objective doc2_strict_total` and **re-run every stack gate from Step 3**. Accepted tags may change.
- If the owner asks "will we hit 55?": quote §5 and the frozen tag's `doc1_final`. Do not round up.

---

## 8. What changed in this session (file index)

**New — `code/next55/`**

| file | purpose |
|---|---|
| `common55.py` | split/labels, metric incl. **`doc1_final`** (cliff at 654 scale), tower auto-discovery, OOF/test loaders (8-fold & full-fit) |
| `decoder55.py` | 17–23-parameter decoder (per-dimension log-bias, P0 floors, global clean gate), coordinate ascent, nested LOFO |
| `diag_oof.py` | per-tower / ensemble nested OOF + per-channel table (`bash run_next55.sh diag`) |
| `diag_split_towers.py` | marginal value of legacy s16–s20 6B/VLM towers on the 2891-image subset |
| `stack55.py` | strictly nested fusion (`mean`; `lr` available but rejected), paired bootstrap vs ref, writes `final.pkl` |
| `build_submission55.py` | submission from fresh tower outputs only; `--deploy 8fold|fullfit`, `--fold-agg median_vote|mean_prob`, runs the official validator |
| `launch_v16_seedbag.py` | V16 8-fold with a new seed (+ `--save-ckpt`), or `--fullfit` |
| `infer_tier1_ckpt.py` | Tier-1 inference from fold ckpts with `--verify-va` (reload reproduced at Δ ≈ 5e-7) |
| `infer_v16_ckpt.py` | V16 inference from `_swa.pt` with `--verify-va` |
| `make_fullfit_splits.py`, `launch_fullfit.py` | 100%-data refit of the frozen towers |
| `prep_final_set.py` | validates the final set, precomputes both physics caches (tested on t150: regenerated cache bit-identical to the original) |
| `make_metric_probes.py` | optional metric identification (2 constant submissions + interpreter) |
| `smoke_iv6b_mem.py` | 6B OOM-fix check |
| `run_next55.sh` (repo root) | step runner used in §6 |

**Modified**

| file | change |
|---|---|
| `code/qingyu/backbones_qingyu.py` | InternViT-6B gradient checkpointing on (`QY_IV6B_GRAD_CKPT=1` default) + embedding-output `requires_grad` hook so LoRA still trains under reentrant checkpointing (OOM fix: >138 GiB → 29.5 GiB) |
| `code/finetune/train_v16_microscope_specialist.py` | `--save-ckpt` (trainable params + buffers), `--extra-template`, extra image dir via `NEXT55_EXTRA_IMG_DIR`, asserts no missing images |
| `code/finetune/extract_native_phys_feats.py` | `NEXT55_EXTRA_IMG_DIR` support (previously missing images were *silently zero-filled*) |

**Pre-existing uncommitted changes, not from this session:** `data/qingyu_splits/split_audit.json`, `pred_nriqa_dataset_test.json`.
