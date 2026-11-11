# R55 Handoff — redesigned experiment framework for 命题18 (replaces NEXT55_HANDOFF.md)

> Written 2026-11-10/11. **Executing agent: follow §5 in order. Do not skip a gate. Do not invent new experiments.**
> `NEXT55_HANDOFF.md` is superseded: its §2 (metric = Doc1) is wrong, and its Steps 1–3 are done.
> Deadline 2026-11-23. Log every result in `results/next55/LEDGER.md`.

---

## 0. Where we are

| | value |
|---|---|
| Scoring-site metric | **Doc2_Strict** (confirmed by constant-prediction probes, see LEDGER) |
| Accepted stack `s1` (pe, cradio, siglip2 × 2 seeds + V16 × 2 seeds) | nested 8-fold OOF **50.94** (my re-run: 51.04) |
| Best online score on the 140/150 practice images | 53.03 (older model with hand edits; pure models 51.8–52.9) |
| `s1` practice file | `results/final/pred_nriqa_dataset_test_next55_s1.json` — hard gate PASS; **owner uploads it once, as a reference only** |
| InternViT-6B 8-fold run | stopped after epoch 2 of 5 (fold 0: Doc2 48.25); not restarted — optional, see §6 |

Owner decisions (2026-11-10): departing from "6B + 4 specialists" is accepted (PE-Core testbed, ≤ 3-tower
deliverable); external unlabelled frames are allowed; one practice upload of `s1` as a reference, nothing tuned on it.

## 1. Hard rules

1. AGENTS.md rules 1–2 stay in force: never read old submission JSONs as a base; never tune on test150 / final-set
   predictions, distributions or online feedback. Thresholds come only from the nested OOF decoder.
2. **File names are never model input.** They leak label priors (pipeline tags such as `v8L3HiVE`, sometimes the
   defect name itself). The only permitted use is as a *training target* for the auxiliary tag head (`--lambda-tag`).
3. Synthetic images never receive a P0/P1/P2 label from their recipe. They carry an ordering, or a teacher soft label.
4. Leak rules for auxiliary data are enforced in `train_testbed.py`; do not bypass them: rows whose seed is in the
   validation fold are dropped, and soft labels for fold k come only from fold-k teacher models.
5. Owner approval before any online submission. Kill finished inference/serving processes and check `nvidia-smi`.
6. Disk is 97% full (≈330 GB free). Run `df -h .` before every generation step; stop below 80 GB.
   `data/r55_aux/` holds ≈5 GB today; the FLUX ladder adds ≈3 GB; an external pool of 4000 frames ≈10–15 GB.
7. Never `git add` results, checkpoints, `.pt`, `.npz` or `data/r55_aux/`. Commit code and docs only when asked.
8. One pe_core_large training job takes ≈66 GB of GPU memory: **at most 2 jobs per GPU.**

## 2. Why the framework changed (all measured on `s1` OOF, 2026-11-10)

| finding | evidence |
|---|---|
| Grading is the headroom, not detection | oracle grade on detected cells 60.5 (+9.4); oracle detection with model grades 54.6 (+3.5) |
| The P0 boundary is worth the most | oracle on P0 cells 58.8 (+7.7); P0 decisions drive ≈30.7 of 65 points |
| 过度生成 is detected but not graded | detection AUC 0.96; P0-vs-P1 AUC 0.75, P1-vs-P2 0.74 (other dimensions 0.82–0.92) |
| The task is data-limited | old learning curve 25/50/75 % → 47.4 / 48.3 / 49.4, ≈ +1 per 900 images, not flattening |
| Ensembling is saturating | doubling the towers gave +0.46; learned stacking loses |
| **Seed noise is large** | two seeds of the same pe recipe: 49.60 vs 48.91, and the paired bootstrap calls that "significant" (P = 0.99) |
| The final score is noisy | random 654-image draws: sd 0.66 |

Consequences: (a) stop adding seeds/towers, add *information* about severity; (b) every gate compares ≥ 2 seeds per
side and uses a threshold above the measured seed-noise floor.

Already in the Tier-1 trainer (so NOT new experiments): a latent severity score per dimension, a cross-batch
Thurstone ranking loss, RPS, soft ordinal targets.

## 3. What is new

| id | experiment (`run_r55.sh exp <name>`) | what it adds | cost |
|---|---|---|---|
| E0 | `lc` | learning curve on the canonical split (confirms the data-limited claim under Doc2_Strict) | 3 × 8 folds |
| E1 | `rank` | same-scene ladders for 模糊 / 纹理损失 / 过锐 at native resolution, ranking loss only | built (4185 images) |
| E2 | `rankkd` | E1 + fold-teacher soft labels on the ladder images | — |
| E3 | `og` | FLUX image-conditioned re-render ladder for 过度生成, ranking + teacher soft labels | ≈ 2–4 h generation |
| E4 | `ext` | self-training on external unlabelled frames (fold-teacher soft labels) | needs an ingested pool |
| E5 | `clean` | certified-clean external frames as low-weight clean negatives | needs E4's pool |
| E6 | `tag` | train-only auxiliary head predicting the pipeline tag | none |
| E7 | `combo` | union of the flags that passed | — |

**Finding truly clean images (owner's question).** Clean means all six dimensions 无问题 under the organisers'
standard; the only ground truth is the 513 labelled clean images.
- Ranking ladders do **not** need clean seeds: "stronger ≥ weaker ≥ seed" holds whatever the seed's grade.
- Only E5 needs clean images, a few hundred. `certify_clean.py` keeps an external frame only if **all 8 fold-teacher
  ensembles** give it a clean score above a threshold calibrated on labelled OOF data for ≥ 99 % precision
  (on `s1`, the top 300 by this score were 99.3 % clean). It then writes contact sheets for a human scan; rejects go
  into a text file and are removed on rerun.
- Residual risk is bounded by a small loss weight (0.1) and by the gate (S_clean and S_p0 may not fall).
- Stakes are modest: S_clean is 19.4/20 on OOF and was 20.00 on every online submission.

## 4. Files

| file | purpose |
|---|---|
| `code/r55/r55_common.py` | paths, auxiliary manifest format, teacher tower list |
| `code/r55/build_rank_tuples.py` | E1 ladders (CPU). Done: 1395 tuples / 4185 images in `data/r55_aux/images` |
| `code/r55/build_overgen_ladder.py` | E3 ladder (`venv_flux/bin/python`, 1 GPU per rank; `--merge` afterwards) |
| `code/r55/ingest_external_pool.py` | external images/videos → pool rows; de-duplicates against labelled + practice images |
| `code/r55/prep_aux.py` | template → physics meta → fold-teacher predictions → `results/r55/aux_teacher.npz` |
| `code/r55/certify_clean.py` | certified-clean list + contact sheets |
| `code/r55/train_testbed.py` | Tier-1 trainer + the optional additions; no flags = baseline |
| `code/r55/gate_r55.py` | paired gate, `--noise` for the seed-noise floor |
| `run_r55.sh` | step runner |

Unchanged and still used: `code/next55/{common55,decoder55,stack55,build_submission55,launch_fullfit,infer_*}.py`.
Note `stack55.py` and `run_next55.sh` default objectives: always pass / export `doc2_strict_total`
(`export NEXT55_OBJECTIVE=doc2_strict_total`).

## 5. Execution plan

Conventions: run from the repo root; every step is **CMD → PASS gate → on FAIL**. Experiments run on
`pe_core_large` with seeds 2026 and 2027; baselines are `results/qingyu/pe_core_large` (2026),
`results/next55/towers/qy_s2027/pe_core_large` (2027), `results/next55/towers/qy_s2028/pe_core_large` (2028).

### Step 0 — noise floor and trainer check (started 2026-11-10 23:30 UTC)
```bash
bash run_r55.sh base 2028          # third baseline seed, 8 folds, ≈35 min  (launched by the author; check logs/r55/base_s2028_f*.log)
bash run_r55.sh noise              # prints the sd of two-seed baselines and the MIN_DELTA to use
export MIN_DELTA=<printed value>   # use it for EVERY later gate
```
**PASS:** 8 × `fold_k_probs.npz` under `results/next55/towers/qy_s2028/pe_core_large`, and
`python code/r55/gate_r55.py --name s2028 --base results/qingyu/pe_core_large --exp results/next55/towers/qy_s2028/pe_core_large`
shows Doc2S within ±1.0 of the base (same recipe; this also proves `train_testbed.py` without flags equals the Tier-1
trainer — a reproduction run of seed 2027 fold 0 gave epoch-1 loss 1.003 vs 1.006 in the original log).
**FAIL** (more than 1.0 below): the testbed trainer diverges from Tier-1 — stop and diff the two training loops.

### Step 1 — auxiliary pool ready
```bash
bash run_r55.sh prep               # ≈25 min; launched by the author for the E1 ladders
```
**PASS:** `results/r55/aux_teacher.npz` exists with `probs_folds` of shape (8, N, 6, 4), N = images in the manifest.
Rerun `prep` every time the manifest grows (Steps 4 and 5).

### Step 2 — E1 `rank` and E6 `tag` (no extra data needed)
```bash
bash run_r55.sh exp rank 2026 & bash run_r55.sh exp rank 2027 & wait      # 16 jobs = 2 per GPU, ≈60 min
bash run_r55.sh gate rank
bash run_r55.sh exp tag 2026 & bash run_r55.sh exp tag 2027 & wait
bash run_r55.sh gate tag
```
Read the verdict line. **PASS** = Δ Doc2S ≥ `MIN_DELTA`, P(Δ ≤ 0) ≤ 0.20, S_p0 and S_clean each not down more than
0.10, 2 seeds, 8 folds. Record PASS/FAIL and the per-label deltas in the LEDGER.
On FAIL: record it and move on. Do not retune loss weights more than once (allowed single retry for `rank`:
`--lambda-tuple 0.25`, by adding it to the preset line).

### Step 3 — E2 `rankkd` (only if E1 did not FAIL on S_p0)
```bash
bash run_r55.sh exp rankkd 2026 & bash run_r55.sh exp rankkd 2027 & wait && bash run_r55.sh gate rankkd
```

### Step 4 — E3 过度生成 ladder
```bash
df -h . ; bash run_r55.sh gen_overgen 400     # 8 GPUs, venv_flux; needs the GPUs free of training jobs
ls data/r55_aux/images | grep -c r55og_       # expect ≈1200
bash run_r55.sh prep
bash run_r55.sh exp og 2026 & bash run_r55.sh exp og 2027 & wait && bash run_r55.sh gate og
```
Before training, open 10 random `r55og_L1/L2/L3` triples next to their seeds. **Stop and report** if L3 is not
visibly more re-rendered than L1, or if layout/text changed — do not train on a broken ladder.
Extra acceptance for `og`: the 过度生成 per-label deltas must be positive on at least 2 of 3 grades.

### Step 5 — E4 `ext` and E5 `clean` (external pool)
```bash
bash run_r55.sh ingest hf:<repo> <tag> 2000    # or: ingest /path/to/frames <tag> 2000 ; repeat for 2–3 sources
bash run_r55.sh prep
bash run_r55.sh certify                        # then scan results/r55/clean_sheets/*.jpg
bash run_r55.sh certify rejects.txt            # after the human scan (file may be empty)
bash run_r55.sh exp ext 2026 & bash run_r55.sh exp ext 2027 & wait && bash run_r55.sh gate ext
bash run_r55.sh exp clean 2026 & bash run_r55.sh exp clean 2027 & wait && bash run_r55.sh gate clean
```
Candidate Hub sources are listed in `ingest_external_pool.py` — **contents and licences are not verified**; check
each before downloading, prefer short-form/UGC video frames, and aim for 4000–8000 frames in total.
If `certify` keeps fewer than 100 frames, skip E5.

### Step 6 — E0 learning curve (run whenever 8 GPUs are idle for ≈90 min)
```bash
bash run_r55.sh lc
```
Record the three Doc2S values next to the full-data baseline. If 75 % → 100 % gains less than +0.3, say so in the
LEDGER: the data-scale pillar (E4) is then expected to be weak.

### Step 7 — combine and propagate
1. `export R55_COMBO_FLAGS="<union of the flags of every PASSED experiment>"`, then
   `bash run_r55.sh exp combo 2026 & bash run_r55.sh exp combo 2027 & wait && bash run_r55.sh gate combo`.
   The combo must beat the best single experiment by ≥ +0.10, else keep the best single experiment's flags.
2. Propagate the winning flags to the other deliverable tower:
   `BACKBONE=cradio_v4_h bash run_r55.sh exp <winner> 2026` and `... 2027`, gate with `BACKBONE=cradio_v4_h`.
3. Stack gate against `s1` (objective Doc2_Strict):
   ```bash
   python code/next55/stack55.py --tag r1 --objective doc2_strict_total \
     --towers r55_<winner>_s2026_pe_core_large r55_<winner>_s2027_pe_core_large \
              r55_<winner>_s2026_cradio_v4_h   r55_<winner>_s2027_cradio_v4_h v16_full8 v16_s700 \
     --ref-towers <the 8 towers of s1, from results/next55/stack/s1/report.json>
   ```
   **ACCEPT `r1`** if Δ ≥ +0.15 with bootstrap `p_le0` ≤ 0.20. Otherwise `s1` stays the frozen configuration.

### Step 8 — deliverable (unchanged from NEXT55 Steps 7, 9, 10)
Freeze the accepted tag → parameter budget ≤ 30B → `bash run_next55.sh fullfit <tag>` → final-set inference twice
for determinism. The `s1` full-fit should be started as soon as GPUs are free regardless of Steps 2–7, so a
shippable model always exists. For R55 towers, the full-fit run needs the same auxiliary flags; `launch_fullfit.py`
does not know them yet — **stop and ask the owner/author before full-fitting an R55 tower.**

## 6. Optional / not scheduled
- InternViT-6B 8-fold (`bash run_next55.sh iv6b_8fold`, ≈4 h on 8 GPUs). Run only if the GPUs would otherwise idle;
  gate it into the stack as in NEXT55 Step 4.
- Dense native-resolution tiles for 过度生成. Not implemented; propose it to the owner only if E3 fails.

## 7. Stop rules and honest outlook
- A FAIL that the single documented retry does not fix → record it, move on. Never edit gate thresholds,
  the decoder, or the leak filters to make something pass.
- Expected (author's estimate, not a measurement): E1 + E6 + E5 ≈ +0.3 to +0.6, E3 0 to +0.5, E4 0 to +0.7, not
  additive. A good outcome is OOF 52–53. Whether that means 55 online depends on the OOF→online offset, which
  the single `s1` upload will measure (with ±1.6 noise on 140 images). Do not round up when reporting.
