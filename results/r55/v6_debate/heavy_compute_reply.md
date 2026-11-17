# Ranking the 8×H200 paths beyond `resident_v6_core`

**Short version:** compute isn't what's holding us back. Data and label noise are. The project's record so far: single-tower gains get absorbed by the ensemble (R1 again: solo RPS −0.0014 became 0.0000 inside the ensemble), synthetic data didn't help, and the one KD run that worked gave only −0.00030 RPS. So the first H200 job shouldn't train a deployable model. It should measure how much a mega-teacher could add at most. If that ceiling is within noise, it closes options 1–3 in one run.

## Option 2a: swap `p1_qy` for a 27B/26B VLM. Ruled out by VRAM and the 30B cap

| Model | bf16 on disk | NF4 resident (est.) | Swap result (est.) |
|---|---|---|---|
| `JEV-27B-VL` (dense, 64L×5120) | 53 GB | ~15–17 GB (embeddings usually stay bf16) | ~26–28 GB resident, ~30 GiB `nvidia-smi` |
| `JEV-Gemma4-26B-A4B` (MoE, 128 experts, top-8) | 49 GB | ~14–15 GB | ~24–25 GB resident |

- **MoE doesn't save VRAM.** The "4B active" figure only reduces compute. All 26B weights still have to stay resident to meet `N=1 == N=150` without reloading per image.
- **27B also breaks the 30B active-param cap.** 27B plus InternViT-6B is about 33B. That rules it out as a deployed model no matter how much VRAM we free.
- **Gemma4 could only fit by dropping 2–3 vision towers.** We already know each tower adds diversity, so this only makes sense if Gemma4 alone beats the whole ensemble. Nothing we've seen suggests it would.

Verdict: don't run it as a deployment candidate. Use these models only as **teachers**.

## Option 2b: mega-teacher → 7-of-8 nested cross-fitted KD. The only path with real upside, but gate it first

Its advantage stands: zero added inference VRAM and latency. Two caveats from earlier work:

- **KD can't beat the teacher.** At best it recovers part of what the teacher would add to the ensemble directly; for logit KD into a different-family student that's typically 30–60%. If adding the (non-deployable) teacher's OOF predictions straight into `resident_v6_core` gains less than about 0.0002 RPS, KD will be noise.
- **Cross-fitting is expensive.** The soft labels on student training images have to come from teachers that never saw those images. Otherwise KD just copies the hard labels back, which is presumably why the `ogk`-style teacher signal was ~0 before. Full nesting means 8×7 = 56 teacher runs per teacher model. Only pay for that after the ceiling probe passes.

## Option 1 (R3: `p1_qy v2` with r=64 / 1024px / vision unfreeze). Low priority

R1 just showed that making `p1_qy` better on its own doesn't move the ensemble. Higher resolution and ROIs mainly help *detection*, but the NEXT55 audit found the bottleneck is *grading*. Raising the LoRA rank on ~2.7k images mostly adds variance. The only version worth trying is "R3 = KD target", which is part of 2b, not a separate track.

## Option 3 (`v16` / `cradio_v3` with better data, resolution, or multi-task heads). Lowest priority

Every single-tower recipe change so far has been ineffective (R55, explore-3, the P3–P6 screens). Auxiliary region/pairwise heads add training signal but no new *labels*, and the limit is the labels. I'd only revisit this as a KD student after 2b passes.

## Option 4: the realistic ceiling

- Inter-frame human agreement on `过度生成` is 44–61%, and the earlier audits already estimated 55 as unreachable.
- At the 8-fold OOF scale, noise is roughly ±0.00005–0.0001 RPS and roughly ±0.1 Doc2S.
- My estimate for the best case of mega-teacher + KD: **+0.2 to +0.5 orig**, with maybe a 30–40% chance of finding anything above noise. This is a judgment call, not something we've measured. Anything above +1 would be a surprise.

## The pilot to run first: teacher-ceiling probe (P0)

1. **Fine-tune both teachers.** `JEV-Gemma4-26B-A4B` and `JEV-27B-VL` in bf16 on H200, each with the `c233397` DoRA + 63-avg+1-summary + SWA recipe and the same views as `p1_qy`.
2. **Use 2 outer folds of the existing video-grouped 8-fold split, not 2-fold CV.** Training on 7/8 of the data keeps the learning curve comparable to production, and the grouping avoids the same-video leakage.
3. **Measure four things:**
   - each teacher's solo OOF RPS against `p1_qy`
   - ΔRPS_ALL / ΔRPS_ORIG / ΔDoc2S when the teacher is **added** to `resident_v6_core` with the frozen decoder
   - the same deltas when the teacher **replaces** `p1_qy`
   - the paired bootstrap CI for each
4. **Pre-register the gate before looking at results:** add-teacher ΔRPS_ORIG ≤ −0.00020 with the CI excluding 0, averaged over ≥2 seeds, compared against the mean of 3 control seeds (the SynthLab rule).
   - **If it passes:** go to 2b for the single best teacher. Spend the nested teacher runs on those 2 folds first, KD into `p1_qy` and `v16`, and keep it only if the frozen-decoder gate passes again.
   - **If it fails:** close R3, 2a, 2b and option 3 together. Ship `resident_v6_core`.

**Cost:** about 4 teacher fine-tunes × 2 seeds. That's roughly a day of 8×H200 time, versus weeks for the full nested KD it either justifies or saves.

I haven't started anything. Tell me if you want me to write and launch the P0 script. `code/r55/build_clean_kd_teacher.py` and `run_wave3_kd_g14.sh` already have the KD plumbing to build on.
