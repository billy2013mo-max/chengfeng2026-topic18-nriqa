# Round 3 reply: skip Plan A and E2 tonight; use the GPUs for 38B F2–F5 and a third control seed

I read the per-seed numbers in `p0_teacher_gate_results.json` and the loss code in `vlm_c233397_qy8.py`. I did not run anything new.

## Q1: what the dual result actually shows

**1. The solo gap is mostly averaging, not diversity.** `dual_32b_38b` averages 4 training runs, while each family's "2seed" averages 2. Averaging partly independent models always lowers solo RPS. What matters is the gain on top of the ensemble, and there the dual adds almost nothing over the better single family:

| add on `resident_v6_core` | a=0.15 | a=0.25 |
|---|---|---|
| intern38b_2seed | −0.00033 | −0.00038 |
| qwen32b_2seed | −0.00025 | −0.00029 |
| dual | −0.00033 | −0.00044 |

At a=0.15 the dual equals 38B alone, and at a=0.25 it is only 0.00006 better. That is well inside noise, so "teacher diversity" is not shown.

**2. Training noise is as large as the effect.** The single-seed gains at a=0.15 are:
- 32B: −0.00029 (s2026) vs −0.00014 (s2027)
- 38B: −0.00015 (s2026) vs −0.00047 (s2027)

Two seeds of the same family differ by 0.00015–0.00032. We have one control seed (`s2027`, −0.00004), and our rule needs the mean of three. The −0.00044 also comes from choosing the best of 3 weights × 5 configs on the same two folds that are being scored, so it is optimistic.

**3. No version meets the −0.0005 threshold we set for KD.** The dual's best is −0.00044. The only pass is `intern38b_s2027` at a=0.25, which is one seed and a picked configuration.

**4. Plan A cannot give a usable answer tonight.**
- **The expected effect is too small to detect.** A perfect student is roughly the replace test (`REP dual 0.18` = −0.00025, CI already crosses 0). A student keeping 30–60% of that gives about −0.0001. Fold 0 has ~228 orig frames, so the CI90 half-width is about 0.0003–0.0004. A null or a pass would both be uninterpretable.
- **It won't finish in 4.5h.** Four inner 32B teachers have to finish before the student can start. A 32B fold here takes several hours (F6–F7 still have 5.2h to go), so the teacher stage alone uses the whole window.
- **The trainer has no KD support yet.** `vlm_c233397_qy8.py` would need new loss and target-loading code plus a smoke test first.

**5. E2 CoT-RL: still no.** Nothing has changed since Round 2: decision-layer RL collapsed in E1 (Doc2S 34.9 and 10.5), and the test set is too small to resolve the effect.

**6. You need to decide one rule question.** Round 1 noted the statement limits trained models to ≤30B parameters. The answer changes what tonight's runs are for:
- **If 32B/38B may ship as members:** KD is unnecessary. Refit the decoder with them as members under 8-fold nested CV.
- **If they cannot ship:** KD is the only route, and its realistic ceiling is about −0.0001 RPS. Then it is a low-priority line, not tonight's main use of the GPUs.

## Recommended use of the 4.5h window

| GPUs | When free | Run | Why |
|---|---|---|---|
| 0–3 | ~40 min | **38B s2026 on F2–F5** | Needed either way (member decision or full-fit KD targets). Won't finish by morning, but the output is reusable, unlike Plan A's. |
| 5, 6 | ~25 min | **8B control s2029 on F0 and F1** | Gives the 3-seed control mean (s2027/28/29). Then re-run `eval_p0_teacher_gate.py` against that mean. |
| 4, 7 | ~5.2h | 32B F6–F7 (unchanged) | Completes 32B on all 8 folds. |

Fix this decision rule before anything is scored:
- Refit the decoder with the teachers as members, choosing α in the inner loop, never on F0–F1.
- Pass only if `dRPS_orig` CI90 < 0 against the 3-seed control mean, and nested `Doc2S_orig` ≥ +0.2.

## Q2: KD design to fix in advance (for whenever KD runs)

Mix the teacher into the target of the existing loss instead of adding a separate KD loss. That leaves one parameter and keeps the recipe otherwise unchanged:

- **Target mixing.** `q_mix = (1−λ)·q_gold + λ·q_teacher`, passed through `DeQAOrdinalDistributionLossC233397` (L257). Reasons:
  - The KL term's cross-entropy part is linear in the target, so this equals λ-weighted KD.
  - The existing `beta_emd=1.5` already serves as the CDF-L2 weight, so **β_cdf = 1.5** stays consistent with the gold term. A separate β would only add a mismatch.
- **λ = 0.5** on labelled rows, fixed in advance and not tuned on the outer fold. If a grid is ever wanted, tune it on inner folds only. **λ = 1** on unlabelled frames.
- **T = 1.** The teacher is a 4-run average, so its targets are already soft. With 4 ordinal grades, T > 1 mostly flattens the P0 tail the decoder depends on. T² scaling doesn't matter at T = 1.
- **Clean head.** Target = `(1−λ)·clean_gt + λ·teacher_P(clean)`. Take P(clean) from the teachers' own clean heads, not from the product of the six marginals.
- **Leave these on gold only:**
  - `p0_sample_boost`: keyed on gold `y==3`; weight 1 on unlabelled frames.
  - The 0.35 Thurstone ranking loss.
  - Per-dimension weights stay as they are.
- **Teacher outputs.** Use raw `probs_6x4` averaged over the 4 runs, taken before the decoder.
- **Leakage**
  - Labelled-row targets must come from inner cross-fit teachers, otherwise the teacher just hands back its own training labels.
  - Full-fit targets: the standard 8-fold OOF predictions.
  - Unlabelled frames: only from training-fold videos or external sources. Frames from validation videos would leak through sibling frames (≈10% RPS).
- **Evaluation.** 2 student seeds against the 3-seed control mean; pass threshold −0.0005 as already set.

---

**中文摘要：** 双教师的单模型优势主要来自 4 个模型平均，不是家族多样性。叠加到集成上，dual 在 a=0.15 时与 38B 单家族相同（−0.00033），a=0.25 时只好 0.00006，属于噪声。同家族两个种子之间差 0.00015–0.00032，与效应量相当；对照只有 1 个种子（规则要求 3 个）；−0.00044 是在同两折上从 3×5 个配置里挑出来的。没有任何组合稳定达到 KD 门槛 −0.0005。方案 A 今晚不可行：学生预期增益约 −0.0001，单折 CI 半宽约 0.0003–0.0004，测不出来；4 个内层 32B 教师本身就要占满窗口；训练脚本还没有 KD 代码。E2 维持否决。需要你决定：32B/38B 能否作为成员上线（≤30B 规则）？能上线就不需要 KD；不能上线的话，KD 上限约 −0.0001，优先级很低。今晚安排：GPU 0–3 跑 38B s2026 F2–F5；GPU 5–6 跑 8B 对照 s2029 F0/F1，凑齐 3 种子均值后重新评估门控；GPU 4/7 照旧跑完 32B F6–F7。KD 预注册：把教师混入目标 q_mix=(1−λ)q_gold+λq_teacher，沿用原损失（β_cdf 即现有 1.5），λ=0.5（无标签帧 λ=1），T=1；clean 头同样混合；P0 加权和排序损失只用金标；带标签行用内层交叉拟合教师，全量训练用 8 折 OOF；2 个学生种子对比 3 种子对照均值。
