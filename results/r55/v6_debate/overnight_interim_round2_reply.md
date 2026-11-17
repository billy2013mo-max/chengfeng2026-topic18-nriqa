# Recommendation: Choice 2, with one fix to the decision rule

I worked this out from the numbers you posted. I didn't open the scripts or the prediction files.

## Q1: Reading the Qwen3-VL-32B P0 result

**Why the gain shows up on orig frames but not on all frames.** Split the all-frames change into its two parts. There are 872 frames in total, 456 orig and 416 non-orig.

- Total change: −0.00010 × 872 ≈ −0.087
- Orig part: −0.00025 × 456 ≈ −0.114
- So the non-orig part is +0.027 / 416 ≈ **+0.00007**

On non-orig frames (synthetic, overgen or derived), adding 32B makes the ensemble slightly worse. All of the gain is on real frames. That fits two things we already know:

- 32B's solo lead over p1_qy_8b is larger on orig (−0.0020) than on all frames (−0.0014). Its advantage is a better understanding of real images, not better sensitivity to artificial artifacts.
- The current ensemble already covers the synthetic and overgen distribution through its specialists. 32B adds nothing there and only dilutes them.

**Orig is the slice that matters.** The test set is real frames, so `RPS_orig` and `Doc2S_orig` are the gating metrics. The non-orig harm is a reason to consider routing 32B only to orig-like inputs, not a reason to reject it.

**The pass is weaker than "6x the control" suggests:**
1. **Only one control seed.** Our own SynthLab rule says gates must use the mean of 3 control seeds. A single seed at −0.00004, with a CI about 0.0005 wide, can't support a 6x ratio.
2. **Several configs were tried on F0–F1.** Five variants were tested on the same two folds and the best ones were reported. The CI upper bounds are −0.00001 and −0.00002, so they only just exclude zero. After that selection, treat it as "promising", not "passed".
3. **Two seeds did not help the ensemble.** The 2-seed average helps solo (0.03931 vs 0.04044/0.04081), but in the ensemble a single seed gives −0.00029 and two seeds give −0.00025. So the gain is a different, complementary signal, not reduced noise. That means **one seed is enough** for the remaining folds.

**Does it justify KD? No, not yet, and probably not at all:**
- The solo gap (−0.0020) is the wrong measure. What matters is how much the model adds to the ensemble, which is only −0.00025, because the ensemble (0.03673) already captures most of what makes 32B good.
- A KD student recovers only part of the teacher's solo gap, and its ensemble contribution would be smaller again. Realistically that is under 0.0001, which neither a single fold nor the KD-ROI threshold (−0.00050) can detect.
- We already have direct evidence: `p12_g14_kd` and `p12_pe_kd` moved the unconstrained ceiling from 0.04200 to 0.04197, essentially zero. The prior for "KD from a stronger VLM carries over to the ensemble" is poor.
- The more direct route is using 32B itself as an ensemble member. Rule 3 was relaxed in the R55 decisions, and a 32B forward pass on 150 test images is cheap. That only needs more outer folds, not inner cross-fitting.

## Q2: How to use GPUs 0–3 for the next 4–5 hours

| | What it settles | Expected value tonight |
|---|---|---|
| **Choice 1** (inner cross-fit for KD on F0) | Whether KD keeps part of a marginal gain, on one fold (~228 orig frames) | **Low.** It can't resolve effects under 0.0001, the prior is poor, and its output can't be reused. |
| **Choice 2** (32B s2026 on F2–F5) | Whether the F0–F1 signal is real across 6 folds, plus the first meaningful Doc2S estimate | **High.** It narrows the CI by about √3, de-biases the F0–F1 config choice, and its output is reusable. |
| **Choice 3** (E2 CoT-RL on 300 images, plus 1 or 2) | Headroom for CoT-RL | **Low.** E1 already showed decision-layer RL collapsing (Doc2S 34.9 and 10.5). 300 images is too few to resolve the effect. It would also take a GPU from Choice 2 and push F2–F5 into two rounds. |

**Choose Choice 2.** Three reasons:
1. **Any route to deploying 32B needs it.** The decoder can't be properly refit with 32B as a member until 32B has out-of-fold predictions on the folds the decoder is fitted on. The current test (decoder fitted on F2–F7 without 32B, fixed weight α=0.15 added on top) is only a stand-in.
2. **The F0–F1 result is too weak to decide on.** With 6 folds (~1,370 orig frames), the CI will either clearly exclude zero or the effect will vanish. Either way, tomorrow's decision rests on evidence instead of a one-seed comparison.
3. **Doc2S is the actual objective, and it is noisy.** At 456 frames the corpus-level F1 parts (P0 and Clean) are too unstable to read. A gain of +0.15 in Doc2S on F0–F1 tells us nothing.

**Plan to fix before looking at results:**
- **GPUs 0–3 now:** 32B s2026 on F2, F3, F4 and F5, single seed.
- **GPUs 4–7 when InternVL3_5-38B finishes (~4h):** 32B on F6 and F7. That completes all 8 folds by morning. Use the other two GPUs for two more `p1_8b` control seeds, so the gate meets the 3-seed rule.
- **Decision rule:** refit the decoder with 32B as a member, using 8-fold nested CV and choosing the weight inside the inner loop, not on F0–F1. Ship only if both hold:
  - the CI90 upper bound of `dRPS_orig` is below 0, compared against the mean of the 3 control seeds;
  - nested `Doc2S_orig` is ≥ +0.2.
- **Also report** the non-orig slice. If the +0.00007 harm persists, test routing 32B only to orig-like inputs, using a routing rule learned on the validation set, never the test set.
- **If InternVL3_5-38B passes as well,** compare `dual_32b_38b` as a member under the same rule. That is still cheaper than any KD route.

For context on headroom: the sibling-pair estimate of the irreducible RPS (0.013–0.020) is a lower bound, because frames from the same video are labelled almost identically (clean agreement 0.983–1.000). So the gap to 0.0367 overstates how much can actually be gained. Real but small gains of about 0.0003, like 32B's, are the realistic target. Choice 2 is the option that can confirm one.

---

**中文摘要：** 我是根据你给出的数字推算的，没有打开脚本或预测文件。全部帧的提升只有原始帧的一半左右，是因为在非原始帧（合成/过生成）上加入 32B 反而略微变差（约 +0.00007）。增益全部来自真实帧，而测试集正是真实帧，所以应以原始帧指标为准。这次通过只是勉强：对照只有 1 个种子，不符合我们"3 个对照种子取均值"的规则，而且 5 个配置都在同两个折上挑选过，CI 上界刚好低于 0。暂不建议做 KD：真正该看的是对集成的边际贡献，只有 −0.00025，不是单模型的 −0.0020；历史上的 KD 模型对集成几乎没有贡献。GPU 0–3 选方案 2，用单种子跑 F2–F5；InternVL 跑完后用 GPU 4–7 补 F6–F7，再加两个对照种子。之后在 8 折嵌套交叉验证下直接把 32B 作为集成成员来评估，判定标准要在看结果之前定死。
