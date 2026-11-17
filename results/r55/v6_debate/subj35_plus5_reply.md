I'm treating this as a real audit rather than a translation request. The most important finding is that the S_cost rubric quoted in your question is not the current one.

# Subjective 35 + Semantic 5: audit

## 0. The rubric premise is outdated

There are two rules files under `documents/`, dated 10-25 and 11-05. The newer one (`(2).md`, line 329) says:

- **Tier ① (0.7–1.0):** the model runs stably on one 4090 under the official test load, with no extra GPU or paid service, and **资源使用有余量** (has headroom). It adds: "是否采用量化、蒸馏、裁剪、串行或分批执行，**不作为降档依据**". Using quantization, distillation, pruning, serial execution or chunking is not a reason to move down a tier.
- **Tier ② (0.3–0.6):** "资源余量不足、吞吐受限或频繁模型换入换出" (not enough headroom, limited throughput, or frequent model swapping in and out). It also says "**不因使用轻量化技术本身降档**": lightweight techniques by themselves don't cause a downgrade.
- **Semantic score:** prediction consistency 0–1, location accuracy 0–2, explanation matching image evidence 0–2. Saying "uncertain" explicitly is allowed, and **"无证据的因果断言不得作为加分依据"**: causal claims without evidence earn nothing.
- The link-matching criterion now names only 极目 / TAT, not VQA.

The "需量化、裁剪、分批 → Tier ②" wording in your question doesn't appear in the 11-05 rubric (line 329); it most likely comes from the older 10-25 file. Under the current text, NF4 and `chunk=2` carry **zero** downgrade risk. The real risk is **headroom**.

## 1. S_cost: should we drop `p1_qy`?

**Recommendation: keep `p1_qy`, but cut the peak and measure it on a real 4090.**

- **Headroom risk:** 21.16 GiB peak out of 24 GiB is 88%, measured on a **H200**. The 4090's own CUDA context and allocator behaviour differ, so the real peak there is unknown. A strict judge could call 2.8 GiB "余量不足". That is the only plausible route to Tier ② (0.5–0.6, about 10–12 pts), and an OOM would be far worse.
- **What dropping `p1_qy` actually buys:** it does not move us from 12 to 20 points. We're already Tier ① on the rubric text, so the gain is *within* the tier, roughly +1 to +3 pts and judge-dependent. The cost is about −0.25 objective, which is certain and also eats into the top-10 cutoff margin.
- **Cheaper fix:** close most of the 3.4 GiB gap between torch peak (17.74) and `nvidia-smi` peak (21.16). Options are `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, calling `empty_cache()` after load, and trying `v16_chunk=1` (it's an exact-match change, TV=0). Target `nvidia-smi` ≤ 18–19 GiB on a real 4090.
- **Decision rule:** if the measured 4090 peak is above about 22 GiB, or the P95 latency is above about 12 s, *then* drop `p1_qy`. Otherwise keep it. Also document a "lite profile" without `p1_qy` (measured VRAM and score, number of parallel workers) as a disclosed scaling option, but submit and reproduce exactly one model.

## 2. S_deploy: what's missing (worst first)

1. **No real 4090 measurement.** Every latency number comes from a H200; `final_engine.py:822` even prints "not a 4090". A H200 has roughly 4–5× the memory bandwidth, so the 1.98 s mean / 3.70 s P95 could plausibly become something like 6–10 s / 12–17 s on a 4090. That puts the 15 s limit at risk. This is the top item.
2. **No service at all.** A grep across `code/` and the docs found no FastAPI, Flask, gRPC or `http.server`. The "online single-image mode" criterion is currently unmet. Needed: a thin FastAPI `/infer` (single image) plus `/infer_batch`, a resident engine loaded once, a `/healthz` endpoint, and the cold-start `load_s` reported.
3. **Attribution and TAT output are batch-only.** They're produced after the batch (`_run_stage4_sidecars`), not per request, and their time is excluded from the reported latency. They need to move into the per-image path, with the end-to-end time measured.
4. **Inference depends on training data.** The clean-footprint calibration reads `data/manifest_combined.csv` and 24 training images at runtime (`final_engine.py:737`). It should be precomputed into a small `.npz` that ships with the weights.
5. **Unverified dependencies.** `bitsandbytes` comes from `third_party/pydeps`. Needed: a lockfile or Dockerfile, a check that the bitsandbytes binary supports sm_89, and an offline wheelhouse.
6. **The 极目 / TAT schema is our own invention** (`export_tat_jimu_payload`). It should be presented as an adapter with an explicit field-mapping table, not claimed as the official format. The video aggregator (`EffectiveFrameORVideoAggregator`) is a real plus for VQA-style chaining; keep it.

## 3. Semantic 5/5: cleanest, lowest-risk path

What we have now would likely score about 2–3/5:

- **Explanations are fixed per-class templates** (`_EVIDENCE_CUES`). Every 花屏 image says "宏块解码异常、条带撕裂或色度通道溢出" regardless of what's in it. That's weak on "evidence consistency" and close to an "assertion without evidence".
- **`reason_40ch` is a hard `[:40]` slice**, so it cuts sentences mid-phrase.
- **Localization is only validated for 花屏** (dimension 3), on synthetic glitch twins, using a *different* tower from the deployed `peg`. Pointing accuracy there was 0.81. Localization for 过度生成 and 虚化突兀 has no ground-truth check; the overgen60 audit only measures self-consistency.

Recommended design: **structured attribution with measured evidence.** Don't use free VLM text by default.

1. **Labels stay untouched.** The attribution only reads `pred_6d`, which is already true. Add a CI check that the label JSON is byte-identical with and without attribution turned on.
2. **Location:** keep the box plus nine-grid region from `defect_attribution_map`, but only emit a box when the map is concentrated; otherwise output "全图", as the code already does.
3. **Evidence:** replace the templates with numbers measured inside the box, compared against percentiles from the clean training set. Examples:
   - 过锐: edge overshoot amplitude
   - 模糊: Laplacian variance
   - 花屏: blockiness / chroma-jump index
   - 虚化突兀: sharpness jump across the matte boundary

   This reuses the `semantic_roi_meta` physics. Every claim is then checkable from the image, and causes stay explicitly uncertain, which the rubric rewards.
4. **Optional `Qwen3-VL-8B` use** (already resident): only for predicted P0 or a localized defect. Give it a constrained ≤ 16-token prompt that **only names the content inside the box** (e.g. "人物面部"). That improves the location wording without the model inventing causes. Turn it on only if the 4090 latency budget still holds.
5. **Validate before claiming 4–5:** hand-label boxes on about 60 real 过度生成 / 虚化突兀 training images and report the pointing accuracy.

## Priority order

1. Real 4090 run: peak VRAM, mean and P95 latency, cold start.
2. Allocator / peak reduction.
3. FastAPI online + batch service, with attribution inside the timed per-image path.
4. Measured-evidence attribution plus the real-image localization check.
5. Dependency lock and Docker image, and the adapter mapping doc.

I haven't run anything on a 4090 or built any of the above yet. I couldn't get permission to run `nvidia-smi` here, so I don't know whether this machine has a 4090. Item 1 needs one.
