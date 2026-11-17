# Strategy Alignment with Fable 5.2: Self-Created Data & Better Utilization of the 3,573 Images (Zero New Human Labels)

The user states a hard constraint:
> "我们现在不可能还有真实标注数据，只能我们自己造或者更好的利用这些数据，这块的方案你跟claude对一下"

Let's align ruthlessly on what is **actually viable vs dead-end** when we have **ZERO new human labels**, given everything we already tested and proved in this repo:

## What We Already Tested & Closed
1. **External unlabelled IQA data (`KVQ` 5.6k, `KonIQ/Inter4K` 4k)**: `kvq`, `shiftkvq`, `kvqcert` ALL failed (`shiftkvq` +0.0033 harm on unseen pipelines because external datasets lack Kuaishou-specific `过度生成`/`虚化突兀`).
2. **External synthetic / generative data**:
   - `glitch2` (`peg`, corrupted bitstream + matched clean decodes): **SUCCEEDED** (`0.885 -> 0.9999 AUC` on unseen source) and is already shipped in `resident_v6_core`.
   - `og` (FLUX re-render), `ogsr1` (Real-ESRGAN/BSRGAN + bicubic twins), `bkseg1` (cut-out bokeh, though note `bkseg1` accidentally kept `过锐`/`过度生成` labels from non-clean seeds): ALL **FAILED** (`HARM` or `NULL`).
   - `benign1`, `baug`, `tw-benign1`, `shiftrc` (mild transcode / tone / random crops): ALL **FAILED** (`HARM` on unseen pipelines).
3. **Model/Encoder/KD/RL scaling on the 3,573 images**:
   - `Qwen3-VL-32B` + `InternVL3_5-38B`, `PE-G14 + KD`, `AIMv2-L/1B`, `WebSSL-MAE-1B`, `FLUX.2-klein-4B DiT`, `E1 RL`: ALL either failed gate or gave `<= 0.0003` marginal RPS (`< 0.3 Doc2S`).

## Questions to Align On (Be Brutally Honest, No Wishful Thinking)
Given **only** the 3,573 labelled images (no new human labels, no access to Kuaishou's internal production overgen operator), please evaluate:

1. **Can we create NEW synthetic data that actually passes our 3-control-seed gate?**
   - Specifically: what about fixing `bkseg1` to use **Clean-only seeds** (`y == 0` on all 6 dims, so zero conflicting `过锐`/`过度生成` labels)? Worth a 1-shot test for `虚化突兀` or not?
   - What about **in-domain local patch / defect-preserving CutMix** (e.g., pasting a verified defective patch onto a Clean image, or pairing Clean+Defective crops) vs **cross-video ordinal ranking pairs**?
   - For `过度生成`: without Kuaishou's actual production enhancement operator, is there *any* synthetic recipe worth trying, or should we declare synthetic `过度生成` dead?

2. **Can we make BETTER USE of the existing 3,573 labelled images?**
   - **Cross-fitted OOF soft-target denoising / label smoothing** on the 3,573 rows (using `resident_v6` 9m OOF or `dual_32b_38b`? Wait: we saw `p12_pe_kd` and `p12_g14_kd` gave `+0.002 Doc2S` in `resident_v6 + kd`, and `xd` helped single towers only). Is there *any* way to squeeze more out of the 3,573 labels (e.g. **disagreement-sample downweighting / focal-on-clean-errors / multi-view TTA at inference**)?
   - **Inference-time / Multi-view utilization** (without changing training data): e.g., currently `final_engine.py` uses 2x h-flip TTA; what about 4-view / 5-crop TTA on 4090 within the 15s/img budget?

3. **Bottom-Line Verdict**:
   Give a clear ranked list of **(a) 0–2 concrete, low-cost data/utilization experiments still worth running (if any)**, vs **(b) whether the honest engineering conclusion is that the 3,573-image ceiling (~51.9–52.2 Doc2S_orig) is already reached with `resident_v6_core` (51.873)**, so further data-hacking without new labels has <15% chance of beating noise.
