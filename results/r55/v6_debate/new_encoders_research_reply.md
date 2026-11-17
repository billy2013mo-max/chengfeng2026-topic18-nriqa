# Which 2025–2026 encoders are worth testing for the 6-class NR-IQA task

**Short answer:** most of the list will duplicate what the pool already has. Three candidates have a genuinely different inductive bias. The chance that any of them moves Doc2S above the noise floor is low, about 10–25%. I'd test them only with a cheap frozen-feature screen and a fixed stop rule.

## 0. Corrections to the brief (checked in the repo)

- **We have already tested pixel- and generative-prior evidence.** E2 (`code/r55/probe_ae_recon.py`, `logs/r55/probes/ae_recon_eval.log`) ran FLUX-AE reconstruction error at native resolution on all 3,573 images. On its own it reached 过度生成 detection AUC 0.766, but adjacent-grade AUC was only 0.62/0.60. Added to the pool it made 过度生成 RPS worse (+0.0016, CI excludes 0) and changed nothing on any other defect. `sr_idem`, `iqa_bank`/`iqa_bank2`, `mos_expert` and FGResQ probes also exist and were null. The "0% representation" claim in the brief is wrong.
- **`eupe-vit-b` is not untested.** It is wired into `code/finetune/models.py:69` and dozens of training scripts.
- **Intermediate-layer PE features were already tried.** `--pe-taps` (`etaps`) came out harmful in the 2026-11-14 screen. That weakens the case for PE-Spatial in particular.
- I could not verify a few named models from memory: "V-JEPA 2.1", "GenLV (arXiv:2507)", the exact Gemma-4 ViT specs, and small PE-Spatial-L sizes. Check those IDs before anyone spends GPU time on them.

## 1. Ruthless filter: pseudo-new candidates (skip)

| Candidate | Why it duplicates what we have |
|---|---|
| MoonViT / MoonViT-V2 | Initialised from SigLIP-SO400M; we already tested SigLIP2-NaFlex and SigLIP2-Giant. |
| Qwen3.5 / 3.6 ViT | The Qwen3-VL ViT line is SigLIP2-derived; we already have the 8B ViT (shipped) and the 32B ViT (null). |
| Gemma-4 ViT | The Gemma line uses SigLIP-family vision towers; same contrastive bias. |
| MetaCLIP 2 | Plain CLIP contrastive ViT-H, a weaker relative of PE-Core. |
| SAM 3 encoder | It is a Perception Encoder backbone, not Hiera (SAM 2 was the Hiera one). C-RADIOv4 already distils SAM-style dense features. |
| PE-Spatial | Distilled from PE-Core-G intermediate layers plus SAM masks. Both halves are covered (C-RADIO, plus PE taps that were harmful). |
| V-JEPA 2.x | Predicts in latent space, so by design it discards pixel-level detail: the wrong bias for blur, texture and sharpening. Video-oriented at about 256 px. |
| SeedViT | Not released as open weights, so we can't use it. |
| GenLV | Unverified. Restoration-trained priors also look a lot like the null `sr_idem` / IQA-bank probes. |

## 2. Candidates with a genuinely different inductive bias, ranked

1. **Apple AIMv2**: `apple/aimv2-large-patch14-448` (~0.3B), `apple/aimv2-large-patch14-native` (native resolution), `apple/aimv2-1B-patch14-448`.
   - **Why it's different:** it is pretrained by autoregressively regressing normalised pixel patches plus text, so its features keep high-frequency information that contrastive towers throw away.
   - **Targets:** 纹理损失, 过锐 and 模糊 grading. A natural fit for the Spec B / Spec C texture and sharpness slot.
   - **On a 4090:** L/448 is about 1k tokens, roughly 1–2 GB bf16 and a few ms per view. 1B/448 is about 3 GB. Both are cheap.
   - **Assessment:** the best bet of the three, but the prior is still low.
2. **Meta WebSSL-MAE**: `facebook/webssl-mae1b-full2b-224`, or the 300M/700M variants.
   - **Why it's different:** pure pixel reconstruction with no language. It is the most pixel-faithful representation on this list.
   - **Problems:** it is trained at 224 px, so you must interpolate position embeddings to 448 or tile it. MAE features are poor frozen and need fine-tuning, which is risky on 3.5k images (we already saw a PE-G-style clean collapse).
   - **On a 4090:** 1B at 448 is fine (about 1k tokens).
   - **Targets:** 纹理损失 and 过锐.
3. **FLUX.2-klein DiT intermediate features at small noise levels** (already local as `flux3-klein`). This is a probe, not a backbone.
   - **Why it's different:** it uses the denoiser's internal features, not the AE reconstruction error that E2 measured. Its only target is 过度生成.
   - **Prior:** low, because E2 already showed the generative-manifold signal is weak and mostly in the pool.
   - **Cost:** about 8 GB bf16 and tens of ms per image per timestep on a 4090. That is a real latency cost for the deliverable.
   - **Rule:** run it only if (1) or (2) shows anything at all.
4. **Optional, not an encoder: a depth-versus-blur consistency feature for 虚化突兀.** This would be monocular depth plus a local blur map at the mask boundary. It is the one defect where explicit geometry is different evidence from appearance features. A cheap probe could reuse the masks from `build_bokeh_seg.py`.

**Architecture rule:** any winner has to replace or augment one of the four Spec heads on top of InternViT-6B, per AGENTS.md rule 3. It cannot be a fifth tower stacked on.

## 3. Reality check

- **Every new evidence source so far has been null:** 32B+38B ViTs (−0.00033 RPS), PE-G14 + KD (+0.002 Doc2S), FLUX-AE, FGResQ, the IQA and MOS banks, KVQ, and synthetic data apart from 花屏 robustness. The 2026-11-12 global stop rule was triggered for this reason.
- **The bottleneck is grading, not detection.** Every family sits at about 0.70–0.75 adjacent-grade AUC on 过度生成. That pattern points to a label-noise or data ceiling, not missing features. A new encoder adds features; it doesn't add labels.
- **My estimate:** about 10% per candidate, and about 20% that at least one of the three passes a 3-control-seed gate. If one does pass, expect +0.1 to +0.3 Doc2S, not points.

**How I'd run it:**
1. Extract frozen features for AIMv2-L-448 and WebSSL-MAE-1B on all 3,573 images, about 1–2 GPU-hours each.
2. Evaluate them with the same nested add-to-pool protocol as E2, on the honest `v16_qy` OOF.
3. Use a pre-registered gate: per-defect ΔRPS < 0 with CI90 excluding 0, averaged over 3 control seeds. For 过度生成, also require adjacent-grade AUC ≥ +0.015.
4. If both fail, stop the encoder direction completely and skip the FLUX-DiT probe.
5. Only a candidate that passes earns a fine-tuned Spec-slot run.
