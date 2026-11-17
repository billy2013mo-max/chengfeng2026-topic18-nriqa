# Model selection and architecture levers for local defect detection and grading (single RTX 4090, ~9 GB free)

Scope note. "Project" findings come from the repository (authoritative for what was tried). Web findings are cited inline. Anything without a citation or a repo path is marked *speculation* or put under Gaps. Objective = Doc2_Strict (65 pts). OOF = nested out-of-fold on 3,573 labelled frames. RPS = ranked probability score of pooled probabilities, lower is better.

---

## Q1. Which backbones / families could add information that is orthogonal to RADIO / PE / SigLIP2, and what does the evidence say about diversity gains?

### Takeaway
Most "different" backbones people propose are already in the pool, inside V16 and Phoenix p3: ConvNeXt-V2 (FCMAE), DINOv3-L, EVA02, PE-Core-G and C-RADIO, plus Haar-wavelet and physics maps. Families that use a different input or pretraining *domain* are still untried as fine-tuned members: high-pass / forensic streams on uncropped pixels, restoration-pretrained encoders and MAE-style pixel-reconstruction ViTs. The literature says diversity of training method decorrelates errors more than diversity of architecture. The project's own evidence is sobering, though. All families share 61 % of their errors and have the same per-defect skill, and every frozen "new evidence" bank tested was null. Realistic gain per new family: 0 to +0.3 objective points, and probably nearer the bottom of that range. This is speculation anchored on the one measured +0.41.

### Cited Findings
**Project (repo) — what is already in the ensemble**
- V16 (`v16_full8`) is itself a 5-backbone fusion: `pecoreg:lastk4, dinov3l:lastk6, cradio:lastk4, eva02:lastk6, convnext:full` at 448 px with `tiles=2` (2×2 native quads) plus 3 one-to-one native "microscope" crops and 60-D physics calipers — `code/finetune/launch_v16_fulldata_8gpus.py` L23-25, `code/finetune/train_v16_microscope_specialist.py` docstring. Measured 3.305B parameters — `docs/reports/R55_REVIEW_2026-11-11.md` L198-206, L260.
- ConvNeXt-V2 in V16 is `convnextv2_base.fcmae_ft_in22k_in1k` (MAE-style fully-convolutional pretraining) — `code/finetune/models.py` L76-78.
- Phoenix p3 = ConvNeXtV2-Base + C-RADIO + PE-Core-L + DINOv3 with a learned backbone gate, a 4-view 1:1 "contrast-residual microscope", 1:1 Haar-wavelet high-frequency subbands / artifact maps and a Depth-Anything-S defocus (CoC) probe — `code/phoenix/train_pillar3_hires_physics.py` docstring; `code/phoenix/native_physics_and_views.py` L83-108; `code/r55/final_engine.py` L137-151.
- So convolutional, MAE-pretrained, DINO-family, EVA and wavelet-domain inputs are **already represented**. Not represented: restoration-trained encoders (SwinIR / Restormer / NAFNet), forensic / AIGC-detector streams (NPR, SRM, DCT patch selection) and Swin-V2 or MAE-ViT trunks (grep of `code/` finds no fine-tuned member of these kinds).

**Project — prior results on backbones**
- Tier-1 8-fold single-tower OOF (Doc1 / Doc2): cradio 51.70/49.50, pe 50.39/49.70, siglip 51.30/48.73, convnext 49.96/47.41, dinov3 48.26/47.26 — `NEXT55_HANDOFF.md` §3. Leave-one-out from the 6-mean: convnextv2_base +0.06 (slightly harmful), dinov3 +0.14 (harmful) — `NEXT55_HANDOFF.md` §4.1.
- PE-Core-G: 1/10 runs clean-collapse to 42.49; rejected. DINOv3: negative. PE-Core-L vs fixed base: only +0.16 — memory `backbone-screen-pe-core-dinov3.md`.
- InternViT-6B: best frozen linear probe in the 14-backbone screen (mAP14 0.779; 过锐_P0 AP 0.712 vs SigLIP2 0.585) — `docs/OVERNIGHT_AGENT_HANDOFF_PLAN.md` L90-96. Fine-tuned legacy 6B towers on top of the 8-fold ensemble gave +0.08 Doc2 (v11) / −0.42 (v12fix), "6B adds ≈0 on top of the ensemble" — `NEXT55_HANDOFF.md` §3, §4.3.
- Frozen-feature screen files exist for clip-vit-large, convnextv2-base, dinov2-base/large, eupe-vit-b, eva02-base, internvit-6b, pe-core-g, metaclip2, siglip2 and c-radio (`feats/`, `code/extract_feats.py`). Single-split frozen-encoder results (`results/enc_split_*.json`): val totals cluster at 41-44 (convnextv2-base 40.98, eupe 41.49, siglip2 41.02, c-radio 43.66, internvit-6b 44.21).
- Ensemble structure: 61 % of the pool's cell errors are wrong in all three families. Per-defect skill is the same for every family (过度生成 adjacent-grade AUC 0.70-0.75 for all seven members). Per-defect family weights gain ΔRPS −0.00004. "More families of the same kind buy variance reduction only (≈−0.0004 RPS each, falling)" — `docs/REASSESSMENT_AND_PLAN_2026-11-12.md` §2 item 2.
- The only modelling step that ever helped was a new family (Phoenix): cross-family family-balanced ensemble RPS 0.04298 → 0.04192, OOF 51.04 → 51.45 — same doc §1.
- Composition search over all 28 models with complete OOF (incl. ConvNeXt-V2, DINOv3, extra seeds, pillar 2, second V16): greedy to 29.6B gives −0.00032 RPS, "at most ≈+0.1 Doc2S" — same doc §8.
- Frozen "new-information" banks were all null inside the pool:
  - 19 public IQA metrics (MUSIQ×3, MANIQA×3, ARNIQA×4, HyperIQA, DBCNN, TReS, NIMA×2, PaQ-2-PiQ, NIQE, BRISQUE, PIQE): mean ΔRPS +0.0004.
  - FGResQ / A-FINE / TOPIQ×3 / UNIQUE / WaDIQaM / CNNIQA: +0.00013.
  - FLUX-AE reconstruction (AEROBLADE idea): 过度生成 ΔRPS +0.0016, worse.
  - KVQ-MOS expert: +0.0002.
  - 33 Real-ESRGAN/BSRGAN idempotence features: +0.0006.
  
  Sources: same doc §6, §9, §11.1.
- The VLM (Qwen3-VL-8B, p1) is as good as the best tower alone (RPS 0.0480) but redundant in the pool: dropping it costs −0.11 decoded (CI90 [−0.37, +0.15]) — same doc §11.8 addendum.
- Zero-shot VLM screen: going from 768² to 1024² did not help (Qwen3.8-27B mAP14 0.280 → 0.278) — `results/decision_model_screen.md`.

**Literature — diversity**
- In a study of 82 models, ensemble performance rises as error correlation falls. Models that differ more in *training methodology* (objective, data) make less correlated errors. ResNet-50 (76.5 %) + ALIGN-ZS (75.5 %) ensemble to 83.4 % (+7 %). Differently trained models learn "overlapping but non-nested" features — [Gontijo-Lopes et al., ICLR 2022](https://arxiv.org/abs/2110.12899).
- Architecture diversity produced more diversity than negative-correlation learning on every measured metric (predictions, attributions, activations) — [Rosales et al., ICCVW 2023](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/html/Rosales_Assessing_the_Impact_of_Diversity_on_the_Resilience_of_Deep_ICCVW_2023_paper.html). Diverse-architecture ensembles were more accurate and more robust to corruptions than single-architecture ensembles of comparable members — [Filipiuk & Singh, CEUR 2023](https://ceur-ws.org/Vol-3215/16.pdf).
- Quality-challenge winners use heterogeneous feature mixes:
  - NTIRE 2025 KVQ track, 1st place (SharpMind): teacher concatenates SlowFast, FAST-VQA, LIQE, DeQA, HVS edge/content features and a Swin-B, then distils to a Swin-T student.
  - 2nd place (ZQE): semantic + local-texture + global branches, plus slicing that drops low-texture slices by Laplacian density.
  - 3rd place (ECNU-SJTU): dual Swin-T, one semantic branch at 384 px and one FAST-VQA-style 32×32-fragment branch.
  - 5th place (TenVQA): a single ConvNeXtV2-Tiny at 576 crops.
  - No ablation isolates the ensemble gain.
  
  Source: [NTIRE 2025 Short-form UGC VQA & Enhancement report](https://arxiv.org/html/2504.13131v1).
- NTIRE 2024 AIGC quality, image track: 2nd place ensembles ConvNeXt + EVA02 backbones; 3rd place fuses ConvNeXt + ViT vision features with CLIP text encoders — [NTIRE 2024 AIGC QA report](https://arxiv.org/html/2404.16687v1).

**Literature — candidate orthogonal encoders**
- NPR (CVPR 2024): upsampling in GANs and diffusion models leaves local neighbouring-pixel dependencies. A lightweight CNN on neighbouring-pixel differences generalises across 28 generators when trained on one (ProGAN). The headline gain is reported inconsistently as 11.6 % / 12.8 % — [Tan et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html).
- Diffusion and VQ-GAN images show Fourier-domain artifacts and differ from real images mainly in mid-high frequencies. Detectors trained on one architecture often fail on others — [Corvi et al., ICASSP 2023 / "Intriguing properties"](https://arxiv.org/abs/2304.06408v2).
- SAFE: replacing resize with crop (random crop in training, centre crop at test) avoids distorting subtle artifacts. A DWT high-frequency stream plus ColorJitter / rotation gives +4.5 % accuracy / +2.9 % AP across 26 generators — [Li et al., arXiv 2408.06741](https://arxiv.org/html/2408.06741v2).
- AIDE scores patches by DCT, takes the top-k highest- and lowest-frequency patches through SRM filters, and concatenates them with CLIP semantics. Reported +3.5 % / +4.6 % on AIGCDetectBenchmark / GenImage. On the hard "Chameleon" set all 9 off-the-shelf detectors dropped sharply — [Yan et al., arXiv 2406.19435](https://arxiv.org/html/2406.19435v2). Details of k and patch size come from secondary reviews.
- DeSRA (ICML 2023) locates GAN-SR "inference artifacts" by comparing local variance between the MSE-SR and GAN-SR outputs. These artifacts are out of the training distribution — [Xie et al., ICML 2023](https://proceedings.mlr.press/v202/xie23c/xie23c.pdf).
- DA-CLIP: a trainable controller on frozen CLIP outputs a degradation embedding that classifies 10 degradation types accurately — [Luo et al., ICLR 2024](https://arxiv.org/html/2310.01018v2).
- ConvNeXt V2 (FCMAE + GRN): Atto 3.7M to Huge 650M params, Huge 88.9 % ImageNet top-1 with public data only — [Woo et al., arXiv 2301.00808](https://arxiv.org/abs/2301.00808).
- Swin-V2 is used as a frozen or fine-tuned IQA/VQA backbone with results "comparable" to UNIQUE/TReS. ConvNeXt-T vs Swin-T on IQA: close, with Swin slightly ahead — [arXiv 2401.08522](https://arxiv.org/abs/2401.08522v1); [arXiv 2309.14868](https://arxiv.org/pdf/2309.14868).

### Inferences
- **Ranked candidate list.** Expected Δ is in objective points after decoding. All Δ figures are *speculation*, bounded by the project's +0.41 (first new family) and ≈+0.1 (greedy 28-model composition).

  | Rank | Option | Why it may be orthogonal | Prior project result | Expected Δ | Added VRAM (bf16, batch 1) | Added latency on 4090 | Risk |
  |---|---|---|---|---|---|---|---|
  | 1 | **Forensic / high-pass native-pixel stream** trained end-to-end on our labels (NPR or SRM / DWT residual input, small CNN or ConvNeXt-V2-Atto/Nano). Uses crop-not-resize (SAFE) and DCT top/bottom-k patch selection (AIDE) over the whole frame. Targets 过度生成, 过锐 and 花屏. | Different input domain (residuals, not RGB) and objective; generative-upsampling traces are known to sit in mid-high frequencies. | Not tested as a trained member. Related nulls: frozen FLUX-AE and SR-idempotence features; p3 already sees Haar HF maps as inputs. | 0 to +0.3 | <0.5 GB weights; <1.5 GB activations with tiles | ~0.05-0.3 s | Low cost. Main risk: redundancy with p3's wavelet channel and V16 physics; weak alone (single-model accuracy may be ~45-48 Doc2). |
  | 2 | **Restoration-pretrained encoder family** (NAFNet / Restormer / SwinIR encoder, or DA-CLIP controller) fine-tuned with a grading head on 1:1 tiles. | Pretraining objective is pixel reconstruction / degradation removal, the "different training method" axis that Gontijo-Lopes found most decorrelating. | Frozen restoration "instrument" features were null (+0.0006 RPS). A fine-tuned trunk was never tried. | 0 to +0.3 | weights ≤0.2 GB; activations at native res need tiling (448 tiles: ~1-3 GB) | ~0.2-0.8 s | Medium. These encoders are not trained for semantics, so 过度生成 (anatomy / text) may need the ViT pool anyway; implementation effort for the heads. |
  | 3 | **MAE-pretrained ViT-L/H** (pixel-reconstruction ViT) or **Swin-V2-L** as a tier-1 tower using the existing guided-crop recipe. | MAE: different objective (masked pixels). Swin-V2: local-window inductive bias; KVQ winners lean on Swin. | Neither tested. Closest analogues: ConvNeXt-V2 FCMAE (harmful in the 6-mean, LOO +0.06) and DINOv3 (harmful). | 0 to +0.15 | ViT-L ~0.6 GB, ViT-H ~1.3 GB, Swin-V2-L ~0.4 GB (approx. from parameter counts, not verified here) | ~0.2-0.6 s | Medium. Prior FCMAE / DINO evidence suggests "another strong RGB trunk" behaves like the existing families. |
  | 4 | **ConvNeXt-V2-L/H at higher native resolution** as a standalone family | Convolutional at native res | ConvNeXt-V2-B already in V16 and p3; standalone tower OOF Doc2 46.81, LOO harmful | ≈0 | L ~0.4 GB, H 650M → ~1.3 GB | ~0.3-0.8 s | Low implementation risk, low value |
  | 5 | DINOv2-reg-L | SSL family | DINOv2 frozen features screened; DINOv3 rejected | ≈0 | ~0.6 GB | ~0.2 s | Low value |
  | — | InternViT-6B | strongest frozen probe | fine-tuned +0.08 / −0.42 on top | ≈0 | ~12 GB weights: **does not fit** in 9 GB without quantisation | — | Reject |
  | — | PE-Core-G as standalone | — | clean collapse; already in V16 | <0 | ~3.8 GB | — | Reject |
  | — | Another VLM | — | p1 redundant (−0.11 n.s. when dropped) | ≈0 | 5.6 GB (4-bit) / ~16 GB bf16 | ~1.5 s | Reject; costs the no-quantisation cost tier |

- **Screening rule (recommended).** A candidate earns a place only through error *decorrelation* with the pool, not single-model accuracy. The +0.41 family and the 6B "best probe, adds ≈0" case show this. Cheapest falsification:
  1. Train 2 of 8 folds of the candidate.
  2. Compute the correlation of its residuals with the pool residual, then ΔRPS inside the pool with `code/r55/stack_plus.py` / `gate_plus.py`.
  3. Use a 3-seed control (SynthLab gate rule).
  4. Proceed to 8 folds only if ΔRPS ≤ −0.0003 with CI90 excluding 0, which matches the bar of the E1 / E2 plan.
- The repeated nulls of *frozen* banks do not rule out *fine-tuned* new-domain trunks. The banks were fitted by second-stage boosting, which the project showed is itself worse than raw pooling (`REASSESSMENT` §10.2). A fine-tuned member pooled by averaging is a different test. This is the main argument that ranks 1-2 are still open; it is an inference, not evidence of a gain.

### Gaps
- No study found that compares architecture diversity against seed diversity at fixed accuracy for IQA or defect grading. The ensemble-diversity evidence is from ImageNet-style classification.
- Exact NPR / AIDE / SAFE numbers on restoration-type over-generation (as opposed to fully generated images) were not found. Whether the production enhancer leaves upsampling traces is unknown. The FLUX-AE null hints the generative-prior signal is weak.
- Parameter counts for ConvNeXt-V2-L, Swin-V2-L, MAE ViT-L/H, NAFNet and Restormer were not verified in this session (only ConvNeXt-V2-Huge 650M is cited).

---

## Q2. Native high-resolution patching / dense tiling with MIL or attention pooling vs global pooling: evidence, and the project's v12_abmil / tiles3 results

### Takeaway
The literature favours native-resolution patches for quality (MUSIQ, FAST-VQA fragments, SAFE crop-not-resize). The project has already moved most of the way there: guided 1:1 crops in every tier-1 tower, 2×2 native quads in V16, 4-view microscope in p3, and ABMIL in V12. Its direct tests show the towers' answers barely depend on which regions they see. A dense whole-frame tile/fragment stream for 过度生成 remains untested in exactly that form. Expected value is small (0 to +0.2, speculation). The best rationale is the audit finding that missed P0 over-generation consists of tiny local inventions (pseudo-glyph captions, sky squiggles).

### Cited Findings
- MUSIQ processes the native-resolution image plus aspect-preserving resized versions as fixed-size patches. It uses a hash-based 2D spatial embedding and a scale embedding, and was SOTA on PaQ-2-PiQ / SPAQ / KonIQ — [Ke et al., ICCV 2021](https://arxiv.org/abs/2108.05997).
- FAST-VQA: resizing and cropping corrupt quality information. Grid mini-patch sampling takes raw-resolution patches from a uniform grid and splices them into "fragments". Reported ~10 % accuracy gain at 1080p with 99.5 % fewer FLOPs — [Wu et al., ECCV 2022](https://arxiv.org/abs/2207.02595). A FAST-VQA-style fragment branch (32×32 from 12×12 partitions) was part of NTIRE 2025 KVQ 3rd place — [NTIRE 2025 report](https://arxiv.org/html/2504.13131v1).
- SAFE: crop instead of resize preserves artifacts for synthetic-image detection — [arXiv 2408.06741](https://arxiv.org/html/2408.06741v2).
- Project — the tested max-over-regions proxy: across the 8 random crop-set passes of two PE towers, the spread of P(过度生成) is 0.008 / 0.013 (0.03 / 0.04 on P2 frames). Max over passes vs mean gives detection AUC +0.0006 / +0.0014, none|P2 AUC +0.001 / +0.002 and unchanged grade AUCs, so "the towers' answer does not depend on which regions they see"; no tile model was built — `docs/REASSESSMENT_AND_PLAN_2026-11-12.md` §10.2.
- Project — visual audit: only 12 of 190 过度生成 P0 frames are called clean. The misses show small local inventions on otherwise normal frames, such as caption text turned into pseudo-glyphs (#1943), squiggles in a plain sky (#1800, #1802) and altered faces. P2 misses have no common cue — same doc §10.1.
- Project — V12 ABMIL (`results/v12_abmil/`): pecoreg + cradio + dinov3l with gated ABMIL, 2×2 tiles at 448 px. Old-protocol test totals per split s16-s20: 48.92 / 49.67 / 42.70 (clean collapse) / 50.14 / 48.97. A family-7 "3-layer pyramid ABMIL" (1×1 + 2×2 + 3×3 = 14 patches) was reported by an earlier agent. Legacy 6B ABMIL on top of the 8-fold ensemble gave −0.42 (v12fix) — `docs/reports/NR_IQA_Task_Status_and_Dilemma_Report.md` L222; `NEXT55_HANDOFF.md` §4.3. These used the older protocol with binary-like clean scores, so they are not comparable to Doc2 OOF.
- Project — `results/tiles3/` (eupe + siglip2 + cradio, splits s11-s15 × 2 seeds) vs `results/size448/` (siglip2 + cradio): tiles3 test totals 34.8-50.3 and size448 35.9-50.2. These are dominated by clean-score cliffs, and the backbone sets differ, so they are not a controlled tiling ablation.
- Project — the current views are a global native-aspect view plus four 448-px 1:1 guided crops chosen by physics maps (peak sharpness, MEOR halo, cut-out boundary, low-gradient texture), see `code/qingyu/dual_engine_preprocess.py`. V16 uses `tiles=2`; p3 uses `[Global, Peak0_Anatomy_Bokeh, Peak1_Halo, Mid_Interior_Tex]`.
- Zero-shot VLM: resolution 768² → 1024² made no difference — `results/decision_model_screen.md`.

### Inferences
- What remains untested is **coverage**, not resolution. Four guided crops cover a small fraction of a 1080p frame, while a 12×12 fragment mosaic or 3×3-4×4 1:1 tiles cover all of it.
  - The random crop-set result argues against large gains. It measured variation across *random* crop sets of a model trained on guided crops, and the spread was tiny.
  - A model *trained* with whole-frame tile MIL (gated attention + top-k pooling) is a different hypothesis. It is worth one cheap screen for 过度生成 P0 recall only.
  - Expected 0 to +0.2 (speculation).
- Implementation suggestion:
  - Use an existing trunk (pe_core_large, 0.33B) on 9-16 native 448 tiles at batch 1 in bf16.
  - Pool with a gated ABMIL head plus top-k. Add a separate tile-level auxiliary loss? Not possible, since there are no tile labels; rely on MIL.
  - VRAM ≈0.65 GB weights + ~1-2 GB activations. Latency is roughly 9-16× one 448 forward of a 0.3B ViT. This is *speculation*, about 0.2-0.6 s on a 4090; to be measured.
- Note: as a separate family it would be "same backbone, different view". The project's twins / second-seed evidence says same-trunk variants add ≈0 after decoding (`REASSESSMENT` §10.8, §11.6). A different trunk (rank 1-2 in Q1) on the dense tiles combines both levers and is the more promising test.

### Gaps
- No found study measures MIL tile pooling vs global pooling specifically for small-area hallucination in restored or enhanced frames.
- No controlled tiles-vs-no-tiles ablation exists in the repo under the current Doc2 OOF protocol.

---

## Q3. Frequency-domain branches (FFT / DCT / wavelet) for blur, over-sharpening halos, compression and glitch: evidence of gains on top of strong ViTs

### Takeaway
Frequency streams have strong evidence in synthetic-image forensics (AIDE, SAFE, NPR, Corvi). No published blind-IQA ablation was found that isolates a frequency branch's gain over a strong spatial transformer. In the project, wavelet / physics inputs already exist in p3 and V16, and 花屏 is solved (F1 0.98). The open niche is a *learned* high-pass stream for 过度生成 / 过锐 rather than hand-crafted maps.

### Cited Findings
- Corvi et al.: diffusion / VQ-GAN images show Fourier artifacts and differ in mid-high frequency content. Single-generator detectors generalise poorly — [arXiv 2304.06408](https://arxiv.org/abs/2304.06408v2).
- SAFE uses a DWT high-frequency stream as one of three components — [arXiv 2408.06741](https://arxiv.org/html/2408.06741v2). AIDE uses DCT-scored patches plus SRM noise filters, fused with CLIP — [arXiv 2406.19435](https://arxiv.org/html/2406.19435v2). NPR uses neighbouring-pixel differences — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html).
- A frequency-decomposition ablation exists only in restoration (Freqformer demoiréing): full model 25.26 vs Block-DCT 23.22 vs Haar-DWT 23.01, metric unnamed in the snippet — [arXiv 2505.19120](https://arxiv.org/pdf/2505.19120). This is not IQA.
- Project:
  - p3 computes 1:1 2D Haar HF subbands (LH, HL, HH) and artifact maps as inputs — `code/phoenix/native_physics_and_views.py` L83-108.
  - The current 9-D physics maps drive crop selection.
  - The 19 public IQA metrics (incl. NIQE / BRISQUE, which are spatial-statistics models) were informative alone (detection AUC 模糊 0.91, 过锐 0.85, 过度生成 0.86) but added nothing to the pool — `REASSESSMENT` §6.
  - Simulated 花屏 already fixed unseen-source glitch detection (AUC 0.885 → 0.9999) — same doc §6.
  - The behavioural probe found blur, codec loss and sharpening detected in order — same doc §9.

### Inferences
- For 模糊, 纹理损失 and 过锐 the project shows data saturation (+0.003 to +0.017 F1 per label doubling, §2 item 4). Frequency inputs carry information the pool already has (the IQA bank and physics nulls), so expect ≈0 there.
- The only defect where a frequency or forensic stream plausibly adds something is 过度生成, the biggest loss at 3.55 points. The forensic literature is about generator traces, which is closest to "AI-restoration hallucinated texture". Fold this into candidate 1 of Q1 rather than building a frequency branch inside an existing ViT tower: same-tower changes vanish in the pool.
- *Speculation*: 0 to +0.2.

### Gaps
- No BIQA paper was found with a clean ablation of an FFT / DCT branch on top of a strong pretrained ViT.
- Whether the production enhancer is GAN-, diffusion- or regression-based is unknown. This determines whether upsampling-trace detectors apply.

---

## Q4. Cross-view attention vs late averaging; two-stage cascades decoupling detection from grading; what is realistic with ~1.9-3.6k labels?

### Takeaway
With these label counts, every learned fusion layer the project tried was worse than plain averaging. V16 and p3 already implement within-model cross-view contrast residuals. Decoding already acts as a calibrated second stage with at most ≈0.4 head-room. A cascade cannot fix one-grade-off errors without more grade labels. Neither lever is recommended; expected Δ ≤ 0.

### Cited Findings
- Project:
  - Learned stacking (per-dimension multinomial LR, nested): −0.11 / −0.32 / −1.12 vs mean at C = 0.005 / 0.05 / 0.5 — `NEXT55_HANDOFF.md` §4.3.
  - Boosting on all 24 pooled probabilities: worse on all six defects. Linear cross-defect stage: +0.0031 RPS (worse). Pooling rules: arithmetic beats geometric (+0.0008), log-odds (+0.0009) and others — `REASSESSMENT` §10.2, §11.1.
  - Per-image expert gating has "nothing learnable in the disagreement"; an oracle family picker would score 56.9 — same doc §2, §4.
  - Decoder in-sample 51.87 vs nested 51.45 vs argmax 51.28: ≤0.4 head-room — same doc §11.1.
  - 78-92 % of errors are one grade off — same doc §2 item 5.
  - Within-model cross-view fusion exists: V16 `r_edge_glob = edge_max − g_view`, `r_edge_tex`; p3 6×6 contrast-residual tokens and learned backbone softmax gate — training-script docstrings.
- Literature: in transformer BIQA, two complementary branches (error map + quality token) fused inside one model beat either alone — [arXiv 2305.09353](https://arxiv.org/pdf/2305.09353). KVQ 2nd place fused semantic / texture / global branches with attention inside one model — [NTIRE 2025 report](https://arxiv.org/html/2504.13131v1). Neither compares this with late averaging of separate models.

### Inferences
- Cross-view attention belongs *inside* a new member, as V16 and p3 already do. It should not be a new fusion layer across members: at 1.9-3.6k labels any fusion layer over members has been worse than averaging.
- A detection → grading cascade is in effect what the cumulative decoder does. The binding errors are P1/P2 boundaries, which the label-saturation evidence ties to label noise or rarity. No architectural decoupling creates grade information. Δ ≤ 0 (inference from project evidence).

### Gaps
- No literature found on cascades vs joint ordinal heads at this data scale for multi-defect grading.

---

## Q5. VRAM / latency budget for top candidates (RTX 4090, batch 1, bf16, no quantisation)

### Takeaway
The fallback engine leaves about 9 GB (24 − 14.8 GB peak). All ranked candidates except InternViT-6B and a bf16 VLM fit comfortably, since weights are ≤1.3 GB each. The binding constraint is more likely **latency on a real 4090**, which has never been measured: 3.7 s mean / 6.7 s max was measured on a shared H200.

### Cited Findings
- Fallback (cradio + pe + siglip2 + v16_full8 + p3): resident 13.6 GB, peak 14.8 GB, 3.7 s mean / 6.6-6.7 s max per image. Per-member mean: cradio 0.52 s, pe 0.31 s, siglip 0.58 s, v16 1.10 s, p3 1.13 s. Cumulative resident: p3 3.96, +cradio 5.33, +pe 5.98, +sig 6.86, +v16 13.60 GB — `results/r55/final/fallback_bench_half.json`. Measured on an H200, "no run on an actual 24 GB card" — `docs/reports/R55_REVIEW_2026-11-11.md` L276-282.
- Measured weights-to-memory ratio ≈2 bytes/param in bf16: pe_core_large 0.332B → +0.65 GB, cradio 0.68B → +1.37 GB — same files.
- The primary engine with the 4-bit VLM: 20.4 GB peak, 5.5 s mean. The cost score favours no quantisation (0.7-1.0 vs 0.3-0.6 of 20 points) — `REASSESSMENT` §11.8.
- ConvNeXt-V2-Huge 650M params — [arXiv 2301.00808](https://arxiv.org/abs/2301.00808).

### Inferences
Estimates; *speculation* except where noted. Weights use 2 B/param. Parameter counts other than ConvNeXt-V2-H are recalled, not verified this session.

| Candidate | Params (approx.) | Weights bf16 | Activation peak, batch 1 | Est. 4090 latency incl. 2× flip TTA | Fits 9 GB? |
|---|---|---|---|---|---|
| Forensic high-pass CNN (NPR-style / ConvNeXt-V2-Nano) on 16 native tiles | 1-15M | <0.05 GB | ~0.5-1 GB | 0.05-0.2 s | yes |
| Restoration encoder (NAFNet / Restormer) + head, 9-16 × 448 tiles | 25-70M | ~0.1 GB | ~1-3 GB (full-res conv features; tile to bound) | 0.2-0.8 s | yes |
| MAE ViT-L / Swin-V2-L tower, global + 4 crops | ~0.2-0.3B | ~0.4-0.6 GB | ~0.5-1 GB | 0.2-0.5 s | yes |
| MAE ViT-H / ConvNeXt-V2-H tower | 0.63-0.65B | ~1.3 GB | ~1-2 GB | 0.4-0.9 s | yes |
| PE-Core-L dense-tile MIL (16 × 448) | 0.33B | 0.65 GB (measured ratio) | ~1-2 GB | 0.3-0.6 s | yes |
| InternViT-6B | ~5.9B | ~12 GB | — | — | **no** (needs quantisation) |
| Qwen3-VL-8B bf16 | ~8.8B (p1 measured 8.77B) | ~16-17.5 GB | — | — | **no** |

- Total parameter rule (≤30B trained): fallback is ≈6.2B (`r55-review` memory), so any candidate above stays well under.
- Before adding anything, measure fallback latency on a real 4090. If the H200-to-4090 slowdown is 2-3× for these small-batch, partly CPU-bound paths (unmeasured), fallback would sit at about 7-11 s mean with the max near or above 15 s. That would leave little room for a sixth member. Physics preprocessing and CoC depth run per image on the fly.

### Gaps
- No 4090 measurement exists for the fallback engine or any candidate. All latency numbers above are extrapolations.
- Activation memory at native 1080p for restoration encoders was not looked up and depends on tiling choice.
