# Research & Technical Debate: Untested 2025–2026 Vision Encoders / ViT Modules for NR-IQA

The user asks: beyond the vision encoders we have already used/tested, what **never-used 2025–2026 vision encoders or VLM ViT modules** are genuinely worth trying for our 6-class NR-IQA BadCase task (`模糊`, `纹理损失`, `过度生成`, `虚化突兀`, `过锐`, `花屏`), especially if we use **standalone encoders + custom heads** rather than huge VLM LLMs?

## 1. Encoders ALREADY Used or Tested in Our Repo (DO NOT re-recommend!)
- **In `resident_v6_core` (shipped 4090)**: `C-RADIOv4-H` (`cradio_v3`), `PE-Core-L14-336/448` (`peg`, `ntg_pe`, `p12_pe`), `InternViT-6B` (`v16`), `Qwen3-VL-8B` ViT (`p1_qy`).
- **Also trained/tested & dropped**: `SigLIP2-SO400M-NaFlex` (`sig_v3`), `SigLIP2-Giant` (`p12_siggiant`), `PE-Core-G14-448` (`p12_g14_kd`, 1.88B), `DINOv3-ViT-L16` (`ntg_dinov3_e6`, failed B1 screen), `ConvNeXt-Large` (`ntg_convnext_e8`, failed B1), `InternVL3_5-38B` (`InternViT-6B`), `Qwen3-VL-32B` ViT, `Clef / Clef-Flash`.
- **Present in `/wanqing-develop/luowenjing/18/models/` or `/wanqing-models/` (check if any are untested)**: `eupe-vit-b`, `flux3-klein` / `FLUX.2-klein-4B`, `JEV-27B-VL`, `JEV-Gemma4-26B-A4B`, `nemotron-omni`, `Qwen3.5-27B`, `Qwen3.6-27B`, `Qwen3.8-27B`, `gemma-3-27b-it`.

## 2. 2025–2026 Candidates Found via Web Search
Please critically evaluate these **untested** families specifically for **low-level perceptual distortion & AI-artifact detection** (not generic ImageNet/VQA!), and tell us which ones have a **genuinely different inductive bias** from `PE-Core / SigLIP2 / C-RADIOv4 / InternViT-6B / DINOv3`:

1. **Pixel-Reconstruction / Autoregressive-Pixel Encoders (0% representation in our current pool!)**:
   - **Apple `AIMv2`** (`apple/aimv2-large-patch14-448` or `aimv2-1B`, multimodal autoregressive pretraining predicting **raw image patches/pixels** + text).
   - **Meta `WebSSL-MAE`** (`facebook/webssl-mae1b-full2b-224` / `mae3b`, pure pixel-reconstruction MAE on 2B web images, language-free).
   - **Diffusion / Flow-Matching Backbones** (e.g., **`FLUX.2-klein-4B` / `SD3.5`** intermediate UNet/DiT features or VAE latent residuals — already locally present as `FLUX.2-klein-4B` / `flux3-klein`! Could generative-model features specifically catch `过度生成` and `纹理损失`?).
2. **Dense / Spatial / Hierarchical Encoders**:
   - **Meta `PE-Spatial`** (`facebook/PE-Spatial-L14-448` / `G14-448`, surfaces intermediate-layer dense spatial features of PE rather than final contrastive layer).
   - **Meta `V-JEPA 2.1`** (`vjepa2.1-vitl/vitg`, dense predictive loss + deep intermediate-layer self-supervision).
   - **Meta `SAM 3` Perception Encoder** (Hiera hierarchical multi-scale ViT, late 2025/2026).
3. **2025–2026 Native-Resolution VLM ViT Modules (used standalone without LLM)**:
   - **`MoonViT` / `MoonViT-V2`** (from `Kimi-VL`, NaViT native-res packing + 2D RoPE, ~400M — initialized from SigLIP-SO400M).
   - **`SeedViT`** (from `Seed1.5-VL`, 532M native-resolution ViT).
   - **`Gemma-4` ViT** (550M–570M 27-layer ViT with axial 2D RoPE + NaFlex) / **`Qwen3.5 / Qwen3.6` ViT** (416M+ ViT).
4. **Low-Level / Restoration Pretrained Encoders**:
   - **`GenLV`** (`arXiv:2507`, trained on 100+ low-level restoration/enhancement tasks) or **`MetaCLIP 2`**.

## Questions to Answer & Synthesize
1. **Ruthless Filtering**: Which of the above are *pseudo-new* (just another contrastive ViT like `PE-Core`/`SigLIP2` that will duplicate what we already have, e.g., `MoonViT` initialized from SigLIP-SO400M, `MetaCLIP 2`, `Qwen3.5/3.6` ViT)?
2. **Top 3–4 Genuinely Orthogonal Encoders Worth Testing**: Rank the top candidates by expected marginal gain on our 6 NR-IQA defects (especially `过度生成`, `虚化突兀`, `纹理损失`, `过锐`), noting exact HF/local model IDs, native resolution support, VRAM/latency on 4090, and *which* existing slot or defect each targets.
3. **Reality Check**: Given that `32B+38B` only added `-0.00033` RPS and `pe_core_g14` + `p12_pe_kd` added `+0.002 Doc2S` on our 3,573-image dataset, be honest about the realistic probability of *any* new encoder moving `Doc2S` above the noise floor without more labelled data.
