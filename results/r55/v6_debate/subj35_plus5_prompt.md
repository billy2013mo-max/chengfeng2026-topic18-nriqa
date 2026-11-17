# Subjective (35 pts) + Semantic Bonus (5 pts) Adversarial Audit with Fable 5.2

We have locked the objective track (`resident_v6_core`, honest 8-fold nested `Doc2S_orig=51.873`, `15.77 GB` resident / `21.16 GiB` `nvidia-smi` peak, `1.98s/img` on H200). Now the user wants to **maximize all 35 subjective points (`S_deploy` 15 + `S_cost` 20) + the 5-point `Semantic Score` bonus = 40 points**.

Be the **harshest possible competition judge** reading Appendix 2 of the official rules:

## Official Rubric (Appendix 2)
1. **`S_deploy` (15 pts, normalized 0–1 × 15)**:
   - ① Single-GPU end-to-end latency (incl. pre/post-processing) `<= 15s` per image/pair;
   - ② Format compatibility with existing QA pipelines (`极目 / TAT / VQA`);
   - ③ Engineering modification cost: serviceization, dependencies, both batch and online single-image modes.
2. **`S_cost` (20 pts, normalized 0–1 × 20, baseline = 1× RTX 4090 24GB)**:
   - **Tier ① (`0.7~1.0` = `14~20 pts`)**: "小模型或多模型或小型大语言模型方案可完全在单张 4090 上部署并并行运行、无需额外 GPU 资源"
   - **Tier ② (`0.3~0.6` = `6~12 pts`)**: "需量化、裁剪、分批等轻量优化后可在单张 4090 运行"
   - **Tier ③ (`0`)**: Requires multiple GPUs / higher-spec GPU / paid API.
3. **`Semantic Score` (额外 5 分, 0–5 pts)**:
   - "依据模型在判类之外输出的**缺陷类型、发生位置与成因描述**人工评分（0~5 分）；不输出语义归因则记 0 分。"

## Critical Questions to Argue Ruthlessly
1. **`S_cost` Tier ① (`14–20`) vs Tier ② (`6–12`) Risk on `resident_v6_core`**:
   - `resident_v6_core` uses **4-bit NF4 (`bitsandbytes` from `third_party/pydeps`)** for `p1_qy` (8B), `chunk=2` for `v16` (6B), and hits **`21.16 GiB` peak** (only 2.8 GiB margin below 24 GiB 4090).
   - Could a strict judge classify `resident_v6_core` into **Tier ② (`0.5–0.6` = `10–12 pts`)** literally because it uses 4-bit quantization + chunking + pruning?
   - Compare that against **`resident_v6_core` WITHOUT `p1_qy`** (5 pure vision towers, `~3.8B` total params in native `bf16` with **ZERO quantization**, ~`10 GB` VRAM, ~`1s/img`, could even run 2–3 workers in parallel on ONE 4090!):
     - Losing `p1_qy` costs `~0.2–0.3` objective `Doc2S_orig` (and `-0.00048 RPS_orig`).
     - Moving from `S_cost` Tier ② (`12 pts`) to Tier ① (`18–20 pts`) gains **+6 to +8 total points**!
     - Wait: OR is `resident_v6_core` ALREADY eligible for Tier ① (`0.7–0.85`) if `p1_qy` 8B is a "小型大语言模型" running resident on a single 4090 (`<= 24 GiB`, no multi-GPU)? Let's argue both sides brutally! (Note: remember the top-10 gate first filters by **objective score**! We must NOT lose objective rank out of top-10, AND remember we want maximum **Objective + Subjective** total among top-10!).
2. **Maximizing `S_deploy` (aiming for `14–15 / 15`)**:
   - What concrete engineering deliverables in our repo/submission package (`final_engine.py`, HTTP/gRPC/FastAPI service wrapper, TAT/极目 adapter schema, self-contained wheelhouse/Docker/dependency check for `bitsandbytes`, exact `N=1` latency & VRAM benchmark) are missing or weak right now?
3. **Capturing the `5/5` `Semantic Score` Bonus safely**:
   - Remember Rule: **ZERO modification to the 6-class `label` predictions** (and keep total `N=1` latency `<= 15s` on 4090 and VRAM `<= 24 GiB`).
   - Notice we ALREADY have `code/r55/defect_attribution_map.py` (fast spatial defect localization `< 1.0s`, `< 1.0 GB` VRAM) AND `p1_qy` (`Qwen3-VL-8B` already in memory, or a lightweight causal template + spatial box + severity rationale).
   - What is the **cleanest, highest-scoring (4–5/5), lowest-risk** way to output `{defect_type, location_bbox_or_region, cause_description}` alongside `label` in our inference pipeline — and should it use `defect_attribution_map.py` + `Qwen3-VL-8B` text generation (only when any defect is predicted), or structured attribution + spatial heatmap?
