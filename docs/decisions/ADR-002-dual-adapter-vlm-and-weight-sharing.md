# ADR-002: 采用冻结层逐位共享、双适配器共享 4-bit VLM 底座与词表主存卸载

## Status
Accepted

## Date
2026-11-19

## Context
`resident_v8_boundary` 需要在一张 24 GB GPU 上同时常驻 14 个专精成员，其中包括 7 个含 `PE-Core-Large` 的塔、4 个含 `C-RADIOv4-H` 的塔，以及两套 8B 多模态大模型微调产物（`p1_qy` DoRA `r=24` 与 `qwen_hr` LoRA `r=64`）。若按常规独立加载，仅两套 4-bit `Qwen3-VL-8B` 就需约 `9.0 GB`，加上 12 个视觉模型（约 `17+ GB`）必然超出 24 GB 显存硬上限。

## Decision
在 [`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py) 与 [`code/r55/dual_vlm_shared_4090.py`](file:///wanqing-develop/luowenjing/18/code/r55/dual_vlm_shared_4090.py) 中联合实施四项常驻显存控制策略：
1. **冻结张量逐位相等物理存储共享（`--share-weights`）**：每加载完一个成员即运行 `_share_identical_weights()`，跳过 `uint8`（4-bit 打包权重）与 `< 64 KiB` 小张量，通过采样哈希桶 + 精确 `torch.equal()` 校验后将重复冻结权重 `t.data = canon` 指向单一显存存储；
2. **单 4-bit NF4 底座双适配器共享（`DualAdapterQwen3VL4090`）**：仅加载一份 `Qwen3-VL-8B`（`252` 个 LLM 线性层原地 4-bit NF4 量化 + BF16 ViT），同时注册 `p1_qy`（DoRA `r=24`）与 `qwen_hr`（LoRA `r=64`）两套 PEFT 适配器，并在主机/显存中保留各自 24 个非 LoRA merger 张量，推理切换时原地 `.copy_()` 覆盖；
3. **VLM 静态词表主机内存卸载（`--embed-on-host`）**：`_move_token_embedding_to_host()` 将 `151,936 × 4096` BF16 词表（`1.159 GB`）替换为 `HostEmbedding`，每次仅按固定 Prompt 的 `input_ids` 索引对应行回传 GPU；
4. **加载序与激活分块控制**：优先加载并量化 `dual_vlm` 避开 BF16 瞬态加载叠加峰；推理期设置 `v16_chunk=2`、`BoundaryTileTower(view_chunk=8)` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## Alternatives Considered

### 备选 A：双 8B VLM 分别独立常驻（各一份 4-bit NF4 底座）
- 优点：`p1_qy` 可保持训练后“先合并 DoRA 再 4-bit 量化”的完全零数值差路径。
- 缺点：额外消耗 `~3.40 GB` 常驻显存，叠加 `bndry_pe` 与 `bndry_dual` 后总 `nvidia-smi` 峰值将突破 `24.4 GiB`，在 24 GB 卡上触发 OOM。
- 结论：拒绝。

### 备选 B：成员间动态 CPU-GPU 换入换出（Offload / Swapping）
- 优点：常驻显存极低。
- 缺点：逐图 PCIe 往返拷贝 `> 10 GB` 权重会严重拖慢延迟并引发显存碎片，危及 `<= 15s/图` 预算。
- 结论：拒绝。

## Consequences
- **已实测收益（H200 NVML，`resident_v8_boundary_bench_half.json`）**：冻结层共享无损释放 `5.808 GB`，双适配器 VLM 合计常驻 `5.67 GB`（含词表卸载 `1.159 GB`，较双独立实例省 `~3.40 GB`），全 14 成员常驻 `17.50 GB`、`nvidia-smi` 加载峰值 `20.23 GiB`、推理峰值 `21.02 GiB`（HTTP 含归因峰值 `21.03~21.04 GiB < 24 GiB`），单图压测均值 `4.84s`。
- **已知数值权衡**：`DualAdapterQwen3VL4090` 走 `NF4(W) + bf16(DoRA)` 而非独立版 `NF4(W + DoRA)`，使 `p1_qy` 相对独立合并版产生 `mean|Δp| = 0.0019`（独立核验 `argmax` 一致率 `100.0%`）；在 `test_150` 上有 1 张处于临界线的图像因此跨过 Clean 阈值（`19 -> 20`），线上 `Clean Score` 仍保持 `20.00 / 20` 满分。
- **待验证项**：当前显存峰值与延迟均在 H200 上按 24 GB 预算测得，真实 RTX 4090（`sm_89`）硬件上的实际延迟与驱动级显存余量待真机复测。
