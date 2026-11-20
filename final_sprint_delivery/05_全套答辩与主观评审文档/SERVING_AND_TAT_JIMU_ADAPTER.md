# 在线/批量服务与 极目 / TAT / VQA 适配映射（`resident_v8_boundary` 共享主干+双适配器 VLM + `deconf_v6_boundary`）

> **声明**：本文件给出的是 **适配层（Adapter）字段映射建议**。我们没有、也不声称掌握 极目 / TAT 平台的内部私有接口规范；下表右侧的目标字段名是按通用音视频质检平台约定拟定的 **占位名**，实际接入时只需以平台方正式接口文档为准修改薄适配层，不影响底层模型推理与 `label` 输出。左侧字段均来自 [`code/r55/subjective_engineering.py`](file:///wanqing-develop/luowenjing/18/code/r55/subjective_engineering.py) 中 `export_tat_jimu_payload` 与 [`code/r55/serve_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/serve_engine.py) 的真实输出（已端到端实测验证）。

---

## 1. 常驻在线 / 批量 HTTP 服务入口 [`code/r55/serve_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/serve_engine.py)

默认加载主推 **`resident_v8_boundary`（开启 `--share-weights` 无损同源主干显存合并省 `5.81 GB` + `DualAdapterQwen3VL4090` 双适配器共享 4-bit 底座 + `--embed-on-host` 词表主存卸载 + 纯 PyTorch `INT8-g128` 量化未共享 `v16[pecoreg]` 省 `1.72 GiB` + 比特精确激活分块 + `--decoder deconf_v6_boundary`）**，单张 RTX 4090（24 GB）常驻显存仅 **`15.78 GB`**、`nvidia-smi` 实时服务显存 **`18.56 GiB`**（含冷启动加载全链路峰值 **`19.98 GiB < 24 GiB`**，推理余量 `~5.50 GiB`），启动一次即可持续处理单帧与多帧视频请求（同时提供离线生产 [`Dockerfile`](file:///wanqing-develop/luowenjing/18/Dockerfile) 支持一键容器化部署）：

```bash
# 启动常驻服务（自动启用 BF16 + 同源冻结主干合并 + 双适配器共享底座 + 词表卸载 + 合成图冷启动预热）
python3 code/r55/serve_engine.py --set resident_v8_boundary --decoder deconf_v6_boundary --gpu 0 --port 8055

# 1) 健康检查与实时显存/吞吐监控
curl -s localhost:8055/healthz

# 2) 单图在线检测（支持本地路径或 Base64 上传，严格走 N=1 推理 + deconf_v6_boundary 校准 + 语义归因）
curl -s -XPOST localhost:8055/infer       -H 'Content-Type: application/json' -d '{"image_path": "/abs/a.png"}'
curl -s -XPOST localhost:8055/infer       -H 'Content-Type: application/json' -d '{"image_base64": "<b64>", "image_name": "a.png"}'

# 3) 批量评测 / 视频多帧抽帧聚合（自动关联同一视频帧并输出 EffectiveFrameORVideoAggregator 视频级结论）
curl -s -XPOST localhost:8055/infer_batch -H 'Content-Type: application/json' -d '{"image_paths": ["/abs/v_frame_0.png", "/abs/v_frame_1.png"], "video_id": "clip_001"}'
```

| 接口 | 作用 | 关键返回字段 |
| :--- | :--- | :--- |
| `GET /healthz` | 存活、显存余量与同源主干省显存监控 | `status, set, members, decoder, params_b, shared_weights_gb, embed_host_gb, torch_peak_gb, nvidia_smi_mib, nvidia_smi_peak_mib, requests, uptime_s` |
| `POST /infer` | 单帧在线检测（与 `final_engine.py` 完全同一条 `N=1` 推理链） | `label, pred_6d, probs_6x4, attribution, tat_jimu_payload, latency_ms{infer,decode,attr,total,request}` |
| `POST /infer_batch` | 批量图片 / VQA 视频多帧聚合检测 | `results[]`（同 `/infer`）+ `videos[]`（`EffectiveFrameORVideoAggregator`：`video_id, num_frames, video_label, trigger_frames, m_eff_p0`） |

### 工程可靠性设计：
- **双传输后端零依赖回退**：检测到 `FastAPI + uvicorn` 时自动启用异步 HTTP 服务；缺省环境下自动回退 Python 标准库 `ThreadingHTTPServer`（`--backend stdlib` 可显式指定），零额外第三方服务依赖。
- **并发安全与零文件名污染**：HTTP 层支持多线程并发接收，GPU 前向通过互斥锁串行共享单份常驻权重；所有请求文件以唯一前缀 `srv_<uuid>_<原名>` 暂存于 `--spool-dir` 并在请求返回时立即清理，绝不与任何同名图片冲突。
- **目录越权防护与优雅退出**：默认仅绑定 `127.0.0.1`；配置 `--path-root` 时强制校验路径前缀防越权读取（非法路径返回 `403`）；捕获 `SIGTERM / SIGINT` 后优雅释放全部显存并以退出码 `0` 退出。

---

## 2. `tat_jimu_payload` $\rightarrow$ 极目 / TAT / VQA 字段映射表（适配建议）

| `tat_jimu_payload` 实际字段 | 类型 / 取值 | 极目质检（占位名） | TAT 批量任务（占位名） | VQA 旁路链路（占位名） | 转换与业务联动说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `image` | `str` | `media_id` | `item_id` | `frame_id` | 原样透传 |
| `label.{模糊,纹理损失,过锐,花屏,过度生成,虚化突兀}` | `无问题/P2/P1/P0` | `defect_labels[dim]` | `annotations[dim].level` | `frame_defects[dim]` | 原样；数值映射：`无问题=0, P2=1, P1=2, P0=3` |
| `tat_jimu_ext.is_badcase` | `bool` | `is_bad` | `result.reject` | `has_defect` | 任一维 `> 无问题` 即为 `true` |
| `tat_jimu_ext.worst_severity` | `无问题/P2/P1/P0` | `severity` | `result.priority` | `max_severity` | 用于质检工单优先级排序 |
| `tat_jimu_ext.p0_alert` | `bool` | `alarm` | `result.urgent` | `trigger_rollback` | **P0 高优拦截信号**：服务端处理产物检出即触发管线回退原片；源内容触发降权/治理 |
| `tat_jimu_ext.clean_prob` | `float ∈ [0,1]` | `quality_score` | `result.confidence` | `clean_posterior` | 六维无问题联合概率 $\prod_d P(\text{无问题}_d)$ |
| `tat_jimu_ext.latency_ms` | `float` | `cost_ms` | `meta.latency_ms` | `iqa_latency_ms` | 单帧端到端总耗时（含 2×TTA + 校准 + 归因） |
| `attribution.summary_40ch` | `str (≤40字)` | `reason` | `result.comment` | `rationale_short` | 按构造严格 `≤40` 字（零截断），供排障看板直接展示 |
| `attribution.defects[i].defect / severity` | `str` | `regions[i].label / level` | `annotations[dim]` | `defect_items[i]` | **100% 锁定 `label` 解码结论**（零不一致） |
| `attribution.defects[i].box_xyxy` | `[x1,y1,x2,y2]` 或 `null` | `regions[i].bbox` | `annotations[dim].bbox` | `grounding_box` | 局部伪影输出原生像素框；全图弥散退化（如全图压缩模糊/涂抹）输出 `null`，绝不强行臆造小框 |
| `attribution.defects[i].location` | 九宫格中文 或 `全图` | `regions[i].position` | `annotations[dim].pos` | `spatial_tag` | 基于未膨胀 `4×4` 反事实粗糙度检验 + `8×8` 子格精定位（弥散模糊/纹理损失收紧门控，自动标 `全图`） |
| `attribution.defects[i].region_cue` | 视觉内容提示 或 `null` | `regions[i].content_cue` | `annotations[dim].cue` | `roi_cue` | 零幻觉确定性像素先验提示（`人像肤色区/高反差边缘/暗部区域/亮部区域/平滑背景区/主体细节区`） |
| `attribution.defects[i].reason_40ch` | `str (≤40字)` | `regions[i].reason` | `annotations[dim].comment` | `item_rationale` | 分层洁净分位超阈时报实测值与分位；局部/全图相对偏差 ≥20% 时报相对对比；其余报区域提示与置信度（去模板唯一率 `87.4%~94.5%`） |
| `attribution.defects[i].evidence.measurements[]` | `{metric,name,value,clean_pct,supports}` | `regions[i].evidence` | `annotations[dim].evidence` | `physics_audit` | 12 项原生分辨率像素与场景归一化物理量，完全可审计 |
| `attribution.defects[i].causal_uncertainty / uncertainty_note` | `bool / str` | `need_review / review_note` | `result.uncertainty` | `pipeline_uncertainty` | 明确说明单帧无法区分的上游环节成因（如源拍摄失焦 vs 编码模糊；源 AIGC vs 超分重绘） |
| `videos[].video_label / trigger_frames` (`/infer_batch`) | 6 维等级 / 触发帧下标 | `video_result / key_frames` | `video.annotations` | `vqa_aggregated_decision` | 保障单帧严重 `P0` 高召回，同时用帧间相关系数 $\rho_d$ 抑制长视频多帧误报膨胀 |

---

## 3. 原生像素与场景归一化实测证据体系 [`code/r55/measured_evidence.py`](file:///wanqing-develop/luowenjing/18/code/r55/measured_evidence.py)

为消除不同内容场景对比度/字幕干扰，我们在 `448×448` **原生 1:1 像素块（零缩放）** 上同时计算经典梯度统计与**场景无量纲归一化比值**，并与离线预计算的训练集全无问题帧基线（`results/r55/final/resident_v8_boundary_clean_footprint.npz`，推理期 **0 次训练数据读取**）做分位数比对：

| 缺陷维度 | 主测量指标（场景归一化） & 辅助指标 | 异常方向 | 实测物理支持判据 | 空间定位专职算子 |
| :--- | :--- | :---: | :---: | :--- |
| **模糊** | `acutance`（边缘锐度比 `\|Lap\|/\|Sobel\|`）、`lapvar` | 偏低 | `≤ 洁净 P10` | `pe_v3` `4×4→8×8` 反事实图（全图弥散自动标 `全图`） |
| **纹理损失** | `tex_norm`（对比度归一化微纹理比）、`tex_hf` | 偏低 | `≤ 洁净 P10` | `pe_v3` 反事实图（全图降噪自动标 `全图`） |
| **过锐** | `overshoot_p90`（强边缘过冲比 P90）、`overshoot` | 偏高 | `≥ 洁净 P90` | `pe_v3` 反事实图（捕捉 1–3px 亮白振铃光晕） |
| **花屏** | `blocky`（8px 宏块边界跳变比）、`chroma`（色度撕裂比） | 偏高 | `任一 ≥ 洁净 P90` | **`peg` 花屏专训塔** 专属反事实空间定位 |
| **过度生成** | `coh`（结构张量相干度 P90）、`hf_ratio`（框内/全局高频比） | 偏高 | `任一 ≥ 洁净 P90` | `pe_v3` 局部聚焦定位 + 多塔结构语义置信度 |
| **虚化突兀** | `cliff`（清晰度断崖梯度 P99）、`dof_bimod`（虚实双峰跨度比） | 偏高 | `任一 ≥ 洁净 P90` | **`pe_bkseg1` 虚化分割专训塔** 专属边界定位 |

- **严格零标签侵入**：归因模块只读解码后的 `pred_6d`，无论开启或关闭 `--emit-attribution / --tat-out / --video-mode`，主输出预测 JSON **100% 逐字节一致**。
- **洁净样本零开销跳过**：六维均判为“无问题”的样本直接跳过反事实定位与像素物理测量（耗时 `< 0.1 ms`）。
