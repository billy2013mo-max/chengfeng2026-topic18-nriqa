# 在线/批量服务与 极目 / TAT / VQA 适配映射（`resident_v6_core`）

> **声明**：本文件给出的是 **适配层（adapter）字段映射建议**。我们没有、也不声称掌握 极目 / TAT 平台的内部接口规范；
> 下表右侧的目标字段名是按通用质检平台约定拟定的 **占位名**，接入时应以平台方正式接口文档为准，只需改动适配层，
> 不影响模型与 `label` 输出。左侧字段是 `code/r55/subjective_engineering.py::export_tat_jimu_payload` 的实际输出（已实测）。

## 1. 服务入口 `code/r55/serve_engine.py`

```bash
python3 code/r55/serve_engine.py --set resident_v6_core --gpu 0 --port 8055      # 模型常驻，启动一次
curl -s localhost:8055/healthz
curl -s -XPOST localhost:8055/infer       -H 'Content-Type: application/json' -d '{"image_path": "/abs/a.png"}'
curl -s -XPOST localhost:8055/infer       -H 'Content-Type: application/json' -d '{"image_base64": "<b64>", "image_name": "a.png"}'
curl -s -XPOST localhost:8055/infer_batch -H 'Content-Type: application/json' -d '{"image_paths": ["/abs/v_frame_0.png", "/abs/v_frame_1.png"]}'
```

| 接口 | 作用 | 关键返回字段 |
| :--- | :--- | :--- |
| `GET /healthz` | 存活与资源探针 | `status, set, members, decoder, load_s, torch_peak_gb, nvidia_smi_mib, nvidia_smi_peak_mib, requests` |
| `POST /infer` | 单图在线（与 `final_engine.py` 完全同一 N=1 路径：推理→逐行解码→实测证据归因） | `label, pred_6d, probs_6x4, attribution, tat_jimu_payload, latency_ms{infer,decode,attr,total,request}` |
| `POST /infer_batch` | 批量 / 视频多帧 | `results[]`（同 `/infer`）+ `videos[]`（`EffectiveFrameORVideoAggregator`：`video_label, trigger_frames, m_eff_p0`） |

- 传输层：FastAPI+uvicorn（已安装时）否则自动回退标准库 `ThreadingHTTPServer`，零额外依赖；`--backend stdlib` 可强制。
- GPU 推理由一把锁串行（单份常驻模型），HTTP 层可并发接收。
- 上传图片以唯一名 `srv_<uuid>_<原名>` 暂存于 `--spool-dir`（路径请求为软链接），请求结束即删除，不会与任何缓存的训练帧同名冲突。
- 默认只监听 `127.0.0.1`；对外暴露时用 `--path-root` 限制 `image_path` 可读目录。
- 视频分组：请求带 `video_id` 时整批视为一个视频，否则按文件名 `infer_video_group_id()`（如 `xxx.mp4_frame_3.png`）自动分组。

## 2. `tat_jimu_payload` → 极目 / TAT / VQA 字段映射（适配建议）

| `tat_jimu_payload` 实际字段 | 类型 / 取值 | 极目质检（占位名） | TAT 批量任务（占位名） | VQA 问答式（占位名） | 转换说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `image` | str | `media_id` | `item_id` | `image_id` | 原样 |
| `label.{模糊,纹理损失,过锐,花屏,过度生成,虚化突兀}` | `无问题/P2/P1/P0` | `defect_labels[dim]` | `annotations[dim].level` | 每维一条 `answer` | 原样；数值化：无问题=0,P2=1,P1=2,P0=3 |
| `tat_jimu_ext.is_badcase` | bool | `is_bad` | `result.reject` | — | 原样 |
| `tat_jimu_ext.worst_severity` | `无问题/P2/P1/P0` | `severity` | `result.priority` | — | P0→最高优先级 |
| `tat_jimu_ext.p0_alert` | bool | `alarm` | `result.urgent` | — | 原样（P0 告警通道） |
| `tat_jimu_ext.clean_prob` | float∈[0,1] | `quality_score` | `result.confidence` | — | `∏_d P(无问题_d)`，非校准“总分”，仅作排序 |
| `tat_jimu_ext.latency_ms` | float | `cost_ms` | `meta.latency_ms` | — | 端到端（推理+解码+归因） |
| `attribution.summary_40ch` | str ≤40 字 | `reason` | `result.comment` | `rationale` | 按构造 ≤40 字，无截断 |
| `attribution.defects[i].defect / severity` | str | `regions[i].label / level` | `annotations[dim]` | `question/answer` | 与 `label` 严格一致（同一解码结果） |
| `attribution.defects[i].box_xyxy` | `[x1,y1,x2,y2]` 原图像素 或 `null` | `regions[i].bbox`（如需 `xywh`：`[x1,y1,x2-x1,y2-y1]`） | `annotations[dim].bbox_norm`（`/W,/H` 归一化） | `grounding_box` | `null` = 全图/弥散缺陷，不臆造框 |
| `attribution.defects[i].location` | 九宫格中文 或 `全图` | `regions[i].position` | — | 文本描述 | 原样 |
| `attribution.defects[i].reason_40ch` | str ≤40 字 | `regions[i].reason` | `annotations[dim].comment` | `rationale` | 含实测量值与洁净集分位，如 `右下区P1过锐：边缘过冲比0.31（洁净P98）` |
| `attribution.defects[i].evidence.measurements[]` | `{metric,name,value,clean_pct,supports,global_value?}` | `regions[i].evidence`（JSON 透传） | `annotations[dim].evidence` | — | 实测物理量（见 §3），可审计 |
| `attribution.defects[i].evidence.measured_support` / `verdict` | bool / `supports·inconclusive·contradicts` | `evidence_confirmed` / `evidence_verdict` | `annotations[dim].verified` | — | 非 `supports` 时提示“物理证据不支持，待复核”；`contradicts` = 物理量落在缺陷相反一侧 |
| `attribution.(defects[i].)causal_uncertainty / uncertainty_note` | bool / str | `need_review / review_note` | `result.need_manual_check` | `uncertainty` | 成因不唯一或临界等级时为 `true` |
| `videos[].video_label / trigger_frames` (`/infer_batch`) | 6 维等级 / 帧下标 | `video_result / key_frames` | `video.annotations` | — | 单帧 P0 OR 规则保证召回；`m_eff_p0` 控制长视频误报 |

## 3. 实测证据（`code/r55/measured_evidence.py`）

每个被判为缺陷的维度，在 448×448 原生像素块上计算经典物理量，并与 **洁净训练帧**（离线预计算于
`results/r55/final/resident_v6_core_clean_footprint.npz`，推理时不读取任何训练数据）的分布比较：

| 维度 | 指标 | 方向 | 判定“实测支持” |
| :--- | :--- | :--- | :--- |
| 模糊 | 亮度拉普拉斯方差 `lapvar` | 低 | ≤ 洁净 P10 |
| 纹理损失 | 中等梯度区 \|Laplacian\| 中位数 `tex_hf` | 低 | ≤ 洁净 P10 |
| 过锐 | 强边缘处 \|y−G₁.₅(y)\|/局部对比度 `overshoot` | 高 | ≥ 洁净 P90 |
| 花屏 | 8px 块边界跳变比 `blocky`、色度/亮度梯度比 `chroma` | 高 | 任一 ≥ 洁净 P90 |
| 过度生成 | 结构张量相干度 P90 `coh`、框内/全局高频比 `hf_ratio` | 高 | 任一 ≥ 洁净 P90 |
| 虚化突兀 | log 高频能量梯度 P99 `cliff`（抠图边界清晰度断崖） | 高 | ≥ 洁净 P90 |

- 有集中热区（`conc≥2, area≤0.45`）时比较 **框内值 vs 洁净块分布**（并给出全局值）；否则比较 **全图 3×3 原生块中位数 vs 洁净图分布**。
- 不满足阈值时**不编造成因**：`measured_support=false`，`uncertainty_note` 明确写明“判级主要来自模型语义特征，具体成因无法由像素统计确证”。
- 全部 6 维为无问题的图片跳过定位与物理计算（零额外开销）。
- 归因只读取解码结果，**不改变任何 `label`**（test_150 上开/关归因输出 JSON 字节一致，见 `results/r55/v6_debate/claude_engineering_report.md`）。
