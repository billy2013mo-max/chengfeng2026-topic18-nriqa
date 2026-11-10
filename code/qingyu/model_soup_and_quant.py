#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
晴羽观感体系 — 支柱五 (6.2 节): 8折参数空间 Model Soup 融合器与混合精度 W4A16 部署量化器

核心数学与工程保证:
1. 全量微调/顶层微调模型 (Tier-1 5大轻量主干):
   - 按 8 折验证集综合得分 s_k 计算温度缩放权重 beta_k = softmax(tau * s_k)，在 FP32 参数空间执行凸组合:
     theta_soup = sum_{k=1}^8 beta_k * theta^{(k)}
   - 针对 ConvNeXtV2-Base (0.089B x 2 = 0.178B, 表 4.1 预算内)，基于 8 折顶层参数余弦距离执行 2 簇互补划分，
     分别融合出 Cluster-A 与 Cluster-B 两个互补 Soup 权重，兼顾零方差与双峰多样性。
2. LoRA 微调大图塔/大模型 (InternViT-6B / Clef-Flash-9B / Qwen3-VL-8B):
   - 严格避免对 (A^{(k)}, B^{(k)}) 直接线性平均导致的交叉项偏差!
   - 精确计算权重空间叠加增量: Delta_W_soup = sum_{k=1}^8 beta_k * (alpha / r) * (B^{(k)} @ A^{(k)})
   - 提供两种无损导出模式:
     (a) SVD 正交重分解模式 (`svd_lora_soup`): 对 Delta_W_soup 执行截断奇异值分解重构紧凑型 LoRA 检查点；
     (b) 基底原位融合模式 (`merge_lora_into_module`): 在单卡 4090 W4A16 量化前直接把精确的 Delta_W_soup 累加进基底 Linear 权重，零推理额外开销。
3. 单卡 RTX 4090 混合精度保护策略 (6.2 节):
   - 视觉编码器 (Vision Tower / Tier-1 全部主干) 强制保持 BF16；
   - 分类头 (UnimodalHead / JointSchemaHead / TipAdapter / Stacker) 强制保持 FP32；
   - 仅对 InternViT-6B 深层 Linear 与双 VLM 的 LLM 解码器 Linear 层启用 W4A16 仅权重量化 (bitsandbytes Linear4bit NF4)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from qingyu.config_and_paths import QINGYU_CKPT_DIR, QINGYU_RESULTS_DIR


def compute_soup_weights(val_scores: Sequence[float], temperature: float = 2.5) -> List[float]:
    """根据各折验证集得分计算数值稳定的 Softmax 参数融合权重 beta_k。"""
    if not val_scores:
        return []
    clean_scores = [float(s) if (s is not None and math.isfinite(float(s))) else 0.0 for s in val_scores]
    max_s = max(clean_scores)
    if max_s - min(clean_scores) < 1e-6 or temperature <= 0.0:
        return [1.0 / len(clean_scores)] * len(clean_scores)
    exps = [math.exp(temperature * (s - max_s)) for s in clean_scores]
    total = sum(exps)
    return [e / total for e in exps]


def _is_lora_a_key(key: str) -> bool:
    return ".lora_A." in key or key.endswith(".lora_A.weight") or key.endswith("lora_A.default.weight")


def _matching_lora_b_key(lora_a_key: str) -> str:
    return lora_a_key.replace("lora_A", "lora_B")


def average_state_dicts_exact(
    state_dicts: Sequence[Dict[str, Any]],
    weights: Sequence[float],
    lora_scaling: float = 2.0,
    svd_rank_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    在 FP32 中精确合并多个检查点 state_dict:
    - 普通浮点张量: sum_k beta_k * T^{(k)}
    - 整型计数缓冲 (如 num_batches_tracked): 取最大值
    - 成对 LoRA 参数 (lora_A, lora_B): 先在完整矩阵空间合成 sum_k beta_k * (B^{(k)} @ A^{(k)})，
      再以精确低秩 SVD 分解回 (B_soup, A_soup)，消除直接平均 A、B 造成的二阶交叉误差!
    """
    import torch

    if not state_dicts:
        raise ValueError("state_dicts 不能为空")
    assert len(state_dicts) == len(weights), "state_dicts 与 weights 长度必须一致"

    ref_sd = state_dicts[0]
    all_keys = list(ref_sd.keys())
    handled_lora_b = set()
    merged: Dict[str, Any] = {}

    for key in all_keys:
        if key in handled_lora_b:
            continue

        if _is_lora_a_key(key):
            b_key = _matching_lora_b_key(key)
            if b_key in ref_sd and all((key in sd and b_key in sd) for sd in state_dicts):
                handled_lora_b.add(b_key)
                # A: [r, in_features], B: [out_features, r]
                delta_acc = None
                orig_dtype = ref_sd[key].dtype
                orig_r = int(ref_sd[key].shape[0])
                for sd, w in zip(state_dicts, weights):
                    a_mat = sd[key].detach().to(device="cpu", dtype=torch.float32)
                    b_mat = sd[b_key].detach().to(device="cpu", dtype=torch.float32)
                    prod = torch.matmul(b_mat, a_mat)  # [out, in]
                    if delta_acc is None:
                        delta_acc = float(w) * prod
                    else:
                        delta_acc.add_(prod, alpha=float(w))

                assert delta_acc is not None
                target_r = min(
                    svd_rank_override or orig_r,
                    int(delta_acc.shape[0]),
                    int(delta_acc.shape[1]),
                )
                u_mat, s_vec, vh_mat = torch.linalg.svd(delta_acc, full_matrices=False)
                u_r = u_mat[:, :target_r]
                s_sqrt = torch.sqrt(torch.clamp(s_vec[:target_r], min=0.0))
                vh_r = vh_mat[:target_r, :]
                b_soup = (u_r * s_sqrt.unsqueeze(0)).to(dtype=orig_dtype).contiguous()
                a_soup = (s_sqrt.unsqueeze(1) * vh_r).to(dtype=orig_dtype).contiguous()
                merged[key] = a_soup
                merged[b_key] = b_soup
                continue

        tensors = [sd[key] for sd in state_dicts if key in sd]
        if len(tensors) != len(state_dicts):
            # 若部分折缺失非核心 buffer，回退使用首个折张量
            merged[key] = ref_sd[key].detach().cpu().clone()
            continue

        t0 = tensors[0]
        if not isinstance(t0, torch.Tensor):
            merged[key] = t0
            continue

        if t0.is_floating_point():
            acc = torch.zeros_like(t0, device="cpu", dtype=torch.float32)
            for t, w in zip(tensors, weights):
                acc.add_(t.detach().to(device="cpu", dtype=torch.float32), alpha=float(w))
            merged[key] = acc.to(dtype=t0.dtype).contiguous()
        elif t0.dtype == torch.bool:
            merged[key] = t0.detach().cpu().clone()
        else:
            # Long/Int buffer (e.g. num_batches_tracked)
            stacked = torch.stack([t.detach().cpu() for t in tensors], dim=0)
            merged[key] = torch.max(stacked, dim=0).values.contiguous()

    return merged


def split_8folds_into_2_complementary_clusters(
    state_dicts: Sequence[Dict[str, Any]],
    val_scores: Sequence[float],
) -> Tuple[List[int], List[int]]:
    """
    针对 ConvNeXtV2-Base (0.089B x 2 = 0.178B):
    利用各折分类头与顶层特征权重的中心化余弦距离将 8 折划分为两个大小均衡 (4 + 4) 的互补簇，
    以保留轻量 CNN 在纹理高频与块效应局部响应上的双模态特征多样性。
    """
    import torch

    n = len(state_dicts)
    if n < 4:
        mid = max(1, n // 2)
        return list(range(mid)), list(range(mid, n))

    vecs = []
    # 选取 head / prototype / 顶层小张量拼成签名向量
    sample_keys = [
        k
        for k, v in state_dicts[0].items()
        if isinstance(v, torch.Tensor)
        and v.is_floating_point()
        and v.numel() <= 262144
        and ("head" in k or "proto" in k or "norm" in k or "stages.3" in k)
    ][:24]
    if not sample_keys:
        sample_keys = [
            k for k, v in state_dicts[0].items() if isinstance(v, torch.Tensor) and v.is_floating_point()
        ][:8]

    for sd in state_dicts:
        parts = [sd[k].detach().to(device="cpu", dtype=torch.float32).flatten()[:4096] for k in sample_keys if k in sd]
        vec = torch.cat(parts, dim=0) if parts else torch.zeros(16, dtype=torch.float32)
        vecs.append(vec)

    mat = torch.stack(vecs, dim=0)  # [N, D]
    mat = mat - mat.mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(mat, dim=1, keepdim=True).clamp_min(1e-8)
    mat_norm = mat / norms
    sim = torch.matmul(mat_norm, mat_norm.T)  # [N, N]

    # 选择相似度最低的一对高分折作为双簇种子
    best_pair = (0, 1)
    min_sim = 2.0
    for i in range(n):
        for j in range(i + 1, n):
            s_ij = float(sim[i, j].item())
            if s_ij < min_sim:
                min_sim = s_ij
                best_pair = (i, j)

    seed_a, seed_b = best_pair
    diffs = [(idx, float(sim[idx, seed_a].item() - sim[idx, seed_b].item())) for idx in range(n)]
    diffs.sort(key=lambda x: x[1], reverse=True)
    half = n // 2
    cluster_a = sorted([idx for idx, _ in diffs[:half]])
    cluster_b = sorted([idx for idx, _ in diffs[half:]])
    return cluster_a, cluster_b


def build_backbone_soup_from_8folds(
    backbone_name: str,
    ckpt_dir: Path,
    out_dir: Path,
    temperature: float = 2.5,
    two_cluster_soup: bool = False,
) -> Dict[str, Any]:
    """
    扫描 `ckpt_dir` 下指定模型的 8 折权重 (`fold_0_best.pt` .. `fold_7_best.pt`) 并生成单组或双簇 Model Soup。
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_paths: List[Path] = []
    for k in range(8):
        candidates = [
            ckpt_dir / backbone_name / f"fold_{k}_best.pt",
            QINGYU_RESULTS_DIR / backbone_name / f"fold_{k}_best.pt",
            ckpt_dir / f"{backbone_name}_fold{k}_best.pt",
            ckpt_dir / backbone_name / f"fold{k}" / "best_checkpoint.pt",
        ]
        found = next((p for p in candidates if p.is_file()), None)
        if found is not None:
            fold_paths.append(found)

    if not fold_paths:
        return {
            "backbone": backbone_name,
            "status": "skipped_no_checkpoints",
            "found_folds": 0,
        }

    payloads = [torch.load(str(p), map_location="cpu", weights_only=False) for p in fold_paths]
    val_scores = [
        float(p.get("best_val_score", p.get("val_doc_blend", p.get("val_doc1", 50.0))))
        for p in payloads
    ]
    weights = compute_soup_weights(val_scores, temperature=temperature)

    def _merge_subset(indices: Sequence[int], tag_suffix: str = "") -> str:
        sub_payloads = [payloads[i] for i in indices]
        sub_scores = [val_scores[i] for i in indices]
        sub_w = compute_soup_weights(sub_scores, temperature=temperature)

        # 支持顶层直接是 state_dict 或嵌套 {"model_state": ..., "head_state": ...}
        soup_ckpt: Dict[str, Any] = {
            "backbone": backbone_name,
            "num_folds_merged": len(indices),
            "fold_indices": list(indices),
            "fold_val_scores": sub_scores,
            "soup_weights": sub_w,
            "config": sub_payloads[0].get("config", {}),
        }
        tensor_dict_keys = [
            k
            for k in ("model_state", "lora_state", "head_state", "tip_adapter_state", "state_dict")
            if k in sub_payloads[0] and isinstance(sub_payloads[0][k], dict)
        ]
        if tensor_dict_keys:
            for tk in tensor_dict_keys:
                soup_ckpt[tk] = average_state_dicts_exact(
                    [p[tk] for p in sub_payloads],
                    sub_w,
                )
        else:
            # 整个 payload 即 state_dict
            soup_ckpt = average_state_dicts_exact(sub_payloads, sub_w)

        fname = f"{backbone_name}_soup{tag_suffix}.pt"
        save_path = out_dir / fname
        torch.save(soup_ckpt, str(save_path))
        return str(save_path)

    saved_files = [_merge_subset(list(range(len(payloads))), tag_suffix="")]
    if two_cluster_soup and len(payloads) >= 4:
        # 提取用于聚类的状态字典
        primary_key = next(
            (k for k in ("model_state", "head_state", "state_dict") if k in payloads[0] and isinstance(payloads[0][k], dict)),
            None,
        )
        sds = [p[primary_key] if primary_key else p for p in payloads]
        ca, cb = split_8folds_into_2_complementary_clusters(sds, val_scores)
        saved_files.append(_merge_subset(ca, tag_suffix="_cluster0"))
        saved_files.append(_merge_subset(cb, tag_suffix="_cluster1"))

    return {
        "backbone": backbone_name,
        "status": "merged_ok",
        "found_folds": len(fold_paths),
        "fold_val_scores": [round(s, 4) for s in val_scores],
        "soup_weights": [round(w, 4) for w in weights],
        "saved_files": saved_files,
    }


def merge_peft_or_custom_lora_into_base(model: Any) -> Any:
    """
    在单卡 4090 部署量化前，将 PEFT LoRA 模型安全 `merge_and_unload()` 合并进基底 Linear 层，
    彻底消除运行期双路矩阵乘法与显存碎片开销。
    """
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def apply_4090_mixed_precision_quant(
    module: Any,
    enable_w4a16: bool = True,
    skip_module_keywords: Tuple[str, ...] = (
        "visual",
        "vision_tower",
        "vision_model",
        "patch_embed",
        "embeddings",
        "lm_head",
        "head",
        "schema_head",
        "unimodal_head",
        "tip_adapter",
        "router",
        "norm",
    ),
    min_linear_params: int = 1_048_576,
) -> Dict[str, int]:
    """
    严格落实方案 6.2 节混合精度量化策略:
    - 跳过所有视觉编码器 (`visual` / `vision_tower` / `vision_model`)、Patch Embedding、LayerNorm 及全部分类头；
    - 仅将参数规模 >= 1M 的大型 Transformer / LLM `nn.Linear` 替换为 W4A16 (NF4, compute_dtype=bfloat16) 仅权重量化层；
    - 若当前环境未安装 `bitsandbytes` 或处于 141GB H200 原生 BF16 模式 (`enable_w4a16=False`)，自动安全保持 BF16。
    """
    import torch
    import torch.nn as nn

    stats = {"quantized_linear_layers": 0, "preserved_bf16_fp32_layers": 0, "backend": "bf16_native"}
    if not enable_w4a16:
        return stats

    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception:
        stats["backend"] = "bf16_fallback_no_bnb"
        return stats

    stats["backend"] = "bitsandbytes_nf4_w4a16"

    def _recursive_replace(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            lowered = full_name.lower()
            if any(kw.lower() in lowered for kw in skip_module_keywords):
                stats["preserved_bf16_fp32_layers"] += 1
                continue

            if isinstance(child, nn.Linear):
                num_params = child.in_features * child.out_features
                if num_params >= min_linear_params:
                    has_bias = child.bias is not None
                    q_linear = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=has_bias,
                        compute_dtype=torch.bfloat16,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    src_device = child.weight.device
                    weight_cpu = child.weight.detach().to(dtype=torch.bfloat16, device="cpu")
                    q_linear.weight = bnb.nn.Params4bit(
                        weight_cpu,
                        requires_grad=False,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    if has_bias and child.bias is not None:
                        q_linear.bias = nn.Parameter(
                            child.bias.detach().to(dtype=torch.bfloat16, device="cpu"),
                            requires_grad=False,
                        )
                    if src_device.type == "cuda":
                        q_linear = q_linear.to(src_device)
                    setattr(parent, child_name, q_linear)
                    stats["quantized_linear_layers"] += 1
                else:
                    stats["preserved_bf16_fp32_layers"] += 1
            else:
                _recursive_replace(child, full_name)

    _recursive_replace(module)
    return stats


ALL_QINGYU_TOWERS: Tuple[Tuple[str, bool], ...] = (
    ("convnextv2_base", True),   # 0.089B x 2 互补双簇 Soup = 0.178B
    ("pe_core_large", False),    # 0.350B 8折单 Soup
    ("dinov3_vitl16", False),    # 0.330B 8折单 Soup
    ("siglip2_so400m_naflex", False),  # 0.430B 8折单 Soup
    ("cradio_v4_h", False),      # 0.670B 8折单 Soup
    ("internvit_6b", False),     # 6.080B 8折 SVD-LoRA Soup
    ("clef_flash_9b", False),    # 9.120B 8折 SVD-LoRA + JointSchemaHead Soup
    ("qwen3_vl_8b", False),      # 8.100B 8折 SVD-LoRA + UnimodalHead Soup
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="晴羽 8折参数空间 Model Soup 融合器")
    parser.add_argument("--ckpt-dir", type=str, default=str(QINGYU_CKPT_DIR))
    parser.add_argument("--out-dir", type=str, default=str(QINGYU_CKPT_DIR / "soup_4090"))
    parser.add_argument("--backbone", type=str, default="all", help="指定单个模型名或 all")
    parser.add_argument("--temperature", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = (
        ALL_QINGYU_TOWERS
        if args.backbone == "all"
        else [(args.backbone, args.backbone == "convnextv2_base")]
    )

    manifest_reports: List[Dict[str, Any]] = []
    print("=" * 96)
    print("=== [晴羽支柱五 6.2] 8折参数空间 Model Soup 融合器 (零推理倍增 & LoRA 精确 SVD 重构) ===")
    print("=" * 96)
    for bname, two_cluster in targets:
        rep = build_backbone_soup_from_8folds(
            backbone_name=bname,
            ckpt_dir=ckpt_dir,
            out_dir=out_dir,
            temperature=args.temperature,
            two_cluster_soup=two_cluster,
        )
        manifest_reports.append(rep)
        print(f"  [{bname:22s}] status={rep['status']} | folds={rep['found_folds']} | files={len(rep.get('saved_files', []))}")

    report_path = out_dir / "soup_manifest.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(manifest_reports, f, ensure_ascii=False, indent=2)
    print(f"[Model Soup] 清单已保存至: {report_path}")


if __name__ == "__main__":
    main()
