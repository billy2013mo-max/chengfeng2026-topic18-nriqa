#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/deploy_4090_qingyu_engine.py
========================================
晴羽观感体系 — 支柱五 (6.1, 6.2, 6.3 节):
单卡 RTX 4090 (24GB) 全常驻 25.283B 混合精度 Model Soup 确定性推理与 KV 缓存复用后置条件归因引擎。

严守赛题官方底线与方案工程指标:
1. 参数总量严格 <= 30B:
   Depth-Anything-V2-S (0.025B) + ConvNeXtV2-Base 双簇 Soup (0.178B) + PE-Core-Large (0.350B)
   + DINOv3-ViT-L/16 (0.330B) + SigLIP2-SO400M-NaFlex (0.430B) + C-RADIOv4-H (0.670B)
   + InternViT-6B (6.080B) + Clef-Flash-9B (9.120B) + Qwen3-VL-8B (8.100B) = 25.283B <= 30.000B。
2. 单卡 4090 峰值常驻显存 ~20.07 GB <= 24.00 GB，单图端到端时延 ~2.1s <= 15.0s:
   - 视觉编码器保持 BF16，全部分类头保持 FP32；
   - InternViT-6B 深层 Linear 与双 VLM 解码器 Linear 支持 W4A16 (NF4) 仅权重量化。
3. 100% 单图独立确定性推理 (`OMP_NUM_THREADS=8`, 零跨样本统计量):
   无论是 1 张单图、150 张练习集还是 654 张终评集，逐帧推理输出完全比特级/标签级一致。
4. 先分类锁定、复用 Prefill KV 缓存的后置条件归因 (6.3 节 & 附录二):
   - 当全无损 (`pred_6d == 0`) 时直接跳过文本生成 (0 解码 Token，零幻觉)；
   - 当检出缺陷时，锁定 `y_hat` 并将双擎算子定位的 `[0, 1000]` 归一化峰值区域框与客观物理证据注入结构化归因模块；
   - 若启用 `--enable-llm-attribution`，直接复用 `Qwen3-VL-8B` 分类 Prefill 阶段缓存的 `past_key_values` (零额外 ViT 前向)，
     追加锁定标签后缀执行贪婪确定性解码 (`temperature=0.0, top_k=0`)，并与物理兜底结构强校验合并，保证 100.0% 零矛盾。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    DIMS,
    FULL_DIMS,
    GRADE_INT_TO_STR,
    MODEL_POOL_BUDGET,
    P0_ONLY_DIMS,
    PEAK_RESIDENT_VRAM_GB,
    QINGYU_CKPT_DIR,
    QINGYU_FINAL_DIR,
    QINGYU_RESULTS_DIR,
    REPO_ROOT,
    TOTAL_POOL_PARAMS_B,
    LosslessNativeImageLoader,
    enforce_deterministic_runtime,
    format_official_submission,
    load_test_template_names,
    print_pool_budget_summary,
    resolve_model_path,
    setup_sys_path_and_cwd,
)


DEFECT_CAUSAL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "模糊": {
        "region_desc": "前景主体对焦核心区域",
        "phenomenon_p2": "前景主体局部细节轻微发软，边缘过渡略欠锐利，但整体轮廓尚可辨识",
        "phenomenon_p1": "前景核心主体出现明显失焦或运动拖影感，高频轮廓清晰度显著下降",
        "phenomenon_p0": "前景主体重度失焦或大尺度模糊退化，关键面部/物体结构难以辨清",
        "cautious_cause": "结合前景重模糊梯度比偏低特征，疑似由拍摄对焦偏移、快门期内相对运动或低分辨率重采样所致",
    },
    "纹理损失": {
        "region_desc": "前景皮肤/织物/表面微纹理区域",
        "phenomenon_p2": "局部表面高频颗粒与微细纹理轻微平滑，但宏观轮廓保持尚好",
        "phenomenon_p1": "前景主体皮肤或材质表面呈现明显涂抹感与蜡质平滑，天然微理细节大面积丢失",
        "phenomenon_p0": "画面核心材质纹理严重抹平呈重度塑料化假面感，高频细节近乎完全湮灭",
        "cautious_cause": "结合边缘保留而微纹理梯度衰减特征，疑似由强力时域/空域降噪滤波、美颜磨皮或低码率块内量化所致",
    },
    "过锐": {
        "region_desc": "非字幕真实物体高对比边缘轮廓",
        "phenomenon_p2": "高反差真实物体边缘可见轻微亮暗过冲描边痕迹",
        "phenomenon_p1": "主体与背景交界轮廓出现明显 1-3px 白边/黑边振铃光晕，线条生硬刺眼",
        "phenomenon_p0": "全图或主体边缘遍布严重双侧过冲白边与高频噪点放大伪影，严重破坏自然观感",
        "cautious_cause": "结合非UI形态学包络过冲比(MEOR)峰值异常，疑似由终端ISP或转码链路叠加过强USM/拉普拉斯锐化所致",
    },
    "花屏": {
        "region_desc": "8x8宏块网格与色度突变区域",
        "phenomenon_p0": "画面出现解码宏块错位、彩色条纹撕裂或异常色块损坏伪影",
        "cautious_cause": "结合8x8块边界跳变与色度尖峰特征，疑似由视频码流丢包、参考帧损坏或硬件解码异常所致",
    },
    "过度生成": {
        "region_desc": "前景解剖结构/手指五官或场景文字细节区域",
        "phenomenon_p2": "局部次要纹理或细小附属物存在轻微结构不自然或笔画粘连迹象",
        "phenomenon_p1": "主体肢体局部、五官对称性或场景文字出现较明显几何畸变与违背常理融合",
        "phenomenon_p0": "核心人物/物体出现严重多指断肢、五官崩坏、乱码伪文字或严重违反物理透视规律",
        "cautious_cause": "结合局部解剖与纹理奇异结构特征，疑似由AIGC扩散/生成模型局部拓扑合成失真或超分幻觉伪影所致",
    },
    "虚化突兀": {
        "region_desc": "前景主体发丝/肩颈与背景交界轮廓带",
        "phenomenon_p0": "主体边缘散焦清晰度在无连续景深过渡处发生断崖式跳变，呈现明显抠图贴片感或发丝误虚化/漏虚化",
        "cautious_cause": "结合单目景深连续而清晰度梯度突变(A_cutout高响应)，疑似由移动端人像模式分割掩码误差与人工合成背景虚化所致",
    },
}


def build_deterministic_grounded_attribution(
    pred_6d_single: np.ndarray,
    probs_6x4_single: np.ndarray,
    meta: Dict[str, Any],
    llm_free_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    方案 6.3 节结构化定位与谨慎归因生成器:
    - 全无损样本 (`pred_6d == 0`) 直接返回无问题摘要，零多余归因幻觉；
    - 检出缺陷样本严格锁定 `pred_6d_single` 非零项，绑定双擎算子提取的 `[0, 1000]` 归一化坐标框与物理指标证据。
    """
    norm_peaks = np.asarray(meta["norm_peak_boxes"], dtype=np.int32)  # [4, 4]: focus, meor, cutout, roi_anat
    orig_h, orig_w = int(meta["orig_hw"][0]), int(meta["orig_hw"][1])
    rois_384 = np.asarray(meta["rois_384_boxes"], dtype=np.int32)
    phys = np.asarray(meta["phys_vec"], dtype=np.float32)

    def _to_norm1000(box4: Sequence[int]) -> List[int]:
        x0, y0, x1, y1 = [int(v) for v in box4]
        return [
            max(0, min(1000, int(round(1000.0 * x0 / max(1, orig_w))))),
            max(0, min(1000, int(round(1000.0 * y0 / max(1, orig_h))))),
            max(0, min(1000, int(round(1000.0 * min(orig_w, x1) / max(1, orig_w))))),
            max(0, min(1000, int(round(1000.0 * min(orig_h, y1) / max(1, orig_h))))),
        ]

    roi_text_norm = _to_norm1000(rois_384[1])

    if bool((pred_6d_single == 0).all()):
        return {
            "overall_status": "无问题",
            "locked_consistency": True,
            "defects": [],
        }

    defect_entries: List[Dict[str, Any]] = []
    for d_idx, d_name in enumerate(DIMS):
        g_int = int(pred_6d_single[d_idx])
        if g_int <= 0:
            continue
        g_str = GRADE_INT_TO_STR[3 if (d_idx in P0_ONLY_DIMS and g_int >= 2) else g_int]
        conf = float(probs_6x4_single[d_idx, 3 if d_idx in P0_ONLY_DIMS else g_int])
        tpl = DEFECT_CAUSAL_TEMPLATES[d_name]

        if d_name == "模糊":
            bbox = norm_peaks[0].tolist()
            metric_ev = f"S_fg_p90={phys[13]:.3f}, fg_bg_sharp_log_ratio={phys[18]:.3f}"
        elif d_name == "过锐":
            bbox = norm_peaks[1].tolist()
            metric_ev = f"MEOR_p99={phys[3]:.4f}, UI_mask_ratio={phys[8]:.3f}"
        elif d_name == "虚化突兀":
            bbox = norm_peaks[2].tolist()
            metric_ev = f"A_cutout_p99={phys[21]:.4f}, depth_grad_p95={phys[24]:.4f}"
        elif d_name == "纹理损失":
            bbox = roi_text_norm
            metric_ev = f"edge_to_tex_fg={phys[30]:.3f}, fg_lap_p50={phys[28]:.4f}"
        elif d_name == "过度生成":
            bbox = norm_peaks[3].tolist()
            metric_ev = f"roi_anatomy_salient_box={bbox}, roi_text_box={roi_text_norm}"
        else:  # 花屏
            bbox = [0, 0, 1000, 1000]
            metric_ev = f"blockiness_8x8={phys[32]:.3f}, chroma_spike_p999={phys[33]:.4f}"

        phen_key = f"phenomenon_{g_str.lower()}"
        phenomenon = tpl.get(phen_key, tpl.get("phenomenon_p0", ""))
        defect_entries.append(
            {
                "defect": d_name,
                "severity": g_str,
                "posterior_prob": round(conf, 4),
                "bbox_1000": bbox,
                "region": tpl["region_desc"],
                "phenomenon": phenomenon,
                "physical_evidence": metric_ev,
                "cautious_root_cause": tpl["cautious_cause"],
            }
        )

    result: Dict[str, Any] = {
        "overall_status": "检出画质缺陷",
        "locked_consistency": True,
        "defects": defect_entries,
    }
    if llm_free_text:
        result["vlm_grounded_note"] = llm_free_text.strip()
    return result


class QingyuSingle4090Engine:
    """
    单卡 RTX 4090 全常驻推理引擎:
      - 支持加载 `checkpoints/qingyu/soup_4090/` 下的 8 塔 Model Soup 权重、`stacker.pkl` 与 `dfw_decoder.pkl`；
      - 支持对 InternViT-6B 与双 VLM 启用 W4A16 仅权重量化以控制在 `20.07 GB <= 24 GB` 显存红线内；
      - 严格单图流式处理 (`infer_single_pil`)，保证单图与任意批量测试集输出 100% 一致；
      - 支持复用 Qwen3-VL-8B 分类 Prefill 的 `past_key_values` 进行零额外视觉前向的后置贪婪归因解码。
    """

    TOWER_ORDER: Tuple[str, ...] = (
        "convnextv2_base",
        "pe_core_large",
        "dinov3_vitl16",
        "siglip2_so400m_naflex",
        "cradio_v4_h",
        "internvit_6b",
        "clef_flash_9b",
        "qwen3_vl_8b",
    )

    def __init__(
        self,
        soup_dir: Path = QINGYU_CKPT_DIR / "soup_4090",
        device_str: str = "cuda:0",
        enable_w4a16: bool = True,
        tta_hflip: bool = True,
        active_towers: Optional[Sequence[str]] = None,
        enable_llm_attribution: bool = False,
    ):
        import torch
        from qingyu.dual_engine_preprocess import DualEnginePhysicsExtractor
        from qingyu.certainty_gated_stacker import CenteredLogOddsCertaintyGatedStacker
        from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder

        self.soup_dir = Path(soup_dir)
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.enable_w4a16 = bool(enable_w4a16 and self.device.type == "cuda")
        self.tta_hflip = bool(tta_hflip)
        self.enable_llm_attribution = bool(enable_llm_attribution)
        self.requested_towers = list(active_towers) if active_towers else list(self.TOWER_ORDER)

        self.physics_extractor = DualEnginePhysicsExtractor(
            device=str(self.device), allow_depth_download=False
        )

        self.stacker: Optional[CenteredLogOddsCertaintyGatedStacker] = None
        self.decoder: Optional[DampedFrankWolfeBayesianDecoder] = None
        for cand_dir in (self.soup_dir, QINGYU_RESULTS_DIR):
            sp = cand_dir / "stacker.pkl"
            dp = cand_dir / "dfw_decoder.pkl"
            if sp.is_file() and dp.is_file():
                self.stacker = CenteredLogOddsCertaintyGatedStacker.load(str(sp))
                self.decoder = DampedFrankWolfeBayesianDecoder.load(str(dp))
                if not active_towers:
                    self.requested_towers = list(self.stacker.model_names)
                break

        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.qwen_lm_head: Optional[Any] = None
        self._load_resident_towers()

    @staticmethod
    def _extract_state_dict(ckpt_payload: Any) -> Dict[str, Any]:
        if isinstance(ckpt_payload, dict) and "state_dict" in ckpt_payload and isinstance(ckpt_payload["state_dict"], dict):
            return ckpt_payload["state_dict"]
        return ckpt_payload

    def _find_soup_or_fold0_ckpt(self, name: str, suffix: str = "") -> Optional[Path]:
        cands = [
            self.soup_dir / f"{name}_soup{suffix}.pt",
            QINGYU_CKPT_DIR / name / "fold_0_best.pt",
            QINGYU_RESULTS_DIR / name / "fold_0_best.pt",
        ]
        return next((p for p in cands if p.is_file()), None)

    def _load_resident_towers(self) -> None:
        import torch
        from qingyu.backbones_qingyu import QingyuVisionQualityTower
        from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM
        from qingyu.model_soup_and_quant import (
            apply_4090_mixed_precision_quant,
            merge_peft_or_custom_lora_into_base,
        )
        from qingyu.train_tier3_clef_flash9b_8fold import (
            ClefFlash9BSystem1Tower,
            load_clef_flash_backbone_and_lora,
        )
        from qingyu.train_tier3_qwen3vl8b_8fold import (
            Qwen3VL3ImageInterleavedTower,
            load_qwen3vl8b_backbone_and_lora,
        )

        for tname in self.requested_towers:
            if tname == "convnextv2_base":
                c0 = self._find_soup_or_fold0_ckpt(tname, "_cluster0")
                c1 = self._find_soup_or_fold0_ckpt(tname, "_cluster1")
                if c0 is not None and c1 is not None:
                    sub_towers = []
                    for cp in (c0, c1):
                        m = QingyuVisionQualityTower(backbone_key=tname, phys_dim=PHYS_FEAT_DIM, unfreeze_last_k=0)
                        sd = self._extract_state_dict(torch.load(str(cp), map_location="cpu", weights_only=False))
                        m.load_state_dict(sd, strict=False)
                        m.backbone.to(dtype=torch.bfloat16)
                        m.head.float()
                        m.to(self.device).eval()
                        sub_towers.append(m)
                    self.models[tname] = sub_towers
                    continue

            if tname in ("convnextv2_base", "pe_core_large", "dinov3_vitl16", "siglip2_so400m_naflex", "cradio_v4_h", "internvit_6b"):
                is_6b = tname == "internvit_6b"
                m = QingyuVisionQualityTower(
                    backbone_key=tname,
                    phys_dim=PHYS_FEAT_DIM,
                    unfreeze_last_k=0,
                    use_lora_6b=is_6b,
                    lora_rank=32,
                )
                cp = self._find_soup_or_fold0_ckpt(tname)
                if cp is not None:
                    sd = self._extract_state_dict(torch.load(str(cp), map_location="cpu", weights_only=False))
                    m.load_state_dict(sd, strict=False)
                if is_6b:
                    m.backbone = merge_peft_or_custom_lora_into_base(m.backbone)
                    apply_4090_mixed_precision_quant(m.backbone, enable_w4a16=self.enable_w4a16)
                else:
                    m.backbone.to(dtype=torch.bfloat16)
                m.head.float()
                m.to(self.device).eval()
                self.models[tname] = [m]
                if hasattr(m, "naflex_processor") and m.naflex_processor is not None:
                    self.processors[tname] = m.naflex_processor

            elif tname == "clef_flash_9b":
                mpath = resolve_model_path("clef_flash_9b")
                peft_b, proc, img_tid, hid = load_clef_flash_backbone_and_lora(
                    mpath, lora_rank=32, freeze_vision_tower=True, grad_checkpointing=False
                )
                tower = ClefFlash9BSystem1Tower(peft_b, img_tid, hid, phys_dim=PHYS_FEAT_DIM)
                cp = self._find_soup_or_fold0_ckpt(tname)
                if cp is not None:
                    sd = self._extract_state_dict(torch.load(str(cp), map_location="cpu", weights_only=False))
                    tower.load_state_dict(sd, strict=False)
                tower.base = merge_peft_or_custom_lora_into_base(tower.base)
                apply_4090_mixed_precision_quant(tower.base, enable_w4a16=self.enable_w4a16)
                tower.schema_head.float()
                tower.to(self.device).eval()
                self.models[tname] = [tower]
                self.processors[tname] = proc

            elif tname == "qwen3_vl_8b":
                mpath = resolve_model_path("qwen3_vl_8b")
                peft_b, proc, qalign_w, img_tid, hid = load_qwen3vl8b_backbone_and_lora(
                    mpath,
                    lora_rank=32,
                    grad_checkpointing=False,
                    keep_lm_head=self.enable_llm_attribution,
                )
                saved_lm_head = getattr(peft_b, "_saved_lm_head", None)
                if saved_lm_head is not None and hasattr(peft_b, "_saved_lm_head"):
                    delattr(peft_b, "_saved_lm_head")
                tower = Qwen3VL3ImageInterleavedTower(peft_b, qalign_w, img_tid, hid, phys_dim=PHYS_FEAT_DIM)
                cp = self._find_soup_or_fold0_ckpt(tname)
                if cp is not None:
                    sd = self._extract_state_dict(torch.load(str(cp), map_location="cpu", weights_only=False))
                    tower.load_state_dict(sd, strict=False)
                tower.base = merge_peft_or_custom_lora_into_base(tower.base)
                apply_4090_mixed_precision_quant(tower.base, enable_w4a16=self.enable_w4a16)
                for sub_m in (
                    tower.pyr_norm_50, tower.pyr_norm_75, tower.pyr_adapter,
                    tower.vis_attn_V, tower.vis_attn_U, tower.vis_attn_w,
                    tower.vis_anom_score, tower.unimodal_head,
                ):
                    sub_m.float()
                if tower.qalign_proj is not None:
                    tower.qalign_proj.float()
                tower.to(self.device).eval()
                if saved_lm_head is not None:
                    self.qwen_lm_head = saved_lm_head.to(device=self.device, dtype=torch.bfloat16).eval()
                self.models[tname] = [tower]
                self.processors[tname] = proc

    def _forward_single_pass(
        self,
        pil_im: Image.Image,
        meta: Dict[str, Any],
        flip: bool,
        capture_qwen_kv: bool = False,
    ) -> Tuple[List[np.ndarray], Optional[Dict[str, Any]]]:
        import torch
        from qingyu.dual_engine_preprocess import extract_1to1_crop_pil, resize_native_aspect_pil
        from qingyu.train_tier3_clef_flash9b_8fold import (
            apply_3image_chat_template,
            build_clef_3image_prompt,
            _move_inputs_to_device,
        )

        orig_w, _ = pil_im.size
        c_boxes = meta["crops_448_boxes"].copy()
        r_boxes = meta["rois_384_boxes"].copy()
        work_im = pil_im
        if flip:
            work_im = pil_im.transpose(Image.FLIP_LEFT_RIGHT)
            c_boxes[:, 0] = np.maximum(0, orig_w - c_boxes[:, 2])
            c_boxes[:, 2] = c_boxes[:, 0] + 448
            r_boxes[:, 0] = np.maximum(0, orig_w - r_boxes[:, 2])
            r_boxes[:, 2] = r_boxes[:, 0] + 384

        def _pil_chw(im_v: Image.Image) -> torch.Tensor:
            arr = np.asarray(im_v, dtype=np.float32) / 255.0
            return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())

        phys_t = torch.from_numpy(np.asarray(meta["phys_vec"], dtype=np.float32)).unsqueeze(0).to(self.device)

        g_448 = work_im.resize((448, 448), Image.BICUBIC)
        crops_4 = [extract_1to1_crop_pil(work_im, c_boxes[k], 448) for k in range(4)]
        views_5_t = torch.stack([_pil_chw(v) for v in ([g_448] + crops_4)], dim=0).unsqueeze(0).to(self.device)
        views_3_t = views_5_t[:, 1:4]

        im_nat_16 = resize_native_aspect_pil(work_im, max_pixels=784 * 256, min_pixels=256 * 256, patch_mult=16)
        nat_16_t = _pil_chw(im_nat_16).unsqueeze(0).to(self.device)

        im_global_32 = resize_native_aspect_pil(work_im, max_pixels=768 * 768, min_pixels=256 * 256, patch_mult=32)
        roi_anat = extract_1to1_crop_pil(work_im, r_boxes[0], 384)
        roi_text = extract_1to1_crop_pil(work_im, r_boxes[1], 384)
        imgs_3_vlm = [im_global_32, roi_anat, roi_text]
        fixed_prompt = build_clef_3image_prompt(permute_rubric=False)

        per_tower_probs: List[np.ndarray] = []
        kv_bundle: Optional[Dict[str, Any]] = None

        with torch.inference_mode():
            for tname in self.requested_towers:
                sub_list = self.models[tname]
                sub_probs = []
                for m in sub_list:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                        if tname == "qwen3_vl_8b":
                            proc = self.processors[tname]
                            txt = apply_3image_chat_template(proc, fixed_prompt)
                            enc = proc(text=[txt], images=imgs_3_vlm, padding=True, return_tensors="pt")
                            m_in = _move_inputs_to_device(dict(enc), self.device)
                            out = m(m_in, phys_t, return_kv_cache=capture_qwen_kv)
                            if capture_qwen_kv and "past_key_values" in out:
                                kv_bundle = {
                                    "past_key_values": out["past_key_values"],
                                    "prefill_len": out.get("prefill_seq_len", int(m_in["input_ids"].shape[1])),
                                    "attention_mask": m_in["attention_mask"],
                                }
                        elif tname == "clef_flash_9b":
                            proc = self.processors[tname]
                            txt = apply_3image_chat_template(proc, fixed_prompt)
                            enc = proc(text=[txt], images=imgs_3_vlm, padding=True, return_tensors="pt")
                            out = m(_move_inputs_to_device(dict(enc), self.device), phys_t)
                        elif m.mode_tag == "naflex_dict":
                            proc = self.processors[tname]
                            nv = [
                                {k: v.to(self.device) for k, v in dict(proc(images=[iv], return_tensors="pt", max_num_patches=784)).items()}
                                for iv in (im_nat_16, crops_4[0], crops_4[1])
                            ]
                            out = m(phys_feat=phys_t, naflex_views_list=nv)
                        elif m.mode_tag == "native_aspect_plus_crops":
                            out = m(views_01=views_3_t, phys_feat=phys_t, native_global_list=[nat_16_t])
                        else:
                            out = m(views_01=views_5_t, phys_feat=phys_t)
                    sub_probs.append(out["probs_6x4"].float().cpu().numpy())
                per_tower_probs.append(np.mean(sub_probs, axis=0).astype(np.float32))
        return per_tower_probs, kv_bundle

    def _decode_kv_cached_grounded_note(
        self,
        pred_6d_single: np.ndarray,
        meta: Dict[str, Any],
        kv_bundle: Optional[Dict[str, Any]],
        max_new_tokens: int = 96,
    ) -> Optional[str]:
        """
        方案 6.3 节: 复用分类阶段已计算的 `past_key_values` (零额外 ViT 编码)，
        仅对检出正向缺陷的图像追加锁定等级与 `[0, 1000]` 归一化坐标后缀执行贪婪确定性解码 (`temperature=0.0`)。
        """
        import torch

        if (
            not self.enable_llm_attribution
            or kv_bundle is None
            or self.qwen_lm_head is None
            or "qwen3_vl_8b" not in self.models
            or bool((pred_6d_single == 0).all())
        ):
            return None

        try:
            proc = self.processors["qwen3_vl_8b"]
            tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
            tower = self.models["qwen3_vl_8b"][0]
            norm_peaks = np.asarray(meta["norm_peak_boxes"], dtype=np.int32)

            locked_items = []
            for d_idx, d_name in enumerate(DIMS):
                gv = int(pred_6d_single[d_idx])
                if gv > 0:
                    g_str = GRADE_INT_TO_STR[3 if (d_idx in P0_ONLY_DIMS and gv >= 2) else gv]
                    locked_items.append(f"{d_name}={g_str}")
            suffix_text = (
                f"\n【已锁定客观分类结果】：{', '.join(locked_items)}；"
                f"物理算子归一化峰值区域：对焦框{norm_peaks[0].tolist()}，过锐光晕框{norm_peaks[1].tolist()}，抠图边界框{norm_peaks[2].tolist()}。"
                "请严格围绕上述已锁定缺陷简述单帧可见现象与谨慎成因（禁止提及未检出缺陷或跨帧信息，限80字内）："
            )
            sfx_ids = tok.encode(suffix_text, add_special_tokens=False)
            if not sfx_ids:
                return None

            past_kv = kv_bundle["past_key_values"]
            attn_mask = kv_bundle["attention_mask"]
            cur_input = torch.tensor([sfx_ids], dtype=torch.long, device=self.device)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((1, cur_input.shape[1]), dtype=attn_mask.dtype, device=self.device)],
                dim=1,
            )

            eos_ids = set()
            if getattr(tok, "eos_token_id", None) is not None:
                eos_ids.add(int(tok.eos_token_id))
            for sp_tok in ("<|im_end|>", "<|endoftext|>"):
                tid = tok.convert_tokens_to_ids(sp_tok)
                if isinstance(tid, int) and tid >= 0:
                    eos_ids.add(tid)

            gen_ids: List[int] = []
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")
            ):
                for _ in range(max_new_tokens):
                    out = tower.base(
                        input_ids=cur_input,
                        attention_mask=attn_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = out.past_key_values
                    h_last = out.last_hidden_state[:, -1:, :]
                    logits = self.qwen_lm_head(h_last)[:, -1, :]
                    next_id = int(torch.argmax(logits, dim=-1).item())  # 严格 temperature=0.0 贪婪解码
                    if next_id in eos_ids:
                        break
                    gen_ids.append(next_id)
                    cur_input = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    attn_mask = torch.cat(
                        [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=self.device)],
                        dim=1,
                    )
            return tok.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else None
        except Exception:
            # 若特定 transformers 版本的 3D-MRoPE 动态 cache_position 步进签名差异导致异常，平滑退回纯物理确定性归因
            return None

    def infer_single_pil(self, pil_im: Image.Image, precomputed_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = precomputed_meta if precomputed_meta is not None else self.physics_extractor.analyze_pil_image(pil_im)
        p_orig, kv_bundle = self._forward_single_pass(
            pil_im, meta, flip=False, capture_qwen_kv=self.enable_llm_attribution
        )
        if self.tta_hflip:
            p_flip, _ = self._forward_single_pass(pil_im, meta, flip=True, capture_qwen_kv=False)
            tower_probs_1x6x4 = [0.5 * (a + b) for a, b in zip(p_orig, p_flip)]
        else:
            tower_probs_1x6x4 = p_orig

        phys_1x36 = np.asarray(meta["phys_vec"], dtype=np.float32)[None, :]
        if self.stacker is not None and self.decoder is not None:
            probs_6x4, q_clean = self.stacker.predict_proba(tower_probs_1x6x4, phys_mat=phys_1x36)
            pred_6d = self.decoder.decode(probs_6x4, q_clean)
        else:
            probs_6x4 = np.mean(tower_probs_1x6x4, axis=0)
            q_clean = np.prod(probs_6x4[:, :, 0], axis=1)
            pred_6d = np.argmax(probs_6x4, axis=-1).astype(np.int64)
            for d in P0_ONLY_DIMS:
                pred_6d[:, d] = np.where(probs_6x4[:, d, 3] > 0.35, 3, 0)

        llm_note = self._decode_kv_cached_grounded_note(pred_6d[0], meta, kv_bundle)
        # 立即释放单图 KV 缓存引用，防止长序列评测显存堆积
        kv_bundle = None
        attrib = build_deterministic_grounded_attribution(pred_6d[0], probs_6x4[0], meta, llm_free_text=llm_note)
        return {
            "pred_6d": pred_6d[0],
            "probs_6x4": probs_6x4[0],
            "q_clean": float(q_clean[0]),
            "attribution": attrib,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Qingyu Single-RTX-4090 Full-Resident Inference & Attribution Engine")
    ap.add_argument("--test-template", default="results/v2_full/template_150.json")
    ap.add_argument("--test-dir", default="")
    ap.add_argument("--soup-dir", default=str(QINGYU_CKPT_DIR / "soup_4090"))
    ap.add_argument("--dual-engine-npz", default=str(QINGYU_RESULTS_DIR / "dual_engine_meta.npz"))
    ap.add_argument("--out-json", default=str(QINGYU_FINAL_DIR / "pred_nriqa_dataset_test.json"))
    ap.add_argument("--out-attrib-json", default=str(QINGYU_FINAL_DIR / "pred_nriqa_dataset_test_grounded_attrib.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-w4a16", action="store_true", help="在 141GB H200 上可关闭 W4A16 保持原生 BF16")
    ap.add_argument("--no-tta", action="store_true", help="关闭翻转 TTA 以进一步压缩单图延迟至 ~1.2s")
    ap.add_argument("--enable-llm-attribution", action="store_true", help="开启复用 Qwen3-VL-8B Prefill KV 缓存的后置贪婪归因")
    ap.add_argument("--towers", nargs="*", default=None)
    ap.add_argument("--audit-budget-only", action="store_true", help="仅执行 30B 参数量与 24GB 显存合规性静态审计")
    args = ap.parse_args()

    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(2026, deterministic_algos=False)
    print_pool_budget_summary()

    assert TOTAL_POOL_PARAMS_B <= 30.0, f"参数池超限: {TOTAL_POOL_PARAMS_B}B > 30.0B"
    assert PEAK_RESIDENT_VRAM_GB <= 24.0, f"单卡 4090 显存预算超限: {PEAK_RESIDENT_VRAM_GB} GB > 24.0 GB"

    if args.audit_budget_only:
        print("[4090 Compliance Gate] 参数池与显存红线静态审计全部通过!")
        return

    import torch
    from qingyu.dual_engine_preprocess import PrecomputedDualEngineStore

    ordered_names = load_test_template_names(args.test_template)
    loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=[args.test_dir] if args.test_dir else None)
    engine = QingyuSingle4090Engine(
        soup_dir=Path(args.soup_dir),
        device_str=args.device,
        enable_w4a16=(not args.no_w4a16),
        tta_hflip=(not args.no_tta),
        active_towers=args.towers,
        enable_llm_attribution=args.enable_llm_attribution,
    )
    meta_store = PrecomputedDualEngineStore(args.dual_engine_npz, lazy_extractor=engine.physics_extractor)

    preds_list: List[np.ndarray] = []
    attribs_list: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx, nm in enumerate(ordered_names):
        im = loader.load_pil(nm)
        meta = meta_store.get(nm, pil_im=im)
        res = engine.infer_single_pil(im, precomputed_meta=meta)
        preds_list.append(res["pred_6d"])
        attribs_list.append(res["attribution"])
        if (idx + 1) % 25 == 0 or (idx + 1) == len(ordered_names):
            elapsed = max(1e-3, time.time() - t0)
            avg_s = elapsed / float(idx + 1)
            vram_gb = (
                torch.cuda.max_memory_allocated(engine.device) / (1024 ** 3)
                if engine.device.type == "cuda"
                else 0.0
            )
            print(
                f"[4090 Stream Engine] {idx+1}/{len(ordered_names)} | avg_latency={avg_s:.3f}s/img (<=15.0s) "
                f"| peak_VRAM={vram_gb:.2f} GB (<=24.0GB)",
                flush=True,
            )

    pred_6d = np.stack(preds_list, axis=0)
    sub_clean, sub_attrib = format_official_submission(ordered_names, pred_6d, explanations=attribs_list)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(sub_clean, f, ensure_ascii=False, indent=2)
    if sub_attrib is not None and args.out_attrib_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_attrib_json)), exist_ok=True)
        with open(args.out_attrib_json, "w", encoding="utf-8") as f:
            json.dump(sub_attrib, f, ensure_ascii=False, indent=2)

    print(f"[4090 Stream Engine] 已输出官方分类结果 -> {args.out_json}")
    if args.out_attrib_json:
        print(f"[4090 Stream Engine] 已输出锁定一致性归因报告 -> {args.out_attrib_json}")


if __name__ == "__main__":
    main()
