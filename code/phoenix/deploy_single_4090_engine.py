#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/deploy_single_4090_engine.py
=========================================
Breakthrough 7: Compliant Single-RTX-4090 (`<=24GB`, `19.425B <= 30.0B`, `~5.8s/img <= 15s`)
All-Resident Deployment Engine (with 2x Horizontal-Flip Single-Image TTA & Dual-Caliber Export),
Post-Hoc Label-Locked <=40-Char Attribution Generator, and CUSUM Video Hysteresis Aggregator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(THIS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(THIS_DIR))

from phoenix.config import (
    DIMS,
    GRADE_INT_TO_STR,
    P0_ONLY_DIMS,
    PEAK_4090_VRAM_GB,
    PHOENIX_CKPT_DIR,
    PHOENIX_SUB_DIR,
    PureRealImageResolver,
    TOTAL_PARAMS_B,
    export_and_verify_submission,
    load_ordered_test_names,
    print_budget_audit,
    resolve_local_or_hf_model,
    set_deterministic_seed,
)
from phoenix.inductive_anchor_and_joint_decoder import (
    InductiveDualAnchorBank,
    JointCompetitionUtilityDecoder,
    fuse_multi_pillar_outputs,
)
from phoenix.native_physics_and_views import (
    CoCDepthDefocusProbe,
    compute_1to1_wavelet_and_artifact_maps,
    resize_native_aspect_pil,
    select_scrutiny_1to1_patches,
)
from phoenix.train_pillar12_native_vlm import (
    NativeVLMPillarModel,
    detect_lora_rank_from_state_dict,
)
from phoenix.train_pillar3_hires_physics import (
    Pillar3HiResPhysicsSpecialist,
    build_pillar3_sample_tensors,
)

_DEFECT_EVIDENCE_PHRASES: Dict[str, str] = {
    "模糊": "边缘梯度弥散且高频细节清晰度下降",
    "纹理损失": "细节涂抹呈蜡质感且微纹理丢失",
    "过锐": "边缘呈双极性振铃白边与锐化光晕",
    "花屏": "局部宏块撕裂且色度通道异常溢出",
    "过度生成": "局部结构伪影与非自然生成畸变",
    "虚化突兀": "同景深主体边缘清晰度断崖跳变",
}


def generate_locked_attribution(
    pred_6d_row: np.ndarray,
    probs_6x4_row: np.ndarray,
    crop_meta: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    active_items = []
    region_hint = str(crop_meta[0].get("region_zh", "画面主体区")) if crop_meta else "画面主体区"
    hf_region = str(crop_meta[1].get("region_zh", region_hint)) if len(crop_meta) > 1 else region_hint

    for d_idx, d_name in enumerate(DIMS):
        g_int = int(pred_6d_row[d_idx])
        if g_int <= 0:
            continue
        g_str = GRADE_INT_TO_STR[g_int]
        conf = float(probs_6x4_row[d_idx, g_int])
        loc = hf_region if d_name in ("过锐", "纹理损失", "花屏") else region_hint
        cue = _DEFECT_EVIDENCE_PHRASES[d_name]
        summary_40 = f"{loc}检出{g_str}级{d_name}：{cue}"[:40]
        active_items.append(
            {
                "defect": d_name,
                "severity": g_str,
                "confidence": round(conf, 4),
                "region": loc,
                "reason_40ch": summary_40,
            }
        )

    if not active_items:
        return {
            "is_clean": True,
            "summary_40ch": "全局与局部1:1切片边缘自然、纹理连续，未见画质缺陷",
            "defects": [],
        }

    top_item = max(active_items, key=lambda x: (x["severity"] == "P0", x["confidence"]))
    return {
        "is_clean": False,
        "summary_40ch": top_item["reason_40ch"],
        "crop_boxes": [m.get("box_xyxy") for m in crop_meta],
        "defects": active_items,
    }


class PhoenixVideoHysteresisAggregator:
    def __init__(
        self,
        drift_k: float = 0.35,
        cusum_threshold: float = 2.40,
        instant_glitch_prob: float = 0.96,
        scene_cut_cos_sim: float = 0.65,
    ) -> None:
        self.drift_k = drift_k
        self.cusum_threshold = cusum_threshold
        self.instant_glitch_prob = instant_glitch_prob
        self.scene_cut_cos_sim = scene_cut_cos_sim

    def aggregate_video_clip(
        self,
        frame_probs_6x4: np.ndarray,
        frame_embs: Optional[np.ndarray] = None,
        decoder: Optional[JointCompetitionUtilityDecoder] = None,
    ) -> Dict[str, object]:
        t_len = frame_probs_6x4.shape[0]
        assert t_len >= 1
        probs = np.clip(frame_probs_6x4, 1e-6, 1.0 - 1e-6)

        cusum_peak = np.zeros(6, dtype=np.float32)
        s_cur = np.zeros(6, dtype=np.float32)

        for t in range(t_len):
            if frame_embs is not None and t > 0:
                cos_prev = float(np.dot(frame_embs[t], frame_embs[t - 1]))
                if cos_prev < self.scene_cut_cos_sim:
                    s_cur *= 0.25

            p_clean_d = probs[t, :, 0]
            p_def_d = 1.0 - p_clean_d
            llr = np.log(p_def_d / p_clean_d)
            s_cur = np.maximum(0.0, s_cur + llr - self.drift_k)
            cusum_peak = np.maximum(cusum_peak, s_cur)

        entropy = -np.sum(probs * np.log(probs), axis=-1)
        frame_w = np.exp(-1.2 * entropy)
        frame_w = frame_w / np.maximum(frame_w.sum(axis=0, keepdims=True), 1e-8)
        pooled_logp = np.sum(frame_w[:, :, None] * np.log(probs), axis=0)

        if decoder is not None:
            clean_l = np.array([float(np.sum(pooled_logp[:, 0]))], dtype=np.float32)
            vid_pred = decoder.decode(pooled_logp[None, :, :], clean_l)[0]
        else:
            vid_pred = np.argmax(pooled_logp, axis=-1)

        for d in range(6):
            if vid_pred[d] > 0 and cusum_peak[d] < self.cusum_threshold:
                if d == 3 and float(np.max(probs[:, 3, 3])) >= self.instant_glitch_prob:
                    vid_pred[d] = 3
                else:
                    vid_pred[d] = 0

        return {
            "video_grade_6d": vid_pred.astype(np.int64),
            "cusum_peak_6d": np.round(cusum_peak, 4).tolist(),
            "n_frames": t_len,
        }


class Single4090PhoenixEngine:
    def __init__(
        self,
        ckpt_dir: str = str(PHOENIX_CKPT_DIR),
        device_str: Optional[str] = None,
        mock_backbones: bool = False,
        enable_2x_hflip_tta: bool = True,
    ) -> None:
        assert TOTAL_PARAMS_B <= 30.0, f"Budget violation: {TOTAL_PARAMS_B}B > 30B"
        assert PEAK_4090_VRAM_GB <= 24.0, f"VRAM violation: {PEAK_4090_VRAM_GB}GB > 24GB"

        self.device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mock_backbones = mock_backbones
        self.enable_2x_hflip_tta = enable_2x_hflip_tta
        self.coc_probe = CoCDepthDefocusProbe(
            model_path=None if mock_backbones else resolve_local_or_hf_model("pillar3_depth_anything_s"),
            device=str(self.device),
        )

        ckpts_loaded: Dict[str, dict] = {}
        for name in ("pillar1_fulldata_swa.pt", "pillar2_fulldata_swa.pt", "pillar3_fulldata_swa.pt"):
            p = os.path.join(ckpt_dir, name)
            if os.path.exists(p) and not mock_backbones:
                ckpts_loaded[name] = torch.load(p, map_location="cpu")

        r1 = detect_lora_rank_from_state_dict(ckpts_loaded.get("pillar1_fulldata_swa.pt", {}), default_rank=32)
        r2 = detect_lora_rank_from_state_dict(ckpts_loaded.get("pillar2_fulldata_swa.pt", {}), default_rank=32)
        p3_raw = ckpts_loaded.get("pillar3_fulldata_swa.pt", {})
        p3_inner = p3_raw.get("state_dict", p3_raw) if isinstance(p3_raw, dict) else {}
        r3 = detect_lora_rank_from_state_dict(p3_inner, default_rank=24)
        self.p3_legacy_3view = bool(p3_inner and ("vision_tower.backbone_gate_logits" not in p3_inner))

        self.pillar1 = NativeVLMPillarModel(
            pillar_name="pillar1",
            quantize_4bit_infer=not mock_backbones,
            apply_dora=not mock_backbones,
            lora_rank=r1,
            mock_backbone=mock_backbones,
        ).to(self.device)
        self.pillar2 = NativeVLMPillarModel(
            pillar_name="pillar2",
            quantize_4bit_infer=not mock_backbones,
            apply_dora=not mock_backbones,
            lora_rank=r2,
            mock_backbone=mock_backbones,
        ).to(self.device)
        self.pillar3 = Pillar3HiResPhysicsSpecialist(
            mock_backbone=mock_backbones,
            lora_rank=r3,
            legacy_3view=self.p3_legacy_3view,
        ).to(self.device)

        for name, mod in (
            ("pillar1_fulldata_swa.pt", self.pillar1),
            ("pillar2_fulldata_swa.pt", self.pillar2),
            ("pillar3_fulldata_swa.pt", self.pillar3),
        ):
            if name in ckpts_loaded:
                sd = ckpts_loaded[name]
                mod.load_state_dict(sd.get("state_dict", sd), strict=False)
            mod.eval()

        bank_path = os.path.join(ckpt_dir, "inductive_anchor_bank.npz")
        if os.path.exists(bank_path):
            self.anchor_bank: Optional[InductiveDualAnchorBank] = InductiveDualAnchorBank.from_dict(
                dict(np.load(bank_path, allow_pickle=True))
            )
        else:
            self.anchor_bank = None

        dec_path = os.path.join(ckpt_dir, "joint_1024_decoder.json")
        self.strict_decoder: Optional[JointCompetitionUtilityDecoder] = None
        if os.path.exists(dec_path):
            with open(dec_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.decoder = JointCompetitionUtilityDecoder.from_json_dict(payload["decoder"])
            if "strict_decoder" in payload:
                self.strict_decoder = JointCompetitionUtilityDecoder.from_json_dict(payload["strict_decoder"])
        else:
            self.decoder = JointCompetitionUtilityDecoder()

    def _forward_one_orientation(self, pil_img):
        rgb_np = np.asarray(pil_img, dtype=np.uint8)
        phys_maps = compute_1to1_wavelet_and_artifact_maps(rgb_np)
        coc_field = self.coc_probe.compute_coc_jacobian_field(pil_img, phys_maps)
        crops, crop_meta = select_scrutiny_1to1_patches(
            pil_img=pil_img,
            phys_maps=phys_maps,
            coc_field=coc_field,
            patch_size=384,
            num_patches=2,
        )
        global_vlm = resize_native_aspect_pil(pil_img, max_pixels=784 * 784, grid_factor=28)
        vlm_views = [[global_vlm, crops[0], crops[1]]]

        out1 = self.pillar1(vlm_views, batch_ids=None, apply_batch_bias=False, device=self.device)
        out2 = self.pillar2(vlm_views, batch_ids=None, apply_batch_bias=False, device=self.device)

        p3_pack = build_pillar3_sample_tensors(
            pil_img,
            coc_probe=self.coc_probe,
            include_mid_tex=not self.p3_legacy_3view,
        )
        out3 = self.pillar3(
            views_01=p3_pack["views_tensor"].unsqueeze(0).to(self.device),
            phys_9ch=p3_pack["phys_9ch"].unsqueeze(0).to(self.device),
            probe_8ch=p3_pack["probe_8ch"].unsqueeze(0).to(self.device),
            batch_ids=None,
            apply_batch_bias=False,
        )
        return (out1, out2, out3), crop_meta

    @torch.no_grad()
    def infer_single_pil_image(self, pil_img) -> Dict[str, object]:
        from PIL import Image

        outs_orig, crop_meta = self._forward_one_orientation(pil_img)
        if self.enable_2x_hflip_tta and not self.mock_backbones:
            outs_flip, _ = self._forward_one_orientation(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            outs_flip = None

        pillar_np = []
        for m_idx in range(3):
            o0 = outs_orig[m_idx]
            if outs_flip is not None:
                o1 = outs_flip[m_idx]
                p = 0.5 * (o0["probs_6x4"].cpu().numpy() + o1["probs_6x4"].cpu().numpy())
                z = np.log(np.clip(p, 1e-6, 1.0))
                for d in P0_ONLY_DIMS:
                    z[:, d, 1] = -1e4
                    z[:, d, 2] = -1e4
                cl = 0.5 * (o0["clean_logit"].cpu().numpy() + o1["clean_logit"].cpu().numpy())
                vr = 0.5 * (o0["var_6d"].cpu().numpy() + o1["var_6d"].cpu().numpy())
                eb = o0["quality_emb"].cpu().numpy() + o1["quality_emb"].cpu().numpy()
                eb = eb / (np.linalg.norm(eb, axis=-1, keepdims=True) + 1e-8)
            else:
                p = o0["probs_6x4"].cpu().numpy()
                z = o0["logits_6x4"].cpu().numpy()
                cl = o0["clean_logit"].cpu().numpy()
                vr = o0["var_6d"].cpu().numpy()
                eb = o0["quality_emb"].cpu().numpy()
            pillar_np.append(
                {
                    "probs_6x4": p.astype(np.float32),
                    "logits_6x4": z.astype(np.float32),
                    "clean_logit": cl.astype(np.float32),
                    "var_6d": vr.astype(np.float32),
                    "quality_emb": eb.astype(np.float32),
                }
            )

        fused = fuse_multi_pillar_outputs(pillar_np, self.decoder.pillar_weights.tolist())
        logits_6x4 = fused["logits_6x4"]
        clean_logit = fused["clean_logit"]

        if self.anchor_bank is not None:
            logits_6x4, clean_logit = self.anchor_bank.transform_single_or_batch(
                fused["quality_emb"], logits_6x4, clean_logit
            )

        pred_6d = self.decoder.decode(logits_6x4, clean_logit)[0]
        pred_strict_6d = (
            self.strict_decoder.decode(logits_6x4, clean_logit)[0]
            if self.strict_decoder is not None
            else pred_6d
        )
        cal_probs, _, _ = self.decoder.calibrated_probabilities(logits_6x4, clean_logit)
        attrib = generate_locked_attribution(pred_6d, cal_probs[0], crop_meta)

        return {
            "pred_6d": pred_6d,
            "pred_strict_6d": pred_strict_6d,
            "probs_6x4": cal_probs[0],
            "quality_emb": fused["quality_emb"][0],
            "attribution": attrib,
        }


def run_self_test() -> None:
    from PIL import Image

    set_deterministic_seed(2026)
    print_budget_audit()

    engine = Single4090PhoenixEngine(mock_backbones=True)
    img_a = Image.fromarray(np.random.randint(0, 255, (720, 405, 3), dtype=np.uint8))
    img_b = Image.fromarray(np.random.randint(0, 255, (640, 360, 3), dtype=np.uint8))

    res_a1 = engine.infer_single_pil_image(img_a)
    _ = engine.infer_single_pil_image(img_b)
    res_a2 = engine.infer_single_pil_image(img_a)
    assert np.array_equal(res_a1["pred_6d"], res_a2["pred_6d"]), "Single-image decision drift!"
    assert np.allclose(res_a1["probs_6x4"], res_a2["probs_6x4"], atol=1e-6), "Probability drift!"
    assert len(res_a1["attribution"]["summary_40ch"]) <= 40, "Attribution exceeded 40 chars!"

    test150_names = load_ordered_test_names("data/manifest_test_150.csv")
    assert len(test150_names) == 150
    rng = np.random.RandomState(2026)
    synth_preds_150 = np.zeros((150, 6), dtype=np.int64)
    synth_preds_150[0] = [2, 1, 0, 3, 0, 0]

    with tempfile.TemporaryDirectory() as tmpd:
        p150 = os.path.join(tmpd, "pred_nriqa_dataset_test.json")
        export_and_verify_submission(test150_names, synth_preds_150, p150)
        names_3573 = load_ordered_test_names("public-2777fd37c166-blank-template.json")
        assert len(names_3573) == 3573
        p3573 = os.path.join(tmpd, "pred_3573_selfeval.json")
        export_and_verify_submission(names_3573, np.zeros((3573, 6), dtype=np.int64), p3573)

    agg = PhoenixVideoHysteresisAggregator()
    naive_or_fp, cusum_fp, n_vids = 0, 0, 200
    for _ in range(n_vids):
        clip_p = np.zeros((30, 6, 4), dtype=np.float32)
        clip_p[:, :, 0] = 0.90
        clip_p[:, :, 1:] = 0.10 / 3.0
        if rng.rand() < 0.22:
            t_spike = rng.randint(0, 30)
            clip_p[t_spike, 0, :] = [0.28, 0.12, 0.20, 0.40]
        if np.any(np.argmax(clip_p, axis=-1) > 0):
            naive_or_fp += 1
        if np.any(agg.aggregate_video_clip(clip_p)["video_grade_6d"] > 0):
            cusum_fp += 1

    assert cusum_fp < naive_or_fp
    print(
        f"[Single-4090 Engine Self-Test PASS] test150={len(test150_names)} & template3573={len(names_3573)} "
        f"JSON Schema Verified | Video 30-Frame Clean FPR: Naive-OR={naive_or_fp/n_vids*100:.1f}% -> "
        f"CUSUM-Hysteresis={cusum_fp/n_vids*100:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix Compliant Single-RTX-4090 Inference & Submission Exporter")
    parser.add_argument("--template", default="pred_nriqa_dataset_test.json")
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--ckpt-dir", default=str(PHOENIX_CKPT_DIR))
    parser.add_argument("--precomputed-npzs", nargs="*", default=[])
    parser.add_argument("--out-json", default=str(PHOENIX_SUB_DIR / "pred_nriqa_dataset_test.json"))
    parser.add_argument("--emit-attribution", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    set_deterministic_seed(2026)
    print_budget_audit()
    ordered_names = load_ordered_test_names(args.template)

    if args.precomputed_npzs:
        dec_path = os.path.join(args.ckpt_dir, "joint_1024_decoder.json")
        strict_dec = None
        if os.path.exists(dec_path):
            with open(dec_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            decoder = JointCompetitionUtilityDecoder.from_json_dict(payload["decoder"])
            if "strict_decoder" in payload:
                strict_dec = JointCompetitionUtilityDecoder.from_json_dict(payload["strict_decoder"])
        else:
            decoder = JointCompetitionUtilityDecoder()

        bank_path = os.path.join(args.ckpt_dir, "inductive_anchor_bank.npz")
        bank = (
            InductiveDualAnchorBank.from_dict(dict(np.load(bank_path, allow_pickle=True)))
            if os.path.exists(bank_path)
            else None
        )

        pillar_list = []
        for p in args.precomputed_npzs:
            z = np.load(p, allow_pickle=True)
            pos = {str(nm): i for i, nm in enumerate(z["names"])}
            idx = [pos[nm] for nm in ordered_names]
            pillar_list.append(
                {
                    "probs_6x4": z["probs_6x4"][idx],
                    "logits_6x4": z["logits_6x4"][idx],
                    "clean_logit": z["clean_logit"][idx],
                    "var_6d": z["var_6d"][idx],
                    "quality_emb": z["quality_emb"][idx],
                }
            )
        w = decoder.pillar_weights[: len(pillar_list)].tolist()
        fused = fuse_multi_pillar_outputs(pillar_list, w)
        logits_6x4, clean_l = fused["logits_6x4"], fused["clean_logit"]
        if bank is not None:
            logits_6x4, clean_l = bank.transform_single_or_batch(fused["quality_emb"], logits_6x4, clean_l)
        preds_6d = decoder.decode(logits_6x4, clean_l)
        export_and_verify_submission(ordered_names, preds_6d, args.out_json)
        if strict_dec is not None:
            strict_out = os.path.splitext(args.out_json)[0] + "_strict_doc2.json"
            export_and_verify_submission(ordered_names, strict_dec.decode(logits_6x4, clean_l), strict_out)
        print(f"[Phoenix Precomputed Export] Verified {len(ordered_names)} predictions -> {args.out_json}", flush=True)
        return

    resolver = PureRealImageResolver(extra_dirs=[args.test_dir] if args.test_dir else None)
    engine = Single4090PhoenixEngine(ckpt_dir=args.ckpt_dir, enable_2x_hflip_tta=not args.no_tta)

    t0 = time.time()
    preds_all = np.zeros((len(ordered_names), 6), dtype=np.int64)
    preds_strict_all = np.zeros((len(ordered_names), 6), dtype=np.int64)
    attribs_all: List[Dict[str, object]] = []

    for i, name in enumerate(ordered_names):
        pil_img = resolver.load_rgb_pil(name)
        res = engine.infer_single_pil_image(pil_img)
        preds_all[i] = res["pred_6d"]
        preds_strict_all[i] = res["pred_strict_6d"]
        attribs_all.append(res["attribution"])
        if (i + 1) % 25 == 0 or (i + 1) == len(ordered_names):
            elapsed = time.time() - t0
            sec_per_img = elapsed / (i + 1)
            assert sec_per_img <= 15.0, f"Latency {sec_per_img:.2f}s/img exceeded 15s single-4090 limit!"
            print(f"  [{i+1:4d}/{len(ordered_names)}] avg_latency={sec_per_img:.2f}s/img (<= 15.0s limit)", flush=True)

    attrib_path = (
        os.path.splitext(args.out_json)[0] + "_with_attribution.json"
        if args.emit_attribution
        else None
    )
    export_and_verify_submission(
        ordered_names=ordered_names,
        pred_6d=preds_all,
        out_json_path=args.out_json,
        attributions=attribs_all if args.emit_attribution else None,
        out_attrib_json_path=attrib_path,
    )
    if engine.strict_decoder is not None:
        strict_json = os.path.splitext(args.out_json)[0] + "_strict_doc2.json"
        export_and_verify_submission(ordered_names, preds_strict_all, strict_json)
        print(f"[Phoenix Dual-Caliber Export] Also saved pure Doc2-Strict JSON -> {strict_json}", flush=True)
    print(f"[Phoenix Single-4090 Deployment Complete] Saved cliff-protected JSON -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
