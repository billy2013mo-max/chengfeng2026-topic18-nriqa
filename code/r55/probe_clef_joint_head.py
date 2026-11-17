#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_clef_joint_head.py
=================================
Zero-training empirical probe of `models/clef-flash` WITH its native 233 MB pretrained
`joint_head.safetensors` (`joint_schema_model.py`: `score` for 4-grade dims, `noul` for P0-only dims)
and 3 native views `[img_global_768, roi_anatomy_384, roi_text_tex_384]`.

Shards the 3,573 labelled images across 8 GPUs (`--rank R --world-size W` or `--merge W`)
and evaluates:
  1) Standalone RPS, mAP14, per-defect AP, ECE, and nested EU Doc2_Strict;
  2) Ensemble blend delta when replacing `p1` (Qwen3-VL-8B) or blending into `fallback`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(REPO, "models", "clef-flash")
for _p in (MODEL_DIR, os.path.join(REPO, "code"), os.path.join(REPO, "code/next55"), os.path.join(REPO, "code/r55")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


QUESTIONS_SCHEMA: Dict[str, Dict[str, Any]] = {
    "模糊": {
        "type": "score",
        "instructions": "评估图像前景主体的真实失焦或动态模糊程度（注意：前景清晰但背景因光学浅景深自然虚化属于无问题）。",
        "criteria": [
            "无问题：前景主体对焦清晰，细节边缘扎实（允许正常的光学浅景深背景虚化）。",
            "P2（轻微模糊）：前景主体存在轻微失焦、轻微运动拖影或分辨力轻度下降，放大可见但不严重影响整体观看。",
            "P1（中度模糊）：前景主体明显模糊或重影，五官、文字或关键轮廓清晰度显著受损，正常观看即易察觉。",
            "P0（严重模糊）：全图或核心主体极度失焦糊成一片，完全无法看清基本结构与语义内容。",
        ],
    },
    "纹理损失": {
        "type": "score",
        "instructions": "评估前景皮肤、发丝、织物、地面等自然高频微纹理是否因降噪、超分或编码涂抹而丢失（塑料感/磨皮涂抹感）。",
        "criteria": [
            "无问题：自然微纹理（毛孔、发丝、布料纤维等）保留真实完整，无算法涂抹或塑料蜡像感。",
            "P2（轻微纹理损失）：局部皮肤或暗部/背景存在轻度降噪涂抹感，高频微纹理略有平滑但主体质感尚存。",
            "P1（中度纹理损失）：大面积前景皮肤、发丝或衣物纹理被明显抹平呈蜡像感或水彩涂抹状，但宏观轮廓尚存。",
            "P0（严重纹理损失）：全图高频细节彻底丧失，画面严重油画化、块状平坦化或极度塑料假脸感。",
        ],
    },
    "过锐": {
        "type": "score",
        "instructions": "评估真实景物与人物边缘是否存在锐化过度造成的黑白光晕振铃（Halo/Ringing）或生硬锯齿白边（严格排除后期叠加的清晰矢量花字/字幕描边）。",
        "criteria": [
            "无问题：真实物体边缘过渡自然，无黑白镶边光晕或高频颗粒噪点放大（正常锐利的后期字幕/UI描边计为无问题）。",
            "P2（轻微过锐）：局部高对比真实轮廓出现轻微 1-2 像素白边/黑边光晕或发丝轻微生硬感。",
            "P1（中度过锐）：人物轮廓、五官或前景边缘普遍存在明显黑白振铃光晕与刺眼锐化痕迹，观感生硬粗糙。",
            "P0（严重过锐）：全图真实边缘严重双轮廓黑白光晕、噪点被剧烈锐化成麻点砂砾状，画面极度刺眼失真。",
        ],
    },
    "花屏": {
        "type": "noul",
        "instructions": "图像是否出现视频解码损坏、绿屏/紫块色块异常、局部彩色条纹撕裂或宏块错位（花屏 P0）？",
        "criteria": {
            "true": "存在解码宏块错位、局部彩色/绿色/紫色异常条纹色块撕裂或像素阵列损坏（P0 花屏）。",
            "false": "画面完整正常，无任何解码损坏、异常色块撕裂或条纹花屏。",
        },
    },
    "过度生成": {
        "type": "score",
        "instructions": "评估图像是否出现 AI 生成或超分增强导致的伪造幻觉伪影（如字幕/背景文字笔画粘连乱码、多指/断肢/畸形手部、扭曲假五官、发丝/纹理违背物理规律的塑料漩涡）。",
        "criteria": [
            "无问题：人物解剖结构（手指、五官、肢体）与画面文字笔画均真实自然，无 AI 臆造乱码或伪纹理。",
            "P2（轻微过度生成）：仅在小面积局部（如边缘小字笔画轻微粘连变形、远景小手/配饰细节微瑕或局部发丝假纹理）出现轻度生成伪影。",
            "P1（中度过度生成）：主体或醒目位置出现易察觉的字幕乱码伪字、手指粘连多指畸变、五官局部错乱或明显不自然的 AI 假纹理漩涡。",
            "P0（严重过度生成）：核心主体解剖结构严重崩坏（怪物脸/严重断肢多手）、大面积恐怖谷假五官或满屏荒诞乱码幻觉。",
        ],
    },
    "虚化突兀": {
        "type": "noul",
        "instructions": "图像是否存在算法人像抠图虚化导致的发丝、肩颈、手指边缘断崖式生硬阶跃伪影或前景局部误虚化（虚化突兀 P0，注意区分镜头自然连续渐变的光学浅景深）？",
        "criteria": {
            "true": "存在算法抠图假虚化造成的发丝/边缘断崖式模糊阶跃、光斑切割生硬或主体边缘被误糊掉（P0 虚化突兀）。",
            "false": "无算法假虚化抠图伪影（全清晰画面，或符合物理景深连续渐变的自然镜头光学虚化均属无问题）。",
        },
    },
}

DIMS_ORDER = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")


def run_shard(args):
    import torch
    from joint_schema_model import collate_records, encode_record, load_release_model
    from qingyu.config_and_paths import LosslessNativeImageLoader
    from qingyu.dual_engine_preprocess import DualEnginePhysicsExtractor, PrecomputedDualEngineStore

    import common55 as c

    names, gold, fold, _ = c.load_split()
    shard_idx = list(range(args.rank, len(names), args.world_size))
    shard_names = [names[i] for i in shard_idx]

    device = torch.device("cuda:0")
    model, processor = load_release_model(MODEL_DIR, device=device, dtype=torch.bfloat16)
    loader = LosslessNativeImageLoader(REPO)
    ext = DualEnginePhysicsExtractor(device="cpu", allow_depth_download=False)
    store = PrecomputedDualEngineStore("results/qingyu/dual_engine_meta.npz", lazy_extractor=ext)

    probs = np.zeros((len(shard_names), 6, 4), dtype=np.float32)
    t0 = time.time()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    for si, nm in enumerate(shard_names):
        pil_im = loader.load_pil(nm).convert("RGB")
        meta = store.get(nm, pil_im=pil_im)
        imgs, _ = ext.prepare_tier3_multi_images(pil_im, precomputed=meta, global_max_pixels=768 * 768)
        rec = {
            "id": nm,
            "state": {
                "views": [
                    "图1：保持原始宽高比的全图视图",
                    "图2：前景人物解剖结构（五官/手部/皮肤）1:1 原生局部放大图",
                    "图3：文字字幕/高频边缘纹理 1:1 原生局部放大图",
                ]
            },
            "images": imgs,
            "questions": QUESTIONS_SCHEMA,
        }
        enc = encode_record(processor.tokenizer, rec, max_length=16384, processor=processor)
        batch = collate_records([enc], pad_id, device)
        with torch.inference_mode():
            logits_list = model(batch)[0]
        q_map = {q.question_id: (q, lg.float().softmax(-1).cpu().numpy()) for q, lg in zip(enc.questions, logits_list)}
        for di, dname in enumerate(DIMS_ORDER):
            q_obj, sm = q_map[dname]
            opt_map = dict(zip(q_obj.option_ids, sm.tolist()))
            if dname in ("花屏", "虚化突兀"):
                p_true = float(opt_map["true"])
                probs[si, di, 0] = 1.0 - p_true
                probs[si, di, 3] = p_true
            else:
                for g in range(4):
                    probs[si, di, g] = float(opt_map[str(g)])
        if (si + 1) % 50 == 0 or (si + 1) == len(shard_names):
            el = time.time() - t0
            print(f"[Clef-JointHead shard {args.rank}/{args.world_size}] {si + 1}/{len(shard_names)} ({(si + 1) / max(1e-3, el):.2f} img/s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    part = args.out.replace(".npz", f".part{args.rank}of{args.world_size}.npz")
    np.savez_compressed(part, names=np.array(shard_names), probs_6x4=probs)
    print(f"[Clef-JointHead] Saved {part}", flush=True)


def merge_and_eval(args):
    from sklearn.metrics import average_precision_score
    import common55 as c
    import final_engine as fe
    import decoder_eu as E

    names, gold, fold, _ = c.load_split()
    idx = {n: i for i, n in enumerate(names)}
    probs = np.zeros((len(names), 6, 4), dtype=np.float32)
    seen = 0
    for r in range(args.merge):
        part = args.out.replace(".npz", f".part{r}of{args.merge}.npz")
        z = np.load(part, allow_pickle=True)
        for nm, p in zip(z["names"], z["probs_6x4"]):
            probs[idx[str(nm)]] = p
            seen += 1
    assert seen == len(names), f"Expected {len(names)}, got {seen}"
    np.savez_compressed(args.out, names=np.array(names), probs_6x4=probs)

    split_map = {row["name"]: row["split"] for row in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    orig_mask = np.array([split_map[n] == "orig" for n in names])

    # Compute mAP14 on orig_1902 (directly comparable to decision_model_screen.md!)
    aps = []
    dim_aps = {}
    for d, dname in enumerate(DIMS_ORDER):
        gs = (1, 2, 3) if d in (0, 1, 2, 4) else (3,)
        sub = []
        for g in gs:
            yt = (gold[orig_mask, d] == g).astype(np.int32)
            ap = float(average_precision_score(yt, probs[orig_mask, d, g]))
            aps.append(ap)
            sub.append(ap)
        dim_aps[dname] = float(np.mean(sub))

    oh = np.eye(4)[gold]
    rps_all = float(np.mean((np.cumsum(probs, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2))
    rps_orig = float(np.mean((np.cumsum(probs[orig_mask], -1)[..., :3] - np.cumsum(oh[orig_mask], -1)[..., :3]) ** 2))
    print(f"=== Clef-Flash + 233MB Pretrained JointHead Zero-Shot Results ===")
    print(f"  ORIG_1902 mAP14 = {np.mean(aps):.4f} (vs old lm_head MCQ 0.234!) | per-dim AP: " +
          " ".join(f"{k}={v:.3f}" for k, v in dim_aps.items()))
    print(f"  RPS_all = {rps_all:.5f} | RPS_orig = {rps_orig:.5f}")

    # Check ensemble blend value on top of honest fallback (no VLM)
    _, _, _, _, P_base = fe.oof_pool(["cradio", "pe", "sig", "v16", "p3"])
    import glob
    P_v16 = np.zeros((len(names), 6, 4))
    for f in sorted(glob.glob(os.path.join(REPO, "results/r55/v16_qy/V16_Full_F*_probs.npz"))):
        z = np.load(f, allow_pickle=True)
        for n, v in zip(z["va_names"], c.p012_to_6x4_simplex(z["va_p0"], z["va_p1"], z["va_p2"])):
            P_v16[idx[str(n)]] = v
    P_base["v16"] = P_v16
    Q_fb = fe.pool(P_base, ["cradio", "pe", "sig", "v16", "p3"])
    for w in (0.0, 0.05, 0.10, 0.15):
        Qb = c.renorm((1.0 - w) * Q_fb + w * probs)
        rps_b = float(np.mean((np.cumsum(Qb, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2))
        pred = np.zeros_like(gold)
        for k in range(8):
            tr, va = fold != k, fold == k
            pred[va] = E.DecoderEU().fit(Qb[tr], gold[tr]).predict(Qb[va])
        sa = c.score(pred, gold)["doc2_strict_total"]
        so = c.score(pred[orig_mask], gold[orig_mask])["doc2_strict_total"]
        print(f"  Blend w={w:.2f} into fallback | RPS_all={rps_b:.5f} | EU all={sa:.3f} orig={so:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--merge", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "results/r55/clef_joint_head_zeroshot.npz"))
    args = ap.parse_args()
    if args.merge > 0:
        merge_and_eval(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
