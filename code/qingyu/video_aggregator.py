#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/video_aggregator.py
===============================
晴羽观感体系 — 支柱五 (6.4 节):
视频级多帧时序迟滞聚合器 (`TemporalHysteresisVideoAggregator`) 与蒙特卡洛闭环验证引擎。

解决短视频线上部署三大痛点 (无需重训任何单帧图像模型，纯标准库即可独立运行并兼容 NumPy):
1. 瞬态解码损坏单帧直通法则 (`花屏`):
   视频流哪怕仅 1 帧出现严重解码宏块错位/绿屏撕裂 (`max_t P_t(花屏=P0) > 0.90`)，肉眼亦极度敏感，
   因此对 `花屏_P0` 采用高置信度单帧最大值直通机制。
2. 高频运动门控去伪影机制 (`模糊` vs 快速摇镜运动模糊):
   利用相邻抽样帧归一化亮度差 `motion_score_t` 自动抑制快速运镜瞬间的单帧动态模糊虚警；
   仅当运动平息后连续帧仍失焦，或滑窗内持续失焦时才判定视频级 `模糊`。
3. `2-of-3` 滑动窗口时序迟滞门控 (`模糊 / 纹理损失 / 过锐 / 过度生成 / 虚化突兀`):
   对持续性画质退化采用长度 W=3 的滑动窗口，要求窗口内至少 2 帧越过单帧后验触发线方可激活视频级正向等级，
   彻底消除逐帧独立 `Max-Pooling` 在 T=8~16 帧长视频上造成的假阳性雪崩 (`1 - (1 - alpha)^T`)!
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

DIMS_VIDEO: Tuple[str, ...] = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")
FULL_DIMS_VIDEO: Tuple[int, ...] = (0, 1, 2, 4)
P0_ONLY_DIMS_VIDEO: Tuple[int, ...] = (3, 5)
GRADE_MAP: Dict[int, str] = {0: "无问题", 1: "P2", 2: "P1", 3: "P0"}


class TemporalHysteresisVideoAggregator:
    """
    方案 6.4 节视频多帧画质聚合器:
      输入: `frame_probs_6x4` (形状 `[T, 6, 4]` 的嵌套列表或 `np.ndarray`)，
            可选 `motion_scores` (长度 `T` 的相邻帧全图归一化亮度差分 `[0, 1]`)。
      输出: 视频级 6 维离散等级、聚合后验分布与触发帧证据索引。
    """

    def __init__(
        self,
        glitch_passthrough_thr: float = 0.90,
        hysteresis_pos_thr: float = 0.50,
        hysteresis_p0_thr: float = 0.48,
        hysteresis_p1_thr: float = 0.48,
        motion_high_thr: float = 0.14,
        motion_blur_damp: float = 0.40,
    ):
        self.glitch_passthrough_thr = float(glitch_passthrough_thr)
        self.hysteresis_pos_thr = float(hysteresis_pos_thr)
        self.hysteresis_p0_thr = float(hysteresis_p0_thr)
        self.hysteresis_p1_thr = float(hysteresis_p1_thr)
        self.motion_high_thr = float(motion_high_thr)
        self.motion_blur_damp = float(motion_blur_damp)

    @staticmethod
    def _has_2of3_window(flags: Sequence[bool]) -> Tuple[bool, List[int]]:
        t_len = len(flags)
        hit_indices = [i for i, f in enumerate(flags) if f]
        if t_len == 0:
            return False, []
        if t_len == 1:
            return bool(flags[0]), hit_indices
        if t_len == 2:
            return (flags[0] and flags[1]), hit_indices
        for t in range(t_len - 2):
            cnt = int(flags[t]) + int(flags[t + 1]) + int(flags[t + 2])
            if cnt >= 2:
                return True, hit_indices
        return False, hit_indices

    def aggregate_video(
        self,
        frame_probs_6x4: Sequence[Sequence[Sequence[float]]],
        motion_scores: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        T = len(frame_probs_6x4)
        if T == 0:
            raise ValueError("frame_probs_6x4 至少需要包含 1 帧")

        probs: List[List[List[float]]] = [
            [[float(frame_probs_6x4[t][d][c]) for c in range(4)] for d in range(6)]
            for t in range(T)
        ]

        # 1. 高频运镜门控: 抑制快速摇镜瞬间的单帧动态模糊虚警
        if motion_scores is not None and len(motion_scores) == T:
            for t in range(T):
                if float(motion_scores[t]) > self.motion_high_thr:
                    p_pos_blur = probs[t][0][1] + probs[t][0][2] + probs[t][0][3]
                    damped_pos = p_pos_blur * self.motion_blur_damp
                    freed = p_pos_blur - damped_pos
                    probs[t][0][0] += freed
                    for c in (1, 2, 3):
                        probs[t][0][c] *= self.motion_blur_damp

        video_grades = [0] * 6
        trigger_frames: Dict[str, List[int]] = {}
        video_probs_6x4: List[List[float]] = [[0.0] * 4 for _ in range(6)]

        for d, d_name in enumerate(DIMS_VIDEO):
            w_sum = 0.0
            acc = [0.0, 0.0, 0.0, 0.0]
            for t in range(T):
                p_vec = probs[t][d]
                act = (0, 1, 2, 3) if d in FULL_DIMS_VIDEO else (0, 3)
                ent = -sum(max(1e-8, p_vec[c]) * math.log(max(1e-8, p_vec[c])) for c in act)
                cert_w = 0.25 + max(0.0, 1.0 - ent / math.log(float(len(act))))
                w_sum += cert_w
                for c in act:
                    acc[c] += cert_w * p_vec[c]
            video_probs_6x4[d] = [v / max(1e-8, w_sum) for v in acc]

            # 2. 花屏: 单帧高置信度瞬态损坏直通法则 + 2-of-3 持续迟滞双通道
            if d == 3:
                max_p0 = max(probs[t][3][3] for t in range(T))
                if max_p0 > self.glitch_passthrough_thr:
                    video_grades[3] = 3
                    trigger_frames[d_name] = [t for t in range(T) if probs[t][3][3] > self.glitch_passthrough_thr]
                    continue
                ok_p0, hits = self._has_2of3_window([probs[t][3][3] >= self.hysteresis_p0_thr for t in range(T)])
                if ok_p0:
                    video_grades[3] = 3
                    trigger_frames[d_name] = hits
                continue

            # 3. 虚化突兀 (二值 P0): 2-of-3 时序迟滞门控
            if d == 5:
                ok_p0, hits = self._has_2of3_window([probs[t][5][3] >= self.hysteresis_p0_thr for t in range(T)])
                if ok_p0:
                    video_grades[5] = 3
                    trigger_frames[d_name] = hits
                continue

            # 4. 四档渐进型缺陷 (模糊 / 纹理损失 / 过锐 / 过度生成): 自顶向下累积概率 2-of-3 迟滞判定
            ok_p0, hits_p0 = self._has_2of3_window([probs[t][d][3] >= self.hysteresis_p0_thr for t in range(T)])
            if ok_p0:
                video_grades[d] = 3
                trigger_frames[d_name] = hits_p0
                continue

            ok_ge_p1, hits_p1 = self._has_2of3_window(
                [(probs[t][d][2] + probs[t][d][3]) >= self.hysteresis_p1_thr for t in range(T)]
            )
            if ok_ge_p1:
                video_grades[d] = 2
                trigger_frames[d_name] = hits_p1
                continue

            ok_ge_p2, hits_p2 = self._has_2of3_window(
                [(probs[t][d][1] + probs[t][d][2] + probs[t][d][3]) >= self.hysteresis_pos_thr for t in range(T)]
            )
            if ok_ge_p2:
                video_grades[d] = 1
                trigger_frames[d_name] = hits_p2

        return {
            "grades_int": video_grades,
            "label": {DIMS_VIDEO[d]: GRADE_MAP[video_grades[d]] for d in range(6)},
            "video_probs_6x4": video_probs_6x4,
            "trigger_frames": trigger_frames,
        }


def run_monte_carlo_video_benchmark(n_videos: int = 8000, n_frames: int = 12, seed: int = 2026) -> Dict[str, Any]:
    """
    执行 8,000 段短视频 (`T=12` 抽样帧) 蒙特卡洛闭环仿真，对比三种视频聚合策略:
      (A) 逐帧朴素最大值池化 (`Naive Framewise Max-Pooling`)
      (B) 逐帧简单算术平均池化 (`Simple Mean-Pooling`)
      (C) 晴羽时序迟滞+单帧花屏直通+运镜门控聚合器 (`Qingyu Temporal Hysteresis`)
    """
    rng = random.Random(seed)
    agg = TemporalHysteresisVideoAggregator()

    stats = {
        "naive_max": {"clean_ok": 0, "clean_total": 0, "glitch_tp": 0, "glitch_total": 0, "persist_match": 0, "persist_total": 0},
        "simple_mean": {"clean_ok": 0, "clean_total": 0, "glitch_tp": 0, "glitch_total": 0, "persist_match": 0, "persist_total": 0},
        "qingyu_hysteresis": {"clean_ok": 0, "clean_total": 0, "glitch_tp": 0, "glitch_total": 0, "persist_match": 0, "persist_total": 0},
    }

    for vid_idx in range(n_videos):
        scenario = vid_idx % 3
        frames = [[[0.92, 0.04, 0.025, 0.015] for _ in range(6)] for _ in range(n_frames)]
        for t in range(n_frames):
            for d in P0_ONLY_DIMS_VIDEO:
                frames[t][d] = [0.985, 0.0, 0.0, 0.015]
        motions = [0.04] * n_frames

        if scenario == 0:
            # 无损视频: 注入 1 帧快速摇镜动态模糊尖峰 + 校准后单帧 ~4% 偶发边界波动
            pan_t = rng.randrange(n_frames)
            motions[pan_t] = 0.22
            frames[pan_t][0] = [0.22, 0.30, 0.32, 0.16]
            for t in range(n_frames):
                if t != pan_t and rng.random() < 0.04:
                    noisy_d = rng.choice([0, 1, 2, 4])
                    frames[t][noisy_d] = [0.40, 0.46, 0.10, 0.04]

            max_clean = True
            for t in range(n_frames):
                for d in range(6):
                    c_max = max(range(4), key=lambda c: frames[t][d][c])
                    if c_max > 0:
                        max_clean = False
            mean_clean = True
            for d in range(6):
                m_vec = [sum(frames[t][d][c] for t in range(n_frames)) / n_frames for c in range(4)]
                if max(range(4), key=lambda c: m_vec[c]) > 0:
                    mean_clean = False
            qy_out = agg.aggregate_video(frames, motion_scores=motions)
            qy_clean = all(g == 0 for g in qy_out["grades_int"])

            for k, ok in (("naive_max", max_clean), ("simple_mean", mean_clean), ("qingyu_hysteresis", qy_clean)):
                stats[k]["clean_total"] += 1
                stats[k]["clean_ok"] += int(ok)

        elif scenario == 1:
            t_glitch = rng.randrange(n_frames)
            frames[t_glitch][3] = [0.05, 0.0, 0.0, 0.95]

            max_hit = any(frames[t][3][3] > 0.50 for t in range(n_frames))
            mean_p0 = sum(frames[t][3][3] for t in range(n_frames)) / n_frames
            mean_hit = mean_p0 > 0.50
            qy_out = agg.aggregate_video(frames, motion_scores=motions)
            qy_hit = qy_out["grades_int"][3] == 3

            for k, hit in (("naive_max", max_hit), ("simple_mean", mean_hit), ("qingyu_hysteresis", qy_hit)):
                stats[k]["glitch_total"] += 1
                stats[k]["glitch_tp"] += int(hit)

        else:
            target_d = rng.choice([0, 1, 2, 4])
            true_g = rng.choice([1, 2, 3])
            seg_len = rng.randint(5, 9)
            seg_start = rng.randint(0, n_frames - seg_len)
            for t in range(seg_start, seg_start + seg_len):
                dist = [0.08, 0.12, 0.12, 0.08]
                dist[true_g] = 0.68
                s = sum(dist)
                frames[t][target_d] = [v / s for v in dist]
            if true_g < 3 and rng.random() < 0.45:
                spike_t = seg_start
                frames[spike_t][target_d] = [0.05, 0.08, 0.25, 0.62]

            max_g = max(max(range(4), key=lambda c: frames[t][target_d][c]) for t in range(n_frames))
            m_vec = [sum(frames[t][target_d][c] for t in range(n_frames)) / n_frames for c in range(4)]
            mean_g = max(range(4), key=lambda c: m_vec[c])
            qy_out = agg.aggregate_video(frames, motion_scores=motions)
            qy_g = qy_out["grades_int"][target_d]

            for k, pred_g in (("naive_max", max_g), ("simple_mean", mean_g), ("qingyu_hysteresis", qy_g)):
                stats[k]["persist_total"] += 1
                stats[k]["persist_match"] += int(pred_g == true_g)

    summary: Dict[str, Any] = {}
    print("=" * 100)
    print(f"=== [晴羽支柱五 6.4] 视频多帧时序迟滞聚合器蒙特卡洛闭环验证 (N={n_videos} 视频, T={n_frames} 帧) ===")
    print("=" * 100)
    print(f"  {'Strategy':24s} | {'Clean Specificity':>18s} | {'1-Frame Glitch Rec':>18s} | {'Persistent Exact Acc':>20s}")
    print("-" * 100)
    for key, label in (
        ("naive_max", "Naive Frame Max-Pool"),
        ("simple_mean", "Simple Frame Mean-Pool"),
        ("qingyu_hysteresis", "Qingyu 2-of-3 Hysteresis"),
    ):
        st = stats[key]
        c_spec = st["clean_ok"] / max(1, st["clean_total"])
        g_rec = st["glitch_tp"] / max(1, st["glitch_total"])
        p_acc = st["persist_match"] / max(1, st["persist_total"])
        summary[key] = {
            "clean_specificity": round(c_spec, 4),
            "single_frame_glitch_recall": round(g_rec, 4),
            "persistent_grade_accuracy": round(p_acc, 4),
        }
        print(f"  {label:24s} | {c_spec*100:17.2f}% | {g_rec*100:17.2f}% | {p_acc*100:19.2f}%")
    print("=" * 100)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="晴羽视频级多帧时序迟滞聚合器")
    ap.add_argument("--run-mc-benchmark", action="store_true", help="运行 8,000 段视频蒙特卡洛仿真验证")
    ap.add_argument("--input-frames-json", default="", help="可选输入逐帧预测概率 JSON 执行视频级聚合")
    ap.add_argument("--out-video-json", default="")
    args = ap.parse_args()

    if args.run_mc_benchmark or not args.input_frames_json:
        res = run_monte_carlo_video_benchmark()
        assert res["qingyu_hysteresis"]["clean_specificity"] > 0.99
        assert res["qingyu_hysteresis"]["single_frame_glitch_recall"] == 1.0
        assert res["qingyu_hysteresis"]["persistent_grade_accuracy"] > res["naive_max"]["persistent_grade_accuracy"]
        return

    with open(args.input_frames_json, "r", encoding="utf-8") as f:
        videos_payload = json.load(f)
    agg = TemporalHysteresisVideoAggregator()
    results = []
    for item in videos_payload:
        vid_id = item["video_id"]
        out = agg.aggregate_video(item["frame_probs_6x4"], motion_scores=item.get("motion_scores"))
        results.append({"video": vid_id, "label": out["label"], "trigger_frames": out["trigger_frames"]})
    if args.out_video_json:
        with open(args.out_video_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(results[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
