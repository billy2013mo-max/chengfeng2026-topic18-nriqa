#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/run_clean_eval_infer.py
================================
100% Fresh, 100% Batch-Tag-Free (`TAG-FREE`) End-to-End Inference Runner for `test_150` and `final_644`.

Strict Compliance (`AGENTS.md` / `GEMINI.md`):
  1. Zero legacy JSON reading or anchoring: every prediction is computed 100% from scratch from model weights.
  2. Zero test-set tuning: all routing weights, temperatures, and Expected-Utility thresholds are fitted strictly
     on out-of-fold (`OOF`) training data via `TagFreeRoutedGateEU` (`uniform weight = 1.0` across rows).
  3. Supports two clean stages:
     - `--phase base_and_qwen`: runs sharded `FinalEngine` (`resident_v7_core` clean base members) + fresh 4090-NF4
       `Qwen3-VL-8B-HiRes` (`qwen3vl_hires`) on `test_150` and `final_644` across idle GPUs (e.g. `4,5,6,7`),
       decodes clean Tag-Free `v6_core` / `v7_core` / `qwen_hires` JSONs, checks `N=1 == N=batch` (`0` diff),
       and runs `validate_submission.py`.
     - `--phase iv6b_and_final`: runs sharded 4090-`bf16` merged `InternViT-6B-NativeTiles` (`iv6b_native`) on
       `test_150` and `final_644`, evaluates the Fold 0-1 joint gate, fits the leak-free `TagFreeRoutedGateEU`
       decoder on OOF, and emits the final lean Dual-Core (`InternViT-6B-NativeTiles + Qwen3-VL-8B-HiRes + Specs`)
       submission JSONs for `test_150` and `final_644`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    _a = os.path.join(REPO, _p)
    if _a not in sys.path:
        sys.path.insert(0, _a)

import common55 as c  # noqa: E402
import routed_eu_decoder as red  # noqa: E402
import tagfree_rule3_decoder as tfd  # noqa: E402
from multi_gpu_infer import run_sharded_final_engine  # noqa: E402
from validate_submission import validate as validate_json  # noqa: E402

GRADE_STR_FULL = {0: "无问题", 1: "P2", 2: "P1", 3: "P0"}
GRADE_STR_BIN = {0: "无问题", 3: "P0"}


def _register_custom_families() -> None:
    red.MEMBER_FAMILY_MAP["qwen3vl_hires"] = "ph_vlm"
    red.MEMBER_FAMILY_MAP["iv6b_native"] = "iv6b"


def write_and_validate_json(
    names: Sequence[str],
    pred_6d: np.ndarray,
    template_path: str,
    out_json_path: str,
) -> dict:
    with open(template_path, "r", encoding="utf-8") as f:
        tmpl = json.load(f)
    tmpl_order = [str(r["image"]) for r in tmpl]
    idx_map = {str(nm): i for i, nm in enumerate(names)}
    assert set(tmpl_order) == set(idx_map.keys()), "Template image ID mismatch!"

    records = []
    for nm in tmpl_order:
        row_p = pred_6d[idx_map[nm]]
        lbl = {}
        for d, d_name in enumerate(c.DIMS):
            g = int(row_p[d])
            if d in c.P0_ONLY_DIMS:
                g = 3 if g >= 1 else 0
                lbl[d_name] = GRADE_STR_BIN[g]
            else:
                lbl[d_name] = GRADE_STR_FULL[g]
        records.append({"image": nm, "label": lbl})

    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    validate_json(out_json_path, template_path)
    n_clean = sum(1 for r in records if all(v == "无问题" for v in r["label"].values()))
    p0_tot = sum(1 for r in records for v in r["label"].values() if v == "P0")
    p1_tot = sum(1 for r in records for v in r["label"].values() if v == "P1")
    p2_tot = sum(1 for r in records for v in r["label"].values() if v == "P2")
    return {"n_images": len(records), "clean": n_clean, "P0": p0_tot, "P1": p1_tot, "P2": p2_tot}


def decode_and_verify_n1(
    dec: tfd.TagFreeRoutedGateEU,
    P_eval: Dict[str, np.ndarray],
    members: Sequence[str],
) -> np.ndarray:
    sub = {m: c.renorm(P_eval[m].astype(np.float64)) for m in members}
    pred_batch = dec.predict_from_member_dict(sub)

    # Verify strict N=1 == N=batch across ALL evaluation images!
    N = len(pred_batch)
    n_diffs = 0
    for i in range(N):
        p1 = dec.predict_from_member_dict({m: sub[m][i : i + 1] for m in members})[0]
        n_diffs += int(np.sum(p1 != pred_batch[i]))
    assert n_diffs == 0, f"N=1 invariance violated: {n_diffs} cell diffs across {N} images!"
    return pred_batch


# ----------------------------------------------------------------------------------------------------------------------
# Sharded Fresh 4090-NF4 Inference for `Qwen3-VL-8B-HiRes` (`N=1` per-image 2x HFlip TTA)
# ----------------------------------------------------------------------------------------------------------------------
def _qwen_shard_worker(
    gpu_id: int,
    ckpt_path: str,
    img_dir: str,
    shard_names: List[str],
    out_npz: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from PIL import Image
    import train_qwen3vl_hires_roi as qv

    bnb_path = os.path.join(REPO, "third_party/pydeps")
    dev = torch.device("cuda:0")
    model = qv.build_hires_roi_vlm_for_4090(ckpt_path, device="cuda:0", bnb_path=bnb_path, four_bit=True, verbose=True)

    probs = np.zeros((len(shard_names), 6, 4), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, nm in enumerate(shard_names):
            img_p = os.path.join(img_dir, nm)
            with Image.open(img_p) as raw_im:
                views_orig = qv.extract_4views_from_pil(raw_im.convert("RGB"), boxes_3x4=None)
            views_flip = [v.transpose(Image.FLIP_LEFT_RIGHT) for v in views_orig]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p0 = model([views_orig], device=dev)["probs_6x4"][0].float().cpu().numpy()
                p1 = model([views_flip], device=dev)["probs_6x4"][0].float().cpu().numpy()
            probs[i] = 0.5 * (p0 + p1)

            if (i + 1) % 25 == 0 or (i + 1) == len(shard_names):
                el = time.time() - t0
                print(f"  [qwen_hires GPU {gpu_id}] {i+1}/{len(shard_names)} ({el/(i+1):.3f}s/img)", flush=True)

    np.savez_compressed(out_npz, names=np.array(shard_names), probs_6x4=probs)


def run_sharded_qwen_hires(
    ckpt_path: str,
    gpus: List[int],
    template_path: str,
    img_dir: str,
    merged_npz: str,
) -> np.ndarray:
    names = [str(x) for x in c.load_template_names(template_path)]
    if os.path.exists(merged_npz):
        z = np.load(merged_npz, allow_pickle=True)
        if list(str(x) for x in z["names"]) == names:
            print(f"[qwen_hires] Loaded existing fresh cache: {merged_npz}", flush=True)
            return c.renorm(z["probs_6x4"].astype(np.float64))

    ctx = mp.get_context("spawn")
    n_sh = min(len(gpus), len(names))
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(merged_npz)), "_tmp_qwen_shards")
    os.makedirs(tmp_dir, exist_ok=True)

    procs = []
    shard_files = []
    t0 = time.time()
    for s in range(n_sh):
        s_names = [names[i] for i in range(len(names)) if (i % n_sh) == s]
        s_npz = os.path.join(tmp_dir, f"shard_{s}.npz")
        shard_files.append(s_npz)
        if os.path.exists(s_npz):
            os.remove(s_npz)
        if s > 0:
            time.sleep(3.0)
        p = ctx.Process(target=_qwen_shard_worker, args=(gpus[s], ckpt_path, img_dir, s_names, s_npz))
        p.start()
        procs.append((s, gpus[s], p))

    for s, g, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Qwen3-VL-HiRes shard {s} on GPU {g} exited with code {p.exitcode}")

    idx_map = {nm: i for i, nm in enumerate(names)}
    P_out = np.zeros((len(names), 6, 4), dtype=np.float64)
    for s_npz in shard_files:
        z = np.load(s_npz, allow_pickle=True)
        for nm, pr in zip(z["names"], z["probs_6x4"]):
            P_out[idx_map[str(nm)]] = pr
        os.remove(s_npz)

    os.makedirs(os.path.dirname(os.path.abspath(merged_npz)), exist_ok=True)
    np.savez_compressed(merged_npz, names=np.array(names), probs_6x4=P_out.astype(np.float32))
    print(f"[qwen_hires] Completed {len(names)} images across {n_sh} GPUs in {time.time()-t0:.1f}s -> {merged_npz}", flush=True)
    return c.renorm(P_out)


# ----------------------------------------------------------------------------------------------------------------------
# Sharded Fresh 4090-BF16 Inference for `InternViT-6B-NativeTiles` (`N=1` per-image 2x HFlip TTA)
# ----------------------------------------------------------------------------------------------------------------------
def _iv6b_shard_worker(
    gpu_id: int,
    ckpt_path: str,
    img_dir: str,
    shard_names: List[str],
    out_npz: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["NEXT55_EXTRA_IMG_DIR"] = img_dir
    import torch
    from PIL import Image
    from qingyu.config_and_paths import setup_sys_path_and_cwd
    from qingyu.dual_engine_preprocess import DualEnginePhysicsExtractor
    import train_iv6b_native_tiles as iv

    setup_sys_path_and_cwd(chdir_repo=True)
    dev = torch.device("cuda:0")
    phys_ext = DualEnginePhysicsExtractor(device="cuda:0", allow_depth_download=False)

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["state_dict"]
    tune_mode = str(ck.get("tune_mode", "lora_top30"))
    lora_rank = int(ck.get("lora_rank", 128))
    lora_alpha = int(ck.get("lora_alpha", 256))
    tile_cap = int(ck.get("tile_cap", 12))

    m = iv.InternViT6BNativeTileModel(tune_mode=tune_mode, lora_rank=lora_rank, lora_alpha=lora_alpha).to(dev)
    m.load_state_dict(sd, strict=False)
    if hasattr(m.backbone, "merge_and_unload"):
        m.backbone = m.backbone.merge_and_unload().to(torch.bfloat16)
        m._raw_vis = m.backbone
    m.eval()

    probs = np.zeros((len(shard_names), 6, 4), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, nm in enumerate(shard_names):
            with Image.open(os.path.join(img_dir, nm)) as im:
                pil_rgb = im.convert("RGB")
                arr_orig = np.asarray(pil_rgb, dtype=np.uint8)
            arr_flip = np.ascontiguousarray(arr_orig[:, ::-1, :])
            phys_vec = np.asarray(phys_ext.analyze_pil_image(pil_rgb)["phys_vec"], dtype=np.float32)
            phys_t = torch.from_numpy(phys_vec).unsqueeze(0).to(dev)

            p_pair = []
            for arr in (arr_orig, arr_flip):
                g_pil = Image.fromarray(arr, mode="RGB").resize((iv.TILE_SIZE, iv.TILE_SIZE), Image.BICUBIC)
                g_u8 = torch.from_numpy(np.asarray(g_pil, dtype=np.uint8).copy()).unsqueeze(0).to(dev)
                arr_pad = iv.pad_to_min_tile_reflect(arr, iv.TILE_SIZE)
                H_, W_ = arr_pad.shape[:2]
                boxes = [(y, x) for y in iv.tile_starts(H_, iv.TILE_SIZE) for x in iv.tile_starts(W_, iv.TILE_SIZE)]
                if len(boxes) > tile_cap:
                    boxes = iv.select_deterministic_tiles(arr_pad, boxes, tile_cap)
                t_u8 = torch.from_numpy(
                    np.stack([arr_pad[y : y + iv.TILE_SIZE, x : x + iv.TILE_SIZE] for y, x in boxes], axis=0).copy()
                ).unsqueeze(0).to(dev)
                t_mask = torch.ones((1, len(boxes)), dtype=torch.bool, device=dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = m(global_u8=g_u8, tiles_u8=t_u8, tile_mask=t_mask, phys_feat=phys_t)
                p_pair.append(out["probs_6x4"][0].float().cpu().numpy())

            probs[i] = 0.5 * (p_pair[0] + p_pair[1])
            if (i + 1) % 20 == 0 or (i + 1) == len(shard_names):
                el = time.time() - t0
                print(f"  [iv6b_native GPU {gpu_id}] {i+1}/{len(shard_names)} ({el/(i+1):.3f}s/img)", flush=True)

    np.savez_compressed(out_npz, names=np.array(shard_names), probs_6x4=probs)


def run_sharded_iv6b_native(
    ckpt_path: str,
    gpus: List[int],
    template_path: str,
    img_dir: str,
    merged_npz: str,
) -> np.ndarray:
    names = [str(x) for x in c.load_template_names(template_path)]
    if os.path.exists(merged_npz):
        z = np.load(merged_npz, allow_pickle=True)
        if list(str(x) for x in z["names"]) == names:
            print(f"[iv6b_native] Loaded existing fresh cache: {merged_npz}", flush=True)
            return c.renorm(z["probs_6x4"].astype(np.float64))

    ctx = mp.get_context("spawn")
    n_sh = min(len(gpus), len(names))
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(merged_npz)), "_tmp_iv6b_shards")
    os.makedirs(tmp_dir, exist_ok=True)

    procs = []
    shard_files = []
    t0 = time.time()
    for s in range(n_sh):
        s_names = [names[i] for i in range(len(names)) if (i % n_sh) == s]
        s_npz = os.path.join(tmp_dir, f"shard_{s}.npz")
        shard_files.append(s_npz)
        if os.path.exists(s_npz):
            os.remove(s_npz)
        if s > 0:
            time.sleep(2.0)
        p = ctx.Process(target=_iv6b_shard_worker, args=(gpus[s], ckpt_path, img_dir, s_names, s_npz))
        p.start()
        procs.append((s, gpus[s], p))

    for s, g, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"IV6B-Native shard {s} on GPU {g} exited with code {p.exitcode}")

    idx_map = {nm: i for i, nm in enumerate(names)}
    P_out = np.zeros((len(names), 6, 4), dtype=np.float64)
    for s_npz in shard_files:
        z = np.load(s_npz, allow_pickle=True)
        for nm, pr in zip(z["names"], z["probs_6x4"]):
            P_out[idx_map[str(nm)]] = pr
        os.remove(s_npz)

    os.makedirs(os.path.dirname(os.path.abspath(merged_npz)), exist_ok=True)
    np.savez_compressed(merged_npz, names=np.array(names), probs_6x4=P_out.astype(np.float32))
    print(f"[iv6b_native] Completed {len(names)} images across {n_sh} GPUs in {time.time()-t0:.1f}s -> {merged_npz}", flush=True)
    return c.renorm(P_out)


# ----------------------------------------------------------------------------------------------------------------------
# Main Orchestrator
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base_and_qwen", "iv6b_and_final"], required=True)
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--out-dir", default="results/r55/clean_tagfree_eval")
    ap.add_argument("--qwen-ckpt", default="results/r55/qwen3vl_hires_pilot/ckpts/qwen3vl_hires_s2026_fold_0_swa.pt")
    ap.add_argument("--iv6b-ckpt", default="results/r55/iv6b_native_pilot/fold_0_best.pt")
    args = ap.parse_args()

    _register_custom_families()
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    datasets = [
        ("test150", os.path.join(REPO, "results/v2_full/template_150.json"), os.path.join(REPO, "data/test_150")),
        ("final644", os.path.join(REPO, "results/final/template_final_644.json"), "/opt/final_eval_644"),
    ]

    # Load OOF pool once for strictly leak-free Tag-Free decoder fitting
    base_members = ["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy", "pe_bkseg1", "ntg_convnext"]
    oof_names, oof_gold, oof_fold, oof_groups, P_oof = tfd.load_member_oof(base_members)

    if args.phase == "base_and_qwen":
        qwen_oof, qwen_ok = tfd.load_candidate(os.path.join(REPO, "results/r55/qwen3vl_hires_pilot"), oof_names)
        P_oof["qwen3vl_hires"] = qwen_oof

        # Fit 100% Tag-Free decoders on OOF (all 3,573 rows for full-OOF stacks; covered OOF rows for pilot-covered stacks)
        v6_mems = ["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy"]
        v6_qwen_mems = ["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "qwen3vl_hires"]
        v7_qwen_mems = ["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "qwen3vl_hires", "pe_bkseg1", "ntg_convnext"]

        dec_v6_clean = tfd.TagFreeRoutedGateEU(**tfd.build_cfg(v6_mems)).fit(
            {m: P_oof[m] for m in v6_mems}, oof_gold, oof_groups
        )
        dec_v6_qwen = tfd.TagFreeRoutedGateEU(**tfd.build_cfg(v6_qwen_mems)).fit(
            {m: P_oof[m][qwen_ok] for m in v6_qwen_mems}, oof_gold[qwen_ok], oof_groups[qwen_ok]
        )
        dec_v7_qwen = tfd.TagFreeRoutedGateEU(**tfd.build_cfg(v7_qwen_mems)).fit(
            {m: P_oof[m][qwen_ok] for m in v7_qwen_mems}, oof_gold[qwen_ok], oof_groups[qwen_ok]
        )

        summary = {}
        for ds_tag, tmpl_p, img_d in datasets:
            base_npz = os.path.join(out_dir, f"{ds_tag}_resident_v7_core_fresh_probs.npz")
            if os.path.exists(base_npz):
                z = np.load(base_npz, allow_pickle=True)
                ds_names = [str(x) for x in z["names"]]
                P_ds = {m: z[f"p_{m}"].astype(np.float64) for m in base_members}
                print(f"[{ds_tag}] Reusing fresh base probs: {base_npz}", flush=True)
            else:
                P_ds = run_sharded_final_engine(
                    set_name="resident_v7_core",
                    gpus=gpus,
                    template_path=tmpl_p,
                    img_dir=img_d,
                    merged_probs_npz=base_npz,
                    log_dir=os.path.join(out_dir, f"logs_{ds_tag}"),
                    half=True,
                )
                ds_names = [str(x) for x in c.load_template_names(tmpl_p)]

            qwen_npz = os.path.join(out_dir, f"{ds_tag}_qwen3vl_hires_fresh_probs.npz")
            P_ds["qwen3vl_hires"] = run_sharded_qwen_hires(
                ckpt_path=os.path.join(REPO, args.qwen_ckpt),
                gpus=gpus,
                template_path=tmpl_p,
                img_dir=img_d,
                merged_npz=qwen_npz,
            )

            summary[ds_tag] = {}
            for st_tag, dec_obj, mems in [
                ("tagfree_v6_core", dec_v6_clean, v6_mems),
                ("tagfree_v6_qwen_hires", dec_v6_qwen, v6_qwen_mems),
                ("tagfree_v7_qwen_hires", dec_v7_qwen, v7_qwen_mems),
            ]:
                pr = decode_and_verify_n1(dec_obj, P_ds, mems)
                out_j = os.path.join(out_dir, f"pred_{ds_tag}_{st_tag}.json")
                summary[ds_tag][st_tag] = write_and_validate_json(ds_names, pr, tmpl_p, out_j)

        with open(os.path.join(out_dir, "phase1_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("=== [Phase 1 DONE] All fresh test_150 & final_644 JSONs generated & validated! ===", flush=True)

    elif args.phase == "iv6b_and_final":
        qwen_oof, qwen_ok = tfd.load_candidate(os.path.join(REPO, "results/r55/qwen3vl_hires_pilot"), oof_names)
        iv6b_oof, iv6b_ok = tfd.load_candidate(os.path.join(REPO, "results/r55/iv6b_native_pilot"), oof_names)
        both_ok = qwen_ok & iv6b_ok
        P_oof["qwen3vl_hires"] = qwen_oof
        P_oof["iv6b_native"] = iv6b_oof

        # Lean 5-member Dual-Core Ideal Stack + 6-member Stack
        dual5_mems = ["iv6b_native", "qwen3vl_hires", "p12_pe", "ntg_convnext", "peg"]
        dual6_mems = ["cradio_v3", "iv6b_native", "qwen3vl_hires", "p12_pe", "ntg_convnext", "peg"]
        dual7_mems = ["cradio_v3", "v16", "iv6b_native", "qwen3vl_hires", "p12_pe", "ntg_convnext", "peg"]

        orig_disp = tfd._eval_only_orig_mask(oof_names)
        cov_folds = sorted(int(k) for k in np.unique(oof_fold[both_ok]))
        oof_eval_report = {}
        for st_tag, mems in [
            ("dualcore_5m_ideal", dual5_mems),
            ("dualcore_6m_plus_cradio", dual6_mems),
            ("dualcore_7m_plus_v16", dual7_mems),
        ]:
            _, pred_cv, rows_cv = tfd.nested_oof(P_oof, mems, oof_gold, oof_fold, oof_groups, eval_folds=cov_folds)
            s_cv = tfd.summarize(pred_cv[rows_cv], oof_gold[rows_cv], orig_disp[rows_cv])
            oof_eval_report[st_tag] = s_cv
            print(tfd._line(f"[Fold {cov_folds} OOF] {st_tag}", s_cv), flush=True)

        decs = {
            st_tag: (
                tfd.TagFreeRoutedGateEU(**tfd.build_cfg(mems)).fit(
                    {m: P_oof[m][both_ok] for m in mems}, oof_gold[both_ok], oof_groups[both_ok]
                ),
                mems,
            )
            for st_tag, mems in [
                ("dualcore_5m_ideal", dual5_mems),
                ("dualcore_6m_plus_cradio", dual6_mems),
                ("dualcore_7m_plus_v16", dual7_mems),
            ]
        }

        summary = {"oof_eval": oof_eval_report, "datasets": {}}
        for ds_tag, tmpl_p, img_d in datasets:
            base_npz = os.path.join(out_dir, f"{ds_tag}_resident_v7_core_fresh_probs.npz")
            qwen_npz = os.path.join(out_dir, f"{ds_tag}_qwen3vl_hires_fresh_probs.npz")
            assert os.path.exists(base_npz) and os.path.exists(qwen_npz), f"Run --phase base_and_qwen first ({ds_tag})!"
            zb = np.load(base_npz, allow_pickle=True)
            zq = np.load(qwen_npz, allow_pickle=True)
            ds_names = [str(x) for x in zb["names"]]
            P_ds = {m: zb[f"p_{m}"].astype(np.float64) for m in base_members}
            P_ds["qwen3vl_hires"] = c.renorm(zq["probs_6x4"].astype(np.float64))

            iv6b_npz = os.path.join(out_dir, f"{ds_tag}_iv6b_native_fresh_probs.npz")
            P_ds["iv6b_native"] = run_sharded_iv6b_native(
                ckpt_path=os.path.join(REPO, args.iv6b_ckpt),
                gpus=gpus,
                template_path=tmpl_p,
                img_dir=img_d,
                merged_npz=iv6b_npz,
            )

            summary["datasets"][ds_tag] = {}
            for st_tag, (dec_obj, mems) in decs.items():
                pr = decode_and_verify_n1(dec_obj, P_ds, mems)
                out_j = os.path.join(out_dir, f"pred_{ds_tag}_{st_tag}.json")
                summary["datasets"][ds_tag][st_tag] = write_and_validate_json(ds_names, pr, tmpl_p, out_j)

        with open(os.path.join(out_dir, "phase2_dualcore_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("=== [Phase 2 DONE] All Dual-Core test_150 & final_644 JSONs generated & validated! ===", flush=True)


if __name__ == "__main__":
    main()
