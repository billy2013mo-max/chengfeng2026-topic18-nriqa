#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/final_engine.py — one process, one GPU, one image at a time: the recommended cross-family ensemble.

Members (all trained on 100 % of the labelled data; nothing is tuned here):
  tier1    cradio_v4_h, pe_core_large, siglip2_so400m_naflex   results/next55/fullfit/qy/<backbone>/fold_0_best.pt
  v16      v16_full8                                            results/next55/fullfit/v16_full8/*_swa.pt
  phoenix  p1 (Qwen3-VL-8B; adapters merged, language model 4-bit: phoenix_pillar_4090.py)   checkpoints/phoenix/pillar1_fulldata_swa.pt
           p3 (hi-res physics specialist, bf16)                                             checkpoints/phoenix/pillar3_fulldata_swa.pt
Sets:  primary = all six,  fallback = primary without p1.
Pooling: family-balanced arithmetic mean (tier1 / v16 / phoenix each sum to 1; equal split inside a family) —
the fixed prior that stack_plus.py showed to be at the optimum.  Decoding: Decoder55 (objective Doc2_Strict,
best of four starting points) fitted on the pooled OUT-OF-FOLD probabilities of the same members (`--fit-decoder`,
CPU), never on a test set.

Every model is resident on the GPU; per image the engine computes all preprocessing on the fly (no precomputed
physics meta, no feature cache), runs every member with its own 2x h-flip TTA, pools and decodes.

Modes
  --fit-decoder                          fit + save results/r55/final/<set>_decoder.pkl (CPU, OOF only)
  --bench N                              time N labelled images end to end at batch 1, report per-member seconds,
                                         peak GPU memory and an in-sample sanity check (weights loaded correctly?)
  --template T --img-dir D --out J       predictions for a template in the official format (fresh inference only)

`bitsandbytes` is needed for p1 only and lives in third_party/pydeps (not in the shared environment).

Usage:
  python code/r55/final_engine.py --set primary --fit-decoder
  python code/r55/final_engine.py --set primary --gpu 7 --bench 40
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Allocator: expandable segments let the caching allocator grow/shrink one mapping instead of keeping many
# fragmented fixed-size blocks, which closes most of the gap between torch peak and nvidia-smi peak.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    sys.path.insert(0, os.path.join(REPO, _p))

# "peg" = pe_core_large trained with simulated 花屏 + matched uncorrupted frames (build_glitch.py; E3 of the reassessment):
# same recipe otherwise.  The *_v2 sets use it in place of "pe" and let it raise the pooled 花屏 probability (see pool()).
SETS = {
    "primary": ["cradio", "pe", "sig", "v16", "p1", "p3"],
    "fallback": ["cradio", "pe", "sig", "v16", "p3"],
    "primary_v2": ["cradio", "peg", "sig", "v16", "p1", "p3"],
    "fallback_v2": ["cradio", "peg", "sig", "v16", "p3"],
    "resident_v3": ["cradio_v3", "pe_v3", "sig_v3", "v16", "p3", "p12_pe"],
    "resident_v3_giant": ["cradio_v3", "pe_v3", "sig_v3", "v16", "p3", "p12_pe", "p12_g14"],
    "resident_v4": ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "p12_pe"],
    "resident_v4_pegonly": ["cradio_v3", "peg", "sig_v3", "v16", "p3_qy", "p12_pe"],
    "resident_v5": ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe"],
    "resident_v6": ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe", "p1_qy"],
    "resident_v6_core": ["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy"],
    "resident_v7": ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe", "p1_qy", "pe_bkseg1", "ntg_convnext"],
}

V16_CHUNK_BY_SET = {
    "resident_v6": 2,
    "resident_v6_core": 2,
    "resident_v7": 2,
}
CLEAN_FOOTPRINT_N = 24          # clean training frames used for the PE-operator footprint (unchanged from the runtime version)
CLEAN_PHYS_N = 160              # clean training frames for the measured-physics percentiles


def clean_footprint_path(set_name):
    return _resolve_pack_path(f"results/r55/final/{set_name}_clean_footprint.npz", f"decoders/{set_name}_clean_footprint.npz")


class NvmlPeakSampler:
    """Background sampler of device-level used memory (what nvidia-smi reports) for the GPU torch is using."""

    def __init__(self, interval_s=0.02):
        import threading
        self.interval_s, self.peak_mib, self.cur_mib, self.ok = interval_s, 0.0, 0.0, False
        self._stop = threading.Event()
        try:
            import pynvml
            import torch
            pynvml.nvmlInit()
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            uuid = str(getattr(props, "uuid", ""))
            h = None
            if uuid:
                for cand in (uuid if uuid.startswith("GPU-") else "GPU-" + uuid, uuid):
                    try:
                        h = pynvml.nvmlDeviceGetHandleByUUID(cand)
                        break
                    except Exception:
                        h = None
            if h is None:
                vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                h = pynvml.nvmlDeviceGetHandleByIndex(int(vis) if vis.isdigit() else 0)
            self._nv, self._h, self.ok = pynvml, h, True
        except Exception as e:  # pragma: no cover
            print(f"  [nvml] unavailable ({e}); nvidia-smi peak will not be reported", flush=True)
        if self.ok:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()

    def read_mib(self):
        return self._nv.nvmlDeviceGetMemoryInfo(self._h).used / 2 ** 20 if self.ok else 0.0

    def _run(self):
        while not self._stop.is_set():
            try:
                self.cur_mib = self.read_mib()
                self.peak_mib = max(self.peak_mib, self.cur_mib)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def reset(self):
        self.peak_mib = self.read_mib() if self.ok else 0.0

    def stop(self):
        self._stop.set()
FAMILY = {
    "cradio": "tier1", "pe": "tier1", "peg": "tier1", "sig": "tier1",
    "cradio_v3": "tier1", "pe_v3": "tier1", "sig_v3": "tier1", "pe_bkseg1": "tier1",
    "ntg_pe": "ntg", "ntg_convnext": "ntg",
    "p12_pe": "p12", "p12_g14": "p12",
    "v16": "v16", "p1": "phoenix", "p3": "phoenix", "p3_qy": "phoenix",
    "p1_qy": "phoenix_vlm",
}
TIER1 = {
    "cradio": "cradio_v4_h", "pe": "pe_core_large", "peg": "pe_core_large", "sig": "siglip2_so400m_naflex",
    "cradio_v3": "cradio_v4_h", "pe_v3": "pe_core_large", "sig_v3": "siglip2_so400m_naflex", "pe_bkseg1": "pe_core_large",
}
TIER1_CKPT = {
    "cradio": "results/next55/fullfit/qy/cradio_v4_h/fold_0_best.pt",
    "pe": "results/next55/fullfit/qy/pe_core_large/fold_0_best.pt",
    "sig": "results/next55/fullfit/qy/siglip2_so400m_naflex/fold_0_best.pt",
    "peg": "results/next55/towers/r55_fullglitch_s2026/pe_core_large/fold_0_best.pt",
    "cradio_v3": "results/r55/fullfit_v3_tier1/cradio_v4_h/fold_0_best.pt",
    "pe_v3": "results/r55/fullfit_v3_tier1/pe_core_large/fold_0_best.pt",
    "sig_v3": "results/r55/fullfit_v3_tier1/siglip2_so400m_naflex/fold_0_best.pt",
    "pe_bkseg1": "results/r55/fullfit_v7_bkseg1/pe_core_large/fold_0_best.pt",
}
P12_SPEC = {"p12_pe": ("pe_core_large", 9), "p12_g14": ("pe_core_g14", 12)}
P12_CKPT = {
    "p12_pe": "results/r55/fullfit_v3_p12/pe_core_large/fold_0_best.pt",
    "p12_g14": "results/r55/fullfit_v3_p12/pe_core_g14/fold_0_best.pt",
}
NTG_SPEC = {"ntg_pe": "pe_core_large", "ntg_convnext": "convnextv2_base"}
NTG_CKPT = {
    "ntg_pe": "results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt",
    "ntg_convnext": "results/r55/fullfit_v7_ntg_convnext/convnextv2_base/fold_0_best.pt",
}
V3_OOF_DIRS = {
    "cradio_v3": "results/next55/towers/r55_fixed_e4_s2026/cradio_v4_h",
    "pe_v3": "results/next55/towers/r55_fixed_s2026/pe_core_large",
    "sig_v3": "results/next55/towers/r55_fixed_s2026/siglip2_so400m_naflex",
    "ntg_pe": "results/next55/towers/r55_ntg_s2026/pe_core_large",
    "p12_pe": "results/next55/towers/r55_p12_kd_e5_s2026/pe_core_large",
    "p12_g14": "results/next55/towers/r55_p12_kd_e4_s2026/pe_core_g14",
    "pe_bkseg1": "results/next55/towers/r55_synth-bkseg1_s2026/pe_core_large",
    "ntg_convnext": "results/next55/towers/r55_ntg_convnext_e8_s2026/convnextv2_base",
}
OOF_KEY = {"cradio": "qy_cradio_v4_h", "pe": "qy_pe_core_large", "peg": "r55_glitch2_s2026_pe_core_large",
           "sig": "qy_siglip2_so400m_naflex", "v16": "v16_full8"}
BNB = os.path.join(REPO, "third_party/pydeps")
_REPO_OUT = os.path.join(REPO, "results/r55/final")
_PACK_OUT = os.path.join(os.path.dirname(REPO), "03_模型产物/decoders")
OUT = _REPO_OUT if os.path.exists(_REPO_OUT) else (_PACK_OUT if os.path.exists(_PACK_OUT) else _REPO_OUT)


def _resolve_pack_path(repo_rel: str, pack_rel: str) -> str:
    p_repo = os.path.join(REPO, repo_rel)
    if os.path.exists(p_repo):
        return p_repo
    p_pack = os.path.join(os.path.dirname(REPO), "03_模型产物", pack_rel)
    if os.path.exists(p_pack):
        return p_pack
    return p_repo


def _base_family_weights(base_m):
    from routed_eu_decoder import compute_family_base_prior
    return compute_family_base_prior(base_m)


def weights(members):
    if any(m in P12_SPEC for m in members):
        base_m = [m for m in members if m not in P12_SPEC]
        p12_m = [m for m in members if m in P12_SPEC]
        p12_tot = 0.20 if len(p12_m) >= 2 else 0.18
        bw_arr = _base_family_weights(base_m) * (1.0 - p12_tot)
        bw = {m: float(w) for m, w in zip(base_m, bw_arr)}
        if len(p12_m) == 2 and set(p12_m) == {"p12_pe", "p12_g14"}:
            pw = {"p12_pe": p12_tot * 0.65, "p12_g14": p12_tot * 0.35}
        else:
            pw = {m: p12_tot / len(p12_m) for m in p12_m}
        return np.array([bw[m] if m in bw else pw[m] for m in members], dtype=np.float64)
    return _base_family_weights(members)


def pool(probs, members):
    """probs: dict member -> (N, 6, 4).  Family-balanced arithmetic mean, renormalised.
    For v4/v5/v6/v6_core ensembles (containing pe_v3, p12_pe, ntg_pe, or p1_qy), uses
    high-confidence peg >= 0.90 unseen-source 花屏 rescue.
    If ONLY legacy *_v2 / resident_v4_pegonly is used without p12/ntg/p1_qy, preserves
    the exact unthresholded max-rescue rule."""
    import common55 as c
    use_thr_rescue = "peg" in members and (
        "pe_v3" in members
        or any(m in P12_SPEC for m in members)
        or "ntg_pe" in members
        or "p1_qy" in members
    )
    if use_thr_rescue:
        p = c.renorm(c.mean_pool([np.asarray(probs[m], dtype=np.float64) for m in members], list(weights(members))))
        pg = c.renorm(np.asarray(probs["peg"], dtype=np.float64))
        m_res = pg[:, 3, 3] >= 0.90
        if np.any(m_res):
            q = np.maximum(p[m_res, 3, 3], pg[m_res, 3, 3])
            p[m_res, 3, :] = np.stack([1.0 - q, np.full_like(q, 1e-6), np.full_like(q, 1e-6), q], 1)
            p = c.renorm(p)
        return p
    p = c.renorm(c.mean_pool([np.asarray(probs[m], dtype=np.float64) for m in members], list(weights(members))))
    if "peg" in members:
        g = c.renorm(np.asarray(probs["peg"], dtype=np.float64))
        q = np.maximum(p[:, 3, 3], g[:, 3, 3])
        p[:, 3, :] = np.stack([1.0 - q, np.full_like(q, 1e-6), np.full_like(q, 1e-6), q], 1)
        p = c.renorm(p)
    return p


def oof_pool(members, stage="stage2-rl2r"):
    import glob
    import common55 as c
    names, gold, fold, _ = c.load_split()
    use_v3 = any(m in V3_OOF_DIRS for m in members)
    legacy_keys = [OOF_KEY[m] for m in members if (m in OOF_KEY and not (use_v3 and m == "v16"))]
    bank = c.build_bank(only=legacy_keys) if legacy_keys else {}
    idx = {n: i for i, n in enumerate(names)}
    P = {}
    for m in members:
        if m in V3_OOF_DIRS:
            from eval_specialist_gate import load_oof_dir
            arr, valid = load_oof_dir(V3_OOF_DIRS[m], names)
            assert valid.all(), f"Incomplete OOF for {m}: {valid.sum()}/{len(names)}"
            P[m] = c.renorm(arr)
        elif use_v3 and m == "v16":
            # Strictly video-grouped zero-leakage V16 OOF (`v16_qy`, Audit 14)
            pv = np.zeros((len(names), 6, 4), dtype=np.float64)
            for f in sorted(glob.glob(os.path.join(REPO, "results/r55/v16_qy/V16_Full_F*_probs.npz"))):
                z = np.load(f, allow_pickle=True)
                for n, v in zip(z["va_names"], c.p012_to_6x4_simplex(z["va_p0"], z["va_p1"], z["va_p2"])):
                    pv[idx[str(n)]] = v
            P[m] = c.renorm(pv)
        elif m == "p3_qy":
            # Strictly video-grouped zero-leakage Pillar-3 8-fold OOF (`p3_qy`, Stage 1A)
            p3_oof_path = os.path.join(REPO, "results/r55/p3_qy_honest/pillar3_best_oof.npz")
            assert os.path.exists(p3_oof_path), f"Missing honest 8-fold Pillar3 OOF: {p3_oof_path}"
            z = np.load(p3_oof_path, allow_pickle=True)
            mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
            P[m] = c.renorm(np.stack([mp[n] for n in names]))
        elif m == "p1_qy":
            # Strictly video-grouped zero-leakage c233397 Qwen3-VL-8B 8-fold OOF (`p1_qy`, v6 Wave 1)
            p1_oof_path = os.path.join(REPO, "results/r55/vlm_qy8/p1_c233397_s2026_oof.npz")
            assert os.path.exists(p1_oof_path), f"Missing honest 8-fold p1_qy OOF: {p1_oof_path}"
            z = np.load(p1_oof_path, allow_pickle=True)
            assert bool(np.all(z["valid_mask"])), f"Incomplete p1_qy OOF: {z['valid_mask'].sum()}/{len(names)}"
            mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
            P[m] = c.renorm(np.stack([mp[n] for n in names]))
        elif m in OOF_KEY:
            P[m] = c.stack_oof(bank, [OOF_KEY[m]], names)[0]
        else:
            z = np.load(os.path.join(REPO, f"results/phoenix/pillar{m[1]}_{stage}_oof.npz"), allow_pickle=True)
            mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
            P[m] = c.renorm(np.stack([mp[n] for n in names]))
    return names, gold, fold, pool(P, members), P


def fit_decoder(set_name):
    import csv
    import common55 as c
    from decoder55 import argmax_decode, nested_oof_decode
    members = SETS[set_name]
    names, gold, fold, p, P_dict = oof_pool(members)
    from decoder_select import fit_multistart
    dec = fit_multistart(p, gold)          # Decoder55 from four starting points; nested it is +0.06 over a single start (n.s.), never worse
    K = "doc2_strict_total"
    rep = dict(set=set_name, members=members, weights=[float(x) for x in weights(members)],
               oof_argmax=float(c.score(argmax_decode(p), gold)[K]),
               oof_nested=float(c.score(nested_oof_decode(p, gold, fold, K), gold)[K]),
               oof_insample_decoder=float(c.score(dec.predict(p), gold)[K]))
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(dict(decoder=dec, members=members, weights=rep["weights"]), open(os.path.join(OUT, f"{set_name}_decoder.pkl"), "wb"))
    json.dump(rep, open(os.path.join(OUT, f"{set_name}_decoder.json"), "w"), indent=2)
    print(f"[final_engine] decoder for '{set_name}': OOF argmax {rep['oof_argmax']:.2f} | nested-decoder {rep['oof_nested']:.2f} "
          f"| decoder fitted on all OOF (optimistic) {rep['oof_insample_decoder']:.2f} -> {OUT}/{set_name}_decoder.pkl")

    if any(m in V3_OOF_DIRS for m in members):
        import decoder_eu as E
        from eval_specialist_gate import load_p0_fallback_base
        from routed_eu_decoder import (
            RoutedEUDecoder,
            RoutedGateEUDecoder,
            evaluate_nested_routed_eu,
            evaluate_nested_routed_gate_eu,
        )
        from subjective_engineering import EffectiveFrameORVideoAggregator

        _, _, _, groups = c.load_split()
        split_map = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
        is_orig = np.array([split_map[n] == "orig" for n in names])

        tag_fold = None
        tag_csv = os.path.join(REPO, "data/r55_tagsplit/folds.csv")
        if os.path.exists(tag_csv):
            tm = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(tag_csv))}
            tag_fold = np.array([tm[n] for n in names], dtype=np.int64)

        # Plain EU decoder on pooled p
        eu_dec = E.DecoderEU().fit(p, gold)
        pickle.dump(dict(decoder=eu_dec, members=members, weights=rep["weights"]),
                    open(os.path.join(OUT, f"{set_name}_decoder_eu.pkl"), "wb"))

        # RoutedEUDecoder & RoutedGateEUDecoder
        base_m = [m for m in members if m not in P12_SPEC]
        base_prior = list(_base_family_weights(base_m))
        p12_m = [m for m in members if m in P12_SPEC]
        p12_int = [0.65, 0.35] if set(p12_m) == {"p12_pe", "p12_g14"} else [1.0 / max(1, len(p12_m))] * len(p12_m)
        p12_tot = 0.20 if len(p12_m) >= 2 else 0.18
        proto_cfg = dict(
            base_members=base_m,
            base_prior=base_prior,
            p12_members=p12_m,
            p12_internal_weights=p12_int,
            p12_total_prior=p12_tot,
            l2_reg=0.015,
        )
        P_p0 = load_p0_fallback_base(names)
        Q_p0 = pool(P_p0, ["cradio", "pe", "sig", "v16", "p3"])

        nested_rep = evaluate_nested_routed_eu(P_dict, gold, fold, groups, is_orig, proto_cfg, Q_p0)
        routed_dec = RoutedEUDecoder(**proto_cfg).fit(P_dict, gold, is_orig=is_orig)
        nested_rep.update(
            set=set_name,
            members=members,
            temps=[float(x) for x in routed_dec.temps],
            W=routed_dec.W.tolist(),
            eu_lam=float(routed_dec.eu.lam),
            oof_insample_routed_eu=float(c.score(routed_dec.predict_from_member_dict(P_dict), gold)[K]),
        )
        pickle.dump(dict(decoder=routed_dec, members=members), open(os.path.join(OUT, f"{set_name}_decoder_routed_eu.pkl"), "wb"))
        json.dump(nested_rep, open(os.path.join(OUT, f"{set_name}_decoder_routed_eu.json"), "w"), indent=2, ensure_ascii=False)
        print(f"[final_engine] routed_eu for '{set_name}': honest nested Doc2S all={nested_rep['oof_nested_doc2_all']:.3f} "
              f"orig={nested_rep['oof_nested_doc2_orig']:.3f} | RPS={nested_rep['oof_rps_all']:.5f} "
              f"(Δ={nested_rep['delta_rps_vs_p0fb']:+.5f} CI90={nested_rep['delta_rps_ci90']}) "
              f"mAP14={nested_rep['oof_mAP14']:.4f} -> {OUT}/{set_name}_decoder_routed_eu.pkl")

        # Stage 3 Breakthrough: RoutedGateEUDecoder (TrueGateConstrainedEU + Hybrid PE + peg>=0.90 rescue)
        proto_gate_cfg = dict(proto_cfg)
        if "p1_qy" in members:
            proto_gate_cfg["lams_cl"] = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)
        if "pe_bkseg1" in members:
            proto_gate_cfg["bkseg1_alpha"] = 0.25
        gate_rep = evaluate_nested_routed_gate_eu(
            P_dict, gold, fold, groups, is_orig, proto_gate_cfg, Q_p0, tag_fold=tag_fold
        )
        gate_dec = RoutedGateEUDecoder(**proto_gate_cfg).fit(P_dict, gold, is_orig=is_orig, groups=groups)
        eu_st_json = {
            k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in gate_dec.eu.st.items()
        }
        gate_rep.update(
            set=set_name,
            members=members,
            base_members=list(gate_dec.base_members),
            base_prior=[float(x) for x in gate_dec.base_prior],
            p12_members=list(gate_dec.p12_members),
            p12_internal_weights=[float(x) for x in gate_dec.p12_internal_weights],
            p12_total_prior=float(gate_dec.p12_total_prior),
            peg_rescue_thr=float(gate_dec.peg_rescue_thr),
            temps=[float(x) for x in gate_dec.temps],
            W=gate_dec.W.tolist(),
            dirdiag_slopes=[float(x) for x in np.exp(gate_dec.dirdiag_params[:, 0])] if gate_dec.dirdiag_params is not None else [],
            dirdiag_params=gate_dec.dirdiag_params.tolist() if gate_dec.dirdiag_params is not None else [],
            eu_lam_cl=float(gate_dec.eu.lam_cl),
            eu_p0_scale=float(gate_dec.eu.p0_scale),
            eu_st=eu_st_json,
            oof_insample_routed_gate_eu=float(c.score(gate_dec.predict_from_member_dict(P_dict), gold)[K]),
        )
        pickle.dump(dict(decoder=gate_dec, members=members), open(os.path.join(OUT, f"{set_name}_decoder_routed_gate_eu.pkl"), "wb"))
        json.dump(gate_rep, open(os.path.join(OUT, f"{set_name}_decoder_routed_gate_eu.json"), "w"), indent=2, ensure_ascii=False)
        print(f"[final_engine] routed_gate_eu for '{set_name}': honest nested Doc2S all={gate_rep['oof_nested_doc2_all']:.3f} "
              f"orig={gate_rep['oof_nested_doc2_orig']:.3f} (Pos_orig={gate_rep['oof_nested_pos_orig']:.2f}, "
              f"P0_orig={gate_rep['oof_nested_p0_orig']:.2f}, Cln_orig={gate_rep['oof_nested_clean_orig']:.2f}) | "
              f"Perm654(Pos<15)={gate_rep['perm654_orig_pos_lt15_pct']:.2f}% ClBoot654(Pos<15)={gate_rep['boot654_orig_pos_fail_pct']:.2f}% "
              f"-> {OUT}/{set_name}_decoder_routed_gate_eu.pkl")

        # Stage 4: Official §5 Compliant Video Aggregator fitted on honest pooled OOF
        Q_full = gate_dec.pool_from_member_dict(P_dict)
        vid_agg = EffectiveFrameORVideoAggregator().fit(Q_full, gold, groups, gate_dec.eu.predict(Q_full))
        pickle.dump(dict(video_aggregator=vid_agg, members=members), open(os.path.join(OUT, f"{set_name}_video_aggregator.pkl"), "wb"))


class FinalEngine:
    def __init__(self, members, device="cuda", img_dir="", half=False, v16_chunk=0):
        import torch
        from qingyu.backbones_qingyu import QingyuVisionQualityTower
        from qingyu.config_and_paths import REPO_ROOT, LosslessNativeImageLoader, setup_sys_path_and_cwd
        from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor, PrecomputedDualEngineStore
        from qingyu.train_tier1_5backbones_8fold import make_tier1_collate_fn
        setup_sys_path_and_cwd(chdir_repo=True)
        self.torch, self.members, self.dev = torch, list(members), torch.device(device)
        self.img_dir = img_dir
        if img_dir:
            os.environ["NEXT55_EXTRA_IMG_DIR"] = img_dir          # V16 feature / image lookup (read at import)
        self.mem = {}
        t0 = time.time()

        def note(tag):
            torch.cuda.synchronize()
            gc.collect()                                          # drop CPU-side state dicts / temporaries of this member
            torch.cuda.empty_cache()                              # return freed blocks (e.g. p1_qy bf16->4bit build) to the driver
            self.mem[tag] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
            print(f"  [load] {tag:8s} resident {self.mem[tag]:6.2f} GB | peak so far {torch.cuda.max_memory_allocated() / 2 ** 30:6.2f} GB "
                  f"| {time.time() - t0:5.0f}s", flush=True)

        # Phoenix first: building p1/p1_qy needs the bf16 model (~16 GB) before it shrinks to ~6 GB
        self.p3_key = "p3_qy" if "p3_qy" in members else ("p3" if "p3" in members else None)
        if "p1" in members or "p1_qy" in members or self.p3_key is not None:
            from phoenix.config import PHOENIX_CKPT_DIR, resolve_local_or_hf_model
            from phoenix.native_physics_and_views import CoCDepthDefocusProbe
            self.coc = CoCDepthDefocusProbe(model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"), device=str(self.dev))
        if "p1" in members:
            from phoenix_pillar_4090 import build_pillar_for_4090
            self.p1 = build_pillar_for_4090("pillar1", device=str(self.dev), bnb_path=BNB, verbose=False)
            note("p1")
        if "p1_qy" in members:
            from vlm_c233397_qy8 import build_c233397_pillar_for_4090
            p1_qy_ckpt = os.environ.get(
                "R55_P1_QY_CKPT_OVERRIDE",
                _resolve_pack_path(
                    "results/r55/vlm_qy8/ckpts/p1_c233397_s2026_fullfit_swa.pt",
                    "checkpoints/p1_qy/p1_c233397_s2026_fullfit_swa.pt",
                ),
            )
            assert os.path.exists(p1_qy_ckpt), f"Missing honest full-fit p1_qy checkpoint: {p1_qy_ckpt}"
            self.p1_qy = build_c233397_pillar_for_4090(
                ckpt_path=p1_qy_ckpt, pillar="pillar1", device=str(self.dev), bnb_path=BNB, four_bit=True, verbose=False
            )
            note("p1_qy")
        if self.p3_key is not None:
            from phoenix.train_pillar12_native_vlm import detect_lora_rank_from_state_dict
            from phoenix.train_pillar3_hires_physics import Pillar3HiResPhysicsSpecialist
            if self.p3_key == "p3_qy":
                p3_ckpt = os.path.join(REPO, "results/r55/p3_qy_honest_fullfit/pillar3_fulldata_swa.pt")
                assert os.path.exists(p3_ckpt), f"Missing honest full-fit Pillar3 checkpoint: {p3_ckpt}"
            else:
                p3_ckpt = os.path.join(str(PHOENIX_CKPT_DIR), "pillar3_fulldata_swa.pt")
            sd = torch.load(p3_ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd)
            self.p3_legacy = "vision_tower.backbone_gate_logits" not in sd
            self.p3 = Pillar3HiResPhysicsSpecialist(mock_backbone=False, lora_rank=detect_lora_rank_from_state_dict(sd, default_rank=24),
                                                    legacy_3view=self.p3_legacy)
            res = self.p3.load_state_dict(sd, strict=False)
            assert not res.unexpected_keys, res.unexpected_keys[:5]
            if half:
                for sub_n in ("dinov3", "cradio", "convnext", "pecore"):
                    sub_m = getattr(self.p3.vision_tower, sub_n, None)
                    if sub_m is not None:
                        for fn, mod in sub_m.named_modules():
                            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)) and "lora" not in fn.lower():
                                mod.to(torch.bfloat16)
            self.p3.to(self.dev).eval()
            note(self.p3_key)

        # Tier-1 towers: empty meta store -> the physics meta of every image is computed on the fly, once, and shared
        self.loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=[img_dir] if img_dir else None)
        self.store = PrecomputedDualEngineStore("__no_precomputed_meta__.npz",
                                                lazy_extractor=DualEnginePhysicsExtractor(device=str(self.dev), allow_depth_download=False))
        self.t1 = {}
        for m in [x for x in members if x in TIER1]:
            bb = TIER1[m]
            model = QingyuVisionQualityTower(backbone_key=bb, phys_dim=PHYS_FEAT_DIM)
            _, unexpected = model.load_state_dict(
                torch.load(_resolve_pack_path(TIER1_CKPT[m], f"checkpoints/{m}/fold_0_best.pt"), map_location="cpu"),
                strict=False,
            )
            assert not unexpected, unexpected[:5]
            if half:                                             # converted on the CPU so the GPU never holds the fp32 copy
                model.backbone.to(torch.bfloat16)                # weights only; the forward already runs under bf16 autocast
                for n, mod in model.backbone.named_modules():    # read-out modules run in fp32 by design (autocast disabled)
                    if any(t in n.split(".")[-1] for t in ("tap3", "dual_tap", "abmil", "spatial_proj")):
                        mod.float()
            model.to(self.dev).eval()
            self.t1[m] = (model, make_tier1_collate_fn(model.mode_tag, naflex_processor=getattr(model, "naflex_processor", None), max_num_patches=784))
            note(m)

        self.ntg = {}
        ntg_ms = [x for x in members if x in NTG_SPEC]
        if ntg_ms:
            from native_tiles import NativeTileTower
            from qingyu.train_tier1_5backbones_8fold import Tier1DualEngineDataset
            Tier1DualEngineDataset.TILE_GLOBAL = True
            for m in ntg_ms:
                bb = NTG_SPEC[m]
                model = NativeTileTower(backbone_key=bb, phys_dim=PHYS_FEAT_DIM, use_global=True)
                _, unexpected = model.load_state_dict(
                    torch.load(_resolve_pack_path(NTG_CKPT[m], f"checkpoints/{m}/fold_0_best.pt"), map_location="cpu"),
                    strict=False,
                )
                assert not unexpected, unexpected[:5]
                if half:
                    model.backbone.to(torch.bfloat16)
                    for n, mod in model.backbone.named_modules():
                        if any(t in n.split(".")[-1] for t in ("tap3", "dual_tap", "abmil", "spatial_proj")):
                            mod.float()
                model.to(self.dev).eval()
                self.ntg[m] = (model, make_tier1_collate_fn(model.mode_tag, naflex_processor=getattr(model, "naflex_processor", None), max_num_patches=784))
                note(m)

        self.p12 = {}
        self.p12_store = None
        p12_ms = [x for x in members if x in P12_SPEC]
        if p12_ms:
            from semantic_roi_meta import SEMANTIC_PHYS_TOTAL_DIM, SemanticRoiLazyExtractor
            from train_p12_specialist import P12SemanticRoiTower, make_p12_collate
            self.p12_store = PrecomputedDualEngineStore(
                "__no_precomputed_p12_meta__.npz",
                lazy_extractor=SemanticRoiLazyExtractor(base_extractor=self.store.lazy_extractor),
            )
            for m in p12_ms:
                bb, unf = P12_SPEC[m]
                model = P12SemanticRoiTower(backbone_key=bb, phys_dim=SEMANTIC_PHYS_TOTAL_DIM, unfreeze_last_k=unf)
                _, unexpected = model.load_state_dict(
                    torch.load(_resolve_pack_path(P12_CKPT[m], f"checkpoints/{m}/fold_0_best.pt"), map_location="cpu"),
                    strict=False,
                )
                assert not unexpected, unexpected[:5]
                if half:
                    model.backbone.to(torch.bfloat16)
                    for n, mod in model.backbone.named_modules():
                        if any(t in n.split(".")[-1] for t in ("tap3", "dual_tap", "abmil", "masked_abmil")):
                            mod.float()
                model.to(self.dev).eval()
                self.p12[m] = (model, make_p12_collate(model.mode_tag, naflex_processor=getattr(model, "naflex_processor", None), max_num_patches=784))
                note(m)

        if "v16" in members:
            import glob
            import train_v16_microscope_specialist as v16
            from extract_native_phys_feats import compute_one_image_feats
            os.chdir(REPO)
            v16_cands = glob.glob(os.path.join(REPO, "results/next55/fullfit/v16_full8/V16_Full_F0_*_swa.pt"))
            v16_ckpt = v16_cands[0] if v16_cands else _resolve_pack_path(
                "results/next55/fullfit/v16_full8/V16_Full_F0_AllGrade_Microscope_swa.pt",
                "checkpoints/v16/V16_Full_F0_AllGrade_Microscope_swa.pt",
            )
            ck = torch.load(v16_ckpt, map_location="cpu", weights_only=False)
            tr = ck["args"]
            net = v16.MultiNetV16([v16.build_v9_backbone(it.split(":")[0]) for it in tr["backbones"].split(",")], phys_dim=60,
                                  dropout=tr["dropout"], n_quad=tr["tiles"] * tr["tiles"])
            _, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
            assert not unexpected, unexpected[:5]
            if half:
                net.bs.to(torch.bfloat16)                        # the five backbones; heads stay fp32
                if "p1_qy" in members and not v16_chunk:
                    v16_chunk = 2                                # bound 8-view activation spike in 9-member resident_v6 (TV == 0.0)
            net.to(self.dev)
            if v16_chunk:                                        # the 8 views of an image go through each backbone in small chunks
                net.fwds = [(lambda z, f=f: torch.cat([f(z[j:j + v16_chunk]) for j in range(0, z.shape[0], v16_chunk)], 0)) for f in net.fwds]
            net.eval()
            self.v16 = (v16, net, ck["phys_mu"], ck["phys_sd"], tr, compute_one_image_feats)
            note("v16")

        # Count resident PyTorch parameters across all loaded modules
        seen_p = set()
        n_params = 0
        mods_all = []
        if hasattr(self, "p1"):
            mods_all.append(self.p1)
        if hasattr(self, "p1_qy"):
            mods_all.append(self.p1_qy)
        if hasattr(self, "p3"):
            mods_all.append(self.p3)
        for mod_t1, _ in self.t1.values():
            mods_all.append(mod_t1)
        for mod_ntg, _ in self.ntg.values():
            mods_all.append(mod_ntg)
        for mod_p12, _ in self.p12.values():
            mods_all.append(mod_p12)
        if hasattr(self, "v16"):
            mods_all.append(self.v16[1])
        for mod_obj in mods_all:
            if hasattr(mod_obj, "parameters"):
                for prm in mod_obj.parameters():
                    if id(prm) not in seen_p:
                        seen_p.add(id(prm))
                        n_params += int(prm.numel())
        self.total_params_b = round(n_params / 1e9, 3)
        gc.collect()
        torch.cuda.empty_cache()
        self.load_s = time.time() - t0

    # ------------------------------------------------------------------ members
    def _tier1(self, m, name):
        from qingyu.train_tier1_5backbones_8fold import predict_2x_tta
        model, collate = self.t1[m]
        p, _, _, _ = predict_2x_tta(model, [{"name": name, "group_id": name, "grades": [0] * 6}], self.loader, self.store, collate,
                                    self.dev, batch_size=1, workers=0)
        return p[0]

    def _ntg(self, m, name):
        from qingyu.train_tier1_5backbones_8fold import predict_2x_tta
        model, collate = self.ntg[m]
        p, _, _, _ = predict_2x_tta(model, [{"name": name, "group_id": name, "grades": [0] * 6}], self.loader, self.store, collate,
                                    self.dev, batch_size=1, workers=0)
        return p[0]

    def _p12(self, m, name):
        from semantic_roi_meta import analyze_semantic_rois_pil
        from train_p12_specialist import predict_p12_tta
        if str(name) not in self.p12_store.map and str(name) in self.store.map:
            im = self.loader.load_pil(name).convert("RGB")
            self.p12_store.map[str(name)] = analyze_semantic_rois_pil(im, base_phys_36=self.store.map[str(name)]["phys_vec"])
        model, collate = self.p12[m]
        p, _, _, _ = predict_p12_tta(model, [{"name": name, "group_id": name, "grades": [0] * 6}], self.loader, self.p12_store, collate,
                                     self.dev, batch_size=1, workers=0)
        return p[0]

    def _v16(self, name):
        from torch.utils.data import DataLoader
        from qingyu.certainty_gated_stacker import p012_to_6x4_simplex
        v16, net, mu, sd, tr, feats = self.v16
        nm, vec = feats(name)
        assert np.any(vec), f"V16 cannot find image {name}"
        rows = [{"name": name, **{c: 0 for c in v16.CLASSES}}]
        outs = []
        for hf in (False, True):
            ds = v16.V16NativeMicroscopeDataset(rows, {name: vec}, mu, sd, tr["size"], False, hflip=hf, tiles=tr["tiles"])
            outs.append(v16.predict_v16(net, DataLoader(ds, batch_size=1, shuffle=False, num_workers=0), self.dev, tr["alpha_ex"]))
        p0, p1, p2 = [0.5 * (outs[0][i] + outs[1][i]) for i in range(3)]
        return p012_to_6x4_simplex(p0, p1, p2)[0]

    def _phoenix(self, pil):
        """p1/p1_qy and p3/p3_qy averaged over the two orientations."""
        from PIL import Image
        from phoenix.native_physics_and_views import compute_1to1_wavelet_and_artifact_maps, resize_native_aspect_pil, select_scrutiny_1to1_patches
        from phoenix.train_pillar3_hires_physics import build_pillar3_sample_tensors
        torch = self.torch
        acc, sec = {}, {"p1": 0.0, "p1_qy": 0.0}
        if self.p3_key is not None:
            sec[self.p3_key] = 0.0
        if "p1_qy" in self.members:
            t = time.perf_counter()
            maps0 = compute_1to1_wavelet_and_artifact_maps(np.asarray(pil, dtype=np.uint8))
            crops0, _ = select_scrutiny_1to1_patches(
                pil_img=pil, phys_maps=maps0, coc_field=self.coc.compute_coc_jacobian_field(pil, maps0), patch_size=384, num_patches=2
            )
            g0 = resize_native_aspect_pil(pil, max_pixels=784 * 784, grid_factor=28)
            views_orig = [g0, crops0[0], crops0[1]]
            views_flip = [v.transpose(Image.FLIP_LEFT_RIGHT) for v in views_orig]
            for vw in (views_orig, views_flip):
                oq = self.p1_qy([vw], batch_ids=None, apply_batch_bias=False, device=self.dev)
                acc.setdefault("p1_qy", []).append(oq["probs_6x4"].float().cpu().numpy()[0])
            torch.cuda.synchronize()
            sec["p1_qy"] += time.perf_counter() - t
        for im in (pil, pil.transpose(Image.FLIP_LEFT_RIGHT)):
            if "p1" in self.members:
                t = time.perf_counter()
                maps = compute_1to1_wavelet_and_artifact_maps(np.asarray(im, dtype=np.uint8))
                crops, _ = select_scrutiny_1to1_patches(pil_img=im, phys_maps=maps, coc_field=self.coc.compute_coc_jacobian_field(im, maps),
                                                        patch_size=384, num_patches=2)
                o = self.p1([[resize_native_aspect_pil(im, max_pixels=784 * 784, grid_factor=28), crops[0], crops[1]]],
                            batch_ids=None, apply_batch_bias=False, device=self.dev)
                acc.setdefault("p1", []).append(o["probs_6x4"].float().cpu().numpy()[0])
                torch.cuda.synchronize()
                sec["p1"] += time.perf_counter() - t
            if self.p3_key is not None:
                t = time.perf_counter()
                pk = build_pillar3_sample_tensors(im, coc_probe=self.coc, include_mid_tex=not self.p3_legacy)
                o = self.p3(views_01=pk["views_tensor"].unsqueeze(0).to(self.dev), phys_9ch=pk["phys_9ch"].unsqueeze(0).to(self.dev),
                            probe_8ch=pk["probe_8ch"].unsqueeze(0).to(self.dev), batch_ids=None, apply_batch_bias=False)
                acc.setdefault(self.p3_key, []).append(o["probs_6x4"].float().cpu().numpy()[0])
                torch.cuda.synchronize()
                sec[self.p3_key] += time.perf_counter() - t
        return {k: np.mean(v, 0) for k, v in acc.items()}, sec

    def infer(self, name):
        """-> (dict member -> (6,4) probabilities, dict member -> seconds)."""
        torch = self.torch
        probs, sec = {}, {}
        self.peak = getattr(self, "peak", {})

        def mark(tag):
            self.peak[tag] = max(self.peak.get(tag, 0.0), torch.cuda.max_memory_allocated() / 2 ** 30)
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            for m in self.t1:
                t = time.perf_counter()
                probs[m] = self._tier1(m, name)
                torch.cuda.synchronize()
                sec[m] = time.perf_counter() - t
                mark(m)
            for m in self.ntg:
                t = time.perf_counter()
                probs[m] = self._ntg(m, name)
                torch.cuda.synchronize()
                sec[m] = time.perf_counter() - t
                mark(m)
            for m in self.p12:
                t = time.perf_counter()
                probs[m] = self._p12(m, name)
                torch.cuda.synchronize()
                sec[m] = time.perf_counter() - t
                mark(m)
            if "v16" in self.members:
                t = time.perf_counter()
                probs["v16"] = self._v16(name)
                torch.cuda.synchronize()
                sec["v16"] = time.perf_counter() - t
                mark("v16")
            if "p1" in self.members or "p1_qy" in self.members or self.p3_key is not None:
                pp, ps = self._phoenix(self.loader.load_pil(name).convert("RGB"))
                probs.update(pp)
                sec.update({k: v for k, v in ps.items() if k in pp})
                mark("phoenix")
        self.store.map.pop(str(name), None)                    # keep the on-the-fly meta store from growing
        if self.p12_store is not None:
            self.p12_store.map.pop(str(name), None)
        return probs, sec


# ---------------------------------------------------------------------------------------------- decoding (N = 1 safe)
def load_decoder(set_name, rule):
    """-> decode(P_dict) -> (pred (N, 6) int, probabilities the rule decoded (N, 6, 4)).  Row-wise, so N = 1 == N = 150."""
    members = SETS[set_name]
    if rule == "d55":
        dp = os.path.join(OUT, f"{set_name}_decoder.pkl")
        assert os.path.exists(dp), f"run --fit-decoder first ({dp})"
        dec = pickle.load(open(dp, "rb"))["decoder"]

        def decode(P):
            q = pool(P, members)
            return dec.predict(q), q
        return decode, dec
    if rule in ("routed_eu", "routed_gate_eu"):
        import routed_eu_decoder  # noqa: F401
        rp = os.path.join(OUT, f"{set_name}_decoder_{rule}.pkl")
        assert os.path.exists(rp), f"run `python code/r55/final_engine.py --set {set_name} --fit-decoder` first ({rp})"
        d = pickle.load(open(rp, "rb"))["decoder"]

        def decode(P):
            q = d.pool_from_member_dict(P)
            return d.eu.predict(q), q
        return decode, d.eu
    import decoder_eu  # noqa: F401  (classes needed to unpickle)
    rp = os.path.join(OUT, f"{set_name}_decoder_{rule}.pkl")
    assert os.path.exists(rp), f"run `python code/r55/decoder_eu.py --set {set_name}` first ({rp})"
    d = pickle.load(open(rp, "rb"))["decoder"]

    def decode(P):
        q = pool(P, members)
        return d.predict(q), q
    return decode, d


# ---------------------------------------------------------------------------------------------- attribution runtime
_FP_MODES = ("midblock_4x4", "sub8_blk15")


def _attr_tower_keys(t1_members):
    pe_key = "pe_v3" if "pe_v3" in t1_members else ("peg" if "peg" in t1_members else ("pe" if "pe" in t1_members else None))
    peg_extra = "peg" if ("peg" in t1_members and pe_key != "peg") else None
    return pe_key, peg_extra


def build_clean_footprint(set_name, half=True):
    """OFFLINE, run once: Clean-footprint of the PE attribution operator(s) + clean physics percentiles, from clean
    TRAINING frames only -> results/r55/final/<set>_clean_footprint.npz.  Inference then never touches training data."""
    import csv
    import torch
    from defect_attribution_map import PEAttributionOperator
    import measured_evidence as ME
    from qingyu.config_and_paths import DIMS
    pe_key, peg_extra = _attr_tower_keys([m for m in SETS[set_name] if m in TIER1])
    assert pe_key is not None, f"set {set_name} has no PE tower for attribution"
    towers = [pe_key] + ([peg_extra] if peg_extra else [])
    eng = FinalEngine(towers, half=half)
    clean = []
    for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8")):
        if all(int(r[d]) == 0 for d in DIMS):
            clean.append(r["name"])
    fp_names = clean[:CLEAN_FOOTPRINT_N]
    ph_names = clean[:CLEAN_PHYS_N]
    out = {}
    t0 = time.time()
    fp_pils = [eng.loader.load_pil(n).convert("RGB") for n in fp_names]
    with torch.no_grad():
        for tk in towers:
            op = PEAttributionOperator(eng.t1[tk][0], eng.dev)
            op.fit_clean_footprint(fp_pils, mode="midblock_hier8")
            for mode in _FP_MODES:
                out[f"fp__{tk}__{mode}"] = op.footprints[mode].astype(np.float32)
    t_fp = time.time() - t0
    del fp_pils
    t0 = time.time()
    out.update(ME.build_clean_physics_reference([eng.loader.load_pil(n).convert("RGB") for n in ph_names]))
    t_ph = time.time() - t0
    meta = dict(set=set_name, towers=towers, tower_ckpts={tk: TIER1_CKPT[tk] for tk in towers}, half=bool(half),
                footprint_modes=list(_FP_MODES), footprint_names=fp_names, physics_names=ph_names,
                source="data/manifest_combined.csv rows with all six grades == 0 (training split 'orig'); no test data",
                built=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(OUT, exist_ok=True)
    op_path = os.path.join(OUT, f"{set_name}_clean_footprint.npz")
    np.savez_compressed(op_path, meta_json=np.array(json.dumps(meta, ensure_ascii=False)), **out)
    print(f"[final_engine] clean footprint ({len(fp_names)} frames, {t_fp:.0f}s) + physics reference ({len(ph_names)} frames, "
          f"{t_ph:.0f}s) -> {op_path} ({os.path.getsize(op_path) / 1024:.1f} KiB)")
    return op_path


class AttributionRuntime:
    """Per-image (N = 1) measured-evidence attribution.  Needs only the resident engine + the precomputed npz."""

    def __init__(self, eng, set_name, fp_path=None):
        from defect_attribution_map import PEAttributionOperator
        fp_path = fp_path or clean_footprint_path(set_name)
        assert os.path.exists(fp_path), (f"missing {fp_path}: run `python code/r55/final_engine.py --set {set_name} "
                                         f"--build-clean-footprint --half --gpu 0` once (offline, training frames only)")
        z = np.load(fp_path, allow_pickle=False)
        self.meta = json.loads(str(z["meta_json"]))
        self.fp_path = fp_path
        self.clean_ref = {k: z[k] for k in z.files if k.startswith("phys_")}
        self.pe_key, self.peg_extra = _attr_tower_keys(list(eng.t1))
        assert self.pe_key in self.meta["towers"], (self.pe_key, self.meta["towers"])
        self.eng = eng

        def mk(tk):
            op = PEAttributionOperator(eng.t1[tk][0], eng.dev)
            for mode in _FP_MODES:
                op.footprints[mode] = z[f"fp__{tk}__{mode}"]
            op.footprints["midblock_hier8"] = np.zeros((6, 8, 8), dtype=np.float32)
            return op
        self.op_main = mk(self.pe_key)
        self.op_peg = mk(self.peg_extra) if self.peg_extra else None

    def attribute(self, pil, pred_6d, probs_6x4):
        """-> (attribution dict, seconds).  Clean predictions skip the GPU localisation and the physics entirely."""
        from subjective_engineering import generate_evidence_attribution
        torch = self.eng.torch
        t = time.perf_counter()
        ow, oh = pil.size
        cal_map = None
        if np.any(pred_6d > 0):
            with torch.no_grad():
                cal_map, _ = self.op_main.calibrated_map(pil, mode="midblock_hier8")
                if self.op_peg is not None and int(pred_6d[3]) > 0:
                    peg_map, _ = self.op_peg.calibrated_map(pil, mode="midblock_hier8")
                    cal_map[3] = peg_map[3]
        att = generate_evidence_attribution(pred_6d, probs_6x4, ow, oh, cal_map_6xGxG=cal_map, pil=pil, clean_ref=self.clean_ref)
        return att, time.perf_counter() - t


class N1Service:
    """Engine + decoder + attribution, one image at a time (shared by the template loop and serve_engine.py)."""

    def __init__(self, eng, set_name, rule="routed_gate_eu", attribution=True):
        self.eng, self.set, self.rule = eng, set_name, rule
        self.decode, self.sf_decoder = load_decoder(set_name, rule)
        self.attr = AttributionRuntime(eng, set_name) if attribution else None

    def predict(self, name, with_attribution=True):
        from qingyu.config_and_paths import DIMS, GRADE_INT_TO_STR
        from subjective_engineering import export_tat_jimu_payload
        t0 = time.perf_counter()
        P1, sec = self.eng.infer(name)
        t1 = time.perf_counter()
        pred, q = self.decode({m: np.asarray(v)[None] for m, v in P1.items()})
        pred, q = np.asarray(pred)[0], np.asarray(q)[0]
        t2 = time.perf_counter()
        rec = dict(image=name, label={DIMS[d]: GRADE_INT_TO_STR[int(pred[d])] for d in range(6)},
                   pred=pred, probs=q, member_probs=P1, member_sec=sec)
        att, ta = None, 0.0
        if with_attribution and self.attr is not None:
            att, ta = self.attr.attribute(self.eng.loader.load_pil(name).convert("RGB"), pred, q)
        t3 = time.perf_counter()
        rec.update(infer_s=t1 - t0, decode_s=t2 - t1, attr_s=ta, total_s=t3 - t0, attribution=att)
        if att is not None:
            rec["tat_jimu_payload"] = export_tat_jimu_payload(name, pred, q, att, latency_ms=(t3 - t0) * 1000.0)
        return rec


def _run_stage4_sidecars(a, eng, names, P, pp, primary_pred, primary_probs, per_img_secs, attr_records=None, tat_records=None):
    """
    Strictly opt-in Stage 4 Subjective Engineering sidecars (`--video-mode`, `--emit-attribution`, `--tat-out`).
    NEVER modifies `primary_pred` or the single-image objective submission JSON (`a.out`).
    """
    from qingyu.config_and_paths import DIMS, GRADE_INT_TO_STR
    from subjective_engineering import (
        EffectiveFrameORVideoAggregator,
        infer_video_group_id,
        render_attribution_contact_sheet,
    )

    if a.video_mode:
        vap = os.path.join(OUT, f"{a.set}_video_aggregator.pkl")
        if os.path.exists(vap):
            vid_agg = pickle.load(open(vap, "rb"))["video_aggregator"]
        else:
            vid_agg = EffectiveFrameORVideoAggregator()

        # Load single-frame decoder object for multi-frame calibration
        rule0 = a.decoder[0]
        if rule0 in ("routed_gate_eu", "routed_eu"):
            rp = os.path.join(OUT, f"{a.set}_decoder_{rule0}.pkl")
            sf_dec = pickle.load(open(rp, "rb"))["decoder"].eu
        else:
            rp = os.path.join(OUT, f"{a.set}_decoder.pkl")
            sf_dec = pickle.load(open(rp, "rb"))["decoder"]

        vid_groups = {}
        for idx_i, nm in enumerate(names):
            gid = infer_video_group_id(nm)
            vid_groups.setdefault(gid, []).append(idx_i)

        vid_records = []
        for gid, ix_list in vid_groups.items():
            res = vid_agg.aggregate_video(primary_probs[ix_list], sf_dec)
            vg = res["video_grade_6d"]
            vid_records.append(
                {
                    "video_id": gid,
                    "num_frames": len(ix_list),
                    "frames": [names[i] for i in ix_list],
                    "video_label": {DIMS[d]: GRADE_INT_TO_STR[int(vg[d])] for d in range(6)},
                    "trigger_frames": res["trigger_frames"],
                    "m_eff_p0": res["m_eff_p0"],
                }
            )
        v_out = os.path.splitext(a.out)[0] + "_video_summary.json"
        json.dump(vid_records, open(v_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[final_engine][Stage4] wrote §5-compliant video summary -> {v_out} ({len(vid_records)} video clips)")

    if a.emit_attribution or a.tat_out:
        # attr_records / tat_records were produced per image inside the N = 1 loop (AttributionRuntime; precomputed
        # clean footprint, no training data read at inference time)
        assert attr_records is not None and tat_records is not None
        if a.emit_attribution:
            os.makedirs(os.path.dirname(os.path.abspath(a.emit_attribution)), exist_ok=True)
            json.dump(attr_records, open(a.emit_attribution, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            sheet_p = os.path.splitext(a.emit_attribution)[0] + "_sheet.jpg"
            rendered = render_attribution_contact_sheet(attr_records, eng.loader, sheet_p)
            print(f"[final_engine][Stage4] wrote causal attribution -> {a.emit_attribution}" + (f" & {rendered}" if rendered else ""))

        if a.tat_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.tat_out)), exist_ok=True)
            json.dump(tat_records, open(a.tat_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"[final_engine][Stage4] wrote 极目/TAT inspection payload -> {a.tat_out}")


_TRAIN_PATTERNS = ("manifest_combined.csv", "/data/images/", "/data/images_supp/", "/data/qingyu_splits", "/data/r55_tagsplit",
                   "_oof.npz", "/data/manifest")
TRAIN_READS = []


def install_training_read_audit():
    """R55_AUDIT_TRAIN_READS=1: record every Python-level file open of training labels / training images."""
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            pth = os.fsdecode(args[0])
            if any(t in pth for t in _TRAIN_PATTERNS):
                TRAIN_READS.append(pth)
    sys.addaudithook(hook)


def main():
    if os.environ.get("R55_AUDIT_TRAIN_READS") == "1":
        install_training_read_audit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="primary", choices=list(SETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fit-decoder", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--v16-chunk", type=int, default=0, help="run V16 backbones on this many views at a time (lower activation peak)")
    ap.add_argument("--half", action="store_true", help="store Tier-1 and V16 backbone weights in bf16 (halves their memory)")
    ap.add_argument("--template", default="")
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--decoder", nargs="+", default=["d55"], choices=["d55", "eu", "vote", "routed_eu", "routed_gate_eu"],
                    help="decision rule(s): d55 = Decoder55; eu / vote = decoder_eu.py; routed_eu = RoutedEUDecoder; "
                         "routed_gate_eu = Stage-3 RoutedGateEUDecoder. Several rules -> one file per rule from ONE inference pass")
    ap.add_argument("--video-mode", action="store_true", help="Stage 4: emit §5-compliant multi-frame video summary sidecar JSON")
    ap.add_argument("--emit-attribution", default="", help="Stage 4: emit causal spatial defect attribution sidecar JSON + contact sheet")
    ap.add_argument("--tat-out", default="", help="Stage 4: emit Kuaishou 极目/TAT platform payload JSON")
    ap.add_argument("--build-clean-footprint", action="store_true",
                    help="OFFLINE: precompute the attribution clean footprint + physics percentiles from clean training frames")
    ap.add_argument("--bench-tag", default="", help="suffix for the --bench output files (e.g. _chunk1)")
    a = ap.parse_args()
    if a.fit_decoder:
        return fit_decoder(a.set)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    import common55 as c
    if a.build_clean_footprint:
        return build_clean_footprint(a.set, half=a.half)
    members = SETS[a.set]
    print(f"=== final_engine | set {a.set}: {members} | weights {np.round(weights(members), 3)} | "
          f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')} ===", flush=True)
    v16_chunk = a.v16_chunk if a.v16_chunk else (V16_CHUNK_BY_SET.get(a.set, 0) if a.half else 0)
    torch.cuda.init()
    nv = NvmlPeakSampler()
    eng = FinalEngine(members, img_dir=a.img_dir, half=a.half, v16_chunk=v16_chunk)
    peak_load = torch.cuda.max_memory_allocated() / 2 ** 30
    nv_peak_load = nv.peak_mib
    nv.reset()
    dp = os.path.join(OUT, f"{a.set}_decoder.pkl")
    dec = pickle.load(open(dp, "rb"))["decoder"] if os.path.exists(dp) else None

    if a.bench:
        names, gold, fold, p_oof, P_oof = oof_pool(members)
        rs = np.random.RandomState(0)
        idx = sorted(rs.choice(len(names), a.bench, replace=False))
        eng.infer(names[idx[0]])                                 # warm-up (cuDNN, lazy inits), not timed
        eng.peak = {}
        nv_after_warm = nv.read_mib()
        nv.reset()
        rows, secs, tot = [], [], []
        for i in idx:
            t = time.perf_counter()
            pr, sc = eng.infer(names[i])
            tot.append(time.perf_counter() - t)
            rows.append(pr)
            secs.append(sc)
        P = {m: np.stack([r[m] for r in rows]) for m in members}
        g = gold[idx]
        print(f"--- batch-1 latency over {len(idx)} labelled images (shared H200, not a 4090; other jobs may share this GPU)")
        for m in members:
            s = np.array([x[m] for x in secs])
            print(f"  {m:7s} mean {s.mean():5.2f}s  max {s.max():5.2f}s")
        tot = np.array(tot)
        print(f"  TOTAL   mean {tot.mean():5.2f}s  median {np.median(tot):5.2f}s  p95 {np.quantile(tot, .95):5.2f}s  max {tot.max():5.2f}s")
        peak_inf = max(eng.peak.values())
        nv_peak_inf = nv.peak_mib
        nv.stop()
        res_gb = torch.cuda.memory_allocated() / 2 ** 30
        reserved_gb = torch.cuda.memory_reserved() / 2 ** 30
        print(f"--- GPU memory & params: active params {eng.total_params_b:.3f}B (<=30B) | resident {res_gb:.2f} GB | "
              f"peak load {peak_load:.2f} GB | peak infer {peak_inf:.2f} GB (<=24GB) | load {eng.load_s:.0f}s | half={a.half} v16_chunk={v16_chunk}")
        print(f"--- nvidia-smi (NVML device used): peak during load {nv_peak_load:.0f} MiB | after warm-up {nv_after_warm:.0f} MiB | "
              f"peak during timed inference {nv_peak_inf:.0f} MiB ({nv_peak_inf / 1024:.2f} GiB) | torch reserved now {reserved_gb:.2f} GiB")
        print("    peak while each member runs (everything resident): " + "  ".join(f"{k} {v:.2f}" for k, v in eng.peak.items()))
        tagp = os.path.join(OUT, f"{a.set}_bench_probs{'_half' if a.half else ''}{a.bench_tag}.npz")
        np.savez_compressed(tagp, names=np.array([names[i] for i in idx]), **{f"p_{m}": P[m] for m in members})
        other = os.path.join(OUT, f"{a.set}_bench_probs{'' if a.half else '_half'}.npz")
        if os.path.exists(other) and not a.bench_tag:
            z = np.load(other, allow_pickle=True)
            if list(z["names"]) == [names[i] for i in idx]:
                print("--- same images, the other precision (fp32 weights vs bf16 weights):")
                for m in members:
                    d = np.abs(P[m] - z[f"p_{m}"])
                    print(f"  {m:7s} mean TV {0.5 * d.sum(-1).mean():.4f}  max|dp| {d.max():.3f}  images with a changed grade "
                          f"{(P[m].argmax(-1) != z[f'p_{m}'].argmax(-1)).any(1).mean() * 100:.1f}%")
        print("--- sanity (these images were in the full-fit training data, so agreement with the labels must be HIGHER than out-of-fold)")
        for m in members:
            a_in = (P[m].argmax(-1) == g).mean()
            a_oof = (P_oof[m][idx].argmax(-1) == g).mean()
            tv = 0.5 * np.abs(c.renorm(P[m]) - c.renorm(P_oof[m][idx])).sum(-1).mean()
            print(f"  {m:7s} grade accuracy full-fit {a_in:.3f} vs out-of-fold {a_oof:.3f} | mean TV to its OOF prediction {tv:.3f}")
        pp = pool(P, members)
        print(f"  pooled  grade accuracy full-fit {(pp.argmax(-1) == g).mean():.3f} vs out-of-fold {(p_oof[idx].argmax(-1) == g).mean():.3f}")
        os.makedirs(OUT, exist_ok=True)
        json.dump(dict(set=a.set, n=len(idx), total_params_b=float(eng.total_params_b), total_mean=float(tot.mean()),
                       total_p95=float(np.quantile(tot, .95)), total_max=float(tot.max()),
                       per_member={m: float(np.mean([x[m] for x in secs])) for m in members}, resident_gb=eng.mem,
                       resident_total_gb=round(float(res_gb), 2), peak_load_gb=float(peak_load), peak_infer_gb=float(peak_inf),
                       peak_by_member=eng.peak, half=bool(a.half), v16_chunk=int(v16_chunk),
                       load_s=float(eng.load_s), total_median=float(np.median(tot)),
                       nvidia_smi_peak_load_mib=float(nv_peak_load), nvidia_smi_after_warmup_mib=float(nv_after_warm),
                       nvidia_smi_peak_infer_mib=float(nv_peak_inf), torch_reserved_gb=round(float(reserved_gb), 3),
                       pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
                  open(os.path.join(OUT, f"{a.set}_bench{'_half' if a.half else ''}{a.bench_tag}.json"), "w"), indent=2)
        return

    assert a.template and a.out, "--template and --out are required for prediction"
    if "d55" in a.decoder:
        assert dec is not None, f"run --fit-decoder first ({dp})"
    want_attr = bool(a.emit_attribution or a.tat_out)
    svc = N1Service(eng, a.set, rule=a.decoder[0], attribution=want_attr)   # N = 1 decode (+ attribution) per image
    names = c.load_template_names(a.template)
    rows, per_img_secs, n1_preds, attr_records, tat_records, t0 = [], [], [], [], [], time.time()
    attr_secs = []
    for k, n in enumerate(names):
        r = svc.predict(n, with_attribution=want_attr)
        rows.append(r["member_probs"])
        per_img_secs.append(r["total_s"])
        n1_preds.append(r["pred"])
        if want_attr:
            attr_secs.append(r["attr_s"])
            attr_records.append({"image": n, "label": r["label"], "attribution": r["attribution"],
                                 "latency_s": {kk: round(float(r[kk]), 4) for kk in ("infer_s", "decode_s", "attr_s", "total_s")}})
            tat_records.append(r["tat_jimu_payload"])
        if (k + 1) % 25 == 0:
            print(f"  [{k + 1}/{len(names)}] {(time.time() - t0) / (k + 1):.2f}s per image", flush=True)
    P = {m: np.stack([r[m] for r in rows]) for m in members}
    pp = pool(P, members)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    primary_pred = None
    primary_probs = pp
    for idx_r, rule in enumerate(a.decoder):
        decode_r, _ = load_decoder(a.set, rule)
        pred, q_used = decode_r(P)                                # batch (N = len(template)) decode
        if idx_r == 0:
            primary_pred = pred
            primary_probs = q_used
        op = a.out if len(a.decoder) == 1 else os.path.splitext(a.out)[0] + f"_{rule}.json"
        json.dump(c.format_official_submission(names, pred)[0], open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[final_engine] wrote {op} (rule {rule}, {len(names)} images, {(time.time() - t0) / len(names):.2f}s per image)")
    n1 = np.stack(n1_preds)
    n_diff = int((n1 != np.asarray(primary_pred)).sum())
    print(f"[final_engine] N=1 vs N={len(names)} decode ({a.decoder[0]}): {n_diff} / {n1.size} cells differ"
          + ("  [batch-invariant]" if n_diff == 0 else "  [WARNING: decoder is batch-dependent]"), flush=True)
    tot = np.asarray(per_img_secs)
    print(f"[final_engine] per-image end-to-end{' (incl. attribution)' if want_attr else ''}: mean {tot.mean():.2f}s "
          f"p95 {np.quantile(tot, .95):.2f}s max {tot.max():.2f}s" +
          (f" | attribution mean {np.mean(attr_secs):.3f}s p95 {np.quantile(attr_secs, .95):.3f}s max {np.max(attr_secs):.3f}s "
           f"(defect images {int(np.any(n1 > 0, 1).sum())}/{len(names)})" if want_attr else ""), flush=True)

    np.savez_compressed(
        os.path.splitext(a.out)[0] + "_probs.npz",
        names=np.array(names),
        pooled=pp,
        routed=primary_probs,
        **{f"p_{m}": v for m, v in P.items()},
    )

    if (a.video_mode or a.emit_attribution or a.tat_out) and primary_pred is not None:
        _run_stage4_sidecars(a, eng, names, P, pp, primary_pred, primary_probs, per_img_secs, attr_records, tat_records)
    if os.environ.get("R55_AUDIT_TRAIN_READS") == "1":
        print(f"[final_engine][audit] training-data file opens during this inference run: {len(TRAIN_READS)}"
              + (f" e.g. {sorted(set(TRAIN_READS))[:5]}" if TRAIN_READS else ""), flush=True)


if __name__ == "__main__":
    main()
