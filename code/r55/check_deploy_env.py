#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/check_deploy_env.py — pre-flight check of an inference host for `final_engine.py` / `serve_engine.py`.

Checks (FAIL = inference will not run; WARN = runs, but differs from the verified environment):
  1. Python + every package pinned in requirements_infer.txt (missing -> FAIL, other version -> WARN)
  2. vendored bitsandbytes (third_party/pydeps): importable, version, CUDA native library for this torch's CUDA
  3. CUDA: device, SM capability, bf16, VRAM vs the 24 GB budget; whether this torch build carries SASS/PTX usable
     on the current GPU AND on sm_89 (RTX 4090): sm_89 runs sm_80/sm_86 SASS (same major, binary compatible)
  4. artifacts: decoder pkls (+ video aggregator), precomputed clean footprint npz (keys + tower match), every
     member checkpoint, local base-model weights
  5. inference does not depend on training data: data/manifest_combined.csv is NOT required (reported only)

Usage:  python3 code/r55/check_deploy_env.py [--set resident_v6_core] [--decoder routed_gate_eu] [--json out.json]
Exit code 0 when there is no FAIL.
"""
import argparse
import glob
import importlib
import importlib.metadata as md
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "r55"))

RES = []


def rec(status, item, detail=""):
    RES.append(dict(status=status, item=item, detail=str(detail)))
    print(f"  [{status:4s}] {item:52s} {detail}", flush=True)


def pkg_dist_name_to_import(dist):
    return {"opencv-python-headless": "cv2", "pillow": "PIL", "scikit-learn": "sklearn", "huggingface-hub": "huggingface_hub",
            "nvidia-ml-py": "pynvml", "pyyaml": "yaml"}.get(dist, dist.replace("-", "_"))


def check_python_deps():
    print("== 1. Python packages (requirements_infer.txt)")
    rec("OK" if sys.version_info[:2] == (3, 11) else "WARN", "python", sys.version.split()[0] + " (verified: 3.11.11)")
    req = os.path.join(REPO, "requirements_infer.txt")
    if not os.path.exists(req):
        return rec("FAIL", "requirements_infer.txt", "missing")
    optional = {"fastapi", "uvicorn", "anyio", "pydantic"}
    for ln in open(req, encoding="utf-8"):
        ln = ln.split("#")[0].strip()
        if not ln or "==" not in ln:
            continue
        name, ver = ln.split("==")
        try:
            have = md.version(name)
            importlib.import_module(pkg_dist_name_to_import(name))
            rec("OK" if have == ver else "WARN", f"{name}", f"{have}" + ("" if have == ver else f" (pinned {ver})"))
        except Exception as e:
            rec("WARN" if name in optional else "FAIL", f"{name}", f"not importable ({type(e).__name__}); pinned {ver}"
                + (" — serve_engine.py falls back to stdlib http.server" if name in optional else ""))


def check_bnb():
    print("== 2. bitsandbytes (vendored, third_party/pydeps)")
    import torch
    pyd = os.path.join(REPO, "third_party", "pydeps")
    if not os.path.isdir(os.path.join(pyd, "bitsandbytes")):
        return rec("FAIL", "third_party/pydeps/bitsandbytes", "missing (needed by p1_qy 4-bit LM)")
    sys.path.insert(0, pyd)
    try:
        import bitsandbytes as bnb
        rec("OK", "import bitsandbytes", f"{bnb.__version__} from {os.path.relpath(os.path.dirname(bnb.__file__), REPO)}")
    except Exception as e:
        return rec("FAIL", "import bitsandbytes", f"{type(e).__name__}: {e}")
    cu = (torch.version.cuda or "").replace(".", "")
    lib = os.path.join(pyd, "bitsandbytes", f"libbitsandbytes_cuda{cu}.so")
    rec("OK" if os.path.exists(lib) else "FAIL", f"native lib for torch CUDA {torch.version.cuda}", os.path.relpath(lib, REPO))
    if torch.cuda.is_available():
        try:
            lin = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
            y = lin(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
            rec("OK", "4-bit NF4 matmul on GPU", f"out {tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
        except Exception as e:
            rec("FAIL", "4-bit NF4 matmul on GPU", f"{type(e).__name__}: {e}")


def _arch_ok(arch_list, major, minor):
    sass = [a for a in arch_list if a.startswith("sm_")]
    ptx = [a for a in arch_list if a.startswith("compute_")]
    same_major = [a for a in sass if int(a[3:]) // 10 == major and int(a[3:]) % 10 <= minor]
    ptx_ok = [a for a in ptx if int(a[8:]) <= major * 10 + minor]
    return (f"sm_{major}{minor}" in sass) or bool(same_major) or bool(ptx_ok), same_major, ptx_ok


def check_cuda():
    print("== 3. CUDA / GPU")
    import torch
    rec("OK", "torch build", f"{torch.__version__} CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        return rec("FAIL", "torch.cuda.is_available()", "False")
    arch = torch.cuda.get_arch_list()
    rec("OK", "compiled arch list", " ".join(arch))
    p = torch.cuda.get_device_properties(0)
    ok, sm, pt = _arch_ok(arch, p.major, p.minor)
    rec("OK" if ok else "FAIL", f"current GPU {p.name}", f"sm_{p.major}{p.minor}, {p.total_memory / 2 ** 30:.1f} GiB, "
        f"usable kernels: {'native' if f'sm_{p.major}{p.minor}' in arch else (sm or pt)}")
    ok89, sm89, pt89 = _arch_ok(arch, 8, 9)
    rec("OK" if ok89 else "FAIL", "RTX 4090 target sm_89 (Ada)", "native sm_89" if "sm_89" in arch else
        f"binary-compatible SASS {sm89} (same major 8, minor <= 9)" + (f", PTX {pt89}" if pt89 else ""))
    rec("OK" if torch.cuda.is_bf16_supported() else "FAIL", "bf16 support", torch.cuda.is_bf16_supported())
    rec("OK" if p.total_memory / 2 ** 30 >= 23.5 else "WARN", "VRAM >= 24 GB budget (RTX 4090 = 23.99 GiB)",
        f"{p.total_memory / 2 ** 30:.1f} GiB on this device")
    rec("OK" if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") else "WARN",
        "PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(unset; final_engine.py sets expandable_segments:True)"))
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    rec("OK", "bf16 GEMM smoke test", f"{float((x @ x).float().abs().mean()):.3f}")


def check_artifacts(set_name, rule):
    print(f"== 4. artifacts for set '{set_name}' (decoder {rule})")
    import numpy as np
    import final_engine as fe
    members = fe.SETS[set_name]
    rec("OK", "members", members)
    dp = os.path.join(fe.OUT, f"{set_name}_decoder_{rule}.pkl" if rule != "d55" else f"{set_name}_decoder.pkl")
    rec("OK" if os.path.exists(dp) else "FAIL", f"decoder ({rule})", os.path.relpath(dp, REPO))
    vp = os.path.join(fe.OUT, f"{set_name}_video_aggregator.pkl")
    rec("OK" if os.path.exists(vp) else "WARN", "video aggregator (serve /infer_batch, --video-mode)", os.path.relpath(vp, REPO))
    fp = fe.clean_footprint_path(set_name)
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        pe_key, peg_extra = fe._attr_tower_keys([m for m in members if m in fe.TIER1])
        need = [f"fp__{t}__{m}" for t in [pe_key] + ([peg_extra] if peg_extra else []) for m in fe._FP_MODES]
        need += [k for k in ("phys_tile__overshoot", "phys_glob__lapvar")]
        miss = [k for k in need if k not in z.files]
        rec("FAIL" if miss else "OK", "clean footprint npz (attribution; no training data at runtime)",
            f"{os.path.relpath(fp, REPO)} {os.path.getsize(fp) / 1024:.1f} KiB towers={meta['towers']} "
            f"n_fp={len(meta['footprint_names'])} n_phys={len(meta['physics_names'])}" + (f" MISSING {miss}" if miss else ""))
    else:
        rec("FAIL", "clean footprint npz", f"{os.path.relpath(fp, REPO)} missing -> final_engine.py --build-clean-footprint")

    def ck(tag, rel, pack=None):
        p = fe._resolve_pack_path(rel, pack) if pack else os.path.join(REPO, rel)
        rec("OK" if os.path.exists(p) else "FAIL", f"ckpt {tag}", f"{os.path.relpath(p, REPO)}"
            + (f" ({os.path.getsize(p) / 2 ** 30:.2f} GiB)" if os.path.isfile(p) else ""))
    for m in members:
        if m in fe.TIER1_CKPT:
            ck(m, fe.TIER1_CKPT[m], f"checkpoints/{m}/fold_0_best.pt")
        elif m in fe.NTG_CKPT:
            ck(m, fe.NTG_CKPT[m], f"checkpoints/{m}/fold_0_best.pt")
        elif m in fe.P12_CKPT:
            ck(m, fe.P12_CKPT[m], f"checkpoints/{m}/fold_0_best.pt")
        elif m == "v16":
            c = glob.glob(os.path.join(REPO, "results/next55/fullfit/v16_full8/V16_Full_F0_*_swa.pt"))
            if c:
                rec("OK", "ckpt v16", os.path.relpath(c[0], REPO))
            else:
                ck("v16", "results/next55/fullfit/v16_full8/V16_Full_F0_AllGrade_Microscope_swa.pt", "checkpoints/v16/V16_Full_F0_AllGrade_Microscope_swa.pt")
        elif m == "p1_qy":
            ck("p1_qy", "results/r55/vlm_qy8/ckpts/p1_c233397_s2026_fullfit_swa.pt", "checkpoints/p1_qy/p1_c233397_s2026_fullfit_swa.pt")
        elif m == "p3_qy":
            ck("p3_qy", "results/r55/p3_qy_honest_fullfit/pillar3_fulldata_swa.pt")
    print("   base-model weights")
    sys.path.insert(0, os.path.join(REPO, "code"))
    from qingyu.config_and_paths import resolve_model_path
    from phoenix.config import resolve_local_or_hf_model
    bbs = sorted({fe.TIER1.get(m) or fe.NTG_SPEC.get(m) or (fe.P12_SPEC[m][0] if m in fe.P12_SPEC else None) for m in members} - {None})
    for bb in bbs:
        p = resolve_model_path(bb)
        rec("OK" if os.path.exists(p) else "WARN", f"backbone {bb}", p if os.path.exists(p) else f"{p} (not local; needs HF cache)")
    keys = (["pillar1_qwen3vl_8b"] if any(m in members for m in ("p1", "p1_qy")) else []) + \
        (["pillar3_depth_anything_s"] if any(m in members for m in ("p1", "p1_qy", "p3", "p3_qy")) else [])
    for k in keys:
        p = resolve_local_or_hf_model(k)
        if os.path.exists(p):
            rec("OK", f"base {k}", p)
        else:
            try:
                from huggingface_hub import snapshot_download
                lp = snapshot_download(p, local_files_only=True)
                rec("OK", f"base {k}", f"HF cache {lp}")
            except Exception as e:
                rec("FAIL", f"base {k}", f"{p} neither local nor in HF cache ({type(e).__name__})")


def check_training_independence():
    print("== 5. runtime independence from training data")
    man = os.path.join(REPO, "data/manifest_combined.csv")
    rec("OK", "data/manifest_combined.csv needed at inference?", "no (only by --fit-decoder / --build-clean-footprint / --bench; "
        f"present on this host: {os.path.exists(man)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="resident_v6_core")
    ap.add_argument("--decoder", default="routed_gate_eu")
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    print(f"=== check_deploy_env | repo {REPO} | set {a.set} | decoder {a.decoder} ===")
    for fn in (check_python_deps, check_bnb, check_cuda, lambda: check_artifacts(a.set, a.decoder), check_training_independence):
        try:
            fn()
        except Exception as e:
            rec("FAIL", getattr(fn, "__name__", "check"), f"{type(e).__name__}: {e}")
    n_fail = sum(r["status"] == "FAIL" for r in RES)
    n_warn = sum(r["status"] == "WARN" for r in RES)
    print(f"=== {len(RES)} checks: {n_fail} FAIL, {n_warn} WARN -> {'READY' if n_fail == 0 else 'NOT READY'} ===")
    if a.json:
        json.dump(dict(set=a.set, decoder=a.decoder, n_fail=n_fail, n_warn=n_warn, checks=RES), open(a.json, "w"), indent=2, ensure_ascii=False)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
