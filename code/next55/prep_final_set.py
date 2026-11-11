#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/prep_final_set.py — run ONCE when the hidden final set (654 imgs) is delivered. 8 GPUs, ~10-15 min.

  1. validates the template (unique names, every image exists + decodes) in --img-dir
  2. qingyu dual-engine physics for train + t150 + final  -> results/next55/dual_engine_meta_final.npz
     (8 ranks + merge; avoids lazy CUDA extraction inside forked DataLoader workers)
  3. V16 60-d native physics cache for the final images (single process -> no concurrent npz writes)

Usage:
  python code/next55/prep_final_set.py --template <final.json> --img-dir <dir_with_final_images> [--gpus 8]
Then export (printed at the end):  FINAL_TEMPLATE, FINAL_IMG_DIR, NEXT55_EXTRA_IMG_DIR, NEXT55_DUAL_NPZ
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DUAL_OUT = "results/next55/dual_engine_meta_final.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    a = ap.parse_args()
    os.chdir(REPO)
    tpl, img_dir = os.path.abspath(a.template), os.path.abspath(a.img_dir)

    from PIL import Image
    data = json.load(open(tpl, encoding="utf-8"))
    names = [str(e["image"] if isinstance(e, dict) else e) for e in data]
    assert len(names) == len(set(names)), "duplicate image names in template"
    bad = []
    for n in names:
        p = os.path.join(img_dir, n)
        try:
            with Image.open(p) as im:
                im.convert("RGB").size
        except Exception as e:  # noqa: BLE001
            bad.append((n, repr(e)[:80]))
    assert not bad, f"{len(bad)} unreadable/missing images, e.g. {bad[:3]}"
    print(f"[prep] template OK: {len(names)} images in {img_dir}")

    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "code"))
    os.makedirs(os.path.join(REPO, "results/next55"), exist_ok=True)
    procs = []
    for r in range(a.gpus):
        log = open(os.path.join(REPO, f"results/next55/dual_engine_final_shard{r}.log"), "w")
        procs.append(subprocess.Popen([sys.executable, "-u", "code/qingyu/dual_engine_preprocess.py", "--rank", str(r),
                                       "--world-size", str(a.gpus), "--extra-template", tpl, "--extra-test-dir", img_dir,
                                       "--out", DUAL_OUT], env=env, stdout=log, stderr=subprocess.STDOUT))
    rc = [p.wait() for p in procs]
    assert all(c == 0 for c in rc), f"dual-engine shards failed: {rc} (see results/next55/dual_engine_final_shard*.log)"
    subprocess.check_call([sys.executable, "-u", "code/qingyu/dual_engine_preprocess.py", "--merge-shards", str(a.gpus), "--out", DUAL_OUT], env=env)
    import numpy as np
    d = np.load(DUAL_OUT, allow_pickle=True)
    have = set(map(str, d["names"]))
    miss = [n for n in names if n not in have and os.path.basename(n) not in have]
    assert not miss, f"{len(miss)} final images missing from {DUAL_OUT}"
    print(f"[prep] dual-engine meta OK -> {DUAL_OUT} ({len(have)} images)")

    os.environ["NEXT55_EXTRA_IMG_DIR"] = img_dir
    sys.path.insert(0, os.path.join(REPO, "code/finetune"))
    from extract_native_phys_feats import extract_for_names  # noqa: E402  (reads NEXT55_EXTRA_IMG_DIR at import)
    feats = extract_for_names(names)
    zero = [n for n in names if not np.any(feats[n])]
    assert not zero, f"{len(zero)} final images got all-zero V16 physics (not found)"
    print("[prep] V16 physics cache OK")
    print("\nexport FINAL_TEMPLATE=%s\nexport FINAL_IMG_DIR=%s\nexport NEXT55_EXTRA_IMG_DIR=%s\nexport NEXT55_DUAL_NPZ=%s"
          % (tpl, img_dir, img_dir, os.path.join(REPO, DUAL_OUT)))


if __name__ == "__main__":
    main()
