#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/certify_clean.py — decide which external frames may be used as CLEAN negatives (CPU).

"Clean" here means the organisers' meaning: all six dimensions 无问题.  The only ground truth for that is the
513 labelled clean images, so the certificate is calibrated on them:

  score(x)  = min over the 8 fold-teacher ensembles of  prod_d P(dimension d = 无问题 | x)
  threshold = the smallest value at which, on the labelled out-of-fold predictions of the SAME teacher towers,
              at least --min-precision of the images above it are truly clean (default 0.99).
              On the s1 stack the top 300 by this score were 99.3% clean (2026-11-10).

Three guards against the certificate failing on a different domain:
  1. all 8 fold ensembles must agree (min, not mean);
  2. contact sheets are written for a human to scan (full frame + a 1:1 centre crop per image); names put in
     --reject-list are removed on the next run;
  3. certified images are only ever used with a small loss weight (train_testbed.py --lambda-clean-neg), and the
     gate requires S_clean and S_p0 not to fall.

Ranking tuples do NOT need clean seeds, so this certificate matters only for the clean-negative experiment.

Usage:  python code/r55/certify_clean.py [--min-precision 0.99] [--max-keep 1500] [--reject-list rejects.txt]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_CLEAN, AUX_IMG, AUX_TEACHER, R55_OUT, REPO, TEACHER_TOWERS, read_manifest  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "code/next55"))


def oof_threshold(min_precision: float):
    """Threshold on prod_d p_none from the teachers' own labelled OOF predictions."""
    import common55 as c
    names, gold, _, _ = c.load_split()
    acc = []
    for _, d in TEACHER_TOWERS:
        m = {}
        for f in range(8):
            z = np.load(os.path.join(REPO, d, f"fold_{f}_probs.npz"), allow_pickle=True)
            m.update({str(n): p for n, p in zip(z["va_names"], z["va_probs_6x4"])})
        acc.append(np.log(np.clip(np.stack([m[n] for n in names]).astype(np.float64), 1e-6, 1.0)))
    p = np.exp(np.mean(acc, 0))
    p = p / p.sum(-1, keepdims=True)
    score = np.prod(p[:, :, 0], axis=1)
    clean = gold.sum(1) == 0
    order = np.argsort(-score)
    prec = np.cumsum(clean[order]) / np.arange(1, len(order) + 1)
    ok = np.where(prec >= min_precision)[0]
    k = int(ok.max()) + 1 if len(ok) else 0
    assert k >= 50, "teacher cannot certify clean images at this precision"
    return float(score[order[k - 1]]), k, float(prec[k - 1]), int(clean.sum())


def contact_sheets(names, out_dir, per_sheet=24, thumb=360):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for s in range(0, len(names), per_sheet):
        chunk = names[s:s + per_sheet]
        cols = 4
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb * 2, rows * thumb), (20, 20, 20))
        for i, n in enumerate(chunk):
            im = Image.open(os.path.join(AUX_IMG, n)).convert("RGB")
            w, h = im.size
            t = im.copy()
            t.thumbnail((thumb, thumb))
            cx, cy = w // 2, h // 2
            crop = im.crop((cx - thumb // 2, cy - thumb // 2, cx + thumb // 2, cy + thumb // 2))  # 1:1 pixels
            x0, y0 = (i % cols) * thumb * 2, (i // cols) * thumb
            sheet.paste(t, (x0, y0))
            sheet.paste(crop, (x0 + thumb, y0))
        sheet.save(os.path.join(out_dir, f"sheet_{s // per_sheet:03d}.jpg"), quality=92)
        with open(os.path.join(out_dir, f"sheet_{s // per_sheet:03d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-precision", type=float, default=0.99)
    ap.add_argument("--max-keep", type=int, default=1500)
    ap.add_argument("--reject-list", default="", help="text file, one image name per line, from the human scan")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()

    thr, k, prec, n_clean = oof_threshold(a.min_precision)
    print(f"[certify] labelled OOF: threshold {thr:.4f} keeps {k} images at precision {prec:.3f} "
          f"(recall {k * prec / n_clean:.2f} of {n_clean} clean)")

    z = np.load(AUX_TEACHER, allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(z["names"])}
    ext = [r["name"] for r in read_manifest() if r["kind"] == "ext" and r["name"] in pos]
    assert ext, "no external rows with teacher predictions — run ingest_external_pool.py then prep_aux.py"
    p = z["probs_folds"][:, [pos[n] for n in ext]]                 # (8, M, 6, 4)
    score = np.prod(p[..., 0], axis=2).min(axis=0)                 # worst fold ensemble
    reject = set()
    if a.reject_list and os.path.exists(a.reject_list):
        reject = {x.strip() for x in open(a.reject_list, encoding="utf-8") if x.strip()}
    keep = [(n, float(s)) for n, s in zip(ext, score) if s >= thr and n not in reject]
    keep.sort(key=lambda t: -t[1])
    keep = keep[: a.max_keep]
    with open(AUX_CLEAN, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "score"])
        w.writerows(keep)
    print(f"[certify] external pool {len(ext)} -> certified clean {len(keep)} "
          f"({len(reject)} human rejects applied) -> {AUX_CLEAN}")
    if not a.no_sheets and keep:
        out = os.path.join(R55_OUT, "clean_sheets")
        contact_sheets([n for n, _ in keep], out)
        print(f"[certify] contact sheets for the human scan -> {out}  (list rejects in a text file, rerun with --reject-list)")


if __name__ == "__main__":
    main()
