#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_glitch.py — E3: 花屏 produced the way it is produced in reality (CPU, ffmpeg + libx264).

docs/REASSESSMENT_AND_PLAN_2026-11-12.md §4 C.  花屏 in the labelled set is decoder corruption (smeared macroblocks,
wrong references).  209 of its 211 positives come from one source family, and when that family is held out the
detector collapses (AUC 0.995 -> 0.885).  Decoder corruption can be reproduced exactly: take a real short video,
re-encode a short window as one long GOP, damage or drop a few P-slices in the bitstream, decode with error
concealment, keep the frame at the position of an existing clean KVQ frame.

For KVQ video <v> and frame slot k (the same frame index as data/r55_aux/images/r55kvq_<v>_<k>.png):
  r55glitch_<v>_<k>.png          the corrupted frame, native resolution
  r55glitchref_<v>_<k>.png       the same re-encoded window decoded WITHOUT damage (matched negative; added after the
                                 first run flagged blurry / screen-recording frames as 花屏: 7 false alarms on 872 images)
  manifest_glitch.csv            kind=glitch, seed_fold=-1, op=<mode>:<details>
  glitch_frames.csv              name, clean (the KVQ frame it was made from), mode, corrupted area share
Four kinds of damage: flipped bits in the key frame, flipped bits in inter frames, dropped inter frames, and a
key frame spliced in from another moment of the same video (inter frames then predict from the wrong picture).
A frame is kept only if the damage is clearly visible against the clean decode of the same window (>= 10 % of the
16x16 blocks differ, or >= 5 % differ strongly) and the picture is not blank.  No labels of ours and no test image are involved.

Usage:  python code/r55/build_glitch.py --n-videos 700 --workers 48
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_DIR, AUX_IMG, append_manifest  # noqa: E402

MAN = os.path.join(AUX_DIR, "manifest_glitch.csv")
SIDE = os.path.join(AUX_DIR, "glitch_frames.csv")
VDIR = os.path.join(AUX_DIR, "raw/kvq/videos/train")


def nal_units(buf):
    """[(start_of_payload, end, nal_type)] of an Annex-B H.264 stream."""
    out, i, n = [], 0, len(buf)
    starts = []
    while True:
        j = buf.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for a, b in zip(starts, starts[1:] + [n]):
        end = b - 3 if b != n else n
        while end > a and buf[end - 1] == 0:                      # trailing zero of a 4-byte start code
            end -= 1
        out.append((a, end, buf[a] & 0x1F))
    return out


def job(arg):
    try:
        return _job(arg)
    except Exception:                                            # one odd video must not stop the pool
        return None


def _job(arg):
    vid, k, seed = arg
    rng = random.Random(seed)
    src = os.path.join(VDIR, vid + ".mp4")
    clean_path = os.path.join(AUX_IMG, f"r55kvq_{vid}_{k}.png")
    if not (os.path.exists(src) and os.path.exists(clean_path)):
        return None
    cap = cv2.VideoCapture(src)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    idx = int(n * (k + 1) / 3)
    win = rng.randint(18, 40)
    if idx - win < 1:
        return None
    tmp = tempfile.mkdtemp(prefix="glitch_", dir=os.environ.get("GLITCH_TMP", "/dev/shm"))
    try:
        seg = os.path.join(tmp, "seg.h264")
        sel = f"select='between(n\\,{idx - win}\\,{idx})',setpts=N/FRAME_RATE/TB"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", sel, "-an", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", str(rng.choice([23, 26, 29])), "-bf", "0", "-g", "9999",
                            "-x264-params", "scenecut=0:slices=1:ref=1", "-f", "h264", seg], capture_output=True)
        if r.returncode != 0 or not os.path.exists(seg):
            return None
        buf = bytearray(open(seg, "rb").read())
        nu = nal_units(bytes(buf))
        sl = [(x, y) for x, y, t in nu if t == 1]                              # P slices, one per frame
        idr = [(x, y) for x, y, t in nu if t == 5]
        if len(sl) < 8 or len(idr) != 1:
            return None

        def flip(x, y, lo_frac, n_flip):
            lo = x + max(8, int((y - x) * lo_frac))
            for _ in range(n_flip):
                if lo < y - 1:
                    q = rng.randint(lo, y - 1)
                    buf[q] ^= 1 << rng.randint(0, 7)

        mode = rng.choice(["idr_flip", "idr_flip", "splice", "splice", "p_flip", "p_drop"])
        c0, m = 0, 0
        if mode == "idr_flip":                                   # damaged key frame: the rest of the slice is concealed, then propagates
            flip(idr[0][0], idr[0][1], rng.uniform(0.10, 0.85), rng.randint(1, 4))
        elif mode == "p_flip":                                   # damaged inter frames
            c0 = rng.randint(1, max(1, len(sl) - 6))
            m = rng.randint(2, 8)
            for x, y in sl[c0:c0 + m]:
                flip(x, y, 0.10, rng.randint(1, 6))
        elif mode == "p_drop":                                   # lost inter frames: later frames predict from the wrong picture
            c0 = rng.randint(1, max(1, len(sl) - 10))
            m = rng.randint(3, 8)
            for x, y in reversed(sl[c0:c0 + m]):
                st = x - 4 if buf[x - 4:x] == b"\x00\x00\x00\x01" else x - 3
                del buf[st:y]
        else:                                                    # splice: key frame of another moment, inter frames of this one (wrong reference)
            j0 = rng.choice([i for i in range(win + 1, n - 1, 7) if abs(i - idx) > 2 * win] or [0])
            if j0 == 0:
                return None
            seg2 = os.path.join(tmp, "seg2.h264")
            sel2 = f"select='between(n\\,{j0 - win}\\,{j0})',setpts=N/FRAME_RATE/TB"
            r2 = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", sel2, "-an", "-c:v", "libx264", "-preset", "veryfast",
                                 "-crf", "26", "-bf", "0", "-g", "9999", "-x264-params", "scenecut=0:slices=1:ref=1", "-f", "h264", seg2],
                                capture_output=True)
            if r2.returncode != 0:
                return None
            b2 = open(seg2, "rb").read()
            first_p2 = min(x for x, y, t in nal_units(b2) if t == 1)
            first_p = sl[0][0]
            st2 = first_p2 - 4 if b2[first_p2 - 4:first_p2] == b"\x00\x00\x00\x01" else first_p2 - 3
            st1 = first_p - 4 if buf[first_p - 4:first_p] == b"\x00\x00\x00\x01" else first_p - 3
            keep_from = rng.randint(0, max(0, len(sl) - 12))     # optionally skip a few inter frames so less of the picture is refreshed
            cut = sl[keep_from][0]
            cut = cut - 4 if buf[cut - 4:cut] == b"\x00\x00\x00\x01" else cut - 3
            buf = bytearray(b2[:st2]) + buf[max(st1, cut):]
            c0 = keep_from
        bad = os.path.join(tmp, "bad.h264")
        open(bad, "wb").write(bytes(buf))
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-err_detect", "ignore_err", "-ec", "guess_mvs+deblock", "-i", bad,
                        "-vsync", "0", os.path.join(tmp, "f_%03d.png")], capture_output=True)
        fr = sorted(x for x in os.listdir(tmp) if x.startswith("f_"))
        if len(fr) < 4:
            return None
        g = cv2.imread(os.path.join(tmp, fr[-1]), cv2.IMREAD_COLOR)
        c = cv2.imdecode(np.fromfile(clean_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if g is None or c is None or g.shape != c.shape:
            return None
        # compare with the uncorrupted decode of the same segment, so that re-encoding loss is not counted as corruption
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", seg, "-vsync", "0", os.path.join(tmp, "c_%03d.png")], capture_output=True)
        cf = sorted(x for x in os.listdir(tmp) if x.startswith("c_"))
        ref = cv2.imread(os.path.join(tmp, cf[-1]), cv2.IMREAD_COLOR) if cf else c
        d = np.abs(g.astype(np.int16) - ref.astype(np.int16)).mean(2)
        h, w = d.shape
        blk = d[:h - h % 16, :w - w % 16].reshape(h // 16, 16, w // 16, 16).mean((1, 3))
        share = float((blk > 12).mean())
        strong = float((blk > 30).mean())
        if not ((share >= 0.10 or strong >= 0.05) and share <= 0.97) or float(g.std()) < 12:
            return None                                          # must be clearly visible, must not be a blank picture
        name = f"r55glitch_{vid}_{k}.png"
        cv2.imencode(".png", g, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(AUX_IMG, name))
        # matched negative: the SAME re-encoded window decoded without damage.  Positive and negative then differ only
        # by the corruption, so content style and re-encoding softness cannot be used as shortcuts for 花屏.
        cv2.imencode(".png", ref, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(AUX_IMG, f"r55glitchref_{vid}_{k}.png"))
        return dict(name=name, clean=f"r55kvq_{vid}_{k}.png", mode=f"{mode}:c{c0}m{m}w{win}", share=f"{share:.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=700)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    vids = sorted(os.path.splitext(f)[0] for f in os.listdir(VDIR) if f.endswith(".mp4"))
    rs = random.Random(a.seed)
    rs.shuffle(vids)
    jobs = [(v, k, a.seed * 100003 + i * 2 + k) for i, v in enumerate(vids[:a.n_videos]) for k in (0, 1)]
    out = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            if r:
                out.append(r)
            if (i + 1) % 200 == 0:
                print(f"[glitch] {i + 1}/{len(jobs)} tried, {len(out)} kept", flush=True)
    out.sort(key=lambda r: r["name"])
    rows = [dict(name=r["name"], kind="glitch", seed=r["clean"], seed_fold=-1, tuple_id="", dim=3, level=3, op=r["mode"]) for r in out]
    rows += [dict(name=r["name"].replace("r55glitch_", "r55glitchref_"), kind="glitch", seed=r["clean"], seed_fold=-1, tuple_id="",
                  dim=3, level=0, op="clean_decode") for r in out]
    n = append_manifest(rows, MAN)
    with open(SIDE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "clean", "mode", "share"])
        w.writeheader()
        w.writerows(out)
    sh = np.array([float(r["share"]) for r in out]) if out else np.zeros(1)
    print(f"[glitch] kept {n} of {len(jobs)} attempts | corrupted area share quartiles {np.round(np.quantile(sh, [.1, .25, .5, .75, .9]), 2)}")


if __name__ == "__main__":
    main()
