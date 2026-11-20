#!/usr/bin/env python3
"""No-calibration-chart Imatest-style over-sharpening probe (cardless edge SFR / overshoot / ringing).

Reproduces the Imatest slanted-edge measurement chain on arbitrary natural images (no calibration target required):
  1. Auto-select strong edges (horizontal/vertical directions, top-K, requiring subpixel fit linearity)
  2. Slight tilt + 8x supersampling ESF binning
  3. LSF = d(ESF)/dx with Hann window -> normalized SFR (fft)
  4. Outputs: SFR50/SFR30, edge overshoot%, undershoot%, ringing index; plus a few cheap baseline physics metrics (Laplacian/Tenengrad/high-frequency energy).

sRGB domain measurement (no EOTF linearization; the report includes SFR results after gamma=2.2 linearization).

Usage:
  python3 probe_oversharp_imatest.py --extract [--limit N] [--workers 12]
  python3 probe_oversharp_imatest.py --eval
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_CSV = os.path.join(REPO, "data/qingyu_splits/qingyu_8fold.csv")
IMG_DIRS = (
    os.path.join(REPO, "data/images"),
    os.path.join(REPO, "data/images_supp"),
)
OUT_DIR = os.path.join(REPO, "results/probes/oversharp_imatest")

DIMS = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")
FOCUS_DIM = "过锐"

HALF_BAND = 8          # Number of rows before/after an edge for tilt fitting
SAMPLE_W = 24          # Pixels sampled on each side along the edge (must cover the undershoot zone)
OVERSAMPLE = 8         # Supersampled points per pixel
ESF_RANGE = SAMPLE_W * 2 * OVERSAMPLE
N_EDGES_PER_DIR = 5    # Number of strong edges selected per direction
MIN_EDGE_LEN = 12      # Edge tilt validity: |slope| <= 0.5 (enough rows within the window)
OS_SMOOTH_SIGMA = 4.0 * OVERSAMPLE   # Overshoot "ideal edge" smoothing sigma (4px "sharpening ripples averaged out")


def imread_gray(name, dst_dir=None):
    if dst_dir is not None:
        path = os.path.join(dst_dir, name)
    else:
        path = next((os.path.join(d, name) for d in IMG_DIRS if os.path.exists(os.path.join(d, name))), None)
        if path is None:
            return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def _find_peaks_1d(energy, k, min_dist):
    """Returns the indices of the top k local energy peaks (min_dist apart)."""
    e = np.asarray(energy, dtype=np.float64)
    order = np.argsort(e)[::-1]
    picked: list = []
    for i in order:
        if e[i] < np.percentile(e, 80):
            break
        if all(abs(i - j) >= min_dist for j in picked):
            picked.append(int(i))
            if len(picked) == k:
                break
    return sorted(picked)


def _subpixel_edge_pos(row, center, win=3):
    """Interpolated estimate of the gradient-peak position in this one-dimensional row (relative to index)."""
    g = np.abs(np.gradient(row))
    lo, hi = max(0, center - win), min(len(row), center + win + 1)
    seg, pos = g[lo:hi], np.arange(lo, hi, dtype=np.float64)
    seg = seg - seg.min()
    if seg.sum() < 1e-8:
        return np.nan
    return float(np.sum(seg * pos) / np.sum(seg))


def _measure_edge(gray, center, vertical=True, span=None):
    """Measures one edge. vertical=True: segment is near-vertical in gray coords.
    center: segment midpoint coordinate on the measured axis; span=(lo, hi) scan range of that axis."""
    g = gray.T if vertical else gray
    n_rows, n_cols = g.shape
    if center < SAMPLE_W or center >= n_cols - SAMPLE_W:
        return None
    if span is None:
        span = (HALF_BAND, n_rows - HALF_BAND)
    y0, y1 = max(HALF_BAND, int(span[0])), min(n_rows - HALF_BAND, int(span[1]))
    if y1 - y0 < MIN_EDGE_LEN:
        return None
    rows = np.arange(y0, y1)
    # Slope fitting: subpixel edge center per row -> linear fit slope
    centers = np.array([_subpixel_edge_pos(g[y, center - SAMPLE_W:center + SAMPLE_W], SAMPLE_W) for y in rows])
    ok = np.isfinite(centers)
    if ok.sum() < MIN_EDGE_LEN:
        return None
    y_used, c_used = rows[ok].astype(np.float64), centers[ok]
    k, b = np.polyfit(y_used, c_used, 1)
    if abs(k) > 0.5:                      # Edge too tilted
        return None
    # Supersampled ESF: for each row, re-align the sampling window to that row's sub-pixel edge center (symmetric coverage of both sides)
    nb = ESF_RANGE + 2 * OVERSAMPLE
    num = np.zeros(nb)
    den = np.zeros(nb)
    px = np.arange(2 * SAMPLE_W, dtype=np.float64)
    n_cols = g.shape[1]
    for y in y_used.astype(int):
        X = center - SAMPLE_W + (b + k * y)          # Global sub-pixel edge position for this row
        col0 = int(round(X)) - SAMPLE_W
        if col0 < 0 or col0 + 2 * SAMPLE_W > n_cols:
            continue
        row = g[y, col0:col0 + 2 * SAMPLE_W]
        xc_in = X - col0                              # Edge position inside the window (≈24)
        u = (px - xc_in) * OVERSAMPLE + SAMPLE_W * OVERSAMPLE
        i0 = np.floor(u).astype(np.int64)
        fr = u - i0
        valid = (i0 >= 0) & (i0 + 1 < nb)
        np.add.at(num, i0[valid], row[valid] * (1.0 - fr[valid]))
        np.add.at(num, i0[valid] + 1, row[valid] * fr[valid])
        np.add.at(den, i0[valid], 1.0)
        np.add.at(den, i0[valid] + 1, 1.0)
    esf_bins = num / np.maximum(den, 1.0)
    if den[SAMPLE_W * OVERSAMPLE:-SAMPLE_W * OVERSAMPLE].mean() < 0.6 * OVERSAMPLE:
        return None
    idx = np.where(den > 0)[0]
    esf = esf_bins[idx[0]:idx[-1] + 1]
    if len(esf) < ESF_RANGE * 0.9:
        return None
    return _esf_metrics(esf)


def _esf_metrics(esf):
    """ESF -> SFR and overshoot family of metrics. Plateau-benchmark overshoot/undershoot (Imatest edge-sharpness style)."""
    # Light smoothing (suppress binning noise, preserving sharpening ripples) + gradient
    k = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    esf_s = np.convolve(np.pad(esf, 1, mode="edge"), k, mode="valid")
    lsf = np.gradient(esf_s)
    w = np.hanning(len(lsf))
    lsf_w = lsf * w
    spec = np.abs(np.fft.rfft(lsf_w))
    spec = spec / (spec[0] + 1e-12)
    n = len(spec)
    # Sampling interval dx = 1/OVERSAMPLE px -> k-th spectrum frequency = k*OVERSAMPLE/len(lsf) cycles/px; digital Nyquist = OVERSAMPLE/2
    fstep = OVERSAMPLE / len(lsf_w)

    def first_below(level):
        idx = np.argmax(spec < level) if np.any(spec < level) else -1
        return idx * fstep if idx > 0 else 0.5

    f50, f30 = first_below(0.5), first_below(0.3)
    # Plateau-benchmark overshoot/undershoot: front/back plateau = mean of 15% bins; overshoot = ESF extreme exceeding the plateau / contrast
    wq = max(4, int(0.15 * len(esf_s)))
    front, back = float(esf_s[:wq].mean()), float(esf_s[-wq:].mean())
    rg = back - front
    if abs(rg) < 10.0:                    # Low-contrast edge (<4% brightness swing), discarded
        return None
    sgn = float(np.sign(rg))
    # Monotonic step-edge gate: reject pulse-shaped edges (text strokes/thin lines) and texture pseudo-edges
    mono = float(np.corrcoef(np.arange(len(esf_s), dtype=np.float64), esf_s)[0, 1])
    if mono * sgn < 0.60:
        return None
    peak_val = float(esf_s.max()) if sgn > 0 else float(esf_s.min())
    valley_val = float(esf_s.min()) if sgn > 0 else float(esf_s.max())
    overshoot = (peak_val - back) * sgn / abs(rg) * 100.0
    undershoot = (front - valley_val) * sgn / abs(rg) * 100.0
    # Ringing index: main-lobe peak vs first side-lobe peak after the main lobe (|LSF|)
    main = int(np.argmax(np.abs(lsf)))
    side0, side1 = main + 2, min(len(lsf), main + 2 + OVERSAMPLE * 4)
    side_peak = np.abs(lsf[side0:side1]).max() if side1 - side0 > 4 else 0.0
    ringing = float(side_peak / (np.abs(lsf[main]) + 1e-12))
    return dict(
        f50=float(np.clip(f50, 0, 0.5)), f30=float(np.clip(f30, 0, 0.5)),
        overshoot=float(np.clip(overshoot, -50, 150)), undershoot=float(np.clip(undershoot, -50, 150)), ringing=float(np.clip(ringing, 0, 3)),
        sfr_near_nyq=float(spec[min(n - 1, int(0.42 / fstep))]),
    )


def _hough_edge_candidates(gray, max_per_dir=12):
    """Canny+HoughLinesP: find long straight edge segments -> [(center, vertical, y0, y1, length)].
    vertical=True means a near-vertical segment in gray coords (is measured on the transposed view)."""
    out = []
    edges = cv2.Canny(np.clip(gray, 0, 255).astype(np.uint8), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=64,
                            minLineLength=max(48, min(gray.shape) // 10), maxLineGap=4)
    if lines is None:
        return out
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = abs(int(x2 - x1)), abs(int(y2 - y1))
        ln = float(np.hypot(dx, dy))
        if max(dx, dy) == 0:
            continue
        if dy / max(dx, dy) > 0.92:      # Nearly-vertical segment -> scan within its y-range
            out.append((int(round((x1 + x2) / 2)), False,
                        min(y1, y2) - HALF_BAND, max(y1, y2) + HALF_BAND, ln))
        elif dx / max(dx, dy) > 0.92:    # Nearly-horizontal segment -> transposed view
            out.append((int(round((y1 + y2) / 2)), True,
                        min(x1, x2) - HALF_BAND, max(x1, x2) + HALF_BAND, ln))
    out.sort(key=lambda t: -t[4])
    seen_v: list = []
    seen_h: list = []
    res = []
    cnt_v = cnt_h = 0
    for c, ver, s0, s1, ln in out:
        seen = seen_v if ver else seen_h
        if any(abs(c - s) < 12 for s in seen):
            continue
        if ver and cnt_v >= max_per_dir:
            continue
        if not ver and cnt_h >= max_per_dir:
            continue
        seen.append(c)
        res.append((c, ver, s0, s1, ln))
        if ver:
            cnt_v += 1
        else:
            cnt_h += 1
    return res


def image_feats(name):
    gray = imread_gray(name)
    if gray is None:
        return None
    out: dict = {}
    results = []
    for vertical in (True, False):
        cands = [(c, s0, s1) for c, ver, s0, s1, _ln in _hough_edge_candidates(gray) if ver == vertical]
        got = 0
        for c, s0, s1 in cands:
            if got >= N_EDGES_PER_DIR:
                break
            m = _measure_edge(gray, c, vertical=vertical, span=(s0, s1))
            if m is None:
                continue
            results.append((m, vertical))
            got += 1
    if not results:
        return None
    vals = {}
    for key in ("f50", "f30", "overshoot", "undershoot", "ringing", "sfr_near_nyq"):
        v = np.array([r[0][key] for r in results])
        vals[f"{key}_med"] = float(np.median(v))
        vals[f"{key}_max"] = float(v.max()) if key in ("overshoot", "undershoot", "ringing") else float(np.median(v))
    vals["n_edges"] = float(len(results))
    # Cheap baseline physics
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    vals["lap_var"] = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
    vals["tenengrad"] = float(np.mean(gx * gx + gy * gy))
    hf = np.sqrt(gx * gx + gy * gy).ravel()
    vals["hf_p98"] = float(np.percentile(hf, 98))
    return vals


def load_rows():
    return list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))


def _worker(args):
    name = args
    try:
        v = image_feats(name)
        return name, v
    except Exception:
        return name, None


def cmd_extract(limit, workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    names = [r["name"] for r in rows]
    if limit:
        names = names[:limit]
    t0 = time.time()
    done = 0
    with mp.Pool(workers) as pool, open(os.path.join(OUT_DIR, "feats.csv"), "w", newline="") as f:
        w = None
        for name, v in pool.imap_unordered(_worker, names, chunksize=16):
            done += 1
            if v is None:
                continue
            if w is None:
                w = csv.DictWriter(f, fieldnames=["name"] + sorted(v.keys()))
                w.writeheader()
            w.writerow({"name": name, **v})
            if done % 500 == 0:
                print(f"  {done}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"extract done: {done} imgs in {time.time() - t0:.0f}s -> {OUT_DIR}/feats.csv")


def cmd_eval():
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    df = pd.read_csv(os.path.join(OUT_DIR, "feats.csv")).set_index("name")
    meta = pd.read_csv(SPLITS_CSV).set_index("name").loc[df.index]
    y_reg = meta[FOCUS_DIM].astype(int).to_numpy()
    y_pos = (y_reg >= 1).astype(int)
    y_p0 = (y_reg == 3).astype(int)
    groups = meta["group_id"].to_numpy()

    feat_cols = [c for c in df.columns if c != "n_edges"]
    imatest_cols = [c for c in feat_cols if not c.endswith(("_med", "_max")) or
                    c.split("_")[0] in ("f50", "f30", "overshoot", "undershoot", "ringing", "sfr")]
    base_cols = ["lap_var", "tenengrad", "hf_p98"]

    report = {"n_images": int(len(df)), "dim": FOCUS_DIM,
              "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y_reg, return_counts=True))}}
    rows = []
    for c in feat_cols:
        v = df[c].values
        if np.nanstd(v) < 1e-9:
            continue
        rows.append({
            "feature": c,
            "spearman_vs_grade": round(float(spearmanr(v, y_reg).statistic), 4),
            "auc_pos": round(float(roc_auc_score(y_pos, np.nan_to_num(v, nan=0.0))), 4),
            "auc_p0": round(float(roc_auc_score(y_p0, np.nan_to_num(v, nan=0.0))), 4),
        })
    report["single_feature"] = sorted(rows, key=lambda r: -abs(r["spearman_vs_grade"]))

    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    X_base = df[base_cols].fillna(0).values
    X_full = df[imatest_cols + base_cols].fillna(0).values
    gkf = GroupKFold(n_splits=5)
    res = {}
    for tag, X in (("base_cheap_phys", X_base), ("base_plus_imatest", X_full)):
        ap_pos, ap_p0 = [], []
        for tr, te in gkf.split(X, y_pos, groups):
            for yb, acc in ((y_pos, ap_pos), (y_p0, ap_p0)):
                lr = LogisticRegression(max_iter=500, C=1.0).fit(X[tr], yb[tr])
                acc.append(float(average_precision_score(yb[te], lr.predict_proba(X[te])[:, 1])))
        res[tag] = {"AP_pos_median": round(float(np.median(ap_pos)), 4), "AP_p0_median": round(float(np.median(ap_p0)), 4)}
    report["cv_AP_group5fold"] = res

    out = os.path.join(OUT_DIR, "eval_report.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("n_images", "label_counts", "cv_AP_group5fold")}, ensure_ascii=False, indent=1))
    print("top single-feature (by |spearman|):")
    for r in report["single_feature"][:12]:
        print(f"  {r['feature']:<18} rho={r['spearman_vs_grade']:+.3f}  AUC_pos={r['auc_pos']:.3f}  AUC_p0={r['auc_p0']:.3f}")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", dest="eval_", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.extract:
        cmd_extract(a.limit, a.workers)
    elif a.eval_:
        cmd_eval()
    else:
        print(__doc__)
