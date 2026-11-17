#!/usr/bin/env python3
# Point-loss breakdown of the FALLBACK ensemble (nested Decoder55 OOF, Doc2_Strict) on the 3,573 labelled images.
# CPU only; reads only the OOF dump written by dump_fallback.py + labels/manifest.  No test set, no legacy JSON.
# Usage: python3 error_analysis.py fallback_pool.npz
import csv
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import Decoder55  # noqa: E402

K = "doc2_strict_total"
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FULL = [0, 1, 2, 4]
GN = {0: "none", 1: "P2", 2: "P1", 3: "P0"}
z = np.load(sys.argv[1], allow_pickle=True)
names = [str(x) for x in z["names"]]
gold, fold, grp, P, pred = z["gold"], z["fold"], z["grp"], z["P"], z["nested"].copy()
pin = z["insample"]
decs = pickle.load(open(sys.argv[1].replace(".npz", "_decoders.pkl"), "rb"))
N = len(names)
sp = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
orig = np.array([sp[n] == "orig" for n in names])
clean = (gold == 0).all(1)
OUT = {}


def S(p, g=gold, m=None):
    if m is not None:
        p, g = p[m], g[m]
    return c.score(p, g)


def comp(ev):
    return dict(total=round(ev[K], 3), pos=round(ev["doc2_S_pos"], 3), p0=round(ev["doc2_S_p0"], 3), clean=round(ev["doc2_Clean2"], 3))


def vid(n):
    m = re.match(r"^(?:\d{3}_)?([0-9a-f]{8,}|\d{9,})", n)
    return m.group(1) if m else n


TAG_RE = re.compile(r"_(v\d+[A-Za-z0-9]*?)(?:\.mp4|_|\.)")


def tag(n):
    m = TAG_RE.search(n)
    if m:
        return m.group(1)
    if re.search(r"[一-鿿]", n):
        return "cn_defectword"
    if re.search(r"\d+\.\d+s\.png$", n):
        return "timestamp"
    if re.match(r"^[a-z]+_\d+\.jpe?g$", n):
        return "stock_jpg"
    return "other_untagged"


vids = np.array([vid(n) for n in names])
tags = np.array([tag(n) for n in names])
ug, ginv = np.unique(grp, return_inverse=True)
rs = np.random.RandomState(0)
NB = 1000


def boot_idx(mask, n=NB):
    mem = [np.where((ginv == i) & mask)[0] for i in range(len(ug))]
    mem = [m for m in mem if len(m)]
    return [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(n)]


BALL = boot_idx(np.ones(N, bool))


def ci(fn, boots=BALL):
    v = np.array([fn(b) for b in boots])
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


ev = S(pred)
print("=" * 100)
print("REFERENCE nested:", c.score_line(ev))
print("first batch rows:", comp(S(pred, m=orig)), " supplement rows:", comp(S(pred, m=~orig)))
OUT["reference"] = dict(all=comp(ev), orig=comp(S(pred, m=orig)), supp=comp(S(pred, m=~orig)))

# ---------------------------------------------------------------- 1. per-component / per-label losses
print("\n## 1. Points lost per component (of 65)")
print(f"  S_pos lost {25 - ev['doc2_S_pos']:.2f} / 25 | S_p0 lost {20 - ev['doc2_S_p0']:.2f} / 20 | clean lost {20 - ev['doc2_Clean2']:.2f} / 20 | total lost {65 - ev[K]:.2f}")
print("  S_p0 sub-terms: R=%.3f F1=%.3f P=%.3f -> lost 10*(1-R)=%.2f 6*(1-F1)=%.2f 4*(1-P)=%.2f" % (
    ev["p0_recall"], ev["p0_f1"], ev["p0_precision"], 10 * (1 - ev["p0_recall"]), 6 * (1 - ev["p0_f1"]), 4 * (1 - ev["p0_precision"])))
rows = []
for x in ev["per_14"]:
    d, g = x["dim_idx"], x["grade"]
    prev = int((gold[:, d] == g).sum())
    lpos = 25 / 14 * (1 - x["F1"])
    lp0 = 20 / 6 * (0.5 * (1 - x["R"]) + 0.3 * (1 - x["F1"]) + 0.2 * (1 - x["P"])) if g == 3 else 0.0
    rows.append(dict(label=x["channel"], n_gold=prev, n_pred=x["tp"] + x["fp"], tp=x["tp"], fp=x["fp"], fn=x["fn"], P=round(x["P"], 3),
                     R=round(x["R"], 3), F1=round(x["F1"], 3), loss_pos=round(lpos, 3), loss_p0=round(lp0, 3), loss=round(lpos + lp0, 3)))
# CI on per-label loss
for r in rows:
    pass
print(f"  {'label':12s} {'n_gold':>6s} {'n_pred':>6s} {'tp':>5s} {'fp':>5s} {'fn':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'L_pos':>6s} {'L_p0':>6s} {'L_tot':>6s}")
for r in sorted(rows, key=lambda r: -r["loss"]):
    print(f"  {r['label']:12s} {r['n_gold']:6d} {r['n_pred']:6d} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d} {r['P']:6.3f} {r['R']:6.3f} {r['F1']:6.3f} {r['loss_pos']:6.2f} {r['loss_p0']:6.2f} {r['loss']:6.2f}")
OUT["per_label"] = rows
# CI on the loss of each label (cluster bootstrap)
lab_ci = {}
for i, r in enumerate(rows):
    def f(b, i=i):
        e = c.score(pred[b], gold[b])
        x = e["per_14"][i]
        l = 25 / 14 * (1 - x["F1"])
        if x["grade"] == 3:
            l += 20 / 6 * (0.5 * (1 - x["R"]) + 0.3 * (1 - x["F1"]) + 0.2 * (1 - x["P"]))
        return l
    lab_ci[r["label"]] = ci(f, BALL[:300])
print("  95% cluster-bootstrap CI of each label's loss:", lab_ci)
OUT["per_label_loss_ci"] = lab_ci

# per-dimension loss
print("\n  per-dimension total loss (pos + p0 terms):")
dl = defaultdict(float)
for r in rows:
    dl[r["label"].rsplit("_", 1)[0]] += r["loss"]
for k, v in sorted(dl.items(), key=lambda t: -t[1]):
    print(f"    {k:6s} {v:6.2f}")
OUT["per_dim_loss"] = {k: round(v, 3) for k, v in dl.items()}

# ---------------------------------------------------------------- 2. oracle substitutions
print("\n## 2. Oracle substitutions (gain in Doc2_Strict over nested %.3f)" % ev[K])
base = ev[K]
orc = {}
# (i) prediction-level: one full dimension perfect
for d in range(6):
    q = pred.copy()
    q[:, d] = gold[:, d]
    e = S(q)
    orc[f"dim_perfect:{DIMS[d]}"] = dict(gain=round(e[K] - base, 3), **comp(e))
# (ii) prediction-level: one label (d,g) perfect -> rows with gold==g get g, rows falsely predicted g get gold
for d in range(6):
    for g in ((1, 2, 3) if d in FULL else (3,)):
        q = pred.copy()
        m = (gold[:, d] == g) | (q[:, d] == g)
        q[m, d] = gold[m, d]
        e = S(q)
        orc[f"label_perfect:{DIMS[d]}_{GN[g]}"] = dict(gain=round(e[K] - base, 3), **comp(e))
# (iii) clean images perfect
q = pred.copy()
q[clean] = 0
e = S(q)
orc["clean_rows_perfect"] = dict(gain=round(e[K] - base, 3), **comp(e))
# (iv) metric-level: set that label's F1 (and P/R) to 1 holding everything else fixed (pure accounting)
for r in rows:
    orc[f"metric_F1=1:{r['label']}"] = dict(gain=r["loss"])
for k, v in sorted(orc.items(), key=lambda t: -t[1]["gain"]):
    if not k.startswith("metric"):
        print(f"  {k:32s} +{v['gain']:6.3f}   -> {v}")
OUT["oracle_substitution"] = orc

# ---------------------------------------------------------------- 3. confusion matrices, detection vs grading
print("\n## 3. Confusion matrices (rows=gold none/P2/P1/P0, cols=pred) and detection vs grading errors")
conf = {}
for d in range(6):
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (gold[:, d], pred[:, d]), 1)
    gd, pd_ = gold[:, d] > 0, pred[:, d] > 0
    det_fp = int((~gd & pd_).sum())
    det_fn = int((gd & ~pd_).sum())
    both = gd & pd_
    gerr = both & (gold[:, d] != pred[:, d])
    off1 = int((gerr & (np.abs(gold[:, d] - pred[:, d]) == 1)).sum())
    off2 = int((gerr & (np.abs(gold[:, d] - pred[:, d]) == 2)).sum())
    over = int((gerr & (pred[:, d] > gold[:, d])).sum())
    tp_det = int(both.sum())
    detF1 = 2 * tp_det / max(1, 2 * tp_det + det_fp + det_fn)
    conf[DIMS[d]] = dict(cm=cm.tolist(), det_fp=det_fp, det_fn=det_fn, det_tp=tp_det, det_F1=round(detF1, 3),
                         grade_err=int(gerr.sum()), off1=off1, off2=off2, over_graded=over, under_graded=int(gerr.sum()) - over,
                         grade_acc_given_detected=round(1 - gerr.sum() / max(1, tp_det), 3))
    print(f"  {DIMS[d]}: cm={cm.tolist()}")
    print(f"     detection: TP {tp_det} FP {det_fp} FN {det_fn} (F1 {detF1:.3f}) | grading errors among detected: {int(gerr.sum())} "
          f"(off-by-1 {off1}, off-by-2 {off2}; over {over}, under {int(gerr.sum()) - over}) grade-acc|det {conf[DIMS[d]]['grade_acc_given_detected']}")
OUT["confusion"] = conf

# ---------------------------------------------------------------- 4. headroom oracles (a)(b)(c)
print("\n## 4. Headroom oracles")
def fold_predict_restricted(d_idx):
    """grade among P2/P1/P0 using each fold decoder's biased logits (model grading for an oracle-detected defect)."""
    out = np.zeros(N, int)
    for f, st in decs["fold_decoders"].items():
        dec = Decoder55.from_state(st)
        m = fold == f
        s = c.renorm(P[m])
        lg = np.log(s[:, d_idx]) + dec.bias[d_idx][None]
        if d_idx in FULL:
            out[m] = 1 + lg[:, 1:].argmax(-1)
        else:
            out[m] = 3
    return out

hd = {}
hd["nested"] = comp(ev)
hd["a_decoder_insample"] = comp(S(pin))
qb = pred.copy()
for d in range(6):
    g = fold_predict_restricted(d)
    qb[gold[:, d] == 0, d] = 0
    m = (gold[:, d] > 0) & (qb[:, d] == 0)
    qb[m, d] = g[m]
hd["b_perfect_detection_model_grading"] = comp(S(qb))
qc = pred.copy()
m = (gold > 0) & (pred > 0)
qc[m] = gold[m]
hd["c_model_detection_perfect_grading"] = comp(S(qc))
# per-dimension versions of (b) and (c)
for d in range(6):
    q = pred.copy()
    g = fold_predict_restricted(d)
    q[gold[:, d] == 0, d] = 0
    mm = (gold[:, d] > 0) & (q[:, d] == 0)
    q[mm, d] = g[mm]
    hd[f"b_detect_perfect:{DIMS[d]}"] = comp(S(q))
    if d in FULL:
        q = pred.copy()
        mm = (gold[:, d] > 0) & (pred[:, d] > 0)
        q[mm, d] = gold[mm, d]
        hd[f"c_grade_perfect:{DIMS[d]}"] = comp(S(q))
for k, v in hd.items():
    print(f"  {k:40s} {v}  (gain {v['total'] - ev[K]:+.3f})")
OUT["headroom"] = hd
# CI for in-sample vs nested gap
OUT["headroom_ci"] = dict(
    a_minus_nested=ci(lambda b: c.score(pin[b], gold[b])[K] - c.score(pred[b], gold[b])[K], BALL[:300]),
    b_minus_nested=ci(lambda b: c.score(qb[b], gold[b])[K] - c.score(pred[b], gold[b])[K], BALL[:300]),
    c_minus_nested=ci(lambda b: c.score(qc[b], gold[b])[K] - c.score(pred[b], gold[b])[K], BALL[:300]))
print("  CIs:", OUT["headroom_ci"])

# ---------------------------------------------------------------- 5. clean images
print("\n## 5. Clean images (n=%d)" % clean.sum())
anyd = (pred[clean] > 0).any(1)
anyp0 = (pred[clean] == 3).any(1)
by = {DIMS[d]: dict(any=int((pred[clean, d] > 0).sum()), p0=int((pred[clean, d] == 3).sum()),
                    only_this=int(((pred[clean, d] > 0) & ((pred[clean] > 0).sum(1) == 1)).sum())) for d in range(6)}
gr = Counter(int(x) for x in pred[clean].max(1))
print(f"  predicted all-clean {int((~anyd).sum())}, any defect {int(anyd.sum())}, any P0 {int(anyp0.sum())}; worst grade histogram {dict(gr)}")
print("  by dimension:", by)
print(f"  clean score {ev['doc2_Clean2']:.3f} / 20 (ceiling); each wrong clean image costs {20 / clean.sum():.4f}")
# images predicted clean that are NOT clean (the other side of the gate)
pc = (pred == 0).all(1)
print(f"  predicted all-clean: {int(pc.sum())}, of which gold-defective {int((pc & ~clean).sum())} (gold worst-grade {dict(Counter(int(x) for x in gold[pc & ~clean].max(1)))})")
OUT["clean"] = dict(n=int(clean.sum()), pred_any_defect=int(anyd.sum()), pred_any_p0=int(anyp0.sum()), by_dim=by,
                    pred_all_clean=int(pc.sum()), pred_clean_but_defective=int((pc & ~clean).sum()))

# ---------------------------------------------------------------- 6. first batch vs supplement
print("\n## 6. First batch vs supplement")
eo, es = S(pred, m=orig), S(pred, m=~orig)
print("  all ", comp(ev), "\n  orig", comp(eo), "\n  supp", comp(es))
OUT["batch_ci"] = dict(orig_minus_all=ci(lambda b: c.score(pred[b], gold[b])[K] - 0, boot_idx(orig, 300)))
d_ci = {}
bo = boot_idx(orig, 300)
d_ci["orig_total"] = ci(lambda b: c.score(pred[b], gold[b])[K], bo)
bs = boot_idx(~orig, 300)
d_ci["supp_total"] = ci(lambda b: c.score(pred[b], gold[b])[K], bs)
d_ci["supp_pos"] = ci(lambda b: c.score(pred[b], gold[b])["doc2_S_pos"], bs)
d_ci["orig_pos"] = ci(lambda b: c.score(pred[b], gold[b])["doc2_S_pos"], bo)
d_ci["supp_p0"] = ci(lambda b: c.score(pred[b], gold[b])["doc2_S_p0"], bs)
d_ci["orig_p0"] = ci(lambda b: c.score(pred[b], gold[b])["doc2_S_p0"], bo)
print("  95% cluster CIs:", d_ci)
OUT["batch_ci"] = d_ci
bt = []
print(f"  {'label':12s} {'prev_o':>7s} {'prev_s':>7s} {'F1_o':>6s} {'F1_s':>6s} {'F1_all':>6s} {'TPR_o':>6s} {'TPR_s':>6s} {'FPR_o':>7s} {'FPR_s':>7s} {'F1(all-acc,o-prev)':>18s}")
for i, x in enumerate(ev["per_14"]):
    d, g = x["dim_idx"], x["grade"]
    r = {}
    for nm, m in (("o", orig), ("s", ~orig), ("all", np.ones(N, bool))):
        yt, yp = gold[m, d] == g, pred[m, d] == g
        tp, fp, fn = (yt & yp).sum(), (~yt & yp).sum(), (yt & ~yp).sum()
        r[f"prev_{nm}"] = yt.mean()
        r[f"TPR_{nm}"] = tp / max(1, yt.sum())
        r[f"FPR_{nm}"] = fp / max(1, (~yt).sum())
        r[f"F1_{nm}"] = 2 * tp / max(1, 2 * tp + fp + fn)
    pi = r["prev_o"]
    r["F1_allacc_origprev"] = 2 * r["TPR_all"] * pi / max(1e-9, r["TPR_all"] * pi + pi + r["FPR_all"] * (1 - pi))
    r["label"] = x["channel"]
    bt.append({k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in r.items()})
    print(f"  {x['channel']:12s} {r['prev_o']:7.3f} {r['prev_s']:7.3f} {r['F1_o']:6.3f} {r['F1_s']:6.3f} {r['F1_all']:6.3f} {r['TPR_o']:6.3f} {r['TPR_s']:6.3f} {r['FPR_o']:7.4f} {r['FPR_s']:7.4f} {r['F1_allacc_origprev']:18.3f}")
OUT["batch_labels"] = bt
# decomposition of orig - all in S_pos: prevalence vs accuracy
pos_prev = 25 / 14 * sum(r["F1_allacc_origprev"] - r["F1_all"] for r in bt)
pos_acc = 25 / 14 * sum(r["F1_o"] - r["F1_allacc_origprev"] for r in bt)
print(f"  S_pos(orig)-S_pos(all) = {eo['doc2_S_pos'] - ev['doc2_S_pos']:+.3f}  ~ prevalence part {pos_prev:+.3f} + accuracy part {pos_acc:+.3f}")
OUT["batch_pos_decomp"] = dict(diff=round(eo["doc2_S_pos"] - ev["doc2_S_pos"], 3), prevalence=round(pos_prev, 3), accuracy=round(pos_acc, 3))
# same for P0 sub-terms per P0 label
p0d = {}
for nm, e in (("all", ev), ("orig", eo), ("supp", es)):
    p0d[nm] = dict(R=round(e["p0_recall"], 4), P=round(e["p0_precision"], 4), F1=round(e["p0_f1"], 4))
print("  P0 macro R/P/F1:", p0d)
OUT["batch_p0"] = p0d
# per-row error rate (any dim wrong) by batch -- decomposable, with CI
err = (pred != gold).any(1)
nerr = (pred != gold).sum(1)
for nm, m in (("orig", orig), ("supp", ~orig), ("orig_defective", orig & ~clean)):
    print(f"  {nm:15s} n={m.sum():5d} any-dim-wrong {err[m].mean():.3f}  mean #dims wrong {nerr[m].mean():.3f}  per-dim acc {[round(float(x), 3) for x in (pred[m] == gold[m]).mean(0)]}")

# per-source/pipeline tag
print("\n  per tag (counts, gold defect rate, error rates, oracle gain if this tag's rows were perfect):")
tg = {}
for t, n in Counter(tags).most_common():
    m = tags == t
    q = pred.copy()
    q[m] = gold[m]
    gain = S(q)[K] - base
    tg[t] = dict(n=int(n), n_orig=int((m & orig).sum()), n_clean=int((m & clean).sum()), any_wrong=round(float(err[m].mean()), 3),
                 dims_wrong=round(float(nerr[m].mean()), 3), oracle_gain=round(gain, 3), gain_per_100=round(100 * gain / n, 3),
                 sub_score=round(S(pred, m=m)[K], 2) if n >= 40 else None)
    print(f"    {t:18s} {tg[t]}")
OUT["tags"] = tg

# ---------------------------------------------------------------- 7. concentration over videos
print("\n## 7. Error concentration over videos")
uv, vinv = np.unique(vids, return_inverse=True)
print(f"  {len(uv)} videos/sources; frames per video: {dict(Counter(Counter(vids).values()))}")
gains = np.zeros(len(uv))
for i in range(len(uv)):
    m = vinv == i
    if not err[m].any():
        continue
    q = pred.copy()
    q[m] = gold[m]
    gains[i] = S(q)[K] - base
order = np.argsort(-gains)
tot_gap = 65 - base
conc = {}
cum = pred.copy()
for k in (10, 25, 50, 100, 200, 400, 800):
    m = np.isin(vinv, order[:k])
    q = pred.copy()
    q[m] = gold[m]
    g = S(q)[K] - base
    conc[k] = dict(frac_rows=round(float(m.mean()), 3), gain=round(g, 3), frac_of_gap=round(g / tot_gap, 3))
    print(f"  fixing the top-{k:3d} videos ({m.sum()} rows, {m.mean():.1%}) gains +{g:.2f} = {g / tot_gap:.1%} of the {tot_gap:.2f}-pt gap")
# baseline: random videos of the same row share
rnd = []
for k in (100, 400):
    gs = []
    for _ in range(20):
        sel = rs.choice(len(uv), k, replace=False)
        m = np.isin(vinv, sel)
        q = pred.copy()
        q[m] = gold[m]
        gs.append(S(q)[K] - base)
    rnd.append((k, round(float(np.mean(gs)), 3)))
print("  random-video baseline (k, mean gain):", rnd)
# row-level error concentration: share of wrong (row,dim) cells in the worst x% of videos
cell = (pred != gold).sum(1)
vc = np.bincount(vinv, weights=cell)
vn = np.bincount(vinv)
srt = np.sort(vc)[::-1]
print("  wrong cells: total %d; top-10%% videos hold %.1f%%, top-25%% hold %.1f%%; videos with 0 errors: %d (%.1f%%)" % (
    cell.sum(), 100 * srt[:len(uv) // 10].sum() / cell.sum(), 100 * srt[:len(uv) // 4].sum() / cell.sum(), (vc == 0).sum(), 100 * (vc == 0).mean()))
# Poisson-ish check: is the per-video error count over-dispersed relative to per-row binomial independence?
rate = cell.sum() / N
exp_var = (vn * rate).sum()
print(f"  over-dispersion of per-video wrong-cell counts: observed var {np.var(vc) * len(uv):.0f} vs independent-rows approx {exp_var * (1 + 0):.0f}")
OUT["concentration"] = dict(top=conc, random=rnd, n_videos=len(uv), top10pct_cells=round(float(srt[:len(uv) // 10].sum() / cell.sum()), 3))

# ---------------------------------------------------------------- 8. confident errors and label noise
print("\n## 8. Confident errors")
ce = {}
for d in range(6):
    s = c.renorm(P)[:, d]
    pg = s[np.arange(N), gold[:, d]]
    wrong = pred[:, d] != gold[:, d]
    pmax = s[np.arange(N), pred[:, d]]
    ce[DIMS[d]] = dict(wrong=int(wrong.sum()), wrong_pred_prob_gt_0_8=int((wrong & (pmax > 0.8)).sum()),
                       wrong_gold_prob_lt_0_1=int((wrong & (pg < 0.1)).sum()),
                       det_conf_fn=int((wrong & (gold[:, d] > 0) & (s[:, 0] > 0.8)).sum()),
                       det_conf_fp=int((wrong & (gold[:, d] == 0) & (s[:, 0] < 0.2)).sum()))
print("  ", ce)
OUT["confident_errors"] = ce

# sibling-frame agreement and a "sibling oracle"
print("\n## 9. Sibling-frame label consistency (frames of one video)")
by_v = defaultdict(list)
for i, v in enumerate(vids):
    by_v[v].append(i)
multi = [ix for ix in by_v.values() if len(ix) >= 2]
pa, pb = [], []
for ix in multi:
    for a in range(len(ix)):
        for b in range(a + 1, len(ix)):
            pa.append(ix[a]); pb.append(ix[b])
pa, pb = np.array(pa), np.array(pb)
agree_all = (gold[pa] == gold[pb]).all(1).mean()
agree_d = (gold[pa] == gold[pb]).mean(0)
agree_det = ((gold[pa] > 0) == (gold[pb] > 0)).mean(0)
print(f"  {len(multi)} videos with >=2 frames, {len(pa)} pairs; all-six agree {agree_all:.3f}; per dim {np.round(agree_d, 3).tolist()}; detection-level {np.round(agree_det, 3).tolist()}")
# model-vs-gold agreement on the same rows
rows_m = np.unique(np.concatenate([pa, pb]))
print(f"  model nested vs gold on those rows: all-six {(pred[rows_m] == gold[rows_m]).all(1).mean():.3f}; per dim {np.round((pred[rows_m] == gold[rows_m]).mean(0), 3).tolist()}")
# sibling oracle: predict each multi-frame row with the gold of a random other frame of the same video (average over 20 draws)
sib_sc, mod_sc = [], []
for rep in range(20):
    q = gold.copy()
    for ix in multi:
        for i in ix:
            j = ix[rs.randint(len(ix) - 1)]
            j = j if j != i else ix[-1] if ix[-1] != i else ix[0]
            q[i] = gold[j]
    sib_sc.append(c.score(q[rows_m], gold[rows_m]))
mod = c.score(pred[rows_m], gold[rows_m])
sm = {k: float(np.mean([e[k] for e in sib_sc])) for k in (K, "doc2_S_pos", "doc2_S_p0", "doc2_Clean2")}
print(f"  'sibling-gold' predictor on multi-frame rows: Doc2S {sm[K]:.2f} (pos {sm['doc2_S_pos']:.2f} p0 {sm['doc2_S_p0']:.2f} cl {sm['doc2_Clean2']:.2f}) vs model {mod[K]:.2f} (pos {mod['doc2_S_pos']:.2f} p0 {mod['doc2_S_p0']:.2f} cl {mod['doc2_Clean2']:.2f})")
OUT["sibling"] = dict(n_videos=len(multi), n_pairs=int(len(pa)), agree_all=round(float(agree_all), 3), agree_dim=np.round(agree_d, 3).tolist(),
                      agree_detect=np.round(agree_det, 3).tolist(), sib_oracle=sm, model_on_same_rows=comp(mod), n_rows=int(len(rows_m)))
# confident errors where a sibling frame's gold equals the model's prediction (label inconsistency candidates)
sib_match = 0
conf_err = 0
for d in range(6):
    s = c.renorm(P)[:, d]
    pmax = s[np.arange(N), pred[:, d]]
    for i in np.where((pred[:, d] != gold[:, d]) & (pmax > 0.8))[0]:
        ix = [j for j in by_v[vids[i]] if j != i]
        if not ix:
            continue
        conf_err += 1
        sib_match += any(gold[j, d] == pred[i, d] for j in ix)
print(f"  confident (p>0.8) errors on multi-frame videos: {conf_err}; a sibling frame carries the predicted label in {sib_match} ({sib_match / max(1, conf_err):.1%})")
# compare: all errors on multi-frame videos
allm, alls = 0, 0
for d in range(6):
    for i in np.where(pred[:, d] != gold[:, d])[0]:
        ix = [j for j in by_v[vids[i]] if j != i]
        if not ix:
            continue
        allm += 1
        alls += any(gold[j, d] == pred[i, d] for j in ix)
print(f"  all errors on multi-frame videos: {allm}; sibling carries predicted label in {alls} ({alls / max(1, allm):.1%})")
OUT["sibling_conf_err"] = dict(conf_err=conf_err, sib_match=sib_match, all_err=allm, all_sib_match=alls)

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "error_analysis.json"), "w"), ensure_ascii=False, indent=1, default=float)
print("\nwritten error_analysis.json")
