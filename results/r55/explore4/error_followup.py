#!/usr/bin/env python3
# Follow-ups to error_analysis.py (CPU, OOF only): batch-gap CI, batch gap inside one pipeline tag,
# sibling-frame error correlation and grade agreement, gate cost.  Usage: python3 error_followup.py fallback_pool.npz
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402

sys.argv_backup = list(sys.argv)
K = "doc2_strict_total"
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FULL = [0, 1, 2, 4]
z = np.load(sys.argv[1], allow_pickle=True)
names = [str(x) for x in z["names"]]
gold, fold, grp, P, pred = z["gold"], z["fold"], z["grp"], z["P"], z["nested"]
N = len(names)
sp = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
orig = np.array([sp[n] == "orig" for n in names])
clean = (gold == 0).all(1)
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


def vid(n):
    m = re.match(r"^(?:\d{3}_)?([0-9a-f]{8,}|\d{9,})", n)
    return m.group(1) if m else n


tags = np.array([tag(n) for n in names])
vids = np.array([vid(n) for n in names])
ug, ginv = np.unique(grp, return_inverse=True)
rs = np.random.RandomState(1)


def boots(mask, n=500):
    mem = [np.where((ginv == i) & mask)[0] for i in range(len(ug))]
    mem = [m for m in mem if len(m)]
    return [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(n)]


def q(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


print("## A. orig - supp and orig - all differences (cluster bootstrap, 500 reps; the two batches share no group?)")
shared = len(set(grp[orig]) & set(grp[~orig]))
print(f"  groups present in both batches: {shared}")
bo, bs = boots(orig), boots(~orig)
for key in (K, "doc2_S_pos", "doc2_S_p0"):
    so = np.array([c.score(pred[b], gold[b])[key] for b in bo])
    ss = np.array([c.score(pred[b], gold[b])[key] for b in bs])
    print(f"  {key:18s} orig {c.score(pred[orig], gold[orig])[key]:.3f}  supp {c.score(pred[~orig], gold[~orig])[key]:.3f}  diff CI {q(so - ss)}")
# supplement scored with the first batch's clean images attached (removes the clean-score convention artefact)
m = ~orig | clean
e = c.score(pred[m], gold[m])
print(f"  supp + the 513 clean images: total {e[K]:.3f} (pos {e['doc2_S_pos']:.3f} p0 {e['doc2_S_p0']:.3f} cl {e['doc2_Clean2']:.3f})")
m = orig & ~clean
e2 = c.score(pred[m], gold[m])
print(f"  orig without its clean images: total {e2[K]:.3f} (pos {e2['doc2_S_pos']:.3f} p0 {e2['doc2_S_p0']:.3f} cl {e2['doc2_Clean2']:.3f})")
# per-row cell error rate on defective rows with CI
cell = (pred != gold).sum(1).astype(float)
for nm, mm in (("orig defective", orig & ~clean), ("supp defective", ~orig & ~clean)):
    bb = boots(mm, 300)
    print(f"  {nm}: mean wrong dims per row {cell[mm].mean():.3f} CI {q([cell[b].mean() for b in bb])}; mean #defective dims (gold) {(gold[mm] > 0).sum(1).mean():.2f}; mean gold severity sum {gold[mm].sum(1).mean():.2f}")

print("\n## B. Same pipeline tag, different batch (wrong dims per row, Doc2S on rows of the tag)")
for t in ("v8L3HiV6", "timestamp", "other_untagged", "v8SlHiV4"):
    for nm, mm in (("orig", orig), ("supp", ~orig)):
        m = (tags == t) & mm & ~clean
        if m.sum() < 15:
            continue
        print(f"  {t:15s} {nm}: n={m.sum():4d} wrong dims/row {cell[m].mean():.3f}  gold defective dims/row {(gold[m] > 0).sum(1).mean():.2f}  "
              f"per-dim acc {np.round((pred[m] == gold[m]).mean(0), 3).tolist()}")
# batch composition by tag
print("  supplement composition:", {t: int(((tags == t) & ~orig).sum()) for t in np.unique(tags[~orig])})

print("\n## C. Sibling frames: error correlation and grade agreement")
by_v = defaultdict(list)
for i, v in enumerate(vids):
    by_v[v].append(i)
pa, pb = [], []
for ix in by_v.values():
    for a in range(len(ix)):
        for b in range(a + 1, len(ix)):
            pa.append(ix[a]); pb.append(ix[b])
pa, pb = np.array(pa), np.array(pb)
for d in range(6):
    wa, wb = pred[pa, d] != gold[pa, d], pred[pb, d] != gold[pb, d]
    base = (pred[:, d] != gold[:, d]).mean()
    both_pos = (gold[pa, d] > 0) & (gold[pb, d] > 0)
    gagree = (gold[pa, d] == gold[pb, d])[both_pos].mean() if both_pos.any() else np.nan
    pboth = (pred[pa, d] > 0) & (pred[pb, d] > 0)
    pagree = (pred[pa, d] == pred[pb, d])[pboth].mean() if pboth.any() else np.nan
    print(f"  {DIMS[d]:5s} P(err) {base:.3f}  P(sib err | err) {wb[wa].mean() if wa.any() else np.nan:.3f}  | gold grade agreement when both frames defective {gagree:.3f} (n={both_pos.sum()}); "
          f"model's predicted-grade agreement when both predicted defective {pagree:.3f} (n={pboth.sum()})")
ok = pred == gold
for d in FULL:
    m = (gold[:, d] > 0) & (pred[:, d] > 0)
    print(f"  {DIMS[d]:5s} model grade accuracy given both gold & pred defective: {ok[m, d].mean():.3f} (n={m.sum()})")

print("\n## D. Clean gate trade-off: defective rows forced all-clean")
pc = (pred == 0).all(1) & ~clean
print(f"  {pc.sum()} defective rows predicted all-clean; their gold cells by dim (#defective): {(gold[pc] > 0).sum(0).tolist()} grades {[np.bincount(gold[pc, d], minlength=4).tolist() for d in range(6)]}")
qq = pred.copy()
qq[pc] = gold[pc]
e = c.score(qq, gold)
print(f"  if those rows were perfect: {e[K]:.3f} (+{e[K] - c.score(pred, gold)[K]:.3f})")

print("\n## E. Which grade-confusion cells cost most (oracle fix of a single gold->pred cell type, per dim)")
base = c.score(pred, gold)[K]
res = []
for d in FULL:
    for g in range(4):
        for p in range(4):
            if g == p:
                continue
            m = (gold[:, d] == g) & (pred[:, d] == p)
            if m.sum() < 5:
                continue
            qq = pred.copy()
            qq[m, d] = g
            res.append((c.score(qq, gold)[K] - base, DIMS[d], g, p, int(m.sum())))
for d in (3, 5):
    for g, p in ((0, 3), (3, 0)):
        m = (gold[:, d] == g) & (pred[:, d] == p)
        qq = pred.copy()
        qq[m, d] = g
        res.append((c.score(qq, gold)[K] - base, DIMS[d], g, p, int(m.sum())))
GN = ["none", "P2", "P1", "P0"]
for gain, dn, g, p, n in sorted(res, reverse=True)[:25]:
    print(f"  {dn:5s} gold {GN[g]:4s} -> pred {GN[p]:4s}  n={n:4d}  fixing gains +{gain:.3f}  ({gain / n * 100:.2f} per 100 cells)")
