"""Review-only helpers (CPU, read-only on the repo): nested 8-fold RoutedGateEU decode for a member list."""
import csv
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402

REPO = "/wanqing-develop/luowenjing/18"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
for p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)

DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
G = ["无", "P2", "P1", "P0"]
GRADED = (0, 1, 2, 4)


def split_info():
    import common55 as c
    names, gold, fold, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    return names, gold, fold, groups, is_orig


def nested(members, keep_decs=False):
    import deconfound_calibrator as dc
    from final_engine import oof_pool
    from routed_eu_decoder import RoutedGateEUDecoder
    names, gold, fold, _, P = oof_pool(list(members))
    _, gold2, fold2, groups, is_orig = split_info()
    assert np.array_equal(gold, gold2) and np.array_equal(fold, fold2)
    cfg = dc._build_routed_gate_cfg(list(members))
    Q = np.zeros((len(names), 6, 4))
    pred = np.zeros((len(names), 6), dtype=np.int64)
    decs = []
    for k in range(8):
        va = fold == k
        tr = ~va
        dec = RoutedGateEUDecoder(**cfg).fit({m: v[tr] for m, v in P.items()}, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
        q = dec.pool_from_member_dict({m: v[va] for m, v in P.items()})
        Q[va] = q
        pred[va] = dec.eu.predict(q)
        if keep_decs:
            decs.append(dec)
    return dict(names=names, gold=gold, fold=fold, groups=groups, is_orig=is_orig, P=P, Q=Q, pred=pred, decs=decs)


def row_rps(Q, gold):
    """Per-row, per-dim ranked probability score (graded dims) / Brier on P0 (binary dims). Shape (N, 6)."""
    N = Q.shape[0]
    out = np.zeros((N, 6))
    ks = np.arange(3)[None, :]
    for d in range(6):
        if d in GRADED:
            cdf_p = np.cumsum(Q[:, d, :3], axis=1)
            cdf_t = (gold[:, d][:, None] <= ks).astype(np.float64)
            out[:, d] = ((cdf_p - cdf_t) ** 2).sum(1) / 3.0
        else:
            out[:, d] = (Q[:, d, 3] - (gold[:, d] == 3).astype(np.float64)) ** 2
    return out


def cluster_boot_idx(groups, B=1000, seed=20261117):
    uniq = np.unique(groups)
    gi = {u: np.where(groups == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    for _ in range(B):
        yield np.concatenate([gi[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
