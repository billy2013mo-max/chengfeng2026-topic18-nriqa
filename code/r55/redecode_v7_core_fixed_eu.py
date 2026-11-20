#!/usr/bin/env python3
"""Re-decode the fresh resident_v7_core probabilities with the EU decision pair fixed to (lam_cl, p0_scale) = (3.0, 1.0)
(the nested-best pair and the most frequent per-fold choice) instead of the pair the full-data fit picked in-sample (3.0, 0.85).
Fusion weights, temperatures, calibration are untouched. Usage: core_redecode.py [--write]"""
import collections
import copy
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # needs rev_common.py (review helper) next to this file
import rev_common as rc  # noqa: E402
import numpy as np  # noqa: E402
import common55 as c  # noqa: E402
import routed_eu_decoder as R  # noqa: E402
from final_engine import oof_pool  # noqa: E402

WRITE = "--write" in sys.argv
SET, LAM, P0, TAG = "resident_v7_core", 3.0, 1.0, "p0s100"
FIN = os.path.join(rc.REPO, "results/r55/final")
K = "doc2_strict_total"
obj = pickle.load(open(os.path.join(FIN, f"{SET}_decoder_routed_gate_eu.pkl"), "rb"))
dec, members = obj["decoder"], obj["members"]
names, gold, fold, _, P = oof_pool(list(members))
_, gold2, _, groups, is_orig = rc.split_info()
assert np.array_equal(gold, gold2)
Qf = dec.pool_from_member_dict(P)
row_w = np.where(is_orig, 1.0, dec.eu.w_supp)
st_chk = R.TrueGateConstrainedEU._fit_stats(Qf, gold, row_w, dec.eu.lam_cl, dec.eu.p0_scale)
print(f"deployed pair ({dec.eu.lam_cl}, {dec.eu.p0_scale}); reproduction of its fitted statistics: max|dF| {np.abs(st_chk['F'] - dec.eu.st['F']).max():.2e} max|dP| {np.abs(st_chk['P'] - dec.eu.st['P']).max():.2e}")
st_new = R.TrueGateConstrainedEU._fit_stats(Qf, gold, row_w, LAM, P0)
dec2 = copy.deepcopy(dec)
dec2.eu.st, dec2.eu.lam_cl, dec2.eu.p0_scale = st_new, LAM, P0
po, pn = dec.eu.predict(Qf), dec2.eu.predict(Qf)
print(f"in-sample (optimistic) Doc2: deployed ALL {c.score(po, gold)[K]:.3f} ORIG {c.score(po[is_orig], gold[is_orig])[K]:.3f} | fixed ({LAM}, {P0}) ALL {c.score(pn, gold)[K]:.3f} ORIG {c.score(pn[is_orig], gold[is_orig])[K]:.3f}")

G = ["无", "P2", "P1", "P0"]
for stem in ("pred_nriqa_dataset_test_r55_resident_v7_core_routed_gate_eu", "pred_nriqa_dataset_final_644_r55_resident_v7_core_routed_gate_eu"):
    z = np.load(os.path.join(FIN, stem + "_probs.npz"), allow_pickle=True)
    nm = [str(x) for x in z["names"]]
    Pt = {m: z[f"p_{m}"] for m in members}
    Qt = dec.pool_from_member_dict(Pt)
    print(f"\n{stem}: n={len(nm)} | pooled probabilities reproduce the saved ones: max abs diff {np.abs(Qt - z['routed']).max():.2e}")
    p_old, p_new = dec.eu.predict(Qt), dec2.eu.predict(Qt)
    old_json = json.load(open(os.path.join(FIN, stem + ".json"), encoding="utf-8"))
    same = c.format_official_submission(nm, p_old)[0] == old_json
    print(f"  re-decode with the deployed decoder equals the existing file: {same}")
    assert same
    ch = np.argwhere(p_old != p_new)
    by = collections.Counter(f"{rc.DIMS[d]} {G[p_old[i, d]]}->{G[p_new[i, d]]}" for i, d in ch)
    print(f"  cells changed {len(ch)} / {p_old.size}: {dict(by)}")
    print(f"  all-clean images {int((p_old == 0).all(1).sum())} -> {int((p_new == 0).all(1).sum())} | P0 cells {int((p_old == 3).sum())} -> {int((p_new == 3).sum())} | P1 {int((p_old == 2).sum())} -> {int((p_new == 2).sum())} | P2 {int((p_old == 1).sum())} -> {int((p_new == 1).sum())}")
    for i, d in ch:
        print(f"    {nm[i]:<60} {rc.DIMS[d]} {G[p_old[i, d]]}->{G[p_new[i, d]]}  probs(无,P2,P1,P0)={np.round(Qt[i, d], 3).tolist()}")
    if WRITE:
        op = os.path.join(FIN, f"{stem}_{TAG}.json")
        json.dump(c.format_official_submission(nm, p_new)[0], open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("  wrote", op)
if WRITE:
    dp = os.path.join(FIN, f"{SET}_decoder_routed_gate_eu_{TAG}.pkl")
    pickle.dump(dict(decoder=dec2, members=members,
                     note=f"same fusion weights/temperatures/calibration as {SET}_decoder_routed_gate_eu.pkl; EU pair fixed to ({LAM}, {P0}) "
                          f"(nested-best fixed pair and modal per-fold choice) instead of the in-sample pick ({dec.eu.lam_cl}, {dec.eu.p0_scale})"), open(dp, "wb"))
    print("wrote", dp)
print("DONE")
