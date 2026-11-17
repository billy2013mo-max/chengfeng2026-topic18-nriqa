"""eval_add.py TAG name=tower_dir[:family] ...  -> v7 members + extra towers, nested RoutedGateEU 8-fold; reports dims 2/4."""
import os, sys, pickle, csv, numpy as np, warnings; warnings.filterwarnings("ignore")
REPO = "/wanqing-develop/luowenjing/18"
for p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval", "code/r55/sprint"): sys.path.insert(0, os.path.join(REPO, p))
import common55 as c, routed_eu_decoder as R
from eval_v6_gates import LAMS_V6, build_proto_cfg
from eval_specialist_gate import load_oof_dir
from harness import D, g, fold, io, groups, names, report
tag = sys.argv[1]; extra = sys.argv[2:]
drop = [e[1:] for e in extra if e.startswith("-")]
P = {m: v for m, v in D["P"].items() if m not in drop}
for e in extra:
    if e.startswith("-"): continue
    nm, rest = e.split("=", 1); d_, fam = (rest.split(":") + ["ntg"])[:2]
    arr, valid = load_oof_dir(d_, names); assert valid.all(), (nm, valid.sum())
    P[nm] = c.renorm(arr); R.MEMBER_FAMILY_MAP[nm] = fam
members = list(P)
cfg = build_proto_cfg(members, "family"); cfg["lams_cl"] = LAMS_V6; cfg["bkseg1_alpha"] = 0.25 if "pe_bkseg1" in P else 0.0
pred = np.zeros_like(g); Qp = np.zeros((len(g), 6, 4))
for k in range(8):
    tr, va = fold != k, fold == k
    dec = R.RoutedGateEUDecoder(**cfg).fit({m: v[tr] for m, v in P.items()}, g[tr], is_orig=io[tr], groups=groups[tr])
    sub = {m: v[va] for m, v in P.items()}
    pred[va] = dec.predict_from_member_dict(sub); Qp[va] = dec.pool_from_member_dict(sub)
s = c.score(pred[io], g[io]); s2 = c.score(pred, g)
report(f"{tag} [{len(members)}m] (ORIG Doc2 {s['doc2_strict_total']:.3f} ALL {s2['doc2_strict_total']:.3f})", {2: pred[:, 2], 4: pred[:, 4]})
from ordinal_dec import nested_cum
pc = pred.copy(); pc[:, 2] = nested_cum(Qp[:, 2], g[:, 2])
report(f"{tag} + 过锐 cum-decode", {2: pc[:, 2], 4: nested_cum(Qp[:, 4], g[:, 4])})
pickle.dump(dict(pred=pred, Q=Qp, pred_cum=pc, members=members), open(f"{REPO}/results/r55/sprint_os_og/pred_{tag}.pkl", "wb"))
