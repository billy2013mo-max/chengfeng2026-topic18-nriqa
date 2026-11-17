#!/usr/bin/env python3
"""
E1 Falsification Experiment (CPU, nested 8-fold CV):
Compares Decision-Layer Policy Gradient (REINFORCE / GRPO with group baseline)
and Differentiable Expected-Doc2 Surrogate against `RoutedGateEUDecoder`
on the exact same honest 8-fold OOF pooled probabilities from `resident_v6_core`.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c
import final_engine as fe
from eval_v6_gates import build_proto_cfg, fit_routed_gate_eu_lams, LAMS_V6


class PolicyNet(nn.Module):
    def __init__(self, in_dim=26, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, 24),
        )

    def forward(self, x):
        logits = self.net(x).view(-1, 6, 4)
        # Mask P0-only dims {3, 5} grades {1, 2}
        mask = torch.zeros_like(logits)
        mask[:, [3, 5], 1:3] = -1e9
        return logits + mask


def build_features(Q: np.ndarray) -> np.ndarray:
    # Q: (N, 6, 4)
    p3_max = Q[:, :, 3].max(axis=1, keepdims=True)
    p_clean = np.prod(Q[:, :, 0], axis=1, keepdims=True)
    return np.concatenate([Q.reshape(len(Q), 24), p3_max, p_clean], axis=1).astype(np.float32)


def per_image_doc2_reward(pred: np.ndarray, gold: np.ndarray, n_cl: float, n_def: float, n_p0_tot: float) -> np.ndarray:
    """Decomposable per-image contribution to Doc2_Strict (Pos + P0 + Clean2)."""
    # Clean2: +10/n_cl if gold clean & pred clean; -10/n_def if gold def & pred clean?
    # Actually Clean2 = 10 * (TN_clean/n_cl - FP_nonclean/n_def) -> let's use exact Clean2 + P0 + Pos proxy:
    y_clean = (gold == 0).all(axis=1).astype(np.float64)
    a_clean = (pred == 0).all(axis=1).astype(np.float64)
    r_cln = 10.0 * (y_clean * a_clean / max(1.0, n_cl) - (1.0 - y_clean) * a_clean / max(1.0, n_def))

    # P0 reward: F1-style per-image match on grade==3 across 6 dims (scaled to 25 pts)
    tp_p0 = ((pred == 3) & (gold == 3)).sum(axis=1).astype(np.float64)
    fp_p0 = ((pred == 3) & (gold != 3)).sum(axis=1).astype(np.float64)
    fn_p0 = ((pred != 3) & (gold == 3)).sum(axis=1).astype(np.float64)
    r_p0 = (25.0 / max(1.0, n_p0_tot)) * (2.0 * tp_p0 - fp_p0 - fn_p0)

    # Pos reward: 14-class F1 proxy (scaled to 25 pts / len(gold))
    pos_match = 0.0
    pos_fp = 0.0
    pos_fn = 0.0
    for d in (0, 1, 2, 4):
        for g in (1, 2, 3):
            pm = (pred[:, d] == g).astype(np.float64)
            gm = (gold[:, d] == g).astype(np.float64)
            pos_match += pm * gm
            pos_fp += pm * (1.0 - gm)
            pos_fn += (1.0 - pm) * gm
    r_pos = (25.0 / max(1.0, len(gold))) * (2.0 * pos_match - 0.5 * pos_fp - 0.5 * pos_fn)
    return r_cln * len(gold) / 10.0 + r_p0 + r_pos


def train_fold_policy(X_tr, gold_tr, mode="grpo", epochs=120, lr=3e-3, seed=2026):
    torch.manual_seed(seed)
    net = PolicyNet(X_tr.shape[1], 64)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(gold_tr)
    n_cl = float((gold_tr == 0).all(axis=1).sum())
    n_def = float(len(gold_tr) - n_cl)
    n_p0_tot = float((gold_tr == 3).sum())

    # Precompute expected class weights for differentiable surrogate
    y_clean = torch.from_numpy((gold_tr == 0).all(axis=1).astype(np.float32))
    y_p0 = torch.from_numpy((gold_tr == 3).astype(np.float32))  # (N, 6)
    y_pos = F.one_hot(yt.clamp(0, 3), 4).float()  # (N, 6, 4)

    for ep in range(epochs):
        opt.zero_grad()
        logits = net(xt)
        probs = F.softmax(logits, dim=-1)  # (N, 6, 4)
        if mode == "surrogate":
            # Differentiable expected Doc2 reward
            p_clean_all = torch.prod(probs[:, :, 0], dim=1)
            r_cln = 10.0 * (y_clean * p_clean_all / max(1.0, n_cl) - (1.0 - y_clean) * p_clean_all / max(1.0, n_def)) * len(gold_tr) / 10.0
            p3 = probs[:, :, 3]
            r_p0 = (25.0 / max(1.0, n_p0_tot)) * (2.0 * (p3 * y_p0).sum(dim=1) - (p3 * (1.0 - y_p0)).sum(dim=1) - ((1.0 - p3) * y_p0).sum(dim=1))
            # Pos match across dims 0,1,2,4 grades 1,2,3
            p_sub = probs[:, [0, 1, 2, 4], 1:4]
            y_sub = y_pos[:, [0, 1, 2, 4], 1:4]
            r_pos = (25.0 / len(gold_tr)) * (2.0 * (p_sub * y_sub).sum(dim=(1, 2)) - 0.5 * (p_sub * (1.0 - y_sub)).sum(dim=(1, 2)) - 0.5 * ((1.0 - p_sub) * y_sub).sum(dim=(1, 2)))
            loss = -(r_cln + r_p0 + r_pos).mean()
        else:
            # GRPO / REINFORCE with G=8 samples per image
            dist = torch.distributions.Categorical(probs=probs)
            G = 8
            actions = dist.sample((G,))  # (G, N, 6)
            logp = dist.log_prob(actions).sum(dim=-1)  # (G, N)
            rewards = []
            act_np = actions.numpy()
            for g_i in range(G):
                rewards.append(per_image_doc2_reward(act_np[g_i], gold_tr, n_cl, n_def, n_p0_tot))
            R = torch.from_numpy(np.stack(rewards, axis=0)).float()  # (G, N)
            adv = (R - R.mean(dim=0, keepdim=True)) / (R.std(dim=0, keepdim=True) + 1e-6)
            loss = -(adv.detach() * logp).mean()
        loss.backward()
        opt.step()
    return net


def main():
    names, gold, fold, grp = c.load_split()
    is_orig = np.array(["_g" not in str(n) for n in names], dtype=bool)
    _, _, _, _, P_v6 = fe.oof_pool(fe.SETS["resident_v6_core"])
    members = fe.SETS["resident_v6_core"]
    cfg = build_proto_cfg(members)

    N = len(names)
    pred_eu = np.zeros((N, 6), dtype=np.int64)
    pred_grpo = np.zeros((N, 6), dtype=np.int64)
    pred_surr = np.zeros((N, 6), dtype=np.int64)

    for fk in range(8):
        m_tr = fold != fk
        m_va = fold == fk
        P_tr = {m: P_v6[m][m_tr] for m in members}
        P_va = {m: P_v6[m][m_va] for m in members}
        dec = fit_routed_gate_eu_lams(P_tr, gold[m_tr], is_orig[m_tr], grp[m_tr], cfg, LAMS_V6)
        Q_tr = dec.pool_from_member_dict(P_tr)
        Q_va = dec.pool_from_member_dict(P_va)
        pred_eu[m_va] = dec.eu.predict(Q_va)

        X_tr = build_features(Q_tr)
        X_va = build_features(Q_va)

        net_grpo = train_fold_policy(X_tr, gold[m_tr], mode="grpo", epochs=120, seed=2026 + fk)
        net_surr = train_fold_policy(X_tr, gold[m_tr], mode="surrogate", epochs=120, seed=2026 + fk)
        with torch.no_grad():
            pred_grpo[m_va] = net_grpo(torch.from_numpy(X_va)).argmax(dim=-1).numpy()
            pred_surr[m_va] = net_surr(torch.from_numpy(X_va)).argmax(dim=-1).numpy()

    for label, pr in [("RoutedGateEU (baseline)", pred_eu), ("E1_Surrogate_Doc2", pred_surr), ("E1_GRPO_REINFORCE", pred_grpo)]:
        sa = c.score(pr, gold)
        so = c.score(pr[is_orig], gold[is_orig])
        print(f"{label:25s} | Doc2S_all={sa['doc2_strict_total']:.3f} (Pos={sa['doc2_S_pos']:.2f} P0={sa['doc2_S_p0']:.2f} Cln={sa['doc2_Clean2']:.2f}) "
              f"| Doc2S_orig={so['doc2_strict_total']:.3f} (Pos={so['doc2_S_pos']:.2f} P0={so['doc2_S_p0']:.2f} Cln={so['doc2_Clean2']:.2f})")


if __name__ == "__main__":
    main()
