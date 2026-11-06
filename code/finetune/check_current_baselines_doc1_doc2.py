import os, sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from cv_decoder_calib import decode_pc, HAND
from optimize_v8_isolated_test import load_healthy_split_models, decode_shielded_pc
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs

def score_both(gold, pred, tau=0.02):
    s1 = cm.score(gold, pred, tau)
    clean = (gold == 0).all(1)
    pred_clean = (pred == 0).all(1)
    s_clean_doc2 = float((clean & pred_clean).sum() / max(1, clean.sum()))
    tot_doc2 = 25.0 * s1['S_pos'] + 20.0 * s1['S_p0'] + 20.0 * s_clean_doc2
    s1['S_clean_doc2'] = s_clean_doc2
    s1['total_doc2'] = tot_doc2
    return s1

for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    p0_te, p1_te, p2_te, cln0_te, cln1_te, cln2_te = build_decoupled_14ch_probs(models, 'te')
    pred_hand = decode_pc(p0_te, p1_te, p2_te, HAND)
    st_hand = score_both(gte, pred_hand)

    pred_cln = decode_pc(cln0_te, cln1_te, cln2_te, HAND)
    st_cln = score_both(gte, pred_cln)

    print(f"=== Split s{sp} Multi-Model Ensembles ===")
    print(f"  Clean_9Model_Ensemble (HAND)   | Doc1={st_cln['total']:.2f} (Clean1={st_cln['S_clean']:.3f}) | Doc2={st_cln['total_doc2']:.2f} (Clean2={st_cln['S_clean_doc2']:.3f}) | S_pos={st_cln['S_pos']:.3f} S_p0={st_cln['S_p0']:.3f}")
    print(f"  Decoupled_14ch_Ensemble (HAND) | Doc1={st_hand['total']:.2f} (Clean1={st_hand['S_clean']:.3f}) | Doc2={st_hand['total_doc2']:.2f} (Clean2={st_hand['S_clean_doc2']:.3f}) | S_pos={st_hand['S_pos']:.3f} S_p0={st_hand['S_p0']:.3f}")
