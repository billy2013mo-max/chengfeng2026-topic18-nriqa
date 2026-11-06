import os, sys, json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from run_scheme1_scheme2_breakthrough import FULL_IDX

def check_cum(p0, p1, p2, label):
    cum = p0.copy()
    for k, ci in enumerate(FULL_IDX):
        cum[:, ci] += p1[:, k] + p2[:, k]
    mx = cum.max(1)
    q = np.quantile(mx, [0.05, 0.10, 0.14, 0.18, 0.25, 0.50]).round(3)
    print(f"{label:55s} | quantiles(5,10,14,18,25,50%)={q} | <0.28: {(mx<0.28).mean():.3f} | <0.38: {(mx<0.38).mean():.3f}")

# Check individual models that have BOTH s16_te, s17_te, AND t150 in the exact same .npz file!
for p in [
    'results/v9_specialists/V9_SpecE_RealHiRes_Giant_s16_probs.npz',
    'results/v9_specialists/V9_SpecE_RealHiRes_Giant_s17_probs.npz',
    'results/v9_specialists/V9_SpecC_AllP0_Tile3_s16_probs.npz',
    'results/v8_experts/V8_W2_ExpC_blurtex_s16_probs.npz',
]:
    d = np.load(p)
    tag = os.path.basename(p).replace('_probs.npz', '')
    check_cum(d['te_p0'], d['te_p1'], d['te_p2'], f"{tag} [te 1000]")
    check_cum(d['t150_p0'], d['t150_p1'], d['t150_p2'], f"{tag} [t150 150]")

# Now check v2_probs, v3_probs, v5_probs on test_150!
v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
for k, (p0, p1, p2, fam, is_iso) in list(v2_probs.items())[:4]:
    check_cum(p0, p1, p2, f"v2_super/{k} (is_iso={is_iso}) [t150]")

v3_probs = np.load('results/v3_super/test_150_v3_models_probs.npz', allow_pickle=True)['data'].item()
for k, val in list(v3_probs.items())[:4]:
    check_cum(val[0], val[1], val[2], f"v3_super/{k} [t150]")

v5_probs = np.load('results/v5_clean/test_150_v5_models_probs.npz', allow_pickle=True)['data'].item()
for k, val in list(v5_probs.items())[:4]:
    check_cum(val[0], val[1], val[2], f"v5_clean/{k} [t150]")
