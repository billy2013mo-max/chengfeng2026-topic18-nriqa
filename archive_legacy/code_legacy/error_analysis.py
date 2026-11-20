import sys, json, os, numpy as np
sys.path.insert(0, 'code/decision_eval')
from comp_metrics import load_probe, argmax_decode
from common import CLASSES

FULL = ['模糊','纹理损失','过锐','过度生成']
MODELS = ['internvit-6b','eupe-vit-b','siglip2-so400m','convnextv2-base','pe-core-l14']
GNAME = ['无','P2','P1','P0']

def analyse(m):
    N, g, p0, p1, p2 = load_probe(f'feats/{m}_combined_oof.npz')
    pred = argmax_decode(p0, p1, p2)
    out = {'model': m, 'n': N, 'classes': {}}
    tot_exact = 0
    for ci, c in enumerate(CLASSES):
        yt, yp = g[:,ci], pred[:,ci]
        cm = np.zeros((4,4), int)
        for a,b in zip(yt,yp): cm[a,b]+=1
        acc = np.trace(cm)/cm.sum()
        # P0
        tp = cm[3,3]; fn = cm[3,:3].sum(); fp = cm[:3,3].sum()
        rec = tp/max(1,tp+fn); prec = tp/max(1,tp+fp)
        # P2 vs none
        p2_none = int(cm[1,0]); none_p2 = int(cm[0,1]); p2_ok = int(cm[1,1])
        out['classes'][c] = dict(acc=round(float(acc),3), p0_recall=round(float(rec),3),
            p0_prec=round(float(prec),3), p0_tp=int(tp), p0_fn=int(fn), p0_fp=int(fp),
            gold=[int(x) for x in cm.sum(1)], pred=[int(x) for x in cm.sum(0)],
            p2_to_none=p2_none, none_to_p2=none_p2, p2_ok=p2_ok,
            cm=cm.tolist())
    out['exact_match'] = round(float((pred==g).all(1).mean()),3)
    out['mean_correct_labels'] = round(float((pred==g).mean()),3)
    # clean false alarm
    clean = (g==0).all(1)
    out['clean_n'] = int(clean.sum())
    out['clean_fp'] = round(float(((pred==3).any(1) & clean).sum()/max(1,clean.sum())),3)
    # confidence: max prob of chosen option
    conf = np.zeros((N,6))
    for ci,c in enumerate(CLASSES):
        if c in FULL:
            j = FULL.index(c)
            pn = np.clip(1-p0[:,ci]-p1[:,j]-p2[:,j],0,None)
            conf[:,ci] = np.stack([pn,p2[:,j],p1[:,j],p0[:,ci]],1).max(1)
        else:
            conf[:,ci] = np.maximum(p0[:,ci], 1-p0[:,ci])
    ok = pred==g
    out['conf_correct'] = round(float(conf[ok].mean()),3)
    out['conf_wrong'] = round(float(conf[~ok].mean()),3)
    # high-confidence errors (P0 relevant)
    hi = (conf>0.8) & (~ok)
    out['high_conf_err'] = int(hi.sum())
    # collect case ids
    cases=[]
    for ci,c in enumerate(CLASSES):
        idx=np.where(hi[:,ci])[0][:3]
        for i in idx: cases.append({'i':int(i),'class':c,'gold':GNAME[int(g[i,ci])],'pred':GNAME[int(pred[i,ci])],'conf':round(float(conf[i,ci]),3)})
    out['cases']=cases
    return out, pred, g

if __name__=='__main__':
    res={}
    preds={}
    for m in MODELS:
        o,p,g=analyse(m); res[m]=o; preds[m]=(p,g)
        print(f"\n=== {m} ===")
        print(f"exact-match {o['exact_match']:.3f}  per-label acc {o['mean_correct_labels']:.3f}  clean-FP {o['clean_fp']:.3f}  conf(correct/wrong) {o['conf_correct']}/{o['conf_wrong']}  high-conf errors {o['high_conf_err']}")
        for c in CLASSES:
            d=o['classes'][c]
            print(f"  {c:5s} acc={d['acc']:.2f} P0 R/P={d['p0_recall']:.2f}/{d['p0_prec']:.2f} (FN={d['p0_fn']} FP={d['p0_fp']})  P2→无={d['p2_to_none']} 无→P2={d['none_to_p2']}  gold={d['gold']} pred={d['pred']}")
    json.dump({k:{'summary':{kk:vv for kk,vv in v.items() if kk not in('classes',)},'classes':v['classes']} for k,v in res.items()},
              open('/tmp/race18/error_analysis.json','w'), ensure_ascii=False, indent=1)
    # pairwise agreement
    print('\n=== pairwise prediction agreement (per label) ===')
    for i in range(len(MODELS)):
        for j in range(i+1,len(MODELS)):
            pi,_=preds[MODELS[i]]; pj,_=preds[MODELS[j]]
            print(f"{MODELS[i]} vs {MODELS[j]}: {(pi==pj).mean():.3f}")
