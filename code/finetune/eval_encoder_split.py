import csv, json, os, sys, numpy as np
sys.path.insert(0,'code/decision_eval')
import comp_metrics as cm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
CLASSES=['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']; FULL=['模糊','纹理损失','过锐','过度生成']
rows=list(csv.DictReader(open('data/split_trainval.csv')))
man={r['name']:r for r in csv.DictReader(open('data/manifest_combined.csv'))}
def grades(names): return np.array([[int(man[n][c]) for c in CLASSES] for n in names])
def cum(G):
    y=[]
    for ci,c in enumerate(CLASSES):
        if c in FULL: y += [(G[:,ci]>=1).astype(int),(G[:,ci]>=2).astype(int),(G[:,ci]==3).astype(int)]
        else: y += [(G[:,ci]==3).astype(int)]
    return np.stack(y,1)
def to_exact(P):
    n=len(P); p0=np.zeros((n,6)); p1=np.zeros((n,4)); p2=np.zeros((n,4)); j=0
    for ci,c in enumerate(CLASSES):
        if c in FULL:
            c2,c1,c0=P[:,j],P[:,j+1],P[:,j+2]; j+=3; k=FULL.index(c)
            p2[:,k]=np.clip(c2-c1,0,None); p1[:,k]=np.clip(c1-c0,0,None); p0[:,ci]=c0
        else: p0[:,ci]=P[:,j]; j+=1
    return p0,p1,p2
for m in sys.argv[1:]:
    d=np.load(f'feats/{m}_combined.npz',allow_pickle=True); X=d['X']; nm=list(d['names']); idx={n:i for i,n in enumerate(nm)}
    tr=[n for n in nm if next((r['split'] for r in rows if r['name']==n),None)=='train']
    # faster: build split map
    sp={r['name']:r['split'] for r in rows}
    tr=[n for n in nm if sp.get(n)=='train']; va=[n for n in nm if sp.get(n)=='val']; te=[n for n in nm if sp.get(n)=='test']
    Xtr=X[[idx[n] for n in tr]]; Xva=X[[idx[n] for n in va]]; Xte=X[[idx[n] for n in te]]
    Gtr,Gva,Gte=grades(tr),grades(va),grades(te); Y=cum(Gtr)
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xva=sc.transform(Xva); Xte=sc.transform(Xte)
    pc=PCA(n_components=512,random_state=0).fit(Xtr); Xtr=pc.transform(Xtr); Xva=pc.transform(Xva); Xte=pc.transform(Xte)
    Pva=np.zeros((len(Xva),14)); Pte=np.zeros((len(Xte),14))
    for j in range(14):
        clf=LogisticRegression(C=0.05,class_weight='balanced',max_iter=3000).fit(Xtr,Y[:,j])
        Pva[:,j]=clf.predict_proba(Xva)[:,1]; Pte[:,j]=clf.predict_proba(Xte)[:,1]
    p0,p1,p2=to_exact(Pva); b,th=cm.optimise(Gva,p0,p1,p2,0.02)
    p0,p1,p2=to_exact(Pte); st=cm.score(Gte,cm.decode(p0,p1,p2,*th),0.02)
    res={'model':m,'val_total':b['total'],'thresholds':[float(x) for x in th],
         'test':{k:st[k] for k in ['total','S_pos','p0R','p0P','p0F','S_p0','S_clean','fpr']},
         'per_p0':{f'{x[0]}_P0':{'P':x[2],'R':x[3],'F1':x[4]} for x in st['per'] if x[1]=='P0'}}
    json.dump(res,open(f'results/enc_split_{m}.json','w'),ensure_ascii=False,indent=1)
    print(m, 'test total %.1f S_pos %.3f P0R %.3f S_p0 %.3f S_clean %.3f'%(st['total'],st['S_pos'],st['p0R'],st['S_p0'],st['S_clean']), flush=True)
