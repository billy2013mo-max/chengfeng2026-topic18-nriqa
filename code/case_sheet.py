import sys, csv, os, numpy as np
sys.path.insert(0,'code/decision_eval')
from comp_metrics import load_probe, argmax_decode
from common import CLASSES
from PIL import Image, ImageDraw
MODELS=['internvit-6b','eupe-vit-b','siglip2-so400m','convnextv2-base','pe-core-l14']
SHORT={'internvit-6b':'Intern','eupe-vit-b':'EUPE','siglip2-so400m':'SigLIP2','convnextv2-base':'ConvNxt','pe-core-l14':'PE-L'}
# row order = combined manifest order (combine_feats guarantees it)
d=np.load('feats/siglip2-so400m_combined.npz',allow_pickle=True)
names=list(d['names'])
rows={r['name']:r for r in csv.DictReader(open('data/manifest_combined.csv'))}
preds={}; gold=None
for m in MODELS:
    N,g,q0,q1,q2=load_probe(f'feats/{m}_combined_oof.npz'); preds[m]=argmax_decode(q0,q1,q2); gold=g
G=['无','P2','P1','P0']
ci={c:i for i,c in enumerate(CLASSES)}
cases=[]
# 1) bokeh P0: SigLIP2 misses entirely, InternViT catches
ix=[i for i in range(len(names)) if gold[i,ci['虚化突兀']]==3 and preds['siglip2-so400m'][i,ci['虚化突兀']]==0 and preds['internvit-6b'][i,ci['虚化突兀']]==3]
cases += [('虚化P0: SigLIP2漏/Intern中', i, '虚化突兀') for i in ix[:4]]
# 2) overgen P0: InternViT misses, SigLIP2 catches
ix=[i for i in range(len(names)) if gold[i,ci['过度生成']]==3 and preds['internvit-6b'][i,ci['过度生成']]!=3 and preds['siglip2-so400m'][i,ci['过度生成']]==3]
cases += [('过度生成P0: Intern漏/SigLIP2中', i, '过度生成') for i in ix[:4]]
# 3) clean false alarms: ConvNeXt/PE-L flag P0 but InternViT clean
ix=[i for i in range(len(names)) if (gold[i]==0).all() and (preds['convnextv2-base'][i]==3).any() and not (preds['internvit-6b'][i]==3).any()]
cases += [('clean误报: ConvNxt误报/Intern不报', i, None) for i in ix[:4]]
# 4) oversharp P0: PE-L misses, Intern catches
ix=[i for i in range(len(names)) if gold[i,ci['过锐']]==3 and preds['pe-core-l14'][i,ci['过锐']]!=3 and preds['internvit-6b'][i,ci['过锐']]==3]
cases += [('过锐P0: PE-L漏/Intern中', i, '过锐') for i in ix[:4]]
TH=300; CW=TH*2
cells=[]
for title,i,cls in cases:
    n=names[i]; r=rows[n]; base='data/images_supp' if r['split']=='supp' else 'data/images'
    im=Image.open(os.path.join(base,n)).convert('RGB'); w,h=im.size; im.thumbnail((CW,TH))
    c=Image.new('RGB',(CW,TH+46),'white'); c.paste(im,((CW-im.width)//2,46+(TH-im.height)//2))
    dr=ImageDraw.Draw(c); dr.text((3,2),title,fill='black')
    gtxt=' '.join(f"{SHORT[m]}={G[preds[m][i,ci[cls]]]}" for m in MODELS) if cls else ' '.join(f"{SHORT[m]}={'P0' if (preds[m][i]==3).any() else 'ok'}" for m in MODELS)
    dr.text((3,16),gtxt,fill='black'); dr.text((3,30),f"{w}x{h} {'/'.join(k for k in CLASSES if gold[i,ci[k]]==3) or 'clean'}",fill='gray')
    cells.append(c)
sheet=Image.new('RGB',(CW*4,(TH+46)*4),(225,225,225))
for k,c in enumerate(cells): sheet.paste(c,((k%4)*CW,(k//4)*(TH+46)))
sheet.save('/tmp/race18/cases.png'); print('cases',len(cells),sheet.size)
