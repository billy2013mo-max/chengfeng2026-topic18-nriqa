import argparse, csv, json, os, sys, numpy as np
os.environ.setdefault('IMAGE_MAX_TOKEN_NUM','1024')
from swift import BaseArguments, InferRequest, TransformersEngine, get_template
from swift.infer_engine import RequestConfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'decision_eval'))
import comp_metrics as cm
CLASSES=['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']; FULL=['模糊','纹理损失','过锐','过度生成']

def infer_rows(engine, rows, bs=8):
    reqs=[InferRequest(messages=r['messages'], images=r['images']) for r in rows]
    outs=[]
    for i in range(0,len(reqs),bs):
        res=engine.infer(reqs[i:i+bs], request_config=RequestConfig(logprobs=True, top_logprobs=14), use_tqdm=False)
        for r in res:
            lp=r.choices[0].logprobs['content'][0]['top_logprobs']
            v=np.zeros(14)
            for it in lp: v[it['index']]=float(np.exp(it['logprob']))
            outs.append(v)
        print(f'{min(i+bs,len(reqs))}/{len(reqs)}', flush=True)
    return np.stack(outs)

def to_exact(V):
    n=len(V); p0=np.zeros((n,6)); p1=np.zeros((n,4)); p2=np.zeros((n,4)); j=0
    for ci,c in enumerate(CLASSES):
        if c in FULL:
            c2,c1,c0=V[:,j],V[:,j+1],V[:,j+2]; j+=3; k=FULL.index(c)
            p2[:,k]=np.clip(c2-c1,0,None); p1[:,k]=np.clip(c1-c0,0,None); p0[:,ci]=c0
        else:
            p0[:,ci]=V[:,j]; j+=1
    return p0,p1,p2

def grades_and_names(rows):
    man={r['name']:r for r in csv.DictReader(open('data/manifest_combined.csv'))}
    names=[os.path.basename(r['images'][0]) for r in rows]
    G=np.array([[int(man[n][c]) for c in CLASSES] for n in names])
    return G,names

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True)
    ap.add_argument('--val',default='data/swift/val.jsonl'); ap.add_argument('--test',default='data/swift/test.jsonl')
    ap.add_argument('--out',default='results/swift_eval.json'); ap.add_argument('--tau',type=float,default=0.02)
    a=ap.parse_args()
    args=BaseArguments.from_pretrained(a.ckpt)
    if os.path.exists(os.path.join(a.ckpt,'model.safetensors')):
        engine=TransformersEngine(a.ckpt, task_type='seq_cls', num_labels=args.num_labels,
                                  problem_type=args.problem_type, model_type=args.model_type, max_batch_size=8)
    else:
        engine=TransformersEngine(args.model, adapters=[a.ckpt], task_type='seq_cls', num_labels=args.num_labels,
                                  problem_type=args.problem_type, model_type=args.model_type, max_batch_size=8)
    engine.template=get_template(engine.processor, args.system, template_type=args.template, use_chat_template=args.use_chat_template)
    vr=[json.loads(l) for l in open(a.val)]; tr=[json.loads(l) for l in open(a.test)]
    Vv=infer_rows(engine,vr); Vt=infer_rows(engine,tr)
    Gv,nv=grades_and_names(vr); Gt,nt=grades_and_names(tr)
    p0,p1,p2=to_exact(Vv); b,th=cm.optimise(Gv,p0,p1,p2,a.tau)
    p0,p1,p2=to_exact(Vt); st=cm.score(Gt,cm.decode(p0,p1,p2,*th),a.tau)
    res={'ckpt':a.ckpt,'val':{k:b[k] for k in ['total','S_pos','p0R','p0P','S_p0','S_clean']},'thresholds':[float(x) for x in th],
         'test':{k:st[k] for k in ['total','S_pos','p0R','p0P','p0F','S_p0','S_clean','fpr']},
         'per_p0':{f'{x[0]}_P0':{'P':x[2],'R':x[3],'F1':x[4]} for x in st['per'] if x[1]=='P0'}}
    json.dump(res,open(a.out,'w'),ensure_ascii=False,indent=1)
    np.savez_compressed(a.out.replace('.json','_probs.npz'), Vv=Vv, Vt=Vt, Gv=Gv, Gt=Gt, nv=np.array(nv), nt=np.array(nt))
    print(json.dumps(res,ensure_ascii=False,indent=1))
