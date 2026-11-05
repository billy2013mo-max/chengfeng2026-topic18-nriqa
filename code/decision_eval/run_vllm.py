"""Run a VLM/decision model on the 6-question multiple-choice task via vLLM.

Outputs one JSONL row per (image, defect) with option probabilities.
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from common import CLASSES, SYSTEM, make_tasks, load_manifest


def candidate_ids(tok, letters):
    out = {}
    for L in letters:
        ids = set()
        for v in [L, ' ' + L, '\n' + L, L + '.', L + '、', '（' + L, '(' + L]:
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[L] = ids
    return out


def probs_from_logprobs(lp, cand):
    """lp: dict token_id -> logprob (float). Returns normalized option probs."""
    p = {}
    for L, ids in cand.items():
        vals = [lp.get(i, -1e9) for i in ids]
        m = max(vals)
        p[L] = float(np.exp(m) * sum(np.exp(v - m) for v in vals)) if m > -1e8 else 0.0
    s = sum(p.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in p.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--imgdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0 = all images')
    ap.add_argument('--chunk', type=int, default=24)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=8192)
    ap.add_argument('--max-pixels', type=int, default=768*768)
    ap.add_argument('--gpu-mem', type=float, default=0.90)
    ap.add_argument('--adapter', default='')
    ap.add_argument('--dtype', default='bfloat16')
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from vllm.lora.request import LoRARequest

    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    kw = dict(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, dtype=a.dtype, trust_remote_code=True,
              limit_mm_per_prompt={'image': 8})
    is_qwen = 'qwen' in a.name.lower()
    try:
        import json as _j
        _cfg = _j.load(open(os.path.join(a.model, 'config.json')))
        is_qwen = is_qwen or str(_cfg.get('model_type','')).startswith('qwen')
    except Exception:
        pass
    if a.max_pixels and is_qwen:
        kw['mm_processor_kwargs'] = {'max_pixels': a.max_pixels, 'min_pixels': 256*256}
    if a.adapter:
        kw['enable_lora'] = True
        kw['max_lora_rank'] = 64
    print('loading', a.name, kw, flush=True)
    llm = LLM(**kw)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    sp = SamplingParams(max_tokens=4, temperature=0.0, logprobs=20)
    lora_req = [LoRARequest('adapter', 1, a.adapter)] if a.adapter else None

    manifest = load_manifest(a.manifest)
    if a.limit:
        manifest = manifest[:a.limit]
    tasks = make_tasks(manifest)

    by_img = {}
    for t in tasks:
        by_img.setdefault(t['image'], []).append(t)

    t0 = time.time()
    n_done = 0
    with open(a.out, 'w') as fout:
        for img_name, group in by_img.items():
            img = Image.open(os.path.join(a.imgdir, img_name)).convert('RGB')
            prompts = []
            for t in group:
                msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                        {'role': 'user', 'content': [{'type': 'image'},
                                                     {'type': 'text', 'text': t['question']}]}]
                try:
                    tpl = getattr(proc, 'chat_template', '') or ''
                except Exception:
                    tpl = ''
                kw = {'enable_thinking': False} if 'enable_thinking' in str(tpl) else {}
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
                prompts.append({'prompt': text, 'multi_modal_data': {'image': [img]}})
            lr = [lora_req[0]] * len(prompts) if lora_req else None
            outs = llm.generate(prompts, sp, lora_request=lr)
            for t, o in zip(group, outs):
                cand = candidate_ids(tok, 'ABCD'[:len(t['options'])])
                lp = {}
                try:
                    gl = o.outputs[0].logprobs; tids = o.outputs[0].token_ids
                    for pos, tid in enumerate(tids):
                        dec = tok.decode([tid]).strip()
                        if len(dec) == 1 and dec in 'ABCD':
                            lp = {k: v.logprob for k, v in gl[pos].items()}
                            break
                except Exception:
                    pass
                p = probs_from_logprobs(lp, cand)
                valid = p is not None
                if not valid:
                    p = {L: 1.0/len(cand) for L in cand}
                fout.write(json.dumps({'id': t['id'], 'image': t['image'], 'defect': t['defect'],
                                       'options': t['options'], 'gold': t['gold'], 'gold_grade': t['gold_grade'],
                                       'probs': p, 'valid': valid,
                                       'raw': {str(k): round(v, 4) for k, v in list(lp.items())[:10]}},
                                      ensure_ascii=False) + '\n')
            n_done += 1
            if n_done % 50 == 0:
                el = time.time() - t0
                print(f'{n_done}/{len(by_img)} images  {el:.0f}s  {n_done/el:.2f} img/s', flush=True)
    print('done', a.name, n_done, 'images', f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
