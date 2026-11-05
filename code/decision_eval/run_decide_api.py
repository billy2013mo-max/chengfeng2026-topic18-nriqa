"""Evaluate JEV's native /v1/decide endpoint on the 6-question task."""
import argparse, base64, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from common import make_tasks, load_manifest

LET = 'ABCD'

def img_data_url(path, max_side):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((max(1,int(w*s)), max(1,int(h*s))), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def decide(url, state, question, options, timeout=120):
    body = {'kind': 'choice', 'state': state, 'question': question,
            'options': options, 'thinking': 'off'}
    r = requests.post(url.rstrip('/') + '/v1/decide', json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:8010')
    ap.add_argument('--manifest', required=True); ap.add_argument('--imgdir', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-side', type=int, default=768)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    manifest = load_manifest(a.manifest)
    if a.limit: manifest = manifest[:a.limit]
    tasks = make_tasks(manifest)
    by_img = {}
    for t in tasks: by_img.setdefault(t['image'], []).append(t)

    results = []
    t0 = time.time(); done = 0
    def work(img_name, group):
        state = ['这是一张待检测的视频截图：', {'image': img_data_url(os.path.join(a.imgdir, img_name), a.max_side)}]
        out = []
        for t in group:
            try:
                r = decide(a.url, state, t['question'], t['options'])
                probs = {LET[i]: float(r['probabilities'][i]) for i in range(len(t['options']))}
                after = r.get('adaptation', '')
                out.append((t, probs, True, after))
            except Exception as e:
                out.append((t, {LET[i]: 1.0/len(t['options']) for i in range(len(t['options']))}, False, str(e)[:80]))
        return out

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, k, v): k for k, v in by_img.items()}
        with open(a.out, 'w') as f:
            for f_ in as_completed(futs):
                for t, probs, valid, note in f_.result():
                    f.write(json.dumps({'id': t['id'], 'image': t['image'], 'defect': t['defect'],
                                        'options': t['options'], 'gold': t['gold'], 'gold_grade': t['gold_grade'],
                                        'probs': probs, 'valid': valid, 'note': note}, ensure_ascii=False) + '\n')
                done += 1
                if done % 50 == 0:
                    el = time.time()-t0
                    print(f'{done}/{len(by_img)} imgs  {el:.0f}s  {done/el:.2f} img/s', flush=True)
    print('done', done, f'{time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
