"""Lossless native-resolution pixel cache: raw RGB uint8 + offset index."""
import argparse, csv, os, numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

CACHE='/dev/shm/race18_native/cache.bin'; INDEX='/dev/shm/race18_native/index.npz'

def work(args):
    idxs, rows, offsets, shapes, orig = args
    fd=os.open(CACHE, os.O_WRONLY)
    for i in idxs:
        r=rows[i]; h,w=shapes[i]
        p=os.path.join('data/images' if r['name'] in orig else 'data/images_supp', r['name'])
        im=Image.open(p).convert('RGB')
        assert im.size==(w,h), (r['name'], im.size, (w,h))
        arr=np.asarray(im, dtype=np.uint8)
        os.pwrite(fd, arr.tobytes(), int(offsets[i]))
    os.close(fd)
    return len(idxs)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=32); a=ap.parse_args()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    rows=list(csv.DictReader(open('data/manifest_combined.csv')))
    names=[r['name'] for r in rows]
    shapes=[(int(r['height']),int(r['width'])) for r in rows]
    sizes=[h*w*3 for h,w in shapes]
    offsets=np.cumsum([0]+sizes)[:-1]
    total=sum(sizes)
    with open(CACHE,'wb') as f: f.truncate(total)
    orig=set(r['name'] for r in csv.DictReader(open('data/manifest.csv')))
    chunks=[list(range(i,len(rows),a.workers)) for i in range(a.workers)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        print('done', sum(ex.map(work, [(c,rows,offsets,shapes,orig) for c in chunks])))
    np.savez_compressed(INDEX, names=np.array(names), offsets=offsets, h=np.array([h for h,w in shapes]),
                        w=np.array([w for h,w in shapes]), total=total)
    print('cache %.2f GB'%(total/1e9))
