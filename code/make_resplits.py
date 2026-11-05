"""Generate grouped, label- and resolution-stratified alternative splits.

Purpose: test whether the HPO-selected hyperparameters generalise beyond the
single fixed 2073/500/1000 split.  Each split keeps the same sizes and the same
class/resolution composition, differs only in which images land where.

Usage: python3 -I code/make_resplits.py --seeds 11,12,13,14,15
"""
import argparse, csv, collections, os, random

CL = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
SIZES = {'train': 2073, 'val': 500, 'test': 1000}
ORDER = ['train', 'val', 'test']


def res_bucket(r):
    m = min(int(r['width']), int(r['height']))
    if m < 720:
        return 'lt720'
    if m < 1080:
        return '720-1080'
    if m < 2160:
        return '1080-2160'
    return 'ge2160'


def source_key(rows):
    """Majority label-presence vector + resolution bucket of the source."""
    bits = collections.Counter(tuple(int(r[c]) > 0 for c in CL) for r in rows)
    b = bits.most_common(1)[0][0]
    rb = collections.Counter(res_bucket(r) for r in rows).most_common(1)[0][0]
    return b + (rb,)


def make_split(rows, seed):
    rng = random.Random(seed)
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r['source']].append(r)
    srcs = list(by_src)
    rng.shuffle(srcs)

    strata = collections.defaultdict(list)
    for s in srcs:
        strata[source_key(by_src[s])].append(s)

    assign = {}
    tot = sum(SIZES.values())
    # round-robin allocation inside each stratum, largest-deficit first
    for key in sorted(strata, key=lambda k: -len(strata[k])):
        group = strata[key]
        # chunk each stratum by target proportions
        n = len(group)
        raw = {s: n * SIZES[s] / tot for s in ORDER}
        quota = {s: int(raw[s]) for s in ORDER}
        rest = n - sum(quota.values())
        for s in sorted(ORDER, key=lambda s: -(raw[s] - quota[s]))[:rest]:
            quota[s] += 1
        i = 0
        for s in ORDER:
            for src in group[i:i + quota[s]]:
                assign[src] = s
            i += quota[s]

    # repair global totals by moving whole sources between splits
    counts = collections.Counter()
    for s, sp in assign.items():
        counts[sp] += len(by_src[s])
    for _ in range(500):
        over = max(ORDER, key=lambda s: counts[s] - SIZES[s])
        under = max(ORDER, key=lambda s: SIZES[s] - counts[s])
        if counts[over] <= SIZES[over] or counts[under] >= SIZES[under]:
            break
        need = SIZES[under] - counts[under]
        cands = [s for s in srcs if assign[s] == over and len(by_src[s]) <= need]
        cands.sort(key=lambda s: (abs(len(by_src[s]) - need), rng.random()))
        if not cands:  # fall back to the smallest source, accepting a small overshoot
            cands = sorted([s for s in srcs if assign[s] == over], key=lambda s: len(by_src[s]))[:1]
        if not cands:
            break
        src = cands[0]
        counts[over] -= len(by_src[src])
        counts[under] += len(by_src[src])
        assign[src] = under

    out = []
    for r in rows:
        out.append({'name': r['name'], 'split': assign[r['source']], 'source': r['source']})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='11,12,13,14,15')
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--outdir', default='data/resplits')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = list(csv.DictReader(open(a.manifest)))
    for seed in [int(x) for x in a.seeds.split(',')]:
        out = make_split(rows, seed)
        path = os.path.join(a.outdir, f'split_s{seed}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
            w.writeheader(); w.writerows(out)
        cnt = collections.Counter(r['split'] for r in out)
        # composition check
        lab = collections.defaultdict(collections.Counter)
        res = collections.defaultdict(collections.Counter)
        mand = {r['name']: r for r in rows}
        for r in out:
            m = mand[r['name']]
            for c in CL:
                lab[r['split']][int(m[c])] += 1
            res[r['split']][res_bucket(m)] += 1
        print(f'seed {seed}: {dict(cnt)} -> {path}')
        for s in ORDER:
            p0 = sum(v for k, v in lab[s].items() if k == 3)
            n = sum(lab[s].values())
            print(f'   {s:5s} n={cnt[s]:4d} P0cells={p0:4d} ({p0/n:.3f})  res={dict(res[s])}')
        # source overlap check
        src_split = collections.defaultdict(set)
        for r in out:
            src_split[r['source']].add(r['split'])
        bad = sum(1 for v in src_split.values() if len(v) > 1)
        print(f'   sources spanning >1 split: {bad}')


if __name__ == '__main__':
    main()
