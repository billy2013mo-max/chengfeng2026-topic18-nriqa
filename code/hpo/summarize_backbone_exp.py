"""Summarise results/backbone_exp against the existing baselines on s16-s20."""
import glob, json, os, statistics as st

BASE = {
    'two-backbone (siglip2+cradio)': 'results/phase5/tb_plain_s*_seed*.json',
    'triple (eupe+siglip2+cradio)': 'results/night/4/final_s*_seed*.json',
}
GROUPS = ['d3add', 'pcladd', 'pcgadd', 'd3swap', 'pclswap', 'pcgswap']


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        try:
            d = json.load(open(f)); t = d['test']
            rows.append(dict(total=t['total'], spos=t['S_pos'], sclean=t['S_clean'],
                             fpr=t['fpr'], f=f, bb=d.get('backbones')))
        except Exception as e:
            print('bad', f, e)
    return rows


def show(name, rows):
    if not rows:
        print(f'{name:34s} (no runs)'); return
    t = [r['total'] for r in rows]
    print(f'{name:34s} n={len(t):2d} mean={st.mean(t):6.2f} median={st.median(t):6.2f} '
          f'min={min(t):6.2f} max={max(t):6.2f} S_pos={st.mean([r["spos"] for r in rows]):.3f} '
          f'S_clean={st.mean([r["sclean"] for r in rows]):.3f}')


def main():
    show('-- baselines --', [])
    for k, p in BASE.items():
        show(k, load(p))
    show('-- new backbones --', [])
    for g in GROUPS:
        show(g, load(f'results/backbone_exp/{g}_s*_seed*.json'))

    print('\nper-split mean test total')
    allp = dict(BASE)
    allp.update({g: f'results/backbone_exp/{g}_s*_seed*.json' for g in GROUPS})
    print(f'{"config":34s} ' + ' '.join(f's{s:>6d}' for s in range(16, 21)))
    for k, p in allp.items():
        rows = load(p)
        cells = []
        for s in range(16, 21):
            v = [r['total'] for r in rows if f's{s}_' in os.path.basename(r['f'])]
            cells.append(f'{st.mean(v):6.2f}' if v else '   -  ')
        print(f'{k:34s} ' + ' '.join(cells))


if __name__ == '__main__':
    main()
