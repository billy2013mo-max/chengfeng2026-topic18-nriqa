#!/usr/bin/env python3
"""
code/audit_final_submission.py
Read-only audit of a candidate submission against the canonical 52.89 baseline and the
later unverified variants. Prints the invariants the 55+ builder is supposed to hold.

Usage:
    python3 code/audit_final_submission.py --candidate pred_nriqa_dataset_test.json
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

GRADE = {'无问题': 0, 'P2': 1, 'P1': 2, 'P0': 3}
GRADE_NAME = {v: k for k, v in GRADE.items()}


def load(path):
    data = json.load(open(path, encoding='utf-8'))
    classes = list(data[0]['label'].keys())
    mat = np.array([[GRADE[e['label'][c]] for c in classes] for e in data], dtype=np.int64)
    names = [e['image'] for e in data]
    return names, classes, mat


def summarize(tag, mat):
    clean = int((mat == 0).all(axis=1).sum())
    p0_imgs = int((mat == 3).any(axis=1).sum())
    p0_cells = int((mat == 3).sum())
    pos_cells = int((mat > 0).sum())
    print(f"  {tag:34s} clean_images={clean:3d}  p0_images={p0_imgs:3d}  "
          f"p0_cells={p0_cells:3d}  positive_cells={pos_cells:3d}")
    return {'clean': clean, 'p0_images': p0_imgs, 'p0_cells': p0_cells, 'pos_cells': pos_cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default='pred_nriqa_dataset_test.json')
    ap.add_argument('--baseline',
                    default='results/final/pred_nriqa_dataset_test_phase1_breakthrough.json')
    args = ap.parse_args()

    paths = [
        ('52.89 Phase-1 baseline', args.baseline),
        ('52.89 + 10 QualiRAG', 'results/final/pred_nriqa_dataset_test_qualirag_consensus.json'),
        ('55+ ultimate (old)', 'results/final/pred_nriqa_dataset_test_55plus_ultimate.json'),
        ('candidate', args.candidate),
    ]
    print("=" * 78)
    print("Submission audit")
    print("=" * 78)
    loaded = {}
    for tag, path in paths:
        if not os.path.exists(path):
            print(f"  [skip] {tag}: {path} missing")
            continue
        names, classes, mat = load(path)
        loaded[tag] = (names, classes, mat)
        summarize(tag, mat)

    base_name, base_classes, base = loaded['52.89 Phase-1 baseline']
    cand_name, cand_classes, cand = loaded['candidate']
    assert base_name == cand_name, "image order differs from baseline"
    assert base_classes == cand_classes, "class order differs from baseline"

    diff = cand != base
    print(f"\nChanged vs 52.89 baseline: {int(diff.sum())} cells across "
          f"{int(diff.any(axis=1).sum())} images")
    trans = Counter((int(base[i, c]), int(cand[i, c])) for i, c in zip(*np.where(diff)))
    for (old, new), n in sorted(trans.items(), key=lambda kv: -kv[1]):
        print(f"  {GRADE_NAME[old]:>4s} -> {GRADE_NAME[new]:<4s} : {n}")

    by_class = Counter()
    for i, c in zip(*np.where(diff)):
        by_class[cand_classes[c]] += 1
    print("  by class:", dict(by_class))

    base_clean = set(np.where((base == 0).all(axis=1))[0])
    cand_clean = set(np.where((cand == 0).all(axis=1))[0])
    base_p0 = set(np.where((base == 3).any(axis=1))[0])
    cand_p0 = set(np.where((cand == 3).any(axis=1))[0])
    print(f"\n  clean image set identical: {base_clean == cand_clean} "
          f"(added={len(cand_clean - base_clean)}, removed={len(base_clean - cand_clean)})")
    print(f"  P0 image set identical   : {base_p0 == cand_p0} "
          f"(added={len(cand_p0 - base_p0)}, removed={len(base_p0 - cand_p0)})")
    huaping = cand_classes.index('花屏')
    print(f"  花屏 P0 count            : {int((cand[:, huaping] == 3).sum())} "
          f"(baseline {int((base[:, huaping] == 3).sum())})")

    changed = []
    for i, c in zip(*np.where(diff)):
        changed.append(f"{cand_name[i]} | {cand_classes[c]} | "
                       f"{GRADE_NAME[int(base[i, c])]} -> {GRADE_NAME[int(cand[i, c])]}")
    print("\nAll changed cells:")
    for line in changed:
        print("  " + line)


if __name__ == '__main__':
    main()
