#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/make_metric_probes.py — OPTIONAL, needs the owner's OK (spends 2 practice-site submissions).

Identifies which formula the scoring website uses.  No model, no labels: the two files are constant
predictions, so nothing about individual test images is learned (AGENTS.md rule 2 is respected).

  probe_A : 模糊=P2 on every image, all else 无问题
            S_clean reported 20.00 -> Doc1 cliff (only P0 on clean images counts)
            S_clean reported  0.00 -> Doc2 strict (any defect on a clean image counts)
  probe_B : 花屏=P0 on every image, all else 无问题   (R=1, P=p, F=2p/(1+p) for 花屏; 0 elsewhere)
            Doc1: S_pos = 25F/6 ,  S_p0 = 20(0.6/6 + 0.4F/6)
            Doc2: S_pos = 25F/14,  S_p0 = 20(0.5/6 + 0.3F/6 + 0.2p/6)

Usage:
  python code/next55/make_metric_probes.py --write                 # -> results/next55/probes/probe_{A,B}.json
  python code/next55/make_metric_probes.py --A-clean 20 --B-pos 0.52 --B-p0 2.17   # interpret the site's numbers
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import OUT_DIR, TEMPLATE_150, load_template_names  # noqa: E402

KEYS = ["模糊", "纹理损失", "过度生成", "虚化突兀", "过锐", "花屏"]


def write():
    names = load_template_names(TEMPLATE_150)
    d = os.path.join(OUT_DIR, "probes")
    os.makedirs(d, exist_ok=True)
    for tag, key, val in (("A", "模糊", "P2"), ("B", "花屏", "P0")):
        sub = [{"image": n, "label": {k: (val if k == key else "无问题") for k in KEYS}} for n in names]
        json.dump(sub, open(os.path.join(d, f"probe_{tag}.json"), "w"), ensure_ascii=False, indent=1)
        print(f"wrote {d}/probe_{tag}.json")


def interpret(a_clean, b_pos, b_p0):
    print(f"probe_A S_clean={a_clean:.2f} -> {'Doc1 cliff (P0-only)' if a_clean > 10 else 'Doc2 strict (all-six-clean)'}")
    for name, k_pos, f_p0 in (
        ("Doc1", 6.0, lambda F, p: 20 * (0.6 / 6 + 0.4 * F / 6)),
        ("Doc2", 14.0, lambda F, p: 20 * (0.5 / 6 + 0.3 * F / 6 + 0.2 * p / 6)),
    ):
        F = b_pos * k_pos / 25.0
        p = F / (2 - F) if F < 2 else float("nan")
        print(f"  {name}: implied 花屏 prevalence p={p:.3f} (F1={F:.3f}) -> predicted S_p0={f_p0(F, p):.3f}  vs reported {b_p0:.3f}"
              f"  |err|={abs(f_p0(F, p) - b_p0):.3f}")
    print("  -> the hypothesis with |err| ~ 0 (and a plausible p in 0.03-0.12) is the website formula.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--A-clean", type=float)
    ap.add_argument("--B-pos", type=float, help="S_pos points (0-25) reported for probe_B")
    ap.add_argument("--B-p0", type=float, help="S_p0 points (0-20) reported for probe_B")
    a = ap.parse_args()
    if a.write:
        write()
    if a.A_clean is not None and a.B_pos is not None and a.B_p0 is not None:
        interpret(a.A_clean, a.B_pos, a.B_p0)


if __name__ == "__main__":
    main()
