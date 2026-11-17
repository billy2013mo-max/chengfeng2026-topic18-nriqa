"""Pilot: can a large open VLM act as a JUDGE or an in-context GRADER for severity?  (local weights, vLLM, one GPU;
nothing leaves the machine.)  Follow-up of results/decision_model_screen.md, which only tested zero-shot absolute grading.

Part 1 — pairwise judge.  Two first-batch frames from different videos whose human grades for one defect differ by
    exactly one step (P0 vs P1, P1 vs P2, P2 vs none).  Question: "which image has the more severe <defect>?".
    Every pair is asked in both orders (position bias cancels).  Reference on the same pairs: the ensemble's
    out-of-fold expected grade (computed by eval_vllm_judge_pilot.py).
Part 2 — in-context grading.  The four-option grade question for a query frame, (a) zero-shot and (b) preceded by
    four labelled exemplar frames (one per grade, other videos).  Balanced sample of queries per grade.

Usage:
  CUDA_VISIBLE_DEVICES=0 python code/decision_eval/run_vllm_judge_pilot.py --model models/JEV-27B-VL --name qwen38-27b \
      --out-dir results/r55/explore4/vlm_pilot
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
for _p in ("code", "code/next55"):
    sys.path.insert(0, os.path.join(REPO, _p))
from common import CLASSES, DEFS, GRADE_DEFS, GRADE_NAME, SYSTEM  # noqa: E402
from run_vllm import candidate_ids, probs_from_logprobs  # noqa: E402

GRADED = ["模糊", "纹理损失", "过锐", "过度生成"]


def load_labels():
    sys.path.insert(0, os.path.join(REPO, "code/r55"))
    import common55 as c
    from probe_eval import DIMS
    names, gold, fold, grp = c.load_split()
    batch = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    names = [str(n) for n in names]
    o = np.array([batch[n] == "orig" for n in names])
    return names, np.asarray(gold), np.asarray(grp), o, list(DIMS)


def build_tasks(n_pairs=150, n_queries=50):
    """Deterministic task lists (seed 2026).  Smaller pilots must take the FIRST k pairs / queries of each cell of the
    default lists (see subset()), so that every model is scored on the same frames."""
    names, gold, grp, o, DIMS = load_labels()
    rs = random.Random(2026)
    a = argparse.Namespace(pairs=n_pairs, queries=n_queries)
    pairs = []
    for defect in GRADED:
        d = DIMS.index(defect)
        for hi, lo in ((3, 2), (2, 1), (1, 0)):
            H = [i for i in range(len(names)) if o[i] and gold[i, d] == hi]
            L = [i for i in range(len(names)) if o[i] and gold[i, d] == lo]
            seen = set()
            while len([p for p in pairs if p["defect"] == defect and p["hi"] == hi]) < a.pairs:
                i, j = rs.choice(H), rs.choice(L)
                if grp[i] == grp[j] or (i, j) in seen:
                    continue
                seen.add((i, j))
                pairs.append(dict(defect=defect, hi=hi, lo=lo, img_hi=names[i], img_lo=names[j]))
    queries = []
    for defect in GRADED:
        d = DIMS.index(defect)
        by = {g: [i for i in range(len(names)) if o[i] and gold[i, d] == g] for g in range(4)}
        for g in range(4):
            for i in rs.sample(by[g], min(a.queries, len(by[g]))):
                ex = []
                for ge in (3, 2, 1, 0):
                    cand = [j for j in by[ge] if grp[j] != grp[i]]
                    ex.append((names[rs.choice(cand)], GRADE_NAME[ge]))
                rs.shuffle(ex)
                order = ["P0", "P1", "P2", "无问题"]
                random.Random(int(hashlib.md5(f"2026|{names[i]}|{defect}".encode()).hexdigest(), 16)).shuffle(order)
                queries.append(dict(defect=defect, image=names[i], gold_grade=int(g), gold=GRADE_NAME[g], exemplars=ex, options=order))
    return pairs, queries


def subset(pairs, queries, k_pairs, k_queries):
    out_p, out_q, cp, cq = [], [], {}, {}
    for p in pairs:
        key = (p["defect"], p["hi"])
        if cp.get(key, 0) < k_pairs:
            cp[key] = cp.get(key, 0) + 1
            out_p.append(p)
    for q in queries:
        key = (q["defect"], q["gold_grade"])
        if cq.get(key, 0) < k_queries:
            cq[key] = cq.get(key, 0) + 1
            out_q.append(q)
    return out_p, out_q


def pair_question(defect):
    d, contrast = DEFS[defect]
    return "\n".join([f"请比较「图片一」和「图片二」在「{defect}」维度上的严重程度。", "", f"【定义】{defect}：{d}", f"【易混淆】{contrast}", "",
                      f"哪一张图片的「{defect}」问题更严重？", "A. 图片一", "B. 图片二", "", "请只输出一个字母（A、B），不要解释。"])


def grade_question(defect, order, fewshot):
    d, contrast = DEFS[defect]
    head = "请参考上面四张已标注等级的示例图片，判断「待判断图片」" if fewshot else "请判断这张图片"
    lines = [f"{head}在「{defect}」维度上的严重程度。", "", f"【定义】{defect}：{d}", f"【易混淆】{contrast}", "", "【等级标准】", GRADE_DEFS, "", "【选项】"]
    lines += [f"{'ABCD'[i]}. {g}" for i, g in enumerate(order)]
    lines += ["", "请只输出一个字母（A、B、C、D），不要解释。"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", type=int, default=150, help="pairs per defect and contrast")
    ap.add_argument("--queries", type=int, default=50, help="query frames per defect and grade")
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many prompts per part")
    ap.add_argument("--dry", action="store_true", help="build the tasks and two prompts with the processor, then stop (no GPU)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    img_dir = os.path.join(REPO, "data/images")
    pairs, queries = build_tasks(a.pairs, a.queries)
    print(f"tasks: {len(pairs)} pairs (x2 orders), {len(queries)} grade queries (x2: zero-shot, 4 exemplars)", flush=True)

    # ---------------- model
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    if a.dry:
        tpl = str(getattr(proc, "chat_template", "") or "")
        tkw = {"enable_thinking": False} if "enable_thinking" in tpl else {}
        q = queries[0]
        content = []
        for k, (n, g) in enumerate(q["exemplars"]):
            content += [{"type": "text", "text": f"示例{k + 1}（「{q['defect']}」等级：{g}）："}, {"type": "image"}]
        content += [{"type": "text", "text": "待判断图片："}, {"type": "image"}, {"type": "text", "text": grade_question(q["defect"], q["options"], True)}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
        print(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw))
        p = pairs[0]
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": [{"type": "text", "text": "图片一："}, {"type": "image"}, {"type": "text", "text": "图片二："}, {"type": "image"}, {"type": "text", "text": pair_question(p["defect"])}]}]
        print(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw))
        import collections
        print("pairs per (defect, contrast):", dict(collections.Counter((x["defect"], x["hi"]) for x in pairs)))
        print("queries per (defect, grade):", dict(collections.Counter((x["defect"], x["gold_grade"]) for x in queries)))
        print("all images exist:", all(os.path.exists(os.path.join(img_dir, n)) for x in pairs for n in (x["img_hi"], x["img_lo"])))
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, dtype="bfloat16",
              trust_remote_code=True, limit_mm_per_prompt={"image": 8}, mm_processor_kwargs={"max_pixels": a.max_pixels, "min_pixels": 256 * 256})
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    sp = SamplingParams(max_tokens=4, temperature=0.0, logprobs=20)
    tpl = str(getattr(proc, "chat_template", "") or "")
    tkw = {"enable_thinking": False} if "enable_thinking" in tpl else {}
    cache = {}

    def img(n):
        if n not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[n] = Image.open(os.path.join(img_dir, n)).convert("RGB")
        return cache[n]

    def prompt(content, images):
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
        return {"prompt": proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw), "multi_modal_data": {"image": [img(n) for n in images]}}

    def run(items, letters, out_path):
        """items: list of (record, prompt dict)."""
        cand = candidate_ids(tok, letters)
        t0, n_bad = time.time(), 0
        with open(out_path, "w", encoding="utf-8") as f:
            for k in range(0, len(items), a.chunk):
                part = items[k:k + a.chunk]
                outs = llm.generate([p for _, p in part], sp)
                for (rec, _), out in zip(part, outs):
                    lp = {}
                    try:
                        gl, tids = out.outputs[0].logprobs, out.outputs[0].token_ids
                        for pos, tid in enumerate(tids):
                            dec = tok.decode([tid]).strip()
                            if len(dec) == 1 and dec in letters:
                                lp = {kk: v.logprob for kk, v in gl[pos].items()}
                                break
                    except Exception:
                        pass
                    p = probs_from_logprobs(lp, cand)
                    if p is None:
                        n_bad += 1
                        p = {L: 1.0 / len(letters) for L in letters}
                        rec = dict(rec, valid=False)
                    f.write(json.dumps(dict(rec, probs=p), ensure_ascii=False) + "\n")
                if (k // a.chunk) % 10 == 0:
                    print(f"  {os.path.basename(out_path)}: {min(len(items), k + a.chunk)}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        print(f"done {out_path}: {len(items)} prompts, {n_bad} without a valid answer, {time.time() - t0:.0f}s", flush=True)

    # ---------------- part 1: pairwise
    items = []
    for p in pairs:
        for first in ("hi", "lo"):
            a_img, b_img = (p["img_hi"], p["img_lo"]) if first == "hi" else (p["img_lo"], p["img_hi"])
            content = [{"type": "text", "text": "图片一："}, {"type": "image"}, {"type": "text", "text": "图片二："}, {"type": "image"},
                       {"type": "text", "text": pair_question(p["defect"])}]
            items.append((dict(p, first=first), prompt(content, [a_img, b_img])))
    run(items[:a.limit] if a.limit else items, "AB", os.path.join(a.out_dir, f"pairs_{a.name}.jsonl"))

    # ---------------- part 2: grading, zero-shot and with four exemplars
    items = []
    for q in queries:
        content = [{"type": "image"}, {"type": "text", "text": grade_question(q["defect"], q["options"], False)}]
        items.append((dict(q, mode="zero"), prompt(content, [q["image"]])))
        content = []
        for k, (n, g) in enumerate(q["exemplars"]):
            content += [{"type": "text", "text": f"示例{k + 1}（「{q['defect']}」等级：{g}）："}, {"type": "image"}]
        content += [{"type": "text", "text": "待判断图片："}, {"type": "image"}, {"type": "text", "text": grade_question(q["defect"], q["options"], True)}]
        items.append((dict(q, mode="few"), prompt(content, [n for n, _ in q["exemplars"]] + [q["image"]])))
    run(items[:a.limit] if a.limit else items, "ABCD", os.path.join(a.out_dir, f"grade_{a.name}.jsonl"))


if __name__ == "__main__":
    main()
