#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG_DIRS = [os.path.join(REPO, "data/images"), os.path.join(REPO, "data/images_supp")]
DEFAULT_CSV = os.path.join(REPO, "data/qingyu_splits/qingyu_8fold.csv")
DEFAULT_OUT = os.path.join(REPO, "results/r55/overgen_http_eval")

GRADE = {3: "P0", 2: "P1", 1: "P2", 0: "无问题"}
GRADE_ID = {v: k for k, v in GRADE.items()}
VALID = ("P0", "P1", "P2", "无问题")

SYSTEM = "你是资深图像与视频画质评测专家。严格按定义和等级标准判断，不要臆测图像外的信息。"

BODY = """请判断这张图片在「过度生成」维度上的严重程度。

【定义】过度生成：画面凭空出现多余、不合理的物体、细节或纹理，违背真实逻辑（如多余肢体、结构崩坏、漂浮物体、文字/纹理乱码），或原有内容被 AI 重绘成非自然的高频伪纹理、塑料感质感。
【易混淆】与过锐的区别：过锐是边缘被强化出白边/振铃，内容本身没有多出东西；过度生成是内容层面多出了不合理的东西，或原有结构被凭空重构。与纹理损失的区别：纹理损失是原有质感被涂抹丢失；过度生成是重绘/新造出非自然纹理。与虚化突兀的区别：后者是虚化过渡边界生硬，不涉及内容多出或重绘。
【等级标准】
P0：明显、大面积或结构性的过度生成，凭空出现不合理物体/结构崩坏/严重伪纹理，明显影响内容理解。
P1：中等程度过度生成，局部有可见的非自然重绘、多余细节或伪纹理，主体内容仍可辨识。
P2：轻微过度生成，整体观感良好，细看可见少量非自然重绘痕迹或轻度塑料感。
无问题：不存在过度生成缺陷，画面内容与纹理自然合理。

【输出格式】严格输出两行，不要输出其他内容：
第一行：等级=P0 或 P1 或 P2 或无问题
第二行：理由=<不超过30字>"""


def find_image(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def data_url(path, max_side):
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_grade(text):
    if not text:
        return None
    for pat in (r"等级\s*[=:：]\s*(P0|P1|P2|无问题)", r"答案\s*[=:：]\s*(P0|P1|P2|无问题)"):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    tail = text.strip().splitlines()[-3:]
    for line in reversed(tail):
        m = re.search(r"(P0|P1|P2|无问题)", line)
        if m:
            return m.group(1)
    return None


def ask(args, key, content):
    import requests
    body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    err = ""
    for attempt in range(args.retries):
        try:
            r = requests.post(args.url.rstrip("/") + "/chat/completions", json=body, timeout=args.timeout,
                              headers={"Authorization": f"Bearer {key}"} if key else {})
            if r.status_code == 200:
                ch = r.json()["choices"][0]
                text = (ch["message"].get("content") or "").strip()
                usage = r.json().get("usage", {})
                finish = ch.get("finish_reason", "")
                if not text:
                    err = f"empty content finish={finish}"
                else:
                    return text, usage, True, "", finish
            err = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(min(30, 2 ** attempt))
    return "", {}, False, err, ""


def load_rows(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    keep = [r for r in rows if int(r["过度生成"]) > 0]
    return keep


def summarize(results):
    cm = {g: {p: 0 for p in VALID} for g in VALID}
    invalid = 0
    for r in results:
        if r.get("pred") not in VALID:
            invalid += 1
            continue
        cm[r["gold"]][r["pred"]] += 1
    per = {}
    for g in ("P0", "P1", "P2", "无问题"):
        n = sum(cm[g].values())
        hit = cm[g][g]
        per[g] = {"n": n, "hit": hit, "recall": (hit / n) if n else 0.0,
                  "predictions": cm[g]}
    pos_n = sum(per[g]["n"] for g in ("P0", "P1", "P2"))
    pos_hit = sum(per[g]["predictions"][g] for g in ("P0", "P1", "P2"))
    defect_hit = sum(per[g]["predictions"][p] for g in ("P0", "P1", "P2") for p in ("P0", "P1", "P2"))
    recalls = [per[g]["recall"] for g in ("P0", "P1", "P2") if per[g]["n"]]
    return {"confusion": cm, "per_class": per, "invalid": invalid,
            "exact_hit": pos_hit, "exact_n": pos_n,
            "defect_detected": defect_hit,
            "macro_recall_P0P1P2": sum(recalls) / len(recalls) if recalls else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://wanqing-prt.internal/api/gateway/v1/endpoints")
    ap.add_argument("--model", default="ep-2bbmni-1794561470122917386")
    ap.add_argument("--key-env", default="OVERGEN_API_KEY")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--name", default="gemini31pro")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    key = os.environ.get(a.key_env, "")
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"overgen_{a.name}.jsonl")
    sum_path = os.path.join(a.out_dir, f"overgen_{a.name}_summary.json")

    rows = load_rows(a.csv)
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if a.resume and os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
    todo = [r for r in rows if r["name"] not in done]
    print(f"overgen positives: {len(rows)}, done: {len(done)}, todo: {len(todo)}, workers: {a.workers}", flush=True)

    lock = threading.Lock()
    t0 = time.time()
    counter = {"n": 0}

    def work(r):
        name = r["name"]
        gold_grade = int(r["过度生成"])
        rec = {"name": name, "gold_grade": gold_grade, "gold": GRADE[gold_grade]}
        p = find_image(name)
        if p is None:
            rec.update(pred=None, ok=False, error="image_not_found")
            return rec
        content = [{"type": "text", "text": BODY},
                   {"type": "image_url", "image_url": {"url": data_url(p, a.max_side)}}]
        ts = time.time()
        text, usage, ok, err, finish = ask(a, key, content)
        pred = parse_grade(text) if ok else None
        rec.update(pred=pred, pred_grade=GRADE_ID.get(pred), ok=bool(ok and pred),
                   raw=text, error=err, usage=usage, finish=finish, elapsed=round(time.time() - ts, 2))
        with lock:
            counter["n"] += 1
            if counter["n"] % 20 == 0:
                print(f"  {counter['n']}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        return rec

    mode = "a" if a.resume else "w"
    with open(out_path, mode, encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    results = []
    seen = set()
    for line in open(out_path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        results.append(r)
    s = summarize(results)
    s.update(model=a.model, url=a.url, images=len(results), elapsed_s=round(time.time() - t0, 1),
             positive_defect_recall=(s["defect_detected"] / s["exact_n"]) if s["exact_n"] else 0.0)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: s[k] for k in ("images", "invalid", "per_class", "macro_recall_P0P1P2",
                                        "positive_defect_recall", "elapsed_s")}, ensure_ascii=False, indent=2), flush=True)
    print("summary ->", sum_path, flush=True)


if __name__ == "__main__":
    main()
