"""Prompt-variant study for the 过度生成 pairwise judge (owner-approved endpoint; SENDS LABELLED FIRST-BATCH FRAMES).

Two stages, so that prompt choice never touches the frames used for the final number:
  dev      development pairs (positions 60-93 of every contrast in the pilot's default pair list; no frame shared
           with the held-out pairs or the anchors) -> every variant scored, best variant picked by mean accuracy
  heldout  the 180 pairs (first 60 per contrast) already scored with the ORIGINAL prompt -> picked variant, once

Variants (all ask "which image has the more severe 过度生成", both orders):
  v1  definition rewritten from the annotators' convention (painted texture, waxy skin, pseudo-glyph text, ...)
  v2  v1 + four labelled anchor frames (P0 / P1 / P2 / none, other videos, ensemble agrees with the label)
  v3  v1 + a native-resolution crop of the most textured region next to each whole frame
  v4  v1 + "describe what you see in each image, then answer" (final line 答案：A/B)

Usage (key from the environment, never from disk):
  JUDGE_API_KEY=... python code/decision_eval/run_http_prompt_variants.py --stage dev --url ... --model ... --name gemini4argon
  JUDGE_API_KEY=... python code/decision_eval/run_http_prompt_variants.py --stage heldout --variant v2 ...
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from common import SYSTEM  # noqa: E402
from run_http_judge_pilot import data_url  # noqa: E402
from run_vllm_judge_pilot import build_tasks, load_labels  # noqa: E402

DEFECT = "过度生成"
IMG_DIR = os.path.join(REPO, "data/images")
POOL = os.environ.get("POOL_NPZ", "")

DEFINITION = (
    "【过度生成的含义（以标注员的实际标准为准）】\n"
    "指画面中出现了真实拍摄/正常处理不会产生的“被生成出来”的内容或质感。常见表现：\n"
    "1. 整个画面或大片区域呈“绘画感/涂抹感”：毛发、衣物、草木像被漩涡状笔触重画，表面像油画或蜡质；\n"
    "2. 人脸、皮肤过度平滑呈塑料/蜡像感，或五官、手指等结构被改变、融合；\n"
    "3. 画面中的小字幕、文字被重绘成伪字形（看起来像字但不是字）；\n"
    "4. 平坦区域（天空、墙面、背景）里出现不存在的波纹、涟漪、涂鸦状纹理或多余物体。\n"
    "【等级】P0：整幅或大面积呈绘画感/生成感，一眼可见；P1：局部明显但主体仍可辨认；P2：轻微、局部、需要仔细看；无问题：没有以上现象。\n"
    "【注意】只比较「过度生成」。不要把单纯的模糊（整体发虚）、纹理被抹平但没有新内容出现、过锐（白边/振铃）当作过度生成。")


def head(variant):
    return f"请比较「图片一」和「图片二」在「{DEFECT}」维度上的严重程度。\n\n{DEFINITION}\n"


def tail(variant):
    q = f"\n哪一张图片的「{DEFECT}」问题更严重？\nA. 图片一\nB. 图片二\n\n"
    if variant == "v4":
        return q + "请先分别用一两句话描述两张图片中与「过度生成」相关的现象，最后单独一行写“答案：A”或“答案：B”。"
    return q + "请只输出一个字母（A、B），不要解释。"


def textured_crop(name, size=448):
    """Native-resolution crop around the block with the highest gradient energy (full frame if smaller than size)."""
    im = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
    w, h = im.size
    if w <= size or h <= size:
        return im
    g = np.asarray(im.convert("L"), dtype=np.float32)
    gx, gy = np.abs(np.diff(g, axis=1))[:-1], np.abs(np.diff(g, axis=0))[:, :-1]
    e = (gx + gy)
    step = size // 2
    best, bxy = -1.0, (0, 0)
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            v = float(e[y:y + size, x:x + size].mean())
            if v > best:
                best, bxy = v, (x, y)
    x, y = bxy
    return im.crop((x, y, x + size, y + size))


def crop_url(name, size=448):
    import base64
    import io
    buf = io.BytesIO()
    textured_crop(name, size).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def select_tasks():
    pairs, _ = build_tasks()
    cells = {}
    for p in pairs:
        if p["defect"] == DEFECT:
            cells.setdefault(p["hi"], []).append(p)
    held = [p for hi in (3, 2, 1) for p in cells[hi][:60]]
    held_imgs = {p["img_hi"] for p in held} | {p["img_lo"] for p in held}
    # anchors: one frame per grade from videos that appear in no held-out pair; ensemble (out of fold) agrees with the label
    names, gold, grp, o, DIMS = load_labels()
    pool = np.load(POOL, allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(pool["names"])}
    P = pool["P"].astype(np.float64)
    P = P / P.sum(-1, keepdims=True)
    d = DIMS.index(DEFECT)
    import random
    rs = random.Random(7)
    anchors = []
    for g in (3, 2, 1, 0):
        cand = [i for i in range(len(names)) if o[i] and gold[i, d] == g and names[i] not in held_imgs]
        cand.sort(key=lambda i: -P[pos[names[i]], d, g])                 # frames the ensemble (out of fold) is most sure of
        anchors.append((names[rs.choice(cand[:10])], g))
    anchor_imgs = {n for n, _ in anchors}
    dev = []
    for hi in (3, 2, 1):
        k = 0
        for p in cells[hi][60:]:
            if p["img_hi"] in held_imgs or p["img_lo"] in held_imgs or p["img_hi"] in anchor_imgs or p["img_lo"] in anchor_imgs:
                continue
            dev.append(p)
            k += 1
            if k == 34:
                break
    return dev, held, anchors


def content(variant, p, first, anchors):
    a, b = (p["img_hi"], p["img_lo"]) if first == "hi" else (p["img_lo"], p["img_hi"])
    img = lambda n: {"type": "image_url", "image_url": {"url": data_url(n, 0)}}
    cr = lambda n: {"type": "image_url", "image_url": {"url": crop_url(n)}}
    c = []
    if variant == "v2":
        gn = {3: "P0", 2: "P1", 1: "P2", 0: "无问题"}
        c.append({"type": "text", "text": f"下面先给出四张已由标注员标注的「{DEFECT}」参考图片，供你校准标准：\n"})
        for k, (n, g) in enumerate(anchors):
            c += [{"type": "text", "text": f"参考{k + 1}（{DEFECT}等级：{gn[g]}）："}, img(n)]
        c.append({"type": "text", "text": "\n现在比较下面两张待判断图片。\n"})
    if variant == "v3":
        c += [{"type": "text", "text": "图片一（整幅）："}, img(a), {"type": "text", "text": "图片一（原始分辨率局部）："}, cr(a),
              {"type": "text", "text": "图片二（整幅）："}, img(b), {"type": "text", "text": "图片二（原始分辨率局部）："}, cr(b)]
    else:
        c += [{"type": "text", "text": "图片一："}, img(a), {"type": "text", "text": "图片二："}, img(b)]
    c.append({"type": "text", "text": head(variant) + tail(variant)})
    return c


def parse(text, variant):
    if variant == "v4":
        m = re.findall(r"答案[:：]\s*\**\s*([AB])", text)
        if m:
            return m[-1]
        m = re.findall(r"\b([AB])\b", text[-40:])
        return m[-1] if m else None
    for ch in text:
        if ch in "AB":
            return ch
    return None


def post(a, key, session, body):
    err = ""
    for attempt in range(a.retries):
        try:
            r = session.post(a.url.rstrip("/") + "/chat/completions", json=body, timeout=a.timeout, headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 200:
                js = r.json()
                return (js["choices"][0]["message"].get("content") or "").strip(), js.get("usage") or {}, ""
            err = f"HTTP {r.status_code}"
        except Exception as e:
            err = type(e).__name__
        time.sleep(min(30, 2 ** attempt))
    return "", {}, err


def run_variant(a, key, variant, pairs, anchors, out_path):
    session = requests.Session()
    session.trust_env = False
    items = [(p, first) for p in pairs for first in ("hi", "lo")]
    usage = {"prompt": 0, "completion": 0}

    def work(it):
        p, first = it
        body = {"model": a.model, "temperature": 0, "max_tokens": 2048 if variant == "v4" else 64,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content(variant, p, first, anchors)}]}
        text, u, err = post(a, key, session, body)
        usage["prompt"] += int(u.get("prompt_tokens") or 0)
        usage["completion"] += int(u.get("completion_tokens") or 0)
        L = parse(text, variant) if text else None
        probs = {"A": 0.5, "B": 0.5} if L is None else {"A": float(L == "A"), "B": float(L == "B")}
        return dict(p, first=first, variant=variant, probs=probs, valid=L is not None, raw=(err or text[-60:]))
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, open(out_path, "w", encoding="utf-8") as f:
        rows = []
        for k, r in enumerate(ex.map(work, items)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows.append(r)
            if (k + 1) % 50 == 0:
                print(f"  {variant}: {k + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {variant}: {len(rows)} requests, {sum(not r['valid'] for r in rows)} invalid, {time.time() - t0:.0f}s, tokens prompt {usage['prompt']} completion {usage['completion']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "heldout"], required=True)
    ap.add_argument("--variant", nargs="+", default=["v1", "v2", "v3", "v4"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results/r55/explore4/vlm_pilot"))
    ap.add_argument("--yes-send-frames", action="store_true")
    a = ap.parse_args()
    dev, held, anchors = select_tasks()
    pairs = dev if a.stage == "dev" else held
    print(f"stage {a.stage}: {len(pairs)} pairs x 2 orders per variant; variants {a.variant}; anchors {[(n, g) for n, g in anchors]}", flush=True)
    if not a.yes_send_frames:
        print("dry run: nothing sent")
        return
    key = os.environ["JUDGE_API_KEY"]
    for v in a.variant:
        run_variant(a, key, v, pairs, anchors, os.path.join(a.out_dir, f"pv_{a.stage}_{v}_{a.name}.jsonl"))


if __name__ == "__main__":
    main()
