"""Judge / in-context grading pilot against a model behind an HTTP interface (closed-source or hosted).

SENDS LABELLED FRAMES TO THE ENDPOINT.  Run it only on an endpoint the project owner has provided and approved for
this data.  Nothing is sent unless --url, --model and --yes-send-frames are all given.

Same tasks as run_vllm_judge_pilot.py (the first k pairs / queries of every cell of its default lists), so the result
is directly comparable with the local 27B model and with the ensemble's out-of-fold predictions:
    python code/decision_eval/eval_vllm_judge_pilot.py --name <name> --pool <pool.npz>

Protocol: OpenAI-compatible chat completions (POST <url>/chat/completions, images as base64 data URLs).  If the
endpoint returns token log-probabilities they are used; otherwise the answer letter becomes a one-hot probability
(for pairs the two presentation orders still give a three-level score).  The API key is read from the environment
variable named by --key-env and is never written to disk or logs.

Usage:
  export JUDGE_API_KEY=...
  python code/decision_eval/run_http_judge_pilot.py --url https://host/v1 --model <model id> --name <short name> \
      --pairs 60 --queries 25 --yes-send-frames
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from common import SYSTEM  # noqa: E402
from run_vllm_judge_pilot import build_tasks, grade_question, pair_question, subset  # noqa: E402

IMG_DIR = os.path.join(REPO, "data/images")
USAGE = {"prompt": 0, "completion": 0, "calls": 0}
SESSION = None


def data_url(name, max_side):
    im = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")                                    # lossless: the evidence is fine texture
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(a, key, content, letters):
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    if a.logprobs:
        body.update(logprobs=True, top_logprobs=10)
    if a.extra_json:
        body.update(json.loads(a.extra_json))
    err = ""
    for attempt in range(a.retries):
        try:
            r = SESSION.post(a.url.rstrip("/") + "/chat/completions", json=body, timeout=a.timeout,
                             headers={"Authorization": f"Bearer {key}"} if key else {})
            if r.status_code == 200:
                js = r.json()
                ch = js["choices"][0]
                text = (ch["message"].get("content") or "").strip()
                u = js.get("usage") or {}
                USAGE["prompt"] += int(u.get("prompt_tokens") or 0)
                USAGE["completion"] += int(u.get("completion_tokens") or 0)
                USAGE["calls"] += 1
                if not text:                                     # e.g. the token budget went to reasoning
                    err = f"empty answer ({ch.get('finish_reason')})"
                    time.sleep(1)
                    continue
                p = None
                try:
                    for tokinfo in ch["logprobs"]["content"]:
                        if tokinfo["token"].strip() in letters:
                            lp = {t["token"].strip(): t["logprob"] for t in tokinfo["top_logprobs"] if t["token"].strip() in letters}
                            m = max(lp.values())
                            z = sum(np.exp(v - m) for v in lp.values())
                            p = {L: float(np.exp(lp.get(L, -1e9) - m) / z) for L in letters}
                            break
                except Exception:
                    p = None
                if p is None:
                    first = next((ch_ for ch_ in text if ch_ in letters), None)
                    if first is None:
                        return {L: 1.0 / len(letters) for L in letters}, False, text[:40]
                    p = {L: float(L == first) for L in letters}
                return p, True, text[:40]
            err = f"HTTP {r.status_code}"
        except Exception as e:                                   # network error, time-out
            err = type(e).__name__
        time.sleep(min(30, 2 ** attempt))
    return {L: 1.0 / len(letters) for L in letters}, False, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="base URL of an OpenAI-compatible API, e.g. https://host/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--name", default="", help="short tag for the output files")
    ap.add_argument("--key-env", default="JUDGE_API_KEY")
    ap.add_argument("--pairs", type=int, default=60, help="pairs per defect and contrast (first k of the 150 of the local pilot)")
    ap.add_argument("--queries", type=int, default=25, help="query frames per defect and grade (first k of 50)")
    ap.add_argument("--max-side", type=int, default=0, help="0 = send frames at native size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8, help="reasoning models need room for their reasoning tokens (e.g. 4096)")
    ap.add_argument("--defects", nargs="+", default=[], help="restrict to these defects (default: all four graded ones)")
    ap.add_argument("--extra-json", default="", help='extra request fields as JSON, e.g. \'{"reasoning_effort": "low"}\'')
    ap.add_argument("--no-proxy", action="store_true", help="ignore proxy environment variables (internal gateways)")
    ap.add_argument("--logprobs", action="store_true", help="ask for token log-probabilities (not every provider supports it)")
    ap.add_argument("--parts", nargs="+", default=["pairs", "grade"], choices=["pairs", "grade"])
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results/r55/explore4/vlm_pilot"))
    ap.add_argument("--yes-send-frames", action="store_true", help="required: confirms the owner approved sending frames to --url")
    a = ap.parse_args()
    pairs, queries = subset(*build_tasks(), a.pairs, a.queries)
    if a.defects:
        pairs = [p for p in pairs if p["defect"] in a.defects]
        queries = [q for q in queries if q["defect"] in a.defects]
    n_req = (2 * len(pairs) if "pairs" in a.parts else 0) + (2 * len(queries) if "grade" in a.parts else 0)
    n_img = (4 * len(pairs) if "pairs" in a.parts else 0) + (6 * len(queries) if "grade" in a.parts else 0)
    print(f"plan: {len(pairs)} pairs x 2 orders, {len(queries)} grade queries x 2 modes -> {n_req} requests, {n_img} image uploads", flush=True)
    if not (a.url and a.model and a.name and a.yes_send_frames):
        print("dry run: nothing was sent (give --url, --model, --name and --yes-send-frames to run)")
        return
    key = os.environ.get(a.key_env, "")
    os.makedirs(a.out_dir, exist_ok=True)
    import requests
    global SESSION
    SESSION = requests.Session()
    if a.no_proxy:
        SESSION.trust_env = False

    def run(items, letters, out_path):
        t0 = time.time()

        def work(it):
            rec, content = it
            p, ok, raw = ask(a, key, content(), letters)
            return dict(rec, probs=p, valid=ok, raw=raw)
        with ThreadPoolExecutor(a.workers) as ex, open(out_path, "w", encoding="utf-8") as f:
            for k, r in enumerate(ex.map(work, items)):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if (k + 1) % 50 == 0:
                    print(f"  {os.path.basename(out_path)}: {k + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        print(f"done {out_path}: {len(rows)} requests, {sum(not r['valid'] for r in rows)} without a valid answer, {time.time() - t0:.0f}s | "
              f"tokens so far: prompt {USAGE['prompt']}, completion {USAGE['completion']} over {USAGE['calls']} calls", flush=True)

    def im(n):
        return {"type": "image_url", "image_url": {"url": data_url(n, a.max_side)}}

    if "pairs" in a.parts:
        items = []
        for p in pairs:
            for first in ("hi", "lo"):
                x, y = (p["img_hi"], p["img_lo"]) if first == "hi" else (p["img_lo"], p["img_hi"])
                items.append((dict(p, first=first), lambda x=x, y=y, d=p["defect"]: [
                    {"type": "text", "text": "图片一："}, im(x), {"type": "text", "text": "图片二："}, im(y), {"type": "text", "text": pair_question(d)}]))
        run(items, "AB", os.path.join(a.out_dir, f"pairs_{a.name}.jsonl"))
    if "grade" in a.parts:
        items = []
        for q in queries:
            items.append((dict(q, mode="zero"), lambda q=q: [im(q["image"]), {"type": "text", "text": grade_question(q["defect"], q["options"], False)}]))

            def few(q=q):
                c = []
                for k, (n, g) in enumerate(q["exemplars"]):
                    c += [{"type": "text", "text": f"示例{k + 1}（「{q['defect']}」等级：{g}）："}, im(n)]
                return c + [{"type": "text", "text": "待判断图片："}, im(q["image"]), {"type": "text", "text": grade_question(q["defect"], q["options"], True)}]
            items.append((dict(q, mode="few"), few))
        run(items, "ABCD", os.path.join(a.out_dir, f"grade_{a.name}.jsonl"))


if __name__ == "__main__":
    main()
