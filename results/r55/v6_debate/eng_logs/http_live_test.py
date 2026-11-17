#!/usr/bin/env python3
"""Live test of code/r55/serve_engine.py: start on GPU 0, exercise /healthz /infer /infer_batch, compare labels with
the submission JSON (fresh N=1 path must reproduce it), then shut the server down with SIGTERM and check the GPU."""
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "/wanqing-develop/luowenjing/18"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8055
BACKEND = sys.argv[2] if len(sys.argv) > 2 else "auto"
LOG = f"{REPO}/results/r55/v6_debate/eng_logs/serve_{BACKEND}.log"
OUT = f"{REPO}/results/r55/v6_debate/eng_runs/http_live_test_{BACKEND}.json"
REF = f"{REPO}/results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json"
IMG = f"{REPO}/data/test_150"
URL = f"http://127.0.0.1:{PORT}"


def call(path, body=None, timeout=600):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t) * 1000
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), (time.perf_counter() - t) * 1000


def gpu0_mib():
    return float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"]).decode().strip())


ref = {r["image"]: r["label"] for r in json.load(open(REF, encoding="utf-8"))}
names = list(ref)
sys.path.insert(0, f"{REPO}/code/r55")
from subjective_engineering import infer_video_group_id  # noqa: E402

groups = {}
for n in names:
    groups.setdefault(infer_video_group_id(n), []).append(n)
multi = sorted([g for g in groups.values() if len(g) >= 2], key=len, reverse=True)
clean = [n for n in names if all(v == "无问题" for v in ref[n].values())]
p0 = [n for n in names if any(v == "P0" for v in ref[n].values())]
single = [clean[0], p0[0], names[1], names[2]]
batch = (multi[0][:4] if multi else []) + (multi[1][:2] if len(multi) > 1 else []) + [clean[1]]

rep = dict(port=PORT, backend=BACKEND, started=time.strftime("%Y-%m-%d %H:%M:%S"))
srv = subprocess.Popen([sys.executable, "-u", f"{REPO}/code/r55/serve_engine.py", "--set", "resident_v6_core", "--gpu", "0",
                        "--port", str(PORT), "--backend", BACKEND], stdout=open(LOG, "w"), stderr=subprocess.STDOUT, cwd=REPO)
t0 = time.time()
while True:
    try:
        st, h, _ = call("/healthz", timeout=5)
        if st == 200:
            break
    except Exception:
        pass
    if srv.poll() is not None:
        raise SystemExit(f"server died, see {LOG}")
    time.sleep(3)
rep["ready_after_s"] = round(time.time() - t0, 1)
rep["healthz_start"] = h
print("healthz:", json.dumps(h, ensure_ascii=False))

res = []
for n in single:
    st, o, ms = call("/infer", {"image_path": f"{IMG}/{n}"})
    ok = st == 200 and o["label"] == ref[n]
    res.append(dict(image=n, mode="image_path", status=st, label_match=ok, latency_ms=o.get("latency_ms"), client_ms=round(ms, 1),
                    summary=(o.get("attribution") or {}).get("summary_40ch")))
    print(f"/infer path  {n}: {st} match={ok} {o.get('latency_ms')} | {res[-1]['summary']}")
    if n == p0[0]:
        rep["example_infer_response"] = o
n = p0[1]
b64 = base64.b64encode(open(f"{IMG}/{n}", "rb").read()).decode()
st, o, ms = call("/infer", {"image_base64": b64, "image_name": n})
res.append(dict(image=n, mode="image_base64", status=st, label_match=st == 200 and o["label"] == ref[n], latency_ms=o.get("latency_ms"),
                client_ms=round(ms, 1), summary=(o.get("attribution") or {}).get("summary_40ch")))
print(f"/infer b64   {n}: {st} match={res[-1]['label_match']} {o.get('latency_ms')}")
rep["infer"] = res

# (a) auto grouping by file name (test_150 frames come from distinct clips -> one video per frame)
# (b) explicit video_id: 6 frames treated as one clip -> EffectiveFrameORVideoAggregator with M = 6
rep["infer_batch"] = {}
for tag, body in (("auto_group", {"image_paths": [f"{IMG}/{x}" for x in names[3:7]]}),
                  ("video_id", {"image_paths": [f"{IMG}/{x}" for x in [p0[2]] + clean[2:4] + names[10:13]], "video_id": "demo_clip_M6"})):
    st, o, ms = call("/infer_batch", body)
    bm = [r["label"] == ref[r["image"]] for r in o.get("results", [])]
    rep["infer_batch"][tag] = dict(status=st, n=o.get("n"), label_match=f"{sum(bm)}/{len(bm)}", client_ms=round(ms, 1),
                                   latency_ms=o.get("latency_ms"), videos=o.get("videos"), video_aggregator=o.get("video_aggregator"))
    print(f"/infer_batch[{tag}] {o.get('n')} imgs: {st} labels match {sum(bm)}/{len(bm)} "
          f"videos={[(v['video_id'], v['num_frames'], v['video_label']) for v in o.get('videos', [])]}")

st, o, _ = call("/infer", {"image_path": f"{IMG}/does_not_exist.png"})
rep["error_missing_path"] = dict(status=st, body=o)
st, o, _ = call("/infer", {"foo": 1})
rep["error_bad_request"] = dict(status=st, body=o)
print("errors:", rep["error_missing_path"]["status"], rep["error_bad_request"]["status"])
st, h, _ = call("/healthz")
rep["healthz_end"] = h
print("healthz end:", json.dumps(h, ensure_ascii=False))
rep["spool_leftovers"] = os.listdir("/tmp/r55_serve_spool")

mib_before = gpu0_mib()
srv.send_signal(signal.SIGTERM)
try:
    rc = srv.wait(timeout=120)
except subprocess.TimeoutExpired:
    srv.kill()
    rc = "killed"
time.sleep(2)
rep["shutdown"] = dict(signal="SIGTERM", returncode=rc, gpu0_mib_before=mib_before, gpu0_mib_after=gpu0_mib(),
                       log_tail=open(LOG).read().strip().splitlines()[-3:])
print("shutdown:", rep["shutdown"])
json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=2)
print("->", OUT)
