#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/serve_engine.py — online + batch HTTP service for the resident engine (default: resident_v6_core,
RoutedGateEUDecoder).  All members are loaded ONCE at startup and stay resident on one GPU; every request is
served image-by-image through exactly the same N = 1 path as `final_engine.py` (FinalEngine.infer ->
row-wise decoder -> AttributionRuntime), so a label returned here equals the label in the submission JSON.

Endpoints
  GET  /healthz       status, set, members, decoder, load_s, torch_peak_gb, nvidia_smi_mib (current / peak), requests
  POST /infer         {"image_path": "/abs/x.png"} | {"image_base64": "...", "image_name": "x.png"}
                      [+ "with_attribution": true]  -> label, probabilities, measured-evidence attribution,
                      tat_jimu_payload, latency_ms
  POST /infer_batch   {"image_paths": [...], "video_id": optional, "with_attribution": true}
                      -> per-image records + video-level EffectiveFrameORVideoAggregator summary per clip
                      (clips grouped by `video_id` if given, else by `infer_video_group_id(file name)`)

Transport: FastAPI + uvicorn when importable, else stdlib `http.server.ThreadingHTTPServer` (zero extra deps);
force with `--backend stdlib|fastapi`.  GPU work is serialised by one lock (one resident model copy).
Incoming images are staged under a unique name in `--spool-dir` (symlink for paths, file for base64) so they can
never collide with a cached training frame of the same name, and are removed after the request.

Usage
  python3 code/r55/serve_engine.py --set resident_v6_core --gpu 0 --port 8055
  curl -s localhost:8055/healthz
  curl -s -XPOST localhost:8055/infer -H 'Content-Type: application/json' -d '{"image_path": "/abs/path/img.png"}'
Bind address defaults to 127.0.0.1 (the /infer image_path mode reads server-side files); use --path-root to
restrict readable paths when exposing the port.
"""
import argparse
import base64
import io
import json
import os
import signal
import sys
import threading
import time
import uuid

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="resident_v6_core")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--decoder", default="routed_gate_eu", choices=["d55", "eu", "vote", "routed_eu", "routed_gate_eu"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8055)
    ap.add_argument("--spool-dir", default="/tmp/r55_serve_spool")
    ap.add_argument("--path-root", default="", help="if set, /infer image_path must lie under this directory")
    ap.add_argument("--no-attribution", action="store_true")
    ap.add_argument("--no-half", action="store_true", help="fp32 backbone weights (default: bf16 weights, like the submission run)")
    ap.add_argument("--backend", default="auto", choices=["auto", "fastapi", "stdlib"])
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--no-warmup", action="store_true", help="skip the start-up warm-up on a synthetic image")
    return ap.parse_args()


A = _parse()
os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)
os.makedirs(A.spool_dir, exist_ok=True)
os.environ["NEXT55_EXTRA_IMG_DIR"] = A.spool_dir          # read at import by the V16 feature extractor
sys.path.insert(0, os.path.join(REPO, "code", "r55"))

import numpy as np  # noqa: E402

import final_engine as fe  # noqa: E402  (sets PYTORCH_CUDA_ALLOC_CONF before torch is imported)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return round(o, 6)
    return o


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


class Service:
    def __init__(self, a):
        import pickle
        import torch
        self.a, self.torch = a, torch
        self.members = fe.SETS[a.set]
        half = not a.no_half
        v16_chunk = fe.V16_CHUNK_BY_SET.get(a.set, 0) if half else 0
        torch.cuda.init()
        self.nv = fe.NvmlPeakSampler()
        t0 = time.time()
        self.eng = fe.FinalEngine(self.members, img_dir=a.spool_dir, half=half, v16_chunk=v16_chunk)
        self.peak_load_gb = torch.cuda.max_memory_allocated() / 2 ** 30
        self.svc = fe.N1Service(self.eng, a.set, rule=a.decoder, attribution=not a.no_attribution)
        vap = os.path.join(fe.OUT, f"{a.set}_video_aggregator.pkl")
        from subjective_engineering import EffectiveFrameORVideoAggregator
        self.vid_agg = pickle.load(open(vap, "rb"))["video_aggregator"] if os.path.exists(vap) else EffectiveFrameORVideoAggregator()
        self.vid_agg_src = vap if os.path.exists(vap) else "default (no fitted aggregator found)"
        self.lock = threading.Lock()
        self.warmup_s = None
        if not a.no_warmup:
            self.warmup_s = self._warmup()
        self.load_s = time.time() - t0
        self.n_req = 0
        self.started = time.time()
        print(f"[serve] {a.set} {self.members} decoder={a.decoder} loaded in {self.load_s:.0f}s | torch peak "
              f"{self.peak_load_gb:.2f} GB | nvidia-smi {self.nv.read_mib():.0f} MiB", flush=True)

    def _warmup(self):
        """One full N = 1 pass (incl. attribution) on a synthetic 1080p image so the first real request does not pay
        cuDNN autotuning / lazy-init cost.  Synthetic pixels only: no training or test data is touched."""
        from PIL import Image
        rs = np.random.RandomState(0)
        yy, xx = np.mgrid[0:1080, 0:1920]
        img = np.clip(127 + 60 * np.sin(xx / 37.0) * np.cos(yy / 23.0) + rs.normal(0, 12, (1080, 1920)), 0, 255)
        nm = f"srv_warmup_{uuid.uuid4().hex[:8]}.png"
        dst = os.path.join(self.a.spool_dir, nm)
        Image.fromarray(np.stack([img, np.roll(img, 7, 1), np.roll(img, 13, 0)], -1).astype(np.uint8)).save(dst)
        t = time.time()
        try:
            rec = self.svc.predict(nm, with_attribution=True)
            if self.svc.attr is not None and not np.any(rec["pred"] > 0):     # make sure the localisation path is warm too
                self.svc.attr.attribute(self.eng.loader.load_pil(nm).convert("RGB"), np.array([1, 0, 0, 0, 0, 0]), rec["probs"])
        finally:
            os.remove(dst)
        print(f"[serve] warm-up (synthetic image) {time.time() - t:.1f}s", flush=True)
        return time.time() - t

    # ------------------------------------------------------------------ helpers
    def _stage(self, item):
        """-> (spool name, original name, path to remove)."""
        if not isinstance(item, dict):
            raise ApiError(400, "each request item must be a JSON object")
        if item.get("image_path"):
            src = os.path.abspath(str(item["image_path"]))
            if self.a.path_root and not src.startswith(os.path.abspath(self.a.path_root) + os.sep):
                raise ApiError(403, f"image_path outside --path-root: {src}")
            if not os.path.isfile(src):
                raise ApiError(404, f"image_path not found: {src}")
            orig = os.path.basename(src)
            spool = f"srv_{uuid.uuid4().hex[:12]}_{orig}"
            dst = os.path.join(self.a.spool_dir, spool)
            os.symlink(src, dst)
        elif item.get("image_base64"):
            try:
                raw = base64.b64decode(str(item["image_base64"]), validate=False)
            except Exception as e:
                raise ApiError(400, f"bad base64: {e}")
            orig = os.path.basename(str(item.get("image_name") or "upload.png"))
            spool = f"srv_{uuid.uuid4().hex[:12]}_{orig}"
            dst = os.path.join(self.a.spool_dir, spool)
            with open(dst, "wb") as f:
                f.write(raw)
        else:
            raise ApiError(400, "need image_path or image_base64")
        try:
            from PIL import Image
            with Image.open(dst) as im:
                im.verify()
        except Exception as e:
            os.remove(dst)
            raise ApiError(400, f"not a decodable image: {e}")
        return spool, orig, dst

    def _one(self, item, with_attr):
        spool, orig, dst = self._stage(item)
        try:
            with self.lock:
                rec = self.svc.predict(spool, with_attribution=with_attr)
                self.n_req += 1
        finally:
            if os.path.lexists(dst):
                os.remove(dst)
        out = dict(image=orig, label=rec["label"], pred_6d=rec["pred"], probs_6x4=np.round(rec["probs"], 5),
                   latency_ms={k[:-2]: round(rec[k] * 1000.0, 2) for k in ("infer_s", "decode_s", "attr_s", "total_s")})
        if rec.get("attribution") is not None:
            out["attribution"] = rec["attribution"]
            pay = dict(rec["tat_jimu_payload"])
            pay["image"] = orig
            out["tat_jimu_payload"] = pay
        return out, rec

    # ------------------------------------------------------------------ endpoints
    def healthz(self):
        pk = max([self.peak_load_gb] + list(getattr(self.eng, "peak", {}).values()))
        return dict(status="ok", set=self.a.set, members=self.members, decoder=self.a.decoder,
                    attribution=not self.a.no_attribution, load_s=round(self.load_s, 2),
                    warmup_s=None if self.warmup_s is None else round(self.warmup_s, 2),
                    torch_peak_gb=round(float(pk), 3),
                    torch_allocated_gb=round(self.torch.cuda.memory_allocated() / 2 ** 30, 3),
                    torch_reserved_gb=round(self.torch.cuda.memory_reserved() / 2 ** 30, 3),
                    nvidia_smi_mib=round(self.nv.read_mib(), 1), nvidia_smi_peak_mib=round(self.nv.peak_mib, 1),
                    params_b=self.eng.total_params_b, requests=self.n_req, uptime_s=round(time.time() - self.started, 1),
                    clean_footprint=os.path.relpath(self.svc.attr.fp_path, fe.REPO) if self.svc.attr else None,
                    pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))

    def infer(self, body):
        t = time.perf_counter()
        out, _ = self._one(body, bool(body.get("with_attribution", True)))
        out["latency_ms"]["request"] = round((time.perf_counter() - t) * 1000.0, 2)
        return out

    def infer_batch(self, body):
        from qingyu.config_and_paths import DIMS, GRADE_INT_TO_STR
        from subjective_engineering import infer_video_group_id
        t = time.perf_counter()
        paths = body.get("image_paths")
        items = body.get("items")
        if items is None:
            if not isinstance(paths, list) or not paths:
                raise ApiError(400, "need non-empty image_paths list (or items list)")
            items = [{"image_path": p} for p in paths]
        if len(items) > self.a.max_batch:
            raise ApiError(413, f"batch larger than --max-batch {self.a.max_batch}")
        with_attr = bool(body.get("with_attribution", True))
        outs, qs = [], []
        for it in items:
            o, rec = self._one(it, with_attr)
            outs.append(o)
            qs.append(rec["probs"])
        groups = {}
        for i, o in enumerate(outs):
            gid = str(body["video_id"]) if body.get("video_id") else infer_video_group_id(o["image"])
            groups.setdefault(gid, []).append(i)
        videos = []
        for gid, ix in groups.items():
            res = self.vid_agg.aggregate_video(np.stack([qs[i] for i in ix]), self.svc.sf_decoder)
            vg = res["video_grade_6d"]
            videos.append(dict(video_id=gid, num_frames=len(ix), frames=[outs[i]["image"] for i in ix],
                               video_label={DIMS[d]: GRADE_INT_TO_STR[int(vg[d])] for d in range(6)},
                               trigger_frames=res["trigger_frames"], m_eff_p0=res["m_eff_p0"]))
        tot = [o["latency_ms"]["total"] for o in outs]
        return dict(n=len(outs), results=outs, videos=videos, video_aggregator=os.path.relpath(self.vid_agg_src, fe.REPO)
                    if os.path.isabs(self.vid_agg_src) else self.vid_agg_src,
                    latency_ms=dict(request=round((time.perf_counter() - t) * 1000.0, 2),
                                    per_image_mean=round(float(np.mean(tot)), 2), per_image_max=round(float(np.max(tot)), 2)))


def _run_fastapi(S):
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    app = FastAPI(title="r55 resident engine", version="1.0")

    def wrap(fn, *args):
        try:
            return JSONResponse(_jsonable(fn(*args)))
        except ApiError as e:
            return JSONResponse({"error": e.msg}, status_code=e.code)
        except Exception as e:  # pragma: no cover
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    @app.get("/healthz")
    def healthz():
        return wrap(S.healthz)

    @app.post("/infer")
    async def infer(req: Request):
        body = await req.json()
        import anyio
        return await anyio.to_thread.run_sync(wrap, S.infer, body)

    @app.post("/infer_batch")
    async def infer_batch(req: Request):
        body = await req.json()
        import anyio
        return await anyio.to_thread.run_sync(wrap, S.infer_batch, body)

    print(f"[serve] FastAPI/uvicorn on http://{A.host}:{A.port}", flush=True)
    # uvicorn shuts down gracefully on SIGTERM/SIGINT and then re-raises the signal to the previous handler;
    # a Python-level handler turns that into a normal exit (code 0) so the `finally` clean-up runs.
    for sg in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, lambda *_: sys.exit(0))
    uvicorn.run(app, host=A.host, port=A.port, log_level="warning", workers=1)


def _run_stdlib(S):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(_jsonable(obj), ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _call(self, fn, *args):
            try:
                self._send(200, fn(*args))
            except ApiError as e:
                self._send(e.code, {"error": e.msg})
            except Exception as e:  # pragma: no cover
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            if self.path.split("?")[0] == "/healthz":
                return self._call(S.healthz)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
            except Exception as e:
                return self._send(400, {"error": f"bad JSON: {e}"})
            p = self.path.split("?")[0]
            if p == "/infer":
                return self._call(S.infer, body)
            if p == "/infer_batch":
                return self._call(S.infer_batch, body)
            self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass

    srv = ThreadingHTTPServer((A.host, A.port), H)

    def stop(*_):
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"[serve] stdlib ThreadingHTTPServer on http://{A.host}:{A.port}", flush=True)
    srv.serve_forever()
    srv.server_close()


def main():
    S = Service(A)
    backend = A.backend
    if backend == "auto":
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
            backend = "fastapi"
        except Exception:
            backend = "stdlib"
    try:
        (_run_fastapi if backend == "fastapi" else _run_stdlib)(S)
    finally:
        S.nv.stop()
        print("[serve] shut down cleanly", flush=True)


if __name__ == "__main__":
    main()
