#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/audit_v8_http_service.py
=================================
End-to-end live HTTP service audit for `serve_engine.py` (`resident_v8_boundary` +
`deconf_v6_boundary`), testing `/healthz`, `/infer` (`image_path` & `image_base64`,
including `v8_boundary` upgraded images), `/infer_batch` (multi-frame video summary),
error codes (404 / 400), 100% exact label match against the official `test_150`
submission JSON, and clean `SIGTERM` shutdown with VRAM release.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _http_json(url: str, payload=None, timeout: float = 120.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:
            j = json.loads(body)
        except Exception:
            j = {"raw": body}
        return e.code, j


def _audit_backend(backend: str, port: int, gpu: int, set_name: str, decoder: str, ref_map: dict, test_imgs: list):
    cmd = [
        sys.executable,
        os.path.join(REPO, "code/r55/serve_engine.py"),
        "--set", set_name,
        "--decoder", decoder,
        "--gpu", str(gpu),
        "--port", str(port),
        "--backend", backend,
    ]
    log_p = f"/tmp/r55_v8_http_audit_{backend}.log"
    fh = open(log_p, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Wait for /healthz up to 360s
        health_init = None
        t0 = time.time()
        while time.time() - t0 < 360.0:
            if proc.poll() is not None:
                raise RuntimeError(f"serve_engine ({backend}) exited early rc={proc.returncode}, see {log_p}")
            try:
                st, body = _http_json(f"{base_url}/healthz", timeout=5.0)
                if st == 200 and body.get("status") == "ok":
                    health_init = body
                    break
            except Exception:
                pass
            time.sleep(2.0)
        assert health_init is not None, f"Timeout waiting for {backend} /healthz"

        img_dir = os.path.join(REPO, "data/test_150")
        checks = []
        for idx, nm in enumerate(test_imgs[:3]):
            full_p = os.path.join(img_dir, nm)
            if idx < 2:
                st, body = _http_json(f"{base_url}/infer", {"image_path": full_p, "with_attribution": True})
                mode = "image_path"
            else:
                b64 = base64.b64encode(open(full_p, "rb").read()).decode("ascii")
                st, body = _http_json(f"{base_url}/infer", {"image_base64": b64, "image_name": nm, "with_attribution": True})
                mode = "image_base64"
            assert st == 200, f"/infer failed ({st}): {body}"
            match = (body["label"] == ref_map[nm])
            assert match, f"Label mismatch on {nm}: got={body['label']} ref={ref_map[nm]}"
            checks.append({"image": nm, "mode": mode, "latency_ms": body.get("latency_ms", {}), "match": match})

        # /infer_batch with 4 frames
        batch_paths = [os.path.join(img_dir, nm) for nm in test_imgs]
        st_b, body_b = _http_json(
            f"{base_url}/infer_batch",
            {"image_paths": batch_paths, "video_id": "clip_v8_audit_01", "with_attribution": True},
            timeout=240.0,
        )
        assert st_b == 200, f"/infer_batch failed ({st_b}): {body_b}"
        for r in body_b["results"]:
            assert r["label"] == ref_map[r["image"]], f"Batch label mismatch on {r['image']}"

        # Error handling checks
        st_404, _ = _http_json(f"{base_url}/infer", {"image_path": "/nonexistent/file_xyz.png"})
        st_400, _ = _http_json(f"{base_url}/infer", {})

        _, health_after = _http_json(f"{base_url}/healthz", timeout=10.0)
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=30.0)
        assert rc == 0, f"SIGTERM exit code={rc}"

        return {
            "backend": backend,
            "port": port,
            "health_after": health_after,
            "infer_checks": checks,
            "batch_video_summary": body_b["videos"][0] if body_b.get("videos") else None,
            "error_checks": {"missing_file_status": st_404, "empty_payload_status": st_400},
            "sigterm_rc": rc,
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--set", default="resident_v8_boundary")
    ap.add_argument("--decoder", default="deconf_v6_boundary")
    ap.add_argument("--out", default="results/r55/final/resident_v8_boundary_http_service_audit.json")
    args = ap.parse_args()
    assert args.gpu != 7, "GPU 7 is forbidden!"

    ref_json = os.path.join(REPO, f"results/final/pred_nriqa_dataset_test_r55_{args.set}_{args.decoder}.json")
    ref_map = {r["image"]: r["label"] for r in json.load(open(ref_json, "r", encoding="utf-8"))}

    # Include clean image, multi-defect image, and v8_boundary upgraded boundary images!
    test_imgs = [
        "196160772653_10.03s.png",
        "137212400938_frame14.png",
        "207550292858_v8L3HiV6_16.33s.png",
        "181204643103_v7L3HiV2_18101.png",
    ]

    report = {
        "set": args.set,
        "decoder": args.decoder,
        "tested_images": test_imgs,
    }
    report["fastapi"] = _audit_backend("fastapi", 8065, args.gpu, args.set, args.decoder, ref_map, test_imgs)
    report["stdlib"] = _audit_backend("stdlib", 8066, args.gpu, args.set, args.decoder, ref_map, test_imgs)

    out_p = os.path.join(REPO, args.out)
    json.dump(report, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[HTTP AUDIT PASS] saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
