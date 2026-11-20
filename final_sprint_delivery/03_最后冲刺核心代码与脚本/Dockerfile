# ==============================================================================
# 2026 乘风挑战赛·命题18：晴羽观感体系·画质BadCase无参检测模型
# Production Air-Gapped Inference & HTTP Microservice Container
# Default target: resident_v8_boundary + deconf_v6_boundary (Single RTX 4090 24GB)
# ==============================================================================
# Build (online or offline wheelhouse):
#   docker build -t topic18-nriqa:v8 .
#   docker build --build-arg PIP_EXTRA_ARGS="--no-index --find-links=/wheels" -t topic18-nriqa:v8 .
#
# Run pre-flight air-gapped verification (65 checks, 0 FAIL / 0 WARN):
#   docker run --rm --gpus '"device=0"' \
#     -v /path/to/models:/app/models:ro \
#     -v /path/to/checkpoints:/app/results:ro \
#     topic18-nriqa:v8 python3 code/r55/check_deploy_env.py --set resident_v8_boundary --decoder deconf_v6_boundary --gpu 0
#
# Run batch evaluation inference (N=1 == N=Batch, zero training-data reads):
#   docker run --rm --gpus '"device=0"' \
#     -v /path/to/models:/app/models:ro \
#     -v /path/to/eval_imgs:/data/eval_imgs:ro \
#     -v /path/to/out:/out \
#     topic18-nriqa:v8 bash run_final_infer.sh /out/template.json /data/eval_imgs /out/pred.json 0
#
# Run resident HTTP microservice (/healthz, /infer, /infer_batch + 极目/TAT payload):
#   docker run -d --gpus '"device=0"' -p 8055:8055 \
#     -v /path/to/models:/app/models:ro \
#     topic18-nriqa:v8
# ==============================================================================

ARG BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG PIP_EXTRA_ARGS=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    R55_SET=resident_v8_boundary \
    R55_DECODER=deconf_v6_boundary \
    R55_PORT=8055

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned inference-only dependencies (bitsandbytes is vendored in third_party/pydeps)
COPY requirements_infer.txt /app/requirements_infer.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      --index-url "${PIP_INDEX_URL}" \
      --extra-index-url "${TORCH_INDEX_URL}" \
      ${PIP_EXTRA_ARGS} \
      -r /app/requirements_infer.txt

# Copy inference code, vendored bitsandbytes, and lightweight decoder / footprint / video-aggregator artifacts.
# Large local base weights (`models/`) and full-fit checkpoints (`results/`) can either be mounted at runtime
# or baked into a standalone air-gapped image by copying them into `/app/models` and `/app/results`.
COPY code /app/code
COPY third_party/pydeps /app/third_party/pydeps
COPY run_final_infer.sh /app/run_final_infer.sh
COPY results/r55/final/resident_v8_boundary_decoder_deconf_v6_boundary.pkl /app/results/r55/final/
COPY results/r55/final/resident_v8_boundary_clean_footprint.npz /app/results/r55/final/
COPY results/r55/final/resident_v8_boundary_video_aggregator.pkl /app/results/r55/final/
RUN chmod +x /app/run_final_infer.sh && mkdir -p /app/models /tmp/r55_spool

EXPOSE 8055

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${R55_PORT}/healthz" || exit 1

CMD ["sh", "-c", "python3 -u code/r55/serve_engine.py --set ${R55_SET} --decoder ${R55_DECODER} --gpu 0 --host 0.0.0.0 --port ${R55_PORT}"]
