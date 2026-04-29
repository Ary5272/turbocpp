# syntax=docker/dockerfile:1.7
#
# turbocpp Docker image — same workflow accessibility as llama.cpp,
# but everything is installed from PREBUILT WHEELS hosted at
# https://huggingface.co/datasets/AIencoder/llama-cpp-wheels.
#
# No C++ compile step → image build is ~30 seconds instead of ~10 minutes,
# and the image works on any x86_64 host with AVX2 + FMA + F16C
# (effectively every cloud / CI / consumer CPU made after 2013).
#
# Targets:
#   :cpu        full turbocpp toolchain — `turbocpp generate|serve|rotate|bench`,
#               plus `convert_hf_to_gguf.py` from llama.cpp's Python utils.
#   :server     inherits cpu; CMD launches the OpenAI-compatible HTTP server
#               on :8080 (via llama-cpp-python's built-in FastAPI app).
#   :turboquant inherits cpu; adds torch + transformers so `turbocpp rotate`
#               actually runs on a HuggingFace model checkpoint.
#
# Build:
#   docker build --target cpu        -t turbocpp:cpu        .
#   docker build --target server     -t turbocpp:server     .
#   docker build --target turboquant -t turbocpp:turboquant .
#
# Run examples:
#   # one-shot inference
#   docker run --rm -v ~/models:/models turbocpp:cpu \
#          generate -m /models/m.gguf -p "Hello" -n 64
#
#   # OpenAI-compat server
#   docker run --rm -p 8080:8080 -v ~/models:/models turbocpp:server \
#          -m /models/m.gguf
#
#   # offline rotation pipeline
#   docker run --rm -v ~/models:/models turbocpp:turboquant \
#          rotate /models/Llama-3-8B /models/Llama-3-8B-tq

ARG PY=3.12

# ============================================================================
# Stage: base — Python + a couple of small native deps llama-cpp-python needs
# ============================================================================
FROM python:${PY}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip

# ============================================================================
# Stage: cpu — turbocpp + llama-cpp-python via prebuilt wheels
# ============================================================================
FROM base AS cpu

# Pinned wheel URLs. Override with --build-arg if you want a different
# CPU feature set (avx512, vnni, amx, …) from AIencoder/llama-cpp-wheels.
ARG LLAMA_CPP_WHEEL_URL=https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/llama_cpp_python-0.3.16%2Bbasic_avx2_fma_f16c-cp312-cp312-manylinux_2_31_x86_64.whl
ARG TURBOCPP_WHEEL_URL=https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/turbocpp/turbocpp-0.3.0-py3-none-any.whl

RUN pip install \
        "${LLAMA_CPP_WHEEL_URL}" \
        "${TURBOCPP_WHEEL_URL}" \
        "huggingface_hub>=0.24,<1.0" \
        "gguf>=0.10"

# Sanity check: the unified CLI is reachable.
RUN turbocpp --help >/dev/null

WORKDIR /work
ENTRYPOINT ["turbocpp"]
CMD ["--help"]

# ============================================================================
# Stage: server — OpenAI-compatible HTTP API on :8080
# ============================================================================
FROM cpu AS server

# llama-cpp-python's built-in server uses FastAPI + Uvicorn.
RUN pip install \
        "uvicorn[standard]>=0.30" \
        "fastapi>=0.110" \
        "sse-starlette>=1.8" \
        "starlette-context>=0.3.6" \
        "pydantic-settings>=2.0"

EXPOSE 8080
ENTRYPOINT ["turbocpp", "serve", "--host", "0.0.0.0", "--port", "8080"]
CMD []

# ============================================================================
# Stage: turboquant — adds the offline rotation pipeline (torch+transformers)
# ============================================================================
FROM cpu AS turboquant

# CPU-only torch wheel keeps the image around 2 GB instead of 7 GB.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.0" \
        "transformers>=4.40" \
        "safetensors>=0.4" \
        "numpy>=1.24"

ENTRYPOINT ["turbocpp"]
CMD ["rotate", "--help"]
