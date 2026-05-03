# syntax=docker/dockerfile:1.7
#
# turbocpp Docker image — same workflow accessibility as llama.cpp,
# but everything is installed from PREBUILT WHEELS hosted at
# https://huggingface.co/datasets/AIencoder/TurboCpp_Wheels.
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

# Both llama-cpp-python and turbocpp ship prebuilt wheels on PyPI.
# We pin llama-cpp-python to a version with a known cp312 wheel and add
# --only-binary=llama-cpp-python so pip will FAIL LOUDLY rather than
# silently fall back to a source build that needs cmake/g++ (which the
# slim base image deliberately does not have).
#
# Override either with --build-arg to use a CPU-feature-tuned wheel:
#   docker build --build-arg LLAMA_CPP_PKG="$(turbocpp pick-wheel)" .
ARG LLAMA_CPP_PKG="llama-cpp-python==0.3.16"
ARG TURBOCPP_PKG="turbocpp"

RUN pip install --only-binary=llama-cpp-python \
        "${LLAMA_CPP_PKG}" \
        "${TURBOCPP_PKG}" \
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

# Torch + transformers + co. --only-binary=:all: forces every package to
# come from a wheel; combined with the cpu stage's llama-cpp-python pin
# this prevents pip's resolver from upgrading + source-rebuilding it.
# Use the official PyTorch CPU index so the torch wheel is the slim
# (~200 MB) CPU build, not the 2 GB CUDA one.
RUN pip install --only-binary=:all: \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.0" \
        "transformers>=4.40" \
        "safetensors>=0.4" \
        "numpy>=1.24"

ENTRYPOINT ["turbocpp"]
CMD ["rotate", "--help"]
