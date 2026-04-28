# syntax=docker/dockerfile:1.7
#
# turbocpp Docker image. Multi-stage, multi-target — same accessibility
# story as llama.cpp's `:full`, `:server`, `:light` variants, plus a
# `:turboquant` target that includes the Python preprocessor.
#
# Targets:
#   cpu        ── llama-cli, llama-server, llama-quantize, llama-bench, …
#                 (the standard llama.cpp toolchain). Default CMD: llama-cli.
#   server     ── inherits cpu; CMD launches llama-server on :8080.
#   turboquant ── inherits cpu; adds Python turboquant + torch/transformers
#                 so you can `python -m turboquant <hf_dir> <out_dir>`.
#
# Build:
#   docker build --target cpu        -t turbocpp:cpu        .
#   docker build --target server     -t turbocpp:server     .
#   docker build --target turboquant -t turbocpp:turboquant .
#
# Run:
#   docker run --rm -v $PWD/models:/models turbocpp:cpu \
#     llama-cli -m /models/model.gguf -p "Hello"
#
#   docker run --rm -p 8080:8080 -v $PWD/models:/models turbocpp:server \
#     -m /models/model.gguf
#
#   docker run --rm -v $PWD/models:/models turbocpp:turboquant \
#     python -m turboquant /models/Llama-3-8B /models/Llama-3-8B-tq
#
# CPU portability: built with -DGGML_NATIVE=OFF and an explicit AVX2/FMA/F16C
# baseline → runs on any x86_64 from 2013 onward. Override at build time:
#   docker build --build-arg LLAMA_CMAKE_FLAGS="-DGGML_AVX512=ON" ...
# ─────────────────────────────────────────────────────────────────────────────

ARG UBUNTU_VERSION=24.04

# ============================================================================
# Stage: base — minimal runtime deps shared by every final image
# ============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libcurl4 \
        libgomp1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage: builder — compile llama.cpp once, share artifacts across targets
# ============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libcurl4-openssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CMAKE_FLAGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"

WORKDIR /src
COPY llama.cpp /src/llama.cpp

RUN cmake -S /src/llama.cpp -B /src/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        ${LLAMA_CMAKE_FLAGS} \
    && cmake --build /src/build --config Release -j"$(nproc)"

# Stage all installable artifacts into /artifacts so the runtime stages
# can do a single COPY. Some llama.cpp versions emit shared libs; copy
# them too if present.
RUN mkdir -p /artifacts/bin /artifacts/lib && \
    cp /src/build/bin/llama-* /artifacts/bin/ && \
    find /src/build -maxdepth 4 -name "*.so" -exec cp {} /artifacts/lib/ \; ; \
    cp /src/llama.cpp/convert_hf_to_gguf.py /artifacts/ ; \
    cp -r /src/llama.cpp/gguf-py /artifacts/ ; \
    true

# ============================================================================
# Target: cpu — the default, full llama.cpp CLI toolchain
# ============================================================================
FROM base AS cpu

COPY --from=builder /artifacts/bin/   /usr/local/bin/
COPY --from=builder /artifacts/lib/   /usr/local/lib/
COPY --from=builder /artifacts/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=builder /artifacts/gguf-py /opt/llama.cpp/gguf-py

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} \
    PYTHONPATH=/opt/llama.cpp/gguf-py

WORKDIR /work
CMD ["llama-cli", "--help"]

# ============================================================================
# Target: server — OpenAI-compatible HTTP API on :8080
# ============================================================================
FROM cpu AS server

EXPOSE 8080
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]

# ============================================================================
# Target: turboquant — adds Python preprocessor (heaviest image)
# ============================================================================
FROM cpu AS turboquant

# CPU-only torch wheel — drops the image from ~7 GB (CUDA torch) to ~2 GB.
RUN pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install --break-system-packages --no-cache-dir \
        transformers safetensors numpy

WORKDIR /app
COPY pyproject.toml /app/
COPY turboquant /app/turboquant
RUN pip install --break-system-packages --no-cache-dir /app

WORKDIR /work
ENTRYPOINT []
CMD ["python", "-m", "turboquant", "--help"]
