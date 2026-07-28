FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CLONE_DIR=/root/pearl

# --- system deps + python 3.12 (Modal's add_python="3.12" equivalent) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
        build-essential pkg-config libssl-dev clang lld \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

ENV PATH="/root/.cargo/bin:/usr/local/go/bin:/root/.local/bin:${PATH}"

# --- python deps ---
RUN python3.12 -m pip install --no-cache-dir uv
RUN python3.12 -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
RUN python3.12 -m pip install --no-cache-dir numpy

# --- rust + go toolchains ---
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN curl -fsSL https://go.dev/dl/go1.26.0.linux-amd64.tar.gz -o /tmp/go.tar.gz \
    && tar -C /usr/local -xzf /tmp/go.tar.gz

# --- clone + build pearl-cion ---
RUN git clone --recurse-submodules https://github.com/shireff/pearl-cion.git ${CLONE_DIR}
RUN cd ${CLONE_DIR}/xmss && make
RUN mkdir -p ${CLONE_DIR}/zk-pow/src/circuit && touch ${CLONE_DIR}/zk-pow/src/circuit/v2_cache.bin
RUN mkdir -p ${CLONE_DIR}/zk-pow/src/v1 && touch ${CLONE_DIR}/zk-pow/src/v1/v1_cache.bin
RUN cd ${CLONE_DIR}/zk-pow/bindings/go && cargo build --release
RUN cd ${CLONE_DIR} && go build -tags xmss,zkpow -o bin/pearld ./node
RUN cd ${CLONE_DIR} && go build -tags xmss,zkpow -o bin/prlctl ./node/cmd/prlctl

# stub the config template pearld looks for at startup (silences the warning)
RUN touch ${CLONE_DIR}/bin/sample-pearld.conf
RUN mkdir -p /root/.pearl

# --- uv + full workspace sync (torch/vllm/etc, same as `uv sync --all-packages`) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN cd ${CLONE_DIR} && uv sync --all-packages

# --- orchestration script (see run_mining.py) ---
COPY run_mining.py ${CLONE_DIR}/run_mining.py

WORKDIR ${CLONE_DIR}