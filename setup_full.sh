#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$REPO_ROOT/bin"
GO_VERSION="1.26.0"
PYTHON_REQUIRED_MAJOR=3
PYTHON_REQUIRED_MINOR=12

info() {
    printf 'setup-full: %s\n' "$*"
}

die() {
    printf 'setup-full: ERROR: %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        die "required command not found: $1"
    fi
}

check_repo_root() {
    if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
        die "setup_full.sh must be executed from the repository root containing pyproject.toml"
    fi
}

detect_linux_arch() {
    local uname_arch
    uname_arch="$(uname -m)"
    case "$uname_arch" in
        x86_64|amd64) echo "amd64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) die "unsupported CPU architecture: $uname_arch" ;;
    esac
}

install_system_packages() {
    info "Installing required system packages..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq

        # Install base packages first (always available)
        sudo apt-get install -y \
            git curl build-essential pkg-config libssl-dev clang lld tmux \
            python3 python3-pip python3-venv python3-dev ninja-build unzip

        # Install distutils/setuptools — try each option, skip on failure
        for pkg in python3-distutils python3.12-distutils python3-setuptools; do
            if apt-cache show "$pkg" >/dev/null 2>&1; then
                sudo apt-get install -y "$pkg" && break || true
            fi
        done
    else
        die "unsupported package manager: apt-get is required on this system"
    fi
}

check_python_version() {
    require_cmd python3
    local version
    version="$(python3 -c 'import sys; print("%s.%s" % sys.version_info[:2])')"
    if [ "$version" != "$PYTHON_REQUIRED_MAJOR.$PYTHON_REQUIRED_MINOR" ]; then
        die "Python 3.12 is required, but python3 is $version. Install Python 3.12 and retry."
    fi
}

install_rust() {
    if command -v cargo >/dev/null 2>&1; then
        info "Rust is already installed"
    else
        info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
    cargo --version
}

install_go() {
    if command -v go >/dev/null 2>&1; then
        info "Go is already installed: $(go version)"
        export PATH="/usr/local/go/bin:$PATH"
        return
    fi

    local go_arch
    go_arch="$(detect_linux_arch)"
    local go_tarball="go${GO_VERSION}.linux-${go_arch}.tar.gz"
    local go_url="https://go.dev/dl/${go_tarball}"
    local tmp_file
    tmp_file="$(mktemp -t go-install-XXXXXX.tar.gz)"

    info "Downloading Go $GO_VERSION for ${go_arch}..."
    curl -fsSL "$go_url" -o "$tmp_file"
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "$tmp_file"
    rm -f "$tmp_file"

    if ! grep -qxF 'export PATH="/usr/local/go/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
        printf '%s\n' 'export PATH="/usr/local/go/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    export PATH="/usr/local/go/bin:$PATH"
    go version
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        info "uv is already installed"
    else
        info "Installing uv workspace manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    if ! grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
        printf '%s\n' 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    uv --version
}

install_python_prereqs() {
    info "Installing Python packaging tools..."
    python3 -m pip install --upgrade --user pip setuptools wheel
}

generate_zk_pow_cache() {
    info "Skipping zk-pow cache generation (embedded_cache feature disabled)..."
    # Cache files are not needed when building without the embedded_cache feature.
    # The verifier loads cache at runtime from disk instead.
    :
}

build_zk_pow_go_bindings() {
    info "Building zk-pow Go bindings..."
    cd "$REPO_ROOT/zk-pow/bindings/go"
    cargo build --release --no-default-features
}

build_xmss() {
    info "Building xmss static library..."
    cd "$REPO_ROOT/xmss"
    make
    if [ ! -f libxmss.a ]; then
        die "xmss build failed: libxmss.a not found after make"
    fi
}

build_go_binaries() {
    info "Building Go binaries..."
    cd "$REPO_ROOT"
    mkdir -p "$BIN_DIR"
    build_xmss
    go build -tags xmss,zkpow -o "$BIN_DIR/pearld" ./node
    go build -tags xmss,zkpow -o "$BIN_DIR/prlctl" ./node/cmd/prlctl
    if [ -d "$REPO_ROOT/wallet" ]; then
        go build -tags xmss,zkpow -o "$BIN_DIR/oyster" ./wallet || true
        CGO_ENABLED=0 go build -o "$BIN_DIR/oystercli" ./wallet/cmd/oystercli || true
    fi
    touch "$BIN_DIR/sample-pearld.conf"
}

build_pearl_gemm() {
    info "Building pearl-gemm CUDA extension..."
    local gemm_dir="$REPO_ROOT/miner/pearl-gemm"
    if [ ! -f "$gemm_dir/setup.py" ]; then
        info "pearl-gemm/setup.py not found, skipping CUDA build"
        return
    fi
    # Auto-detect CUDA installation
    if [ -z "${CUDA_HOME:-}" ]; then
        for cuda_candidate in /usr/local/cuda-12.4 /usr/local/cuda-12.6 /usr/local/cuda; do
            if [ -f "$cuda_candidate/bin/nvcc" ]; then
                CUDA_HOME="$cuda_candidate"
                break
            fi
        done
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    fi
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
    export PEARL_GEMM_ARCH="${PEARL_GEMM_ARCH:-arch=compute_89,code=sm_89}"
    export MAX_JOBS="${MAX_JOBS:-4}"
    export NVCC_PREPEND_FLAGS=""
    export CUDAHOSTCXX=/usr/bin/g++
    export CXX=/usr/bin/g++
    export CC=/usr/bin/gcc
    cd "$gemm_dir"
    python3 -m pip install -e "$REPO_ROOT/miner/pearl-gemm-build-utils"
    rm -rf build src/*.so 2>/dev/null || true
    python3 setup.py build_ext --inplace
    info "pearl-gemm build complete"
}

install_python_workspace() {
    info "Syncing Python workspace with uv..."
    cd "$REPO_ROOT"
    # pearl-gemm is built separately via build.sh — exclude it from uv sync
    # to avoid re-downloading torch in a temp dir (causes disk-space failures)
    uv sync --all-packages --no-install-package pearl-gemm || \
        uv sync --package vllm-miner --no-install-package pearl-gemm
}

main() {
    check_repo_root
    install_system_packages
    check_python_version
    install_rust
    install_go
    install_uv
    install_python_prereqs
    generate_zk_pow_cache
    build_zk_pow_go_bindings
    build_go_binaries
    build_pearl_gemm
    install_python_workspace

    info "Setup complete."
    info "Run 'source ~/.bashrc' or open a new shell to refresh PATH."
    info "Then run: python3 miner/run_mining.py"
}

main "$@"
