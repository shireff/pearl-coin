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
        sudo apt-get update

        local python_distutils_pkg=""
        if apt-cache show python3-distutils >/dev/null 2>&1; then
            python_distutils_pkg="python3-distutils"
        elif apt-cache show python3.12-distutils >/dev/null 2>&1; then
            python_distutils_pkg="python3.12-distutils"
        else
            python_distutils_pkg="python3-setuptools"
        fi

        sudo apt-get install -y \
            git curl build-essential pkg-config libssl-dev clang lld tmux \
            python3 python3-pip python3-venv python3-dev "$python_distutils_pkg" ninja-build \
            unzip
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
    local go_arch
    go_arch="$(detect_linux_arch)"
    local current_go
    if command -v go >/dev/null 2>&1; then
        current_go="$(go version | awk '{print $3}' | sed 's/^go//')"
        if [ "$current_go" = "$GO_VERSION" ]; then
            info "Go $GO_VERSION is already installed"
            return
        fi
        info "Replacing installed Go version $current_go with $GO_VERSION"
    fi

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
    info "Generating zk-pow verifier cache files..."
    cd "$REPO_ROOT/zk-pow"
    mkdir -p src/circuit src/v1
    cargo run --release --no-default-features --bin build_cache src/circuit/v2_cache.bin src/v1/v1_cache.bin
}

build_zk_pow_go_bindings() {
    info "Building zk-pow Go bindings..."
    cd "$REPO_ROOT/zk-pow/bindings/go"
    cargo build --release
}

build_go_binaries() {
    info "Building Go binaries..."
    cd "$REPO_ROOT"
    mkdir -p "$BIN_DIR"
    go build -tags xmss,zkpow -o "$BIN_DIR/pearld" ./node
    go build -tags xmss,zkpow -o "$BIN_DIR/prlctl" ./node/cmd/prlctl
    if [ -d "$REPO_ROOT/wallet" ]; then
        go build -tags xmss,zkpow -o "$BIN_DIR/oyster" ./wallet || true
        CGO_ENABLED=0 go build -o "$BIN_DIR/oystercli" ./wallet/cmd/oystercli || true
    fi
    touch "$BIN_DIR/sample-pearld.conf"
}

install_python_workspace() {
    info "Syncing Python workspace with uv..."
    cd "$REPO_ROOT"
    uv sync --all-packages
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
    install_python_workspace

    info "Setup complete."
    info "Run 'source ~/.bashrc' or open a new shell to refresh PATH."
    info "Then run: python3 miner/run_mining.py"
}

main "$@"
