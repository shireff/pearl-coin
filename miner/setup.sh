#!/usr/bin/env bash
set -e

CLONE_DIR="$HOME/pearl"

echo "[1/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y git curl build-essential pkg-config libssl-dev clang lld tmux

echo "[2/6] Installing torch/numpy (needed by the monitoring script)..."
python3 -m pip install --user torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126 numpy

echo "[3/6] Installing Rust..."
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "[4/6] Installing Go..."
curl -fsSL https://go.dev/dl/go1.26.0.linux-amd64.tar.gz -o /tmp/go.tar.gz
sudo tar -C /usr/local -xzf /tmp/go.tar.gz
export PATH="/usr/local/go/bin:$PATH"
grep -qxF 'export PATH="/usr/local/go/bin:$PATH"' ~/.bashrc || echo 'export PATH="/usr/local/go/bin:$PATH"' >> ~/.bashrc

echo "[5/6] Cloning and building pearl-cion..."
git clone --recurse-submodules https://github.com/shireff/pearl-cion.git "$CLONE_DIR"
cd "$CLONE_DIR/xmss" && make
mkdir -p "$CLONE_DIR/zk-pow/src/circuit" && touch "$CLONE_DIR/zk-pow/src/circuit/v2_cache.bin"
mkdir -p "$CLONE_DIR/zk-pow/src/v1" && touch "$CLONE_DIR/zk-pow/src/v1/v1_cache.bin"
cd "$CLONE_DIR/zk-pow/bindings/go" && cargo build --release
cd "$CLONE_DIR" && go build -tags xmss,zkpow -o bin/pearld ./node
cd "$CLONE_DIR" && go build -tags xmss,zkpow -o bin/prlctl ./node/cmd/prlctl
touch "$CLONE_DIR/bin/sample-pearld.conf"
mkdir -p "$HOME/.pearl"

echo "[6/6] Installing uv and syncing the python workspace (this is the slow step)..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
cd "$CLONE_DIR" && uv sync --all-packages

echo ""
echo "==================================================="
echo "Setup finished successfully."
echo "Next step: copy run_mining_studio.py into $CLONE_DIR"
echo "then start it inside a tmux session."
echo "==================================================="