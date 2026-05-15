#!/usr/bin/env bash
# =============================================================================
# 3D Object Generation — Linux Installer (Rocky Linux 9.4 / RHEL 9)
#
# Usage:
#   sudo bash install.sh
#   sudo HF_TOKEN=hf_xxx bash install.sh           # with HuggingFace token
#   sudo HF_TOKEN=hf_xxx SKIP_MODELS=1 bash install.sh  # skip model download
#
# What this script does:
#   1. Installs system packages (dnf)
#   2. Installs CUDA 12.8 toolkit if not present
#   3. Creates a Python 3.11 virtual environment at .venv/
#   4. Installs Python dependencies + TRELLIS 2 CUDA extensions
#   5. Downloads AI models (~20 GB, skippable with SKIP_MODELS=1)
#   6. Installs and enables a systemd service
# =============================================================================
set -euo pipefail

# --- Colors & helpers --------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${CYAN}[→]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
die()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
SERVICE_NAME="3d-object-generation"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SKIP_MODELS="${SKIP_MODELS:-0}"

# --- Banner ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  3D Object Generation — Rocky Linux 9.4 Installer"
echo "  Repo: $REPO_DIR"
echo "============================================================"
echo ""

# --- Root check --------------------------------------------------------------
[[ "$(id -u)" -eq 0 ]] || die "Please run as root: sudo bash install.sh"

# Determine the non-root user who will run the service
RUN_USER="${SUDO_USER:-}"
if [[ -z "$RUN_USER" ]]; then
    warn "SUDO_USER not set. Defaulting service user to 'root'. Rerun via 'sudo' to use your own account."
    RUN_USER="root"
fi
RUN_HOME=$(getent passwd "$RUN_USER" | cut -d: -f6)
log "Service will run as user: $RUN_USER"

# --- LLM backend selection ---------------------------------------------------
OPENAI_URL=""
OPENAI_MODEL=""
USE_OPENAI_LLM=0

echo ""
echo "  LLM Backend Selection"
echo "  ─────────────────────"
echo "  Option A: Run a local LLM on this machine (requires extra GPU VRAM)"
echo "  Option B: Use an existing OpenAI-compatible API (vLLM, llama.cpp, Ollama...)"
echo ""
read -rp "  Do you have an OpenAI-compatible API endpoint? [y/N] " _ans_openai
if [[ "$_ans_openai" =~ ^[Yy]$ ]]; then
    read -rp "  API base URL (e.g. http://lx-gpu-001.vfx.priv:8000/v1): " OPENAI_URL
    OPENAI_URL="${OPENAI_URL:-http://localhost:8000/v1}"
    OPENAI_MODEL="default"
    USE_OPENAI_LLM=1
    info "Using OpenAI-compatible endpoint: $OPENAI_URL (model will be auto-detected at startup)"
else
    info "Using local LLM (native PyTorch)"
fi
echo ""

# --- GPU check ---------------------------------------------------------------
info "Checking NVIDIA GPU..."
command -v nvidia-smi &>/dev/null || die "nvidia-smi not found. Install NVIDIA drivers before running this script."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | while read line; do
    log "GPU detected: $line"
done

# --- System packages ---------------------------------------------------------
info "Installing system packages..."
dnf install -y epel-release dnf-plugins-core

dnf install -y \
    python3.11 python3.11-devel python3.11-pip \
    gcc gcc-c++ make cmake ninja-build \
    git git-lfs \
    libGL mesa-libGL mesa-libGL-devel \
    openexr openexr-devel \
    curl wget \
    2>/dev/null || warn "Some packages may have failed — continuing"

# Ensure git-lfs is initialized
git lfs install --skip-repo 2>/dev/null || true

log "System packages installed"

# --- CUDA 12.8 ---------------------------------------------------------------
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1)
    log "CUDA already installed: $CUDA_VER"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
else
    info "Installing CUDA 12.8 toolkit..."
    dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    dnf clean expire-cache
    dnf install -y cuda-toolkit-12-8 || die "CUDA toolkit installation failed"
    export CUDA_HOME="/usr/local/cuda"
    # Persist for future sessions
    echo 'export CUDA_HOME=/usr/local/cuda' > /etc/profile.d/cuda.sh
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "CUDA 12.8 installed at $CUDA_HOME"
fi

# --- Python 3.11 venv --------------------------------------------------------
info "Creating Python 3.11 virtual environment at $VENV_DIR ..."
python3.11 -m venv "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

"$PIP" install --upgrade pip setuptools wheel
log "Virtual environment created"

# --- PyTorch 2.7.0 + CUDA 12.8 -----------------------------------------------
info "Installing PyTorch 2.7.0 (CUDA 12.8)..."
"$PIP" install \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
log "PyTorch installed"

# --- Python dependencies -----------------------------------------------------
info "Installing Python dependencies..."
"$PIP" install -r "$REPO_DIR/requirements.txt"

[[ -f "$REPO_DIR/requirements-native.txt" ]] && \
    "$PIP" install -r "$REPO_DIR/requirements-native.txt"

[[ -f "$REPO_DIR/requirements-trellis.txt" ]] && \
    "$PIP" install -r "$REPO_DIR/requirements-trellis.txt" || \
    warn "requirements-trellis.txt not found — skipping"

# Remove deprecated pynvml if present (replaced by nvidia-ml-py, same API)
"$PIP" uninstall -y pynvml 2>/dev/null && info "Removed deprecated pynvml package" || true

log "Python dependencies installed"

# --- Apply LLM backend config ------------------------------------------------
if [[ "$USE_OPENAI_LLM" -eq 1 ]]; then
    info "Configuring OpenAI-compatible LLM backend in config.py..."
    "$PYTHON" - <<PYEOF
import re, sys

config_path = "$REPO_DIR/config.py"
with open(config_path, 'r') as f:
    content = f.read()

content = re.sub(
    r'USE_OPENAI_COMPATIBLE_LLM\s*=\s*\S+',
    'USE_OPENAI_COMPATIBLE_LLM = True',
    content
)
content = re.sub(
    r'OPENAI_COMPATIBLE_BASE_URL\s*=\s*"[^"]*"',
    'OPENAI_COMPATIBLE_BASE_URL = "$OPENAI_URL"',
    content
)
content = re.sub(
    r'OPENAI_COMPATIBLE_MODEL\s*=\s*"[^"]*"',
    'OPENAI_COMPATIBLE_MODEL = "$OPENAI_MODEL"',
    content
)

with open(config_path, 'w') as f:
    f.write(content)
print("  config.py updated: USE_OPENAI_COMPATIBLE_LLM = True")
print("  OPENAI_COMPATIBLE_BASE_URL = '$OPENAI_URL'")
print("  OPENAI_COMPATIBLE_MODEL = '$OPENAI_MODEL'")
PYEOF
    log "LLM backend configured"
fi

# --- TRELLIS 2 CUDA extensions -----------------------------------------------
info "Building TRELLIS 2 CUDA extensions (this takes 10–20 min)..."
export CUDA_HOME
cd "$REPO_DIR"
"$PYTHON" install_dependencies.py
log "TRELLIS 2 CUDA extensions built"

# --- Download AI models ------------------------------------------------------
if [[ "$SKIP_MODELS" == "1" ]]; then
    warn "Skipping model download (SKIP_MODELS=1). Run 'python download_models.py' manually."
else
    info "Downloading AI models (~20 GB — this may take a while)..."
    HF_TOKEN="${HF_TOKEN:-}"
    if [[ -z "$HF_TOKEN" ]]; then
        warn "HF_TOKEN not set. Some models (e.g. Llama) may require authentication."
        warn "Rerun with: sudo HF_TOKEN=hf_xxx bash install.sh"
    fi
    HF_TOKEN="$HF_TOKEN" "$PYTHON" download_models.py || warn "Model download failed — run manually later"
    log "Models downloaded"
fi

# --- systemd service ---------------------------------------------------------
info "Installing systemd service: $SERVICE_NAME ..."

# Build PATH for service: venv bin + CUDA bin + system
SVC_PATH="$VENV_DIR/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=3D Object Generation Service (TRELLIS 2)
Documentation=https://github.com/simonouel/3d-object-generation-T2
After=network.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO_DIR
Environment="PATH=$SVC_PATH"
Environment="CUDA_HOME=/usr/local/cuda"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
Environment="PYTHONUNBUFFERED=1"
Environment="OPENCV_IO_ENABLE_OPENEXR=1"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="HF_TOKEN=${HF_TOKEN:-}"
Environment="OPENAI_COMPATIBLE_BASE_URL=${OPENAI_URL:-}"
Environment="TRELLIS_ASSETS_DIR=${TRELLIS_ASSETS_DIR:-}"
ExecStart=$VENV_DIR/bin/python $REPO_DIR/app.py
Restart=on-failure
RestartSec=30
TimeoutStartSec=300
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

log "systemd service installed, enabled and started"

# --- Summary -----------------------------------------------------------------
echo ""
echo "============================================================"
echo -e "  ${GREEN}Installation complete!${NC}"
echo "============================================================"
echo ""
echo "  Service:  $SERVICE_NAME"
echo "  App URL:  http://localhost:7860"
echo ""
echo "  Useful commands:"
echo "    systemctl status  $SERVICE_NAME"
echo "    systemctl restart $SERVICE_NAME"
echo "    journalctl -u $SERVICE_NAME -f"
echo ""
echo "  To set HuggingFace token or network assets dir after install:"
echo "    sudo systemctl edit $SERVICE_NAME"
echo "    # Add: Environment=\"HF_TOKEN=hf_xxx\""
echo "    # Add: Environment=\"TRELLIS_ASSETS_DIR=/mnt/nfs/3d-assets\""
echo ""
