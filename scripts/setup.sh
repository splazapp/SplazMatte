#!/usr/bin/env bash
#
# Setup the SplazMatte conda environment.
#
# Usage:
#   bash scripts/setup.sh          # auto-detect platform (MPS / CUDA)
#   bash scripts/setup.sh --cuda   # force CUDA install
#   bash scripts/setup.sh --mps    # force MPS (Mac) install
#
set -euo pipefail

ENV_NAME="splazmatte"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Clone required external SDKs
# ---------------------------------------------------------------------------
SDKS_DIR="$PROJECT_ROOT/sdks"
mkdir -p "$SDKS_DIR"

# repo_url -> local_dir_name -> pinned_commit (empty = use default branch HEAD)
SDK_REPOS=(
    "https://github.com/facebookresearch/co-tracker.git|co-tracker|82e02e8"
)

for entry in "${SDK_REPOS[@]}"; do
    IFS='|' read -r repo_url dir_name pinned_commit <<< "$entry"
    target="$SDKS_DIR/$dir_name"
    if [ -d "$target/.git" ]; then
        echo "[INFO] SDK '$dir_name' already cloned. Skipping."
    else
        echo "[INFO] Cloning $repo_url into sdks/$dir_name ..."
        git clone "$repo_url" "$target"
        if [ -n "$pinned_commit" ]; then
            echo "[INFO] Checking out pinned commit $pinned_commit for $dir_name"
            git -C "$target" checkout "$pinned_commit"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
PLATFORM=""
for arg in "$@"; do
    case "$arg" in
        --cuda) PLATFORM="cuda" ;;
        --mps)  PLATFORM="mps"  ;;
        *)      echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Auto-detect platform if not specified
if [ -z "$PLATFORM" ]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        PLATFORM="mps"
    else
        PLATFORM="cuda"
    fi
    echo "[INFO] Auto-detected platform: $PLATFORM"
fi

# ---------------------------------------------------------------------------
# Create conda environment
# ---------------------------------------------------------------------------
if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "[INFO] Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------
# Source conda so `conda activate` works in scripts
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "[INFO] Activated environment: $ENV_NAME ($(python --version))"

# ---------------------------------------------------------------------------
# Install PyTorch (platform-specific)
# ---------------------------------------------------------------------------
echo "[INFO] Installing PyTorch for platform: $PLATFORM"
if [ "$PLATFORM" = "cuda" ]; then
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124
else
    # macOS MPS — default PyPI wheels include MPS support
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
fi

# ---------------------------------------------------------------------------
# Install project dependencies
# ---------------------------------------------------------------------------
echo "[INFO] Installing project dependencies from requirements.txt..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available:  {torch.backends.mps.is_built() and torch.backends.mps.is_available()}')
import huggingface_hub, diffusers, transformers, gradio
print(f'huggingface_hub: {huggingface_hub.__version__}')
print(f'diffusers:       {diffusers.__version__}')
print(f'transformers:    {transformers.__version__}')
print(f'gradio:          {gradio.__version__}')
print()
print('Setup complete!')
"

echo ""
echo "To activate the environment in a new shell:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To download model weights:"
echo "  bash scripts/download_models.sh"
