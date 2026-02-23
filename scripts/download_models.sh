#!/usr/bin/env bash
#
# Download model weights for SplazMatte.
#
# Activates the conda environment and delegates to download_models.py.
# All arguments are forwarded.
#
# Usage:
#   bash scripts/download_models.sh              # download all models
#   bash scripts/download_models.sh --sam3        # SAM3 only
#   bash scripts/download_models.sh --sam2        # SAM2.1 only
#   bash scripts/download_models.sh --matanyone   # MatAnyone only
#   bash scripts/download_models.sh --videomama   # VideoMaMa only
#   bash scripts/download_models.sh --cotracker   # CoTracker3 only
#   bash scripts/download_models.sh --verify      # verify existing downloads
#
# Parallel download (run each in a separate terminal):
#   bash scripts/download_models.sh --sam3        # terminal 1
#   bash scripts/download_models.sh --matanyone   # terminal 2
#   bash scripts/download_models.sh --videomama   # terminal 3
#   bash scripts/download_models.sh --cotracker   # terminal 4
#
set -euo pipefail

ENV_NAME="splazmatte"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
eval "$(conda shell.bash hook)"

if ! conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "[ERROR] Conda environment '$ENV_NAME' not found."
    echo "        Run 'bash scripts/setup.sh' first."
    exit 1
fi

conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# Run download script, forwarding all CLI arguments
# ---------------------------------------------------------------------------
python "$SCRIPT_DIR/download_models.py" "$@"
