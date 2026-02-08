#!/usr/bin/env bash
#
# Start the SplazMatte Gradio app (macOS / Linux).
#
# Usage:
#   bash start.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="splazmatte"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[INFO] Environment: $ENV_NAME ($(python --version))"
echo "[INFO] Starting SplazMatte ..."

python "$SCRIPT_DIR/app.py"
