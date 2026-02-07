"""Centralized paths and defaults for SplazMatte."""

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
SDKS_DIR = PROJECT_ROOT / "sdks"

# SAM2.1 (image predictor for mask annotation)
SAM2_CHECKPOINT = MODELS_DIR / "sam2" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# MatAnyone (video matting)
MATANYONE_CHECKPOINT = MODELS_DIR / "matanyone" / "matanyone.pth"

# Processing defaults
DEFAULT_ERODE = 10
DEFAULT_DILATE = 10
DEFAULT_WARMUP = 10
MAX_VIDEO_DURATION = 60  # seconds
MAX_VIDEO_SHORT_SIDE = 1080

# Cloudflare R2 (all values from environment variables)
R2_ENDPOINT = os.environ.get("SPLAZMATTE_R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("SPLAZMATTE_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("SPLAZMATTE_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("SPLAZMATTE_R2_SECRET_KEY", "")
R2_CDN_DOMAIN = os.environ.get("SPLAZMATTE_R2_CDN_DOMAIN", "")
R2_PREFIX = os.environ.get("SPLAZMATTE_R2_PREFIX", "splazmatte")

# 飞书机器人
FEISHU_WEBHOOK_URL = os.environ.get("SPLAZMATTE_FEISHU_WEBHOOK_URL", "")

# Processing log (displayed in Gradio UI)
PROCESSING_LOG_FILE = WORKSPACE_DIR / "processing.log"


def get_device() -> torch.device:
    """Detect available compute device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Gradio server
GRADIO_SERVER_PORT = int(os.environ.get("SPLAZMATTE_PORT", "7860"))
