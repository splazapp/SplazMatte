"""Centralized paths and defaults for SplazMatte."""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Enable MPS CPU fallback for ops not yet implemented on Apple Silicon.
# Must be set before torch initialises its MPS backend (static cache).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
LOGS_DIR = PROJECT_ROOT / "logs"
SDKS_DIR = PROJECT_ROOT / "sdks"

# Session storage: matting and tracking use separate directories
MATTING_SESSIONS_DIR = WORKSPACE_DIR / "matte_sessions"
TRACKING_SESSIONS_DIR = WORKSPACE_DIR / "tracking_sessions"

# SAM2.1 (image predictor for mask annotation)
SAM2_CHECKPOINT = MODELS_DIR / "sam2" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# SAM3 (image + video predictor with text prompt support)
SAM3_CHECKPOINT = MODELS_DIR / "sam3" / "sam3.pt"

# MatAnyone (video matting)
MATANYONE_CHECKPOINT = MODELS_DIR / "matanyone" / "matanyone.pth"

# VideoMaMa (diffusion-based video matting)
VIDEOMAMA_SVD_PATH = MODELS_DIR / "videomama" / "stable-video-diffusion-img2vid-xt"
VIDEOMAMA_UNET_PATH = MODELS_DIR / "videomama" / "VideoMaMa"
VIDEOMAMA_BATCH_SIZE = (
    40 if torch.backends.mps.is_available() else 8
)
VIDEOMAMA_OVERLAP = (
    2 if torch.backends.mps.is_available() else 0
)
VIDEOMAMA_SEED = 42         # random seed for reproducibility

# CoTracker3 (point tracking)
COTRACKER_CHECKPOINT = MODELS_DIR / "cotracker" / "scaled_online.pth"
COTRACKER_OFFLINE_CHECKPOINT = MODELS_DIR / "cotracker" / "scaled_offline.pth"
COTRACKER_WINDOW_LEN = 16
COTRACKER_OFFLINE_WINDOW_LEN = 60
COTRACKER_INPUT_RESO = (384, 512)  # (H, W) model input resolution
COTRACKER_FRAME_LIMIT = 300        # max frames to process
# CoTracker uses grid_sampler_3d which MPS does not support; fallback causes CPU-GPU
# transfer overhead. Use CPU on MPS for more predictable performance.
COTRACKER_USE_CPU_ON_MPS = True

# SAM2 Hiera backbone uses bicubic interpolation (aten::upsample_bicubic2d) which MPS
# does not support; it falls back to CPU and triggers UserWarning + poor performance.
# Force CPU on MPS for consistent behavior.
SAM2_USE_CPU_ON_MPS = False

# Processing defaults
DEFAULT_MATTING_ENGINE = "VideoMaMa"
DEFAULT_ERODE = 10
DEFAULT_DILATE = 10
DEFAULT_WARMUP = 10
MAX_VIDEO_DURATION = 60  # seconds
MAX_VIDEO_SHORT_SIDE = 1080
MAX_PRELOAD_FRAMES = int(os.environ.get("SPLAZMATTE_MAX_PRELOAD_FRAMES", "600"))
MAX_PROPAGATION_FRAMES = int(os.environ.get("SPLAZMATTE_MAX_PROPAGATION_FRAMES", "2000"))
PREVIEW_MAX_W = int(os.environ.get("SPLAZMATTE_PREVIEW_MAX_W", "800"))
PREVIEW_MAX_H = int(os.environ.get("SPLAZMATTE_PREVIEW_MAX_H", "800"))

# Cloudflare R2 (all values from environment variables)
R2_ENDPOINT = os.environ.get("SPLAZMATTE_R2_ENDPOINT", "")
R2_BUCKET = os.environ.get("SPLAZMATTE_R2_BUCKET", "")
R2_ACCESS_KEY = os.environ.get("SPLAZMATTE_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("SPLAZMATTE_R2_SECRET_KEY", "")
R2_CDN_DOMAIN = os.environ.get("SPLAZMATTE_R2_CDN_DOMAIN", "")
R2_PREFIX = os.environ.get("SPLAZMATTE_R2_PREFIX", "splazmatte")

# Storage backend: "r2" | "oss" | "" (disabled)
STORAGE_BACKEND = os.environ.get("SPLAZMATTE_STORAGE_BACKEND", "").lower()

# Aliyun OSS
OSS_ACCESS_KEY_ID = os.environ.get("SPLAZMATTE_OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.environ.get("SPLAZMATTE_OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET = os.environ.get("SPLAZMATTE_OSS_BUCKET", "")
OSS_ENDPOINT = os.environ.get("SPLAZMATTE_OSS_ENDPOINT", "")
OSS_CDN_DOMAIN = os.environ.get("SPLAZMATTE_OSS_CDN_DOMAIN", "")
OSS_PREFIX = os.environ.get("SPLAZMATTE_OSS_PREFIX", "splazmatte")

# 飞书机器人
FEISHU_WEBHOOK_URL = os.environ.get("SPLAZMATTE_FEISHU_WEBHOOK_URL", "")

# Processing log (displayed in UI)
PROCESSING_LOG_FILE = WORKSPACE_DIR / "processing.log"


def get_device() -> torch.device:
    """Detect available compute device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Web server (NiceGUI)
SERVER_PORT = int(os.environ.get("SPLAZMATTE_PORT", "7870"))

# NiceGUI storage secret (required for app.storage.user and app.storage.browser)
STORAGE_SECRET = os.environ.get("SPLAZMATTE_STORAGE_SECRET", "splazmatte-default-secret-change-in-production")
