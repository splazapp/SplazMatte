"""SplazMatte — 启动入口

日志配置、目录初始化、静态文件挂载、服务启动。
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

import asyncio
import logging
from logging.handlers import RotatingFileHandler

from nicegui import app, ui

from config import (
    LOGS_DIR,
    MATTING_SESSIONS_DIR,
    PROCESSING_LOG_FILE,
    SERVER_PORT,
    STORAGE_SECRET,
    TRACKING_SESSIONS_DIR,
    WORKSPACE_DIR,
)
from utils.feishu_notify import send_feishu_startup

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
PROCESSING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
PROCESSING_LOG_FILE.touch()
_file_handler = logging.FileHandler(str(PROCESSING_LOG_FILE), mode="w")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
))
logging.getLogger().addHandler(_file_handler)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_persistent_handler = RotatingFileHandler(
    str(LOGS_DIR / "splazmatte.log"),
    maxBytes=5_000_000, backupCount=5, encoding="utf-8",
)
_persistent_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
))
logging.getLogger().addHandler(_persistent_handler)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory init
# ---------------------------------------------------------------------------
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
MATTING_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
preview_dir = WORKSPACE_DIR / "preview"
preview_dir.mkdir(exist_ok=True)
tracking_preview_dir = WORKSPACE_DIR / "tracking_preview"
tracking_preview_dir.mkdir(exist_ok=True)
tracking_results_dir = WORKSPACE_DIR / "tracking_results"
tracking_results_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Static file mounts
# ---------------------------------------------------------------------------
app.add_static_files("/sessions", str(MATTING_SESSIONS_DIR))
app.add_static_files("/preview", str(preview_dir))
app.add_static_files("/workspace", str(WORKSPACE_DIR))
app.add_static_files("/tracking_preview", str(tracking_preview_dir))
app.add_static_files("/tracking_results", str(tracking_results_dir))

# ---------------------------------------------------------------------------
# Page registration (importing the modules registers @ui.page routes)
# ---------------------------------------------------------------------------
from pages.matting_page import matting_page      # noqa: F401, E402  @ui.page("/")
from pages.tracking_page import tracking_page    # noqa: F401, E402  @ui.page("/tracking")

# ---------------------------------------------------------------------------
# Startup notification
# ---------------------------------------------------------------------------
async def _notify_startup():
    await asyncio.sleep(1)
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}", timeout=2)
        send_feishu_startup(f"http://127.0.0.1:{SERVER_PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    log.info("SplazMatte 已启动，等待操作...")
    app.on_startup(_notify_startup)
    log.info("Launching NiceGUI (port=%s)...", SERVER_PORT)
    ui.run(
        host="0.0.0.0",
        port=SERVER_PORT,
        title="SplazMatte",
        reload=False,
        show=False,
        storage_secret=STORAGE_SECRET,
        reconnect_timeout=60,
    )
