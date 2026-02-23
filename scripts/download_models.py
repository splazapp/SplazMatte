"""Download model weights for SplazMatte.

Features:
    - tqdm progress bars for each download
    - Skips already-downloaded files (checks sentinel files)
    - Resume support for interrupted downloads (huggingface_hub built-in)

Usage:
    python scripts/download_models.py              # download all models
    python scripts/download_models.py --sam3        # SAM3 only
    python scripts/download_models.py --sam2        # SAM2.1 only
    python scripts/download_models.py --matanyone   # MatAnyone only
    python scripts/download_models.py --videomama   # VideoMaMa only
    python scripts/download_models.py --cotracker   # CoTracker3 only
    python scripts/download_models.py --verify      # verify existing downloads
"""

import argparse
import logging
from pathlib import Path

import requests
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Project root / models directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# HuggingFace repo IDs and filenames
SAM3_REPO = "facebook/sam3"
SAM3_FILES = ["sam3.pt", "config.json"]

MATANYONE_URL = (
    "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
)
MATANYONE_FILE = "matanyone.pth"

SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)
SAM2_FILE = "sam2.1_hiera_large.pt"

SVD_REPO = "stabilityai/stable-video-diffusion-img2vid-xt"
VIDEOMAMA_REPO = "SammyLim/VideoMaMa"

COTRACKER_REPO = "facebook/cotracker3"
COTRACKER_FILE = "scaled_online.pth"
COTRACKER_OFFLINE_FILE = "scaled_offline.pth"

# Sentinel files used to check if a model group is fully downloaded
SENTINEL_FILES: dict[str, list[str]] = {
    "sam3": ["sam3/sam3.pt", "sam3/config.json"],
    "sam2": ["sam2/sam2.1_hiera_large.pt"],
    "matanyone": ["matanyone/matanyone.pth"],
    "videomama_svd": [
        "videomama/stable-video-diffusion-img2vid-xt/model_index.json",
    ],
    "videomama_unet": [
        "videomama/VideoMaMa/unet/config.json",
        "videomama/VideoMaMa/unet/diffusion_pytorch_model.safetensors",
    ],
    "cotracker": ["cotracker/scaled_online.pth", "cotracker/scaled_offline.pth"],
}


def detect_device() -> str:
    """Detect available compute device (cuda / mps / cpu)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"cuda ({name})"
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


def check_hf_auth() -> bool:
    """Check if the user is logged in to HuggingFace Hub."""
    try:
        HfApi().whoami()
        return True
    except Exception:
        return False


def _file_ok(path: Path) -> bool:
    """Check if a file exists and has non-zero size."""
    return path.is_file() and path.stat().st_size > 0


def _is_downloaded(models_dir: Path, sentinel_key: str) -> bool:
    """Check if all sentinel files for a model group exist and are non-empty."""
    paths = SENTINEL_FILES.get(sentinel_key, [])
    return all(_file_ok(models_dir / p) for p in paths)


def _hf_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download a single file via hf_hub_download (tqdm + auto-resume)."""
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        force_download=False,
    ))


def _snapshot_dl(repo_id: str, local_dir: Path) -> Path:
    """Download a full HF repo via snapshot_download (tqdm + auto-resume)."""
    return Path(snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
    ))


def _download_url(url: str, dest_path: Path) -> None:
    """Download a file from a URL with tqdm progress and resume support.

    Uses HTTP Range header to resume partial downloads.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    # Resume from partial download if .tmp file exists
    downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + downloaded
    mode = "ab" if downloaded else "wb"

    with open(tmp_path, mode) as f, tqdm(
        total=total,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))

    tmp_path.rename(dest_path)


def download_sam3(models_dir: Path) -> bool:
    """Download SAM3 weights (gated, requires HF auth). Returns success."""
    log.info("--- SAM3 ---")

    if _is_downloaded(models_dir, "sam3"):
        log.info("  Already downloaded, skipping.")
        return True

    if not check_hf_auth():
        log.error(
            "SAM3 requires HuggingFace login: huggingface-cli login"
        )
        return False
    log.info("  (Gated model — request access at https://huggingface.co/facebook/sam3)")

    dest = models_dir / "sam3"
    dest.mkdir(parents=True, exist_ok=True)

    try:
        for filename in SAM3_FILES:
            log.info(f"  Downloading {filename}...")
            _hf_download(SAM3_REPO, filename, dest)
        log.info("  SAM3 complete.")
        return True
    except Exception as e:
        log.error(f"  SAM3 failed: {e}")
        return False


def download_sam2(models_dir: Path) -> bool:
    """Download SAM2.1 Large weights from Meta CDN (public). Returns success."""
    log.info("--- SAM2.1 ---")

    if _is_downloaded(models_dir, "sam2"):
        log.info("  Already downloaded, skipping.")
        return True

    dest_file = models_dir / "sam2" / SAM2_FILE

    try:
        log.info(f"  Downloading {SAM2_FILE} from Meta CDN...")
        _download_url(SAM2_URL, dest_file)
        log.info("  SAM2.1 complete.")
        return True
    except Exception as e:
        log.error(f"  SAM2.1 failed: {e}")
        return False


def download_matanyone(models_dir: Path) -> bool:
    """Download MatAnyone weights from GitHub Releases (public). Returns success."""
    log.info("--- MatAnyone ---")

    if _is_downloaded(models_dir, "matanyone"):
        log.info("  Already downloaded, skipping.")
        return True

    dest_file = models_dir / "matanyone" / MATANYONE_FILE

    try:
        log.info(f"  Downloading {MATANYONE_FILE} from GitHub Releases...")
        _download_url(MATANYONE_URL, dest_file)
        log.info("  MatAnyone complete.")
        return True
    except Exception as e:
        log.error(f"  MatAnyone failed: {e}")
        return False


def download_videomama(models_dir: Path) -> bool:
    """Download SVD base + VideoMaMa UNet (public). Returns success."""
    log.info("--- VideoMaMa ---")

    dest = models_dir / "videomama"
    dest.mkdir(parents=True, exist_ok=True)
    ok = True

    # SVD base model
    if _is_downloaded(models_dir, "videomama_svd"):
        log.info("  SVD base: already downloaded, skipping.")
    else:
        try:
            log.info("  Downloading stable-video-diffusion-img2vid-xt...")
            _snapshot_dl(
                SVD_REPO,
                dest / "stable-video-diffusion-img2vid-xt",
            )
            log.info("  SVD base complete.")
        except Exception as e:
            log.error(f"  SVD base failed: {e}")
            ok = False

    # VideoMaMa UNet
    if _is_downloaded(models_dir, "videomama_unet"):
        log.info("  VideoMaMa UNet: already downloaded, skipping.")
    else:
        try:
            log.info("  Downloading VideoMaMa UNet...")
            _snapshot_dl(
                VIDEOMAMA_REPO,
                dest / "VideoMaMa",
            )
            log.info("  VideoMaMa UNet complete.")
        except Exception as e:
            log.error(f"  VideoMaMa UNet failed: {e}")
            ok = False

    return ok


def download_cotracker(models_dir: Path) -> bool:
    """Download CoTracker3 Online + Offline weights from HuggingFace (public). Returns success."""
    log.info("--- CoTracker3 ---")

    dest = models_dir / "cotracker"
    dest.mkdir(parents=True, exist_ok=True)
    ok = True

    if not _file_ok(dest / COTRACKER_FILE):
        try:
            log.info(f"  Downloading {COTRACKER_FILE} (Online mode)...")
            _hf_download(COTRACKER_REPO, COTRACKER_FILE, dest)
            log.info("  CoTracker3 Online complete.")
        except Exception as e:
            log.error(f"  CoTracker3 Online failed: {e}")
            ok = False
    else:
        log.info(f"  {COTRACKER_FILE} already present, skipping.")

    if not _file_ok(dest / COTRACKER_OFFLINE_FILE):
        try:
            log.info(f"  Downloading {COTRACKER_OFFLINE_FILE} (Offline / backward mode)...")
            _hf_download(COTRACKER_REPO, COTRACKER_OFFLINE_FILE, dest)
            log.info("  CoTracker3 Offline complete.")
        except Exception as e:
            log.error(f"  CoTracker3 Offline failed: {e}")
            ok = False
    else:
        log.info(f"  {COTRACKER_OFFLINE_FILE} already present, skipping.")

    return ok


def verify_downloads(models_dir: Path) -> dict[str, bool]:
    """Check all sentinel files exist and are non-empty."""
    results: dict[str, bool] = {}
    for name in SENTINEL_FILES:
        results[name] = _is_downloaded(models_dir, name)
    return results



def print_verification(results: dict[str, bool]) -> None:
    """Print verification results in a readable table."""
    log.info("=== Verification Results ===")
    for name, ok in results.items():
        status = "OK" if ok else "MISSING"
        log.info(f"  {name:20s} [{status}]")

    if all(results.values()):
        log.info("All models present.")
    else:
        missing = [n for n, ok in results.items() if not ok]
        log.warning(f"Missing: {', '.join(missing)}")


def main() -> None:
    """Parse CLI args and orchestrate model downloads."""
    parser = argparse.ArgumentParser(
        description="Download model weights for SplazMatte.",
    )
    parser.add_argument("--sam3", action="store_true", help="Download SAM3")
    parser.add_argument("--sam2", action="store_true", help="Download SAM2.1 Large")
    parser.add_argument("--matanyone", action="store_true", help="Download MatAnyone")
    parser.add_argument("--videomama", action="store_true", help="Download VideoMaMa")
    parser.add_argument("--cotracker", action="store_true", help="Download CoTracker3")
    parser.add_argument("--verify", action="store_true", help="Verify downloads")
    args = parser.parse_args()

    log.info(f"Device: {detect_device()}")
    log.info(f"Models directory: {MODELS_DIR}")

    if args.verify:
        results = verify_downloads(MODELS_DIR)
        print_verification(results)
        return

    # No flags = download all
    download_all = not (
        args.sam3 or args.sam2 or args.matanyone or args.videomama or args.cotracker
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    if download_all or args.sam3:
        results["sam3"] = download_sam3(MODELS_DIR)
    if download_all or args.sam2:
        results["sam2"] = download_sam2(MODELS_DIR)
    if download_all or args.matanyone:
        results["matanyone"] = download_matanyone(MODELS_DIR)
    if download_all or args.videomama:
        results["videomama"] = download_videomama(MODELS_DIR)
    if download_all or args.cotracker:
        results["cotracker"] = download_cotracker(MODELS_DIR)

    # Summary
    log.info("=== Download Summary ===")
    for name, ok in results.items():
        status = "SUCCESS" if ok else "FAILED"
        log.info(f"  {name:20s} [{status}]")

    if not all(results.values()):
        log.warning("Some downloads failed. Re-run to retry (supports resume).")


if __name__ == "__main__":
    main()
