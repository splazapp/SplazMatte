"""Download video from URL using yt-dlp."""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def download_video(url: str, dest_dir: Path) -> Path:
    """Download video from URL to dest_dir using yt-dlp.

    Supports direct HTTP links and social media platforms (TikTok, WeChat, etc.).

    Args:
        url: Video URL to download.
        dest_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded video file.

    Raises:
        RuntimeError: If download fails.
    """
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize URL for use as filename prefix
    url_slug = re.sub(r"[^\w]", "_", url)[:40]
    out_template = str(dest_dir / f"{url_slug}.%(ext)s")

    ydl_opts = {
        "outtmpl": out_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": False,
    }

    downloaded_path: Path | None = None

    class _Hook:
        def __call__(self, d: dict) -> None:
            nonlocal downloaded_path
            if d.get("status") == "finished":
                downloaded_path = Path(d["filename"])

    ydl_opts["progress_hooks"] = [_Hook()]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ret = ydl.download([url])
        if ret != 0:
            raise RuntimeError(f"yt-dlp returned exit code {ret}")

    # yt-dlp may rename merged file; scan dest_dir for most recent video file
    if downloaded_path is None or not downloaded_path.exists():
        candidates = sorted(
            (p for p in dest_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("Download succeeded but no video file found")
        downloaded_path = candidates[0]

    log.info("Downloaded video to %s", downloaded_path)
    return downloaded_path
