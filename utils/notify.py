"""Post-matting upload and notification helpers."""

import logging
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_WARMUP
from utils.feishu_notify import send_feishu_failure, send_feishu_success
from utils.storage import upload_session

log = logging.getLogger(__name__)


def upload_and_notify(
    state: dict,
    erode: int,
    dilate: int,
    session_dir: Path,
    processing_time: float,
    start_time: str,
    end_time: str,
) -> None:
    """Upload results to cloud storage and send Feishu success notification.

    Args:
        state: Session state dict containing keyframes, video info, etc.
        erode: Erosion kernel size used for processing.
        dilate: Dilation kernel size used for processing.
        session_dir: Path to the session working directory.
        processing_time: Total processing time in seconds.
        start_time: UTC start time string.
        end_time: UTC end time string.
    """
    # Save keyframe masks as PNGs for upload
    mask_paths: list[Path] = []
    for idx, mask in sorted(state["keyframes"].items()):
        mask_path = session_dir / f"keyframe_{idx:06d}.png"
        # cv2.imwrite cannot handle non-ASCII paths on Windows
        success, buf = cv2.imencode(".png", mask)
        buf.tofile(str(mask_path))
        mask_paths.append(mask_path)

    files_to_upload = [
        state.get("source_video_path"),
        session_dir / "alpha.mp4",
        session_dir / "foreground.mp4",
        *mask_paths,
    ]
    files_to_upload = [Path(f) for f in files_to_upload if f is not None]

    cdn_urls = upload_session(state["session_id"], files_to_upload)

    source_name = state.get("original_filename") or (
        Path(state["source_video_path"]).name
        if state.get("source_video_path")
        else "unknown"
    )
    send_feishu_success(
        session_id=state["session_id"],
        source_filename=source_name,
        video_width=state.get("video_width", 0),
        video_height=state.get("video_height", 0),
        video_duration=state.get("video_duration", 0.0),
        num_frames=state.get("num_frames", 0),
        fps=state.get("fps", 0.0),
        video_format=state.get("video_format", ""),
        file_size=state.get("video_file_size", 0),
        erode=erode,
        dilate=dilate,
        warmup=DEFAULT_WARMUP,
        keyframe_indices=sorted(state["keyframes"].keys()),
        processing_time=processing_time,
        start_time=start_time,
        end_time=end_time,
        cdn_urls=cdn_urls,
        matting_engine=state.get("matting_engine", "MatAnyone"),
        model_type=state.get("model_type", "SAM2"),
        batch_size=state.get("batch_size", 0),
        overlap=state.get("overlap", 0),
        seed=state.get("seed", 0),
    )


def notify_failure(session_id: str, error: Exception) -> None:
    """Send Feishu failure notification, swallowing any errors.

    Args:
        session_id: Current session identifier.
        error: The exception that caused the failure.
    """
    try:
        send_feishu_failure(session_id, error)
    except Exception:
        log.exception("Failed to send failure notification to Feishu")
