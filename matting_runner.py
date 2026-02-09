"""Matting pipeline execution: engine lifecycle, matting, and queue runner.

Pure execution layer with no Gradio dependency. Used by both the Gradio UI
callbacks and the CLI entry point.

Progress callbacks use ``Callable[[float, str], None] | None`` — the first
argument is a fraction (0.0–1.0), the second is a description string.
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    DEFAULT_WARMUP,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
)
from pipeline.video_io import encode_video, load_all_frames_as_tensor
from queue_models import load_queue
from session_store import load_session, save_session_state
from utils.notify import notify_failure, upload_and_notify

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None] | None

# ---------------------------------------------------------------------------
# Queue cancellation flag
# ---------------------------------------------------------------------------
_queue_cancel_requested = False


def request_queue_cancel():
    """Signal the running queue to stop after the current task finishes."""
    global _queue_cancel_requested
    _queue_cancel_requested = True


def reset_queue_cancel():
    """Clear the cancellation flag (called at queue execution start)."""
    global _queue_cancel_requested
    _queue_cancel_requested = False


# ---------------------------------------------------------------------------
# Lazy-loaded engine singletons (matting-pipeline engines only)
# ---------------------------------------------------------------------------
_sam2_video_engine = None
_sam3_video_engine = None
_matanyone_engine = None
_videomama_engine = None


def _get_sam2_video():
    global _sam2_video_engine
    if _sam2_video_engine is None:
        from engines.sam2_video_engine import SAM2VideoEngine
        _sam2_video_engine = SAM2VideoEngine()
    return _sam2_video_engine


def _get_sam3_video():
    global _sam3_video_engine
    if _sam3_video_engine is None:
        from engines.sam3_video_engine import SAM3VideoEngine
        _sam3_video_engine = SAM3VideoEngine()
    return _sam3_video_engine


def get_video_engine(model_type: str):
    """Return the video propagation engine for the given model type."""
    return _get_sam3_video() if model_type == "SAM3" else _get_sam2_video()


def empty_device_cache():
    """Release cached memory on the active device (CUDA or MPS)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_matanyone():
    """Lazily load MatAnyone, unloading competing engines first."""
    global _matanyone_engine
    if _matanyone_engine is None:
        unload_videomama()
        unload_sam_video()
        from engines.matanyone_engine import MatAnyoneEngine
        _matanyone_engine = MatAnyoneEngine()
    return _matanyone_engine


def get_videomama():
    """Lazily load VideoMaMa, unloading competing engines first."""
    global _videomama_engine
    if _videomama_engine is None:
        unload_matanyone()
        unload_sam_video()
        from engines.videomama_engine import VideoMaMaEngine
        _videomama_engine = VideoMaMaEngine()
    return _videomama_engine


def unload_matanyone():
    """Free MatAnyone from VRAM."""
    global _matanyone_engine
    if _matanyone_engine is not None:
        del _matanyone_engine
        _matanyone_engine = None
        empty_device_cache()


def unload_videomama():
    """Free VideoMaMa from VRAM."""
    global _videomama_engine
    if _videomama_engine is not None:
        del _videomama_engine
        _videomama_engine = None
        empty_device_cache()


def unload_sam_video():
    """Free SAM2/SAM3 video engines to reclaim memory."""
    global _sam2_video_engine, _sam3_video_engine
    changed = False
    if _sam2_video_engine is not None:
        del _sam2_video_engine
        _sam2_video_engine = None
        changed = True
    if _sam3_video_engine is not None:
        del _sam3_video_engine
        _sam3_video_engine = None
        changed = True
    if changed:
        empty_device_cache()


# ---------------------------------------------------------------------------
# Matting sub-pipelines
# ---------------------------------------------------------------------------
def run_videomama(
    state: dict,
    batch_size: int,
    overlap: int,
    seed: int,
    progress_callback: ProgressCallback = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run VideoMaMa matting pipeline.

    Args:
        state: Session state dict.
        batch_size: Frames per inference batch.
        overlap: Overlap frames between batches for blending.
        seed: Random seed for reproducibility.
        progress_callback: Optional progress reporter.

    Returns:
        Tuple of (alphas, foregrounds) arrays.

    Raises:
        ValueError: If propagated masks are incomplete.
    """
    masks = state.get("propagated_masks", {})
    if len(masks) < state["num_frames"]:
        raise ValueError(
            "VideoMaMa 需要每帧遮罩，请先运行 SAM2 传播。"
            f"（当前 {len(masks)}/{state['num_frames']} 帧）"
        )

    log.info(
        "开始 VideoMaMa 推理 (%d 帧, batch=%d, overlap=%d, seed=%d)...",
        state["num_frames"], batch_size, overlap, seed,
    )
    engine = get_videomama()
    cb = (
        (lambda f: progress_callback(f, "VideoMaMa 推理中..."))
        if progress_callback else None
    )
    return engine.process(
        frames_dir=state["frames_dir"],
        masks=masks,
        batch_size=batch_size,
        overlap=overlap,
        seed=seed,
        progress_callback=cb,
    )


def run_matanyone(
    state: dict,
    erode: int,
    dilate: int,
    progress_callback: ProgressCallback = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MatAnyone matting pipeline.

    Args:
        state: Session state dict.
        erode: Erosion kernel size.
        dilate: Dilation kernel size.
        progress_callback: Optional progress reporter.

    Returns:
        Tuple of (alphas, foregrounds) arrays.
    """
    log.info("加载帧数据 (%d 帧)...", state["num_frames"])
    frames_tensor = load_all_frames_as_tensor(state["frames_dir"])

    log.info("开始 MatAnyone 推理...")
    engine = get_matanyone()
    cb = (
        (lambda f: progress_callback(f, "MatAnyone 推理中..."))
        if progress_callback else None
    )
    return engine.process(
        frames=frames_tensor,
        keyframe_masks=state["keyframes"],
        erode=erode,
        dilate=dilate,
        warmup=DEFAULT_WARMUP,
        progress_callback=cb,
    )


# ---------------------------------------------------------------------------
# Main matting task
# ---------------------------------------------------------------------------
def run_matting_task(
    state: dict,
    progress_callback: ProgressCallback = None,
    progress_prefix: str = "",
) -> tuple[Path, Path, float]:
    """Execute the full matting pipeline for a session.

    Handles auto-propagation, matting inference, video encoding, and
    upload/notify. Reads matting parameters from ``state`` directly.

    Args:
        state: Session state dict (must contain matting params).
        progress_callback: ``(fraction, description) -> None`` or None.
        progress_prefix: Prefix for progress descriptions (e.g. "[1/3]").

    Returns:
        (alpha_path, fgr_path, processing_time)
    """
    matting_engine = state.get("matting_engine", "MatAnyone")
    erode = int(state.get("erode", DEFAULT_ERODE))
    dilate = int(state.get("dilate", DEFAULT_DILATE))
    batch_size = int(state.get("batch_size", VIDEOMAMA_BATCH_SIZE))
    overlap = int(state.get("overlap", VIDEOMAMA_OVERLAP))
    seed = int(state.get("seed", VIDEOMAMA_SEED))
    model_type = state.get("model_type", "SAM2")

    def _progress(frac: float, desc: str = ""):
        if progress_callback is not None:
            full = f"{progress_prefix} {desc}".strip() if progress_prefix else desc
            progress_callback(frac, full)

    # Auto-run propagation if not done yet
    if not state.get("propagated_masks"):
        log.info("传播尚未执行，自动运行 %s 传播...", model_type)
        engine = get_video_engine(model_type)
        propagated = engine.propagate(
            frames_dir=state["frames_dir"],
            keyframe_masks=state["keyframes"],
            progress_callback=lambda f: _progress(f, "自动传播中..."),
        )
        state["propagated_masks"] = propagated
        log.info("自动传播完成，共 %d 帧遮罩", len(propagated))

    log.info("========== 开始抠像 (引擎: %s) ==========", matting_engine)
    log.info(
        "Session: %s | 关键帧: %s | erode=%d, dilate=%d",
        state["session_id"],
        sorted(state["keyframes"].keys()),
        erode,
        dilate,
    )

    start_ts = time.time()
    start_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if matting_engine == "VideoMaMa":
        alphas, foregrounds = run_videomama(
            state, batch_size, overlap, seed, _progress,
        )
    else:
        alphas, foregrounds = run_matanyone(
            state, erode, dilate, _progress,
        )

    session_dir = WORKSPACE_DIR / "sessions" / state["session_id"]
    alpha_path = session_dir / "alpha.mp4"
    fgr_path = session_dir / "foreground.mp4"

    log.info("编码视频...")
    alpha_rgb = np.repeat(alphas, 3, axis=3)
    encode_video(alpha_rgb, alpha_path, state["fps"])
    encode_video(foregrounds, fgr_path, state["fps"])

    end_ts = time.time()
    end_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    processing_time = end_ts - start_ts

    # Persist timing info so manual re-send can access them
    state["processing_time"] = processing_time
    state["start_time"] = start_dt
    state["end_time"] = end_dt

    log.info("上传至 R2...")
    try:
        upload_and_notify(
            state, erode, dilate, session_dir,
            processing_time, start_dt, end_dt,
        )
    except Exception:
        log.exception("Post-matting upload/notify failed")

    log.info("========== 处理完成 (%.1fs) ==========", processing_time)
    return alpha_path, fgr_path, processing_time


# ---------------------------------------------------------------------------
# Queue execution
# ---------------------------------------------------------------------------
def _read_session_info(session_id: str) -> dict | None:
    """Read meta.json + state.json for a session (lightweight)."""
    session_dir = WORKSPACE_DIR / "sessions" / session_id
    meta_path = session_dir / "meta.json"
    state_path = session_dir / "state.json"
    if not meta_path.exists() or not state_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        state = json.loads(state_path.read_text())
        return {**meta, **state}
    except (json.JSONDecodeError, OSError):
        return None


def execute_queue(
    progress_callback: ProgressCallback = None,
) -> tuple[int, int, list[str]]:
    """Execute all pending queue items sequentially.

    Args:
        progress_callback: ``(fraction, description) -> None`` or None.

    Returns:
        (done_count, error_count, timings) where timings is a list of
        human-readable strings like ``"video.mp4: 12.3s"`` or
        ``"video.mp4: 失败"``.
    """
    queue = load_queue()

    # Filter to pending sessions
    pending_sids: list[str] = []
    for sid in queue:
        info = _read_session_info(sid)
        status = info.get("task_status", "") if info else ""
        if status in ("pending", ""):
            pending_sids.append(sid)

    if not pending_sids:
        return 0, 0, []

    reset_queue_cancel()

    total = len(pending_sids)
    log.info("========== 开始执行队列 (%d 个任务) ==========", total)

    done_count = 0
    error_count = 0
    timings: list[str] = []

    for i, sid in enumerate(pending_sids, start=1):
        if _queue_cancel_requested:
            log.info("用户取消了队列执行")
            break
        loaded = load_session(sid)
        if loaded is None:
            log.warning("Cannot load session %s, skipping", sid)
            timings.append(f"{sid}: 加载失败")
            error_count += 1
            continue

        video_name = loaded.get("original_filename", sid)
        log.info(
            "--- 任务 %d/%d: session=%s, video=%s ---",
            i, total, sid, video_name,
        )

        loaded["task_status"] = "processing"
        loaded["error_msg"] = ""
        save_session_state(loaded)

        try:
            prefix = f"[{i}/{total}] {sid}"
            _, _, elapsed = run_matting_task(loaded, progress_callback, prefix)
            loaded["task_status"] = "done"
            save_session_state(loaded)
            done_count += 1
            timings.append(f"{video_name}: {elapsed:.1f}s")
        except Exception as exc:
            loaded["task_status"] = "error"
            loaded["error_msg"] = str(exc)
            save_session_state(loaded)
            error_count += 1
            log.exception("任务失败 (session=%s): %s", sid, exc)
            notify_failure(sid, exc)
            timings.append(f"{video_name}: 失败")

    log.info(
        "========== 队列执行完毕: %d 成功, %d 失败 ==========",
        done_count, error_count,
    )
    return done_count, error_count, timings
