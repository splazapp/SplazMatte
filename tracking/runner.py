"""Tracking task execution: run tracking + export + upload/notify.

Pure execution layer with no UI dependency. Called by queue runner
and by the tracking page's direct "run" button.
"""

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from tracking.logic import run_tracking
from tracking.export import export_ae_keyframe_data, export_trajectory_summary
from tracking.session_store import save_tracking_results, save_tracking_session

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None] | None


def run_tracking_task(
    state: dict,
    progress_callback: ProgressCallback = None,
    progress_prefix: str = "",
) -> tuple[Path | None, Path | None, float]:
    """Execute the full tracking pipeline for a session.

    Steps: run CoTracker → encode result video → export AE data →
    upload to cloud → send Feishu notification.

    Args:
        state: Tracking session state dict.
        progress_callback: ``(fraction, description) -> None`` or None.
        progress_prefix: Prefix for progress descriptions (e.g. "[1/3]").

    Returns:
        (result_video_path, ae_export_path, processing_time)
    """
    use_grid = state.get("use_grid", False)
    grid_size = state.get("grid_size", 15)
    backward_tracking = state.get("backward_tracking", False)

    def _progress(frac: float, desc: str = ""):
        if progress_callback is not None:
            full = f"{progress_prefix} {desc}".strip() if progress_prefix else desc
            progress_callback(frac, full)

    log.info(
        "========== 开始追踪任务 (session=%s, grid=%s, backward=%s) ==========",
        state.get("session_id", "?"),
        use_grid,
        backward_tracking,
    )

    start_ts = time.time()
    start_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Run tracking
    result = run_tracking(
        state,
        use_grid=use_grid,
        grid_size=grid_size,
        backward_tracking=backward_tracking,
        progress_callback=lambda f, d: _progress(f * 0.8, d),
    )
    state = result["session_state"]

    # Check if tracking produced an error / warning
    notify = result.get("notify")
    if notify and notify[0] in ("error", "warning"):
        raise ValueError(notify[1])

    # 2. Export AE keyframe data
    _progress(0.85, "导出 AE 数据...")
    export_result = export_ae_keyframe_data(state)
    state = export_result["session_state"]
    ae_export_path = export_result.get("export_path")

    # 2a. Export summary trajectory (IQR-filtered average)
    _progress(0.88, "生成整体轨迹...")
    summary_result = export_trajectory_summary(state)
    state = summary_result["session_state"]
    s_notify = summary_result.get("notify")
    if s_notify and s_notify[0] in ("error", "warning"):
        log.warning("export_trajectory_summary: %s", s_notify[1])

    # 3. Save raw tracking results to disk
    save_tracking_results(state)

    end_ts = time.time()
    end_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    processing_time = end_ts - start_ts

    state["processing_time"] = processing_time
    state["start_time"] = start_dt
    state["end_time"] = end_dt
    save_tracking_session(state)

    # 4. Upload + Feishu notification
    _progress(0.9, "上传结果...")
    result_video_path = state.get("result_video_path")
    try:
        upload_and_notify_tracking(state, processing_time, start_dt, end_dt)
    except Exception:
        log.exception("Post-tracking upload/notify failed")

    _progress(1.0, "完成")
    log.info("========== 追踪完成 (%.1fs) ==========", processing_time)

    return (
        Path(result_video_path) if result_video_path else None,
        Path(ae_export_path) if ae_export_path else None,
        processing_time,
    )


def upload_and_notify_tracking(
    state: dict,
    processing_time: float,
    start_time: str,
    end_time: str,
) -> None:
    """Upload tracking results to cloud and send Feishu notification.

    Args:
        state: Tracking session state with result paths.
        processing_time: Total processing time in seconds.
        start_time: UTC start time string.
        end_time: UTC end time string.
    """
    from utils.storage import upload_session

    sid = state.get("session_id", "")
    files_to_upload: list[Path] = []

    # Result video
    if state.get("result_video_path"):
        files_to_upload.append(Path(state["result_video_path"]))
    # AE export
    if state.get("ae_export_path"):
        files_to_upload.append(Path(state["ae_export_path"]))
    # Summary trajectory files
    if state.get("ae_summary_txt_path"):
        files_to_upload.append(Path(state["ae_summary_txt_path"]))
    if state.get("ae_summary_jsx_path"):
        files_to_upload.append(Path(state["ae_summary_jsx_path"]))
    # Original video
    if state.get("video_path"):
        files_to_upload.append(Path(state["video_path"]))

    files_to_upload = [f for f in files_to_upload if f.exists()]
    cdn_urls = upload_session(sid, files_to_upload) if files_to_upload else {}

    from utils.feishu_notify import send_feishu_tracking_success

    orig_h, orig_w = state.get("original_size", (0, 0))
    num_frames = state.get("num_frames", 0)
    keyframes = state.get("keyframes", {})
    total_points = sum(len(kf["points"]) for kf in keyframes.values())

    send_feishu_tracking_success(
        session_id=sid,
        source_filename=state.get("original_filename", ""),
        video_width=orig_w,
        video_height=orig_h,
        num_frames=num_frames,
        fps=state.get("fps", 24.0),
        keyframe_indices=sorted(keyframes.keys()),
        total_points=total_points,
        use_grid=state.get("use_grid", False),
        grid_size=state.get("grid_size", 15),
        backward_tracking=state.get("backward_tracking", False),
        processing_time=processing_time,
        start_time=start_time,
        end_time=end_time,
        cdn_urls=cdn_urls,
    )
