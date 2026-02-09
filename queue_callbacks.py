"""Callback functions for the task queue.

Queue items are session_id strings persisted in workspace/queue.json.
All task data (matting params, status) lives in each session's state.json.
"""

import json
import logging
import zipfile
from pathlib import Path

import gradio as gr

from app_callbacks import (
    _clear_processing_log,
    keyframe_display,
    keyframe_gallery,
    render_frame,
)
from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
)
from matting_runner import execute_queue
from queue_models import load_queue, save_queue
from session_store import empty_state, load_session, save_session_state

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue display helpers
# ---------------------------------------------------------------------------
def _read_session_info(session_id: str) -> dict | None:
    """Read meta.json + state.json for a session. Returns None on failure."""
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


def _queue_status_text(queue: list[str]) -> str:
    """Build a status markdown string from the queue list."""
    if not queue:
        return "队列为空。"
    pending = 0
    done = 0
    error = 0
    for sid in queue:
        info = _read_session_info(sid)
        status = info.get("task_status", "") if info else ""
        if status == "done":
            done += 1
        elif status == "error":
            error += 1
        else:
            pending += 1
    parts = [f"队列中 {len(queue)} 个任务"]
    if pending:
        parts.append(f"待处理: {pending}")
    if done:
        parts.append(f"完成: {done}")
    if error:
        parts.append(f"错误: {error}")
    return " | ".join(parts)


def _queue_table_data(queue: list[str]) -> list[list]:
    """Build dataframe rows from the queue list."""
    rows = []
    for i, sid in enumerate(queue, start=1):
        info = _read_session_info(sid)
        if info is None:
            rows.append([i, sid, 0, 0, "No", "-", "丢失"])
            continue
        video_name = info.get("original_filename", sid)
        num_frames = info.get("num_frames", 0)
        kf_indices = info.get("keyframe_indices", [])
        has_prop = "Yes" if info.get("has_propagation") else "No"
        engine = info.get("matting_engine", "MatAnyone")
        engine_short = "MA" if engine == "MatAnyone" else "VM"
        status = info.get("task_status", "pending")
        rows.append([i, video_name, num_frames, len(kf_indices), has_prop, engine_short, status])
    return rows


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_add_to_queue(
    matting_engine: str,
    erode: int,
    dilate: int,
    batch_size: int,
    overlap: int,
    seed: int,
    session_state: dict,
    queue_state: list[str],
):
    """Save matting params to session, add session_id to queue, reset UI.

    Returns:
        Tuple of 13 values: session_state, queue_state, queue_status,
        queue_table, frame_display, frame_slider, frame_label,
        keyframe_info, kf_gallery, video_input, propagation_preview,
        alpha_output, fgr_output.
    """
    if not session_state.get("keyframes"):
        gr.Warning("请至少保存一个关键帧后再添加到队列。")
        queue = load_queue()
        return (
            session_state, queue,
            _queue_status_text(queue),
            _queue_table_data(queue),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
        )

    # Write matting params + pending status into session state
    session_state["matting_engine"] = matting_engine
    session_state["erode"] = int(erode)
    session_state["dilate"] = int(dilate)
    session_state["batch_size"] = int(batch_size)
    session_state["overlap"] = int(overlap)
    session_state["seed"] = int(seed)
    session_state["task_status"] = "pending"
    session_state["error_msg"] = ""
    save_session_state(session_state)

    # Add to queue (dedup by session_id)
    queue = load_queue()
    sid = session_state["session_id"]
    if sid not in queue:
        queue.append(sid)
    save_queue(queue)

    log.info(
        "Added to queue: session=%s, video=%s, keyframes=%d, engine=%s",
        sid,
        session_state.get("original_filename", ""),
        len(session_state["keyframes"]),
        matting_engine,
    )

    # Reset session state and all annotation UI components
    new_state = empty_state()
    return (
        new_state,
        queue,
        _queue_status_text(queue),
        _queue_table_data(queue),
        None,
        gr.update(value=0, visible=False),
        gr.update(value="请先上传视频。"),
        "尚未保存任何关键帧。",
        [],
        None,
        None,
        None,
        None,
    )


def on_remove_from_queue(
    remove_idx: int,
    queue_state: list[str],
):
    """Remove a queue item by 1-based index.

    Returns:
        Tuple of (queue_state, queue_status, queue_table).
    """
    queue = load_queue()
    idx = int(remove_idx) - 1
    if idx < 0 or idx >= len(queue):
        gr.Warning(f"无效序号: {int(remove_idx)}，队列共 {len(queue)} 项。")
        return (
            queue,
            _queue_status_text(queue),
            _queue_table_data(queue),
        )

    removed_sid = queue[idx]
    queue = [*queue[:idx], *queue[idx + 1:]]
    save_queue(queue)
    log.info("Removed from queue: session=%s", removed_sid)
    return (
        queue,
        _queue_status_text(queue),
        _queue_table_data(queue),
    )


def on_clear_queue(queue_state: list[str]):
    """Clear all items from the queue.

    Returns:
        Tuple of (queue_state, queue_status, queue_table).
    """
    save_queue([])
    log.info("Queue cleared")
    return (
        [],
        _queue_status_text([]),
        _queue_table_data([]),
    )


def on_restore_from_queue(
    restore_idx: int,
    session_state: dict,
    queue_state: list[str],
):
    """Restore a queue item into the editing area for modification.

    Loads the full session state from disk so the user can edit
    keyframes, re-run propagation, change parameters, etc.

    Returns:
        Tuple of 21 values covering all editable UI components.
    """
    # 21 no-update values for early returns
    no_update = (
        session_state, queue_state,
        gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
    )

    queue = load_queue()
    idx = int(restore_idx) - 1
    if idx < 0 or idx >= len(queue):
        gr.Warning(f"无效序号: {int(restore_idx)}，队列共 {len(queue)} 项。")
        return no_update

    sid = queue[idx]
    loaded = load_session(sid)
    if loaded is None:
        gr.Warning(f"无法加载 Session: {sid}")
        return no_update

    frames_dir = loaded["frames_dir"]
    if not Path(frames_dir).exists():
        gr.Warning("帧数据目录不存在，无法恢复编辑。")
        return no_update

    # Navigate to first keyframe
    first_kf = min(loaded["keyframes"].keys()) if loaded["keyframes"] else 0
    loaded["current_frame_idx"] = first_kf
    loaded["current_mask"] = loaded["keyframes"].get(first_kf)

    frame = render_frame(loaded)
    gallery = keyframe_gallery(loaded)
    kf_info = keyframe_display(loaded)

    # Check for existing propagation preview
    session_dir = WORKSPACE_DIR / "sessions" / sid
    preview_path = session_dir / "propagation_preview.mp4"
    prop_preview = str(preview_path) if preview_path.exists() else None

    # Check for existing matting output videos
    alpha_path = session_dir / "alpha.mp4"
    fgr_path = session_dir / "foreground.mp4"
    alpha_video = str(alpha_path) if alpha_path.exists() else None
    fgr_video = str(fgr_path) if fgr_path.exists() else None

    # Parameter visibility based on matting engine
    is_ma = loaded.get("matting_engine", "MatAnyone") == "MatAnyone"

    log.info("Restored queue item for editing: session=%s", sid)

    return (
        loaded,
        queue,
        _queue_status_text(queue),
        _queue_table_data(queue),
        frame,
        gr.update(
            minimum=0, maximum=loaded["num_frames"] - 1,
            value=first_kf, visible=True, interactive=True,
        ),
        f"第 {first_kf} 帧 / 共 {loaded['num_frames'] - 1} 帧",
        kf_info,
        gallery,
        str(loaded["source_video_path"]) if loaded["source_video_path"] else None,
        gr.update(value=loaded["model_type"]),
        gr.update(visible=(loaded["model_type"] == "SAM3")),
        gr.update(value=loaded.get("matting_engine", "MatAnyone")),
        gr.update(value=loaded.get("erode", DEFAULT_ERODE), visible=is_ma),
        gr.update(value=loaded.get("dilate", DEFAULT_DILATE), visible=is_ma),
        gr.update(value=loaded.get("batch_size", VIDEOMAMA_BATCH_SIZE), visible=not is_ma),
        gr.update(value=loaded.get("overlap", VIDEOMAMA_OVERLAP), visible=not is_ma),
        gr.update(value=loaded.get("seed", VIDEOMAMA_SEED), visible=not is_ma),
        prop_preview,
        alpha_video,
        fgr_video,
    )


def _pack_results_zip(queue: list[str]) -> Path | None:
    """Pack alpha.mp4 and foreground.mp4 from all done sessions into a zip.

    Args:
        queue: List of session IDs to check.

    Returns:
        Path to the zip file, or None if no successful results.
    """
    zip_path = WORKSPACE_DIR / "results.zip"
    packed = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sid in queue:
            info = _read_session_info(sid)
            if info is None or info.get("task_status") != "done":
                continue
            session_dir = WORKSPACE_DIR / "sessions" / sid
            video_name = Path(info.get("original_filename", sid)).stem
            for filename in ("alpha.mp4", "foreground.mp4"):
                src = session_dir / filename
                if src.exists():
                    zf.write(src, f"{video_name}/{filename}")
                    packed += 1
    if packed == 0:
        zip_path.unlink(missing_ok=True)
        return None
    return zip_path


def on_execute_queue(
    _queue_progress: str,
    queue_state: list[str],
    progress=gr.Progress(track_tqdm=True),
):
    """Execute all pending queue items sequentially.

    Delegates to ``matting_runner.execute_queue`` for the actual work,
    then updates Gradio UI components with results.

    Returns:
        Tuple of (queue_state, queue_status, queue_table, queue_progress,
        download_file).
    """
    queue = load_queue()

    # Check if there are any pending tasks before clearing the log
    has_pending = any(
        (info := _read_session_info(sid)) is not None
        and info.get("task_status", "") in ("pending", "")
        for sid in queue
    )
    if not has_pending:
        gr.Warning("队列中没有待处理的任务。")
        return (
            queue,
            _queue_status_text(queue),
            _queue_table_data(queue),
            "没有待处理的任务。",
            gr.update(value=None, visible=False),
        )

    _clear_processing_log()

    def progress_cb(frac: float, desc: str = ""):
        progress(frac, desc=desc)

    done_count, error_count, timings = execute_queue(progress_cb)

    # Build summary
    summary_parts = [f"**队列执行完毕**: {done_count} 成功"]
    if error_count:
        summary_parts[0] += f", {error_count} 失败"
    for t in timings:
        summary_parts.append(f"- {t}")
    summary = "\n".join(summary_parts)

    # Re-read queue from disk for latest state
    queue = load_queue()

    # Pack results zip
    zip_path = _pack_results_zip(queue)
    if zip_path is not None:
        download_update = gr.update(value=str(zip_path), visible=True)
    else:
        download_update = gr.update(value=None, visible=False)

    return (
        queue,
        _queue_status_text(queue),
        _queue_table_data(queue),
        summary,
        download_update,
    )


def on_load_queue():
    """Restore queue display on page load.

    Returns:
        Tuple of (queue_state, queue_status, queue_table).
    """
    queue = load_queue()
    return queue, _queue_status_text(queue), _queue_table_data(queue)
