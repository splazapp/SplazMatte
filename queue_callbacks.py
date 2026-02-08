"""Callback functions for the task queue."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np

from app_callbacks import (
    _clear_processing_log,
    _get_video_engine,
    _run_matanyone,
    _run_videomama,
    empty_state,
    keyframe_display,
    keyframe_gallery,
    render_frame,
)
from config import WORKSPACE_DIR
from pipeline.video_io import encode_video
from queue_models import QueueItem, snapshot_to_queue_item
from utils.notify import notify_failure, upload_and_notify

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------
class _TaskProgress:
    """Wrap ``gr.Progress`` to prefix descriptions with task index.

    This allows sub-functions (e.g. ``_run_videomama``) that call
    ``progress(frac, desc=...)`` to automatically include the task
    context like ``[1/3] VideoMaMa 推理中...``.
    """

    def __init__(self, progress: gr.Progress, prefix: str):
        self._progress = progress
        self._prefix = prefix

    def __call__(self, fraction: float, desc: str = ""):
        full_desc = f"{self._prefix} {desc}" if desc else self._prefix
        self._progress(fraction, desc=full_desc)


# ---------------------------------------------------------------------------
# Queue display helpers
# ---------------------------------------------------------------------------
def _queue_status_text(queue: list[QueueItem]) -> str:
    """Build a status markdown string from the queue list."""
    if not queue:
        return "队列为空。"
    pending = sum(1 for item in queue if item.status == "pending")
    done = sum(1 for item in queue if item.status == "done")
    error = sum(1 for item in queue if item.status == "error")
    parts = [f"队列中 {len(queue)} 个任务"]
    if pending:
        parts.append(f"待处理: {pending}")
    if done:
        parts.append(f"完成: {done}")
    if error:
        parts.append(f"错误: {error}")
    return " | ".join(parts)


def _queue_table_data(queue: list[QueueItem]) -> list[list]:
    """Build dataframe rows from the queue list."""
    rows = []
    for i, item in enumerate(queue, start=1):
        video_name = item.original_filename or Path(item.source_video_path).name
        engine_short = "MA" if item.matting_engine == "MatAnyone" else "VM"
        rows.append([
            i,
            video_name,
            item.num_frames,
            len(item.keyframes),
            "Yes" if item.has_propagation else "No",
            engine_short,
            item.status,
        ])
    return rows


def _item_to_state_dict(item: QueueItem) -> dict:
    """Convert a QueueItem back to a state dict for matting functions.

    The returned dict contains the fields that ``_run_matanyone``,
    ``_run_videomama``, and ``upload_and_notify`` read from state.
    """
    return {
        "session_id": item.session_id,
        "source_video_path": item.source_video_path,
        "frames_dir": item.frames_dir,
        "num_frames": item.num_frames,
        "fps": item.fps,
        "video_width": item.video_width,
        "video_height": item.video_height,
        "video_duration": item.video_duration,
        "video_file_size": item.video_file_size,
        "video_format": item.video_format,
        "keyframes": item.keyframes,
        "propagated_masks": item.propagated_masks,
        "model_type": item.model_type,
    }


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
    queue_state: list[QueueItem],
):
    """Snapshot the current session into a queue item, then reset the UI.

    Returns:
        Tuple of 13 values: session_state, queue_state, queue_status,
        queue_table, frame_display, frame_slider, frame_label,
        keyframe_info, kf_gallery, video_input, propagation_preview,
        alpha_output, fgr_output.
    """
    if not session_state.get("keyframes"):
        gr.Warning("请至少保存一个关键帧后再添加到队列。")
        return (
            session_state, queue_state,
            _queue_status_text(queue_state),
            _queue_table_data(queue_state),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
        )

    item = snapshot_to_queue_item(
        session_state, matting_engine,
        int(erode), int(dilate), int(batch_size), int(overlap), int(seed),
    )

    editing_id = session_state.get("_editing_item_id")
    if editing_id and any(q.item_id == editing_id for q in queue_state):
        # Overwrite existing queue item
        queue_state = [item if q.item_id == editing_id else q for q in queue_state]
        log.info(
            "Updated queue item %s: video=%s, keyframes=%d, engine=%s",
            editing_id,
            item.original_filename or Path(item.source_video_path).name,
            len(item.keyframes),
            item.matting_engine,
        )
    else:
        queue_state = [*queue_state, item]
        log.info(
            "Added to queue: session=%s, video=%s, keyframes=%d, engine=%s",
            item.session_id,
            item.original_filename or Path(item.source_video_path).name,
            len(item.keyframes),
            item.matting_engine,
        )

    # Reset session state and all annotation UI components
    new_state = empty_state()
    return (
        new_state,
        queue_state,
        _queue_status_text(queue_state),
        _queue_table_data(queue_state),
        # frame_display
        None,
        # frame_slider
        gr.update(value=0, visible=False),
        # frame_label
        gr.update(value="请先上传视频。"),
        # keyframe_info
        "尚未保存任何关键帧。",
        # kf_gallery
        [],
        # video_input
        None,
        # propagation_preview
        None,
        # alpha_output
        None,
        # fgr_output
        None,
    )


def on_remove_from_queue(
    remove_idx: int,
    queue_state: list[QueueItem],
):
    """Remove a queue item by 1-based index.

    Returns:
        Tuple of (queue_state, queue_status, queue_table).
    """
    idx = int(remove_idx) - 1
    if idx < 0 or idx >= len(queue_state):
        gr.Warning(f"无效序号: {int(remove_idx)}，队列共 {len(queue_state)} 项。")
        return (
            queue_state,
            _queue_status_text(queue_state),
            _queue_table_data(queue_state),
        )

    removed = queue_state[idx]
    queue_state = [*queue_state[:idx], *queue_state[idx + 1:]]
    log.info("Removed from queue: session=%s", removed.session_id)
    return (
        queue_state,
        _queue_status_text(queue_state),
        _queue_table_data(queue_state),
    )


def on_restore_from_queue(
    restore_idx: int,
    session_state: dict,
    queue_state: list[QueueItem],
):
    """Restore a queue item into the editing area for modification.

    Reconstructs the full session state from the queue item so the user
    can edit keyframes, re-run propagation, change parameters, etc.
    When the user clicks "添加到队列" afterwards, the original queue
    item is overwritten with the updated data.

    Args:
        restore_idx: 1-based index of the item to restore.
        session_state: Current session state (will be replaced).
        queue_state: Current queue list.

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

    idx = int(restore_idx) - 1
    if idx < 0 or idx >= len(queue_state):
        gr.Warning(f"无效序号: {int(restore_idx)}，队列共 {len(queue_state)} 项。")
        return no_update

    item = queue_state[idx]

    if not item.frames_dir or not Path(item.frames_dir).exists():
        gr.Warning("帧数据目录不存在，无法恢复编辑。")
        return no_update

    # Reconstruct session state from queue item
    state = empty_state()
    state["session_id"] = item.session_id
    state["frames_dir"] = item.frames_dir
    state["num_frames"] = item.num_frames
    state["fps"] = item.fps
    state["source_video_path"] = item.source_video_path
    state["original_filename"] = item.original_filename
    state["video_file_size"] = item.video_file_size
    state["video_format"] = item.video_format
    state["video_duration"] = item.video_duration
    state["video_width"] = item.video_width
    state["video_height"] = item.video_height
    state["model_type"] = item.model_type
    state["keyframes"] = {k: v.copy() for k, v in item.keyframes.items()}
    state["propagated_masks"] = {
        k: v.copy() for k, v in item.propagated_masks.items()
    }
    # Mark for overwrite on next "添加到队列"
    state["_editing_item_id"] = item.item_id

    # Navigate to first keyframe
    first_kf = min(item.keyframes.keys()) if item.keyframes else 0
    state["current_frame_idx"] = first_kf
    state["current_mask"] = state["keyframes"].get(first_kf)

    frame = render_frame(state)
    gallery = keyframe_gallery(state)
    kf_info = keyframe_display(state)

    # Check for existing propagation preview
    session_dir = WORKSPACE_DIR / "sessions" / item.session_id
    preview_path = session_dir / "propagation_preview.mp4"
    prop_preview = str(preview_path) if preview_path.exists() else None

    # Parameter visibility based on matting engine
    is_ma = item.matting_engine == "MatAnyone"

    log.info(
        "Restored queue item for editing: session=%s, item_id=%s",
        item.session_id, item.item_id,
    )

    return (
        state,
        queue_state,
        _queue_status_text(queue_state),
        _queue_table_data(queue_state),
        # frame_display
        frame,
        # frame_slider
        gr.update(
            minimum=0, maximum=item.num_frames - 1,
            value=first_kf, visible=True, interactive=True,
        ),
        # frame_label
        f"第 {first_kf} 帧 / 共 {item.num_frames - 1} 帧",
        # keyframe_info
        kf_info,
        # kf_gallery
        gallery,
        # video_input
        str(item.source_video_path),
        # model_selector
        gr.update(value=item.model_type),
        # text_prompt_row
        gr.update(visible=(item.model_type == "SAM3")),
        # matting_engine_selector
        gr.update(value=item.matting_engine),
        # erode_slider, dilate_slider
        gr.update(value=item.erode, visible=is_ma),
        gr.update(value=item.dilate, visible=is_ma),
        # vm_batch_slider, vm_overlap_slider, vm_seed_input
        gr.update(value=item.batch_size, visible=not is_ma),
        gr.update(value=item.overlap, visible=not is_ma),
        gr.update(value=item.seed, visible=not is_ma),
        # propagation_preview
        prop_preview,
        # alpha_output
        None,
        # fgr_output
        None,
    )


def _execute_single_item(
    item: QueueItem,
    progress: gr.Progress,
    task_idx: int,
    total_tasks: int,
) -> float:
    """Execute matting for a single queue item with progress tracking.

    Args:
        item: The queue item to process.
        progress: Gradio progress tracker.
        task_idx: 1-based index of this task in the batch.
        total_tasks: Total number of tasks in the batch.

    Returns:
        Processing time in seconds.
    """
    prefix = f"[{task_idx}/{total_tasks}]"
    tp = _TaskProgress(progress, prefix)
    state = _item_to_state_dict(item)

    # Phase 1: Auto-propagate if needed
    if not item.has_propagation:
        log.info("自动运行 %s 传播 (session=%s)...", item.model_type, item.session_id)
        tp(0, "传播中...")
        engine = _get_video_engine(item.model_type)
        propagated = engine.propagate(
            frames_dir=item.frames_dir,
            keyframe_masks=item.keyframes,
            progress_callback=lambda f: tp(f, "传播中..."),
        )
        item.propagated_masks = propagated
        item.has_propagation = True
        state["propagated_masks"] = propagated

    # Phase 2: Run matting
    start_ts = time.time()
    start_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    tp(0, f"{item.matting_engine} 推理中...")

    if item.matting_engine == "VideoMaMa":
        alphas, foregrounds = _run_videomama(
            state, item.batch_size, item.overlap, item.seed, tp,
        )
    else:
        alphas, foregrounds = _run_matanyone(state, item.erode, item.dilate)

    # Phase 3: Encode output videos
    tp(0, "编码视频...")
    session_dir = WORKSPACE_DIR / "sessions" / item.session_id
    alpha_path = session_dir / "alpha.mp4"
    fgr_path = session_dir / "foreground.mp4"

    log.info("编码视频 (session=%s)...", item.session_id)
    alpha_rgb = np.repeat(alphas, 3, axis=3)
    encode_video(alpha_rgb, alpha_path, item.fps)
    encode_video(foregrounds, fgr_path, item.fps)

    end_ts = time.time()
    end_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    processing_time = end_ts - start_ts

    # Phase 4: Upload and notify
    tp(0, "上传至 R2...")
    log.info("上传至 R2 (session=%s)...", item.session_id)
    try:
        upload_and_notify(
            state, item.erode, item.dilate, session_dir,
            processing_time, start_dt, end_dt,
        )
    except Exception:
        log.exception("Post-matting upload/notify failed for session=%s", item.session_id)

    tp(1.0, "完成")
    log.info("任务完成 (session=%s, %.1fs)", item.session_id, processing_time)
    return processing_time


def on_execute_queue(
    _queue_progress: str,
    queue_state: list[QueueItem],
    progress=gr.Progress(track_tqdm=True),
):
    """Execute all pending queue items sequentially.

    Real-time progress is shown via ``gr.Progress`` (progress bar) and
    tqdm interception (``track_tqdm=True``).  The queue table and
    summary markdown are updated once at the end.

    Note: ``_queue_progress`` is an unused visible-component input
    required to work around a Gradio 6 issue where event handlers
    with only ``gr.State`` inputs receive no data from the frontend.

    Returns:
        Tuple of (queue_state, queue_status, queue_table, queue_progress).
    """
    pending = [item for item in queue_state if item.status == "pending"]
    if not pending:
        gr.Warning("队列中没有待处理的任务。")
        return (
            queue_state,
            _queue_status_text(queue_state),
            _queue_table_data(queue_state),
            "没有待处理的任务。",
        )

    _clear_processing_log()
    total = len(pending)
    log.info("========== 开始执行队列 (%d 个任务) ==========", total)

    done_count = 0
    error_count = 0
    timings: list[str] = []

    for i, item in enumerate(pending, start=1):
        video_name = item.original_filename or Path(item.source_video_path).name
        log.info(
            "--- 任务 %d/%d: session=%s, video=%s ---",
            i, total, item.session_id, video_name,
        )
        item.status = "processing"

        try:
            elapsed = _execute_single_item(item, progress, i, total)
            item.status = "done"
            done_count += 1
            timings.append(f"{video_name}: {elapsed:.1f}s")
        except Exception as exc:
            item.status = "error"
            item.error_msg = str(exc)
            error_count += 1
            log.exception("任务失败 (session=%s): %s", item.session_id, exc)
            notify_failure(item.session_id, exc)
            timings.append(f"{video_name}: 失败")

    # Final summary
    summary_parts = [f"**队列执行完毕**: {done_count} 成功"]
    if error_count:
        summary_parts[0] += f", {error_count} 失败"
    for t in timings:
        summary_parts.append(f"- {t}")
    summary = "\n".join(summary_parts)

    log.info("========== 队列执行完毕: %d 成功, %d 失败 ==========", done_count, error_count)

    return (
        list(queue_state),
        _queue_status_text(queue_state),
        _queue_table_data(queue_state),
        summary,
    )
