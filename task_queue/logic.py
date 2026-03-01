"""UI-agnostic queue logic for SplazMatte.

All functions return structured dicts. No Gradio or NiceGUI imports.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Callable

from config import (
    MATTING_SESSIONS_DIR,
    TRACKING_SESSIONS_DIR,
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    DEFAULT_MATTING_ENGINE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
)
from matting.runner import execute_queue, request_queue_cancel
from task_queue.models import QueueItem, load_queue, save_queue
from matting.session_store import empty_state, load_session, save_session_state
from tracking.session_store import (
    load_tracking_session,
    read_tracking_session_info,
    save_tracking_session,
)
from utils.notify import upload_and_notify

from matting.logic import clear_processing_log, keyframe_display, keyframe_gallery, render_frame

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None] | None


def read_item_info(item: QueueItem) -> dict | None:
    """Read lightweight session info for a queue item.

    Dispatches to tracking or matting session stores based on item type.

    Args:
        item: Queue item dict with "type" and "sid" keys.

    Returns:
        Merged meta + state dict, or None on failure.
    """
    if item["type"] == "tracking":
        return read_tracking_session_info(item["sid"])
    # Default: matting session
    session_dir = MATTING_SESSIONS_DIR / item["sid"]
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


def queue_status_text(queue: list[QueueItem]) -> str:
    """Build status string from the queue list."""
    if not queue:
        return "队列为空。"
    pending = done = error = 0
    for item in queue:
        info = read_item_info(item)
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


def queue_table_rows(queue: list[QueueItem]) -> list[list]:
    """Build table rows from the queue list.

    Columns: #, 类型, 文件名, 帧数, 关键帧, 引擎/模式, 状态
    """
    rows = []
    for i, item in enumerate(queue, start=1):
        info = read_item_info(item)
        task_type = item["type"]
        type_label = "追踪" if task_type == "tracking" else "抠像"
        if info is None:
            rows.append([i, type_label, item["sid"], 0, 0, "-", "丢失"])
            continue
        video_name = info.get("original_filename", item["sid"])
        num_frames = info.get("num_frames", 0)
        kf_indices = info.get("keyframe_indices", [])
        status = info.get("task_status", "pending")

        if task_type == "tracking":
            total_pts = info.get("total_points", 0)
            mode = f"点x{total_pts}"
        else:
            engine = info.get("matting_engine", DEFAULT_MATTING_ENGINE)
            mode = "MA" if engine == "MatAnyone" else "VM"
        rows.append([i, type_label, video_name, num_frames, len(kf_indices), mode, status])
    return rows


def load_queue_display() -> dict[str, Any]:
    """Load queue from disk and return display data."""
    queue = load_queue()
    return {
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
    }


def _maybe_add_tracking_task(session_state: dict, queue: list) -> str:
    """若 matting session 有设置追踪关键帧，则创建并添加追踪任务。

    Args:
        session_state: 已完成保存的 matting session state。
        queue: 当前队列列表（会被原地追加）。

    Returns:
        新建追踪 session 的 ID，若未创建则返回空字符串。
    """
    tracking_kp = session_state.get("tracking_keypoints", {})
    if not tracking_kp:
        return ""

    from tracking.logic import preprocess_video, empty_tracking_state
    import colorsys

    video_path = session_state.get("source_video_path")
    if not video_path or not Path(video_path).exists():
        log.warning("No source video for tracking task, skip.")
        return ""

    # 预处理视频，创建追踪 session
    tracking_state = empty_tracking_state()
    result = preprocess_video(str(video_path), tracking_state)
    if result.get("notify", ("", ""))[0] == "error":
        log.warning("preprocess_video failed for tracking task: %s", result)
        return ""

    tracking_state = result["session_state"]
    # 使用抠像任务的原始视频名，而非本地保存的 source.mp4
    tracking_state["original_filename"] = session_state.get(
        "original_filename", Path(str(video_path)).name
    )

    # 坐标转换：原始分辨率 → 追踪预览分辨率
    orig_h = session_state.get("video_height", 0)
    orig_w = session_state.get("video_width", 0)
    if orig_h == 0 or orig_w == 0:
        log.warning("Invalid video dimensions, skip tracking task.")
        return ""

    prev_h, prev_w = tracking_state["preview_size"]
    if prev_h == 0 or prev_w == 0:
        log.warning("Invalid preview dimensions, skip tracking task.")
        return ""
    num_frames = tracking_state["num_frames"]

    for frame_idx_raw, pts in tracking_kp.items():
        frame_idx = int(frame_idx_raw)
        if frame_idx >= num_frames:
            continue

        converted = []
        colors = []
        for i, (x, y) in enumerate(pts):
            tx = float(x) * prev_w / orig_w
            ty = float(y) * prev_h / orig_h
            converted.append((tx, ty, frame_idx))
            hue = (i / max(len(pts), 1)) % 1.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.9)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))

        tracking_state["query_points"][frame_idx] = converted
        tracking_state["query_colors"][frame_idx] = colors
        tracking_state["keyframes"][frame_idx] = {
            "points": converted,
            "colors": colors,
        }

    tracking_state["query_count"] = sum(
        len(v) for v in tracking_state["query_points"] if v
    )
    tracking_state["task_status"] = "pending"
    tracking_state["error_msg"] = ""
    save_tracking_session(tracking_state)

    tsid = tracking_state["session_id"]
    new_item: QueueItem = {"type": "tracking", "sid": tsid}
    if not any(item["sid"] == tsid for item in queue):
        queue.append(new_item)

    log.info(
        "Auto-added tracking task: sid=%s, keyframes=%d",
        tsid,
        len(tracking_kp),
    )
    return tsid


def add_to_queue(
    matting_engine: str,
    erode: int,
    dilate: int,
    batch_size: int,
    overlap: int,
    seed: int,
    session_state: dict,
    queue_state: list[QueueItem],
) -> dict[str, Any]:
    """Save matting params to session, add to queue, return reset UI data."""
    if not session_state.get("keyframes"):
        queue = load_queue()
        return {
            "session_state": session_state,
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": "请至少保存一个关键帧后再添加到队列。",
        }

    session_state["matting_engine"] = matting_engine
    session_state["erode"] = int(erode)
    session_state["dilate"] = int(dilate)
    session_state["batch_size"] = int(batch_size)
    session_state["overlap"] = int(overlap)
    session_state["seed"] = int(seed)
    session_state["task_status"] = "pending"
    session_state["error_msg"] = ""
    save_session_state(session_state)

    queue = load_queue()
    sid = session_state["session_id"]
    new_item: QueueItem = {"type": "matting", "sid": sid}
    if not any(item["sid"] == sid for item in queue):
        queue.append(new_item)
    linked_tsid = _maybe_add_tracking_task(session_state, queue)
    save_queue(queue)

    # Write back-reference only after queue is successfully persisted
    if linked_tsid:
        session_state["linked_tracking_sid"] = linked_tsid
        save_session_state(session_state)

    log.info(
        "Added to queue: session=%s, video=%s, keyframes=%d, engine=%s",
        sid,
        session_state.get("original_filename", ""),
        len(session_state["keyframes"]),
        matting_engine,
    )

    new_state = empty_state()
    return {
        "session_state": new_state,
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
        "frame_image": None,
        "slider_visible": False,
        "slider_value": 0,
        "frame_label": "请先上传视频。",
        "keyframe_info": "尚未保存任何关键帧。",
        "keyframe_gallery": [],
        "video_path": None,
        "propagation_preview_path": None,
        "alpha_path": None,
        "fgr_path": None,
    }


def add_tracking_to_queue(
    session_state: dict,
    queue_state: list[QueueItem],
) -> dict[str, Any]:
    """Save tracking session, add to queue, return reset UI data.

    Args:
        session_state: Current tracking state dict.
        queue_state: Current queue items.

    Returns:
        Dict with session_state (reset), queue_state, queue display data.
    """
    from tracking.logic import empty_tracking_state

    if not session_state.get("keyframes"):
        queue = load_queue()
        return {
            "session_state": session_state,
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": "请至少保存一个关键帧后再添加到队列。",
        }

    session_state["task_status"] = "pending"
    session_state["error_msg"] = ""
    save_tracking_session(session_state)

    queue = load_queue()
    sid = session_state["session_id"]
    new_item: QueueItem = {"type": "tracking", "sid": sid}
    if not any(item["sid"] == sid for item in queue):
        queue.append(new_item)
    save_queue(queue)

    log.info(
        "Added tracking to queue: session=%s, video=%s, keyframes=%d",
        sid,
        session_state.get("original_filename", ""),
        len(session_state["keyframes"]),
    )

    new_state = empty_tracking_state()
    return {
        "session_state": new_state,
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
    }


def remove_from_queue(remove_idx: int, queue_state: list[QueueItem]) -> dict[str, Any]:
    """Remove a queue item by 1-based index."""
    queue = load_queue()
    idx = int(remove_idx) - 1
    if idx < 0 or idx >= len(queue):
        return {
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": f"无效序号: {int(remove_idx)}，队列共 {len(queue)} 项。",
        }

    removed = queue[idx]
    queue = [*queue[:idx], *queue[idx + 1:]]
    save_queue(queue)
    log.info("Removed from queue: %s session=%s", removed["type"], removed["sid"])
    return {
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
    }


def pin_to_top_from_queue(pin_idx: int, queue_state: list[QueueItem]) -> dict[str, Any]:
    """Move a queue item to position 0 (pin to top) by 1-based index."""
    queue = load_queue()
    idx = int(pin_idx) - 1
    if idx < 0 or idx >= len(queue):
        return {
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": f"无效序号: {int(pin_idx)}，队列共 {len(queue)} 项。",
        }
    if idx == 0:
        return {
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "info": "该任务已在队列顶部。",
        }
    item = queue[idx]
    queue = [item, *queue[:idx], *queue[idx + 1:]]
    save_queue(queue)
    log.info("Pinned to top: %s session=%s", item["type"], item["sid"])
    return {
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
        "info": "已置顶。",
    }


def pin_to_bottom_from_queue(pin_idx: int, queue_state: list[QueueItem]) -> dict[str, Any]:
    """Move a queue item to the last position (pin to bottom) by 1-based index."""
    queue = load_queue()
    idx = int(pin_idx) - 1
    if idx < 0 or idx >= len(queue):
        return {
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": f"无效序号: {int(pin_idx)}，队列共 {len(queue)} 项。",
        }
    if idx == len(queue) - 1:
        return {
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "info": "该任务已在队列底部。",
        }
    item = queue[idx]
    queue = [*queue[:idx], *queue[idx + 1:], item]
    save_queue(queue)
    log.info("Pinned to bottom: %s session=%s", item["type"], item["sid"])
    return {
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
        "info": "已置底。",
    }


def clear_queue(queue_state: list[QueueItem]) -> dict[str, Any]:
    """Clear all items from the queue."""
    save_queue([])
    log.info("Queue cleared")
    return {
        "queue_state": [],
        "queue_status_text": queue_status_text([]),
        "queue_table_rows": [],
    }


def restore_from_queue(
    restore_idx: int,
    session_state: dict,
    queue_state: list[QueueItem],
) -> dict[str, Any]:
    """Restore a queue item into the editing area.

    Returns different keys depending on item type (matting vs tracking).
    The UI layer checks ``"restore_type"`` to decide which page to update.
    """
    queue = load_queue()
    idx = int(restore_idx) - 1
    if idx < 0 or idx >= len(queue):
        return {
            "session_state": session_state,
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": f"无效序号: {int(restore_idx)}，队列共 {len(queue)} 项。",
        }

    item = queue[idx]
    sid = item["sid"]
    task_type = item["type"]

    if task_type == "tracking":
        return _restore_tracking_item(sid, session_state, queue)

    return _restore_matting_item(sid, session_state, queue)


def _restore_matting_item(
    sid: str, session_state: dict, queue: list[QueueItem],
) -> dict[str, Any]:
    """Restore a matting queue item into the editing area."""
    loaded = load_session(sid)
    if loaded is None:
        return {
            "session_state": session_state,
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": f"无法加载 Session: {sid}",
        }

    frames_dir = loaded["frames_dir"]
    if not Path(frames_dir).exists():
        return {
            "session_state": session_state,
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": "帧数据目录不存在，无法恢复编辑。",
        }

    first_kf = min(loaded["keyframes"].keys()) if loaded["keyframes"] else 0
    loaded["current_frame_idx"] = first_kf
    loaded["current_mask"] = loaded["keyframes"].get(first_kf)

    session_dir = MATTING_SESSIONS_DIR / sid
    preview_path = session_dir / "propagation_preview.mp4"
    prop_preview = str(preview_path) if preview_path.exists() else None
    alpha_path = session_dir / "alpha.mp4"
    fgr_path = session_dir / "foreground.mp4"
    alpha_video = str(alpha_path) if alpha_path.exists() else None
    fgr_video = str(fgr_path) if fgr_path.exists() else None
    is_ma = loaded.get("matting_engine", DEFAULT_MATTING_ENGINE) == "MatAnyone"

    log.info("Restored matting queue item for editing: session=%s", sid)

    return {
        "restore_type": "matting",
        "session_state": loaded,
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
        "frame_image": render_frame(loaded),
        "slider_visible": True,
        "slider_max": loaded["num_frames"] - 1,
        "slider_value": first_kf,
        "frame_label": f"第 {first_kf} 帧 / 共 {loaded['num_frames'] - 1} 帧",
        "keyframe_info": keyframe_display(loaded),
        "keyframe_gallery": keyframe_gallery(loaded),
        "video_path": str(loaded["source_video_path"]) if loaded["source_video_path"] else None,
        "model_type": loaded["model_type"],
        "text_prompt_visible": loaded["model_type"] == "SAM3",
        "propagation_preview_path": prop_preview,
        "matting_engine": loaded.get("matting_engine", DEFAULT_MATTING_ENGINE),
        "erode": loaded.get("erode", DEFAULT_ERODE),
        "dilate": loaded.get("dilate", DEFAULT_DILATE),
        "batch_size": loaded.get("batch_size", VIDEOMAMA_BATCH_SIZE),
        "overlap": loaded.get("overlap", VIDEOMAMA_OVERLAP),
        "seed": loaded.get("seed", VIDEOMAMA_SEED),
        "erode_dilate_visible": is_ma,
        "vm_params_visible": not is_ma,
        "alpha_path": alpha_video,
        "fgr_path": fgr_video,
    }


def _restore_tracking_item(
    sid: str, session_state: dict, queue: list[QueueItem],
) -> dict[str, Any]:
    """Restore a tracking queue item into the editing area."""
    loaded = load_tracking_session(sid)
    if loaded is None:
        return {
            "session_state": session_state,
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": f"无法加载追踪 Session: {sid}",
        }

    from tracking.logic import tracking_keyframe_display, tracking_keyframe_gallery

    first_kf = min(loaded["keyframes"].keys()) if loaded.get("keyframes") else 0
    num_frames = loaded.get("num_frames", 0)

    log.info("Restored tracking queue item for editing: session=%s", sid)

    return {
        "restore_type": "tracking",
        "session_state": loaded,
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
        "slider_max": max(num_frames - 1, 0),
        "slider_value": first_kf,
        "keyframe_info": tracking_keyframe_display(loaded),
        "keyframe_gallery": tracking_keyframe_gallery(loaded),
    }


def _pack_results_zip(queue: list[QueueItem]) -> Path | None:
    """Pack output files from done sessions into a zip."""
    zip_path = WORKSPACE_DIR / "results.zip"
    packed = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in queue:
            info = read_item_info(item)
            if info is None or info.get("task_status") != "done":
                continue
            sid = item["sid"]
            video_name = Path(info.get("original_filename", sid)).stem

            if item["type"] == "tracking":
                # Pack tracking results
                results_dir = WORKSPACE_DIR / "tracking_results"
                session_dir = TRACKING_SESSIONS_DIR / sid
                # Result video + AE export
                for pattern in ("tracking_*.mp4", "ae_export_*.txt"):
                    for f in results_dir.glob(pattern):
                        zf.write(f, f"{video_name}_tracking/{f.name}")
                        packed += 1
            else:
                # Pack matting results
                session_dir = MATTING_SESSIONS_DIR / sid
                for filename in ("alpha.mp4", "foreground.mp4"):
                    src = session_dir / filename
                    if src.exists():
                        zf.write(src, f"{video_name}/{filename}")
                        packed += 1

                source_video_path = info.get("source_video_path")
                if source_video_path:
                    source_video = Path(source_video_path)
                    if source_video.exists() and source_video.is_file():
                        original_name = info.get("original_filename") or source_video.name
                        original_suffix = Path(original_name).suffix or source_video.suffix or ".mp4"
                        zf.write(source_video, f"{video_name}/original{original_suffix}")
                        packed += 1
    if packed == 0:
        zip_path.unlink(missing_ok=True)
        return None
    return zip_path


def pack_download(queue_state: list[QueueItem]) -> dict[str, Any]:
    """Pack results zip. Returns download_path or warning."""
    queue = load_queue()
    zip_path = _pack_results_zip(queue)
    if zip_path is None:
        return {"download_path": None, "warning": "没有可打包的结果（队列中无已完成的任务）。"}
    return {"download_path": str(zip_path)}


def reset_status(queue_state: list[QueueItem]) -> dict[str, Any]:
    """Reset all queue tasks to pending."""
    queue = load_queue()
    if not queue:
        return {
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "warning": "队列为空。",
        }

    count = 0
    for item in queue:
        if item["type"] == "tracking":
            loaded = load_tracking_session(item["sid"])
            if loaded is None:
                continue
            loaded["task_status"] = "pending"
            loaded["error_msg"] = ""
            save_tracking_session(loaded)
        else:
            loaded = load_session(item["sid"])
            if loaded is None:
                continue
            loaded["task_status"] = "pending"
            loaded["error_msg"] = ""
            save_session_state(loaded)
        count += 1

    return {
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
        "info": f"已重置 {count} 个任务为待处理状态。",
    }


def send_feishu(queue_state: list[QueueItem]) -> dict[str, Any]:
    """Send Feishu notification for each completed task."""
    queue = load_queue()
    done_items = [
        item for item in queue
        if (info := read_item_info(item)) and info.get("task_status") == "done"
    ]

    if not done_items:
        return {"warning": "队列中没有已完成的任务，无法发送通知。"}

    success = 0
    failed = 0
    for item in done_items:
        sid = item["sid"]
        try:
            if item["type"] == "tracking":
                loaded = load_tracking_session(sid)
                if loaded is None:
                    failed += 1
                    continue
                from tracking.runner import upload_and_notify_tracking
                upload_and_notify_tracking(
                    loaded,
                    loaded.get("processing_time", 0.0),
                    loaded.get("start_time", "N/A"),
                    loaded.get("end_time", "N/A"),
                )
            else:
                loaded = load_session(sid)
                if loaded is None:
                    failed += 1
                    continue
                erode = int(loaded.get("erode", DEFAULT_ERODE))
                dilate = int(loaded.get("dilate", DEFAULT_DILATE))
                session_dir = MATTING_SESSIONS_DIR / sid
                upload_and_notify(
                    loaded, erode, dilate, session_dir,
                    loaded.get("processing_time", 0.0),
                    loaded.get("start_time", "N/A"),
                    loaded.get("end_time", "N/A"),
                )
            success += 1
        except Exception:
            log.exception("Feishu notify failed for session %s", sid)
            failed += 1

    if failed:
        return {"info": f"飞书通知: {success} 条发送成功，{failed} 条失败。"}
    return {"info": f"飞书通知: {success} 条全部发送成功。"}


def stop_queue() -> dict[str, Any]:
    """Request cancellation of the running queue execution.

    Stops the current task at the next progress checkpoint (typically within
    a few seconds for tracking, or per-batch for matting).
    """
    request_queue_cancel()
    return {"info": "已请求停止，正在中断当前任务…"}


def run_execute_queue(progress_callback: ProgressCallback = None) -> dict[str, Any]:
    """Execute all pending queue items. Long-running; use progress_callback for UI."""
    queue = load_queue()
    has_pending = any(
        (info := read_item_info(item)) is not None
        and info.get("task_status", "") in ("pending", "")
        for item in queue
    )
    if not has_pending:
        return {
            "queue_state": queue,
            "queue_status_text": queue_status_text(queue),
            "queue_table_rows": queue_table_rows(queue),
            "queue_progress_text": "没有待处理的任务。",
            "warning": "队列中没有待处理的任务。",
        }

    clear_processing_log()

    def progress_cb(frac: float, desc: str = ""):
        if progress_callback:
            progress_callback(frac, desc)

    done_count, error_count, timings = execute_queue(progress_cb)

    summary_parts = [f"队列执行完毕: {done_count} 成功"]
    if error_count:
        summary_parts[0] += f", {error_count} 失败"
    for t in timings:
        summary_parts.append(f"- {t}")
    summary = "\n".join(summary_parts)

    queue = load_queue()
    return {
        "queue_state": queue,
        "queue_status_text": queue_status_text(queue),
        "queue_table_rows": queue_table_rows(queue),
        "queue_progress_text": summary,
    }
