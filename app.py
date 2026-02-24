"""SplazMatte — NiceGUI web app for MatAnyone matting with SAM2/SAM3 multi-frame annotation."""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

import asyncio
import logging
import platform
import queue
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import numpy as np

from nicegui import app, events, ui
from nicegui import run

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    LOGS_DIR,
    MATTING_SESSIONS_DIR,
    PROCESSING_LOG_FILE,
    TRACKING_SESSIONS_DIR,
    SERVER_PORT,
    STORAGE_SECRET,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
    get_device,
)
from session_store import empty_state, list_sessions, load_session
from queue_models import load_queue
from app_logic import (
    keyframe_display,
    keyframe_gallery,
    model_change,
    refresh_sessions,
    restore_session,
    run_propagation,
    save_keyframe,
    slider_change,
    start_matting,
    text_prompt,
    upload_video,
    clear_clicks,
    delete_keyframe,
    frame_click,
    undo_click,
)
from queue_logic import (
    add_to_queue,
    add_tracking_to_queue,
    clear_queue,
    load_queue_display,
    pack_download,
    queue_status_text,
    queue_table_rows,
    remove_from_queue,
    reset_status,
    restore_from_queue,
    run_execute_queue,
    send_feishu,
    stop_queue,
)
from gpu_lock import try_acquire_gpu, release_gpu, get_gpu_status
from utils.feishu_notify import send_feishu_startup
from engines.cotracker_engine import TrackingCancelledError
from matting_runner import MattingCancelledError
from cotracker_logic import (
    empty_tracking_state,
    preprocess_video as ct_preprocess_video,
    refresh_tracking_sessions as ct_refresh_sessions,
    restore_tracking_session as ct_restore_session,
    change_frame as ct_change_frame,
    add_point as ct_add_point,
    undo_point as ct_undo_point,
    clear_frame_points as ct_clear_frame_points,
    clear_all_points as ct_clear_all_points,
    run_tracking as ct_run_tracking,
    sam_click as ct_sam_click,
    sam_undo as ct_sam_undo,
    sam_clear as ct_sam_clear,
    generate_points_from_mask as ct_generate_points_from_mask,
    save_tracking_keyframe as ct_save_kf,
    delete_tracking_keyframe as ct_del_kf,
    tracking_keyframe_display as ct_kf_display,
    tracking_keyframe_gallery as ct_kf_gallery,
)
from tracking_export import export_ae_keyframe_data as ct_ae_export
from tracking_session_store import list_tracking_sessions, save_tracking_session as ct_save_session

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
log = logging.getLogger(__name__)

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
MATTING_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
preview_dir = WORKSPACE_DIR / "preview"
preview_dir.mkdir(exist_ok=True)
tracking_preview_dir = WORKSPACE_DIR / "tracking_preview"
tracking_preview_dir.mkdir(exist_ok=True)
tracking_results_dir = WORKSPACE_DIR / "tracking_results"
tracking_results_dir.mkdir(exist_ok=True)
app.add_static_files("/sessions", str(MATTING_SESSIONS_DIR))
app.add_static_files("/preview", str(preview_dir))
app.add_static_files("/workspace", str(WORKSPACE_DIR))
app.add_static_files("/tracking_preview", str(tracking_preview_dir))
app.add_static_files("/tracking_results", str(tracking_results_dir))


def _session_path_to_url(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        rel = p.relative_to(MATTING_SESSIONS_DIR)
        return "/sessions/" + str(rel).replace("\\", "/")
    except ValueError:
        return path


def _workspace_path_to_url(path: str | Path | None) -> str:
    """Convert a path under workspace to a URL for browser access."""
    if not path:
        return ""
    p = Path(path)
    try:
        rel = p.relative_to(WORKSPACE_DIR)
        return "/workspace/" + str(rel).replace("\\", "/")
    except ValueError:
        return str(path)


def _get_user_email_from_request(request) -> str:
    """Get user email from Cloudflare Access header.
    
    When deployed behind Cloudflare Access, the authenticated user's email
    is available in the 'Cf-Access-Authenticated-User-Email' header.
    For local development without Cloudflare, returns 'admin' as fallback.
    
    Args:
        request: FastAPI Request object from client.request
    """
    if request is not None:
        email = request.headers.get("Cf-Access-Authenticated-User-Email")
        if email:
            return email
    return "admin"


def _get_user_id_from_request(request) -> str:
    """Get unique user ID based on email."""
    email = _get_user_email_from_request(request)
    # Use email as user ID (sanitized for filesystem safety)
    return email.replace("@", "_at_").replace(".", "_")


def _get_user_name_from_request(request) -> str:
    """Get user display name (email or admin)."""
    return _get_user_email_from_request(request)


def _get_session_state() -> dict:
    """Get or create session state for current user.
    
    Session state is stored in memory per-page (in refs), but the session_id
    is persisted in app.storage.user for restoration across page reloads.
    """
    session_id = app.storage.user.get("current_session_id")
    if session_id:
        state = load_session(session_id)
        if state:
            return state
    return empty_state()


def _save_session_id(session_id: str) -> None:
    """Persist current session_id for the user."""
    app.storage.user["current_session_id"] = session_id


def _write_frame_preview(frame: np.ndarray, user_id: str) -> str:
    """Write frame preview to user-specific directory."""
    if frame is None or frame.size == 0:
        return ""
    user_preview_dir = preview_dir / user_id
    user_preview_dir.mkdir(exist_ok=True)
    path = user_preview_dir / "current.png"
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return f"/preview/{user_id}/current.png?t={id(frame)}"


def _build_queue_rows_with_sid() -> list[dict]:
    """Build table rows including session_id for row actions."""
    q = load_queue()
    raw = queue_table_rows(q)
    rows = []
    for i, r in enumerate(raw, start=1):
        item = q[i - 1] if i <= len(q) else {"type": "", "sid": ""}
        rows.append({
            "idx": r[0],
            "type": r[1],
            "video": r[2],
            "frames": r[3],
            "kf": r[4],
            "mode": r[5],
            "status": r[6],
            "session_id": item["sid"] if isinstance(item, dict) else str(item),
            "task_type": item["type"] if isinstance(item, dict) else "matting",
        })
    return rows


def _apply_restore_out(
    out: dict,
    refs: dict,
    user_id: str,
    session_state: dict,
    on_click_frame: callable = None,
) -> dict:
    """Apply restore output and return updated session_state."""
    if "session_state" in out:
        session_state = out["session_state"]
        if session_state.get("session_id"):
            _save_session_id(session_state["session_id"])
    if out.get("frame_image") is not None:
        refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
    if out.get("frame_label") is not None:
        refs["frame_label"].set_text(out["frame_label"])
    if out.get("keyframe_info") is not None:
        refs["keyframe_info"].set_text(out["keyframe_info"])
    if out.get("keyframe_gallery") is not None:
        _refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, on_click_frame)
    if out.get("video_path") is not None:
        refs["video_display"].set_source(_session_path_to_url(out["video_path"]))
    if out.get("slider_visible") is not None:
        refs["frame_slider"].set_visibility(out["slider_visible"])
    if out.get("slider_max") is not None:
        refs["frame_slider"].props["max"] = out["slider_max"]
    if out.get("slider_value") is not None:
        refs["frame_slider"].value = out["slider_value"]
    if out.get("model_type") is not None:
        refs["model_selector"].value = out["model_type"]
    if out.get("text_prompt_visible") is not None:
        refs["text_prompt_row"].set_visibility(out["text_prompt_visible"])
    if out.get("propagation_preview_path") is not None:
        refs["propagation_preview"].set_source(_session_path_to_url(out["propagation_preview_path"]))
    if out.get("matting_engine") is not None:
        refs["matting_engine_selector"].value = out["matting_engine"]
    if out.get("erode") is not None:
        refs["erode_slider"].value = out["erode"]
    if out.get("dilate") is not None:
        refs["dilate_slider"].value = out["dilate"]
    if out.get("batch_size") is not None:
        refs["vm_batch_slider"].value = out["batch_size"]
    if out.get("overlap") is not None:
        refs["vm_overlap_slider"].value = out["overlap"]
    if out.get("seed") is not None:
        refs["vm_seed_input"].value = out["seed"]
    if out.get("erode_dilate_visible") is not None:
        refs["erode_row"].set_visibility(out["erode_dilate_visible"])
    if out.get("vm_params_visible") is not None:
        refs["vm_params_row"].set_visibility(out["vm_params_visible"])
    if out.get("alpha_path") is not None:
        refs["alpha_video"].set_source(_session_path_to_url(out["alpha_path"]))
    if out.get("fgr_path") is not None:
        refs["fgr_video"].set_source(_session_path_to_url(out["fgr_path"]))
    if out.get("session_choices") is not None:
        refs["session_dropdown"].options = {v: l for l, v in out["session_choices"]}
    if out.get("session_value") is not None:
        refs["session_dropdown"].value = out["session_value"]
    return session_state


def _refresh_gallery(
    container: ui.element,
    items: list,
    user_id: str,
    on_click_frame: callable = None,
) -> None:
    """Refresh keyframe gallery with clickable thumbnails.
    
    Args:
        container: The UI container to populate.
        items: List of (image, caption) tuples. Caption format: "第 N 帧".
        user_id: User ID for preview path isolation.
        on_click_frame: Optional callback(frame_idx) when a keyframe is clicked.
    """
    import re
    import time
    
    user_preview_dir = preview_dir / user_id
    user_preview_dir.mkdir(exist_ok=True)
    
    # Clean up old keyframe previews for this user
    for old_file in user_preview_dir.glob("kf_*.png"):
        try:
            old_file.unlink()
        except OSError:
            pass
    
    container.clear()
    timestamp = int(time.time() * 1000)
    
    with container:
        for i, (img, caption) in enumerate(items):
            # Extract frame index from caption like "第 N 帧"
            frame_idx = None
            match = re.search(r"第\s*(\d+)\s*帧", caption)
            if match:
                frame_idx = int(match.group(1))
            
            # Use index-based filename (fixed, not incrementing)
            path = user_preview_dir / f"kf_{i}.png"
            if img is not None and img.size > 0:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            with ui.column().classes("items-center cursor-pointer hover:bg-gray-100 rounded p-1"):
                img_url = f"/preview/{user_id}/kf_{i}.png?t={timestamp}" if img is not None and path.exists() else ""
                # Use fixed size for consistent display
                kf_image = ui.image(img_url).classes("w-20 h-14 object-cover")
                kf_label = ui.label(caption).classes("text-xs")
                
                # Make the whole column clickable to jump to that frame
                if frame_idx is not None and on_click_frame is not None:
                    # Capture frame_idx in closure; use asyncio for async callback
                    def make_click_handler(fidx):
                        def handler():
                            asyncio.create_task(on_click_frame(fidx))
                        return handler
                    kf_image.on("click", make_click_handler(frame_idx))
                    kf_label.on("click", make_click_handler(frame_idx))


def _apply_notify(out: dict) -> None:
    if out.get("warning"):
        ui.notify(out["warning"], type="warning")
    if out.get("info"):
        ui.notify(out["info"], type="positive")


def _update_queue_ui(refs: dict) -> None:
    if "queue_status" not in refs or "queue_table" not in refs:
        return
    q = load_queue()
    refs["queue_status"].set_text(queue_status_text(q))
    refs["queue_table"].rows = _build_queue_rows_with_sid()
    if "queue_pending_count" in refs:
        rows = refs["queue_table"].rows
        pending = sum(1 for row in rows if row.get("status") == "pending")
        done = sum(1 for row in rows if row.get("status") == "done")
        failed = sum(1 for row in rows if row.get("status") == "failed")
        refs["queue_pending_count"].set_text(str(pending))
        refs["queue_done_count"].set_text(str(done))
        refs["queue_failed_count"].set_text(str(failed))


@ui.page("/")
def main_page(client):
    # Get user identity from request headers (Cloudflare Access)
    user_id = _get_user_id_from_request(client.request)
    user_name = _get_user_name_from_request(client.request)
    
    # Per-page session state (mutable dict that persists for this page instance)
    page_state = {"session": _get_session_state()}
    
    device = get_device()
    device_label = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
    hostname = platform.node() or "unknown"

    refs = {}
    page_timers = set()

    def _start_page_timer(interval: float, callback, *, once: bool = False, immediate: bool = True):
        """Start a per-page timer and track it for cleanup."""
        timer = app.timer(interval, callback, once=once, immediate=immediate)
        page_timers.add(timer)
        return timer

    def _stop_page_timer(timer) -> None:
        """Cancel a tracked page timer safely."""
        if timer is None:
            return
        timer.cancel(with_current_invocation=True)
        page_timers.discard(timer)

    def _cleanup_page_timers(*_) -> None:
        """Cancel all timers when page/client lifecycle ends."""
        for timer in list(page_timers):
            timer.cancel(with_current_invocation=True)
        page_timers.clear()

    def _start_until_false_timer(interval: float, poll_fn) -> None:
        """Start a timer and stop it when poll_fn returns False or errors."""
        timer = None

        def wrapped_poll():
            nonlocal timer
            try:
                keep_running = poll_fn()
            except RuntimeError as ex:
                log.warning("Stop poll timer due to runtime error: %s", ex)
                keep_running = False
            except Exception:
                log.exception("Stop poll timer due to unexpected polling error")
                keep_running = False
            if not keep_running and timer is not None:
                _stop_page_timer(timer)
                timer = None

        timer = _start_page_timer(interval, wrapped_poll)

    client.on_disconnect(_cleanup_page_timers)
    client.on_delete(_cleanup_page_timers)
    
    # Jump to a specific frame (used by keyframe gallery clicks)
    async def jump_to_frame(frame_idx: int):
        """Jump to a specific frame when clicking on a keyframe thumbnail."""
        out = await run.io_bound(slider_change, frame_idx, page_state["session"])
        page_state["session"] = out["session_state"]
        if out.get("frame_image") is not None:
            refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
        if out.get("frame_label"):
            refs["frame_label"].set_text(out["frame_label"])
        # Update the slider to match
        if "frame_slider" in refs:
            refs["frame_slider"].value = frame_idx

    # ---- Header ----
    with ui.row().classes("w-full items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.label("SplazMatte").classes("text-2xl font-bold")
            ui.link("抠像", "/").classes("text-blue-600 underline font-medium")
            ui.link("轨迹追踪", "/tracking").classes("text-gray-600 hover:text-blue-600")
        with ui.row().classes("items-center gap-4"):
            # GPU status indicator
            gpu_status_label = ui.label("").classes("text-xs")
            refs["gpu_status_label"] = gpu_status_label
            ui.label(f"用户: {user_name}").classes("text-sm text-gray-600")
            ui.label(f"运行设备: {device_label.get(device.type, device.type)} | 主机: {hostname}").classes("text-sm text-gray-600")
    
    # GPU status polling
    _last_gpu_text = ""

    def update_gpu_status():
        nonlocal _last_gpu_text
        status = get_gpu_status()
        if status["locked"]:
            if status["holder_id"] == user_id:
                text = f"GPU: 您正在使用 ({status['operation']})"
                remove, add = "text-red-500 text-green-500", "text-blue-500"
            else:
                text = f"GPU: {status['holder_name']} 使用中"
                remove, add = "text-blue-500 text-green-500", "text-red-500"
        else:
            text = "GPU: 空闲"
            remove, add = "text-red-500 text-blue-500", "text-green-500"
        if text != _last_gpu_text:
            _last_gpu_text = text
            gpu_status_label.set_text(text)
            gpu_status_label.classes(remove=remove, add=add)
    _start_page_timer(1.0, update_gpu_status)

    ui.separator()

    # ======================================================================
    # 区域 1：上传视频 / 恢复 Session
    # ======================================================================
    with ui.expansion("① 上传视频 / 恢复 Session", icon="folder_open").classes("w-full").props("default-opened"):
        ui.label("上传视频文件开始新任务，或从历史 Session 恢复之前的工作进度。").classes("text-xs text-gray-500 mb-2")
        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-1"):
                video_display = ui.video("").classes("max-w-full max-h-60")
                refs["video_display"] = video_display

                async def on_uploaded(e: events.UploadEventArguments):
                    upload_dir = WORKSPACE_DIR / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    dest = upload_dir / (e.file.name or "video")
                    save_fn = e.file.save(str(dest))
                    if asyncio.iscoroutine(save_fn):
                        await save_fn
                    try:
                        out = upload_video(str(dest), page_state["session"])
                    except Exception as ex:
                        log.exception("Upload video failed")
                        ui.notify(f"视频处理失败: {ex}", type="negative")
                        return
                    page_state["session"] = out["session_state"]
                    if page_state["session"].get("session_id"):
                        _save_session_id(page_state["session"]["session_id"])
                    if out.get("frame_image") is not None:
                        refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
                    if out.get("frame_label"):
                        refs["frame_label"].set_text(out["frame_label"])
                    if out.get("keyframe_info"):
                        refs["keyframe_info"].set_text(out["keyframe_info"])
                    if out.get("keyframe_gallery") is not None:
                        _refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, jump_to_frame)
                    if out.get("slider_visible") is not None:
                        refs["frame_slider"].set_visibility(out["slider_visible"])
                        refs["frame_slider"].props["max"] = out.get("slider_max", 0)
                        refs["frame_slider"].value = out.get("slider_value", 0)
                    if out.get("session_choices") is not None:
                        refs["session_dropdown"].options = {v: l for l, v in out["session_choices"]}
                        refs["session_dropdown"].value = out.get("session_value")
                    st = out["session_state"]
                    if st.get("session_id") and st.get("source_video_path"):
                        src = Path(st["source_video_path"])
                        refs["video_display"].set_source(f"/sessions/{st['session_id']}/{src.name}")
                    _apply_notify(out)

                ui.upload(on_upload=on_uploaded, max_file_size=500_000_000).props("accept='video/*'").classes("w-full")
                ui.label("支持 mp4/mov/avi 等常见格式，上传后自动提取帧。").classes("text-xs text-gray-400")

            with ui.column().classes("w-64"):
                sessions = list_sessions()
                session_dropdown = ui.select(
                    options={v: l for l, v in sessions},
                    value=None,
                    label="恢复 Session",
                ).classes("w-full")
                refs["session_dropdown"] = session_dropdown
                with ui.row():
                    def on_refresh():
                        out = refresh_sessions()
                        refs["session_dropdown"].options = {v: l for l, v in out["session_choices"]}
                    ui.button("刷新", on_click=on_refresh)
                    def on_restore():
                        sid = refs["session_dropdown"].value
                        out = restore_session(sid, page_state["session"])
                        page_state["session"] = _apply_restore_out(out, refs, user_id, page_state["session"], jump_to_frame)
                        _apply_notify(out)
                    ui.button("恢复", on_click=on_restore).props("color=primary")
                ui.label("选择历史 Session 可恢复之前的标注、传播、抠像结果。").classes("text-xs text-gray-400")

    ui.separator()

    # ======================================================================
    # 区域 2：标注关键帧
    # ======================================================================
    with ui.expansion("② 标注关键帧", icon="touch_app").classes("w-full").props("default-opened"):
        ui.label("在关键帧上点击标注目标对象，SAM 模型会自动分割。建议在目标出现/消失/遮挡变化的帧上标注。").classes("text-xs text-gray-500 mb-2")
        # 工具栏
        with ui.row().classes("gap-4 flex-wrap items-center mb-2"):
            with ui.column().classes("gap-0"):
                ui.label("标注模型").classes("text-xs text-gray-500")
                model_selector = ui.radio(["SAM2", "SAM3"], value="SAM2")
                refs["model_selector"] = model_selector
            with ui.column().classes("gap-0"):
                ui.label("点击模式").classes("text-xs text-gray-500")
                point_mode = ui.radio(["Positive", "Negative"], value="Positive")
            ui.label("Positive=选中目标，Negative=排除区域").classes("text-xs text-gray-400")

            async def on_undo():
                acquired, msg = try_acquire_gpu(user_id, user_name, "撤销标注")
                if not acquired:
                    ui.notify(msg, type="warning")
                    return
                if "annotation_loading_overlay" in refs:
                    refs["annotation_loading_overlay"].set_visibility(True)
                try:
                    out = await run.cpu_bound(undo_click, refs["model_selector"].value, page_state["session"])
                    page_state["session"] = out["session_state"]
                    if out.get("frame_image") is not None:
                        refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
                    _apply_notify(out)
                finally:
                    release_gpu(user_id)
                    if "annotation_loading_overlay" in refs:
                        refs["annotation_loading_overlay"].set_visibility(False)

            async def on_clear():
                if "annotation_loading_overlay" in refs:
                    refs["annotation_loading_overlay"].set_visibility(True)
                try:
                    out = await run.io_bound(clear_clicks, page_state["session"])
                    page_state["session"] = out["session_state"]
                    if out.get("frame_image") is not None:
                        refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
                    _apply_notify(out)
                finally:
                    if "annotation_loading_overlay" in refs:
                        refs["annotation_loading_overlay"].set_visibility(False)

            async def on_save_kf():
                out = await run.io_bound(save_keyframe, page_state["session"])
                page_state["session"] = out["session_state"]
                if out.get("keyframe_info"):
                    refs["keyframe_info"].set_text(out["keyframe_info"])
                if out.get("keyframe_gallery") is not None:
                    _refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, jump_to_frame)
                _apply_notify(out)

            async def on_del_kf():
                out = await run.io_bound(delete_keyframe, page_state["session"])
                page_state["session"] = out["session_state"]
                if out.get("frame_image") is not None:
                    refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
                if out.get("keyframe_info"):
                    refs["keyframe_info"].set_text(out["keyframe_info"])
                if out.get("keyframe_gallery") is not None:
                    _refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, jump_to_frame)
                _apply_notify(out)

            ui.button("撤销", on_click=on_undo)
            ui.button("清除", on_click=on_clear)
            ui.button("保存关键帧", on_click=on_save_kf).props("color=primary")
            ui.button("删除关键帧", on_click=on_del_kf)

        # Slider
        frame_slider = ui.slider(min=0, max=1, value=0, step=1).props("label-always").classes("w-full mb-2")
        refs["frame_slider"] = frame_slider
        async def on_slider_val():
            v = refs["frame_slider"].value
            if v is None:
                return
            out = await run.io_bound(slider_change, int(v), page_state["session"])
            page_state["session"] = out["session_state"]
            if out.get("frame_image") is not None:
                refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
            if out.get("frame_label"):
                refs["frame_label"].set_text(out["frame_label"])
        frame_slider.on("update:model-value", on_slider_val)

        # 帧标签
        frame_label = ui.label("请先上传视频。").classes("text-sm text-gray-600 mb-1")
        refs["frame_label"] = frame_label

        # 主内容区
        with ui.row().classes("w-full gap-4 items-start"):
            with ui.column().classes("flex-1"):
                click_state = {"busy": False}

                async def on_mouse(e: events.MouseEventArguments):
                    if e.type != "click" or not page_state["session"].get("frames_dir"):
                        return
                    if click_state["busy"]:
                        return
                    
                    # Try to acquire GPU lock
                    acquired, msg = try_acquire_gpu(user_id, user_name, "SAM 标注")
                    if not acquired:
                        ui.notify(msg, type="warning")
                        return
                    
                    click_state["busy"] = True
                    refs["annotation_loading_overlay"].set_visibility(True)
                    try:
                        out = await run.cpu_bound(
                            frame_click,
                            e.image_x,
                            e.image_y,
                            point_mode.value,
                            refs["model_selector"].value,
                            page_state["session"],
                        )
                        page_state["session"] = out["session_state"]
                        if out.get("frame_image") is not None:
                            refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
                        _apply_notify(out)
                    except Exception as ex:
                        log.exception("Frame annotation failed")
                        ui.notify(f"标注失败: {ex}", type="negative")
                    finally:
                        release_gpu(user_id)
                        refs["annotation_loading_overlay"].set_visibility(False)
                        click_state["busy"] = False

                with ui.element("div").classes("relative w-full"):
                    frame_image = ui.interactive_image("", on_mouse=on_mouse, events=["click"]).classes("w-full")
                    refs["frame_image"] = frame_image
                    # 加载遮罩层
                    loading_overlay = ui.element("div").classes(
                        "absolute inset-0 flex items-center justify-center bg-black/50 z-10 backdrop-blur-sm"
                    )
                    with loading_overlay:
                        with ui.card().classes("p-4 items-center"):
                            ui.spinner("dots", size="xl", color="primary")
                            ui.label("SAM 推理中...").classes("text-sm mt-2")
                            ui.label("请稍候").classes("text-xs text-gray-500")
                    loading_overlay.set_visibility(False)
                    refs["annotation_loading_overlay"] = loading_overlay

                # 文本提示词（SAM3 专用）
                text_prompt_row = ui.row().classes("w-full mt-2")
                refs["text_prompt_row"] = text_prompt_row
                with text_prompt_row:
                    text_prompt_input = ui.input(label="文本提示词", placeholder="person, car, dog...").classes("flex-1")

                    async def on_text_detect():
                        acquired, msg = try_acquire_gpu(user_id, user_name, "文本检测")
                        if not acquired:
                            ui.notify(msg, type="warning")
                            return
                        refs["annotation_loading_overlay"].set_visibility(True)
                        try:
                            out = await run.cpu_bound(text_prompt, text_prompt_input.value or "", page_state["session"])
                            page_state["session"] = out["session_state"]
                            if out.get("frame_image") is not None:
                                refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
                            _apply_notify(out)
                        except Exception as ex:
                            log.exception("Text detection failed")
                            ui.notify(f"文本检测失败: {ex}", type="negative")
                        finally:
                            release_gpu(user_id)
                            refs["annotation_loading_overlay"].set_visibility(False)

                    ui.button("检测", on_click=on_text_detect).props("color=primary")
                    ui.label("SAM3 支持文本描述检测，例如 'person in red'").classes("text-xs text-gray-400 self-center")
                text_prompt_row.set_visibility(False)

            with ui.column().classes("w-48"):
                ui.label("已保存的关键帧").classes("text-xs text-gray-500 font-medium")
                keyframe_info = ui.label("尚未保存任何关键帧。")
                refs["keyframe_info"] = keyframe_info
                keyframe_gallery_container = ui.column().classes("gap-1 max-h-[400px] overflow-auto")
                refs["keyframe_gallery_container"] = keyframe_gallery_container
                ui.label("建议标注 2~5 个关键帧，覆盖目标的不同姿态。").classes("text-xs text-gray-400 mt-1")

        # 模型切换回调
        async def on_model_val():
            acquired, msg = try_acquire_gpu(user_id, user_name, "模型切换")
            if not acquired:
                ui.notify(msg, type="warning")
                return
            if "annotation_loading_overlay" in refs:
                refs["annotation_loading_overlay"].set_visibility(True)
            try:
                out = await run.cpu_bound(model_change, refs["model_selector"].value, page_state["session"])
                page_state["session"] = out["session_state"]
                refs["text_prompt_row"].set_visibility(out.get("text_prompt_visible", False))
                if out.get("frame_image") is not None:
                    refs["frame_image"].set_source(_write_frame_preview(out["frame_image"], user_id))
            finally:
                release_gpu(user_id)
                if "annotation_loading_overlay" in refs:
                    refs["annotation_loading_overlay"].set_visibility(False)
        model_selector.on("update:model-value", on_model_val)

    ui.separator()

    # ======================================================================
    # 区域 3：传播 + 抠像 (左右并排)
    # ======================================================================
    with ui.row().classes("w-full gap-4"):
        # 传播
        with ui.expansion("③ 传播", icon="account_tree").classes("flex-1").props("default-opened"):
            ui.label("将关键帧标注传播到所有帧，生成完整的遮罩序列。").classes("text-xs text-gray-500 mb-2")
            propagation_preview = ui.video("").classes("max-h-48 w-full")
            refs["propagation_preview"] = propagation_preview
            progress_bar = ui.linear_progress(value=0).props("visible=false").classes("w-full")
            progress_label = ui.label("").classes("text-sm")

            def run_propagation_task():
                acquired, msg = try_acquire_gpu(user_id, user_name, "传播")
                if not acquired:
                    ui.notify(msg, type="warning")
                    return
                
                progress_q = queue.Queue()
                def cb(frac: float, desc: str):
                    progress_q.put(("progress", frac, desc))
                def blocking():
                    result = run_propagation(refs["model_selector"].value, page_state["session"], progress_callback=cb)
                    progress_q.put(("done", result))
                async def start():
                    await run.io_bound(blocking)
                asyncio.get_event_loop().create_task(start())
                def poll():
                    try:
                        while True:
                            item = progress_q.get_nowait()
                            if item[0] == "progress":
                                progress_bar.value = item[1]
                                progress_bar.set_visibility(True)
                                progress_label.set_text(item[2])
                            elif item[0] == "done":
                                result = item[1]
                                page_state["session"] = result["session_state"]
                                if result.get("propagation_preview_path"):
                                    refs["propagation_preview"].set_source(_session_path_to_url(result["propagation_preview_path"]))
                                progress_bar.set_visibility(False)
                                release_gpu(user_id)
                                _apply_notify(result)
                                return False
                    except queue.Empty:
                        pass
                    return True
                _start_until_false_timer(0.2, poll)
            ui.button("运行传播", on_click=run_propagation_task).props("color=primary")

        # 抠像参数
        with ui.expansion("④ 抠像", icon="content_cut").classes("flex-1").props("default-opened"):
            ui.label("使用遮罩进行精细抠像，输出透明背景视频。").classes("text-xs text-gray-500 mb-2")
            with ui.column().classes("gap-0"):
                ui.label("抠图引擎").classes("text-xs text-gray-500")
                matting_engine_selector = ui.radio(["MatAnyone", "VideoMaMa"], value="MatAnyone")
                refs["matting_engine_selector"] = matting_engine_selector
            ui.label("MatAnyone: 速度快，适合常规场景 | VideoMaMa: 效果更细腻，适合毛发等复杂边缘").classes("text-xs text-gray-400 my-1")
            erode_row = ui.row().classes("w-full")
            with erode_row:
                with ui.column().classes("flex-1 gap-0"):
                    ui.label("腐蚀 (Erode)").classes("text-xs text-gray-500")
                    erode_slider = ui.slider(min=0, max=30, value=DEFAULT_ERODE, step=1).props("label-always")
                    refs["erode_slider"] = erode_slider
                    ui.label("向内收缩遮罩边缘，去除边缘杂色。值越大收缩越多。").classes("text-xs text-gray-400")
                with ui.column().classes("flex-1 gap-0"):
                    ui.label("膨胀 (Dilate)").classes("text-xs text-gray-500")
                    dilate_slider = ui.slider(min=0, max=30, value=DEFAULT_DILATE, step=1).props("label-always")
                    refs["dilate_slider"] = dilate_slider
                    ui.label("向外扩展遮罩边缘，保留更多边缘细节。值越大扩展越多。").classes("text-xs text-gray-400")
            refs["erode_row"] = erode_row
            vm_params_row = ui.row().classes("w-full gap-4")
            refs["vm_params_row"] = vm_params_row
            with vm_params_row:
                with ui.column().classes("gap-0"):
                    ui.label("批次大小 (Batch)").classes("text-xs text-gray-500")
                    vm_batch_slider = ui.slider(min=4, max=128, value=VIDEOMAMA_BATCH_SIZE, step=4).props("label-always")
                    refs["vm_batch_slider"] = vm_batch_slider
                    ui.label("每批处理的帧数，越大显存占用越高。").classes("text-xs text-gray-400")
                with ui.column().classes("gap-0"):
                    ui.label("重叠帧数 (Overlap)").classes("text-xs text-gray-500")
                    vm_overlap_slider = ui.slider(min=0, max=8, value=VIDEOMAMA_OVERLAP, step=1).props("label-always")
                    refs["vm_overlap_slider"] = vm_overlap_slider
                    ui.label("批次间重叠帧数，增加可减少批次接缝。").classes("text-xs text-gray-400")
                with ui.column().classes("gap-0"):
                    ui.label("随机种子 (Seed)").classes("text-xs text-gray-500")
                    vm_seed_input = ui.number(value=VIDEOMAMA_SEED, format="%.0f")
                    refs["vm_seed_input"] = vm_seed_input
                    ui.label("固定种子可复现结果。").classes("text-xs text-gray-400")
            vm_params_row.set_visibility(False)
            def on_engine_change():
                eng = refs["matting_engine_selector"].value
                refs["erode_row"].set_visibility(eng == "MatAnyone")
                refs["vm_params_row"].set_visibility(eng == "VideoMaMa")
            matting_engine_selector.on("update:model-value", on_engine_change)
            matting_progress = ui.linear_progress(value=0).props("visible=false").classes("w-full")
            with ui.row().classes("gap-2"):
                matting_run_btn = ui.button(
                    "开始抠像",
                    on_click=lambda: _on_matting_run_stop(refs, matting_progress, page_state, user_id, user_name),
                ).props("color=primary")
                refs["matting_run_btn"] = matting_run_btn
                ui.button("添加到队列", on_click=lambda: _on_add_queue(refs, page_state, user_id))
                ui.label("'添加到队列' 可批量处理多个任务").classes("text-xs text-gray-400 self-center")

    # 抠像结果
    with ui.row().classes("w-full gap-4"):
        with ui.column().classes("flex-1"):
            ui.label("Alpha 通道").classes("text-sm font-medium")
            ui.label("白色=完全不透明，黑色=完全透明，灰色=半透明").classes("text-xs text-gray-400")
            alpha_video = ui.video("").classes("max-h-48 w-full")
            refs["alpha_video"] = alpha_video
        with ui.column().classes("flex-1"):
            ui.label("前景视频").classes("text-sm font-medium")
            ui.label("提取的前景，可直接用于合成").classes("text-xs text-gray-400")
            fgr_video = ui.video("").classes("max-h-48 w-full")
            refs["fgr_video"] = fgr_video

    # 处理日志
    with ui.expansion("处理日志", icon="terminal").classes("w-full"):
        ui.label("实时显示当前任务的处理进度和状态信息。").classes("text-xs text-gray-500 mb-2")
        log_display = ui.textarea(value="暂无日志").props("readonly outlined autogrow").classes("w-full font-mono text-xs")
        refs["log_display"] = log_display

        _last_log_content = ""

        def poll_log():
            nonlocal _last_log_content
            content = ""
            if PROCESSING_LOG_FILE.exists():
                content = PROCESSING_LOG_FILE.read_text().strip()
            display = content if content else "暂无日志"
            if display != _last_log_content:
                _last_log_content = display
                log_display.value = display
        _start_page_timer(2.0, poll_log)

    matting_cancel_event: threading.Event | None = None

    def _finish_matting(r: dict, progress_el, uid: str):
        """Reset UI after matting completes or is cancelled."""
        nonlocal matting_cancel_event
        matting_cancel_event = None
        progress_el.set_visibility(False)
        btn = r["matting_run_btn"]
        btn.set_text("开始抠像")
        btn.props("color=primary")
        btn._props["icon"] = ""
        btn.update()
        release_gpu(uid)

    def _on_matting_run_stop(r: dict, progress_el, ps: dict, uid: str, uname: str) -> None:
        nonlocal matting_cancel_event
        # If matting is running, stop it
        if matting_cancel_event is not None:
            matting_cancel_event.set()
            ui.notify("正在停止抠像…", type="info")
            return

        acquired, msg = try_acquire_gpu(uid, uname, "抠像")
        if not acquired:
            ui.notify(msg, type="warning")
            return

        # Switch button to "停止"
        btn = r["matting_run_btn"]
        btn.set_text("停止")
        btn.props("color=negative")
        btn._props["icon"] = "stop"
        btn.update()

        cancel_ev = threading.Event()
        matting_cancel_event = cancel_ev

        progress_q = queue.Queue()

        def cb(frac: float, desc: str):
            progress_q.put(("progress", frac, desc))

        def blocking():
            try:
                result = start_matting(
                    r["matting_engine_selector"].value,
                    int(r["erode_slider"].value), int(r["dilate_slider"].value),
                    int(r["vm_batch_slider"].value), int(r["vm_overlap_slider"].value), int(r["vm_seed_input"].value),
                    r["model_selector"].value, ps["session"],
                    progress_callback=cb, cancel_event=cancel_ev,
                )
                progress_q.put(("done", result))
            except MattingCancelledError:
                progress_q.put(("cancelled",))

        async def start():
            await run.io_bound(blocking)

        asyncio.get_event_loop().create_task(start())

        def poll():
            try:
                while True:
                    item = progress_q.get_nowait()
                    if item[0] == "progress":
                        progress_el.value = item[1]
                        progress_el.set_visibility(True)
                    elif item[0] == "done":
                        result = item[1]
                        ps["session"] = result["session_state"]
                        if result.get("alpha_path"):
                            r["alpha_video"].set_source(_session_path_to_url(result["alpha_path"]))
                        if result.get("fgr_path"):
                            r["fgr_video"].set_source(_session_path_to_url(result["fgr_path"]))
                        _finish_matting(r, progress_el, uid)
                        _apply_notify(result)
                        return False
                    elif item[0] == "cancelled":
                        _finish_matting(r, progress_el, uid)
                        ui.notify("抠像已停止", type="info")
                        return False
            except queue.Empty:
                pass
            return True

        _start_until_false_timer(0.2, poll)

    def _on_add_queue(r: dict, ps: dict, uid: str) -> None:
        q = load_queue()
        out = add_to_queue(
            r["matting_engine_selector"].value,
            int(r["erode_slider"].value), int(r["dilate_slider"].value),
            int(r["vm_batch_slider"].value), int(r["vm_overlap_slider"].value), int(r["vm_seed_input"].value),
            ps["session"], q,
        )
        ps["session"] = out["session_state"]
        r["queue_status"].set_text(out["queue_status_text"])
        r["queue_table"].rows = _build_queue_rows_with_sid()
        _update_queue_counts(r)
        if out.get("frame_image") is not None:
            r["frame_image"].set_source(_write_frame_preview(out["frame_image"], uid) if out["frame_image"] else "")
        if out.get("frame_label"):
            r["frame_label"].set_text(out["frame_label"])
        if out.get("keyframe_info"):
            r["keyframe_info"].set_text(out["keyframe_info"])
        if out.get("keyframe_gallery") is not None:
            _refresh_gallery(r["keyframe_gallery_container"], out["keyframe_gallery"], uid, jump_to_frame)
        r["frame_slider"].set_visibility(out.get("slider_visible", False))
        r["alpha_video"].set_source(_session_path_to_url(out.get("alpha_path")))
        r["fgr_video"].set_source(_session_path_to_url(out.get("fgr_path")))
        _apply_notify(out)

    ui.separator()

    # ======================================================================
    # 区域 4：任务队列
    # ======================================================================
    with ui.expansion("⑤ 任务队列", icon="playlist_play").classes("w-full").props("default-opened"):
        ui.label("抠像和追踪任务共享同一队列，串行执行避免 GPU 过载。").classes("text-xs text-gray-500 mb-2")

        # 队列状态概览
        with ui.card().classes("w-full p-3 mb-2 bg-gray-50"):
            with ui.row().classes("w-full items-center justify-between"):
                queue_status = ui.label("队列为空。").classes("font-medium")
                refs["queue_status"] = queue_status
                with ui.row().classes("gap-4"):
                    with ui.column().classes("items-center gap-0"):
                        queue_pending_count = ui.label("0").classes("text-xl font-bold text-blue-600")
                        refs["queue_pending_count"] = queue_pending_count
                        ui.label("待处理").classes("text-xs text-gray-500")
                    with ui.column().classes("items-center gap-0"):
                        queue_done_count = ui.label("0").classes("text-xl font-bold text-green-600")
                        refs["queue_done_count"] = queue_done_count
                        ui.label("已完成").classes("text-xs text-gray-500")
                    with ui.column().classes("items-center gap-0"):
                        queue_failed_count = ui.label("0").classes("text-xl font-bold text-red-600")
                        refs["queue_failed_count"] = queue_failed_count
                        ui.label("失败").classes("text-xs text-gray-500")

        columns = [
            {"name": "idx", "label": "序号", "field": "idx", "align": "center"},
            {"name": "type", "label": "类型", "field": "type", "align": "center"},
            {"name": "video", "label": "视频名", "field": "video"},
            {"name": "frames", "label": "帧数", "field": "frames", "align": "center"},
            {"name": "kf", "label": "关键帧", "field": "kf", "align": "center"},
            {"name": "mode", "label": "模式", "field": "mode", "align": "center"},
            {"name": "status", "label": "状态", "field": "status", "align": "center"},
            {"name": "action", "label": "操作", "align": "center"},
        ]
        queue_table = ui.table(columns=columns, rows=_build_queue_rows_with_sid(), row_key="session_id").classes("w-full")
        refs["queue_table"] = queue_table
        with queue_table.add_slot("body-cell-action"):
            with queue_table.cell("action"):
                with ui.row().classes("gap-1"):
                    ui.button("恢复", icon="edit").props("flat dense size=sm").on(
                        "click",
                        js_handler="() => emit(props.row.session_id)",
                        handler=lambda e: _on_queue_restore(refs, page_state, user_id, (e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args)),
                    )
                    ui.button("移除", icon="delete").props("flat dense size=sm color=negative").on(
                        "click",
                        js_handler="() => emit(props.row.session_id)",
                        handler=lambda e: _on_queue_remove(refs, (e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args)),
                    )

        # 执行进度区域
        with ui.card().classes("w-full p-3 mt-2"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("hourglass_empty", color="blue").classes("text-lg")
                queue_progress_label = ui.label("等待执行...").classes("text-sm flex-1")
                refs["queue_progress_label"] = queue_progress_label
            queue_progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full mt-1")
            queue_progress_bar.set_visibility(False)
            refs["queue_progress_bar"] = queue_progress_bar
            queue_current_task = ui.label("").classes("text-xs text-gray-500 mt-1")
            refs["queue_current_task"] = queue_current_task

        with ui.row().classes("gap-2 flex-wrap items-center mt-2"):
            ui.button("清空队列", on_click=lambda: _queue_act(refs, clear_queue)).props("color=negative outline")
            ui.button("重置状态", on_click=lambda: _queue_act(refs, reset_status)).props("outline")
            queue_execute_btn = ui.button(
                "开始执行队列",
                on_click=lambda: _on_queue_execute_or_stop(refs, user_id, user_name),
            ).props("color=primary")
            refs["queue_execute_btn"] = queue_execute_btn
            refs["queue_executing"] = False
            ui.button("飞书通知", on_click=lambda: _apply_notify(send_feishu(load_queue()))).props("outline")
            ui.button("打包下载", on_click=lambda: _on_pack(refs))
        ui.label("恢复: 重新编辑该任务 | 移除: 从队列删除 | 新任务加入后自动继续执行").classes("text-xs text-gray-400 mt-1")

    def _on_queue_restore(r: dict, ps: dict, uid: str, session_id: str) -> None:
        q = load_queue()
        match_idx = next((i for i, item in enumerate(q) if item["sid"] == session_id), None)
        if match_idx is None:
            ui.notify("该任务已不在队列中", type="warning")
            _update_queue_ui(r)
            return
        out = restore_from_queue(match_idx + 1, ps["session"], q)
        if out.get("restore_type") == "tracking":
            ui.notify("追踪任务请在追踪页面编辑", type="info")
        else:
            ps["session"] = _apply_restore_out(out, r, uid, ps["session"], jump_to_frame)
        _apply_notify(out)

    def _on_queue_remove(r: dict, session_id: str) -> None:
        q = load_queue()
        match_idx = next((i for i, item in enumerate(q) if item["sid"] == session_id), None)
        if match_idx is None:
            ui.notify("该任务已不在队列中", type="warning")
            _update_queue_ui(r)
            return
        out = remove_from_queue(match_idx + 1, q)
        r["queue_status"].set_text(out["queue_status_text"])
        r["queue_table"].rows = _build_queue_rows_with_sid()
        _update_queue_counts(r)
        _apply_notify(out)

    def _queue_act(r: dict, fn) -> None:
        q = load_queue()
        out = fn(q)
        r["queue_status"].set_text(out["queue_status_text"])
        r["queue_table"].rows = _build_queue_rows_with_sid()
        _update_queue_counts(r)
        _apply_notify(out)

    def _update_queue_counts(r: dict) -> None:
        rows = r["queue_table"].rows
        pending = sum(1 for row in rows if row.get("status") == "pending")
        done = sum(1 for row in rows if row.get("status") == "done")
        failed = sum(1 for row in rows if row.get("status") == "failed")
        r["queue_pending_count"].set_text(str(pending))
        r["queue_done_count"].set_text(str(done))
        r["queue_failed_count"].set_text(str(failed))

    def _on_stop_queue(r: dict) -> None:
        out = stop_queue()
        r["queue_progress_bar"].set_visibility(False)
        r["queue_progress_label"].set_text("已停止")
        r["queue_current_task"].set_text("")
        _apply_notify(out)

    def _on_queue_execute_or_stop(r: dict, uid: str, uname: str) -> None:
        """Toggle: start queue if idle, stop if running."""
        if r.get("queue_executing"):
            _on_stop_queue(r)
        else:
            _run_execute_queue(r, uid, uname)

    def _run_execute_queue(r: dict, uid: str, uname: str) -> None:
        acquired, msg = try_acquire_gpu(uid, uname, "执行队列")
        if not acquired:
            ui.notify(msg, type="warning")
            return

        r["queue_executing"] = True
        btn = r.get("queue_execute_btn")
        if btn:
            btn.set_text("停止")
            btn.props("color=negative")

        progress_q = queue.Queue()

        def cb(frac: float, desc: str):
            progress_q.put(("progress", frac, desc))

        def blocking():
            try:
                result = run_execute_queue(progress_callback=cb)
            except Exception as exc:
                q = load_queue()
                result = {
                    "queue_state": q,
                    "queue_status_text": queue_status_text(q),
                    "queue_table_rows": queue_table_rows(q),
                    "queue_progress_text": f"执行出错: {exc}",
                    "warning": str(exc),
                }
            progress_q.put(("done", result))

        async def start():
            await run.io_bound(blocking)

        r["queue_progress_bar"].set_visibility(True)
        r["queue_progress_bar"].value = 0
        r["queue_progress_label"].set_text("正在执行队列...")
        r["queue_current_task"].set_text("")

        asyncio.get_event_loop().create_task(start())

        def poll():
            try:
                while True:
                    item = progress_q.get_nowait()
                    if item[0] == "progress":
                        frac, desc = item[1], item[2]
                        r["queue_progress_bar"].value = frac
                        r["queue_progress_label"].set_text(f"进度: {frac * 100:.1f}%")
                        r["queue_current_task"].set_text(desc)
                        r["queue_table"].rows = _build_queue_rows_with_sid()
                        _update_queue_counts(r)
                    elif item[0] == "done":
                        result = item[1]
                        r["queue_executing"] = False
                        if btn:
                            btn.set_text("开始执行队列")
                            btn.props("color=primary")
                        r["queue_status"].set_text(result["queue_status_text"])
                        r["queue_table"].rows = _build_queue_rows_with_sid()
                        _update_queue_counts(r)
                        r["queue_progress_bar"].value = 1
                        r["queue_progress_bar"].set_visibility(False)
                        r["queue_progress_label"].set_text("队列执行完成")
                        r["queue_current_task"].set_text("")
                        release_gpu(uid)
                        _apply_notify(result)
                        return False
            except queue.Empty:
                pass
            return True

        _start_until_false_timer(0.2, poll)

    def _on_pack(r: dict) -> None:
        q = load_queue()
        out = pack_download(q)
        if out.get("download_path"):
            ui.download("/workspace/results.zip", "results.zip")
        _apply_notify(out)

    # Auto-refresh queue UI for multi-user visibility
    _last_queue_rows: list[dict] = []

    def auto_refresh_queue():
        nonlocal _last_queue_rows
        rows = _build_queue_rows_with_sid()
        if rows != _last_queue_rows:
            _last_queue_rows = rows
            _update_queue_ui(refs)
    _start_page_timer(3.0, auto_refresh_queue)
    
    _update_queue_ui(refs)
    
    # Restore previous session if available
    if page_state["session"].get("session_id"):
        # Trigger UI update for restored session
        out = restore_session(page_state["session"]["session_id"], empty_state())
        if out.get("session_state"):
            page_state["session"] = _apply_restore_out(out, refs, user_id, page_state["session"], jump_to_frame)


def _write_tracking_preview(frame: np.ndarray, user_id: str) -> str:
    """Write tracking frame preview to user-specific directory."""
    if frame is None or frame.size == 0:
        return ""
    user_preview_dir = tracking_preview_dir / user_id
    user_preview_dir.mkdir(exist_ok=True)
    path = user_preview_dir / "current.png"
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return f"/tracking_preview/{user_id}/current.png?t={id(frame)}"


@ui.page("/tracking")
def tracking_page(client):
    """Point tracking page using CoTracker."""
    user_id = _get_user_id_from_request(client.request)
    user_name = _get_user_name_from_request(client.request)
    
    page_state = {"tracking": empty_tracking_state()}
    
    device = get_device()
    device_label = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
    hostname = platform.node() or "unknown"

    refs = {}
    page_timers = set()

    def _start_page_timer(interval: float, callback, *, once: bool = False, immediate: bool = True):
        timer = app.timer(interval, callback, once=once, immediate=immediate)
        page_timers.add(timer)
        return timer

    def _stop_page_timer(timer) -> None:
        if timer is None:
            return
        timer.cancel(with_current_invocation=True)
        page_timers.discard(timer)

    def _cleanup_page_timers(*_) -> None:
        for timer in list(page_timers):
            timer.cancel(with_current_invocation=True)
        page_timers.clear()

    def _start_until_false_timer(interval: float, poll_fn) -> None:
        timer = None
        def wrapped_poll():
            nonlocal timer
            try:
                keep_running = poll_fn()
            except RuntimeError as ex:
                log.warning("Stop poll timer due to runtime error: %s", ex)
                keep_running = False
            except Exception:
                log.exception("Stop poll timer due to unexpected polling error")
                keep_running = False
            if not keep_running and timer is not None:
                _stop_page_timer(timer)
                timer = None
        timer = _start_page_timer(interval, wrapped_poll)

    client.on_disconnect(_cleanup_page_timers)
    client.on_delete(_cleanup_page_timers)

    def _apply_tracking_notify(out: dict) -> None:
        if out.get("notify"):
            ntype, msg = out["notify"]
            ui.notify(msg, type=ntype)

    # ---- Header ----
    with ui.row().classes("w-full items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.label("SplazMatte").classes("text-2xl font-bold")
            ui.link("抠像", "/").classes("text-gray-600 hover:text-blue-600")
            ui.link("轨迹追踪", "/tracking").classes("text-blue-600 underline font-medium")
        with ui.row().classes("items-center gap-4"):
            gpu_status_label = ui.label("").classes("text-xs")
            refs["gpu_status_label"] = gpu_status_label
            ui.label(f"用户: {user_name}").classes("text-sm text-gray-600")
            ui.label(f"运行设备: {device_label.get(device.type, device.type)} | 主机: {hostname}").classes("text-sm text-gray-600")

    # GPU status polling
    _last_gpu_text = ""

    def update_gpu_status():
        nonlocal _last_gpu_text
        status = get_gpu_status()
        if status["locked"]:
            if status["holder_id"] == user_id:
                text = f"GPU: 您正在使用 ({status['operation']})"
                remove, add = "text-red-500 text-green-500", "text-blue-500"
            else:
                text = f"GPU: {status['holder_name']} 使用中"
                remove, add = "text-blue-500 text-green-500", "text-red-500"
        else:
            text = "GPU: 空闲"
            remove, add = "text-red-500 text-blue-500", "text-green-500"
        if text != _last_gpu_text:
            _last_gpu_text = text
            gpu_status_label.set_text(text)
            gpu_status_label.classes(remove=remove, add=add)
    _start_page_timer(1.0, update_gpu_status)

    ui.separator()

    ui.label("上传视频，点击选择追踪点，运行追踪后查看结果。").classes("text-xs text-gray-500 mb-2")

    # ======================================================================
    # 区域 1：上传视频 / 恢复 Session
    # ======================================================================
    with ui.expansion("① 上传视频 / 恢复 Session", icon="folder_open").classes("w-full").props("default-opened"):
        ui.label("上传视频文件开始新任务，或从历史 Session 恢复之前的追踪进度。").classes("text-xs text-gray-500 mb-2")
        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-1"):
                tracking_video_display = ui.video("").classes("max-w-full max-h-48")
                refs["video_display"] = tracking_video_display

                async def on_tracking_uploaded(e: events.UploadEventArguments):
                    upload_dir = WORKSPACE_DIR / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    dest = upload_dir / (e.file.name or "video")
                    save_fn = e.file.save(str(dest))
                    if asyncio.iscoroutine(save_fn):
                        await save_fn
                    loading_note = ui.notification("正在预处理视频…", type="ongoing", timeout=None, spinner=True)
                    try:
                        out = await run.io_bound(ct_preprocess_video, str(dest), page_state["tracking"])
                    except Exception as ex:
                        loading_note.dismiss()
                        log.exception("Tracking video upload failed")
                        ui.notify(f"视频处理失败: {ex}", type="negative")
                        return
                    loading_note.dismiss()
                    page_state["tracking"] = out["session_state"]
                    if out.get("preview_frame") is not None:
                        refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                    if out.get("slider_max") is not None:
                        refs["frame_slider"].props["max"] = out["slider_max"]
                        refs["frame_slider"].value = out.get("slider_value", 0)
                        refs["frame_slider"].set_visibility(True)
                    refs["frame_label"].set_text(f"帧 1 / {out.get('slider_max', 0) + 1}")
                    refs["video_display"].set_source(_workspace_path_to_url(dest))
                    # Refresh session list and auto-select the new session
                    out_refresh = ct_refresh_sessions()
                    if out_refresh.get("session_choices") and refs.get("tracking_session_dropdown"):
                        dd = refs["tracking_session_dropdown"]
                        dd.options = {v: l for l, v in out_refresh["session_choices"]}
                        new_sid = page_state["tracking"].get("session_id")
                        if new_sid:
                            dd.value = new_sid
                        dd.update()
                    _apply_tracking_notify(out)

                ui.upload(on_upload=on_tracking_uploaded, max_file_size=500_000_000).props("accept='video/*'").classes("w-full")
                ui.label("支持 mp4/mov/avi 等格式。").classes("text-xs text-gray-400")

            with ui.column().classes("w-64"):
                tracking_sessions = list_tracking_sessions()
                tracking_session_dropdown = ui.select(
                    options={v: l for l, v in tracking_sessions},
                    value=None,
                    label="恢复 Session",
                ).classes("w-full")
                refs["tracking_session_dropdown"] = tracking_session_dropdown
                with ui.row():
                    def on_tracking_refresh():
                        out = ct_refresh_sessions()
                        refs["tracking_session_dropdown"].options = {v: l for l, v in out["session_choices"]}
                        refs["tracking_session_dropdown"].update()
                    ui.button("刷新", on_click=on_tracking_refresh)
                    def on_tracking_restore():
                        sid = refs["tracking_session_dropdown"].value
                        out = ct_restore_session(sid, page_state["tracking"])
                        if out.get("warning"):
                            ui.notify(out["warning"], type="warning")
                            return
                        page_state["tracking"] = out["session_state"]
                        if out.get("preview_frame") is not None:
                            refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                        if out.get("slider_max") is not None:
                            refs["frame_slider"].props["max"] = out["slider_max"]
                            refs["frame_slider"].value = out.get("slider_value", 0)
                            refs["frame_slider"].set_visibility(True)
                        refs["frame_label"].set_text(out.get("frame_label", "帧 0 / 0"))
                        if out.get("keyframe_info"):
                            refs["tracking_kf_info"].set_text(out["keyframe_info"])
                        if out.get("keyframe_gallery") is not None:
                            _refresh_gallery(refs["tracking_kf_gallery"], out["keyframe_gallery"], user_id, _jump_to_tracking_frame)
                        refs["point_count_label"].set_text(f"已选择 {out.get('point_count', 0)} 个追踪点")
                        if out.get("video_path"):
                            refs["video_display"].set_source(_workspace_path_to_url(out["video_path"]))
                        if out.get("result_video_path"):
                            refs["result_video"].set_source(_workspace_path_to_url(out["result_video_path"]))
                        if refs.get("backward_tracking") is not None:
                            refs["backward_tracking"].value = out.get("backward_tracking", False)
                        if refs.get("grid_size") is not None:
                            refs["grid_size"].value = out.get("grid_size", 15)
                        ui.notify("Session 恢复成功", type="positive")
                    ui.button("恢复", on_click=on_tracking_restore).props("color=primary")
                ui.label("选择历史 Session 可恢复之前的追踪点、追踪结果。").classes("text-xs text-gray-400")

    ui.separator()

    # ======================================================================
    # 区域 2：选择追踪点
    # ======================================================================
    with ui.expansion("② 选择追踪点", icon="touch_app").classes("w-full").props("default-opened"):
        ui.label("点击图片添加追踪点，使用网格模式自动生成，或用 SAM 选择目标区域。").classes("text-xs text-gray-500 mb-2")

        # 模式选择
        track_mode_toggle = ui.toggle(["手动选点", "网格模式", "SAM目标选择"], value="手动选点").classes("mb-2")
        refs["track_mode"] = track_mode_toggle

        # 手动/网格 工具栏
        with ui.row().classes("gap-2 flex-wrap items-center mb-2") as manual_toolbar:
            async def on_tracking_undo():
                out = await run.io_bound(ct_undo_point, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                refs["point_count_label"].set_text(f"已选择 {out.get('query_count', 0)} 个追踪点")

            async def on_tracking_clear_frame():
                out = await run.io_bound(ct_clear_frame_points, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                refs["point_count_label"].set_text(f"已选择 {out.get('query_count', 0)} 个追踪点")

            async def on_tracking_clear_all():
                out = await run.io_bound(ct_clear_all_points, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                refs["point_count_label"].set_text("已选择 0 个追踪点")

            ui.button("撤销", on_click=on_tracking_undo, icon="undo")
            ui.button("清除当前帧", on_click=on_tracking_clear_frame, icon="clear")
            ui.button("清除全部", on_click=on_tracking_clear_all, icon="delete_sweep")
            ui.separator().props("vertical")
            grid_size_input = ui.number("网格大小", value=15, min=3, max=30, step=1).classes("w-24")
            refs["grid_size"] = grid_size_input
        refs["manual_toolbar"] = manual_toolbar

        # 关键帧保存/删除按钮（始终可见，不受模式切换影响）
        with ui.row().classes("gap-2 items-center mb-2"):
            async def on_save_tracking_kf():
                out = await run.io_bound(ct_save_kf, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                _apply_tracking_notify(out)
                if out.get("keyframe_info"):
                    refs["tracking_kf_info"].set_text(out["keyframe_info"])
                if out.get("keyframe_gallery") is not None:
                    _refresh_gallery(refs["tracking_kf_gallery"], out["keyframe_gallery"], user_id, _jump_to_tracking_frame)
                await run.io_bound(ct_save_session, page_state["tracking"])

            async def on_del_tracking_kf():
                out = await run.io_bound(ct_del_kf, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                _apply_tracking_notify(out)
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                if out.get("keyframe_info"):
                    refs["tracking_kf_info"].set_text(out["keyframe_info"])
                if out.get("keyframe_gallery") is not None:
                    _refresh_gallery(refs["tracking_kf_gallery"], out["keyframe_gallery"], user_id, _jump_to_tracking_frame)
                refs["point_count_label"].set_text(f"已选择 {page_state['tracking'].get('query_count', 0)} 个追踪点")
                await run.io_bound(ct_save_session, page_state["tracking"])

            ui.button("保存关键帧", on_click=on_save_tracking_kf, icon="save").props("color=primary")
            ui.button("删除关键帧", on_click=on_del_tracking_kf, icon="delete")

        # SAM 工具栏（仅 SAM 模式可见）
        with ui.row().classes("gap-2 flex-wrap items-center mb-2") as sam_toolbar:
            with ui.column().classes("gap-0"):
                ui.label("分割模型").classes("text-xs text-gray-500")
                sam_model_select = ui.radio(["SAM2", "SAM3"], value="SAM2")
                refs["sam_model"] = sam_model_select
            with ui.column().classes("gap-0"):
                ui.label("点击模式").classes("text-xs text-gray-500")
                sam_point_mode = ui.radio(["Positive", "Negative"], value="Positive")
                refs["sam_point_mode"] = sam_point_mode
            ui.label("Positive=选中目标，Negative=排除区域").classes("text-xs text-gray-400")

            async def on_sam_undo():
                acquired, msg = try_acquire_gpu(user_id, user_name, "SAM 撤销")
                if not acquired:
                    ui.notify(msg, type="warning")
                    return
                refs["loading_overlay"].set_visibility(True)
                try:
                    out = await run.io_bound(ct_sam_undo, page_state["tracking"], refs["sam_model"].value)
                    page_state["tracking"] = out["session_state"]
                    if out.get("preview_frame") is not None:
                        refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                finally:
                    release_gpu(user_id)
                    refs["loading_overlay"].set_visibility(False)

            async def on_sam_clear():
                out = await run.io_bound(ct_sam_clear, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))

            async def on_generate_from_mask():
                grid_size = int(refs["grid_size_sam"].value or 15)
                out = await run.io_bound(ct_generate_points_from_mask, page_state["tracking"], grid_size)
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                if out.get("query_count") is not None:
                    refs["point_count_label"].set_text(f"已选择 {out['query_count']} 个追踪点")
                if out.get("notify"):
                    ntype, nmsg = out["notify"]
                    ui.notify(nmsg, type=ntype)

            ui.button("撤销", on_click=on_sam_undo, icon="undo")
            ui.button("清除 Mask", on_click=on_sam_clear, icon="clear")
            ui.separator().props("vertical")
            grid_size_sam = ui.number("采样网格", value=15, min=3, max=30, step=1).classes("w-24")
            refs["grid_size_sam"] = grid_size_sam
            ui.button("从 Mask 生成追踪点", on_click=on_generate_from_mask, icon="scatter_plot").props("color=primary")
        refs["sam_toolbar"] = sam_toolbar
        sam_toolbar.set_visibility(False)

        # 模式切换回调
        def on_track_mode_change():
            mode = refs["track_mode"].value
            refs["manual_toolbar"].set_visibility(mode != "SAM目标选择")
            refs["sam_toolbar"].set_visibility(mode == "SAM目标选择")
        track_mode_toggle.on("update:model-value", on_track_mode_change)

        # 帧滑块
        tracking_frame_slider = ui.slider(min=0, max=1, value=0, step=1).props("label-always").classes("w-full mb-2")
        tracking_frame_slider.set_visibility(False)
        refs["frame_slider"] = tracking_frame_slider

        async def on_tracking_slider_change():
            v = refs["frame_slider"].value
            if v is None or page_state["tracking"].get("frames_dir") is None:
                return
            out = await run.io_bound(ct_change_frame, int(v), page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            if out.get("preview_frame") is not None:
                refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
            if out.get("frame_label"):
                refs["frame_label"].set_text(out["frame_label"])

        tracking_frame_slider.on("update:model-value", on_tracking_slider_change)

        # 帧标签和点数
        with ui.row().classes("w-full justify-between items-center mb-1"):
            tracking_frame_label = ui.label("请先上传视频。").classes("text-sm text-gray-600")
            refs["frame_label"] = tracking_frame_label
            point_count_label = ui.label("已选择 0 个追踪点").classes("text-sm text-blue-600")
            refs["point_count_label"] = point_count_label

        # 跳转到指定帧（Gallery 点击回调）
        async def _jump_to_tracking_frame(frame_idx: int):
            refs["frame_slider"].value = frame_idx
            out = await run.io_bound(ct_change_frame, frame_idx, page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            if out.get("preview_frame") is not None:
                refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
            if out.get("frame_label"):
                refs["frame_label"].set_text(out["frame_label"])

        # 交互式图片 + 关键帧 Gallery
        click_state = {"busy": False}

        async def _on_tracking_mouse(e: events.MouseEventArguments):
            if e.type != "click":
                return
            if page_state["tracking"].get("frames_dir") is None:
                ui.notify("请先上传视频", type="warning")
                return
            if click_state["busy"]:
                return

            mode = refs["track_mode"].value

            if mode == "网格模式":
                ui.notify("网格模式下无需手动选点", type="info")
                return

            if mode == "SAM目标选择":
                acquired, msg = try_acquire_gpu(user_id, user_name, "SAM 选择")
                if not acquired:
                    ui.notify(msg, type="warning")
                    return
                click_state["busy"] = True
                refs["loading_overlay"].set_visibility(True)
                try:
                    is_positive = refs["sam_point_mode"].value == "Positive"
                    out = await run.io_bound(
                        ct_sam_click, e.image_x, e.image_y,
                        page_state["tracking"],
                        refs["sam_model"].value,
                        is_positive,
                    )
                    page_state["tracking"] = out["session_state"]
                    if out.get("preview_frame") is not None:
                        refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
                except Exception as ex:
                    log.exception("SAM prediction failed")
                    ui.notify(f"SAM 推理失败: {ex}", type="negative")
                finally:
                    release_gpu(user_id)
                    refs["loading_overlay"].set_visibility(False)
                    click_state["busy"] = False
                return

            # 手动选点模式
            out = await run.io_bound(ct_add_point, e.image_x, e.image_y, page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            if out.get("preview_frame") is not None:
                refs["frame_image"].set_source(_write_tracking_preview(out["preview_frame"], user_id))
            refs["point_count_label"].set_text(f"已选择 {out.get('query_count', 0)} 个追踪点")

        with ui.row().classes("w-full gap-4 items-start"):
            with ui.column().classes("flex-1"):
                with ui.element("div").classes("relative w-full"):
                    tracking_frame_image = ui.interactive_image(
                        "", on_mouse=_on_tracking_mouse, events=["click"],
                    ).classes("w-full")
                    refs["frame_image"] = tracking_frame_image

                    # 追踪加载遮罩
                    tracking_loading_overlay = ui.element("div").classes(
                        "absolute inset-0 flex items-center justify-center bg-black/50 z-10 backdrop-blur-sm"
                    )
                    with tracking_loading_overlay:
                        with ui.card().classes("p-4 items-center"):
                            ui.spinner("dots", size="xl", color="primary")
                            ui.label("追踪中...").classes("text-sm mt-2")
                            tracking_progress_label = ui.label("请稍候").classes("text-xs text-gray-500")
                            refs["progress_label"] = tracking_progress_label
                    tracking_loading_overlay.set_visibility(False)
                    refs["loading_overlay"] = tracking_loading_overlay

            # 关键帧 Gallery 侧边栏
            with ui.column().classes("w-48"):
                ui.label("已保存的关键帧").classes("text-xs text-gray-500 font-medium")
                tracking_kf_info = ui.label("尚未保存任何关键帧。").classes("text-xs text-gray-400")
                refs["tracking_kf_info"] = tracking_kf_info
                tracking_kf_gallery = ui.column().classes("gap-1 max-h-[400px] overflow-auto")
                refs["tracking_kf_gallery"] = tracking_kf_gallery

    ui.separator()

    # ======================================================================
    # 区域 3：运行追踪
    # ======================================================================
    with ui.expansion("③ 运行追踪", icon="play_arrow").classes("w-full").props("default-opened"):
        tracking_progress_bar = ui.linear_progress(value=0).classes("w-full mb-2")
        tracking_progress_bar.set_visibility(False)
        refs["progress_bar"] = tracking_progress_bar

        backward_tracking_checkbox = ui.checkbox("双向追踪（前向 + 后向）", value=False).classes("mb-1")
        refs["backward_tracking"] = backward_tracking_checkbox
        ui.label("前向模式（默认）：从选点帧向后追踪；双向模式：向前+向后都追踪，需 Offline 模型，显存约 12–16 GB").classes("text-xs text-gray-500 mb-2")

        tracking_cancel_event: threading.Event | None = None

        def _finish_tracking():
            """Reset UI after tracking completes or is cancelled."""
            nonlocal tracking_cancel_event
            tracking_cancel_event = None
            refs["loading_overlay"].set_visibility(False)
            refs["progress_bar"].set_visibility(False)
            btn = refs["run_stop_btn"]
            btn.set_text("运行追踪")
            btn.props("color=primary")
            btn._props["icon"] = "play_arrow"
            btn.update()
            release_gpu(user_id)

        def on_run_stop_click():
            nonlocal tracking_cancel_event
            # If tracking is running, stop it
            if tracking_cancel_event is not None:
                tracking_cancel_event.set()
                ui.notify("正在停止追踪…", type="info")
                return

            # --- Start tracking ---
            if page_state["tracking"].get("frames_dir") is None:
                ui.notify("请先上传视频", type="warning")
                return

            mode = refs["track_mode"].value
            use_grid = (mode == "网格模式")
            grid_size = int(refs["grid_size"].value or 15) if use_grid else 0

            page_state["tracking"]["use_grid"] = use_grid
            page_state["tracking"]["grid_size"] = grid_size
            page_state["tracking"]["backward_tracking"] = refs["backward_tracking"].value

            if not use_grid:
                keyframes = page_state["tracking"].get("keyframes", {})
                query_count = page_state["tracking"].get("query_count", 0)
                if not keyframes and query_count == 0:
                    ui.notify("请先保存至少一个关键帧，或启用网格/SAM模式", type="warning")
                    return

            acquired, msg = try_acquire_gpu(user_id, user_name, "轨迹追踪")
            if not acquired:
                ui.notify(msg, type="warning")
                return

            # Switch button to "停止"
            btn = refs["run_stop_btn"]
            btn.set_text("停止")
            btn.props("color=negative")
            btn._props["icon"] = "stop"
            btn.update()

            refs["loading_overlay"].set_visibility(True)
            refs["progress_bar"].set_visibility(True)
            refs["progress_bar"].value = 0

            cancel_ev = threading.Event()
            tracking_cancel_event = cancel_ev

            progress_q = queue.Queue()

            def cb(frac: float, desc: str):
                progress_q.put(("progress", frac, desc))

            def blocking():
                try:
                    result = ct_run_tracking(
                        page_state["tracking"],
                        use_grid=use_grid,
                        grid_size=grid_size,
                        backward_tracking=refs["backward_tracking"].value,
                        progress_callback=cb,
                        cancel_event=cancel_ev,
                    )
                    progress_q.put(("done", result))
                except TrackingCancelledError:
                    progress_q.put(("cancelled",))

            async def start():
                await run.io_bound(blocking)

            asyncio.get_event_loop().create_task(start())

            def poll():
                try:
                    while True:
                        item = progress_q.get_nowait()
                        if item[0] == "progress":
                            refs["progress_bar"].value = item[1]
                            refs["progress_label"].set_text(item[2])
                        elif item[0] == "done":
                            result = item[1]
                            page_state["tracking"] = result["session_state"]
                            if result.get("result_video_path"):
                                rel_path = Path(result["result_video_path"]).relative_to(WORKSPACE_DIR)
                                refs["result_video"].set_source(f"/workspace/{rel_path}")
                            _finish_tracking()
                            _apply_tracking_notify(result)
                            return False
                        elif item[0] == "cancelled":
                            _finish_tracking()
                            ui.notify("追踪已停止", type="info")
                            return False
                except queue.Empty:
                    pass
                return True

            _start_until_false_timer(0.2, poll)

        with ui.row().classes("gap-2 items-center"):
            run_stop_btn = ui.button("运行追踪", on_click=on_run_stop_click, icon="play_arrow").props("color=primary")
            refs["run_stop_btn"] = run_stop_btn

            async def on_add_tracking_to_queue():
                st = page_state["tracking"]
                if st.get("frames_dir") is None:
                    ui.notify("请先上传视频", type="warning")
                    return
                if not st.get("keyframes"):
                    ui.notify("请至少保存一个关键帧后再添加到队列", type="warning")
                    return
                # Store current params in state
                mode = refs["track_mode"].value
                st["use_grid"] = (mode == "网格模式")
                st["grid_size"] = int(refs["grid_size"].value or 15)
                st["backward_tracking"] = refs["backward_tracking"].value
                q = load_queue()
                out = await run.io_bound(add_tracking_to_queue, st, q)
                page_state["tracking"] = out["session_state"]
                if out.get("queue_state") is not None:
                    _update_tracking_queue_ui(refs)
                # Reset frame display
                refs["frame_image"].set_source("")
                refs["frame_slider"].set_visibility(False)
                refs["frame_label"].set_text("请先上传视频。")
                refs["point_count_label"].set_text("已选择 0 个追踪点")
                refs["tracking_kf_info"].set_text("尚未保存任何关键帧。")
                refs["tracking_kf_gallery"].clear()
                _apply_tracking_notify(out)
                if not out.get("notify"):
                    ui.notify("已添加到队列", type="positive")

            ui.button("添加到队列", on_click=on_add_tracking_to_queue, icon="playlist_add").props("outline")

        ui.label("运行追踪立即执行；添加到队列将保存后批量处理。").classes("text-xs text-gray-400")

        ui.separator().classes("my-2")

        ui.label("追踪结果").classes("text-sm font-medium")
        tracking_result_video = ui.video("").classes("max-w-full max-h-96")
        refs["result_video"] = tracking_result_video

        ui.separator().classes("my-2")

        ui.label("导出").classes("text-sm font-medium")

        async def on_ae_export():
            if page_state["tracking"].get("raw_tracks") is None:
                ui.notify("请先运行追踪", type="warning")
                return
            out = await run.io_bound(ct_ae_export, page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            _apply_tracking_notify(out)
            if out.get("export_path"):
                export_path = Path(out["export_path"])
                rel = export_path.relative_to(WORKSPACE_DIR)
                ui.download(f"/workspace/{rel}", filename="ae_tracking_keyframes.txt")

        ui.button("导出 After Effects 关键帧", on_click=on_ae_export, icon="download")
        ui.label("导出 Adobe After Effects 关键帧数据（.txt），可直接粘贴到 AE。").classes("text-xs text-gray-400")

    ui.separator()

    # ======================================================================
    # 区域 4：统一任务队列
    # ======================================================================
    with ui.expansion("④ 任务队列", icon="playlist_play").classes("w-full").props("default-opened"):
        ui.label("抠像和追踪任务共享同一队列，避免同时占用 GPU。").classes("text-xs text-gray-500 mb-2")

        with ui.card().classes("w-full p-3 mb-2 bg-gray-50"):
            with ui.row().classes("w-full items-center justify-between"):
                t_queue_status = ui.label("队列为空。").classes("font-medium")
                refs["queue_status"] = t_queue_status
                with ui.row().classes("gap-4"):
                    with ui.column().classes("items-center gap-0"):
                        t_pending = ui.label("0").classes("text-xl font-bold text-blue-600")
                        refs["queue_pending_count"] = t_pending
                        ui.label("待处理").classes("text-xs text-gray-500")
                    with ui.column().classes("items-center gap-0"):
                        t_done = ui.label("0").classes("text-xl font-bold text-green-600")
                        refs["queue_done_count"] = t_done
                        ui.label("已完成").classes("text-xs text-gray-500")
                    with ui.column().classes("items-center gap-0"):
                        t_failed = ui.label("0").classes("text-xl font-bold text-red-600")
                        refs["queue_failed_count"] = t_failed
                        ui.label("失败").classes("text-xs text-gray-500")

        t_columns = [
            {"name": "idx", "label": "序号", "field": "idx", "align": "center"},
            {"name": "type", "label": "类型", "field": "type", "align": "center"},
            {"name": "video", "label": "视频名", "field": "video"},
            {"name": "frames", "label": "帧数", "field": "frames", "align": "center"},
            {"name": "kf", "label": "关键帧", "field": "kf", "align": "center"},
            {"name": "mode", "label": "模式", "field": "mode", "align": "center"},
            {"name": "status", "label": "状态", "field": "status", "align": "center"},
            {"name": "action", "label": "操作", "align": "center"},
        ]
        t_queue_table = ui.table(
            columns=t_columns, rows=_build_queue_rows_with_sid(), row_key="session_id",
        ).classes("w-full")
        refs["queue_table"] = t_queue_table
        with t_queue_table.add_slot("body-cell-action"):
            with t_queue_table.cell("action"):
                with ui.row().classes("gap-1"):
                    ui.button("移除", icon="delete").props("flat dense size=sm color=negative").on(
                        "click",
                        js_handler="() => emit(props.row.session_id)",
                        handler=lambda e: _on_tracking_queue_remove(
                            refs, (e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args),
                        ),
                    )

        # 执行进度区域
        with ui.card().classes("w-full p-3 mt-2"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("hourglass_empty", color="blue").classes("text-lg")
                t_progress_label = ui.label("等待执行...").classes("text-sm flex-1")
                refs["queue_progress_label"] = t_progress_label
            t_progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full mt-1")
            t_progress_bar.set_visibility(False)
            refs["queue_progress_bar"] = t_progress_bar
            t_current_task = ui.label("").classes("text-xs text-gray-500 mt-1")
            refs["queue_current_task"] = t_current_task

        with ui.row().classes("gap-2 flex-wrap items-center mt-2"):
            ui.button("清空队列", on_click=lambda: _t_queue_act(refs, clear_queue)).props("color=negative outline")
            ui.button("重置状态", on_click=lambda: _t_queue_act(refs, reset_status)).props("outline")
            t_execute_btn = ui.button(
                "开始执行队列",
                on_click=lambda: _on_t_queue_execute_or_stop(refs, user_id, user_name),
            ).props("color=primary")
            refs["queue_execute_btn"] = t_execute_btn
            refs["queue_executing"] = False
            ui.button("飞书通知", on_click=lambda: _apply_notify(send_feishu(load_queue()))).props("outline")
            ui.button("打包下载", on_click=lambda: _on_t_pack(refs))
        ui.label("新任务加入后自动继续执行。").classes("text-xs text-gray-400 mt-1")

    def _update_tracking_queue_ui(r: dict) -> None:
        if "queue_status" not in r or "queue_table" not in r:
            return
        q = load_queue()
        r["queue_status"].set_text(queue_status_text(q))
        r["queue_table"].rows = _build_queue_rows_with_sid()
        if "queue_pending_count" in r:
            rows = r["queue_table"].rows
            pending = sum(1 for row in rows if row.get("status") in ("pending", ""))
            done = sum(1 for row in rows if row.get("status") == "done")
            failed = sum(1 for row in rows if row.get("status") == "error")
            r["queue_pending_count"].set_text(str(pending))
            r["queue_done_count"].set_text(str(done))
            r["queue_failed_count"].set_text(str(failed))

    def _t_queue_act(r: dict, fn) -> None:
        q = load_queue()
        out = fn(q)
        r["queue_status"].set_text(out["queue_status_text"])
        r["queue_table"].rows = _build_queue_rows_with_sid()
        _update_tracking_queue_ui(r)
        _apply_notify(out)

    def _on_tracking_queue_remove(r: dict, session_id: str) -> None:
        q = load_queue()
        match_idx = next((i for i, item in enumerate(q) if item["sid"] == session_id), None)
        if match_idx is None:
            ui.notify("该任务已不在队列中", type="warning")
            _update_tracking_queue_ui(r)
            return
        out = remove_from_queue(match_idx + 1, q)
        r["queue_status"].set_text(out["queue_status_text"])
        r["queue_table"].rows = _build_queue_rows_with_sid()
        _update_tracking_queue_ui(r)
        _apply_notify(out)

    def _on_t_queue_execute_or_stop(r: dict, uid: str, uname: str) -> None:
        if r.get("queue_executing"):
            out = stop_queue()
            r["queue_progress_bar"].set_visibility(False)
            r["queue_progress_label"].set_text("已停止")
            r["queue_current_task"].set_text("")
            _apply_notify(out)
        else:
            _run_t_execute_queue(r, uid, uname)

    def _run_t_execute_queue(r: dict, uid: str, uname: str) -> None:
        acquired, msg = try_acquire_gpu(uid, uname, "执行队列")
        if not acquired:
            ui.notify(msg, type="warning")
            return

        r["queue_executing"] = True
        btn = r.get("queue_execute_btn")
        if btn:
            btn.set_text("停止")
            btn.props("color=negative")

        progress_q = queue.Queue()

        def cb(frac: float, desc: str):
            progress_q.put(("progress", frac, desc))

        def blocking():
            try:
                result = run_execute_queue(progress_callback=cb)
            except Exception as exc:
                q = load_queue()
                result = {
                    "queue_state": q,
                    "queue_status_text": queue_status_text(q),
                    "queue_table_rows": queue_table_rows(q),
                    "queue_progress_text": f"执行出错: {exc}",
                    "warning": str(exc),
                }
            progress_q.put(("done", result))

        async def start():
            await run.io_bound(blocking)

        r["queue_progress_bar"].set_visibility(True)
        r["queue_progress_bar"].value = 0
        r["queue_progress_label"].set_text("正在执行队列...")
        r["queue_current_task"].set_text("")

        asyncio.get_event_loop().create_task(start())

        def poll():
            try:
                while True:
                    item = progress_q.get_nowait()
                    if item[0] == "progress":
                        frac, desc = item[1], item[2]
                        r["queue_progress_bar"].value = frac
                        r["queue_progress_label"].set_text(f"进度: {frac * 100:.1f}%")
                        r["queue_current_task"].set_text(desc)
                        r["queue_table"].rows = _build_queue_rows_with_sid()
                        _update_tracking_queue_ui(r)
                    elif item[0] == "done":
                        result = item[1]
                        r["queue_executing"] = False
                        if btn:
                            btn.set_text("开始执行队列")
                            btn.props("color=primary")
                        r["queue_status"].set_text(result["queue_status_text"])
                        r["queue_table"].rows = _build_queue_rows_with_sid()
                        _update_tracking_queue_ui(r)
                        r["queue_progress_bar"].value = 1
                        r["queue_progress_bar"].set_visibility(False)
                        r["queue_progress_label"].set_text("队列执行完成")
                        r["queue_current_task"].set_text("")
                        release_gpu(uid)
                        _apply_notify(result)
                        return False
            except queue.Empty:
                pass
            return True

        _start_until_false_timer(0.2, poll)

    def _on_t_pack(r: dict) -> None:
        q = load_queue()
        out = pack_download(q)
        if out.get("download_path"):
            ui.download("/workspace/results.zip", "results.zip")
        _apply_notify(out)

    # Auto-refresh queue UI
    _last_tracking_queue_rows: list[dict] = []

    def auto_refresh_tracking_queue():
        nonlocal _last_tracking_queue_rows
        rows = _build_queue_rows_with_sid()
        if rows != _last_tracking_queue_rows:
            _last_tracking_queue_rows = rows
            _update_tracking_queue_ui(refs)
    _start_page_timer(3.0, auto_refresh_tracking_queue)

    _update_tracking_queue_ui(refs)


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
    ui.run(host="0.0.0.0", port=SERVER_PORT, title="SplazMatte", reload=False, show=False, storage_secret=STORAGE_SECRET)
