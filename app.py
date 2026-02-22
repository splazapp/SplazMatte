"""SplazMatte — NiceGUI web app for MatAnyone matting with SAM2/SAM3 multi-frame annotation."""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

import asyncio
import logging
import platform
import queue
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
    PROCESSING_LOG_FILE,
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
preview_dir = WORKSPACE_DIR / "preview"
preview_dir.mkdir(exist_ok=True)
app.add_static_files("/sessions", str(WORKSPACE_DIR / "sessions"))
app.add_static_files("/preview", str(preview_dir))
app.add_static_files("/workspace", str(WORKSPACE_DIR))


def _session_path_to_url(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        rel = p.relative_to(WORKSPACE_DIR / "sessions")
        return "/sessions/" + str(rel).replace("\\", "/")
    except ValueError:
        return path


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
    return [
        {
            "idx": r[0],
            "video": r[1],
            "frames": r[2],
            "kf": r[3],
            "prop": r[4],
            "engine": r[5],
            "status": r[6],
            "session_id": q[i - 1] if i <= len(q) else "",
        }
        for i, r in enumerate(raw, start=1)
    ]


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
        ui.label("SplazMatte").classes("text-2xl font-bold")
        with ui.row().classes("items-center gap-4"):
            # GPU status indicator
            gpu_status_label = ui.label("").classes("text-xs")
            refs["gpu_status_label"] = gpu_status_label
            ui.label(f"用户: {user_name}").classes("text-sm text-gray-600")
            ui.label(f"运行设备: {device_label.get(device.type, device.type)} | 主机: {hostname}").classes("text-sm text-gray-600")
    
    # GPU status polling
    def update_gpu_status():
        status = get_gpu_status()
        if status["locked"]:
            if status["holder_id"] == user_id:
                gpu_status_label.set_text(f"GPU: 您正在使用 ({status['operation']})")
                gpu_status_label.classes(remove="text-red-500 text-green-500", add="text-blue-500")
            else:
                gpu_status_label.set_text(f"GPU: {status['holder_name']} 使用中")
                gpu_status_label.classes(remove="text-blue-500 text-green-500", add="text-red-500")
        else:
            gpu_status_label.set_text("GPU: 空闲")
            gpu_status_label.classes(remove="text-red-500 text-blue-500", add="text-green-500")
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
                ui.button("开始抠像", on_click=lambda: _run_matting(refs, matting_progress, page_state, user_id, user_name)).props("color=primary")
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

        def poll_log():
            if PROCESSING_LOG_FILE.exists():
                content = PROCESSING_LOG_FILE.read_text()
                if content.strip():
                    log_display.value = content
                else:
                    log_display.value = "暂无日志"
            else:
                log_display.value = "暂无日志"
        _start_page_timer(0.5, poll_log)

    def _run_matting(r: dict, progress_el, ps: dict, uid: str, uname: str) -> None:
        acquired, msg = try_acquire_gpu(uid, uname, "抠像")
        if not acquired:
            ui.notify(msg, type="warning")
            return
        
        progress_q = queue.Queue()
        def cb(frac: float, desc: str):
            progress_q.put(("progress", frac, desc))
        def blocking():
            result = start_matting(
                r["matting_engine_selector"].value,
                int(r["erode_slider"].value), int(r["dilate_slider"].value),
                int(r["vm_batch_slider"].value), int(r["vm_overlap_slider"].value), int(r["vm_seed_input"].value),
                r["model_selector"].value, ps["session"], progress_callback=cb,
            )
            progress_q.put(("done", result))
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
                        progress_el.set_visibility(False)
                        release_gpu(uid)
                        _apply_notify(result)
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
        ui.label("批量管理多个抠像任务，可一次性执行或逐个处理。").classes("text-xs text-gray-500 mb-2")

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
            {"name": "video", "label": "视频名", "field": "video"},
            {"name": "frames", "label": "帧数", "field": "frames", "align": "center"},
            {"name": "kf", "label": "关键帧", "field": "kf", "align": "center"},
            {"name": "prop", "label": "已传播", "field": "prop", "align": "center"},
            {"name": "engine", "label": "引擎", "field": "engine", "align": "center"},
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
            ui.button("执行队列", on_click=lambda: _run_execute_queue(refs, user_id, user_name)).props("color=primary")
            ui.button("停止", on_click=lambda: _on_stop_queue(refs)).props("color=negative")
            ui.button("飞书通知", on_click=lambda: _apply_notify(send_feishu(load_queue()))).props("outline")
            ui.button("打包下载", on_click=lambda: _on_pack(refs))
        ui.label("恢复: 重新编辑该任务 | 移除: 从队列删除 | 执行队列: 自动处理所有待处理任务").classes("text-xs text-gray-400 mt-1")

    def _on_queue_restore(r: dict, ps: dict, uid: str, session_id: str) -> None:
        q = load_queue()
        if session_id not in q:
            ui.notify("该任务已不在队列中", type="warning")
            _update_queue_ui(r)
            return
        idx = q.index(session_id) + 1
        out = restore_from_queue(idx, ps["session"], q)
        ps["session"] = _apply_restore_out(out, r, uid, ps["session"], jump_to_frame)
        _apply_notify(out)

    def _on_queue_remove(r: dict, session_id: str) -> None:
        q = load_queue()
        if session_id not in q:
            ui.notify("该任务已不在队列中", type="warning")
            _update_queue_ui(r)
            return
        idx = q.index(session_id) + 1
        out = remove_from_queue(idx, q)
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

    def _run_execute_queue(r: dict, uid: str, uname: str) -> None:
        acquired, msg = try_acquire_gpu(uid, uname, "执行队列")
        if not acquired:
            ui.notify(msg, type="warning")
            return
        
        progress_q = queue.Queue()

        def cb(frac: float, desc: str):
            progress_q.put(("progress", frac, desc))

        def blocking():
            result = run_execute_queue(progress_callback=cb)
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
    def auto_refresh_queue():
        _update_queue_ui(refs)
    _start_page_timer(3.0, auto_refresh_queue)
    
    _update_queue_ui(refs)
    
    # Restore previous session if available
    if page_state["session"].get("session_id"):
        # Trigger UI update for restored session
        out = restore_session(page_state["session"]["session_id"], empty_state())
        if out.get("session_state"):
            page_state["session"] = _apply_restore_out(out, refs, user_id, page_state["session"], jump_to_frame)


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
