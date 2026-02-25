"""SplazMatte 抠像页面 — 从 app.py main_page 提取。

包含视频上传/恢复、关键帧标注（SAM2/SAM3）、遮罩传播、
精细抠像（MatAnyone/VideoMaMa）、结果预览、任务队列等功能区域。
"""

import asyncio
import logging
import platform
import queue
import threading
from pathlib import Path

from nicegui import app, events, ui
from nicegui import run

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    PROCESSING_LOG_FILE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
    get_device,
)
from matting.session_store import empty_state, list_sessions
from task_queue.models import load_queue
from matting.logic import (
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
from task_queue.logic import add_to_queue
from gpu_lock import try_acquire_gpu, release_gpu, get_gpu_status
from matting.runner import MattingCancelledError
from pages.shared_ui import (
    write_frame_preview,
    session_path_to_url,
    get_user_id_from_request,
    get_user_name_from_request,
    refresh_gallery,
    apply_notify,
    build_queue_rows_with_sid,
    get_session_state,
    save_session_id,
    apply_restore_out,
    update_queue_ui,
)
from pages.queue_panel import build_queue_panel

log = logging.getLogger(__name__)


@ui.page("/")
def matting_page(client):
    """抠像主页面：视频上传、标注、传播、抠像、结果预览。"""
    # 从请求头获取用户身份（Cloudflare Access）
    user_id = get_user_id_from_request(client.request)
    user_name = get_user_name_from_request(client.request)

    # 页面级会话状态（可变字典，在当前页面实例中持久化）
    page_state = {"session": get_session_state()}

    device = get_device()
    device_label = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
    hostname = platform.node() or "unknown"

    refs = {}
    page_timers = set()

    def _start_page_timer(interval: float, callback, *, once: bool = False, immediate: bool = True):
        """启动页面级定时器并记录，便于页面关闭时统一清理。"""
        timer = app.timer(interval, callback, once=once, immediate=immediate)
        page_timers.add(timer)
        return timer

    def _stop_page_timer(timer) -> None:
        """安全取消已跟踪的页面定时器。"""
        if timer is None:
            return
        timer.cancel(with_current_invocation=True)
        page_timers.discard(timer)

    def _cleanup_page_timers(*_) -> None:
        """页面/客户端生命周期结束时取消所有定时器。"""
        for timer in list(page_timers):
            timer.cancel(with_current_invocation=True)
        page_timers.clear()

    def _start_until_false_timer(interval: float, poll_fn) -> None:
        """启动轮询定时器，当 poll_fn 返回 False 或抛出异常时自动停止。"""
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

    # 跳转到指定帧（关键帧 Gallery 点击时使用）
    async def jump_to_frame(frame_idx: int):
        """点击关键帧缩略图时跳转到指定帧。"""
        out = await run.io_bound(slider_change, frame_idx, page_state["session"])
        page_state["session"] = out["session_state"]
        if out.get("frame_image") is not None:
            refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
        if out.get("frame_label"):
            refs["frame_label"].set_text(out["frame_label"])
        # 同步更新滑块位置
        if "frame_slider" in refs:
            refs["frame_slider"].value = frame_idx
        if "frame_input" in refs:
            refs["frame_input"].value = frame_idx

    # ---- 顶部导航栏 ----
    with ui.row().classes("w-full items-center justify-between"):
        with ui.row().classes("items-center gap-4"):
            ui.label("SplazMatte").classes("text-2xl font-bold")
            ui.link("抠像", "/").classes("text-blue-600 underline font-medium")
            ui.link("轨迹追踪", "/tracking").classes("text-gray-600 hover:text-blue-600")
        with ui.row().classes("items-center gap-4"):
            # GPU 状态指示器
            gpu_status_label = ui.label("").classes("text-xs")
            refs["gpu_status_label"] = gpu_status_label
            ui.label(f"用户: {user_name}").classes("text-sm text-gray-600")
            ui.label(f"运行设备: {device_label.get(device.type, device.type)} | 主机: {hostname}").classes("text-sm text-gray-600")

    # GPU 状态轮询
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
                        save_session_id(page_state["session"]["session_id"])
                    if out.get("frame_image") is not None:
                        refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
                    if out.get("frame_label"):
                        refs["frame_label"].set_text(out["frame_label"])
                    if out.get("keyframe_info"):
                        refs["keyframe_info"].set_text(out["keyframe_info"])
                    if out.get("keyframe_gallery") is not None:
                        refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, jump_to_frame)
                    if out.get("slider_visible") is not None:
                        refs["frame_slider"].set_visibility(out["slider_visible"])
                        refs["frame_slider"].props["max"] = out.get("slider_max", 0)
                        refs["frame_slider"].value = out.get("slider_value", 0)
                        refs["frame_input"].set_visibility(out["slider_visible"])
                        refs["frame_input"].max = out.get("slider_max", 0)
                        refs["frame_input"].value = out.get("slider_value", 0)
                    if out.get("session_choices") is not None:
                        refs["session_dropdown"].options = {v: l for l, v in out["session_choices"]}
                        refs["session_dropdown"].value = out.get("session_value")
                    st = out["session_state"]
                    if st.get("session_id") and st.get("source_video_path"):
                        src = Path(st["source_video_path"])
                        refs["video_display"].set_source(f"/sessions/{st['session_id']}/{src.name}")
                    apply_notify(out)

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
                        page_state["session"] = apply_restore_out(out, refs, user_id, page_state["session"], jump_to_frame)
                        apply_notify(out)
                    ui.button("恢复", on_click=on_restore).props("color=primary")
                ui.label("选择历史 Session 可恢复之前的标注、传播、抠像结果。").classes("text-xs text-gray-400")

    ui.separator()

    # ======================================================================
    # 区域 2：标注关键帧
    # ======================================================================
    with ui.expansion("② 标注关键帧", icon="touch_app").classes("w-full").props("default-opened"):
        ui.label("在关键帧上点击标注目标对象，SAM 模型会自动分割。建议在目标出现/消失/遮挡变化的帧上标注。").classes("text-xs text-gray-500 mb-2")
        # 标注工具栏
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
                        refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
                    apply_notify(out)
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
                        refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
                    apply_notify(out)
                finally:
                    if "annotation_loading_overlay" in refs:
                        refs["annotation_loading_overlay"].set_visibility(False)

            async def on_save_kf():
                out = await run.io_bound(save_keyframe, page_state["session"])
                page_state["session"] = out["session_state"]
                if out.get("keyframe_info"):
                    refs["keyframe_info"].set_text(out["keyframe_info"])
                if out.get("keyframe_gallery") is not None:
                    refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, jump_to_frame)
                apply_notify(out)

            async def on_del_kf():
                out = await run.io_bound(delete_keyframe, page_state["session"])
                page_state["session"] = out["session_state"]
                if out.get("frame_image") is not None:
                    refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
                if out.get("keyframe_info"):
                    refs["keyframe_info"].set_text(out["keyframe_info"])
                if out.get("keyframe_gallery") is not None:
                    refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, jump_to_frame)
                apply_notify(out)

            ui.button("撤销", on_click=on_undo)
            ui.button("清除", on_click=on_clear)
            ui.button("保存关键帧", on_click=on_save_kf).props("color=primary")
            ui.button("删除关键帧", on_click=on_del_kf)

        # 帧滑块 + 帧号输入框
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            frame_slider = ui.slider(min=0, max=1, value=0, step=1).props("label-always").classes("flex-1")
            refs["frame_slider"] = frame_slider
            frame_input = ui.number(min=0, max=1, value=0, step=1).classes("w-24").props("dense outlined")
            refs["frame_input"] = frame_input

        _slider_busy = {"value": False}
        _slider_pending = {"value": None}

        async def on_slider_val():
            v = refs["frame_slider"].value
            if v is None:
                return
            if _slider_busy["value"]:
                _slider_pending["value"] = v
                return
            _slider_busy["value"] = True
            try:
                current_v = v
                while True:
                    out = await run.io_bound(slider_change, int(current_v), page_state["session"])
                    page_state["session"] = out["session_state"]
                    if out.get("frame_image") is not None:
                        refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
                    if out.get("frame_label"):
                        refs["frame_label"].set_text(out["frame_label"])
                    refs["frame_input"].value = int(current_v)
                    if _slider_pending["value"] is None:
                        break
                    current_v = _slider_pending["value"]
                    _slider_pending["value"] = None
            finally:
                _slider_busy["value"] = False

        frame_slider.on("update:model-value", on_slider_val)

        async def on_frame_input():
            v = refs["frame_input"].value
            if v is None:
                return
            max_v = int(refs["frame_slider"].props.get("max", 1))
            clamped = max(0, min(int(v), max_v))
            refs["frame_slider"].value = clamped
            refs["frame_input"].value = clamped
            await on_slider_val()

        frame_input.on("change", on_frame_input)

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

                    # 尝试获取 GPU 锁
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
                            refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
                        apply_notify(out)
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
                                refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
                            apply_notify(out)
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
                    refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
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
                                    refs["propagation_preview"].set_source(session_path_to_url(result["propagation_preview_path"]))
                                progress_bar.set_visibility(False)
                                release_gpu(user_id)
                                apply_notify(result)
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

    # 抠像结果预览
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
        """抠像完成或取消后重置 UI 状态。"""
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
        # 如果抠像正在运行，则停止
        if matting_cancel_event is not None:
            matting_cancel_event.set()
            ui.notify("正在停止抠像…", type="info")
            return

        acquired, msg = try_acquire_gpu(uid, uname, "抠像")
        if not acquired:
            ui.notify(msg, type="warning")
            return

        # 切换按钮为 "停止" 状态
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
                            r["alpha_video"].set_source(session_path_to_url(result["alpha_path"]))
                        if result.get("fgr_path"):
                            r["fgr_video"].set_source(session_path_to_url(result["fgr_path"]))
                        _finish_matting(r, progress_el, uid)
                        apply_notify(result)
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
        r["queue_table"].rows = build_queue_rows_with_sid()
        update_queue_ui(r)
        if out.get("frame_image") is not None:
            r["frame_image"].set_source(write_frame_preview(out["frame_image"], uid) if out["frame_image"] else "")
        if out.get("frame_label"):
            r["frame_label"].set_text(out["frame_label"])
        if out.get("keyframe_info"):
            r["keyframe_info"].set_text(out["keyframe_info"])
        if out.get("keyframe_gallery") is not None:
            refresh_gallery(r["keyframe_gallery_container"], out["keyframe_gallery"], uid, jump_to_frame)
        r["frame_slider"].set_visibility(out.get("slider_visible", False))
        r["frame_input"].set_visibility(out.get("slider_visible", False))
        r["alpha_video"].set_source(session_path_to_url(out.get("alpha_path")))
        r["fgr_video"].set_source(session_path_to_url(out.get("fgr_path")))
        apply_notify(out)

    ui.separator()

    # ======================================================================
    # 区域 5：任务队列 (shared component)
    # ======================================================================
    build_queue_panel(refs, page_state, user_id, user_name, "matting", _start_page_timer, _start_until_false_timer, jump_to_frame)

    # 如果有之前的会话，自动恢复
    if page_state["session"].get("session_id"):
        # 触发 UI 更新以恢复之前的会话
        out = restore_session(page_state["session"]["session_id"], empty_state())
        if out.get("session_state"):
            page_state["session"] = apply_restore_out(out, refs, user_id, page_state["session"], jump_to_frame)
