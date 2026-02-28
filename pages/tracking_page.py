"""追踪页面 — 使用 CoTracker 进行点追踪。

从 app.py 提取。包含以 ``@ui.page("/tracking")`` 装饰的
``tracking_page`` 函数，支持手动选点、网格模式、SAM 目标选择三种模式。
"""

import asyncio
import logging
import platform
import queue
import threading
from pathlib import Path

from nicegui import app, events, ui
from nicegui import run

from config import WORKSPACE_DIR, get_device
from engines.cotracker_engine import TrackingCancelledError
from gpu_lock import try_acquire_gpu, release_gpu, get_gpu_status
from pages.queue_panel import build_queue_panel
from pages.shared_ui import (
    apply_notify,
    apply_tracking_notify,
    build_queue_rows_with_sid,
    get_user_id_from_request,
    get_user_name_from_request,
    refresh_gallery,
    update_queue_ui,
    workspace_path_to_url,
    write_tracking_preview,
)
from task_queue.logic import (
    add_tracking_to_queue,
    queue_status_text,
    queue_table_rows,
    send_feishu,
)
from task_queue.models import load_queue
from tracking.export import export_ae_keyframe_data as ct_ae_export
from tracking.logic import (
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
from utils.user_context import set_current_user
from tracking.session_store import (
    list_tracking_sessions,
    save_tracking_session as ct_save_session,
)

log = logging.getLogger(__name__)


@ui.page("/tracking")
def tracking_page(client):
    """点追踪页面：上传视频、选择追踪点、运行追踪、导出结果。"""
    user_id = get_user_id_from_request(client.request)
    user_name = get_user_name_from_request(client.request)
    set_current_user(user_name)

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

    # 注入 CSS：覆盖 interactive_image 内部 <img> 的硬编码尺寸
    ui.add_head_html('''<style>
.frame-preview { max-height: 70vh; }
.frame-preview img {
    width: auto !important;
    height: auto !important;
    max-height: 70vh;
    max-width: 100%;
}
</style>''')

    # ---- 顶部导航栏 ----
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
                        refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                    if out.get("slider_max") is not None:
                        refs["frame_slider"].props["max"] = out["slider_max"]
                        refs["frame_slider"].value = out.get("slider_value", 0)
                        refs["frame_slider"].set_visibility(True)
                        refs["frame_input"].max = out["slider_max"]
                        refs["frame_input"].value = out.get("slider_value", 0)
                        refs["frame_input"].set_visibility(True)
                    refs["frame_label"].set_text(f"第 0 帧 / 共 {out.get('slider_max', 0)} 帧")
                    refs["video_display"].set_source(workspace_path_to_url(dest))
                    # 刷新 Session 列表并自动选中新建的 Session
                    out_refresh = ct_refresh_sessions()
                    if out_refresh.get("session_choices") and refs.get("tracking_session_dropdown"):
                        dd = refs["tracking_session_dropdown"]
                        dd.options = {v: l for l, v in out_refresh["session_choices"]}
                        new_sid = page_state["tracking"].get("session_id")
                        if new_sid:
                            dd.value = new_sid
                        dd.update()
                    apply_tracking_notify(out)

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
                    async def on_tracking_restore():
                        sid = refs["tracking_session_dropdown"].value
                        loading_note = ui.notification("正在恢复 Session…", type="ongoing", timeout=None, spinner=True)
                        try:
                            out = await run.io_bound(ct_restore_session, sid, page_state["tracking"])
                            if out.get("warning"):
                                ui.notify(out["warning"], type="warning")
                                return
                            page_state["tracking"] = out["session_state"]
                            if out.get("preview_frame") is not None:
                                refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                            if out.get("slider_max") is not None:
                                refs["frame_slider"].props["max"] = out["slider_max"]
                                refs["frame_slider"].value = out.get("slider_value", 0)
                                refs["frame_slider"].set_visibility(True)
                                refs["frame_input"].max = out["slider_max"]
                                refs["frame_input"].value = out.get("slider_value", 0)
                                refs["frame_input"].set_visibility(True)
                            refs["frame_label"].set_text(out.get("frame_label", "第 0 帧 / 共 0 帧"))
                            if out.get("keyframe_info"):
                                refs["tracking_kf_info"].set_text(out["keyframe_info"])
                            if out.get("keyframe_gallery") is not None:
                                refresh_gallery(refs["tracking_kf_gallery"], out["keyframe_gallery"], user_id, _jump_to_tracking_frame)
                            refs["point_count_label"].set_text(f"已选择 {out.get('point_count', 0)} 个追踪点")
                            if out.get("video_path"):
                                refs["video_display"].set_source(workspace_path_to_url(out["video_path"]))
                            if out.get("result_video_path"):
                                refs["result_video"].set_source(workspace_path_to_url(out["result_video_path"]))
                            if refs.get("backward_tracking") is not None:
                                refs["backward_tracking"].value = out.get("backward_tracking", False)
                            if refs.get("grid_size") is not None:
                                refs["grid_size"].value = out.get("grid_size", 15)
                            ui.notify("Session 恢复成功", type="positive")
                        except Exception as ex:
                            log.exception("Restore tracking session failed")
                            ui.notify(f"恢复失败: {ex}", type="negative")
                        finally:
                            loading_note.dismiss()
                    ui.button("恢复", on_click=on_tracking_restore).props("color=primary")
                ui.label("选择历史 Session 可恢复之前的追踪点、追踪结果。").classes("text-xs text-gray-400")

    ui.separator()

    # ======================================================================
    # 区域 2：选择追踪点
    # ======================================================================
    with ui.expansion("② 选择追踪点", icon="touch_app").classes("w-full").props("default-opened"):
        ui.label("点击图片添加追踪点，使用网格模式自动生成，或用 SAM 选择目标区域。").classes("text-xs text-gray-500 mb-2")

        # 追踪模式选择（手动/网格/SAM）
        track_mode_toggle = ui.toggle(["手动选点", "网格模式", "SAM目标选择"], value="手动选点").classes("mb-2")
        refs["track_mode"] = track_mode_toggle

        # 手动/网格模式工具栏
        with ui.row().classes("gap-2 flex-wrap items-center mb-2") as manual_toolbar:
            async def on_tracking_undo():
                out = await run.io_bound(ct_undo_point, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                refs["point_count_label"].set_text(f"已选择 {out.get('query_count', 0)} 个追踪点")

            async def on_tracking_clear_frame():
                out = await run.io_bound(ct_clear_frame_points, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                refs["point_count_label"].set_text(f"已选择 {out.get('query_count', 0)} 个追踪点")

            async def on_tracking_clear_all():
                out = await run.io_bound(ct_clear_all_points, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                refs["point_count_label"].set_text("已选择 0 个追踪点")

            ui.button("撤销", on_click=on_tracking_undo, icon="undo")
            ui.button("清除当前帧", on_click=on_tracking_clear_frame, icon="clear")
            ui.button("清除全部", on_click=on_tracking_clear_all, icon="delete_sweep")
            ui.separator().props("vertical")
            grid_size_input = ui.number("网格大小", value=15, min=3, max=30, step=1).classes("w-24")
            refs["grid_size"] = grid_size_input
        refs["manual_toolbar"] = manual_toolbar

        # 关键帧回调（按钮放在侧边栏，回调提前定义）
        async def on_save_tracking_kf():
            out = await run.io_bound(ct_save_kf, page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            apply_tracking_notify(out)
            if out.get("keyframe_info"):
                refs["tracking_kf_info"].set_text(out["keyframe_info"])
            if out.get("keyframe_gallery") is not None:
                refresh_gallery(refs["tracking_kf_gallery"], out["keyframe_gallery"], user_id, _jump_to_tracking_frame)
            if "query_count" in out:
                refs["point_count_label"].set_text(f"已选择 {out['query_count']} 个追踪点")
            await run.io_bound(ct_save_session, page_state["tracking"])

        async def on_del_tracking_kf():
            out = await run.io_bound(ct_del_kf, page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            apply_tracking_notify(out)
            if out.get("preview_frame") is not None:
                refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
            if out.get("keyframe_info"):
                refs["tracking_kf_info"].set_text(out["keyframe_info"])
            if out.get("keyframe_gallery") is not None:
                refresh_gallery(refs["tracking_kf_gallery"], out["keyframe_gallery"], user_id, _jump_to_tracking_frame)
            refs["point_count_label"].set_text(f"已选择 {page_state['tracking'].get('query_count', 0)} 个追踪点")
            await run.io_bound(ct_save_session, page_state["tracking"])

        # SAM 目标选择工具栏（仅 SAM 模式可见）
        with ui.column().classes("gap-2 mb-2") as sam_toolbar:
            # 第一行：SAM 模型和点击模式选择
            with ui.row().classes("gap-2 flex-wrap items-center"):
                with ui.column().classes("gap-0"):
                    ui.label("分割模型").classes("text-xs text-gray-500")
                    sam_model_select = ui.radio(["SAM2", "SAM3"], value="SAM2")
                    refs["sam_model"] = sam_model_select
                with ui.column().classes("gap-0"):
                    ui.label("点击模式").classes("text-xs text-gray-500")
                    sam_point_mode = ui.radio(["Positive", "Negative"], value="Positive")
                    refs["sam_point_mode"] = sam_point_mode
                ui.label("Positive=选中目标，Negative=排除区域").classes("text-xs text-gray-400")

            # SAM 回调
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
                        refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                finally:
                    release_gpu(user_id)
                    refs["loading_overlay"].set_visibility(False)

            async def on_sam_clear():
                out = await run.io_bound(ct_sam_clear, page_state["tracking"])
                page_state["tracking"] = out["session_state"]
                # 同时清除当前帧追踪点
                out2 = await run.io_bound(ct_clear_frame_points, page_state["tracking"])
                page_state["tracking"] = out2["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                refs["point_count_label"].set_text(f"已选择 {out2.get('query_count', 0)} 个追踪点")

            async def on_generate_from_mask():
                num_points = int(refs["num_points_sam"].value or 30)
                out = await run.io_bound(ct_generate_points_from_mask, page_state["tracking"], num_points)
                page_state["tracking"] = out["session_state"]
                if out.get("preview_frame") is not None:
                    refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                if out.get("query_count") is not None:
                    refs["point_count_label"].set_text(f"已选择 {out['query_count']} 个追踪点")
                if out.get("notify"):
                    ntype, nmsg = out["notify"]
                    ui.notify(nmsg, type=ntype)

            # 第二行：撤销、追踪点数、生成、清除
            with ui.row().classes("gap-2 items-center"):
                ui.button("撤销", on_click=on_sam_undo, icon="undo")
                num_points_sam = ui.number("Mask 内追踪点数", value=30, min=5, max=200, step=5).classes("w-32")
                refs["num_points_sam"] = num_points_sam
                ui.button("重新从 Mask 生成追踪点", on_click=on_generate_from_mask, icon="scatter_plot").props("color=primary")
                ui.button("清除 Mask 和追踪点", on_click=on_sam_clear, icon="clear")
        refs["sam_toolbar"] = sam_toolbar
        sam_toolbar.set_visibility(False)

        # 模式切换回调
        def on_track_mode_change():
            mode = refs["track_mode"].value
            refs["manual_toolbar"].set_visibility(mode != "SAM目标选择")
            refs["sam_toolbar"].set_visibility(mode == "SAM目标选择")
        track_mode_toggle.on("update:model-value", on_track_mode_change)

        # 帧滑块 + 帧号输入框
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            tracking_frame_slider = ui.slider(min=0, max=1, value=0, step=1).props("label-always").classes("flex-1")
            tracking_frame_slider._props['loopback'] = False
            tracking_frame_slider.set_visibility(False)
            refs["frame_slider"] = tracking_frame_slider
            tracking_frame_input = ui.number(min=0, max=1, value=0, step=1).classes("w-24").props("dense outlined")
            tracking_frame_input.set_visibility(False)
            refs["frame_input"] = tracking_frame_input
            tracking_frame_slider.bind_value(tracking_frame_input)

        def _load_tracking_frame(frame_idx: int):
            """加载追踪帧并更新标签。slider/input 同步由 bind_value 处理。"""
            out = ct_change_frame(frame_idx, page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            if out.get("preview_frame") is not None:
                refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
            if out.get("frame_label"):
                refs["frame_label"].set_text(out["frame_label"])

        def on_tracking_slider_change(e):
            v = e.args
            if v is None or page_state["tracking"].get("frames_dir") is None:
                return
            _load_tracking_frame(int(v))

        tracking_frame_slider.on("update:model-value", on_tracking_slider_change, [None], throttle=0.05)

        def on_tracking_frame_input():
            if page_state["tracking"].get("frames_dir") is None:
                return
            v = refs["frame_input"].value
            if v is None:
                return
            max_v = int(refs["frame_slider"].props.get("max", 1))
            clamped = max(0, min(int(v), max_v))
            refs["frame_slider"].value = clamped  # bind_value 同步 input
            _load_tracking_frame(clamped)

        tracking_frame_input.on("change", on_tracking_frame_input)

        # 帧标签和追踪点计数
        with ui.row().classes("w-full justify-between items-center mb-1"):
            tracking_frame_label = ui.label("请先上传视频。").classes("text-sm text-gray-600")
            refs["frame_label"] = tracking_frame_label
            point_count_label = ui.label("已选择 0 个追踪点").classes("text-sm text-blue-600")
            refs["point_count_label"] = point_count_label

        # 跳转到指定帧（关键帧 Gallery 点击回调）
        def _jump_to_tracking_frame(frame_idx: int):
            refs["frame_slider"].value = frame_idx  # bind_value 同步 input
            _load_tracking_frame(frame_idx)

        # 交互式图片（点击选点）+ 关键帧 Gallery 侧边栏
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
                        refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
                except Exception as ex:
                    log.exception("SAM prediction failed")
                    ui.notify(f"SAM 推理失败: {ex}", type="negative")
                finally:
                    release_gpu(user_id)
                    refs["loading_overlay"].set_visibility(False)
                    click_state["busy"] = False
                return

            # 手动选点模式：直接在图片上添加追踪点
            out = await run.io_bound(ct_add_point, e.image_x, e.image_y, page_state["tracking"])
            page_state["tracking"] = out["session_state"]
            if out.get("preview_frame") is not None:
                refs["frame_image"].set_source(write_tracking_preview(out["preview_frame"], user_id))
            refs["point_count_label"].set_text(f"已选择 {out.get('query_count', 0)} 个追踪点")

        with ui.row().classes("w-full gap-4 items-start"):
            with ui.column().classes("flex-1"):
                with ui.element("div").classes("relative w-full"):
                    tracking_frame_image = ui.interactive_image(
                        "", on_mouse=_on_tracking_mouse, events=["click"],
                    ).classes("frame-preview")
                    refs["frame_image"] = tracking_frame_image

                    # 追踪加载遮罩层（推理时显示）
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

            # 关键帧 Gallery 侧边栏（显示已保存的关键帧缩略图）
            with ui.column().classes("w-48"):
                with ui.row().classes("gap-1 items-center"):
                    ui.button("保存关键帧", on_click=on_save_tracking_kf, icon="save").props("color=primary dense")
                    ui.button("删除关键帧", on_click=on_del_tracking_kf, icon="delete").props("dense")
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
            """追踪完成或取消后重置 UI 状态。"""
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
            # 如果追踪正在运行，则停止
            if tracking_cancel_event is not None:
                tracking_cancel_event.set()
                ui.notify("正在停止追踪…", type="info")
                return

            # --- 启动追踪 ---
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

            # 切换按钮为 "停止" 状态
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
                            apply_tracking_notify(result)
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
                # 将当前参数保存到状态中
                mode = refs["track_mode"].value
                st["use_grid"] = (mode == "网格模式")
                st["grid_size"] = int(refs["grid_size"].value or 15)
                st["backward_tracking"] = refs["backward_tracking"].value
                q = load_queue()
                out = await run.io_bound(add_tracking_to_queue, st, q)
                page_state["tracking"] = out["session_state"]
                if out.get("queue_state") is not None:
                    update_queue_ui(refs)
                # 重置帧显示区域
                refs["frame_image"].set_source("")
                refs["frame_slider"].set_visibility(False)
                refs["frame_input"].set_visibility(False)
                refs["frame_label"].set_text("请先上传视频。")
                refs["point_count_label"].set_text("已选择 0 个追踪点")
                refs["tracking_kf_info"].set_text("尚未保存任何关键帧。")
                refs["tracking_kf_gallery"].clear()
                apply_tracking_notify(out)
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
            apply_tracking_notify(out)
            if out.get("export_path"):
                export_path = Path(out["export_path"])
                rel = export_path.relative_to(WORKSPACE_DIR)
                ui.download(f"/workspace/{rel}", filename="ae_原始轨迹.txt")

        ui.button("导出 After Effects 关键帧", on_click=on_ae_export, icon="download")
        ui.label("导出 Adobe After Effects 关键帧数据（.txt），可直接粘贴到 AE。").classes("text-xs text-gray-400")

        async def on_summary_txt_export():
            txt_path = page_state["tracking"].get("ae_summary_txt_path", "")
            if not txt_path or not Path(txt_path).exists():
                ui.notify("整体轨迹尚未生成，请先运行追踪。", type="warning")
                return
            try:
                rel = Path(txt_path).relative_to(WORKSPACE_DIR)
            except ValueError:
                ui.notify("文件路径异常。", type="negative")
                return
            ui.download(f"/workspace/{rel}", filename="ae_整体轨迹.txt")

        async def on_summary_jsx_export():
            jsx_path = page_state["tracking"].get("ae_summary_jsx_path", "")
            if not jsx_path or not Path(jsx_path).exists():
                ui.notify("整体轨迹尚未生成，请先运行追踪。", type="warning")
                return
            try:
                rel = Path(jsx_path).relative_to(WORKSPACE_DIR)
            except ValueError:
                ui.notify("文件路径异常。", type="negative")
                return
            ui.download(f"/workspace/{rel}", filename="ae_整体轨迹.jsx")

        ui.button("导出整体轨迹 TXT", on_click=on_summary_txt_export, icon="timeline")
        ui.button("导出整体轨迹 JSX", on_click=on_summary_jsx_export, icon="code")
        ui.label("整体轨迹：IQR 离群点剔除后各帧平均位置，适合跟踪物体整体运动。").classes("text-xs text-gray-400")

    ui.separator()

    # ======================================================================
    # 区域 4：统一任务队列
    # ======================================================================
    build_queue_panel(refs, page_state, user_id, user_name, "tracking", _start_page_timer, _start_until_false_timer)
