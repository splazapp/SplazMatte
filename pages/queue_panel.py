"""共享任务队列面板 UI 组件 — 抠像页面与追踪页面共用。"""

import asyncio
import logging
import queue
import threading

from nicegui import app, ui
from nicegui import run

from config import WORKSPACE_DIR
from task_queue.models import load_queue
from task_queue.logic import (
    clear_queue,
    pack_download,
    pin_to_bottom_from_queue,
    pin_to_top_from_queue,
    queue_status_text,
    queue_table_rows,
    remove_from_queue,
    reset_status,
    restore_from_queue,
    run_execute_queue,
    send_feishu,
    stop_queue,
)
from gpu_lock import try_acquire_gpu, release_gpu
from pages.shared_ui import (
    apply_notify,
    apply_restore_out,
    build_queue_rows_with_sid,
    session_path_to_url,
    update_queue_ui,
)

log = logging.getLogger(__name__)


def build_queue_panel(
    refs: dict,
    page_state: dict,
    user_id: str,
    user_name: str,
    panel_type: str,
    start_page_timer_fn: callable,
    start_until_false_timer_fn: callable,
    jump_to_frame_fn: callable = None,
) -> None:
    """构建任务队列面板 UI。

    Args:
        refs: UI 组件引用字典。
        page_state: 页面状态字典。
        user_id: 当前用户 ID。
        user_name: 当前用户显示名称。
        panel_type: "matting"（含恢复按钮）或 "tracking"（不含恢复按钮）。
        start_page_timer_fn: 启动页面周期定时器的函数。
        start_until_false_timer_fn: 启动轮询定时器的函数（返回 False 时自动停止）。
        jump_to_frame_fn: 可选回调，用于抠像页面的帧跳转。
    """
    icon_num = "⑤" if panel_type == "matting" else "④"
    with ui.expansion(f"{icon_num} 任务队列", icon="playlist_play").classes("w-full").props("default-opened"):
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
        queue_table = ui.table(
            columns=columns, rows=build_queue_rows_with_sid(), row_key="session_id",
        ).classes("w-full")
        refs["queue_table"] = queue_table

        # 表格行内操作按钮
        with queue_table.add_slot("body-cell-action"):
            with queue_table.cell("action"):
                with ui.row().classes("gap-1"):
                    ui.button("置顶", icon="vertical_align_top").props("flat dense size=sm").on(
                        "click",
                        js_handler="() => emit(props.row.session_id)",
                        handler=lambda e: _on_queue_pin_top(
                            refs,
                            (e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args),
                        ),
                    )
                    if panel_type == "matting":
                        ui.button("恢复", icon="edit").props("flat dense size=sm").on(
                            "click",
                            js_handler="() => emit(props.row.session_id)",
                            handler=lambda e: _on_queue_restore(
                                refs, page_state, user_id,
                                (e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args),
                                jump_to_frame_fn,
                            ),
                        )
                    ui.button("移除", icon="delete").props("flat dense size=sm color=negative").on(
                        "click",
                        js_handler="() => emit(props.row.session_id)",
                        handler=lambda e: _on_queue_remove(
                            refs,
                            (e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args),
                        ),
                    )
                    ui.button("置底", icon="vertical_align_bottom").props("flat dense size=sm").on(
                        "click",
                        js_handler="() => emit(props.row.session_id)",
                        handler=lambda e: _on_queue_pin_bottom(
                            refs,
                            (e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args),
                        ),
                    )

        # 进度显示区
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
                on_click=lambda: _on_queue_execute_or_stop(refs, user_id, user_name, start_until_false_timer_fn),
            ).props("color=primary")
            refs["queue_execute_btn"] = queue_execute_btn
            refs["queue_executing"] = False
            ui.button("飞书通知", on_click=lambda: apply_notify(send_feishu(load_queue()))).props("outline")
            ui.button("打包下载", on_click=lambda: _on_pack(refs)).props("outline")

        hint = "恢复: 重新编辑该任务 | 移除: 从队列删除 | 新任务加入后自动继续执行" if panel_type == "matting" else "新任务加入后自动继续执行。"
        ui.label(hint).classes("text-xs text-gray-400 mt-1")

    # 自动刷新队列 UI（多用户可见性）
    _last_queue_rows: list[dict] = []

    def auto_refresh_queue():
        nonlocal _last_queue_rows
        rows = build_queue_rows_with_sid()
        if rows != _last_queue_rows:
            _last_queue_rows = rows
            update_queue_ui(refs)
    start_page_timer_fn(3.0, auto_refresh_queue)

    update_queue_ui(refs)


# ---------------------------------------------------------------------------
# 内部回调函数
# ---------------------------------------------------------------------------
def _on_queue_restore(refs, page_state, user_id, session_id, jump_to_frame_fn):
    """将队列中的任务恢复到编辑区域（仅限抠像页面）。"""
    q = load_queue()
    match_idx = next((i for i, item in enumerate(q) if item["sid"] == session_id), None)
    if match_idx is None:
        ui.notify("该任务已不在队列中", type="warning")
        update_queue_ui(refs)
        return
    out = restore_from_queue(match_idx + 1, page_state["session"], q)
    if out.get("restore_type") == "tracking":
        ui.notify("追踪任务请在追踪页面编辑", type="info")
    else:
        page_state["session"] = apply_restore_out(out, refs, user_id, page_state["session"], jump_to_frame_fn)
    apply_notify(out)


def _on_queue_remove(refs, session_id):
    """根据 session_id 从队列中移除任务。"""
    q = load_queue()
    match_idx = next((i for i, item in enumerate(q) if item["sid"] == session_id), None)
    if match_idx is None:
        ui.notify("该任务已不在队列中", type="warning")
        update_queue_ui(refs)
        return
    out = remove_from_queue(match_idx + 1, q)
    refs["queue_status"].set_text(out["queue_status_text"])
    refs["queue_table"].rows = build_queue_rows_with_sid()
    update_queue_ui(refs)
    apply_notify(out)


def _queue_act(refs, fn):
    """执行队列操作（清空/重置）并刷新 UI。"""
    q = load_queue()
    out = fn(q)
    refs["queue_status"].set_text(out["queue_status_text"])
    refs["queue_table"].rows = build_queue_rows_with_sid()
    update_queue_ui(refs)
    apply_notify(out)


def _on_queue_execute_or_stop(refs, user_id, user_name, start_until_false_timer_fn):
    """切换：空闲时启动队列执行，运行中时停止。"""
    if refs.get("queue_executing"):
        out = stop_queue()
        refs["queue_progress_bar"].set_visibility(False)
        refs["queue_progress_label"].set_text("已停止")
        refs["queue_current_task"].set_text("")
        apply_notify(out)
    else:
        _run_execute_queue(refs, user_id, user_name, start_until_false_timer_fn)


def _run_execute_queue(refs, user_id, user_name, start_until_false_timer_fn):
    """在后台启动队列执行，通过线程安全队列传递进度更新。"""
    acquired, msg = try_acquire_gpu(user_id, user_name, "执行队列")
    if not acquired:
        ui.notify(msg, type="warning")
        return

    refs["queue_executing"] = True
    btn = refs.get("queue_execute_btn")
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

    refs["queue_progress_bar"].set_visibility(True)
    refs["queue_progress_bar"].value = 0
    refs["queue_progress_label"].set_text("正在执行队列...")
    refs["queue_current_task"].set_text("")

    asyncio.get_event_loop().create_task(start())

    def poll():
        try:
            while True:
                item = progress_q.get_nowait()
                if item[0] == "progress":
                    frac, desc = item[1], item[2]
                    refs["queue_progress_bar"].value = frac
                    refs["queue_progress_label"].set_text(f"进度: {frac * 100:.1f}%")
                    refs["queue_current_task"].set_text(desc)
                    refs["queue_table"].rows = build_queue_rows_with_sid()
                    update_queue_ui(refs)
                elif item[0] == "done":
                    result = item[1]
                    refs["queue_executing"] = False
                    if btn:
                        btn.set_text("开始执行队列")
                        btn.props("color=primary")
                    refs["queue_status"].set_text(result["queue_status_text"])
                    refs["queue_table"].rows = build_queue_rows_with_sid()
                    update_queue_ui(refs)
                    refs["queue_progress_bar"].value = 1
                    refs["queue_progress_bar"].set_visibility(False)
                    refs["queue_progress_label"].set_text("队列执行完成")
                    refs["queue_current_task"].set_text("")
                    release_gpu(user_id)
                    apply_notify(result)
                    return False
        except queue.Empty:
            pass
        return True

    start_until_false_timer_fn(0.2, poll)


def _on_queue_pin_top(refs, session_id):
    """将指定任务移动到队列顶部（index 0）。"""
    q = load_queue()
    match_idx = next((i for i, item in enumerate(q) if item["sid"] == session_id), None)
    if match_idx is None:
        ui.notify("该任务已不在队列中", type="warning")
        update_queue_ui(refs)
        return
    out = pin_to_top_from_queue(match_idx + 1, q)
    refs["queue_status"].set_text(out["queue_status_text"])
    refs["queue_table"].rows = build_queue_rows_with_sid()
    update_queue_ui(refs)
    apply_notify(out)


def _on_queue_pin_bottom(refs, session_id):
    """将指定任务移动到队列底部（最后一位）。"""
    q = load_queue()
    match_idx = next((i for i, item in enumerate(q) if item["sid"] == session_id), None)
    if match_idx is None:
        ui.notify("该任务已不在队列中", type="warning")
        update_queue_ui(refs)
        return
    out = pin_to_bottom_from_queue(match_idx + 1, q)
    refs["queue_status"].set_text(out["queue_status_text"])
    refs["queue_table"].rows = build_queue_rows_with_sid()
    update_queue_ui(refs)
    apply_notify(out)


def _on_pack(refs):
    """打包结果文件并触发浏览器下载。"""
    q = load_queue()
    out = pack_download(q)
    if out.get("download_path"):
        ui.download("/workspace/results.zip", "results.zip")
    apply_notify(out)
