"""共享 UI 辅助函数 — 抠像页面与追踪页面共用。

包含：路径→URL 转换、关键帧 Gallery 刷新、通知适配器、
用户身份识别、会话状态持久化、帧预览图写入。
"""

import asyncio
import re
import time
from pathlib import Path

import cv2
import numpy as np
from nicegui import app, ui

from config import MATTING_SESSIONS_DIR, WORKSPACE_DIR
from matting.session_store import empty_state, load_session
from task_queue.models import load_queue
from task_queue.logic import queue_status_text, queue_table_rows

# 预览目录（由 app.py 启动时创建）
preview_dir = WORKSPACE_DIR / "preview"
tracking_preview_dir = WORKSPACE_DIR / "tracking_preview"


# ---------------------------------------------------------------------------
# 路径 → URL 转换
# ---------------------------------------------------------------------------
def session_path_to_url(path: str | None) -> str:
    """将 sessions 目录下的文件路径转换为浏览器可访问的 URL。"""
    if not path:
        return ""
    p = Path(path)
    try:
        rel = p.relative_to(MATTING_SESSIONS_DIR)
        return "/sessions/" + str(rel).replace("\\", "/")
    except ValueError:
        return path


def workspace_path_to_url(path: str | Path | None) -> str:
    """将 workspace 目录下的文件路径转换为浏览器可访问的 URL。"""
    if not path:
        return ""
    p = Path(path)
    try:
        rel = p.relative_to(WORKSPACE_DIR)
        return "/workspace/" + str(rel).replace("\\", "/")
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# 用户身份识别（Cloudflare Access）
# ---------------------------------------------------------------------------
def get_user_email_from_request(request) -> str:
    """从 Cloudflare Access 请求头中获取用户邮箱。

    部署在 Cloudflare Access 后，已认证用户的邮箱可通过
    'Cf-Access-Authenticated-User-Email' 请求头获取。
    本地开发环境无 Cloudflare 时，返回 'admin' 作为默认值。

    Args:
        request: FastAPI 的 Request 对象，来自 client.request。
    """
    if request is not None:
        email = request.headers.get("Cf-Access-Authenticated-User-Email")
        if email:
            return email
    return "admin"


def get_user_id_from_request(request) -> str:
    """根据邮箱生成唯一用户 ID（将 @ 和 . 替换为下划线）。"""
    email = get_user_email_from_request(request)
    return email.replace("@", "_at_").replace(".", "_")


def get_user_name_from_request(request) -> str:
    """获取用户显示名称（邮箱或 admin）。"""
    return get_user_email_from_request(request)


# ---------------------------------------------------------------------------
# 会话状态持久化（抠像页面）
# ---------------------------------------------------------------------------
def get_session_state() -> dict:
    """获取或创建当前用户的会话状态。

    会话状态在页面内存中以 refs 字典保存，但 session_id 会持久化到
    app.storage.user 中，以便页面刷新后恢复。
    """
    session_id = app.storage.user.get("current_session_id")
    if session_id:
        state = load_session(session_id)
        if state:
            return state
    return empty_state()


def save_session_id(session_id: str) -> None:
    """将当前 session_id 持久化到用户存储中。"""
    app.storage.user["current_session_id"] = session_id


# ---------------------------------------------------------------------------
# 帧预览图写入
# ---------------------------------------------------------------------------
def write_frame_preview(frame: np.ndarray, user_id: str) -> str:
    """将帧预览图写入用户专属目录，返回浏览器可访问的 URL。"""
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


def write_tracking_preview(frame: np.ndarray, user_id: str) -> str:
    """将追踪帧预览图写入用户专属目录，返回浏览器可访问的 URL。"""
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


# ---------------------------------------------------------------------------
# 队列表格行构建（附带 session_id 用于行操作）
# ---------------------------------------------------------------------------
def build_queue_rows_with_sid() -> list[dict]:
    """构建包含 session_id 的队列表格行数据，供表格行内按钮使用。"""
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


# ---------------------------------------------------------------------------
# 关键帧 Gallery 刷新
# ---------------------------------------------------------------------------
def refresh_gallery(
    container: ui.element,
    items: list,
    user_id: str,
    on_click_frame: callable = None,
) -> None:
    """刷新关键帧 Gallery，生成可点击的缩略图列表。

    Args:
        container: 用于填充缩略图的 UI 容器。
        items: (图片, 标题) 元组列表，标题格式如 "第 N 帧"。
        user_id: 用户 ID，用于预览路径隔离。
        on_click_frame: 可选回调 callback(frame_idx)，点击关键帧时触发。
    """
    user_preview_dir = preview_dir / user_id
    user_preview_dir.mkdir(exist_ok=True)

    # 清理该用户的旧关键帧预览图
    for old_file in user_preview_dir.glob("kf_*.png"):
        try:
            old_file.unlink()
        except OSError:
            pass

    container.clear()
    timestamp = int(time.time() * 1000)

    with container:
        for i, (img, caption) in enumerate(items):
            # 从标题（如 "第 N 帧"）中提取帧索引
            frame_idx = None
            match = re.search(r"第\s*(\d+)\s*帧", caption)
            if match:
                frame_idx = int(match.group(1))

            # 使用基于索引的固定文件名（不递增）
            path = user_preview_dir / f"kf_{i}.png"
            if img is not None and img.size > 0:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            with ui.column().classes("items-center cursor-pointer hover:bg-gray-100 rounded p-1"):
                img_url = f"/preview/{user_id}/kf_{i}.png?t={timestamp}" if img is not None and path.exists() else ""
                kf_image = ui.image(img_url).classes("w-20 h-14 object-cover")
                kf_label = ui.label(caption).classes("text-xs")

                if frame_idx is not None and on_click_frame is not None:
                    def make_click_handler(fidx):
                        def handler():
                            asyncio.create_task(on_click_frame(fidx))
                        return handler
                    kf_image.on("click", make_click_handler(frame_idx))
                    kf_label.on("click", make_click_handler(frame_idx))


# ---------------------------------------------------------------------------
# 通知适配器
# ---------------------------------------------------------------------------
def apply_notify(out: dict) -> None:
    """根据逻辑层返回字典中的 warning/info 键显示 UI 通知。"""
    if out.get("warning"):
        ui.notify(out["warning"], type="warning")
    if out.get("info"):
        ui.notify(out["info"], type="positive")


def apply_tracking_notify(out: dict) -> None:
    """根据追踪逻辑层返回字典中的 notify 键显示 UI 通知。"""
    if out.get("notify"):
        ntype, msg = out["notify"]
        ui.notify(msg, type=ntype)


# ---------------------------------------------------------------------------
# 队列 UI 更新辅助
# ---------------------------------------------------------------------------
def update_queue_ui(refs: dict) -> None:
    """刷新队列状态标签、表格行数据和计数徽章。"""
    if "queue_status" not in refs or "queue_table" not in refs:
        return
    q = load_queue()
    refs["queue_status"].set_text(queue_status_text(q))
    refs["queue_table"].rows = build_queue_rows_with_sid()
    _update_queue_counts(refs)


def _update_queue_counts(refs: dict) -> None:
    """根据表格行数据更新待处理/已完成/失败的计数徽章。"""
    if "queue_pending_count" not in refs:
        return
    rows = refs["queue_table"].rows
    pending = sum(1 for row in rows if row.get("status") in ("pending", ""))
    done = sum(1 for row in rows if row.get("status") == "done")
    failed = sum(1 for row in rows if row.get("status") in ("error", "failed"))
    refs["queue_pending_count"].set_text(str(pending))
    refs["queue_done_count"].set_text(str(done))
    refs["queue_failed_count"].set_text(str(failed))


# ---------------------------------------------------------------------------
# 恢复输出应用器（抠像页面）
# ---------------------------------------------------------------------------
def apply_restore_out(
    out: dict,
    refs: dict,
    user_id: str,
    session_state: dict,
    on_click_frame: callable = None,
) -> dict:
    """将恢复操作的输出应用到抠像页面的 UI 组件，返回更新后的 session_state。"""
    if "session_state" in out:
        session_state = out["session_state"]
        if session_state.get("session_id"):
            save_session_id(session_state["session_id"])
    if out.get("frame_image") is not None:
        refs["frame_image"].set_source(write_frame_preview(out["frame_image"], user_id))
    if out.get("frame_label") is not None:
        refs["frame_label"].set_text(out["frame_label"])
    if out.get("keyframe_info") is not None:
        refs["keyframe_info"].set_text(out["keyframe_info"])
    if out.get("keyframe_gallery") is not None:
        refresh_gallery(refs["keyframe_gallery_container"], out["keyframe_gallery"], user_id, on_click_frame)
    if out.get("video_path") is not None:
        refs["video_display"].set_source(session_path_to_url(out["video_path"]))
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
        refs["propagation_preview"].set_source(session_path_to_url(out["propagation_preview_path"]))
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
        refs["alpha_video"].set_source(session_path_to_url(out["alpha_path"]))
    if out.get("fgr_path") is not None:
        refs["fgr_video"].set_source(session_path_to_url(out["fgr_path"]))
    if out.get("session_choices") is not None:
        refs["session_dropdown"].options = {v: l for l, v in out["session_choices"]}
    if out.get("session_value") is not None:
        refs["session_dropdown"].value = out["session_value"]
    return session_state
