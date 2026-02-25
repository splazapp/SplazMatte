"""UI-agnostic application logic for SplazMatte.

All functions take state/params and return a structured dict. No Gradio or
NiceGUI imports. Used by both Gradio adapters and the NiceGUI app.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import imageio
import numpy as np

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    MATTING_SESSIONS_DIR,
    PREVIEW_MAX_H,
    PREVIEW_MAX_W,
    PROCESSING_LOG_FILE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
)
from matting.runner import get_video_engine, run_matting_task
from pipeline.video_io import extract_frames, load_frame
from matting.session_store import (
    empty_state,
    list_sessions,
    load_session,
    save_propagated_masks,
    save_session_masks,
    save_session_state,
)
from utils.mask_utils import draw_frame_number, draw_points, fit_to_box, overlay_mask
from utils.notify import notify_failure

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None] | None


def clear_processing_log() -> None:
    """Clear the processing log file so the handler's stream position is reset."""
    for handler in logging.getLogger().handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == PROCESSING_LOG_FILE.resolve()
        ):
            handler.acquire()
            try:
                if handler.stream is not None:
                    handler.stream.seek(0)
                    handler.stream.truncate(0)
                else:
                    PROCESSING_LOG_FILE.write_text("")
            finally:
                handler.release()
            return
    PROCESSING_LOG_FILE.write_text("")


# ---------------------------------------------------------------------------
# Lazy-loaded SAM image engines
# ---------------------------------------------------------------------------
_sam2_engine = None
_sam3_engine = None


def _get_sam2():
    global _sam2_engine
    if _sam2_engine is None:
        from engines.sam2_engine import SAM2Engine
        _sam2_engine = SAM2Engine()
    return _sam2_engine


def _get_sam3():
    global _sam3_engine
    if _sam3_engine is None:
        from engines.sam3_engine import SAM3Engine
        _sam3_engine = SAM3Engine()
    return _sam3_engine


def _get_image_engine(model_type: str):
    return _get_sam3() if model_type == "SAM3" else _get_sam2()


def _make_session_id(video_filename: str) -> str:
    date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(video_filename).stem
    stem = unicodedata.normalize("NFKC", stem)
    stem = re.sub(r"[\s\-]+", "_", stem)
    stem = re.sub(r"[^\w]", "", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "video"
    return f"{date_prefix}_{stem}"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def render_frame(state: dict) -> np.ndarray:
    """Render the current frame with mask overlay and click points.

    Returns a preview-sized image (fit within PREVIEW_MAX_W x PREVIEW_MAX_H).
    Click points in state are in original-resolution space; they are scaled
    to preview space only for drawing.
    """
    frame = load_frame(state["frames_dir"], state["current_frame_idx"])
    if state["current_mask"] is not None:
        frame = overlay_mask(frame, state["current_mask"])

    orig_h, orig_w = frame.shape[:2]
    prev_h, prev_w = fit_to_box(orig_h, orig_w, PREVIEW_MAX_H, PREVIEW_MAX_W)
    if (prev_h, prev_w) != (orig_h, orig_w):
        frame = cv2.resize(frame, (prev_w, prev_h), interpolation=cv2.INTER_LINEAR)

    if state["click_points"]:
        scale_x = prev_w / orig_w
        scale_y = prev_h / orig_h
        preview_pts = [
            [int(round(px * scale_x)), int(round(py * scale_y))]
            for px, py in state["click_points"]
        ]
        frame = draw_points(frame, preview_pts, state["click_labels"])
    return frame


def keyframe_display(state: dict) -> str:
    """Build a string showing saved keyframes."""
    if not state["keyframes"]:
        return "尚未保存任何关键帧。"
    indices = sorted(state["keyframes"].keys())
    parts = [f"#{i}" for i in indices]
    return "关键帧: " + "  ".join(parts)


def keyframe_gallery(state: dict) -> list[tuple[np.ndarray, str]]:
    """Build gallery items: list of (overlay_image, caption) for each keyframe."""
    if not state["keyframes"] or state["frames_dir"] is None:
        return []
    items = []
    for idx in sorted(state["keyframes"].keys()):
        frame = load_frame(state["frames_dir"], idx)
        preview = overlay_mask(frame, state["keyframes"][idx])
        items.append((preview, f"第 {idx} 帧"))
    return items


def _sync_frame_clicks(state: dict) -> None:
    idx = state["current_frame_idx"]
    if state["click_points"]:
        state["frame_clicks"][idx] = {
            "points": [p[:] for p in state["click_points"]],
            "labels": state["click_labels"][:],
        }
        if state["current_mask"] is not None:
            state["frame_masks"][idx] = state["current_mask"].copy()
    else:
        state["frame_clicks"].pop(idx, None)
        state["frame_masks"].pop(idx, None)


# ---------------------------------------------------------------------------
# Session restore / refresh
# ---------------------------------------------------------------------------
def restore_session(session_id: str | None, state: dict) -> dict[str, Any]:
    """Restore a session from disk. Returns dict for UI updates."""
    out = _no_restore_out(state)
    if not session_id:
        out["warning"] = "请先选择一个 Session。"
        return out

    loaded = load_session(session_id)
    if loaded is None:
        out["warning"] = f"无法加载 Session: {session_id}"
        return out

    frames_dir = loaded["frames_dir"]
    if not Path(frames_dir).exists():
        out["warning"] = "帧数据目录不存在，无法恢复。"
        return out

    frame = render_frame(loaded)
    gallery = keyframe_gallery(loaded)
    kf_info = keyframe_display(loaded)
    session_dir = MATTING_SESSIONS_DIR / session_id
    preview_path = session_dir / "propagation_preview.mp4"
    prop_preview = str(preview_path) if preview_path.exists() else None
    source_video = str(loaded["source_video_path"]) if loaded["source_video_path"] else None
    alpha_path = session_dir / "alpha.mp4"
    fgr_path = session_dir / "foreground.mp4"
    alpha_video = str(alpha_path) if alpha_path.exists() else None
    fgr_video = str(fgr_path) if fgr_path.exists() else None
    is_ma = loaded.get("matting_engine", "MatAnyone") == "MatAnyone"

    log.info("Restored session from disk: %s", session_id)
    return {
        "session_state": loaded,
        "frame_image": frame,
        "frame_label": f"第 {loaded['current_frame_idx']} 帧 / 共 {loaded['num_frames'] - 1} 帧",
        "keyframe_info": kf_info,
        "keyframe_gallery": gallery,
        "video_path": source_video,
        "slider_visible": True,
        "slider_max": loaded["num_frames"] - 1,
        "slider_value": loaded["current_frame_idx"],
        "model_type": loaded["model_type"],
        "text_prompt_visible": loaded["model_type"] == "SAM3",
        "propagation_preview_path": prop_preview,
        "matting_engine": loaded.get("matting_engine", "MatAnyone"),
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


def _no_restore_out(state: dict) -> dict[str, Any]:
    return {
        "session_state": state,
        "frame_image": None,
        "frame_label": None,
        "keyframe_info": keyframe_display(state),
        "keyframe_gallery": keyframe_gallery(state),
        "video_path": None,
        "slider_visible": False,
        "slider_max": 0,
        "slider_value": 0,
        "model_type": state.get("model_type", "SAM2"),
        "text_prompt_visible": False,
        "propagation_preview_path": None,
        "matting_engine": state.get("matting_engine", "MatAnyone"),
        "erode": state.get("erode", DEFAULT_ERODE),
        "dilate": state.get("dilate", DEFAULT_DILATE),
        "batch_size": state.get("batch_size", VIDEOMAMA_BATCH_SIZE),
        "overlap": state.get("overlap", VIDEOMAMA_OVERLAP),
        "seed": state.get("seed", VIDEOMAMA_SEED),
        "erode_dilate_visible": True,
        "vm_params_visible": False,
        "alpha_path": None,
        "fgr_path": None,
    }


def refresh_sessions() -> dict[str, Any]:
    """Refresh session dropdown choices."""
    return {"session_choices": list_sessions()}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def upload_video(video_path: str | None, state: dict) -> dict[str, Any]:
    """Extract frames from uploaded video and initialize session state."""
    if video_path is None:
        return {
            "session_state": state,
            "frame_image": None,
            "frame_label": None,
            "keyframe_info": keyframe_display(state),
            "keyframe_gallery": keyframe_gallery(state),
            "session_choices": list_sessions(),
            "session_value": None,
        }

    session_id = _make_session_id(Path(video_path).name)
    frames_dir = MATTING_SESSIONS_DIR / session_id / "frames"
    num_frames, fps = extract_frames(Path(video_path), frames_dir)

    state = empty_state()
    state["session_id"] = session_id
    state["frames_dir"] = frames_dir
    state["num_frames"] = num_frames
    state["fps"] = fps

    session_dir = MATTING_SESSIONS_DIR / session_id
    source_path = Path(video_path)
    original_filename = source_path.name
    dest_path = session_dir / f"source{source_path.suffix}"
    shutil.copy2(source_path, dest_path)
    state["source_video_path"] = dest_path
    state["original_filename"] = original_filename
    state["video_file_size"] = dest_path.stat().st_size
    state["video_format"] = source_path.suffix.lstrip(".")
    state["video_duration"] = num_frames / fps if fps else 0.0

    frame = load_frame(frames_dir, 0)
    state["video_height"], state["video_width"] = frame.shape[:2]

    meta = {
        "session_id": session_id,
        "original_filename": original_filename,
        "num_frames": num_frames,
        "fps": fps,
        "video_width": state["video_width"],
        "video_height": state["video_height"],
        "video_duration": state["video_duration"],
        "video_file_size": state["video_file_size"],
        "video_format": state["video_format"],
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
    )
    save_session_state(state)

    first_frame = render_frame(state)
    return {
        "session_state": state,
        "frame_image": first_frame,
        "frame_label": f"第 0 帧 / 共 {num_frames - 1} 帧",
        "keyframe_info": keyframe_display(state),
        "keyframe_gallery": keyframe_gallery(state),
        "slider_visible": True,
        "slider_max": num_frames - 1,
        "slider_value": 0,
        "session_choices": list_sessions(),
        "session_value": session_id,
    }


# ---------------------------------------------------------------------------
# Frame navigation and click
# ---------------------------------------------------------------------------
def slider_change(frame_idx: int, state: dict) -> dict[str, Any]:
    """Navigate to a different frame."""
    if state["frames_dir"] is None:
        return {"session_state": state, "frame_image": None, "frame_label": None}

    frame_idx = int(frame_idx)
    old_idx = state["current_frame_idx"]

    if old_idx == frame_idx:
        return {
            "session_state": state,
            "frame_image": render_frame(state),
            "frame_label": f"第 {frame_idx} 帧 / 共 {state['num_frames'] - 1} 帧",
        }

    _sync_frame_clicks(state)
    state["current_frame_idx"] = frame_idx

    saved = state["frame_clicks"].get(frame_idx)
    if frame_idx in state["keyframes"]:
        state["current_mask"] = state["keyframes"][frame_idx]
        if saved:
            state["click_points"] = [p[:] for p in saved["points"]]
            state["click_labels"] = saved["labels"][:]
        else:
            state["click_points"] = []
            state["click_labels"] = []
    elif saved:
        state["click_points"] = [p[:] for p in saved["points"]]
        state["click_labels"] = saved["labels"][:]
        state["current_mask"] = state["frame_masks"].get(frame_idx)
    else:
        state["click_points"] = []
        state["click_labels"] = []
        state["current_mask"] = None

    save_session_state(state)
    return {
        "session_state": state,
        "frame_image": render_frame(state),
        "frame_label": f"第 {frame_idx} 帧 / 共 {state['num_frames'] - 1} 帧",
    }


def frame_click(x: float, y: float, point_mode: str, model_type: str, state: dict) -> dict[str, Any]:
    """Add a click point, run SAM prediction, update mask.

    x, y are in preview image space. They are mapped to original resolution
    before storing and SAM inference.
    """
    if state["frames_dir"] is None:
        return {"session_state": state, "frame_image": None}

    orig_w, orig_h = state["video_width"], state["video_height"]
    if orig_w == 0 or orig_h == 0:
        return {"session_state": state, "frame_image": None}
    prev_h, prev_w = fit_to_box(orig_h, orig_w, PREVIEW_MAX_H, PREVIEW_MAX_W)
    ix = int(round(x * orig_w / prev_w))
    iy = int(round(y * orig_h / prev_h))

    label = 1 if point_mode == "Positive" else 0
    state["click_points"].append([ix, iy])
    state["click_labels"].append(label)

    engine = _get_image_engine(model_type)
    if state["_sam2_image_idx"] != state["current_frame_idx"]:
        raw_frame = load_frame(state["frames_dir"], state["current_frame_idx"])
        engine.set_image(raw_frame)
        state["_sam2_image_idx"] = state["current_frame_idx"]
    mask = engine.predict(state["click_points"], state["click_labels"])
    state["current_mask"] = mask

    _sync_frame_clicks(state)
    save_session_state(state)
    return {"session_state": state, "frame_image": render_frame(state)}


def undo_click(model_type: str, state: dict) -> dict[str, Any]:
    """Remove last click and re-predict."""
    if state["frames_dir"] is None or not state["click_points"]:
        return {
            "session_state": state,
            "frame_image": render_frame(state) if state["frames_dir"] else None,
        }

    state["click_points"].pop()
    state["click_labels"].pop()

    if state["click_points"]:
        engine = _get_image_engine(model_type)
        if state["_sam2_image_idx"] != state["current_frame_idx"]:
            raw_frame = load_frame(state["frames_dir"], state["current_frame_idx"])
            engine.set_image(raw_frame)
            state["_sam2_image_idx"] = state["current_frame_idx"]
        mask = engine.predict(state["click_points"], state["click_labels"])
        state["current_mask"] = mask
    else:
        state["current_mask"] = None

    _sync_frame_clicks(state)
    save_session_state(state)
    return {"session_state": state, "frame_image": render_frame(state)}


def clear_clicks(state: dict) -> dict[str, Any]:
    """Clear all click points and mask for the current frame."""
    if state["frames_dir"] is None:
        return {"session_state": state, "frame_image": None}

    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = None
    _sync_frame_clicks(state)
    save_session_state(state)
    return {"session_state": state, "frame_image": render_frame(state)}


def save_keyframe(state: dict) -> dict[str, Any]:
    """Save current mask as a keyframe."""
    if state["current_mask"] is None:
        return {
            "session_state": state,
            "keyframe_info": keyframe_display(state),
            "keyframe_gallery": keyframe_gallery(state),
            "warning": "没有可保存的遮罩，请先在帧上点击标注。",
        }

    idx = state["current_frame_idx"]
    state["keyframes"][idx] = state["current_mask"].copy()
    log.info("Saved keyframe at frame %d", idx)
    save_session_state(state)
    save_session_masks(state)
    return {
        "session_state": state,
        "keyframe_info": keyframe_display(state),
        "keyframe_gallery": keyframe_gallery(state),
    }


def delete_keyframe(state: dict) -> dict[str, Any]:
    """Delete the keyframe at the current frame index."""
    idx = state["current_frame_idx"]
    if idx in state["keyframes"]:
        del state["keyframes"][idx]
        state["current_mask"] = None
        log.info("Deleted keyframe at frame %d", idx)

    save_session_state(state)
    save_session_masks(state)
    frame = render_frame(state) if state["frames_dir"] else None
    return {
        "session_state": state,
        "frame_image": frame,
        "keyframe_info": keyframe_display(state),
        "keyframe_gallery": keyframe_gallery(state),
    }


# ---------------------------------------------------------------------------
# Propagation (long-running; progress_callback required for UI)
# ---------------------------------------------------------------------------
def run_propagation(
    model_type: str,
    state: dict,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    """Run bidirectional propagation and generate preview video."""
    if not state["keyframes"]:
        return {
            "session_state": state,
            "propagation_preview_path": None,
            "warning": "请至少保存一个关键帧后再运行传播。",
        }

    clear_processing_log()
    log.info("========== 开始 %s VP 传播 ==========", model_type)
    log.info("关键帧: %s (%d 个)", sorted(state["keyframes"].keys()), len(state["keyframes"]))

    def progress_cb(frac: float):
        if progress_callback:
            progress_callback(frac, "传播中...")

    engine = get_video_engine(model_type)
    propagated = engine.propagate(
        frames_dir=state["frames_dir"],
        keyframe_masks=state["keyframes"],
        progress_callback=progress_cb,
    )
    state["propagated_masks"] = propagated

    log.info("生成传播预览视频...")
    session_dir = MATTING_SESSIONS_DIR / state["session_id"]
    preview_path = session_dir / "propagation_preview.mp4"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(preview_path), fps=state["fps"], quality=7)
    try:
        for idx in range(state["num_frames"]):
            frame = load_frame(state["frames_dir"], idx)
            mask = propagated.get(idx)
            if mask is not None:
                frame = overlay_mask(frame, mask)
            frame = draw_frame_number(frame, idx)
            writer.append_data(frame)
    finally:
        writer.close()

    log.info("传播预览已生成: %s", preview_path.name)
    save_propagated_masks(state)
    return {
        "session_state": state,
        "propagation_preview_path": str(preview_path),
    }


# ---------------------------------------------------------------------------
# Model change and text prompt
# ---------------------------------------------------------------------------
def model_change(model_type: str, state: dict) -> dict[str, Any]:
    """Handle model selector change between SAM2 and SAM3."""
    state["model_type"] = model_type
    state["_sam2_image_idx"] = -1
    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = None
    state["frame_clicks"] = {}
    state["frame_masks"] = {}
    save_session_state(state)

    frame = render_frame(state) if state["frames_dir"] else None
    return {
        "session_state": state,
        "text_prompt_visible": model_type == "SAM3",
        "frame_image": frame,
    }


def text_prompt(prompt: str, state: dict) -> dict[str, Any]:
    """Run SAM3 text-prompt detection on the current frame."""
    if state["frames_dir"] is None:
        return {"session_state": state, "frame_image": None}
    if not prompt or not prompt.strip():
        return {
            "session_state": state,
            "frame_image": render_frame(state),
            "warning": "请输入文本提示词。",
        }

    engine = _get_sam3()
    if state["_sam2_image_idx"] != state["current_frame_idx"]:
        raw_frame = load_frame(state["frames_dir"], state["current_frame_idx"])
        engine.set_image(raw_frame)
        state["_sam2_image_idx"] = state["current_frame_idx"]

    mask = engine.predict_text(prompt.strip())
    state["current_mask"] = mask
    state["click_points"] = []
    state["click_labels"] = []
    save_session_state(state)
    return {"session_state": state, "frame_image": render_frame(state)}


# ---------------------------------------------------------------------------
# Matting (long-running)
# ---------------------------------------------------------------------------
def start_matting(
    matting_engine: str,
    erode: int,
    dilate: int,
    batch_size: int,
    overlap: int,
    seed: int,
    model_type: str,
    state: dict,
    progress_callback: ProgressCallback = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run matting with the selected engine. Auto-runs propagation if needed.

    Args:
        matting_engine: Engine name ("MatAnyone" or "VideoMaMa").
        erode: Erosion kernel size.
        dilate: Dilation kernel size.
        batch_size: VideoMaMa batch size.
        overlap: VideoMaMa overlap frames.
        seed: VideoMaMa random seed.
        model_type: Propagation model type.
        state: Session state dict.
        progress_callback: Optional progress reporter.
        cancel_event: Optional threading.Event; if set, matting is cancelled.

    Returns:
        Dict with session_state, alpha_path, fgr_path.
    """
    if not state["keyframes"]:
        return {
            "session_state": state,
            "alpha_path": None,
            "fgr_path": None,
            "warning": "请至少保存一个关键帧后再开始抠像。",
        }

    state["matting_engine"] = matting_engine
    state["erode"] = int(erode)
    state["dilate"] = int(dilate)
    state["batch_size"] = int(batch_size)
    state["overlap"] = int(overlap)
    state["seed"] = int(seed)
    state["model_type"] = model_type
    state["task_status"] = "processing"
    state["error_msg"] = ""
    save_session_state(state)
    clear_processing_log()

    def progress_cb(frac: float, desc: str = ""):
        if progress_callback:
            progress_callback(frac, desc)

    try:
        alpha_path, fgr_path, _ = run_matting_task(
            state, progress_cb, cancel_event=cancel_event,
        )
        state["task_status"] = "done"
        save_session_state(state)
        return {
            "session_state": state,
            "alpha_path": str(alpha_path),
            "fgr_path": str(fgr_path),
        }
    except Exception as exc:
        state["task_status"] = "error"
        state["error_msg"] = str(exc)
        save_session_state(state)
        notify_failure(state.get("session_id", "unknown"), exc)
        raise
