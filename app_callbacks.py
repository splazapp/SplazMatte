"""Gradio callback functions for SplazMatte app."""

import json
import logging
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import gradio as gr
import imageio
import numpy as np

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    PROCESSING_LOG_FILE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
)
from matting_runner import get_video_engine, run_matting_task
from pipeline.video_io import extract_frames, load_frame
from session_store import (
    empty_state,
    list_sessions,
    load_session,
    save_propagated_masks,
    save_session_masks,
    save_session_state,
)
from utils.mask_utils import draw_frame_number, draw_points, overlay_mask
from utils.notify import notify_failure

log = logging.getLogger(__name__)


def _clear_processing_log():
    """Clear the processing log, properly resetting the FileHandler stream.

    Using ``PROCESSING_LOG_FILE.write_text("")`` truncates the file via a
    separate fd, but the ``logging.FileHandler`` retains its old seek
    position.  Subsequent writes start at that offset, filling the gap
    with null bytes that render as blank in the browser.  Instead, we
    seek + truncate on the handler's own stream so the position is reset.
    """
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
    # Fallback when no matching handler is found
    PROCESSING_LOG_FILE.write_text("")


# ---------------------------------------------------------------------------
# Lazy-loaded SAM image engines (UI annotation only)
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
    """Return the image segmentation engine for the given model type."""
    return _get_sam3() if model_type == "SAM3" else _get_sam2()


def _make_session_id(video_filename: str) -> str:
    """Generate a human-readable session ID from the video filename.

    Format: ``{yyyyMMddHHMM}_{sanitized_name}_{seq}`` where *seq*
    auto-increments until no matching directory exists.

    Args:
        video_filename: Original filename of the uploaded video.

    Returns:
        A unique, filesystem/URL-safe session ID.
    """
    date_prefix = datetime.now().strftime("%Y%m%d%H%M")

    stem = Path(video_filename).stem
    stem = unicodedata.normalize("NFKC", stem)
    stem = re.sub(r"[\s\-]+", "_", stem)
    stem = re.sub(r"[^\w]", "", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "video"

    base = f"{date_prefix}_{stem}"
    sessions_dir = WORKSPACE_DIR / "sessions"
    seq = 1
    while (sessions_dir / f"{base}_{seq}").exists():
        seq += 1
    return f"{base}_{seq}"


# ---------------------------------------------------------------------------
# UI display helpers
# ---------------------------------------------------------------------------
def render_frame(state: dict) -> np.ndarray:
    """Render the current frame with mask overlay and click points."""
    frame = load_frame(state["frames_dir"], state["current_frame_idx"])
    if state["current_mask"] is not None:
        frame = overlay_mask(frame, state["current_mask"])
    if state["click_points"]:
        frame = draw_points(frame, state["click_points"], state["click_labels"])
    return frame


def keyframe_display(state: dict) -> str:
    """Build a markdown string showing saved keyframes."""
    if not state["keyframes"]:
        return "尚未保存任何关键帧。"
    indices = sorted(state["keyframes"].keys())
    parts = [f"**#{i}**" for i in indices]
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
    """Sync current frame's click state to frame_clicks/frame_masks storage."""
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
# Session restore / refresh callbacks
# ---------------------------------------------------------------------------
def on_restore_session(session_id: str | None, state: dict):
    """Restore a session from disk into the editing UI.

    Returns:
        Tuple of 18 values: session_state, frame_display, frame_slider,
        frame_label, keyframe_info, kf_gallery, video_input,
        model_selector, text_prompt_row, propagation_preview,
        matting_engine_selector, erode_slider, dilate_slider,
        vm_batch_slider, vm_overlap_slider, vm_seed_input,
        alpha_output, fgr_output.
    """
    no_update = (
        state, gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(),
    )

    if not session_id:
        gr.Warning("请先选择一个 Session。")
        return no_update

    loaded = load_session(session_id)
    if loaded is None:
        gr.Warning(f"无法加载 Session: {session_id}")
        return no_update

    frames_dir = loaded["frames_dir"]
    if not Path(frames_dir).exists():
        gr.Warning("帧数据目录不存在，无法恢复。")
        return no_update

    # Navigate to saved frame position
    frame = render_frame(loaded)
    gallery = keyframe_gallery(loaded)
    kf_info = keyframe_display(loaded)

    # Check for existing propagation preview
    session_dir = WORKSPACE_DIR / "sessions" / session_id
    preview_path = session_dir / "propagation_preview.mp4"
    prop_preview = str(preview_path) if preview_path.exists() else None

    # Source video for the video input widget
    source_video = (
        str(loaded["source_video_path"])
        if loaded["source_video_path"] else None
    )

    # Check for existing matting output videos
    alpha_path = session_dir / "alpha.mp4"
    fgr_path = session_dir / "foreground.mp4"
    alpha_video = str(alpha_path) if alpha_path.exists() else None
    fgr_video = str(fgr_path) if fgr_path.exists() else None

    # Matting engine parameter visibility
    is_ma = loaded.get("matting_engine", "MatAnyone") == "MatAnyone"

    log.info("Restored session from disk: %s", session_id)

    return (
        loaded,
        frame,
        gr.update(
            minimum=0,
            maximum=loaded["num_frames"] - 1,
            value=loaded["current_frame_idx"],
            visible=True,
            interactive=True,
        ),
        f"第 {loaded['current_frame_idx']} 帧 / 共 {loaded['num_frames'] - 1} 帧",
        kf_info,
        gallery,
        source_video,
        gr.update(value=loaded["model_type"]),
        gr.update(visible=(loaded["model_type"] == "SAM3")),
        prop_preview,
        gr.update(value=loaded.get("matting_engine", "MatAnyone")),
        gr.update(value=loaded.get("erode", DEFAULT_ERODE), visible=is_ma),
        gr.update(value=loaded.get("dilate", DEFAULT_DILATE), visible=is_ma),
        gr.update(value=loaded.get("batch_size", VIDEOMAMA_BATCH_SIZE), visible=not is_ma),
        gr.update(value=loaded.get("overlap", VIDEOMAMA_OVERLAP), visible=not is_ma),
        gr.update(value=loaded.get("seed", VIDEOMAMA_SEED), visible=not is_ma),
        alpha_video,
        fgr_video,
    )


def on_refresh_sessions():
    """Refresh the session dropdown choices."""
    return gr.update(choices=list_sessions())


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_upload(video_path: str, state: dict):
    """Extract frames from uploaded video and initialize session state."""
    if video_path is None:
        return (state, None, gr.update(), gr.update(),
                keyframe_display(state), keyframe_gallery(state),
                gr.update())

    session_id = _make_session_id(Path(video_path).name)
    frames_dir = WORKSPACE_DIR / "sessions" / session_id / "frames"

    num_frames, fps = extract_frames(Path(video_path), frames_dir)

    state = empty_state()
    state["session_id"] = session_id
    state["frames_dir"] = frames_dir
    state["num_frames"] = num_frames
    state["fps"] = fps

    # Copy source video into session dir (Gradio temp files may be cleaned)
    session_dir = WORKSPACE_DIR / "sessions" / session_id
    source_path = Path(video_path)
    original_filename = source_path.name
    dest_path = session_dir / f"source{source_path.suffix}"
    shutil.copy2(source_path, dest_path)
    state["source_video_path"] = dest_path
    state["original_filename"] = original_filename
    state["video_file_size"] = dest_path.stat().st_size
    state["video_format"] = source_path.suffix.lstrip(".")
    state["video_duration"] = num_frames / fps if fps else 0.0

    # Read resolution from the first frame
    frame = load_frame(frames_dir, 0)
    state["video_height"], state["video_width"] = frame.shape[:2]

    # Write session metadata
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
    meta_path = session_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    save_session_state(state)

    slider_update = gr.update(
        minimum=0, maximum=num_frames - 1, value=0, visible=True, interactive=True,
    )
    label_update = gr.update(value=f"第 0 帧 / 共 {num_frames - 1} 帧")
    return (state, frame, slider_update, label_update,
            keyframe_display(state), keyframe_gallery(state),
            gr.update(choices=list_sessions(), value=session_id))


def on_slider_change(frame_idx: int, state: dict):
    """Navigate to a different frame; save old frame clicks, restore new."""
    if state["frames_dir"] is None:
        return None, state, gr.update()

    frame_idx = int(frame_idx)
    old_idx = state["current_frame_idx"]

    # Same frame — no navigation needed (chained event from restore/upload)
    if old_idx == frame_idx:
        frame = render_frame(state)
        label = f"第 {frame_idx} 帧 / 共 {state['num_frames'] - 1} 帧"
        return frame, state, gr.update(value=label)

    # Save old frame's click state before switching
    _sync_frame_clicks(state)

    state["current_frame_idx"] = frame_idx

    # Restore new frame: keyframe mask takes priority, but also restore clicks
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

    frame = render_frame(state)
    label = f"第 {frame_idx} 帧 / 共 {state['num_frames'] - 1} 帧"
    return frame, state, gr.update(value=label)


def on_frame_click(evt: gr.SelectData, point_mode: str, model_type: str, state: dict):
    """Accumulate a click point, run SAM prediction, and show mask overlay."""
    if state["frames_dir"] is None:
        return None, state

    x, y = evt.index[0], evt.index[1]
    label = 1 if point_mode == "Positive" else 0
    state["click_points"].append([x, y])
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
    return render_frame(state), state


def on_undo_click(model_type: str, state: dict):
    """Remove last click point and re-predict."""
    if state["frames_dir"] is None or not state["click_points"]:
        return render_frame(state) if state["frames_dir"] else None, state

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
    return render_frame(state), state


def on_clear_clicks(state: dict):
    """Clear all click points and mask for the current frame."""
    if state["frames_dir"] is None:
        return None, state

    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = None

    _sync_frame_clicks(state)
    save_session_state(state)
    return render_frame(state), state


def on_save_keyframe(state: dict):
    """Save current mask as a keyframe annotation."""
    if state["current_mask"] is None:
        gr.Warning("没有可保存的遮罩，请先在帧上点击标注。")
        return state, keyframe_display(state), keyframe_gallery(state)

    idx = state["current_frame_idx"]
    state["keyframes"][idx] = state["current_mask"].copy()
    log.info("Saved keyframe at frame %d", idx)

    save_session_state(state)
    save_session_masks(state)
    return state, keyframe_display(state), keyframe_gallery(state)


def on_delete_keyframe(state: dict):
    """Delete the keyframe at the current frame index."""
    idx = state["current_frame_idx"]
    if idx in state["keyframes"]:
        del state["keyframes"][idx]
        state["current_mask"] = None
        log.info("Deleted keyframe at frame %d", idx)

    save_session_state(state)
    save_session_masks(state)

    frame = render_frame(state) if state["frames_dir"] else None
    return state, frame, keyframe_display(state), keyframe_gallery(state)


def on_run_propagation(
    model_type: str,
    state: dict,
    progress=gr.Progress(track_tqdm=True),
):
    """Run bidirectional propagation and generate preview video."""
    if not state["keyframes"]:
        gr.Warning("请至少保存一个关键帧后再运行传播。")
        return None, state

    _clear_processing_log()
    log.info("========== 开始 %s VP 传播 ==========", model_type)
    log.info(
        "关键帧: %s (%d 个)",
        sorted(state["keyframes"].keys()),
        len(state["keyframes"]),
    )

    engine = get_video_engine(model_type)

    def progress_cb(frac: float):
        progress(frac, desc="传播中...")

    propagated = engine.propagate(
        frames_dir=state["frames_dir"],
        keyframe_masks=state["keyframes"],
        progress_callback=progress_cb,
    )
    state["propagated_masks"] = propagated

    # Generate preview video: overlay mask + frame number on each frame
    log.info("生成传播预览视频...")
    session_dir = WORKSPACE_DIR / "sessions" / state["session_id"]
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
    return str(preview_path), state


def on_model_change(model_type: str, state: dict):
    """Handle model selector change between SAM2 and SAM3."""
    state["model_type"] = model_type
    # Force re-embedding on next click
    state["_sam2_image_idx"] = -1
    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = None
    state["frame_clicks"] = {}
    state["frame_masks"] = {}

    save_session_state(state)

    frame = render_frame(state) if state["frames_dir"] else None
    return state, gr.update(visible=(model_type == "SAM3")), frame


def on_text_prompt(prompt: str, state: dict):
    """Run SAM3 text-prompt detection on the current frame."""
    if state["frames_dir"] is None:
        return None, state
    if not prompt or not prompt.strip():
        gr.Warning("请输入文本提示词。")
        return render_frame(state), state

    engine = _get_sam3()
    if state["_sam2_image_idx"] != state["current_frame_idx"]:
        raw_frame = load_frame(state["frames_dir"], state["current_frame_idx"])
        engine.set_image(raw_frame)
        state["_sam2_image_idx"] = state["current_frame_idx"]

    mask = engine.predict_text(prompt.strip())
    state["current_mask"] = mask
    # Clear click points — text and point modes don't mix
    state["click_points"] = []
    state["click_labels"] = []

    save_session_state(state)
    return render_frame(state), state


def _gradio_progress_adapter(progress):
    """Wrap a gr.Progress into the (float, str) -> None callback signature."""
    def callback(frac: float, desc: str = ""):
        progress(frac, desc=desc)
    return callback


def on_start_matting(
    matting_engine: str,
    erode: int,
    dilate: int,
    batch_size: int,
    overlap: int,
    seed: int,
    model_type: str,
    state: dict,
    progress=gr.Progress(track_tqdm=True),
):
    """Run matting with the selected engine.

    Auto-runs propagation first if it hasn't been executed yet.

    Args:
        matting_engine: "MatAnyone" or "VideoMaMa".
        erode: Erosion kernel size (MatAnyone only).
        dilate: Dilation kernel size (MatAnyone only).
        batch_size: Frames per batch (VideoMaMa only).
        overlap: Overlap frames between batches (VideoMaMa only).
        seed: Random seed (VideoMaMa only).
        model_type: "SAM2" or "SAM3" (used for auto-propagation).
        state: Session state dict.
        progress: Gradio progress tracker.
    """
    if not state["keyframes"]:
        gr.Warning("请至少保存一个关键帧后再开始抠像。")
        return None, None, state

    # Save matting parameters to state
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

    # Clear log file so the UI panel starts fresh
    _clear_processing_log()

    try:
        alpha_path, fgr_path, _ = run_matting_task(
            state, _gradio_progress_adapter(progress),
        )
        state["task_status"] = "done"
        save_session_state(state)
        return str(alpha_path), str(fgr_path), state

    except Exception as exc:
        state["task_status"] = "error"
        state["error_msg"] = str(exc)
        save_session_state(state)
        notify_failure(state.get("session_id", "unknown"), exc)
        raise
