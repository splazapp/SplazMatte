"""Gradio callback functions for SplazMatte app."""

import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import imageio
import numpy as np
import torch

from config import DEFAULT_WARMUP, PROCESSING_LOG_FILE, WORKSPACE_DIR
from pipeline.video_io import (
    encode_video,
    extract_frames,
    load_all_frames_as_tensor,
    load_frame,
)
from utils.mask_utils import draw_frame_number, draw_points, overlay_mask
from utils.notify import notify_failure, upload_and_notify

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
# Lazy-loaded global engines (initialized on first use)
# ---------------------------------------------------------------------------
_sam2_engine = None
_sam2_video_engine = None
_sam3_engine = None
_sam3_video_engine = None
_matanyone_engine = None
_videomama_engine = None


def _get_sam2():
    global _sam2_engine
    if _sam2_engine is None:
        from engines.sam2_engine import SAM2Engine
        _sam2_engine = SAM2Engine()
    return _sam2_engine


def _get_sam2_video():
    global _sam2_video_engine
    if _sam2_video_engine is None:
        from engines.sam2_video_engine import SAM2VideoEngine
        _sam2_video_engine = SAM2VideoEngine()
    return _sam2_video_engine


def _get_sam3():
    global _sam3_engine
    if _sam3_engine is None:
        from engines.sam3_engine import SAM3Engine
        _sam3_engine = SAM3Engine()
    return _sam3_engine


def _get_sam3_video():
    global _sam3_video_engine
    if _sam3_video_engine is None:
        from engines.sam3_video_engine import SAM3VideoEngine
        _sam3_video_engine = SAM3VideoEngine()
    return _sam3_video_engine


def _get_matanyone():
    global _matanyone_engine
    if _matanyone_engine is None:
        _unload_videomama()
        from engines.matanyone_engine import MatAnyoneEngine
        _matanyone_engine = MatAnyoneEngine()
    return _matanyone_engine


def _get_videomama():
    global _videomama_engine
    if _videomama_engine is None:
        _unload_matanyone()
        from engines.videomama_engine import VideoMaMaEngine
        _videomama_engine = VideoMaMaEngine()
    return _videomama_engine


def _unload_matanyone():
    """Free MatAnyone from VRAM."""
    global _matanyone_engine
    if _matanyone_engine is not None:
        del _matanyone_engine
        _matanyone_engine = None
        torch.cuda.empty_cache()


def _unload_videomama():
    """Free VideoMaMa from VRAM."""
    global _videomama_engine
    if _videomama_engine is not None:
        del _videomama_engine
        _videomama_engine = None
        torch.cuda.empty_cache()


def _get_image_engine(model_type: str):
    """Return the image segmentation engine for the given model type."""
    return _get_sam3() if model_type == "SAM3" else _get_sam2()


def _get_video_engine(model_type: str):
    """Return the video propagation engine for the given model type."""
    return _get_sam3_video() if model_type == "SAM3" else _get_sam2_video()


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def empty_state() -> dict:
    """Create a fresh session state dict."""
    return {
        "session_id": "",
        "frames_dir": None,
        "num_frames": 0,
        "fps": 0.0,
        "current_frame_idx": 0,
        "click_points": [],
        "click_labels": [],
        "current_mask": None,
        "keyframes": {},
        "propagated_masks": {},
        "_sam2_image_idx": -1,
        "model_type": "SAM2",
        "source_video_path": None,
        "original_filename": "",
        "video_file_size": 0,
        "video_format": "",
        "video_duration": 0.0,
        "video_width": 0,
        "video_height": 0,
    }


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


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_upload(video_path: str, state: dict):
    """Extract frames from uploaded video and initialize session state."""
    if video_path is None:
        return (state, None, gr.update(), gr.update(),
                keyframe_display(state), keyframe_gallery(state))

    session_id = str(uuid.uuid4())[:8]
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

    slider_update = gr.update(
        minimum=0, maximum=num_frames - 1, value=0, visible=True, interactive=True,
    )
    label_update = gr.update(value=f"第 0 帧 / 共 {num_frames - 1} 帧")
    return (state, frame, slider_update, label_update,
            keyframe_display(state), keyframe_gallery(state))


def on_slider_change(frame_idx: int, state: dict):
    """Navigate to a different frame; clear click state."""
    if state["frames_dir"] is None:
        return None, state, gr.update()

    frame_idx = int(frame_idx)
    state["current_frame_idx"] = frame_idx
    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = state["keyframes"].get(frame_idx)

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

    return render_frame(state), state


def on_clear_clicks(state: dict):
    """Clear all click points and mask for the current frame."""
    if state["frames_dir"] is None:
        return None, state

    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = None
    return render_frame(state), state


def on_save_keyframe(state: dict):
    """Save current mask as a keyframe annotation."""
    if state["current_mask"] is None:
        gr.Warning("没有可保存的遮罩，请先在帧上点击标注。")
        return state, keyframe_display(state), keyframe_gallery(state)

    idx = state["current_frame_idx"]
    state["keyframes"][idx] = state["current_mask"].copy()
    log.info("Saved keyframe at frame %d", idx)
    return state, keyframe_display(state), keyframe_gallery(state)


def on_delete_keyframe(state: dict):
    """Delete the keyframe at the current frame index."""
    idx = state["current_frame_idx"]
    if idx in state["keyframes"]:
        del state["keyframes"][idx]
        state["current_mask"] = None
        log.info("Deleted keyframe at frame %d", idx)

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

    engine = _get_video_engine(model_type)

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
    return str(preview_path), state


def on_model_change(model_type: str, state: dict):
    """Handle model selector change between SAM2 and SAM3."""
    state["model_type"] = model_type
    # Force re-embedding on next click
    state["_sam2_image_idx"] = -1
    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = None

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
    return render_frame(state), state


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

    # Clear log file so the UI panel starts fresh
    _clear_processing_log()

    # Auto-run propagation if not done yet
    if not state.get("propagated_masks"):
        log.info("传播尚未执行，自动运行 %s 传播...", model_type)
        engine = _get_video_engine(model_type)
        propagated = engine.propagate(
            frames_dir=state["frames_dir"],
            keyframe_masks=state["keyframes"],
            progress_callback=lambda f: progress(f, desc="自动传播中..."),
        )
        state["propagated_masks"] = propagated
        log.info("自动传播完成，共 %d 帧遮罩", len(propagated))

    log.info("========== 开始抠像 (引擎: %s) ==========", matting_engine)
    log.info(
        "Session: %s | 关键帧: %s | erode=%d, dilate=%d",
        state["session_id"],
        sorted(state["keyframes"].keys()),
        erode,
        dilate,
    )

    start_ts = time.time()
    start_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        if matting_engine == "VideoMaMa":
            alphas, foregrounds = _run_videomama(
                state, int(batch_size), int(overlap), int(seed), progress,
            )
        else:
            alphas, foregrounds = _run_matanyone(state, erode, dilate)

        session_dir = WORKSPACE_DIR / "sessions" / state["session_id"]
        alpha_path = session_dir / "alpha.mp4"
        fgr_path = session_dir / "foreground.mp4"

        log.info("编码视频...")
        alpha_rgb = np.repeat(alphas, 3, axis=3)
        encode_video(alpha_rgb, alpha_path, state["fps"])
        encode_video(foregrounds, fgr_path, state["fps"])

        end_ts = time.time()
        end_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        processing_time = end_ts - start_ts

        log.info("上传至 R2...")
        try:
            upload_and_notify(
                state, erode, dilate, session_dir,
                processing_time, start_dt, end_dt,
            )
        except Exception:
            log.exception("Post-matting upload/notify failed")

        log.info(
            "========== 处理完成 (%.1fs) ==========", processing_time,
        )
        return str(alpha_path), str(fgr_path), state

    except Exception as exc:
        notify_failure(state.get("session_id", "unknown"), exc)
        raise


def _run_videomama(
    state: dict,
    batch_size: int,
    overlap: int,
    seed: int,
    progress: gr.Progress,
) -> tuple[np.ndarray, np.ndarray]:
    """Run VideoMaMa matting pipeline.

    Args:
        state: Session state dict.
        batch_size: Frames per inference batch.
        overlap: Overlap frames between batches for blending.
        seed: Random seed for reproducibility.
        progress: Gradio progress tracker.

    Returns:
        Tuple of (alphas, foregrounds) arrays.
    """
    masks = state.get("propagated_masks", {})
    if len(masks) < state["num_frames"]:
        raise gr.Error(
            "VideoMaMa 需要每帧遮罩，请先运行 SAM2 传播。"
            f"（当前 {len(masks)}/{state['num_frames']} 帧）"
        )

    log.info(
        "开始 VideoMaMa 推理 (%d 帧, batch=%d, overlap=%d, seed=%d)...",
        state["num_frames"], batch_size, overlap, seed,
    )
    engine = _get_videomama()
    return engine.process(
        frames_dir=state["frames_dir"],
        masks=masks,
        batch_size=batch_size,
        overlap=overlap,
        seed=seed,
        progress_callback=lambda f: progress(f, desc="VideoMaMa 推理中..."),
    )


def _run_matanyone(
    state: dict, erode: int, dilate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MatAnyone matting pipeline.

    Returns:
        Tuple of (alphas, foregrounds) arrays.
    """
    log.info("加载帧数据 (%d 帧)...", state["num_frames"])
    frames_tensor = load_all_frames_as_tensor(state["frames_dir"])

    log.info("开始 MatAnyone 推理...")
    engine = _get_matanyone()
    return engine.process(
        frames=frames_tensor,
        keyframe_masks=state["keyframes"],
        erode=erode,
        dilate=dilate,
        warmup=DEFAULT_WARMUP,
    )
