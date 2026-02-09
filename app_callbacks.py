"""Gradio callback functions for SplazMatte app."""

import json
import logging
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import imageio
import numpy as np
import torch

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    DEFAULT_WARMUP,
    PROCESSING_LOG_FILE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
)
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
        _unload_sam_video()
        from engines.matanyone_engine import MatAnyoneEngine
        _matanyone_engine = MatAnyoneEngine()
    return _matanyone_engine


def _get_videomama():
    global _videomama_engine
    if _videomama_engine is None:
        _unload_matanyone()
        _unload_sam_video()
        from engines.videomama_engine import VideoMaMaEngine
        _videomama_engine = VideoMaMaEngine()
    return _videomama_engine


def _empty_device_cache():
    """Release cached memory on the active device (CUDA or MPS)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _unload_matanyone():
    """Free MatAnyone from VRAM."""
    global _matanyone_engine
    if _matanyone_engine is not None:
        del _matanyone_engine
        _matanyone_engine = None
        _empty_device_cache()


def _unload_videomama():
    """Free VideoMaMa from VRAM."""
    global _videomama_engine
    if _videomama_engine is not None:
        del _videomama_engine
        _videomama_engine = None
        _empty_device_cache()


def _unload_sam_video():
    """Free SAM2/SAM3 video engines to reclaim memory."""
    global _sam2_video_engine, _sam3_video_engine
    changed = False
    if _sam2_video_engine is not None:
        del _sam2_video_engine
        _sam2_video_engine = None
        changed = True
    if _sam3_video_engine is not None:
        del _sam3_video_engine
        _sam3_video_engine = None
        changed = True
    if changed:
        _empty_device_cache()


def _get_image_engine(model_type: str):
    """Return the image segmentation engine for the given model type."""
    return _get_sam3() if model_type == "SAM3" else _get_sam2()


def _get_video_engine(model_type: str):
    """Return the video propagation engine for the given model type."""
    return _get_sam3_video() if model_type == "SAM3" else _get_sam2_video()


def _make_session_id(video_filename: str) -> str:
    """Generate a human-readable session ID from the video filename.

    Format: ``{yyyyMMdd}_{sanitized_name}_{seq}`` where *seq* auto-increments
    until no matching directory exists under ``workspace/sessions/``.

    Args:
        video_filename: Original filename of the uploaded video
            (e.g. "My Video (1).mp4").

    Returns:
        A unique, filesystem/URL-safe session ID such as
        ``20260209_my_video_1``.
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
        "frame_clicks": {},
        "frame_masks": {},
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
        "matting_engine": "MatAnyone",
        "erode": DEFAULT_ERODE,
        "dilate": DEFAULT_DILATE,
        "batch_size": VIDEOMAMA_BATCH_SIZE,
        "overlap": VIDEOMAMA_OVERLAP,
        "seed": VIDEOMAMA_SEED,
        "task_status": "",
        "error_msg": "",
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
# Session persistence helpers
# ---------------------------------------------------------------------------
def _save_session_state(state: dict) -> None:
    """Write state.json and masks/current_mask.npy to disk.

    Saves only annotation-related dynamic fields (click points, keyframe
    indices, model type, etc.). Video metadata lives in meta.json.
    """
    session_id = state.get("session_id")
    if not session_id:
        return
    session_dir = WORKSPACE_DIR / "sessions" / session_id
    if not session_dir.exists():
        return

    # Serialize frame_clicks with string keys for JSON
    frame_clicks: dict = state.get("frame_clicks", {})
    serialized_fc = {str(k): v for k, v in frame_clicks.items()}

    data = {
        "current_frame_idx": state.get("current_frame_idx", 0),
        "click_points": state.get("click_points", []),
        "click_labels": state.get("click_labels", []),
        "model_type": state.get("model_type", "SAM2"),
        "keyframe_indices": sorted(state.get("keyframes", {}).keys()),
        "has_propagation": bool(state.get("propagated_masks")),
        "frame_clicks": serialized_fc,
        "matting_engine": state.get("matting_engine", "MatAnyone"),
        "erode": state.get("erode", DEFAULT_ERODE),
        "dilate": state.get("dilate", DEFAULT_DILATE),
        "batch_size": state.get("batch_size", VIDEOMAMA_BATCH_SIZE),
        "overlap": state.get("overlap", VIDEOMAMA_OVERLAP),
        "seed": state.get("seed", VIDEOMAMA_SEED),
        "task_status": state.get("task_status", ""),
        "error_msg": state.get("error_msg", ""),
    }
    (session_dir / "state.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
    )

    masks_dir = session_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    mask = state.get("current_mask")
    current_mask_path = masks_dir / "current_mask.npy"
    if mask is not None:
        np.save(current_mask_path, mask)
    elif current_mask_path.exists():
        current_mask_path.unlink()

    # Save frame_masks as click_mask_N.npy; clean up stale files
    frame_masks: dict = state.get("frame_masks", {})
    for idx, fmask in frame_masks.items():
        np.save(masks_dir / f"click_mask_{idx:06d}.npy", fmask)
    for npy_path in masks_dir.glob("click_mask_*.npy"):
        try:
            fidx = int(npy_path.stem.split("_")[-1])
        except (IndexError, ValueError):
            continue
        if fidx not in frame_masks:
            npy_path.unlink()


def _save_session_masks(state: dict) -> None:
    """Write keyframe masks as individual .npy files.

    Also cleans up .npy files for keyframes that have been deleted.
    """
    session_id = state.get("session_id")
    if not session_id:
        return
    masks_dir = WORKSPACE_DIR / "sessions" / session_id / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    keyframes: dict = state.get("keyframes", {})

    # Save each keyframe mask
    for idx, mask in keyframes.items():
        np.save(masks_dir / f"keyframe_{idx:06d}.npy", mask)

    # Clean up .npy files for deleted keyframes
    for npy_path in masks_dir.glob("keyframe_*.npy"):
        try:
            frame_idx = int(npy_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if frame_idx not in keyframes:
            npy_path.unlink()


def _save_propagated_masks(state: dict) -> None:
    """Write propagated masks as a compressed .npz file."""
    session_id = state.get("session_id")
    if not session_id:
        return
    propagated: dict = state.get("propagated_masks", {})
    if not propagated:
        return
    masks_dir = WORKSPACE_DIR / "sessions" / session_id / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        masks_dir / "propagated.npz",
        **{str(k): v for k, v in propagated.items()},
    )


def _load_session(session_id: str) -> dict | None:
    """Restore a full state dict from disk for the given session.

    Reads meta.json, state.json, keyframe masks, current_mask, and
    propagated masks. Returns None if the session directory is invalid.
    """
    sessions_root = (WORKSPACE_DIR / "sessions").resolve()
    session_dir = (WORKSPACE_DIR / "sessions" / session_id).resolve()
    if not session_dir.is_relative_to(sessions_root):
        return None

    meta_path = session_dir / "meta.json"
    state_path = session_dir / "state.json"

    if not meta_path.exists() or not state_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        saved = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    state = empty_state()
    state["session_id"] = session_id
    state["frames_dir"] = session_dir / "frames"
    state["num_frames"] = meta.get("num_frames", 0)
    state["fps"] = meta.get("fps", 0.0)
    state["original_filename"] = meta.get("original_filename", "")
    state["video_file_size"] = meta.get("video_file_size", 0)
    state["video_format"] = meta.get("video_format", "")
    state["video_duration"] = meta.get("video_duration", 0.0)
    state["video_width"] = meta.get("video_width", 0)
    state["video_height"] = meta.get("video_height", 0)

    # Find source video file
    for candidate in session_dir.glob("source.*"):
        state["source_video_path"] = candidate
        break

    # Restore annotation state
    state["current_frame_idx"] = saved.get("current_frame_idx", 0)
    state["model_type"] = saved.get("model_type", "SAM2")
    state["_sam2_image_idx"] = -1

    # Restore matting parameters and task status
    state["matting_engine"] = saved.get("matting_engine", "MatAnyone")
    state["erode"] = saved.get("erode", DEFAULT_ERODE)
    state["dilate"] = saved.get("dilate", DEFAULT_DILATE)
    state["batch_size"] = saved.get("batch_size", VIDEOMAMA_BATCH_SIZE)
    state["overlap"] = saved.get("overlap", VIDEOMAMA_OVERLAP)
    state["seed"] = saved.get("seed", VIDEOMAMA_SEED)
    state["task_status"] = saved.get("task_status", "")
    state["error_msg"] = saved.get("error_msg", "")

    # Restore per-frame click data (keys stored as strings in JSON)
    saved_fc = saved.get("frame_clicks", {})
    state["frame_clicks"] = {int(k): v for k, v in saved_fc.items()}

    # Load mask data (directory may not exist for sessions without keyframes)
    masks_dir = session_dir / "masks"
    if masks_dir.exists():
        for npy_path in sorted(masks_dir.glob("keyframe_*.npy")):
            try:
                frame_idx = int(npy_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            state["keyframes"][frame_idx] = np.load(npy_path)

        # Restore per-frame click masks
        for npy_path in sorted(masks_dir.glob("click_mask_*.npy")):
            try:
                fidx = int(npy_path.stem.split("_")[-1])
            except (IndexError, ValueError):
                continue
            state["frame_masks"][fidx] = np.load(npy_path)

        current_mask_path = masks_dir / "current_mask.npy"
        if current_mask_path.exists():
            state["current_mask"] = np.load(current_mask_path)

        propagated_path = masks_dir / "propagated.npz"
        if propagated_path.exists():
            with np.load(propagated_path) as data:
                state["propagated_masks"] = {
                    int(k): data[k] for k in data.files
                }

    # Populate click_points/click_labels from current frame's saved clicks
    cur_idx = state["current_frame_idx"]
    cur_fc = state["frame_clicks"].get(cur_idx)
    if cur_idx in state["keyframes"]:
        state["current_mask"] = state["keyframes"][cur_idx]
        if cur_fc:
            state["click_points"] = [p[:] for p in cur_fc["points"]]
            state["click_labels"] = cur_fc["labels"][:]
        else:
            state["click_points"] = []
            state["click_labels"] = []
    elif cur_fc:
        state["click_points"] = [p[:] for p in cur_fc["points"]]
        state["click_labels"] = cur_fc["labels"][:]
        if state["current_mask"] is None:
            state["current_mask"] = state["frame_masks"].get(cur_idx)
    else:
        state["click_points"] = saved.get("click_points", [])
        state["click_labels"] = saved.get("click_labels", [])

    return state


def list_sessions() -> list[tuple[str, str]]:
    """List restorable sessions from disk.

    Scans workspace/sessions/ for directories containing state.json.

    Returns:
        List of (label, session_id) tuples sorted by modification time
        (newest first). Label format: ``{session_id} ({original_filename})``.
    """
    sessions_dir = WORKSPACE_DIR / "sessions"
    if not sessions_dir.exists():
        return []

    items: list[tuple[float, str, str]] = []
    for state_file in sessions_dir.glob("*/state.json"):
        session_id = state_file.parent.name
        meta_path = state_file.parent / "meta.json"
        filename = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                filename = meta.get("original_filename", "")
            except (json.JSONDecodeError, OSError):
                pass
        label = f"{session_id} ({filename})" if filename else session_id
        mtime = state_file.stat().st_mtime
        items.append((mtime, label, session_id))

    items.sort(key=lambda x: x[0], reverse=True)
    return [(label, sid) for _, label, sid in items]


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

    loaded = _load_session(session_id)
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
    """Refresh the session dropdown choices.

    Returns:
        gr.update with refreshed choices list.
    """
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

    _save_session_state(state)

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

    _save_session_state(state)

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
    _save_session_state(state)
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
    _save_session_state(state)
    return render_frame(state), state


def on_clear_clicks(state: dict):
    """Clear all click points and mask for the current frame."""
    if state["frames_dir"] is None:
        return None, state

    state["click_points"] = []
    state["click_labels"] = []
    state["current_mask"] = None

    _sync_frame_clicks(state)
    _save_session_state(state)
    return render_frame(state), state


def on_save_keyframe(state: dict):
    """Save current mask as a keyframe annotation."""
    if state["current_mask"] is None:
        gr.Warning("没有可保存的遮罩，请先在帧上点击标注。")
        return state, keyframe_display(state), keyframe_gallery(state)

    idx = state["current_frame_idx"]
    state["keyframes"][idx] = state["current_mask"].copy()
    log.info("Saved keyframe at frame %d", idx)

    _save_session_state(state)
    _save_session_masks(state)
    return state, keyframe_display(state), keyframe_gallery(state)


def on_delete_keyframe(state: dict):
    """Delete the keyframe at the current frame index."""
    idx = state["current_frame_idx"]
    if idx in state["keyframes"]:
        del state["keyframes"][idx]
        state["current_mask"] = None
        log.info("Deleted keyframe at frame %d", idx)

    _save_session_state(state)
    _save_session_masks(state)

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
    _save_propagated_masks(state)
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

    _save_session_state(state)

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

    _save_session_state(state)
    return render_frame(state), state


def _run_matting_task(
    state: dict,
    progress=None,
    progress_prefix: str = "",
) -> tuple[Path, Path, float]:
    """Shared matting execution logic.

    Handles auto-propagation, matting inference, video encoding, and
    upload/notify. Reads matting parameters from ``state`` directly.

    Args:
        state: Session state dict (must contain matting params).
        progress: Gradio progress tracker (or compatible callable).
        progress_prefix: Prefix for progress descriptions (e.g. "[1/3]").

    Returns:
        (alpha_path, fgr_path, processing_time)
    """
    matting_engine = state.get("matting_engine", "MatAnyone")
    erode = int(state.get("erode", DEFAULT_ERODE))
    dilate = int(state.get("dilate", DEFAULT_DILATE))
    batch_size = int(state.get("batch_size", VIDEOMAMA_BATCH_SIZE))
    overlap = int(state.get("overlap", VIDEOMAMA_OVERLAP))
    seed = int(state.get("seed", VIDEOMAMA_SEED))
    model_type = state.get("model_type", "SAM2")

    def _progress(frac: float, desc: str = ""):
        if progress is not None:
            full = f"{progress_prefix} {desc}".strip() if progress_prefix else desc
            progress(frac, desc=full)

    # Auto-run propagation if not done yet
    if not state.get("propagated_masks"):
        log.info("传播尚未执行，自动运行 %s 传播...", model_type)
        engine = _get_video_engine(model_type)
        propagated = engine.propagate(
            frames_dir=state["frames_dir"],
            keyframe_masks=state["keyframes"],
            progress_callback=lambda f: _progress(f, "自动传播中..."),
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

    if matting_engine == "VideoMaMa":
        alphas, foregrounds = _run_videomama(
            state, batch_size, overlap, seed, progress,
        )
    else:
        alphas, foregrounds = _run_matanyone(state, erode, dilate, progress)

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

    log.info("========== 处理完成 (%.1fs) ==========", processing_time)
    return alpha_path, fgr_path, processing_time


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
    _save_session_state(state)

    # Clear log file so the UI panel starts fresh
    _clear_processing_log()

    try:
        alpha_path, fgr_path, _ = _run_matting_task(state, progress)
        state["task_status"] = "done"
        _save_session_state(state)
        return str(alpha_path), str(fgr_path), state

    except Exception as exc:
        state["task_status"] = "error"
        state["error_msg"] = str(exc)
        _save_session_state(state)
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
    state: dict,
    erode: int,
    dilate: int,
    progress: gr.Progress | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MatAnyone matting pipeline.

    Args:
        state: Session state dict.
        erode: Erosion kernel size.
        dilate: Dilation kernel size.
        progress: Gradio progress tracker.

    Returns:
        Tuple of (alphas, foregrounds) arrays.
    """
    log.info("加载帧数据 (%d 帧)...", state["num_frames"])
    frames_tensor = load_all_frames_as_tensor(state["frames_dir"])

    log.info("开始 MatAnyone 推理...")
    engine = _get_matanyone()
    cb = (lambda f: progress(f, desc="MatAnyone 推理中...")) if progress else None
    return engine.process(
        frames=frames_tensor,
        keyframe_masks=state["keyframes"],
        erode=erode,
        dilate=dilate,
        warmup=DEFAULT_WARMUP,
        progress_callback=cb,
    )
