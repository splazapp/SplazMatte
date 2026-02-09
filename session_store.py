"""Session persistence: load, save, and list sessions.

Pure data layer with no Gradio dependency. Used by both the Gradio UI
callbacks and the CLI entry point.
"""

import json
import logging
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_DILATE,
    DEFAULT_ERODE,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    WORKSPACE_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State factory
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


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_session_state(state: dict) -> None:
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
        "processing_time": state.get("processing_time", 0.0),
        "start_time": state.get("start_time", ""),
        "end_time": state.get("end_time", ""),
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


def save_session_masks(state: dict) -> None:
    """Write keyframe masks as individual .npy files.

    Also cleans up .npy files for keyframes that have been deleted.
    """
    session_id = state.get("session_id")
    if not session_id:
        return
    masks_dir = WORKSPACE_DIR / "sessions" / session_id / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    keyframes: dict = state.get("keyframes", {})

    for idx, mask in keyframes.items():
        np.save(masks_dir / f"keyframe_{idx:06d}.npy", mask)

    for npy_path in masks_dir.glob("keyframe_*.npy"):
        try:
            frame_idx = int(npy_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if frame_idx not in keyframes:
            npy_path.unlink()


def save_propagated_masks(state: dict) -> None:
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


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_session(session_id: str) -> dict | None:
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


def read_session_status(session_id: str) -> dict:
    """Read lightweight status info for a session without full load.

    Returns a dict with task_status, original_filename, matting_engine,
    and num_frames. Returns defaults if the session is unreadable.
    """
    session_dir = WORKSPACE_DIR / "sessions" / session_id
    result = {
        "task_status": "",
        "original_filename": "",
        "matting_engine": "",
        "num_frames": 0,
    }
    for name, path in [
        ("meta", session_dir / "meta.json"),
        ("state", session_dir / "state.json"),
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            result.update({
                k: data[k] for k in result if k in data
            })
        except (json.JSONDecodeError, OSError):
            pass
    return result
