"""Tracking session persistence: save, load, and list tracking sessions.

Stores tracking sessions under workspace/tracking_sessions/{sid}/.
"""

import json
import logging
from pathlib import Path

import numpy as np

from config import TRACKING_SESSIONS_DIR

log = logging.getLogger(__name__)


def save_tracking_session(state: dict) -> None:
    """Persist a tracking session to disk.

    Saves meta.json, state.json, keyframes/*.json, and numpy arrays.

    Args:
        state: Full tracking state dict.
    """
    sid = state.get("session_id")
    if not sid:
        return

    session_dir = TRACKING_SESSIONS_DIR / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    # meta.json — video metadata
    frames_dir = state.get("frames_dir")
    meta = {
        "session_id": sid,
        "original_filename": state.get("original_filename", ""),
        "fps": state.get("fps", 24.0),
        "original_size": list(state.get("original_size", (0, 0))),
        "preview_size": list(state.get("preview_size", (0, 0))),
        "num_frames": state.get("num_frames", 0),
        "video_path": state.get("video_path", ""),
        "frames_dir": str(frames_dir) if frames_dir is not None else "",
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
    )

    # state.json — annotation and task state
    keyframe_indices = sorted(state.get("keyframes", {}).keys())
    total_points = sum(
        len(kf["points"]) for kf in state.get("keyframes", {}).values()
    )
    data = {
        "current_frame": state.get("current_frame", 0),
        "keyframe_indices": keyframe_indices,
        "total_points": total_points,
        "task_status": state.get("task_status", ""),
        "error_msg": state.get("error_msg", ""),
        "backward_tracking": state.get("backward_tracking", False),
        "grid_size": state.get("grid_size", 15),
        "use_grid": state.get("use_grid", False),
        "processing_time": state.get("processing_time", 0.0),
        "start_time": state.get("start_time", ""),
        "end_time": state.get("end_time", ""),
        "ae_export_path": state.get("ae_export_path", ""),
    }
    (session_dir / "state.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
    )

    # Save keyframes as individual JSON files
    kf_dir = session_dir / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    keyframes: dict = state.get("keyframes", {})
    for idx, kf in keyframes.items():
        kf_data = {
            "points": [list(p) for p in kf["points"]],
            "colors": [list(c) for c in kf["colors"]],
        }
        (kf_dir / f"{idx:06d}.json").write_text(
            json.dumps(kf_data, ensure_ascii=False),
        )
    # Clean up stale keyframe files
    for f in kf_dir.glob("*.json"):
        try:
            fidx = int(f.stem)
        except ValueError:
            continue
        if fidx not in keyframes:
            f.unlink()

    # Note: preview_frames.npy and input_frames.npy are saved once during
    # preprocess_video(). They are NOT stored in the state dict to keep it
    # lightweight for interactive operations.

    log.info("Tracking session saved: %s (%d keyframes)", sid, len(keyframes))


def save_tracking_results(state: dict) -> None:
    """Save raw tracking results (tracks + visibility) to disk.

    Args:
        state: Tracking state with raw_tracks and raw_visibility.
    """
    sid = state.get("session_id")
    if not sid:
        return
    session_dir = TRACKING_SESSIONS_DIR / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    if state.get("raw_tracks") is not None:
        np.save(session_dir / "raw_tracks.npy", state["raw_tracks"])
    if state.get("raw_visibility") is not None:
        np.save(session_dir / "raw_visibility.npy", state["raw_visibility"])


def load_tracking_session(sid: str) -> dict | None:
    """Restore a full tracking state dict from disk.

    Args:
        sid: Session ID.

    Returns:
        Full tracking state dict, or None if not loadable.
    """
    from tracking.logic import empty_tracking_state

    session_dir = TRACKING_SESSIONS_DIR / sid
    meta_path = session_dir / "meta.json"
    state_path = session_dir / "state.json"

    if not meta_path.exists() or not state_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        saved = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    state = empty_tracking_state()
    state["session_id"] = sid
    state["original_filename"] = meta.get("original_filename", "")
    state["fps"] = meta.get("fps", 24.0)
    state["original_size"] = tuple(meta.get("original_size", (0, 0)))
    state["preview_size"] = tuple(meta.get("preview_size", (0, 0)))
    state["video_path"] = meta.get("video_path", "")
    fd = meta.get("frames_dir", "")
    state["frames_dir"] = Path(fd) if fd and Path(fd).exists() else None
    state["current_frame"] = saved.get("current_frame", 0)
    state["task_status"] = saved.get("task_status", "")
    state["error_msg"] = saved.get("error_msg", "")
    state["backward_tracking"] = saved.get("backward_tracking", False)
    state["grid_size"] = saved.get("grid_size", 15)
    state["use_grid"] = saved.get("use_grid", False)
    state["processing_time"] = saved.get("processing_time", 0.0)
    state["start_time"] = saved.get("start_time", "")
    state["end_time"] = saved.get("end_time", "")
    state["ae_export_path"] = saved.get("ae_export_path", "")

    # Note: preview_frames.npy and input_frames.npy are NOT loaded into
    # state. They are loaded on demand by _load_all_preview_frames() and
    # _load_input_frames() in cotracker_logic.py only when needed.

    # Init per-frame point arrays
    num_frames = meta.get("num_frames", 0)
    state["num_frames"] = num_frames
    if num_frames > 0:
        state["query_points"] = [[] for _ in range(num_frames)]
        state["query_colors"] = [[] for _ in range(num_frames)]

    # Load keyframes
    kf_dir = session_dir / "keyframes"
    if kf_dir.exists():
        for f in sorted(kf_dir.glob("*.json")):
            try:
                idx = int(f.stem)
            except ValueError:
                continue
            try:
                kf_data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            points = [tuple(p) for p in kf_data.get("points", [])]
            colors = [tuple(c) for c in kf_data.get("colors", [])]
            state["keyframes"][idx] = {"points": points, "colors": colors}
            # Also populate query_points for the frame
            if num_frames > 0 and idx < num_frames:
                state["query_points"][idx] = list(points)
                state["query_colors"][idx] = list(colors)

    state["query_count"] = sum(
        len(kf["points"]) for kf in state["keyframes"].values()
    )

    # Load raw results if available
    tracks_path = session_dir / "raw_tracks.npy"
    vis_path = session_dir / "raw_visibility.npy"
    if tracks_path.exists():
        state["raw_tracks"] = np.load(tracks_path)
    if vis_path.exists():
        state["raw_visibility"] = np.load(vis_path)

    log.info("Loaded tracking session: %s (%d keyframes)", sid, len(state["keyframes"]))
    return state


def list_tracking_sessions() -> list[tuple[str, str]]:
    """List restorable tracking sessions from disk.

    Scans workspace/tracking_sessions/ for directories containing state.json.

    Returns:
        List of (label, session_id) tuples sorted by modification time
        (newest first). Label format: ``{session_id} ({original_filename})``.
    """
    if not TRACKING_SESSIONS_DIR.exists():
        return []

    items: list[tuple[float, str, str]] = []
    for state_file in TRACKING_SESSIONS_DIR.glob("*/state.json"):
        sid = state_file.parent.name
        meta_path = state_file.parent / "meta.json"
        filename = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                filename = meta.get("original_filename", "")
            except (json.JSONDecodeError, OSError):
                pass
        label = f"{sid} ({filename})" if filename else sid
        mtime = state_file.stat().st_mtime
        items.append((mtime, label, sid))

    items.sort(key=lambda x: x[0], reverse=True)
    return [(label, sid) for _, label, sid in items]


def read_tracking_session_info(sid: str) -> dict | None:
    """Read lightweight session info without loading numpy data.

    Args:
        sid: Session ID.

    Returns:
        Dict with merged meta + state fields, or None.
    """
    session_dir = TRACKING_SESSIONS_DIR / sid
    meta_path = session_dir / "meta.json"
    state_path = session_dir / "state.json"
    if not meta_path.exists() or not state_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        saved = json.loads(state_path.read_text())
        return {**meta, **saved}
    except (json.JSONDecodeError, OSError):
        return None
