"""UI-agnostic business logic for CoTracker point tracking.

All functions take state/params and return structured dicts. No UI imports.
"""

from __future__ import annotations

import colorsys
import logging
import random
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import imageio
import numpy as np

from config import (
    COTRACKER_INPUT_RESO,
    PREVIEW_MAX_H,
    PREVIEW_MAX_W,
    TRACKING_SESSIONS_DIR,
    WORKSPACE_DIR,
)
from pipeline.video_io import extract_frames, load_frame, preload_all_frames, unload_frames
from utils.mask_utils import draw_points, fit_to_box, overlay_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None] | None


def empty_tracking_state() -> dict[str, Any]:
    """Return an empty tracking state dict."""
    return {
        "session_id": "",
        "original_filename": "",
        "video_path": None,
        "frames_dir": None,
        "num_frames": 0,
        "fps": 24.0,
        "original_size": (0, 0),
        "preview_size": (0, 0),
        "current_frame": 0,
        "query_points": [],
        "query_colors": [],
        "query_count": 0,
        "result_video_path": None,
        # Saved keyframes: {frame_idx: {"points": [...], "colors": [...]}}
        "keyframes": {},
        # SAM object selection state
        "sam_mask": None,
        "sam_points": [],
        "sam_labels": [],
        "sam_image_idx": -1,
        # Raw tracking results for AE export
        "raw_tracks": None,
        "raw_visibility": None,
        "ae_export_path": "",
        "ae_summary_txt_path": "",
        "ae_summary_jsx_path": "",
        # Task queue state
        "task_status": "",
        "error_msg": "",
        "backward_tracking": False,
        "grid_size": 15,
        "use_grid": False,
    }


def _get_colors(num_colors: int) -> list[tuple[int, int, int]]:
    """Generate visually distinct colors for tracking points."""
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / max(num_colors, 1)):
        hue = i / 360.0
        lightness = (50 + random.random() * 10) / 100.0
        saturation = (90 + random.random() * 10) / 100.0
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
    random.shuffle(colors)
    return colors


def _get_preview_frame(state: dict, idx: int) -> np.ndarray:
    """Load a single frame from disk and resize to preview dimensions."""
    frame = load_frame(Path(state["frames_dir"]), idx)
    prev_h, prev_w = state["preview_size"]
    return cv2.resize(frame, (prev_w, prev_h), interpolation=cv2.INTER_LINEAR)


def _load_all_preview_frames(state: dict) -> np.ndarray:
    """Load all preview frames from npy cache (for run_tracking only)."""
    npy_path = Path(state["frames_dir"]).parent / "preview_frames.npy"
    if npy_path.exists():
        return np.load(npy_path)
    prev_h, prev_w = state["preview_size"]
    frames = [
        cv2.resize(load_frame(Path(state["frames_dir"]), i), (prev_w, prev_h))
        for i in range(state["num_frames"])
    ]
    return np.stack(frames)


def _load_input_frames(state: dict) -> np.ndarray:
    """Load CoTracker input frames from npy cache (for run_tracking only)."""
    npy_path = Path(state["frames_dir"]).parent / "input_frames.npy"
    if npy_path.exists():
        return np.load(npy_path)
    input_H, input_W = COTRACKER_INPUT_RESO
    frames = [
        cv2.resize(load_frame(Path(state["frames_dir"]), i), (input_W, input_H))
        for i in range(state["num_frames"])
    ]
    return np.stack(frames)


def preprocess_video(video_path: str, state: dict) -> dict[str, Any]:
    """Read and preprocess video for tracking.

    Uses extract_frames + load_frame (same as matting). Extracts all frames
    to disk, then builds preview/input arrays from loaded frames.

    Args:
        video_path: Path to video file.
        state: Current tracking state.

    Returns:
        Dict with session_state, preview_frame, slider_max, etc.
    """
    log.info("Loading video: %s", video_path)

    sid = uuid.uuid4().hex[:12]
    session_dir = TRACKING_SESSIONS_DIR / sid
    frames_dir = session_dir / "frames"

    num_frames, fps = extract_frames(Path(video_path), frames_dir)
    if num_frames == 0:
        return {"notify": ("error", "视频为空"), "session_state": state}

    old_frames_dir = state.get("frames_dir")
    if old_frames_dir is not None:
        unload_frames(Path(old_frames_dir))
    preload_all_frames(frames_dir, num_frames)
    frame0 = load_frame(frames_dir, 0)
    H, W = frame0.shape[:2]

    new_height, new_width = fit_to_box(H, W, PREVIEW_MAX_H, PREVIEW_MAX_W)
    input_H, input_W = COTRACKER_INPUT_RESO

    preview_frames = []
    input_frames = []
    for i in range(num_frames):
        frame = load_frame(frames_dir, i)
        preview_frames.append(
            cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        )
        input_frames.append(
            cv2.resize(frame, (input_W, input_H), interpolation=cv2.INTER_LINEAR)
        )
    preview_arr = np.stack(preview_frames)
    input_arr = np.stack(input_frames)

    # Save npy caches for bulk operations (run_tracking)
    np.save(session_dir / "preview_frames.npy", preview_arr)
    np.save(session_dir / "input_frames.npy", input_arr)

    # Keep first preview frame for immediate display
    first_preview = preview_arr[0].copy()

    new_state = empty_tracking_state()
    new_state.update({
        "session_id": sid,
        "original_filename": Path(video_path).name,
        "video_path": video_path,
        "frames_dir": frames_dir,
        "num_frames": num_frames,
        "fps": fps,
        "original_size": (H, W),
        "preview_size": (new_height, new_width),
        "current_frame": 0,
        "query_points": [[] for _ in range(num_frames)],
        "query_colors": [[] for _ in range(num_frames)],
        "query_count": 0,
    })

    log.info("Video loaded: %d frames, %dx%d @ %.1f fps", num_frames, W, H, fps)

    from tracking.session_store import save_tracking_session
    save_tracking_session(new_state)

    return {
        "session_state": new_state,
        "preview_frame": first_preview,
        "slider_max": num_frames - 1,
        "slider_value": 0,
        "notify": ("positive", f"视频加载成功: {num_frames} 帧"),
    }


def restore_tracking_session(session_id: str | None, state: dict) -> dict[str, Any]:
    """Restore a tracking session from disk. Returns dict for UI updates."""
    from tracking.session_store import load_tracking_session, list_tracking_sessions

    out = {
        "session_state": state,
        "preview_frame": None,
        "frame_label": "第 0 帧 / 共 0 帧",
        "slider_max": 0,
        "slider_value": 0,
        "slider_visible": False,
        "keyframe_info": tracking_keyframe_display(state),
        "keyframe_gallery": [],
        "video_path": None,
        "result_video_path": None,
        "backward_tracking": False,
        "grid_size": 15,
        "use_grid": False,
        "point_count": 0,
    }
    if not session_id:
        out["warning"] = "请先选择一个 Session。"
        return out

    loaded = load_tracking_session(session_id)
    if loaded is None:
        out["warning"] = f"无法加载 Session: {session_id}"
        return out

    if loaded.get("frames_dir") is None or loaded.get("num_frames", 0) == 0:
        out["warning"] = "帧数据不存在，无法恢复。"
        return out

    current_frame = loaded.get("current_frame", 0)
    num_frames = loaded["num_frames"]
    current_frame = max(0, min(current_frame, num_frames - 1))
    loaded["current_frame"] = current_frame

    new_frames_dir = loaded["frames_dir"]
    old_frames_dir = state.get("frames_dir")
    if old_frames_dir is not None and str(old_frames_dir) != str(new_frames_dir):
        unload_frames(Path(old_frames_dir))
    preload_all_frames(Path(new_frames_dir), num_frames)
    preview = _get_preview_frame(loaded, current_frame)
    kf = loaded.get("keyframes", {}).get(current_frame)
    if kf:
        for pt, color in zip(kf["points"], kf["colors"]):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(preview, (px, py), 4, tuple(color), -1)
    kf_marker = " [已保存]" if current_frame in loaded.get("keyframes", {}) else ""

    log.info("Restored tracking session from disk: %s", session_id)
    return {
        "session_state": loaded,
        "preview_frame": preview,
        "frame_label": f"第 {current_frame} 帧 / 共 {num_frames - 1} 帧{kf_marker}",
        "slider_max": num_frames - 1,
        "slider_value": current_frame,
        "slider_visible": True,
        "keyframe_info": tracking_keyframe_display(loaded),
        "keyframe_gallery": tracking_keyframe_gallery(loaded),
        "video_path": loaded.get("video_path"),
        "result_video_path": loaded.get("result_video_path"),
        "backward_tracking": loaded.get("backward_tracking", False),
        "grid_size": loaded.get("grid_size", 15),
        "use_grid": loaded.get("use_grid", False),
        "point_count": loaded.get("query_count", 0),
    }


def refresh_tracking_sessions() -> dict[str, Any]:
    """Refresh tracking session dropdown choices."""
    from tracking.session_store import list_tracking_sessions

    return {"session_choices": list_tracking_sessions()}


def change_frame(frame_idx: int, state: dict) -> dict[str, Any]:
    """Change the current frame being viewed.

    Args:
        frame_idx: Target frame index.
        state: Current tracking state.

    Returns:
        Dict with session_state and preview_frame.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    num_frames = state["num_frames"]
    frame_idx = max(0, min(frame_idx, num_frames - 1))

    state["current_frame"] = frame_idx

    # 切换帧时清除 SAM 状态，避免旧帧的标注点/mask 残留
    state["sam_points"] = []
    state["sam_labels"] = []
    state["sam_mask"] = None
    state["sam_image_idx"] = -1

    # Restore points from saved keyframe if available
    kf = state.get("keyframes", {}).get(frame_idx)
    if kf is not None:
        state["query_points"][frame_idx] = [tuple(p) for p in kf["points"]]
        state["query_colors"][frame_idx] = [tuple(c) for c in kf["colors"]]

    preview = _get_preview_frame(state, frame_idx)

    for pt, color in zip(
        state["query_points"][frame_idx], state["query_colors"][frame_idx]
    ):
        x, y, _ = pt
        cv2.circle(preview, (int(x), int(y)), 4, color, -1)

    # Indicate if this frame is a saved keyframe
    kf_marker = " [已保存]" if frame_idx in state.get("keyframes", {}) else ""
    return {
        "session_state": state,
        "preview_frame": preview,
        "frame_label": f"第 {frame_idx} 帧 / 共 {num_frames - 1} 帧{kf_marker}",
    }


def add_point(x: float, y: float, state: dict) -> dict[str, Any]:
    """Add a tracking query point at the clicked position.

    Args:
        x: X coordinate in preview image space.
        y: Y coordinate in preview image space.
        state: Current tracking state.

    Returns:
        Dict with session_state and updated preview_frame.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    frame_idx = state["current_frame"]
    color_idx = len(state["query_points"][frame_idx])

    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
    color = cmap(color_idx % 20 / 20)
    color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    state["query_points"][frame_idx].append((x, y, frame_idx))
    state["query_colors"][frame_idx].append(color_rgb)
    state["query_count"] = _effective_point_count(state)

    preview = _get_preview_frame(state, frame_idx)
    for pt, col in zip(
        state["query_points"][frame_idx], state["query_colors"][frame_idx]
    ):
        px, py, _ = pt
        cv2.circle(preview, (int(px), int(py)), 4, col, -1)

    return {
        "session_state": state,
        "preview_frame": preview,
        "query_count": state["query_count"],
    }


def undo_point(state: dict) -> dict[str, Any]:
    """Remove the last added point on the current frame.

    Args:
        state: Current tracking state.

    Returns:
        Dict with session_state and updated preview_frame.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    frame_idx = state["current_frame"]
    if len(state["query_points"][frame_idx]) == 0:
        return {"session_state": state, "preview_frame": _get_preview_frame(state, frame_idx)}

    state["query_points"][frame_idx].pop()
    state["query_colors"][frame_idx].pop()
    state["query_count"] = _effective_point_count(state)

    preview = _get_preview_frame(state, frame_idx)
    for pt, col in zip(
        state["query_points"][frame_idx], state["query_colors"][frame_idx]
    ):
        px, py, _ = pt
        cv2.circle(preview, (int(px), int(py)), 4, col, -1)

    return {
        "session_state": state,
        "preview_frame": preview,
        "query_count": state["query_count"],
    }


def clear_frame_points(state: dict) -> dict[str, Any]:
    """Clear all points on the current frame.

    Args:
        state: Current tracking state.

    Returns:
        Dict with session_state and preview_frame.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    frame_idx = state["current_frame"]
    state["query_points"][frame_idx] = []
    state["query_colors"][frame_idx] = []
    state["query_count"] = _effective_point_count(state)

    preview = _get_preview_frame(state, frame_idx)

    return {
        "session_state": state,
        "preview_frame": preview,
        "query_count": state["query_count"],
    }


def clear_all_points(state: dict) -> dict[str, Any]:
    """Clear all tracking points from all frames.

    Args:
        state: Current tracking state.

    Returns:
        Dict with session_state and preview_frame.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    state["query_points"] = [[] for _ in range(state["num_frames"])]
    state["query_colors"] = [[] for _ in range(state["num_frames"])]
    state["query_count"] = _effective_point_count(state)

    preview = _get_preview_frame(state, state["current_frame"])

    return {
        "session_state": state,
        "preview_frame": preview,
        "query_count": state["query_count"],
    }


def _effective_point_count(state: dict) -> int:
    """Return the number of tracking points that will actually be used when running.

    Mirrors the logic in run_tracking: keyframes take priority over query_points.
    """
    keyframes = state.get("keyframes", {})
    if keyframes:
        return sum(len(kf["points"]) for kf in keyframes.values())
    return sum(len(pts) for pts in state.get("query_points", []))


# ---------------------------------------------------------------------------
# SAM object selection (mirrors app_logic.py frame_click/undo_click/clear)
# Uses original video resolution (same as matting) to avoid MPS bicubic fallback.
# ---------------------------------------------------------------------------
def _get_sam_engine(model_type: str):
    """Lazy-load SAM2 or SAM3 image engine (shared with app_logic)."""
    from matting.logic import _get_image_engine
    return _get_image_engine(model_type)


def _get_original_frame(state: dict, frame_idx: int) -> np.ndarray | None:
    """Get a single frame at original video resolution for SAM.

    Prefers frames_dir (JPEG via load_frame, same as matting). Falls back to
    video_path via imageio.
    Returns RGB uint8 (H, W, 3) or None on failure.
    """
    frames_dir = state.get("frames_dir")
    if frames_dir is not None and Path(frames_dir).exists():
        try:
            return load_frame(Path(frames_dir), frame_idx)
        except Exception as e:
            log.warning("Failed to load frame %d from frames_dir: %s", frame_idx, e)

    video_path = state.get("video_path")
    if video_path and Path(video_path).exists():
        try:
            reader = imageio.get_reader(video_path)
            frame = np.array(reader.get_data(frame_idx))
            reader.close()
            return frame
        except Exception as e:
            log.warning("Failed to load frame %d from video: %s", frame_idx, e)

    return None


def _preview_to_original_coords(
    px: float, py: float, state: dict
) -> tuple[float, float]:
    """Map coordinates from preview space to original frame space."""
    orig_h, orig_w = state.get("original_size", (0, 0))
    prev_h, prev_w = state.get("preview_size", (0, 0))
    if prev_w <= 0 or prev_h <= 0:
        return px, py
    scale_x = orig_w / prev_w
    scale_y = orig_h / prev_h
    return px * scale_x, py * scale_y


def _original_to_preview_coords(
    ox: float, oy: float, state: dict
) -> tuple[float, float]:
    """Map coordinates from original frame space to preview space."""
    orig_h, orig_w = state.get("original_size", (0, 0))
    prev_h, prev_w = state.get("preview_size", (0, 0))
    if orig_w <= 0 or orig_h <= 0:
        return ox, oy
    scale_x = prev_w / orig_w
    scale_y = prev_h / orig_h
    return ox * scale_x, oy * scale_y


def _render_sam_preview(state: dict) -> np.ndarray:
    """Render current frame with SAM mask overlay and click points.

    sam_points are stored in original coords; convert to preview for drawing.
    """
    preview = _get_preview_frame(state, state["current_frame"])
    if state.get("sam_mask") is not None:
        preview = overlay_mask(preview, state["sam_mask"])
    if state.get("sam_points"):
        pts_preview = [
            _original_to_preview_coords(px, py, state)
            for px, py in state["sam_points"]
        ]
        pts_preview = [[int(round(x)), int(round(y))] for x, y in pts_preview]
        preview = draw_points(preview, pts_preview, state["sam_labels"])
    return preview


def sam_click(
    x: float,
    y: float,
    state: dict,
    model_type: str = "SAM2",
    is_positive: bool = True,
) -> dict[str, Any]:
    """Add a SAM click point, run prediction, update mask.

    Uses original video resolution (same as matting) for SAM inference.

    Args:
        x: X coordinate in preview image space.
        y: Y coordinate in preview image space.
        state: Current tracking state.
        model_type: "SAM2" or "SAM3".
        is_positive: True for positive point, False for negative.

    Returns:
        Dict with session_state and preview_frame.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    ix, iy = int(round(x)), int(round(y))
    label = 1 if is_positive else 0
    ox, oy = _preview_to_original_coords(ix, iy, state)
    state["sam_points"].append([int(round(ox)), int(round(oy))])
    state["sam_labels"].append(label)

    frame_idx = state["current_frame"]
    orig_frame = _get_original_frame(state, frame_idx)
    if orig_frame is None:
        orig_frame = _get_preview_frame(state, frame_idx)

    engine = _get_sam_engine(model_type)
    if state.get("sam_image_idx") != frame_idx:
        engine.set_image(orig_frame)
        state["sam_image_idx"] = frame_idx

    mask = engine.predict(state["sam_points"], state["sam_labels"])

    prev_h, prev_w = state["preview_size"]
    if mask.shape[:2] != (prev_h, prev_w):
        mask = cv2.resize(
            mask, (prev_w, prev_h),
            interpolation=cv2.INTER_NEAREST,
        )
    state["sam_mask"] = mask

    return {"session_state": state, "preview_frame": _render_sam_preview(state)}


def sam_undo(state: dict, model_type: str = "SAM2") -> dict[str, Any]:
    """Remove last SAM click point and re-predict.

    Uses original video resolution (same as matting) for SAM inference.

    Args:
        state: Current tracking state.
        model_type: "SAM2" or "SAM3".

    Returns:
        Dict with session_state and preview_frame.
    """
    if state.get("frames_dir") is None or not state.get("sam_points"):
        return {
            "session_state": state,
            "preview_frame": _render_sam_preview(state) if state.get("frames_dir") else None,
        }

    state["sam_points"].pop()
    state["sam_labels"].pop()

    if state["sam_points"]:
        frame_idx = state["current_frame"]
        orig_frame = _get_original_frame(state, frame_idx)
        if orig_frame is None:
            orig_frame = _get_preview_frame(state, frame_idx)

        engine = _get_sam_engine(model_type)
        if state.get("sam_image_idx") != frame_idx:
            engine.set_image(orig_frame)
            state["sam_image_idx"] = frame_idx

        mask = engine.predict(state["sam_points"], state["sam_labels"])

        prev_h, prev_w = state["preview_size"]
        if mask.shape[:2] != (prev_h, prev_w):
            mask = cv2.resize(
                mask, (prev_w, prev_h),
                interpolation=cv2.INTER_NEAREST,
            )
        state["sam_mask"] = mask
    else:
        state["sam_mask"] = None

    return {"session_state": state, "preview_frame": _render_sam_preview(state)}


def sam_clear(state: dict) -> dict[str, Any]:
    """Clear all SAM points and mask.

    Args:
        state: Current tracking state.

    Returns:
        Dict with session_state and preview_frame.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    state["sam_points"] = []
    state["sam_labels"] = []
    state["sam_mask"] = None
    state["sam_image_idx"] = -1

    preview = _get_preview_frame(state, state["current_frame"])
    return {"session_state": state, "preview_frame": preview}


def _sample_contour_uniform(
    contours: list[np.ndarray], n: int
) -> list[tuple[float, float]]:
    """沿轮廓等弧长采样 n 个点。

    Args:
        contours: cv2.findContours 返回的轮廓列表。
        n: 采样点数。

    Returns:
        采样点列表 [(x, y), ...]。
    """
    valid = [c.squeeze(1) for c in contours if len(c) >= 2]
    if not valid:
        # 只有单点轮廓
        for c in contours:
            if len(c) >= 1:
                pt = c.squeeze()
                return [(float(pt[0]), float(pt[1]))]
        return []
    all_pts = np.concatenate(valid)
    if len(all_pts) < 2:
        return [(float(all_pts[0][0]), float(all_pts[0][1]))]

    diffs = np.diff(all_pts, axis=0, append=all_pts[:1])  # 闭合
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-6:
        return [(float(all_pts[0][0]), float(all_pts[0][1]))]

    sample_d = np.linspace(0, total, n, endpoint=False) + total / (2 * n)
    result = []
    for d in sample_d:
        d = d % total
        idx = np.searchsorted(cum, d) - 1
        idx = max(0, min(idx, len(all_pts) - 1))
        t = (d - cum[idx]) / max(seg_len[idx], 1e-6)
        pt = all_pts[idx] + t * diffs[idx]
        result.append((float(pt[0]), float(pt[1])))
    return result


def _farthest_point_sample(
    dist_transform: np.ndarray, n: int, seed_points: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """距离变换加权的最远点采样。

    Args:
        dist_transform: cv2.distanceTransform 结果。
        n: 采样点数。
        seed_points: 已有的种子点（如边缘采样点），用于初始化距离。

    Returns:
        采样点列表 [(x, y), ...]。
    """
    ys, xs = np.where(dist_transform > 1.0)
    if len(ys) == 0:
        return []
    cands = np.stack([xs, ys], axis=1).astype(float)
    dt_vals = dist_transform[ys, xs]

    if seed_points:
        seeds = np.array(seed_points)
        min_dist = np.min(
            np.linalg.norm(cands[:, None] - seeds[None], axis=2), axis=1
        )
    else:
        min_dist = np.full(len(cands), np.inf)

    result = []
    for _ in range(n):
        if len(cands) == 0:
            break
        scores = min_dist * np.sqrt(dt_vals + 1)
        best = np.argmax(scores)
        pt = cands[best]
        result.append((float(pt[0]), float(pt[1])))
        new_d = np.linalg.norm(cands - pt, axis=1)
        min_dist = np.minimum(min_dist, new_d)
    return result


def mask_to_points(mask: np.ndarray, num_points: int = 30) -> list[tuple[float, float]]:
    """从二值 mask 生成追踪点（原始坐标系）。

    使用连通域分析按面积分配点数，每个区域 40% 边缘 + 60% 内部。
    边缘点沿轮廓等弧长采样，内部点使用距离变换加权最远点采样。

    Args:
        mask: 二值或灰度 mask（>127 视为前景）。
        num_points: 期望生成的追踪点数量。

    Returns:
        追踪点列表 [(x, y), ...]，坐标在 mask 坐标系下。
    """
    if num_points <= 0:
        return []

    binary = (mask > 127).astype(np.uint8)
    total_area = int(binary.sum())

    if total_area < 5:
        return []

    # 极小 mask：只放质心
    if total_area < 20:
        ys, xs = np.where(binary)
        return [(float(xs.mean()), float(ys.mean()))]

    # 连通域分析 → 按面积分配点数
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    valid = [
        (i, stats[i, cv2.CC_STAT_AREA])
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 10
    ]
    if not valid:
        return []

    total_valid = sum(a for _, a in valid)
    alloc = {i: max(1, round(num_points * a / total_valid)) for i, a in valid}

    # 修正总数使之等于 num_points
    diff = num_points - sum(alloc.values())
    if diff != 0:
        largest_id = max(alloc, key=alloc.get)
        alloc[largest_id] = max(1, alloc[largest_id] + diff)

    # 对每个连通域：40% 边缘 + 60% 内部
    all_points: list[tuple[float, float]] = []
    for comp_id, n_pts in alloc.items():
        comp_mask = (labels == comp_id).astype(np.uint8)
        n_edge = max(1, round(n_pts * 0.4))
        n_interior = max(0, n_pts - n_edge)

        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        edge_pts = _sample_contour_uniform(contours, n_edge)

        if n_interior > 0:
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            interior_pts = _farthest_point_sample(dist, n_interior, edge_pts)
        else:
            interior_pts = []

        all_points.extend(edge_pts)
        all_points.extend(interior_pts)

    # Clamp to num_points in case alloc correction was insufficient (many small regions)
    return all_points[:num_points]


def generate_points_from_mask(
    state: dict,
    num_points: int = 30,
) -> dict[str, Any]:
    """在 SAM mask 内智能生成追踪点。

    使用连通域分析按面积分配点数，每个区域 40% 边缘 + 60% 内部。
    边缘点沿轮廓等弧长采样，内部点使用距离变换加权最远点采样。

    Args:
        state: Current tracking state (must contain sam_mask).
        num_points: 期望生成的追踪点数量。

    Returns:
        Dict with session_state, preview_frame, query_count, notify.
    """
    if state.get("sam_mask") is None:
        return {
            "session_state": state,
            "notify": ("warning", "请先用 SAM 选择目标区域"),
        }

    mask = state["sam_mask"]

    # 每次生成前清空当前帧已有的追踪点，避免重复点击累积
    frame_idx = state["current_frame"]
    state["query_points"][frame_idx] = []
    state["query_colors"][frame_idx] = []

    all_points = mask_to_points(mask, num_points)

    if not all_points:
        return {
            "session_state": state,
            "notify": ("warning", "Mask 太小，请重新选择目标"),
        }

    if len(all_points) == 1:
        # 极小 mask（质心单点）情况下给出特殊提示
        binary = (mask > 127).astype(np.uint8)
        if int(binary.sum()) < 20:
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
            offset = len(state["query_points"][frame_idx])
            color = cmap(offset % 20 / 20)
            color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            x, y = all_points[0]
            state["query_points"][frame_idx].append((x, y, frame_idx))
            state["query_colors"][frame_idx].append(color_rgb)
            state["query_count"] = _effective_point_count(state)
            preview = _render_sam_preview(state)
            for pt, col in zip(
                state["query_points"][frame_idx], state["query_colors"][frame_idx]
            ):
                px, py, _ = pt
                cv2.circle(preview, (int(px), int(py)), 4, col, -1)
            return {
                "session_state": state,
                "preview_frame": preview,
                "query_count": state["query_count"],
                "notify": ("positive", "Mask 极小，已在质心放置 1 个追踪点"),
            }

    # 添加到 state（复用现有颜色分配逻辑）
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
    color_offset = len(state["query_points"][frame_idx])

    for i, (x, y) in enumerate(all_points):
        color = cmap((color_offset + i) % 20 / 20)
        color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        state["query_points"][frame_idx].append((x, y, frame_idx))
        state["query_colors"][frame_idx].append(color_rgb)

    state["query_count"] = _effective_point_count(state)
    actual = len(all_points)
    log.info("Generated %d tracking points from mask (requested %d)", actual, num_points)

    preview = _render_sam_preview(state)
    for pt, col in zip(
        state["query_points"][frame_idx], state["query_colors"][frame_idx]
    ):
        px, py, _ = pt
        cv2.circle(preview, (int(px), int(py)), 4, col, -1)

    msg = f"已在 Mask 内生成 {actual} 个追踪点"
    if actual < num_points:
        msg += f"（请求 {num_points}，可用像素不足）"

    return {
        "session_state": state,
        "preview_frame": preview,
        "query_count": state["query_count"],
        "notify": ("positive", msg),
    }


# ---------------------------------------------------------------------------
# Keyframe save / delete / display (mirrors app_logic.py save_keyframe etc.)
# ---------------------------------------------------------------------------
def save_tracking_keyframe(state: dict) -> dict[str, Any]:
    """Save current frame's query points as a keyframe.

    Args:
        state: Current tracking state.

    Returns:
        Dict with session_state, keyframe_info, keyframe_gallery.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    idx = state["current_frame"]
    pts = state["query_points"][idx]
    cols = state["query_colors"][idx]

    if not pts:
        return {
            "session_state": state,
            "keyframe_info": tracking_keyframe_display(state),
            "keyframe_gallery": tracking_keyframe_gallery(state),
            "notify": ("warning", "当前帧没有追踪点，请先添加点位。"),
        }

    state["keyframes"][idx] = {
        "points": [tuple(p) for p in pts],
        "colors": [tuple(c) for c in cols],
    }
    state["query_count"] = _effective_point_count(state)
    log.info("Saved tracking keyframe at frame %d (%d points)", idx, len(pts))

    return {
        "session_state": state,
        "keyframe_info": tracking_keyframe_display(state),
        "keyframe_gallery": tracking_keyframe_gallery(state),
        "query_count": state["query_count"],
        "notify": ("positive", f"关键帧已保存: 帧 {idx} ({len(pts)} 个点)"),
    }


def delete_tracking_keyframe(state: dict) -> dict[str, Any]:
    """Delete the keyframe at the current frame.

    Args:
        state: Current tracking state.

    Returns:
        Dict with session_state, preview_frame, keyframe_info, keyframe_gallery.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state}

    idx = state["current_frame"]
    if idx not in state.get("keyframes", {}):
        return {
            "session_state": state,
            "notify": ("warning", "当前帧没有保存的关键帧。"),
        }

    del state["keyframes"][idx]
    state["query_points"][idx] = []
    state["query_colors"][idx] = []

    state["query_count"] = _effective_point_count(state)

    log.info("Deleted tracking keyframe at frame %d", idx)

    preview = _get_preview_frame(state, idx)
    return {
        "session_state": state,
        "preview_frame": preview,
        "keyframe_info": tracking_keyframe_display(state),
        "keyframe_gallery": tracking_keyframe_gallery(state),
    }


def tracking_keyframe_display(state: dict) -> str:
    """Build keyframe info string for display.

    Args:
        state: Current tracking state.

    Returns:
        e.g. "关键帧: #0(5点)  #30(12点)"
    """
    keyframes = state.get("keyframes", {})
    if not keyframes:
        return "尚未保存任何关键帧。"
    parts = [
        f"#{idx}({len(kf['points'])}点)"
        for idx, kf in sorted(keyframes.items())
    ]
    return "关键帧: " + "  ".join(parts)


def tracking_keyframe_gallery(state: dict) -> list[tuple[np.ndarray, str]]:
    """Build gallery items for saved tracking keyframes.

    Args:
        state: Current tracking state.

    Returns:
        List of (thumbnail_image, caption) tuples.
    """
    keyframes = state.get("keyframes", {})
    if not keyframes or state.get("frames_dir") is None:
        return []

    items = []
    for idx in sorted(keyframes.keys()):
        kf = keyframes[idx]
        preview = _get_preview_frame(state, idx)
        for pt, col in zip(kf["points"], kf["colors"]):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(preview, (px, py), 4, tuple(col), -1)
        items.append((preview, f"第 {idx} 帧 ({len(kf['points'])}点)"))
    return items


def _paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw tracking points on video frames.

    Args:
        frames: Video frames (T, H, W, 3) uint8.
        point_tracks: Track coordinates (N, T, 2) float.
        visibles: Visibility mask (N, T) bool.
        colormap: Optional list of RGB colors for each point.

    Returns:
        Video with tracks drawn (T, H, W, 3) uint8.
    """
    num_points, num_frames = point_tracks.shape[0:2]
    if colormap is None:
        colormap = _get_colors(num_points)

    height, width = frames.shape[1:3]
    dot_size_frac = 0.015
    radius = int(round(min(height, width) * dot_size_frac))
    diam = radius * 2 + 1

    quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
    quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
    icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
    sharpness = 0.15
    icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
    icon = 1 - icon[:, :, np.newaxis]
    icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
    icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
    icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
    icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

    video = frames.copy()
    for t in range(num_frames):
        image = np.pad(
            video[t],
            [(radius + 1, radius + 1), (radius + 1, radius + 1), (0, 0)],
        )
        for i in range(num_points):
            x, y = point_tracks[i, t, :] + 0.5
            x = min(max(x, 0.0), width)
            y = min(max(y, 0.0), height)

            if visibles[i, t]:
                x1, y1 = int(np.floor(x)), int(np.floor(y))
                x2, y2 = x1 + 1, y1 + 1

                patch = (
                    icon1 * (x2 - x) * (y2 - y)
                    + icon2 * (x2 - x) * (y - y1)
                    + icon3 * (x - x1) * (y2 - y)
                    + icon4 * (x - x1) * (y - y1)
                )
                x_ub = x1 + 2 * radius + 2
                y_ub = y1 + 2 * radius + 2
                image[y1:y_ub, x1:x_ub, :] = (
                    (1 - patch) * image[y1:y_ub, x1:x_ub, :]
                    + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]
                )

        video[t] = image[radius + 1 : -radius - 1, radius + 1 : -radius - 1].astype(
            np.uint8
        )
    return video


_cotracker_engine = None


def _get_cotracker():
    """Lazy-load CoTracker engine."""
    global _cotracker_engine
    if _cotracker_engine is None:
        from engines.cotracker_engine import CoTrackerEngine
        _cotracker_engine = CoTrackerEngine()
    return _cotracker_engine


def run_tracking(
    state: dict,
    use_grid: bool = False,
    grid_size: int = 15,
    backward_tracking: bool = False,
    progress_callback: ProgressCallback = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Execute point tracking on the video.

    Args:
        state: Current tracking state.
        use_grid: If True, use grid mode instead of user-selected points.
        grid_size: Grid size for grid mode.
        backward_tracking: If True, track both forward and backward from query frames.
        progress_callback: Optional progress callback.
        cancel_event: Optional threading.Event; if set, tracking is cancelled.

    Returns:
        Dict with session_state, result_video_path, notify.
    """
    if state.get("frames_dir") is None:
        return {"session_state": state, "notify": ("error", "请先上传视频")}

    input_frames = _load_input_frames(state)
    preview_frames = _load_all_preview_frames(state)
    preview_H, preview_W = state["preview_size"]
    input_H, input_W = COTRACKER_INPUT_RESO

    if use_grid:
        queries = None
        log.info("Running grid tracking with size %d", grid_size)
    else:
        # Collect points only from saved keyframes
        all_points = []
        all_colors = []
        keyframes = state.get("keyframes", {})
        if keyframes:
            for idx in sorted(keyframes.keys()):
                kf = keyframes[idx]
                all_points.extend(kf["points"])
                all_colors.extend(kf["colors"])
        else:
            # Fallback: use all query_points (for direct run without keyframes)
            for frame_pts, frame_cols in zip(
                state["query_points"], state["query_colors"]
            ):
                all_points.extend(frame_pts)
                all_colors.extend(frame_cols)

        if len(all_points) == 0:
            return {
                "session_state": state,
                "notify": ("warning", "请先保存至少一个关键帧，或使用网格模式"),
            }

        queries_list = []
        for pt in all_points:
            x, y, t = pt
            x_input = x * (input_W / preview_W)
            y_input = y * (input_H / preview_H)
            queries_list.append([t, x_input, y_input])
        queries = np.array(queries_list, dtype=np.float32)
        log.info("Running tracking with %d query points", len(queries))

    engine = _get_cotracker()
    tracks, visibility = engine.track(
        input_frames,
        queries=queries,
        grid_size=grid_size if use_grid else 0,
        backward_tracking=backward_tracking,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    # Store raw tracks in input space for AE export
    state["raw_tracks"] = tracks.copy()
    state["raw_visibility"] = visibility.copy()

    scale_x = preview_W / input_W
    scale_y = preview_H / input_H
    tracks_preview = tracks.copy()
    tracks_preview[:, :, 0] *= scale_x
    tracks_preview[:, :, 1] *= scale_y

    if use_grid:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        num_pts = tracks.shape[0]
        colors = []
        for i in range(num_pts):
            color = cmap(i / float(num_pts))
            colors.append((int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
    else:
        colors = all_colors

    painted_video = _paint_point_track(preview_frames, tracks_preview, visibility, colors)

    output_dir = WORKSPACE_DIR / "tracking_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_filename = f"tracking_{uuid.uuid4().hex[:8]}.mp4"
    video_path = output_dir / video_filename

    fps = state.get("fps", 24.0)
    writer = imageio.get_writer(str(video_path), fps=fps)
    for frame in painted_video:
        writer.append_data(frame)
    writer.close()

    state["result_video_path"] = str(video_path)
    log.info("Tracking result saved to %s", video_path)

    return {
        "session_state": state,
        "result_video_path": str(video_path),
        "notify": ("positive", "追踪完成！"),
    }
