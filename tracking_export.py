"""After Effects keyframe data export for tracking results."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from config import COTRACKER_INPUT_RESO, WORKSPACE_DIR

log = logging.getLogger(__name__)


def _format_ae_keyframes(
    tracks: np.ndarray,
    visibility: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> str:
    """Format tracking data as AE Keyframe Data text.

    Args:
        tracks: Point coordinates (N, T, 2) in original video pixels.
        visibility: Visibility mask (N, T) bool.
        fps: Video frame rate.
        width: Original video width.
        height: Original video height.

    Returns:
        Adobe After Effects Keyframe Data formatted string.
    """
    lines = [
        "Adobe After Effects 8.0 Keyframe Data",
        "",
        f"\tUnits Per Second\t{fps:.6f}",
        f"\tSource Width\t{width}",
        f"\tSource Height\t{height}",
        "\tSource Pixel Aspect Ratio\t1",
        "\tComp Pixel Aspect Ratio\t1",
        "",
    ]

    num_points, num_frames = tracks.shape[:2]
    for i in range(num_points):
        lines.append(f"Effects\tnull_{i + 1}\tPosition")
        lines.append("\tFrame\tX pixels\tY pixels")
        for t in range(num_frames):
            if visibility[i, t]:
                x = tracks[i, t, 0]
                y = tracks[i, t, 1]
                lines.append(f"\t{t}\t{x:.1f}\t{y:.1f}")
        lines.append("")

    lines.append("End of Keyframe Data")
    return "\n".join(lines)


def export_ae_keyframe_data(state: dict) -> dict[str, Any]:
    """Export tracking results as After Effects keyframe data.

    Args:
        state: Tracking state containing raw_tracks and raw_visibility.

    Returns:
        Dict with session_state, export_path, notify.
    """
    raw_tracks = state.get("raw_tracks")
    raw_vis = state.get("raw_visibility")
    if raw_tracks is None or raw_vis is None:
        return {
            "session_state": state,
            "notify": ("warning", "请先运行追踪"),
        }

    orig_h, orig_w = state["original_size"]
    input_h, input_w = COTRACKER_INPUT_RESO

    # Scale from input resolution to original video resolution
    tracks_orig = raw_tracks.copy()
    tracks_orig[:, :, 0] *= orig_w / input_w
    tracks_orig[:, :, 1] *= orig_h / input_h

    fps = state.get("fps", 24.0)
    text = _format_ae_keyframes(tracks_orig, raw_vis, fps, orig_w, orig_h)

    output_dir = WORKSPACE_DIR / "tracking_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ae_export_{uuid.uuid4().hex[:8]}.txt"
    export_path = output_dir / filename
    export_path.write_text(text, encoding="utf-8")

    state["ae_export_path"] = str(export_path)
    log.info("AE keyframe data exported to %s (%d points)", export_path, raw_tracks.shape[0])

    return {
        "session_state": state,
        "export_path": str(export_path),
        "notify": ("positive", f"导出成功：{raw_tracks.shape[0]} 个追踪点"),
    }
