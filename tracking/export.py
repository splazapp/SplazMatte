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
    labels: list[str] | None = None,
) -> str:
    """Format tracking data as AE Keyframe Data text.

    Args:
        tracks: Point coordinates (N, T, 2) in original video pixels.
        visibility: Visibility mask (N, T) bool.
        fps: Video frame rate.
        width: Original video width.
        height: Original video height.
        labels: Optional list of N label strings. Defaults to null_1, null_2, ...

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
    if labels is not None and len(labels) != num_points:
        log.warning("labels length %d != num_points %d, falling back to auto-label", len(labels), num_points)
        labels = None
    for i in range(num_points):
        label = labels[i] if labels is not None else f"null_{i + 1}"
        lines.append(f"Effects\t{label}\tPosition")
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
    sid = state.get("session_id") or uuid.uuid4().hex
    filename = f"ae_原始轨迹_{sid[:8]}.txt"
    export_path = output_dir / filename
    export_path.write_text(text, encoding="utf-8")

    state["ae_export_path"] = str(export_path)
    log.info("AE keyframe data exported to %s (%d points)", export_path, raw_tracks.shape[0])

    return {
        "session_state": state,
        "export_path": str(export_path),
        "notify": ("positive", f"导出成功：{raw_tracks.shape[0]} 个追踪点"),
    }


def compute_trajectory_summary(
    raw_tracks: np.ndarray,
    raw_visibility: np.ndarray,
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
) -> np.ndarray:
    """每帧对可见点做 IQR 离群点剔除后取平均，返回 (T, 2) 原始分辨率坐标。

    Args:
        raw_tracks: Point coordinates (N, T, 2) in input resolution.
        raw_visibility: Visibility mask (N, T) bool.
        orig_w: Original video width.
        orig_h: Original video height.
        input_w: Input (CoTracker) resolution width.
        input_h: Input (CoTracker) resolution height.

    Returns:
        (T, 2) array of mean positions per frame. NaN where no visible points.
    """
    tracks = raw_tracks.copy().astype(float)
    tracks[:, :, 0] *= orig_w / input_w
    tracks[:, :, 1] *= orig_h / input_h

    N, T, _ = tracks.shape
    summary = np.full((T, 2), np.nan)

    for t in range(T):
        visible = raw_visibility[:, t]
        pts = tracks[visible, t, :]
        if len(pts) == 0:
            continue
        if len(pts) <= 2:
            summary[t] = pts.mean(axis=0)
            continue

        # IQR filtering (x and y independently)
        valid = np.ones(len(pts), dtype=bool)
        for dim in range(2):
            q1, q3 = np.percentile(pts[:, dim], [25, 75])
            iqr = q3 - q1
            if iqr < 1e-6:  # all points at same position, skip filtering
                continue
            valid &= (pts[:, dim] >= q1 - 1.5 * iqr) & (pts[:, dim] <= q3 + 1.5 * iqr)

        filtered = pts[valid]
        summary[t] = filtered.mean(axis=0) if len(filtered) > 0 else pts.mean(axis=0)

    return summary


def _format_jsx_script(
    summary: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> str:
    """生成可在 After Effects 中运行的 JSX 脚本，创建整体轨迹 Null 层。

    Args:
        summary: (T, 2) array of positions in original resolution. NaN = invalid.
        fps: Video frame rate.
        width: Original video width.
        height: Original video height.

    Returns:
        JSX script string.
    """
    keyframes = []
    for t in range(len(summary)):
        if not (np.isnan(summary[t, 0]) or np.isnan(summary[t, 1])):
            keyframes.append((t, float(summary[t, 0]), float(summary[t, 1])))

    kf_lines = ",\n        ".join(
        f"[{t}, {x:.2f}, {y:.2f}]" for t, x, y in keyframes
    )

    return f"""\
(function() {{
    var comp = app.project.activeItem;
    if (!comp || !(comp instanceof CompItem)) {{
        alert("请先选中一个 After Effects 合成！");
        return;
    }}
    var fps = {fps};
    var sourceWidth = {width};
    var sourceHeight = {height};
    var layer = comp.layers.addNull(comp.duration);
    layer.name = "整体轨迹";
    var pos = layer.property("Transform").property("Position");
    var keyframes = [
        {kf_lines}
    ];
    for (var i = 0; i < keyframes.length; i++) {{
        var kf = keyframes[i];
        pos.setValueAtTime(kf[0] / fps, [kf[1], kf[2]]);
    }}
    alert("已创建整体轨迹图层，共 " + keyframes.length + " 个关键帧。\\n视频分辨率: " + sourceWidth + "x" + sourceHeight);
}})();
"""


def export_trajectory_summary(state: dict) -> dict[str, Any]:
    """基于 raw_tracks 计算整体轨迹，生成 ae_整体轨迹.txt 和 ae_整体轨迹.jsx。

    Args:
        state: Tracking state containing raw_tracks and raw_visibility.

    Returns:
        Dict with session_state, summary_txt_path, summary_jsx_path, notify.
    """
    raw_tracks = state.get("raw_tracks")
    raw_vis = state.get("raw_visibility")
    if raw_tracks is None or raw_vis is None:
        return {"session_state": state, "notify": ("warning", "请先运行追踪")}

    original_size = state.get("original_size")
    if not original_size or original_size == (0, 0):
        return {"session_state": state, "notify": ("warning", "视频尺寸信息缺失，请重新上传视频")}
    orig_h, orig_w = original_size
    input_h, input_w = COTRACKER_INPUT_RESO
    fps = state.get("fps", 24.0)
    sid = state.get("session_id") or uuid.uuid4().hex

    summary = compute_trajectory_summary(raw_tracks, raw_vis, orig_w, orig_h, input_w, input_h)

    valid_mask = ~np.isnan(summary[:, 0])
    num_valid = int(valid_mask.sum())
    if num_valid == 0:
        return {
            "session_state": state,
            "notify": ("warning", "整体轨迹无有效帧（所有帧均无可见追踪点），请检查追踪结果"),
        }

    output_dir = WORKSPACE_DIR / "tracking_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # TXT file (standard AE keyframe format, single track)
    T = summary.shape[0]
    tracks_1 = summary[np.newaxis].copy()  # (1, T, 2)
    # Replace NaN with 0 for array; visibility mask handles which frames to emit
    tracks_1 = np.nan_to_num(tracks_1, nan=0.0)
    vis_1 = valid_mask[np.newaxis]  # (1, T)
    txt_content = _format_ae_keyframes(tracks_1, vis_1, fps, orig_w, orig_h, labels=["整体轨迹"])

    txt_path = output_dir / f"ae_整体轨迹_{sid[:8]}.txt"
    txt_path.write_text(txt_content, encoding="utf-8")

    # JSX file
    jsx_content = _format_jsx_script(summary, fps, orig_w, orig_h)
    jsx_path = output_dir / f"ae_整体轨迹_{sid[:8]}.jsx"
    jsx_path.write_text(jsx_content, encoding="utf-8")

    state["ae_summary_txt_path"] = str(txt_path)
    state["ae_summary_jsx_path"] = str(jsx_path)

    log.info(
        "Summary trajectory exported: %d valid frames, txt=%s, jsx=%s",
        num_valid, txt_path, jsx_path,
    )

    return {
        "session_state": state,
        "summary_txt_path": str(txt_path),
        "summary_jsx_path": str(jsx_path),
        "notify": ("positive", f"整体轨迹导出成功：{num_valid} 帧有效位置"),
    }
