"""Send structured notifications to Feishu (飞书) via webhook."""

import logging
import platform
import traceback

import requests

from config import DEFAULT_MATTING_ENGINE, FEISHU_WEBHOOK_URL, get_device
from utils.lan_ip import lan_ip as _lan_ip

log = logging.getLogger(__name__)

MAX_STACKTRACE_LEN = 2000


def _device_info() -> str:
    """Return a short string describing the compute device and host."""
    device = get_device()
    device_label = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
    hostname = platform.node() or "unknown"
    return f"{device_label.get(device.type, device.type)} | {hostname}"


def _post_card(card: dict) -> None:
    """POST an interactive card to the Feishu webhook.

    Errors are logged and swallowed so they never affect the main flow.
    """
    payload = {"msg_type": "interactive", "card": card}
    try:
        resp = requests.post(FEISHU_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("Feishu notification sent successfully.")
    except Exception:
        log.exception("Failed to send Feishu notification")


def _format_file_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_duration(seconds: float) -> str:
    """Return duration as ``Xm Ys`` or ``Ys``."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


def send_feishu_startup(local_url: str) -> None:
    """Send a startup notification with the LAN URL."""
    ip = _lan_ip()
    lan_url = local_url.replace("localhost", ip).replace("0.0.0.0", ip).replace("127.0.0.1", ip)
    content_md = (
        f"**局域网链接**: [{lan_url}]({lan_url})\n"
        f"**设备**: {_device_info()}"
    )
    card = {
        "header": {
            "title": {"tag": "plain_text", "content": "SplazMatte 已启动 🚀"},
            "template": "blue",
        },
        "elements": [
            {"tag": "markdown", "content": content_md},
        ],
    }
    _post_card(card)


def _sam_label(model_type: str) -> str:
    """Return human-readable SAM model name."""
    return "SAM 3" if model_type == "SAM3" else "SAM 2.1"


def _engine_params_line(
    engine: str,
    *,
    erode: int,
    dilate: int,
    warmup: int,
    batch_size: int,
    overlap: int,
    seed: int,
) -> str:
    """Return the engine-specific parameter summary line."""
    if engine == "VideoMaMa":
        return f"批次大小: {batch_size} | 重叠帧: {overlap} | 随机种子: {seed}"
    return f"腐蚀核: {erode} | 膨胀核: {dilate} | Warmup: {warmup}"


def send_feishu_success(
    *,
    session_id: str,
    source_filename: str,
    video_width: int,
    video_height: int,
    video_duration: float,
    num_frames: int,
    fps: float,
    video_format: str,
    file_size: int,
    erode: int,
    dilate: int,
    warmup: int,
    keyframe_indices: list[int],
    processing_time: float,
    start_time: str,
    end_time: str,
    cdn_urls: dict[str, str],
    matting_engine: str = DEFAULT_MATTING_ENGINE,
    model_type: str = "SAM2",
    batch_size: int = 0,
    overlap: int = 0,
    seed: int = 0,
) -> None:
    """Send a success notification card to Feishu.

    Args:
        session_id: Unique session identifier.
        source_filename: Original uploaded video filename.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        video_duration: Video duration in seconds.
        num_frames: Total number of frames.
        fps: Frames per second.
        video_format: File extension / container format.
        file_size: Source file size in bytes.
        erode: Erosion kernel size used.
        dilate: Dilation kernel size used.
        warmup: Warmup frame count.
        keyframe_indices: List of annotated keyframe indices.
        processing_time: Total processing seconds.
        start_time: ISO-formatted start time string.
        end_time: ISO-formatted end time string.
        cdn_urls: Mapping of filename → CDN URL.
        matting_engine: Matting engine name (``MatAnyone`` or ``VideoMaMa``).
        model_type: SAM model variant (``SAM2`` or ``SAM3``).
        batch_size: VideoMaMa batch size (only used when engine is VideoMaMa).
        overlap: VideoMaMa overlap frames (only used when engine is VideoMaMa).
        seed: VideoMaMa random seed (only used when engine is VideoMaMa).
    """
    keyframes_str = ", ".join(str(i) for i in sorted(keyframe_indices))

    links_lines = []
    for name, url in cdn_urls.items():
        links_lines.append(f"- [{name}]({url})")
    links_md = "\n".join(links_lines) if links_lines else "无"

    content_md = (
        f"**Session**: {session_id}\n\n"
        f"**原始视频**\n"
        f"- 文件名: {source_filename}\n"
        f"- 分辨率: {video_width}×{video_height}\n"
        f"- 时长: {_format_duration(video_duration)} | "
        f"{num_frames} 帧 | {fps:.2f} fps\n"
        f"- 格式: {video_format} | "
        f"大小: {_format_file_size(file_size)}\n\n"
        f"**模型与参数**\n"
        f"- 模型: {_sam_label(model_type)} + {matting_engine}\n"
        f"- {_engine_params_line(matting_engine, erode=erode, dilate=dilate, warmup=warmup, batch_size=batch_size, overlap=overlap, seed=seed)}\n"
        f"- 关键帧: [{keyframes_str}]\n\n"
        f"**处理耗时**\n"
        f"- 开始: {start_time}\n"
        f"- 结束: {end_time}\n"
        f"- 总耗时: {_format_duration(processing_time)}\n\n"
        f"**运行环境**\n"
        f"- 设备: {_device_info()}\n\n"
        f"**输出文件**\n{links_md}"
    )

    card = {
        "header": {
            "title": {"tag": "plain_text", "content": f"SplazMatte 抠像完成 ✅"},
            "template": "green",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": content_md,
            },
        ],
    }
    _post_card(card)


def send_feishu_tracking_success(
    *,
    session_id: str,
    source_filename: str,
    video_width: int,
    video_height: int,
    num_frames: int,
    fps: float,
    keyframe_indices: list[int],
    total_points: int,
    use_grid: bool,
    grid_size: int,
    backward_tracking: bool,
    processing_time: float,
    start_time: str,
    end_time: str,
    cdn_urls: dict[str, str],
) -> None:
    """Send a tracking success notification card to Feishu.

    Args:
        session_id: Unique session identifier.
        source_filename: Original uploaded video filename.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        num_frames: Total number of frames.
        fps: Frames per second.
        keyframe_indices: List of keyframe frame indices.
        total_points: Total number of tracking query points.
        use_grid: Whether grid mode was used.
        grid_size: Grid size if grid mode.
        backward_tracking: Whether backward tracking was enabled.
        processing_time: Total processing seconds.
        start_time: ISO-formatted start time string.
        end_time: ISO-formatted end time string.
        cdn_urls: Mapping of filename to CDN URL.
    """
    keyframes_str = ", ".join(str(i) for i in sorted(keyframe_indices))
    mode = f"网格 {grid_size}x{grid_size}" if use_grid else f"手动 {total_points} 点"
    direction = "双向" if backward_tracking else "仅前向"

    links_lines = [f"- [{name}]({url})" for name, url in cdn_urls.items()]
    links_md = "\n".join(links_lines) if links_lines else "无"

    content_md = (
        f"**Session**: {session_id}\n\n"
        f"**原始视频**\n"
        f"- 文件名: {source_filename}\n"
        f"- 分辨率: {video_width}×{video_height}\n"
        f"- {num_frames} 帧 | {fps:.2f} fps\n\n"
        f"**追踪参数**\n"
        f"- 模式: {mode}\n"
        f"- 方向: {direction}\n"
        f"- 关键帧: [{keyframes_str}]\n\n"
        f"**处理耗时**\n"
        f"- 开始: {start_time}\n"
        f"- 结束: {end_time}\n"
        f"- 总耗时: {_format_duration(processing_time)}\n\n"
        f"**运行环境**\n"
        f"- 设备: {_device_info()}\n\n"
        f"**输出文件**\n{links_md}"
    )

    card = {
        "header": {
            "title": {"tag": "plain_text", "content": "SplazMatte 追踪完成 ✅"},
            "template": "turquoise",
        },
        "elements": [
            {"tag": "markdown", "content": content_md},
        ],
    }
    _post_card(card)


def send_feishu_failure(session_id: str, error: Exception) -> None:
    """Send a failure notification card to Feishu.

    Args:
        session_id: Unique session identifier.
        error: The exception that caused the failure.
    """
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    stacktrace = "".join(tb)
    if len(stacktrace) > MAX_STACKTRACE_LEN:
        stacktrace = stacktrace[:MAX_STACKTRACE_LEN] + "\n... (truncated)"

    content_md = (
        f"**Session**: {session_id}\n"
        f"**设备**: {_device_info()}\n"
        f"**错误类型**: {type(error).__name__}\n"
        f"**错误信息**: {error}\n\n"
        f"**堆栈**\n```\n{stacktrace}\n```"
    )

    card = {
        "header": {
            "title": {"tag": "plain_text", "content": "SplazMatte 抠像失败 ❌"},
            "template": "red",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": content_md,
            },
        ],
    }
    _post_card(card)
