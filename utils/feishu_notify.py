"""Send structured notifications to Feishu (飞书) via webhook."""

import logging
import platform
import socket
import traceback

import requests

from config import FEISHU_WEBHOOK_URL, get_device

log = logging.getLogger(__name__)

MAX_STACKTRACE_LEN = 2000


def _lan_ip() -> str:
    """Detect the LAN IP address via UDP socket routing (no data sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


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
    lan_url = local_url.replace("0.0.0.0", _lan_ip()).replace("127.0.0.1", _lan_ip())
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
    """
    keyframes_str = ", ".join(str(i) for i in sorted(keyframe_indices))

    links_lines = []
    for name, url in cdn_urls.items():
        links_lines.append(f"- [{name}]({url})")
    links_md = "\n".join(links_lines) if links_lines else "无"

    content_md = (
        f"**原始视频**\n"
        f"- 文件名: {source_filename}\n"
        f"- 分辨率: {video_width}×{video_height}\n"
        f"- 时长: {_format_duration(video_duration)} | "
        f"{num_frames} 帧 | {fps:.2f} fps\n"
        f"- 格式: {video_format} | "
        f"大小: {_format_file_size(file_size)}\n\n"
        f"**模型与参数**\n"
        f"- 模型: SAM 2.1 + MatAnyone\n"
        f"- 腐蚀核: {erode} | 膨胀核: {dilate} | Warmup: {warmup}\n"
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
            {
                "tag": "note",
                "elements": [
                    {"tag": "plain_text", "content": f"Session: {session_id}"},
                ],
            },
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
            {
                "tag": "note",
                "elements": [
                    {"tag": "plain_text", "content": f"Session: {session_id}"},
                ],
            },
        ],
    }
    _post_card(card)
