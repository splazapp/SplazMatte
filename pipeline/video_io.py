"""Video frame extraction and encoding utilities."""

import logging
import subprocess
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def extract_frames(video_path: Path, output_dir: Path) -> tuple[int, float]:
    """Extract video frames to JPEG files using ffmpeg.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to write JPEG frames (named 000000.jpg, ...).

    Returns:
        Tuple of (num_frames, fps).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe fps
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    num, den = probe.stdout.strip().split("/")
    fps = float(num) / float(den)

    # Extract frames
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-q:v", "2",
            str(output_dir / "%06d.jpg"),
        ],
        capture_output=True, check=True,
    )

    num_frames = len(list(output_dir.glob("*.jpg")))
    log.info("Extracted %d frames at %.2f fps from %s", num_frames, fps, video_path.name)
    return num_frames, fps


def load_frame(frames_dir: Path, frame_idx: int) -> np.ndarray:
    """Load a single frame as an RGB numpy array (H, W, 3).

    Args:
        frames_dir: Directory containing JPEG frames.
        frame_idx: 0-based frame index.

    Returns:
        RGB uint8 array of shape (H, W, 3).
    """
    # ffmpeg outputs 1-based filenames
    path = frames_dir / f"{frame_idx + 1:06d}.jpg"
    # cv2.imread cannot handle non-ASCII paths on Windows; use imdecode
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_all_frames_as_tensor(frames_dir: Path) -> torch.Tensor:
    """Load all frames as a (T, C, H, W) float tensor for MatAnyone.

    Pixel values are in [0, 255] float32, matching MatAnyone's expected input.

    Args:
        frames_dir: Directory containing JPEG frames.

    Returns:
        Float tensor of shape (T, 3, H, W).
    """
    paths = sorted(frames_dir.glob("*.jpg"))
    log.info("Loading %d frames from %s", len(paths), frames_dir)
    frames = []
    for p in tqdm(paths, desc="加载帧数据", unit="帧"):
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    # (T, H, W, C) -> (T, C, H, W)
    arr = np.stack(frames)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float()


def encode_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    """Encode numpy frames to MP4 via imageio.

    Args:
        frames: Array of shape (T, H, W, C) uint8.
        output_path: Output .mp4 file path.
        fps: Frames per second for the output video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, quality=7)
    try:
        for frame in tqdm(frames, desc=f"编码 {output_path.stem}", unit="帧"):
            writer.append_data(frame)
    finally:
        writer.close()
    log.info("Encoded %d frames to %s", len(frames), output_path.name)
