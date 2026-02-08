"""Data models for the task queue."""

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QueueItem:
    """Snapshot of one matting task.

    Mask arrays are deep-copied from session state so the queue item
    is independent of subsequent annotation edits.  Status fields
    (``status``, ``error_msg``, ``propagated_masks``) are updated
    during queue execution.
    """

    item_id: str
    session_id: str
    source_video_path: Path
    original_filename: str
    frames_dir: Path
    num_frames: int
    fps: float
    video_width: int
    video_height: int
    video_duration: float
    video_file_size: int
    video_format: str
    keyframes: dict[int, np.ndarray]
    propagated_masks: dict[int, np.ndarray]
    has_propagation: bool
    model_type: str
    matting_engine: str
    erode: int
    dilate: int
    batch_size: int
    overlap: int
    seed: int
    status: str = "pending"
    error_msg: str = ""


def snapshot_to_queue_item(
    state: dict,
    matting_engine: str,
    erode: int,
    dilate: int,
    batch_size: int,
    overlap: int,
    seed: int,
) -> QueueItem:
    """Create a QueueItem by deep-copying masks from session state.

    Args:
        state: Current session state dict.
        matting_engine: "MatAnyone" or "VideoMaMa".
        erode: Erosion kernel size.
        dilate: Dilation kernel size.
        batch_size: VideoMaMa batch size.
        overlap: VideoMaMa overlap frames.
        seed: VideoMaMa random seed.

    Returns:
        A new QueueItem with independent mask copies.
    """
    keyframes = {idx: mask.copy() for idx, mask in state["keyframes"].items()}
    propagated = {
        idx: mask.copy()
        for idx, mask in state.get("propagated_masks", {}).items()
    }

    return QueueItem(
        item_id=str(uuid.uuid4())[:8],
        session_id=state["session_id"],
        source_video_path=state["source_video_path"],
        original_filename=state.get("original_filename", ""),
        frames_dir=state["frames_dir"],
        num_frames=state["num_frames"],
        fps=state["fps"],
        video_width=state["video_width"],
        video_height=state["video_height"],
        video_duration=state["video_duration"],
        video_file_size=state["video_file_size"],
        video_format=state["video_format"],
        keyframes=keyframes,
        propagated_masks=propagated,
        has_propagation=len(propagated) > 0,
        model_type=state.get("model_type", "SAM2"),
        matting_engine=matting_engine,
        erode=erode,
        dilate=dilate,
        batch_size=batch_size,
        overlap=overlap,
        seed=seed,
    )
