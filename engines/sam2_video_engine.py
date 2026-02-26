"""SAM2.1 video predictor wrapper for multi-object mask propagation."""

import sys
import logging
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from config import SAM2_CHECKPOINT, SAM2_CONFIG, MAX_PROPAGATION_FRAMES, SDKS_DIR, get_device

log = logging.getLogger(__name__)

# Add SAM2 SDK to path so hydra can find configs
_sam2_root = SDKS_DIR / "sam2"
if str(_sam2_root) not in sys.path:
    sys.path.insert(0, str(_sam2_root))

# Suppress SAM2 _C CUDA extension warning on non-CUDA platforms (e.g. Mac MPS).
# The _C module is only needed for post-processing hole filling and its absence
# is harmless — SAM2 skips that step automatically.
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")


def _reset_hydra_for_sam2() -> None:
    """Rebind Hydra config search path to the SAM2 module."""
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    hydra = GlobalHydra.instance()
    if hydra.is_initialized():
        hydra.clear()
    initialize_config_module("sam2", version_base="1.2")


class SAM2VideoEngine:
    """Wrapper around SAM2.1 video predictor for mask propagation.

    Usage:
        engine = SAM2VideoEngine()
        masks = engine.propagate(frames_dir, {0: mask_a, 10: mask_b})
    """

    def __init__(
        self,
        checkpoint: Path = SAM2_CHECKPOINT,
        config: str = SAM2_CONFIG,
        device: torch.device | None = None,
    ) -> None:
        """Initialize SAM2 video predictor.

        Args:
            checkpoint: Path to SAM2.1 checkpoint file.
            config: Config name relative to sam2/configs/ (hydra).
            device: Torch device. Auto-detected if None.
        """
        self.device = device or get_device()

        # MatAnyone may have initialized Hydra with its own config root.
        # Reset here so SAM2 configs resolve from the sam2 package.
        _reset_hydra_for_sam2()

        from sam2.build_sam import build_sam2_video_predictor

        log.info("Loading SAM2 Video Predictor from %s", checkpoint)
        self.predictor = build_sam2_video_predictor(
            config_file=config,
            ckpt_path=str(checkpoint),
            device=str(self.device),
        )
        log.info("SAM2 Video Predictor ready on %s", self.device)

    @torch.inference_mode()
    def propagate(
        self,
        frames_dir: Path,
        keyframe_masks: dict[int, np.ndarray],
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[int, np.ndarray]:
        """Run bidirectional propagation from keyframe masks.

        Args:
            frames_dir: Directory containing JPEG frames.
            keyframe_masks: {frame_idx: mask (H, W) uint8 0/255}.
            progress_callback: Called with fraction [0, 1].

        Returns:
            Per-frame binary masks: {frame_idx: mask (H, W) uint8 0/255}.
        """
        if not keyframe_masks:
            raise ValueError("At least one keyframe mask is required.")

        num_jpgs = len(list(Path(frames_dir).glob("*.jpg")))
        if num_jpgs > MAX_PROPAGATION_FRAMES:
            raise ValueError(
                f"帧数 ({num_jpgs}) 超过传播上限 ({MAX_PROPAGATION_FRAMES})。"
                f"请使用更短的视频或调整 SPLAZMATTE_MAX_PROPAGATION_FRAMES 环境变量。"
            )

        non_cuda = self.device.type != "cuda"
        state = self.predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=non_cuda,
            offload_state_to_cpu=non_cuda,
            async_loading_frames=non_cuda,
        )
        num_frames = state["num_frames"]

        # Register all keyframe masks as obj_id=1
        for frame_idx, mask_np in keyframe_masks.items():
            mask_bool = torch.from_numpy((mask_np > 127).astype(np.uint8)).bool()
            self.predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=1,
                mask=mask_bool,
            )
        log.info(
            "Registered %d keyframe masks at frames %s",
            len(keyframe_masks),
            sorted(keyframe_masks.keys()),
        )

        result: dict[int, np.ndarray] = {}
        completed = 0

        def _to_mask(masks_tensor: torch.Tensor) -> np.ndarray:
            """Convert SAM2 output (num_obj, 1, H, W) to (H, W) uint8."""
            return (
                (masks_tensor[0] > 0.0).cpu().numpy().squeeze(0) * 255
            ).astype(np.uint8)

        # Forward propagation
        log.info("Forward propagation...")
        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(
            state
        ):
            result[frame_idx] = _to_mask(masks)
            completed += 1
            if progress_callback:
                progress_callback(completed / num_frames)

        # Reverse propagation: only fill frames not covered by forward pass
        log.info("Reverse propagation...")
        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(
            state, reverse=True
        ):
            if frame_idx not in result:
                result[frame_idx] = _to_mask(masks)
            completed += 1
            if progress_callback:
                progress_callback(min(1.0, completed / num_frames))

        self.predictor.reset_state(state)

        log.info("Propagation complete: %d frames", len(result))
        return result
