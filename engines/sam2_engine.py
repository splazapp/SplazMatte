"""SAM2.1 image predictor wrapper for interactive mask annotation."""

import sys
import logging
from pathlib import Path

import numpy as np
import torch

from config import SAM2_CHECKPOINT, SAM2_CONFIG, SDKS_DIR, get_device

log = logging.getLogger(__name__)

# Add SAM2 SDK to path so hydra can find configs via initialize_config_module
_sam2_root = SDKS_DIR / "sam2"
if str(_sam2_root) not in sys.path:
    sys.path.insert(0, str(_sam2_root))


class SAM2Engine:
    """Wrapper around SAM2.1 image predictor for point-based segmentation.

    Usage:
        engine = SAM2Engine()
        engine.set_image(rgb_array)
        mask = engine.predict(points=[[x, y]], labels=[1])
    """

    def __init__(
        self,
        checkpoint: Path = SAM2_CHECKPOINT,
        config: str = SAM2_CONFIG,
        device: torch.device | None = None,
    ) -> None:
        """Initialize SAM2 model and image predictor.

        Args:
            checkpoint: Path to SAM2.1 checkpoint file.
            config: Config name relative to sam2/configs/ (hydra).
            device: Torch device. Auto-detected if None.
        """
        self.device = device or get_device()

        # Import SAM2 build utilities (triggers hydra config init via sam2/__init__.py)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        log.info("Loading SAM2 model from %s", checkpoint)
        model = build_sam2(
            config_file=config,
            ckpt_path=str(checkpoint),
            device=str(self.device),
        )
        self.predictor = SAM2ImagePredictor(model)
        log.info("SAM2 engine ready on %s", self.device)

    def set_image(self, image: np.ndarray) -> None:
        """Compute image embeddings for a new frame.

        Args:
            image: RGB uint8 array of shape (H, W, 3).
        """
        self.predictor.set_image(image)

    def predict(
        self,
        points: list[list[int]],
        labels: list[int],
    ) -> np.ndarray:
        """Predict a binary mask from point prompts.

        Args:
            points: List of [x, y] pixel coordinates.
            labels: List of labels (1=foreground, 0=background).

        Returns:
            Binary mask of shape (H, W) as uint8 (0 or 255).
        """
        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Pick the mask with the highest predicted IoU
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8) * 255
        return mask

    def reset(self) -> None:
        """Clear image embeddings."""
        self.predictor.reset_predictor()
