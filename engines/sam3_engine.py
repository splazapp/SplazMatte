"""SAM3 image predictor wrapper for interactive mask annotation.

Supports both point-click prompts (same as SAM2) and text prompts.
"""

import logging
from pathlib import Path

import numpy as np
from config import SAM3_CHECKPOINT, get_device  # must precede torch for MPS env var

import torch

from engines._sam3_deps import setup_sam3_deps

log = logging.getLogger(__name__)

setup_sam3_deps()


class SAM3Engine:
    """Wrapper around SAM3 for point-based and text-based segmentation.

    Usage:
        engine = SAM3Engine()
        engine.set_image(rgb_array)
        mask = engine.predict(points=[[x, y]], labels=[1])
        mask = engine.predict_text("person")
    """

    def __init__(
        self,
        checkpoint: Path = SAM3_CHECKPOINT,
        device: torch.device | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize SAM3 model with both point and text prediction.

        Args:
            checkpoint: Path to SAM3 unified checkpoint file.
            device: Torch device. Auto-detected if None.
            confidence_threshold: Minimum score for text-prompt detections.
        """
        self.device = device or get_device()
        device_str = str(self.device)

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        log.info("Loading SAM3 model from %s", checkpoint)
        model = build_sam3_image_model(
            enable_inst_interactivity=True,
            load_from_HF=False,
            checkpoint_path=str(checkpoint),
            device=device_str,
        )

        # The SDK's _setup_device_and_mode only moves weights when
        # device=="cuda".  On MPS we must move explicitly.
        if self.device.type != "cuda":
            model = model.to(self.device)

        # The SDK builds the interactive tracker without a backbone
        # (build_tracker defaults with_backbone=False).  Share the main
        # model's VL backbone so forward_image() works.
        model.inst_interactive_predictor.model.backbone = model.backbone

        # Point-click predictor (SAM2-compatible API)
        self.predictor = model.inst_interactive_predictor
        # Text-prompt processor
        self.processor = Sam3Processor(
            model, device=device_str, confidence_threshold=confidence_threshold,
        )
        self._processor_state: dict | None = None
        log.info("SAM3 engine ready on %s", self.device)

    def set_image(self, image: np.ndarray) -> None:
        """Compute image embeddings for a new frame.

        Prepares both point predictor and text processor.

        Args:
            image: RGB uint8 array of shape (H, W, 3).
        """
        self.predictor.set_image(image)
        self._processor_state = self.processor.set_image(image)
        # Fix: SDK's set_image uses image.shape[-2:] which gives (W, C)
        # for numpy (H, W, C) arrays instead of (H, W).
        h, w = image.shape[:2]
        self._processor_state["original_height"] = h
        self._processor_state["original_width"] = w

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
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8) * 255
        return mask

    def predict_text(self, prompt: str) -> np.ndarray | None:
        """Predict a binary mask from a text prompt.

        All detections above the confidence threshold are merged into a
        single binary mask (union).

        Args:
            prompt: Text description, e.g. "person", "car".

        Returns:
            Merged binary mask (H, W) uint8 (0 or 255), or None if no
            detections pass the threshold.
        """
        if self._processor_state is None:
            raise RuntimeError("Call set_image() before predict_text().")

        # Reset previous prompts so we get a clean text-only prediction
        self.processor.reset_all_prompts(self._processor_state)
        state = self.processor.set_text_prompt(prompt, self._processor_state)

        masks = state.get("masks")  # (N, 1, H, W) bool tensor
        if masks is None or masks.numel() == 0:
            return None

        # Union all detection masks into one binary mask
        merged = masks.any(dim=0).squeeze(0).cpu().numpy()  # (H, W) bool
        return (merged.astype(np.uint8)) * 255

    def reset(self) -> None:
        """Clear image embeddings for both predictors."""
        self.predictor.reset_predictor()
        self._processor_state = None
