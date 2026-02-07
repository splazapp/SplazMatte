"""MatAnyone video matting engine with multi-keyframe mask injection."""

import sys
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import MATANYONE_CHECKPOINT, SDKS_DIR, get_device

log = logging.getLogger(__name__)

# Add MatAnyone SDK to path for model/config imports
_matanyone_root = SDKS_DIR / "MatAnyone"
if str(_matanyone_root) not in sys.path:
    sys.path.insert(0, str(_matanyone_root))


def _gen_dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate a binary mask. Adapted from MatAnyone inference_utils."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    fg = np.not_equal(mask, 0).astype(np.float32)
    return (cv2.dilate(fg, kernel, iterations=1) * 255).astype(np.float32)


def _gen_erode(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Erode a binary mask. Adapted from MatAnyone inference_utils."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    fg = np.equal(mask, 255).astype(np.float32)
    return (cv2.erode(fg, kernel, iterations=1) * 255).astype(np.float32)


class MatAnyoneEngine:
    """MatAnyone wrapper supporting keyframe mask injection at arbitrary frames.

    Usage:
        engine = MatAnyoneEngine()
        alphas, foregrounds = engine.process(
            frames_tensor, {0: mask0, 30: mask30}
        )
    """

    def __init__(
        self,
        checkpoint: Path = MATANYONE_CHECKPOINT,
        device: torch.device | None = None,
    ) -> None:
        """Load the MatAnyone model.

        Args:
            checkpoint: Path to matanyone.pth weights.
            device: Torch device. Auto-detected if None.
        """
        self.device = device or get_device()

        # Clear Hydra global state to avoid conflict with SAM2's initialization.
        # SAM2's __init__.py calls initialize_config_module("sam2") and
        # MatAnyone's get_matanyone_model calls initialize() without checking.
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        from matanyone.utils.get_default_model import get_matanyone_model

        log.info("Loading MatAnyone model from %s", checkpoint)
        self.model = get_matanyone_model(str(checkpoint), self.device)
        log.info("MatAnyone engine ready on %s", self.device)

    @torch.inference_mode()
    def process(
        self,
        frames: torch.Tensor,
        keyframe_masks: dict[int, np.ndarray],
        erode: int = 10,
        dilate: int = 10,
        warmup: int = 10,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run matting with keyframe masks injected at specified frames.

        Args:
            frames: (T, C, H, W) float tensor, pixel values in [0, 255].
            keyframe_masks: {frame_idx: mask_array (H, W) uint8 0/255}.
            erode: Erosion kernel size for masks.
            dilate: Dilation kernel size for masks.
            warmup: Number of warmup repetitions for the first keyframe.
            progress_callback: Called with fraction [0, 1] for progress.

        Returns:
            Tuple of (alphas, foregrounds):
                alphas: (T', H, W, 1) uint8 alpha mattes.
                foregrounds: (T', H, W, 3) uint8 green-screen composites.
        """
        from matanyone.inference.inference_core import InferenceCore

        processor = InferenceCore(self.model, cfg=self.model.cfg)

        sorted_kf_indices = sorted(keyframe_masks.keys())
        if not sorted_kf_indices:
            raise ValueError("At least one keyframe mask is required.")

        first_kf = sorted_kf_indices[0]

        # Prepare masks: apply erode/dilate and convert to tensors
        kf_masks_tensor: dict[int, torch.Tensor] = {}
        for idx, mask_np in keyframe_masks.items():
            m = mask_np.astype(np.float32)
            if dilate > 0:
                m = _gen_dilate(m, dilate)
            if erode > 0:
                m = _gen_erode(m, erode)
            kf_masks_tensor[idx] = torch.from_numpy(m).float().to(self.device)

        # Build the frame sequence: prepend warmup copies of the first keyframe
        first_kf_frame = frames[first_kf].unsqueeze(0)
        warmup_frames = first_kf_frame.repeat(warmup, 1, 1, 1)
        # Only process from first_kf onward
        active_frames = frames[first_kf:]
        all_frames = torch.cat([warmup_frames, active_frames], dim=0).float()
        total_len = all_frames.shape[0]

        # Green background for foreground composite
        bgr = (
            np.array([120, 255, 155], dtype=np.float32) / 255
        ).reshape(1, 1, 3)

        objects = [1]
        alphas_list: list[np.ndarray] = []
        fgrs_list: list[np.ndarray] = []

        for ti in tqdm(range(total_len), desc="抠像推理", unit="帧"):
            image = all_frames[ti]
            image_np = image.permute(1, 2, 0).numpy()
            image_norm = (image / 255.0).float().to(self.device)

            # Map ti back to the original frame index
            # ti in [0, warmup) -> warmup region (first keyframe)
            # ti >= warmup -> original_idx = first_kf + (ti - warmup)
            original_idx = first_kf + (ti - warmup) if ti >= warmup else None

            # Check if this is a keyframe (in original frame indices)
            is_keyframe = original_idx is not None and original_idx in kf_masks_tensor

            if ti == 0:
                # First warmup frame: encode the first keyframe's mask
                mask = kf_masks_tensor[first_kf]
                processor.step(image_norm, mask, objects=objects)
                output_prob = processor.step(image_norm, first_frame_pred=True)
            elif ti <= warmup:
                # Warmup region: keep re-initializing (matches reference behavior)
                output_prob = processor.step(image_norm, first_frame_pred=True)
            elif is_keyframe and original_idx != first_kf:
                # Subsequent keyframe: inject new mask
                mask = kf_masks_tensor[original_idx]
                processor.step(image_norm, mask, objects=objects)
                output_prob = processor.step(
                    image_norm, first_frame_pred=True
                )
            else:
                # Normal propagation frame
                output_prob = processor.step(image_norm)

            mask_out = processor.output_prob_to_mask(output_prob)
            pha = mask_out.unsqueeze(2).cpu().numpy()  # (H, W, 1)
            com = image_np / 255.0 * pha + bgr * (1 - pha)

            # Only collect frames after warmup
            if ti >= warmup:
                pha_u8 = np.clip(pha * 255, 0, 255).astype(np.uint8)
                com_u8 = np.clip(com * 255, 0, 255).astype(np.uint8)
                alphas_list.append(pha_u8)
                fgrs_list.append(com_u8)

            if progress_callback is not None:
                progress_callback((ti + 1) / total_len)

        return np.stack(alphas_list), np.stack(fgrs_list)
