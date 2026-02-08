"""VideoMaMa video matting engine with batched inference and overlap blending."""

import sys
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from config import (
    SDKS_DIR,
    VIDEOMAMA_BATCH_SIZE,
    VIDEOMAMA_OVERLAP,
    VIDEOMAMA_SEED,
    VIDEOMAMA_SVD_PATH,
    VIDEOMAMA_UNET_PATH,
    get_device,
)
from pipeline.video_io import load_frame

log = logging.getLogger(__name__)

# Add VideoMaMa SDK to path for pipeline imports
_videomama_root = SDKS_DIR / "VideoMaMa"
if str(_videomama_root) not in sys.path:
    sys.path.insert(0, str(_videomama_root))

# Model input resolution (matches SVD training)
_MODEL_W, _MODEL_H = 1024, 576


class VideoMaMaEngine:
    """VideoMaMa matting engine with batched inference and overlap blending.

    Usage:
        engine = VideoMaMaEngine()
        alphas, foregrounds = engine.process(
            frames_dir, masks, progress_callback=cb,
        )
    """

    def __init__(
        self,
        svd_path: Path = VIDEOMAMA_SVD_PATH,
        unet_path: Path = VIDEOMAMA_UNET_PATH,
        device: torch.device | None = None,
    ) -> None:
        """Load the VideoMaMa pipeline.

        Args:
            svd_path: Path to Stable Video Diffusion base model.
            unet_path: Path to fine-tuned VideoMaMa UNet checkpoint.
            device: Torch device. Auto-detected if None.
        """
        self.device = device or get_device()

        if self.device.type == "mps":
            log.info("Using MPS device for VideoMaMa (fp32 mode).")

        # Patch CUDA calls before importing the SDK (diffusers uses autocast)
        from engines._cuda_compat import patch_cuda_to_device
        patch_cuda_to_device(self.device)

        from pipeline_svd_mask import VideoInferencePipeline

        log.info("Loading VideoMaMa pipeline (SVD: %s, UNet: %s)", svd_path, unet_path)
        self.pipeline = VideoInferencePipeline(
            base_model_path=str(svd_path),
            unet_checkpoint_path=str(unet_path),
            device=str(self.device),
            weight_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        log.info("VideoMaMa engine ready on %s", self.device)

    def process(
        self,
        frames_dir: Path,
        masks: dict[int, np.ndarray],
        batch_size: int = VIDEOMAMA_BATCH_SIZE,
        overlap: int = VIDEOMAMA_OVERLAP,
        seed: int = VIDEOMAMA_SEED,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run batched matting with overlap blending.

        Args:
            frames_dir: Directory of JPEG frames (000001.jpg, ...).
            masks: Per-frame masks {frame_idx: (H, W) uint8 0/255}.
            batch_size: Frames per inference batch.
            overlap: Overlap frames between adjacent batches for blending.
            seed: Random seed for reproducibility.
            progress_callback: Called with fraction [0, 1] for progress.

        Returns:
            Tuple of (alphas, foregrounds):
                alphas: (T, H, W, 1) uint8 alpha mattes.
                foregrounds: (T, H, W, 3) uint8 green-screen composites.
        """
        num_frames = len(masks)
        if num_frames == 0:
            raise ValueError("At least one frame mask is required.")
        if overlap >= batch_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than batch_size ({batch_size})."
            )

        stride = batch_size - overlap
        all_alphas: list[np.ndarray | None] = [None] * num_frames
        prev_tail: list[np.ndarray] | None = None
        batches_done = 0

        # Count total batches for progress
        if num_frames <= batch_size:
            total_batches = 1
        else:
            total_batches = 1 + (num_frames - batch_size + stride - 1) // stride

        start = 0
        while start < num_frames:
            end = min(start + batch_size, num_frames)
            actual_len = end - start
            needs_padding = actual_len < batch_size

            # Build frame/mask lists for this batch
            batch_frames, batch_masks = self._load_batch(
                frames_dir, masks, start, end, batch_size,
            )

            # Run inference
            batch_alphas = self._process_batch(batch_frames, batch_masks, seed)

            # Trim padding if needed
            if needs_padding:
                batch_alphas = batch_alphas[:actual_len]

            # Overlap blending with previous batch
            if prev_tail is not None and overlap > 0:
                blend_len = min(overlap, len(batch_alphas))
                for i in range(blend_len):
                    w = (i + 1) / (overlap + 1)
                    blended = (
                        prev_tail[i].astype(np.float32) * (1 - w)
                        + batch_alphas[i].astype(np.float32) * w
                    )
                    batch_alphas[i] = np.clip(blended, 0, 255).astype(np.uint8)

            # Store results
            for i, alpha in enumerate(batch_alphas):
                all_alphas[start + i] = alpha

            # Save tail for next batch's blending
            if overlap > 0 and end < num_frames and len(batch_alphas) >= overlap:
                prev_tail = batch_alphas[-overlap:]
            else:
                prev_tail = None

            batches_done += 1
            if progress_callback is not None:
                progress_callback(batches_done / total_batches)

            start += stride

        # Composite foregrounds
        alphas_stack = np.stack(all_alphas)  # (T, H, W)
        alphas_out = alphas_stack[..., np.newaxis]  # (T, H, W, 1)
        foregrounds = self._composite_foregrounds(frames_dir, alphas_stack, num_frames)

        return alphas_out, foregrounds

    def _load_batch(
        self,
        frames_dir: Path,
        masks: dict[int, np.ndarray],
        start: int,
        end: int,
        batch_size: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Load frames and masks for a batch, padding if needed.

        Args:
            frames_dir: Frame directory.
            masks: Per-frame masks dict.
            start: Start frame index (inclusive).
            end: End frame index (exclusive).
            batch_size: Target batch size (pad to this if short).

        Returns:
            Tuple of (frames_list, masks_list) as numpy arrays.
        """
        batch_frames = []
        batch_masks = []
        for idx in range(start, end):
            batch_frames.append(load_frame(frames_dir, idx))
            batch_masks.append(masks[idx])

        # Pad with last frame/mask if batch is short
        while len(batch_frames) < batch_size:
            batch_frames.append(batch_frames[-1])
            batch_masks.append(batch_masks[-1])

        return batch_frames, batch_masks

    def _process_batch(
        self,
        frames_np: list[np.ndarray],
        masks_np: list[np.ndarray],
        seed: int,
    ) -> list[np.ndarray]:
        """Run VideoMaMa inference on a single batch.

        Args:
            frames_np: List of (H, W, 3) uint8 RGB frames.
            masks_np: List of (H, W) uint8 masks.
            seed: Random seed.

        Returns:
            List of (H, W) uint8 alpha arrays at original resolution.
        """
        orig_h, orig_w = frames_np[0].shape[:2]

        # Convert to PIL and resize to model resolution
        cond_frames = [
            Image.fromarray(f).resize((_MODEL_W, _MODEL_H), Image.Resampling.BILINEAR)
            for f in frames_np
        ]
        mask_frames = [
            Image.fromarray(m, mode="L").resize(
                (_MODEL_W, _MODEL_H), Image.Resampling.BILINEAR,
            )
            for m in masks_np
        ]

        # Run pipeline
        output_pils = self.pipeline.run(
            cond_frames=cond_frames,
            mask_frames=mask_frames,
            seed=seed,
        )

        # Resize back and extract alpha (grayscale channel)
        alphas = []
        for pil_img in output_pils:
            resized = pil_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            alpha = np.array(resized.convert("L"))
            alphas.append(alpha)

        return alphas

    def _composite_foregrounds(
        self,
        frames_dir: Path,
        alphas: np.ndarray,
        num_frames: int,
    ) -> np.ndarray:
        """Composite frames onto green background using alpha mattes.

        Args:
            frames_dir: Frame directory.
            alphas: (T, H, W) uint8 alpha mattes.
            num_frames: Number of frames.

        Returns:
            (T, H, W, 3) uint8 green-screen composites.
        """
        bgr = np.array([120, 255, 155], dtype=np.float32).reshape(1, 1, 3) / 255
        composites = []

        for idx in range(num_frames):
            frame = load_frame(frames_dir, idx)
            alpha_f = alphas[idx].astype(np.float32)[..., np.newaxis] / 255.0
            comp = frame / 255.0 * alpha_f + bgr * (1 - alpha_f)
            composites.append(np.clip(comp * 255, 0, 255).astype(np.uint8))

        return np.stack(composites)
