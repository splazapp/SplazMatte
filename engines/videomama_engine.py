"""VideoMaMa video matting engine with batched inference and overlap blending."""

import sys
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision import transforms

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
            log.info("Using MPS device for VideoMaMa.")

        # Patch CUDA calls before importing the SDK (diffusers uses autocast)
        from engines._cuda_compat import patch_cuda_to_device
        patch_cuda_to_device(self.device)

        from pipeline_svd_mask import VideoInferencePipeline

        log.info("Loading VideoMaMa pipeline (SVD: %s, UNet: %s)", svd_path, unet_path)
        weight_dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.pipeline = VideoInferencePipeline(
            base_model_path=str(svd_path),
            unet_checkpoint_path=str(unet_path),
            device=str(self.device),
            weight_dtype=weight_dtype,
        )

        # The SDK forces CPU when CUDA is unavailable. Override to use
        # the actual device (e.g. MPS) and move all models there.
        if self.pipeline.device != self.device:
            log.info("Moving VideoMaMa models to %s...", self.device)
            self.pipeline.device = self.device
            self.pipeline.image_encoder.to(self.device, dtype=weight_dtype).eval()
            self.pipeline.vae.to(self.device, dtype=weight_dtype).eval()
            self.pipeline.unet.to(self.device, dtype=weight_dtype).eval()

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

        # Count total batches for progress (ceiling division)
        total_batches = max(1, -(-num_frames // stride))

        start = 0
        while start < num_frames:
            end = min(start + batch_size, num_frames)
            actual_len = end - start
            needs_padding = actual_len < batch_size

            # Build frame/mask lists for this batch
            batch_frames, batch_masks = self._load_batch(
                frames_dir, masks, start, end, batch_size,
            )

            # Build sub-batch progress reporter
            def _sub_progress(frac: float, _base=batches_done, _total=total_batches):
                if progress_callback is not None:
                    progress_callback((_base + frac) / _total)

            # Run inference
            batch_alphas = self._process_batch(
                batch_frames, batch_masks, seed, _sub_progress,
            )

            # Free intermediate tensors cached by MPS/CUDA
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

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

    def _offload(self, *models: torch.nn.Module) -> None:
        """Move models to CPU and free device cache (MPS only)."""
        for m in models:
            m.to("cpu")
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def _encode_to_latents(
        self, video_tensor: torch.Tensor, chunk_size: int = 8,
    ) -> torch.Tensor:
        """VAE encode with chunking (mirrors the existing chunked decode).

        Args:
            video_tensor: (B, F, C, H, W) tensor in [-1, 1].
            chunk_size: Frames per encoding chunk.

        Returns:
            (B, F, Cl, Hl, Wl) latent tensor, scaled by VAE factor.
        """
        vae = self.pipeline.vae
        b, f, c, h, w = video_tensor.shape
        frames = rearrange(video_tensor, "b f c h w -> (b f) c h w")
        chunks = []
        for i in range(0, frames.shape[0], chunk_size):
            chunk = vae.encode(frames[i : i + chunk_size]).latent_dist.sample()
            chunks.append(chunk)
        latents = torch.cat(chunks, dim=0)
        return rearrange(
            latents, "(b f) c h w -> b f c h w", f=f,
        ) * vae.config.scaling_factor

    @torch.inference_mode()
    def _process_batch(
        self,
        frames_np: list[np.ndarray],
        masks_np: list[np.ndarray],
        seed: int,
        stage_callback: Callable[[float], None] | None = None,
    ) -> list[np.ndarray]:
        """Run VideoMaMa inference on a single batch.

        Inlines the SDK pipeline.run() logic so we can offload models
        between stages and free intermediate tensors on MPS.

        Args:
            frames_np: List of (H, W, 3) uint8 RGB frames.
            masks_np: List of (H, W) uint8 masks.
            seed: Random seed.
            stage_callback: Called with fraction [0, 1] after each stage.

        Returns:
            List of (H, W) uint8 alpha arrays at original resolution.
        """
        def _report(frac: float) -> None:
            if stage_callback is not None:
                stage_callback(frac)

        orig_h, orig_w = frames_np[0].shape[:2]
        pipe = self.pipeline
        dtype = pipe.weight_dtype
        is_mps = self.device.type == "mps"

        # --- 1. PIL → tensor ---
        cond_pils = [
            Image.fromarray(f).resize(
                (_MODEL_W, _MODEL_H), Image.Resampling.BILINEAR,
            )
            for f in frames_np
        ]
        mask_pils = [
            Image.fromarray(m, mode="L").resize(
                (_MODEL_W, _MODEL_H), Image.Resampling.BILINEAR,
            )
            for m in masks_np
        ]
        cond_video = pipe._pil_to_tensor(cond_pils).to(self.device)
        mask_video = pipe._pil_to_tensor(mask_pils).to(self.device)
        if mask_video.shape[2] != 3:
            mask_video = mask_video.repeat(1, 1, 3, 1, 1)

        # --- 2. CLIP encode first frame ---
        first_frame = cond_video[:, 0, :, :, :]
        clip_input = pipe._resize_with_antialiasing(first_frame, (224, 224))
        clip_input = ((clip_input + 1.0) / 2.0).clamp(0, 1)
        pixel_values = pipe.feature_extractor(
            images=clip_input, return_tensors="pt",
        ).pixel_values
        image_embeddings = pipe.image_encoder(
            pixel_values.to(self.device, dtype=dtype),
        ).image_embeds
        encoder_hidden = torch.zeros_like(image_embeddings).unsqueeze(1)
        del first_frame, clip_input, pixel_values, image_embeddings
        _report(0.1)

        # --- 3. Offload CLIP (no longer needed) ---
        if is_mps:
            self._offload(pipe.image_encoder)

        # --- 4. VAE encode (chunked) ---
        cond_latents = self._encode_to_latents(cond_video.to(dtype))
        cond_latents = cond_latents / pipe.vae.config.scaling_factor
        mask_latents = self._encode_to_latents(mask_video.to(dtype))
        mask_latents = mask_latents / pipe.vae.config.scaling_factor
        del cond_video, mask_video
        if is_mps:
            torch.mps.empty_cache()

        _report(0.3)

        # --- 5. Offload VAE (not needed during UNet) ---
        if is_mps:
            self._offload(pipe.vae)

        # --- 6. UNet single-step inference ---
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noisy_latents = torch.randn(
            cond_latents.shape, generator=generator,
            device=self.device, dtype=dtype,
        )
        timesteps = torch.full((1,), 1.0, device=self.device, dtype=torch.long)
        added_time_ids = pipe._get_add_time_ids(
            fps=7, motion_bucket_id=127, noise_aug_strength=0.0, batch_size=1,
        )
        unet_input = torch.cat(
            [noisy_latents, cond_latents, mask_latents], dim=2,
        )
        del noisy_latents, cond_latents, mask_latents
        if is_mps:
            torch.mps.empty_cache()

        pred_latents = pipe.unet(
            unet_input, timesteps, encoder_hidden,
            added_time_ids=added_time_ids,
        ).sample
        del unet_input, encoder_hidden, timesteps, added_time_ids
        if is_mps:
            torch.mps.empty_cache()
        _report(0.7)

        # --- 7. Offload UNet, bring VAE back for decode ---
        if is_mps:
            self._offload(pipe.unet)
            pipe.vae.to(self.device, dtype=dtype)

        # --- 8. VAE decode (chunked, same as SDK) ---
        pred_latents = (
            (1 / pipe.vae.config.scaling_factor) * pred_latents.squeeze(0)
        )
        decoded_chunks = []
        for i in range(0, pred_latents.shape[0], 8):
            chunk = pred_latents[i : i + 8]
            decoded_chunks.append(
                pipe.vae.decode(chunk, num_frames=chunk.shape[0]).sample,
            )
        video_tensor = torch.cat(decoded_chunks, dim=0)
        del pred_latents, decoded_chunks
        video_tensor = (
            (video_tensor / 2.0 + 0.5).clamp(0, 1)
            .mean(dim=1, keepdim=True)
            .repeat(1, 3, 1, 1)
        )

        _report(0.9)

        # --- 9. Restore all models to device for next batch ---
        if is_mps:
            pipe.image_encoder.to(self.device, dtype=dtype)
            pipe.unet.to(self.device, dtype=dtype)

        # --- 10. To PIL → resize → alpha ---
        # Move to CPU and free device memory before PIL conversion
        video_tensor = video_tensor.cpu().float()
        if is_mps:
            torch.mps.empty_cache()

        alphas = []
        for frame_tensor in video_tensor:
            pil_img = transforms.ToPILImage()(frame_tensor)
            resized = pil_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            alphas.append(np.array(resized.convert("L")))

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
