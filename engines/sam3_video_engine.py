"""SAM3 video tracker wrapper for mask propagation."""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
from config import SAM3_CHECKPOINT, get_device  # must precede torch for MPS env var

import torch
from engines._sam3_deps import setup_sam3_deps

log = logging.getLogger(__name__)

setup_sam3_deps()


def _fix_plain_tensors_and_rope(module: torch.nn.Module, device: torch.device) -> None:
    """Post-process a model tree for MPS compatibility.

    1. Move plain tensor attributes (not params/buffers) to *device*.
       The SDK stores RoPE frequencies as plain ``self.freqs_cis``
       which ``nn.Module.to()`` ignores.
    2. Switch RoPEAttention modules to real-valued mode.  MPS does not
       support ``repeat()`` on complex tensors.
    """
    from sam3.sam.transformer import RoPEAttention

    param_and_buf = {id(p) for p in module.parameters()}
    param_and_buf |= {id(b) for b in module.buffers()}
    for sub in module.modules():
        for name in list(vars(sub)):
            val = getattr(sub, name)
            if isinstance(val, torch.Tensor) and id(val) not in param_and_buf:
                setattr(sub, name, val.to(device))

        # Force real-valued RoPE on MPS (complex ops unsupported)
        if isinstance(sub, RoPEAttention) and not sub.use_rope_real:
            sub.use_rope_real = True
            sub.freqs_cis_real = sub.freqs_cis.real
            sub.freqs_cis_imag = sub.freqs_cis.imag


class SAM3VideoEngine:
    """Wrapper around SAM3 tracker for bidirectional mask propagation.

    Usage:
        engine = SAM3VideoEngine()
        masks = engine.propagate(frames_dir, {0: mask_a, 10: mask_b})
    """

    def __init__(
        self,
        checkpoint: Path = SAM3_CHECKPOINT,
        device: torch.device | None = None,
    ) -> None:
        """Initialize SAM3 video tracker.

        Builds the standalone tracker module and loads weights from the
        unified SAM3 checkpoint (extracting ``tracker.*`` keys).

        Args:
            checkpoint: Path to SAM3 unified checkpoint file.
            device: Torch device. Auto-detected if None.
        """
        self.device = device or get_device()

        from sam3.model_builder import build_tracker

        log.info("Loading SAM3 Video Tracker from %s", checkpoint)
        self.tracker = build_tracker(
            apply_temporal_disambiguation=True, with_backbone=True,
        )

        # Load tracker weights + visual backbone from unified checkpoint.
        # The unified ckpt stores tracker weights under "tracker.*" and the
        # visual backbone under "detector.backbone.*".  The standalone
        # tracker (with_backbone=True) keeps its backbone at "backbone.*".
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        tracker_ckpt = {
            k.replace("tracker.", ""): v
            for k, v in ckpt.items()
            if k.startswith("tracker.")
        }
        # Map detector visual backbone → tracker backbone
        backbone_ckpt = {
            k.replace("detector.backbone.", "backbone."): v
            for k, v in ckpt.items()
            if k.startswith("detector.backbone.")
        }
        tracker_ckpt.update(backbone_ckpt)
        self.tracker.load_state_dict(tracker_ckpt, strict=False)
        self.tracker.eval()
        self.tracker.to(device=self.device)
        _fix_plain_tensors_and_rope(self.tracker, self.device)
        log.info("SAM3 Video Tracker ready on %s", self.device)

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

        # The SDK hardcodes storage_device=torch.device("cuda") when
        # offload_state_to_cpu=False.  We pass True to dodge the crash
        # during frame loading, then immediately switch to MPS.
        non_cuda = self.device.type != "cuda"
        state = self.tracker.init_state(
            video_path=str(frames_dir),
            offload_state_to_cpu=non_cuda,
            offload_video_to_cpu=non_cuda,
        )
        if non_cuda:
            state["storage_device"] = self.device
        num_frames = state["num_frames"]

        # Register all keyframe masks as obj_id=1
        for frame_idx, mask_np in keyframe_masks.items():
            mask_bool = torch.from_numpy(
                (mask_np > 127).astype(np.uint8)
            ).bool()
            self.tracker.add_new_mask(
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
        min_kf = min(keyframe_masks.keys())

        def _to_mask(video_res_masks: torch.Tensor) -> np.ndarray:
            """Convert tracker output (num_obj, 1, H, W) to (H, W) uint8."""
            return (
                (video_res_masks[0] > 0.0).cpu().numpy().squeeze(0) * 255
            ).astype(np.uint8)

        # Forward propagation from the earliest keyframe
        log.info("Forward propagation...")
        for frame_idx, obj_ids, _low, video_res, _scores in (
            self.tracker.propagate_in_video(
                state,
                start_frame_idx=min_kf,
                max_frame_num_to_track=None,
                reverse=False,
                propagate_preflight=True,
            )
        ):
            result[frame_idx] = _to_mask(video_res)
            completed += 1
            if progress_callback:
                progress_callback(completed / num_frames)

        # Reverse propagation: fill frames before the earliest keyframe
        log.info("Reverse propagation...")
        for frame_idx, obj_ids, _low, video_res, _scores in (
            self.tracker.propagate_in_video(
                state,
                start_frame_idx=min_kf,
                max_frame_num_to_track=None,
                reverse=True,
            )
        ):
            if frame_idx not in result:
                result[frame_idx] = _to_mask(video_res)
            completed += 1
            if progress_callback:
                progress_callback(min(1.0, completed / num_frames))

        log.info("Propagation complete: %d frames", len(result))
        return result
