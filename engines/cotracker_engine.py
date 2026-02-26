"""CoTracker3 Online/Offline mode wrapper for point tracking in videos."""

import logging
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    COTRACKER_CHECKPOINT,
    COTRACKER_OFFLINE_CHECKPOINT,
    COTRACKER_OFFLINE_WINDOW_LEN,
    COTRACKER_USE_CPU_ON_MPS,
    COTRACKER_WINDOW_LEN,
    SDKS_DIR,
    get_device,
)

log = logging.getLogger(__name__)


class TrackingCancelledError(Exception):
    """Raised when tracking is cancelled via cancel_event."""

_cotracker_root = SDKS_DIR / "co-tracker"
if str(_cotracker_root) not in sys.path:
    sys.path.insert(0, str(_cotracker_root))


def _get_points_on_a_grid(
    size: int,
    extent: tuple[float, float],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a grid of points covering a rectangular region.

    Args:
        size: Grid size (size x size points).
        extent: (H, W) of the region.
        device: Torch device.

    Returns:
        Tensor of shape (1, size*size, 2) with (x, y) coordinates.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    center = [extent[0] / 2, extent[1] / 2]
    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


class CoTrackerEngine:
    """CoTracker3 Online/Offline point tracking engine.

    Usage:
        engine = CoTrackerEngine()
        tracks, visibility = engine.track(
            video,           # (T, H, W, 3) RGB uint8
            queries=points,  # (N, 3) format [t, x, y] or None for grid
            grid_size=15,    # used if queries is None
            backward_tracking=False,  # True for bidirectional tracking
        )
    """

    def __init__(
        self,
        checkpoint: Path = COTRACKER_CHECKPOINT,
        window_len: int = COTRACKER_WINDOW_LEN,
        device: torch.device | None = None,
    ) -> None:
        """Initialize CoTracker3 Online model (Offline loaded lazily when needed).

        Args:
            checkpoint: Path to Online model checkpoint.
            window_len: Sliding window length.
            device: Torch device. Auto-detected if None.
        """
        _device = device or get_device()
        if COTRACKER_USE_CPU_ON_MPS and _device.type == "mps":
            self.device = torch.device("cpu")
            log.info("CoTracker: using CPU (MPS lacks grid_sampler_3d support)")
        else:
            self.device = _device
        self.window_len = window_len
        self._online_lock = threading.Lock()
        self._online_model = None
        self._offline_predictor = None

        from cotracker.models.build_cotracker import build_cotracker

        log.info("Loading CoTracker3 Online model from %s", checkpoint)
        self._online_model = build_cotracker(
            checkpoint=str(checkpoint),
            offline=False,
            window_len=window_len,
            v2=False,
        )
        self._online_model.to(self.device)
        self._online_model.eval()

        self.interp_shape = self._online_model.model_resolution
        self.step = self.window_len // 2
        self.support_grid_size = 6
        log.info(
            "CoTracker3 engine ready on %s (resolution=%s, step=%d)",
            self.device,
            self.interp_shape,
            self.step,
        )

    def _get_offline_predictor(self):
        """Lazy-load Offline predictor for backward tracking."""
        if self._offline_predictor is None:
            from cotracker.predictor import CoTrackerPredictor

            log.info("Loading CoTracker3 Offline model for backward tracking...")
            self._offline_predictor = CoTrackerPredictor(
                checkpoint=str(COTRACKER_OFFLINE_CHECKPOINT),
                offline=True,
                window_len=COTRACKER_OFFLINE_WINDOW_LEN,
                v2=False,
            )
            self._offline_predictor.to(self.device)
            self._offline_predictor.eval()
        return self._offline_predictor

    @torch.no_grad()
    def track(
        self,
        video: np.ndarray,
        queries: np.ndarray | None = None,
        grid_size: int = 15,
        backward_tracking: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track points through a video.

        Args:
            video: RGB video array of shape (T, H, W, 3), uint8.
            queries: Optional query points of shape (N, 3) in [t, x, y] format.
                If None, a grid of points is generated.
            grid_size: Grid size when queries is None. Default 15 means 15x15=225 points.
            backward_tracking: If True, track both forward and backward from query frames.
            progress_callback: Optional callback(fraction, description).
            cancel_event: Optional threading.Event; if set, tracking is cancelled.

        Returns:
            tracks: Point coordinates of shape (N, T, 2) as float32.
            visibility: Visibility mask of shape (N, T) as bool.

        Raises:
            CancelledError: If cancel_event is set during tracking.
        """
        if backward_tracking:
            return self._track_offline(video, queries, grid_size, progress_callback, cancel_event)

        return self._track_online(video, queries, grid_size, progress_callback, cancel_event)

    def _track_offline(
        self,
        video: np.ndarray,
        queries: np.ndarray | None,
        grid_size: int,
        progress_callback: Callable[[float, str], None] | None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Offline mode with backward tracking support."""
        T, H, W, _ = video.shape
        log.info("Tracking video (Offline, backward): %d frames, %dx%d", T, H, W)

        if cancel_event and cancel_event.is_set():
            raise TrackingCancelledError()

        if progress_callback:
            progress_callback(0.1, "加载 Offline 模型...")

        predictor = self._get_offline_predictor()

        video_tensor = (
            torch.from_numpy(video)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        if queries is not None:
            queries_tensor = torch.from_numpy(queries).float().unsqueeze(0).to(self.device)
            N = queries.shape[0]
            if progress_callback:
                progress_callback(0.3, "追踪中（双向）...")
            tracks, visibilities = predictor(
                video_tensor,
                queries=queries_tensor,
                grid_size=0,
                backward_tracking=True,
            )
        else:
            N = grid_size * grid_size
            if progress_callback:
                progress_callback(0.3, "追踪中（双向网格）...")
            tracks, visibilities = predictor(
                video_tensor,
                queries=None,
                grid_size=grid_size,
                grid_query_frame=0,
                backward_tracking=True,
            )

        tracks = tracks[0].permute(1, 0, 2).cpu().numpy()
        visibilities = visibilities[0].permute(1, 0).cpu().numpy()

        if progress_callback:
            progress_callback(1.0, "完成")

        log.info("Tracking complete (backward): %d points, %d frames", tracks.shape[0], tracks.shape[1])
        return tracks, visibilities

    def _track_online(
        self,
        video: np.ndarray,
        queries: np.ndarray | None,
        grid_size: int,
        progress_callback: Callable[[float, str], None] | None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Online mode (forward only)."""
        T, H, W, _ = video.shape
        log.info("Tracking video (Online): %d frames, %dx%d", T, H, W)

        video_tensor = (
            torch.from_numpy(video)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        if queries is not None:
            queries_tensor = torch.from_numpy(queries).float().unsqueeze(0).to(self.device)
            queries_tensor[:, :, 1:] *= queries_tensor.new_tensor([
                (self.interp_shape[1] - 1) / (W - 1),
                (self.interp_shape[0] - 1) / (H - 1),
            ])
            add_support_grid = True
            N = queries.shape[0]
        else:
            grid_pts = _get_points_on_a_grid(grid_size, self.interp_shape, device=self.device)
            queries_tensor = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts],
                dim=2,
            )
            add_support_grid = False
            N = grid_size * grid_size

        if add_support_grid:
            support_pts = _get_points_on_a_grid(
                self.support_grid_size, self.interp_shape, device=self.device
            )
            support_pts = torch.cat(
                [torch.zeros_like(support_pts[:, :, :1]), support_pts],
                dim=2,
            )
            queries_tensor = torch.cat([queries_tensor, support_pts], dim=1)

        with self._online_lock:
            self._online_model.init_video_online_processing()

            total_steps = max(1, (T - self.step * 2) // self.step + 1)
            step_count = 0

            tracks = None
            visibility = None

            for ind in range(0, T - self.step, self.step):
                if cancel_event and cancel_event.is_set():
                    raise TrackingCancelledError()

                end_ind = min(ind + self.step * 2, T)
                chunk = video_tensor[:, ind:end_ind]

                B, chunk_T, C, chunk_H, chunk_W = chunk.shape
                chunk = chunk.reshape(B * chunk_T, C, chunk_H, chunk_W)
                chunk = F.interpolate(
                    chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
                )
                chunk = chunk.reshape(B, chunk_T, 3, self.interp_shape[0], self.interp_shape[1])

                pred_tracks, pred_vis, pred_conf, _ = self._online_model(
                    video=chunk, queries=queries_tensor, iters=6, is_online=True
                )

                pred_vis = pred_vis * pred_conf
                pred_vis = pred_vis > 0.6

                pred_tracks = pred_tracks * pred_tracks.new_tensor([
                    (W - 1) / (self.interp_shape[1] - 1),
                    (H - 1) / (self.interp_shape[0] - 1),
                ])

                tracks = pred_tracks
                visibility = pred_vis

                step_count += 1
                if progress_callback:
                    frac = min(step_count / total_steps, 1.0)
                    progress_callback(frac, f"追踪中 {step_count}/{total_steps}")

            if add_support_grid:
                tracks = tracks[:, :, :N]
                visibility = visibility[:, :, :N]

            tracks = tracks[0].permute(1, 0, 2).cpu().numpy()
            visibility = visibility[0].permute(1, 0).cpu().numpy()

        log.info("Tracking complete: %d points, %d frames", tracks.shape[0], tracks.shape[1])
        return tracks, visibility
