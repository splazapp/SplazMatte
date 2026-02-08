"""Redirect hardcoded CUDA calls to the actual compute device.

The SAM3 SDK assumes CUDA everywhere (``tensor.cuda()``, ``device="cuda"``).
On Apple-Silicon Macs the real device is ``mps``; on CPU-only boxes it is
``cpu``.  This module monkey-patches PyTorch **once** so that all those
SDK calls land on the correct device automatically.

Call ``patch_cuda_to_device()`` before any ``from sam3 import …``.
"""

import functools
import logging
from typing import Any

import torch

log = logging.getLogger(__name__)

_patched = False


def patch_cuda_to_device(device: torch.device) -> None:
    """Apply global monkey-patches if the real device is not CUDA.

    Safe to call multiple times — only the first call takes effect.

    Args:
        device: The actual device to use (e.g. ``mps`` or ``cpu``).
    """
    global _patched
    if _patched or device.type == "cuda":
        return
    _patched = True
    log.info("Patching CUDA → %s for SAM3 SDK compatibility", device)

    # --- 1. Tensor.cuda() → Tensor.to(device) ---
    def _tensor_cuda(self: torch.Tensor, *_args: Any, **_kw: Any) -> torch.Tensor:
        return self.to(device)

    torch.Tensor.cuda = _tensor_cuda  # type: ignore[assignment]

    # --- 2. Module.cuda() → Module.to(device) ---
    _orig_module_cuda = torch.nn.Module.cuda

    def _module_cuda(self: torch.nn.Module, *_args: Any, **_kw: Any) -> torch.nn.Module:
        return self.to(device)

    torch.nn.Module.cuda = _module_cuda  # type: ignore[assignment]

    # --- 3. Factory functions: replace device="cuda" in kwargs ---
    _factories = [
        "zeros", "ones", "empty", "randn", "rand", "tensor", "full",
        "arange", "linspace", "logspace", "eye", "as_tensor",
    ]

    def _wrap_factory(original):
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
            if kwargs.get("device") in ("cuda", "cuda:0"):
                kwargs["device"] = device
            return original(*args, **kwargs)
        return wrapper

    for name in _factories:
        original = getattr(torch, name, None)
        if original is not None:
            setattr(torch, name, _wrap_factory(original))

    # --- 4. torch.jit.script → no-op (eager mode) ---
    # torch.jit.script recursively compiles functions and tries to
    # inspect.getsourcelines() on every callable it encounters.  Our
    # functools.wraps wrappers expose __wrapped__ pointing to C builtins
    # (e.g. the original torch.arange), which have no Python source →
    # TypeError.  JIT scripting is a CUDA performance optimisation; on
    # MPS / CPU eager execution is fine.
    _orig_jit_script = torch.jit.script

    def _eager_jit_script(obj, *args, **kwargs):
        """Return *obj* unchanged — skip JIT compilation on non-CUDA."""
        return obj

    torch.jit.script = _eager_jit_script  # type: ignore[assignment]

    # --- 5. torch.autocast: redirect device_type="cuda" → "cpu" ---
    # SAM3 uses torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # in several places. MPS does not support autocast; CPU autocast with
    # bfloat16 is supported and is the safe fallback.
    _OrigAutocast = torch.autocast

    class _PatchedAutocast(_OrigAutocast):  # type: ignore[misc]
        """Autocast wrapper that silently remaps ``cuda`` → ``cpu``."""

        def __init__(self, device_type: str, *args: Any, **kwargs: Any):
            if device_type in ("cuda", "cuda:0"):
                device_type = "cpu"
            super().__init__(device_type, *args, **kwargs)

    torch.autocast = _PatchedAutocast  # type: ignore[misc]

    # --- 6. F.grid_sample: fall back to CPU on MPS ---
    # MPS has a bug where grid_sample crashes with "Placeholder tensor is
    # empty!" on zero-element inputs.  PYTORCH_ENABLE_MPS_FALLBACK only
    # catches NotImplementedError, not internal assertion failures.
    if device.type == "mps":
        _orig_grid_sample = torch.nn.functional.grid_sample

        @functools.wraps(_orig_grid_sample)
        def _safe_grid_sample(
            input: torch.Tensor, grid: torch.Tensor, **kwargs: Any,
        ) -> torch.Tensor:
            if input.numel() == 0 or grid.numel() == 0:
                return _orig_grid_sample(
                    input.cpu(), grid.cpu(), **kwargs,
                ).to(device)
            return _orig_grid_sample(input, grid, **kwargs)

        torch.nn.functional.grid_sample = _safe_grid_sample  # type: ignore[assignment]

    # --- 7. Tensor.pin_memory() → no-op ---
    # pin_memory() is a CUDA optimisation for async CPU→GPU copies.
    # Not available on MPS; just return self.
    def _noop_pin_memory(self: torch.Tensor, *_a: Any, **_kw: Any) -> torch.Tensor:
        return self

    torch.Tensor.pin_memory = _noop_pin_memory  # type: ignore[assignment]
