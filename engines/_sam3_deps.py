"""One-time setup for SAM3 SDK: sys-path, module stubs, and CUDA compat.

Call ``setup_sam3_deps()`` at module level in any engine that imports from
the SAM3 SDK.  Safe to call multiple times — only the first call acts.
"""

import sys
import types

from config import SDKS_DIR, get_device

_done = False


def setup_sam3_deps() -> None:
    """Register sys-path, stub missing packages, and patch CUDA calls."""
    global _done
    if _done:
        return
    _done = True

    # --- 1. Add SAM3 SDK to sys.path ---
    sam3_root = str(SDKS_DIR / "sam3")
    if sam3_root not in sys.path:
        sys.path.insert(0, sam3_root)

    # --- 2. Stub training-only deps (decord, pycocotools, iopath) ---
    _stub_decord()
    _stub_pycocotools()
    _stub_iopath()

    # --- 3. Redirect hardcoded .cuda() / device="cuda" to real device ---
    from engines._cuda_compat import patch_cuda_to_device

    patch_cuda_to_device(get_device())


# ------------------------------------------------------------------
# Module stubs
# ------------------------------------------------------------------

def _stub_decord() -> None:
    if "decord" in sys.modules:
        return
    mod = types.ModuleType("decord")
    mod.cpu = None  # type: ignore[attr-defined]
    mod.VideoReader = None  # type: ignore[attr-defined]
    sys.modules["decord"] = mod


def _stub_pycocotools() -> None:
    if "pycocotools" in sys.modules:
        return
    pkg = types.ModuleType("pycocotools")
    mask = types.ModuleType("pycocotools.mask")
    sys.modules["pycocotools"] = pkg
    sys.modules["pycocotools.mask"] = mask


def _stub_iopath() -> None:
    """Functional stub — tokenizer calls ``g_pathmgr.open()`` at runtime."""
    if "iopath" in sys.modules:
        return

    class _PathManager:
        @staticmethod
        def open(path, mode="r", **_kw):
            return open(path, mode)  # noqa: SIM115

    pkg = types.ModuleType("iopath")
    common = types.ModuleType("iopath.common")
    file_io = types.ModuleType("iopath.common.file_io")
    file_io.g_pathmgr = _PathManager()  # type: ignore[attr-defined]
    pkg.common = common  # type: ignore[attr-defined]
    common.file_io = file_io  # type: ignore[attr-defined]
    sys.modules["iopath"] = pkg
    sys.modules["iopath.common"] = common
    sys.modules["iopath.common.file_io"] = file_io
