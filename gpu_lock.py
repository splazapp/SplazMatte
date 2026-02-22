"""GPU lock for multi-user access control.

Implements a global mutex to ensure only one user can run GPU-intensive
operations (SAM inference, propagation, matting) at a time.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPULockInfo:
    """Information about the current GPU lock holder."""

    holder_id: str
    holder_name: str
    operation: str
    start_time: float = field(default_factory=time.time)

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time


_gpu_lock = threading.Lock()
_lock_info: Optional[GPULockInfo] = None
_lock_guard = threading.Lock()


def try_acquire_gpu(user_id: str, user_name: str, operation: str) -> tuple[bool, str]:
    """Try to acquire the GPU lock.

    Args:
        user_id: Unique identifier for the user (from browser session).
        user_name: Display name for the user.
        operation: Description of the operation (e.g., "SAM 标注", "传播", "抠像").

    Returns:
        (success, message) tuple. If success is False, message describes who
        is currently holding the lock.
    """
    global _lock_info

    with _lock_guard:
        if _lock_info is not None and _lock_info.holder_id != user_id:
            elapsed = _lock_info.elapsed_seconds()
            return (
                False,
                f"GPU 正在被 {_lock_info.holder_name} 使用，"
                f"正在执行「{_lock_info.operation}」({elapsed:.0f}秒)，请稍候...",
            )

        if _lock_info is not None and _lock_info.holder_id == user_id:
            _lock_info.operation = operation
            _lock_info.start_time = time.time()
            return True, "已持有锁"

        acquired = _gpu_lock.acquire(blocking=False)
        if acquired:
            _lock_info = GPULockInfo(
                holder_id=user_id,
                holder_name=user_name,
                operation=operation,
            )
            return True, "获取锁成功"
        else:
            return False, "GPU 正忙，请稍候..."


def release_gpu(user_id: str) -> bool:
    """Release the GPU lock.

    Args:
        user_id: The user attempting to release the lock.

    Returns:
        True if the lock was released, False if the user didn't hold it.
    """
    global _lock_info

    with _lock_guard:
        if _lock_info is None:
            return False

        if _lock_info.holder_id != user_id:
            return False

        _lock_info = None
        try:
            _gpu_lock.release()
        except RuntimeError:
            pass
        return True


def force_release_gpu() -> bool:
    """Force release the GPU lock (admin operation).

    Returns:
        True if a lock was released, False if no lock was held.
    """
    global _lock_info

    with _lock_guard:
        if _lock_info is None:
            return False

        _lock_info = None
        try:
            _gpu_lock.release()
        except RuntimeError:
            pass
        return True


def get_gpu_status() -> dict:
    """Get current GPU lock status.

    Returns:
        Dict with keys:
        - locked: bool
        - holder_id: str or None
        - holder_name: str or None
        - operation: str or None
        - elapsed_seconds: float or None
    """
    with _lock_guard:
        if _lock_info is None:
            return {
                "locked": False,
                "holder_id": None,
                "holder_name": None,
                "operation": None,
                "elapsed_seconds": None,
            }

        return {
            "locked": True,
            "holder_id": _lock_info.holder_id,
            "holder_name": _lock_info.holder_name,
            "operation": _lock_info.operation,
            "elapsed_seconds": _lock_info.elapsed_seconds(),
        }


def is_holder(user_id: str) -> bool:
    """Check if the given user currently holds the GPU lock."""
    with _lock_guard:
        return _lock_info is not None and _lock_info.holder_id == user_id
