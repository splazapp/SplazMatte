"""GPU lock for multi-user access control.

Implements category-based locking to allow interactive operations (SAM annotation)
to run concurrently with batch operations (propagation, matting, queue execution).

Categories:
- batch: Long-running GPU tasks (propagation, matting, tracking, queue).
         Globally exclusive — only one batch operation at a time.
- interactive: Short GPU tasks (SAM annotation, text detection, model switch).
               Per-user exclusive — each user can hold one interactive lock.
               Does NOT conflict with batch locks.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPULockInfo:
    """Information about a GPU lock holder."""

    holder_id: str
    holder_name: str
    operation: str
    category: str
    start_time: float = field(default_factory=time.time)

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time


_batch_lock_info: Optional[GPULockInfo] = None
_interactive_locks: dict[str, GPULockInfo] = {}
_lock_guard = threading.Lock()


def try_acquire_gpu(
    user_id: str,
    user_name: str,
    operation: str,
    category: str = "batch",
) -> tuple[bool, str]:
    """Try to acquire the GPU lock.

    Args:
        user_id: Unique identifier for the user (from browser session).
        user_name: Display name for the user.
        operation: Description of the operation (e.g., "SAM 标注", "传播").
        category: "batch" for long-running tasks, "interactive" for short tasks.

    Returns:
        (success, message) tuple.
    """
    global _batch_lock_info

    with _lock_guard:
        if category == "interactive":
            existing = _interactive_locks.get(user_id)
            if existing is not None:
                elapsed = existing.elapsed_seconds()
                return (
                    False,
                    f"你已有一个交互操作正在进行「{existing.operation}」"
                    f"({elapsed:.0f}秒)，请等待完成后再操作。",
                )
            _interactive_locks[user_id] = GPULockInfo(
                holder_id=user_id,
                holder_name=user_name,
                operation=operation,
                category="interactive",
            )
            return True, "获取锁成功"

        # category == "batch"
        if _batch_lock_info is not None:
            info = _batch_lock_info
            elapsed = info.elapsed_seconds()
            if info.holder_id == user_id:
                return (
                    False,
                    f"你已有一个 GPU 操作正在进行「{info.operation}」"
                    f"({elapsed:.0f}秒)，请等待完成后再操作。",
                )
            return (
                False,
                f"GPU 正在被 {info.holder_name} 使用，"
                f"正在执行「{info.operation}」({elapsed:.0f}秒)，请稍候...",
            )

        _batch_lock_info = GPULockInfo(
            holder_id=user_id,
            holder_name=user_name,
            operation=operation,
            category="batch",
        )
        return True, "获取锁成功"


def release_gpu(user_id: str, category: str = "batch") -> bool:
    """Release the GPU lock.

    Args:
        user_id: The user attempting to release the lock.
        category: "batch" or "interactive".

    Returns:
        True if the lock was released, False if the user didn't hold it.
    """
    global _batch_lock_info

    with _lock_guard:
        if category == "interactive":
            if user_id in _interactive_locks:
                del _interactive_locks[user_id]
                return True
            return False

        # category == "batch"
        if _batch_lock_info is None:
            return False
        if _batch_lock_info.holder_id != user_id:
            return False
        _batch_lock_info = None
        return True


def force_release_gpu() -> bool:
    """Force release all GPU locks (admin operation).

    Returns:
        True if any lock was released, False if no locks were held.
    """
    global _batch_lock_info

    with _lock_guard:
        had_locks = _batch_lock_info is not None or len(_interactive_locks) > 0
        _batch_lock_info = None
        _interactive_locks.clear()
        return had_locks


def get_gpu_status() -> dict:
    """Get current GPU lock status (batch only).

    Interactive locks are too short-lived to display in the UI.

    Returns:
        Dict with keys: locked, holder_id, holder_name, operation, elapsed_seconds.
    """
    with _lock_guard:
        if _batch_lock_info is None:
            return {
                "locked": False,
                "holder_id": None,
                "holder_name": None,
                "operation": None,
                "elapsed_seconds": None,
            }

        return {
            "locked": True,
            "holder_id": _batch_lock_info.holder_id,
            "holder_name": _batch_lock_info.holder_name,
            "operation": _batch_lock_info.operation,
            "elapsed_seconds": _batch_lock_info.elapsed_seconds(),
        }


def is_holder(user_id: str) -> bool:
    """Check if the given user currently holds the batch GPU lock."""
    with _lock_guard:
        return _batch_lock_info is not None and _batch_lock_info.holder_id == user_id
