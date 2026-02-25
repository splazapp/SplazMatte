"""Queue persistence helpers.

Manages workspace/queue.json — an ordered list of task items.
Each item is ``{"type": "matting"|"tracking", "sid": "<session_id>"}``.
Legacy entries (plain strings) are treated as matting tasks.
"""

import json
import logging
from pathlib import Path
from typing import Any

from config import WORKSPACE_DIR

log = logging.getLogger(__name__)

QUEUE_FILE: Path = WORKSPACE_DIR / "queue.json"

QueueItem = dict[str, str]  # {"type": ..., "sid": ...}


def _normalize_item(raw: Any) -> QueueItem:
    """Convert a raw queue entry to the canonical dict format."""
    if isinstance(raw, dict) and "type" in raw and "sid" in raw:
        return {"type": raw["type"], "sid": str(raw["sid"])}
    # Legacy format: plain session-id string → matting task
    return {"type": "matting", "sid": str(raw)}


def load_queue() -> list[QueueItem]:
    """Read queue items from workspace/queue.json."""
    if not QUEUE_FILE.exists():
        return []
    try:
        data = json.loads(QUEUE_FILE.read_text())
        if isinstance(data, list):
            return [_normalize_item(item) for item in data]
    except (json.JSONDecodeError, OSError):
        log.warning("Failed to load queue.json, returning empty queue")
    return []


def save_queue(queue: list[QueueItem]) -> None:
    """Write the queue to workspace/queue.json."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2),
    )
