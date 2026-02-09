"""Queue persistence helpers.

Manages workspace/queue.json — a simple ordered list of session IDs.
"""

import json
import logging
from pathlib import Path

from config import WORKSPACE_DIR

log = logging.getLogger(__name__)

QUEUE_FILE: Path = WORKSPACE_DIR / "queue.json"


def load_queue() -> list[str]:
    """Read the queue (session_id list) from workspace/queue.json."""
    if not QUEUE_FILE.exists():
        return []
    try:
        data = json.loads(QUEUE_FILE.read_text())
        if isinstance(data, list):
            return [str(sid) for sid in data]
    except (json.JSONDecodeError, OSError):
        log.warning("Failed to load queue.json, returning empty queue")
    return []


def save_queue(queue: list[str]) -> None:
    """Write the queue to workspace/queue.json."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2),
    )
