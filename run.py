#!/usr/bin/env python3
"""CLI entry point for SplazMatte matting tasks.

Usage:
    python run.py session <session_id>    Run matting on a single session
    python run.py queue                   Run all pending queue tasks
    python run.py list                    List all sessions and their status

Exit codes:
    0  All tasks succeeded
    1  One or more tasks failed
    2  Argument error / missing data
"""

import argparse
import logging
import sys

from config import PROCESSING_LOG_FILE, WORKSPACE_DIR


def _setup_logging():
    """Configure logging to stderr and the processing log file."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(str(PROCESSING_LOG_FILE), mode="w"),
        ],
    )


def _make_progress_callback():
    """Create a CLI progress callback using tqdm if available."""
    try:
        from tqdm import tqdm
        bar = tqdm(total=100, unit="%", bar_format="{l_bar}{bar}| {n:.0f}%")
        last_pct = [0.0]

        def callback(frac: float, desc: str = ""):
            pct = frac * 100
            delta = pct - last_pct[0]
            if delta > 0:
                bar.update(delta)
            last_pct[0] = pct
            if desc:
                bar.set_description(desc)

        callback._bar = bar  # keep reference for cleanup
        return callback
    except ImportError:
        def callback(frac: float, desc: str = ""):
            pct = frac * 100
            suffix = f" - {desc}" if desc else ""
            print(f"\r  [{pct:5.1f}%]{suffix}", end="", flush=True)
            if frac >= 1.0:
                print()
        return callback


def cmd_session(args):
    """Run matting on a single session."""
    from matting_runner import run_matting_task
    from session_store import load_session, save_session_state

    session_id = args.session_id
    state = load_session(session_id)
    if state is None:
        print(f"Error: Session '{session_id}' not found.", file=sys.stderr)
        sys.exit(2)

    if not state["keyframes"]:
        print(
            f"Error: Session '{session_id}' has no keyframes.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(
        f"Session: {session_id}\n"
        f"  Video: {state['original_filename']}\n"
        f"  Frames: {state['num_frames']}  Keyframes: {len(state['keyframes'])}\n"
        f"  Engine: {state.get('matting_engine', 'MatAnyone')}"
    )

    state["task_status"] = "processing"
    state["error_msg"] = ""
    save_session_state(state)

    progress_cb = _make_progress_callback()

    try:
        alpha_path, fgr_path, elapsed = run_matting_task(
            state, progress_callback=progress_cb,
        )
        state["task_status"] = "done"
        save_session_state(state)
        print(f"\nDone in {elapsed:.1f}s")
        print(f"  Alpha:      {alpha_path}")
        print(f"  Foreground: {fgr_path}")
    except Exception as exc:
        state["task_status"] = "error"
        state["error_msg"] = str(exc)
        save_session_state(state)
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_queue(args):
    """Run all pending queue tasks."""
    from matting_runner import execute_queue

    done, errors, timings = execute_queue(
        progress_callback=_make_progress_callback(),
    )

    if done == 0 and errors == 0:
        print("No pending tasks in queue.")
        return

    print(f"\nQueue complete: {done} done, {errors} failed")
    for t in timings:
        print(f"  {t}")

    if errors:
        sys.exit(1)


def cmd_list(args):
    """List all sessions and their status."""
    from session_store import list_sessions, read_session_status

    sessions = list_sessions()
    if not sessions:
        print("No sessions found.")
        return

    print(f"{'SESSION ID':<40} {'STATUS':<12} {'VIDEO'}")
    print("-" * 80)
    for _label, sid in sessions:
        info = read_session_status(sid)
        status = info["task_status"] or "new"
        filename = info["original_filename"] or "-"
        print(f"{sid:<40} {status:<12} {filename}")


def main():
    """Parse arguments and dispatch to the appropriate sub-command."""
    parser = argparse.ArgumentParser(
        description="SplazMatte CLI — run matting tasks without the web UI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # session sub-command
    sp_session = subparsers.add_parser(
        "session", help="Run matting on a single session",
    )
    sp_session.add_argument("session_id", help="The session ID to process")
    sp_session.set_defaults(func=cmd_session)

    # queue sub-command
    sp_queue = subparsers.add_parser(
        "queue", help="Run all pending queue tasks",
    )
    sp_queue.set_defaults(func=cmd_queue)

    # list sub-command
    sp_list = subparsers.add_parser(
        "list", help="List all sessions and their status",
    )
    sp_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(2)

    _setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
