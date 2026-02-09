# SplazMatte — command runner
# Usage: just <recipe>    (run `just --list` to see all recipes)

# Conda environment name
env_name := "splazmatte"

# Python binary inside the conda env
python := "/Users/wangtong/anaconda3/envs/" + env_name + "/bin/python"

# Project root (where this justfile lives)
root := justfile_directory()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# Start the Gradio web UI
app:
    {{python}} {{root}}/app.py

# Start the Gradio web UI with a custom port
[doc("Start web UI on a custom port (default 7870)")]
app-port port="7870":
    SPLAZMATTE_PORT={{port}} {{python}} {{root}}/app.py

# ---------------------------------------------------------------------------
# CLI matting
# ---------------------------------------------------------------------------

# Run matting on a single session
run session_id:
    {{python}} {{root}}/run.py session {{session_id}}

# Run all pending tasks in the queue
run-queue:
    {{python}} {{root}}/run.py queue

# List all sessions and their status
list:
    @{{python}} {{root}}/run.py list

# ---------------------------------------------------------------------------
# Setup & models
# ---------------------------------------------------------------------------

# Set up the conda environment and install dependencies
setup *args:
    bash {{root}}/scripts/setup.sh {{args}}

# Download model weights (pass --sam3, --matanyone, --videomama, or --verify)
download-models *args:
    bash {{root}}/scripts/download_models.sh {{args}}

# Install/update Python dependencies only (skip env creation)
install-deps:
    {{python}} -m pip install -r {{root}}/requirements.txt

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

# Show project version
version:
    @cat {{root}}/VERSION

# Verify all Python files compile without errors
check:
    @echo "Checking syntax..."
    @{{python}} -m py_compile {{root}}/config.py
    @{{python}} -m py_compile {{root}}/session_store.py
    @{{python}} -m py_compile {{root}}/matting_runner.py
    @{{python}} -m py_compile {{root}}/run.py
    @{{python}} -m py_compile {{root}}/app_callbacks.py
    @{{python}} -m py_compile {{root}}/queue_callbacks.py
    @{{python}} -m py_compile {{root}}/queue_models.py
    @{{python}} -m py_compile {{root}}/app.py
    @echo "All files OK"

# Show device info (CUDA / MPS / CPU)
device:
    @{{python}} -c "\
    import torch; \
    dev = 'CUDA' if torch.cuda.is_available() else ('MPS' if torch.backends.mps.is_built() and torch.backends.mps.is_available() else 'CPU'); \
    print(f'PyTorch {torch.__version__}  Device: {dev}')"

# Clean Python caches and __pycache__ directories
clean:
    find {{root}} -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find {{root}} -name '*.pyc' -delete 2>/dev/null || true
    @echo "Cleaned __pycache__ and .pyc files"

# Clean workspace processing artifacts (keeps sessions intact)
clean-logs:
    rm -f {{root}}/workspace/processing.log
    rm -f {{root}}/logs/*.log
    @echo "Cleaned log files"
