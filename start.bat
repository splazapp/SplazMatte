@echo off
REM Start the SplazMatte Gradio app (Windows).
REM
REM Usage:
REM   start.bat
REM

set SCRIPT_DIR=%~dp0
set ENV_NAME=splazmatte

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment "%ENV_NAME%".
    echo         Please run "scripts/setup.sh --cuda" first.
    pause
    exit /b 1
)

echo [INFO] Environment: %ENV_NAME%
echo [INFO] Starting SplazMatte ...

python "%SCRIPT_DIR%app.py"
pause
