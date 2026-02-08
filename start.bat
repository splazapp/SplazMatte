@echo off
REM Start the SplazMatte Gradio app (Windows).
REM
REM Usage:
REM   start.bat
REM
setlocal

set SCRIPT_DIR=%~dp0
set ENV_NAME=splazmatte

REM ---------------------------------------------------------------------------
REM Check conda environment exists
REM ---------------------------------------------------------------------------
conda info --envs 2>nul | findstr /c:"%ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda environment '%ENV_NAME%' not found.
    echo         Run 'scripts\setup.bat' first.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Launch app via conda run (avoids activate issues in cmd.exe)
REM ---------------------------------------------------------------------------
echo [INFO] Environment: %ENV_NAME%
echo [INFO] Starting SplazMatte ...
conda run -n %ENV_NAME% python "%SCRIPT_DIR%app.py"
pause
