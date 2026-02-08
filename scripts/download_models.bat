@echo off
REM Download model weights for SplazMatte (Windows).
REM
REM Activates the conda environment and delegates to download_models.py.
REM All arguments are forwarded.
REM
REM Usage:
REM   scripts\download_models.bat              # download all models
REM   scripts\download_models.bat --sam3       # SAM3 only
REM   scripts\download_models.bat --matanyone  # MatAnyone only
REM   scripts\download_models.bat --videomama  # VideoMaMa only
REM   scripts\download_models.bat --verify     # verify existing downloads
REM
setlocal

set ENV_NAME=splazmatte
set SCRIPT_DIR=%~dp0

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
REM Run download script via conda run (avoids activate issues in cmd.exe)
REM ---------------------------------------------------------------------------
echo [INFO] Running download_models.py in '%ENV_NAME%' environment...
conda run -n %ENV_NAME% python "%SCRIPT_DIR%download_models.py" %*
if errorlevel 1 (
    echo [ERROR] Model download failed.
    pause
    exit /b 1
)

pause
