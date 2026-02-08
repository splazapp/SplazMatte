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

set ENV_NAME=splazmatte
set SCRIPT_DIR=%~dp0

REM ---------------------------------------------------------------------------
REM Activate conda environment via activate.bat (supports real-time output)
REM ---------------------------------------------------------------------------
for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i
if not defined CONDA_BASE (
    echo [ERROR] conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)
call "%CONDA_BASE%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment '%ENV_NAME%'.
    echo         Run 'scripts\setup.bat' first.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Run download script, forwarding all CLI arguments
REM ---------------------------------------------------------------------------
echo [INFO] Running download_models.py in '%ENV_NAME%' environment...
python "%SCRIPT_DIR%download_models.py" %*
if errorlevel 1 (
    echo [ERROR] Model download failed.
    pause
    exit /b 1
)

pause
