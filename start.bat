@echo off
REM Start the SplazMatte Gradio app (Windows).
REM
REM Usage:
REM   start.bat
REM

set SCRIPT_DIR=%~dp0
set ENV_NAME=splazmatte

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

echo [INFO] Environment: %ENV_NAME%
echo [INFO] Starting SplazMatte ...
python "%SCRIPT_DIR%app.py"
pause
