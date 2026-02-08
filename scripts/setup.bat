@echo off
REM Setup the SplazMatte conda environment (Windows).
REM
REM Usage:
REM   scripts\setup.bat            # defaults to CUDA
REM   scripts\setup.bat --cuda     # force CUDA install
REM
setlocal enabledelayedexpansion

set ENV_NAME=splazmatte
set PYTHON_VERSION=3.11
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM ---------------------------------------------------------------------------
REM Find conda installation
REM ---------------------------------------------------------------------------
for /f "tokens=*" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i
if not defined CONDA_BASE (
    echo [ERROR] conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Parse args (Windows always uses CUDA)
REM ---------------------------------------------------------------------------
set PLATFORM=cuda
if "%~1"=="--cuda" set PLATFORM=cuda
if "%~1"=="--mps" (
    echo [WARN] MPS is not available on Windows. Falling back to CUDA.
    set PLATFORM=cuda
)
echo [INFO] Platform: %PLATFORM%

REM ---------------------------------------------------------------------------
REM Create conda environment
REM ---------------------------------------------------------------------------
conda info --envs 2>nul | findstr /c:"%ENV_NAME%" >nul 2>&1
if %errorlevel%==0 (
    echo [INFO] Conda environment '%ENV_NAME%' already exists. Skipping creation.
) else (
    echo [INFO] Creating conda environment '%ENV_NAME%' ^(Python %PYTHON_VERSION%^)...
    conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        pause
        exit /b 1
    )
)

REM ---------------------------------------------------------------------------
REM Activate environment via activate.bat
REM ---------------------------------------------------------------------------
call "%CONDA_BASE%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment '%ENV_NAME%'.
    pause
    exit /b 1
)
echo [INFO] Activated environment: %ENV_NAME%

REM ---------------------------------------------------------------------------
REM Install PyTorch (CUDA) via conda to ensure native DLL dependencies resolve
REM ---------------------------------------------------------------------------
echo [INFO] Installing PyTorch for platform: %PLATFORM%
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Install project dependencies
REM ---------------------------------------------------------------------------
echo [INFO] Installing project dependencies from requirements.txt...
pip install -r "%PROJECT_ROOT%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install project dependencies.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Verify installation
REM ---------------------------------------------------------------------------
echo.
echo === Verification ===
python -c "import torch; print(f'PyTorch:        {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import huggingface_hub, diffusers, transformers, gradio; print(f'huggingface_hub: {huggingface_hub.__version__}'); print(f'diffusers:       {diffusers.__version__}'); print(f'transformers:    {transformers.__version__}'); print(f'gradio:          {gradio.__version__}'); print(); print('Setup complete!')"

echo.
echo To activate the environment in a new shell:
echo   conda activate %ENV_NAME%
echo.
echo To download model weights:
echo   scripts\download_models.bat
echo.
pause
