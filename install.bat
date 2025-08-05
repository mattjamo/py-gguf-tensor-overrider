@echo off
setlocal enabledelayedexpansion

echo GGUF Tensor Overrider - Windows Installation Script
echo ====================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

REM Check if nvidia-smi is available
nvidia-smi --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed.
    echo The tool will not work without NVIDIA GPUs and drivers.
)

REM Create installation directory
set INSTALL_DIR=%LOCALAPPDATA%\gguf-tensor-overrider
set BIN_FILE=%LOCALAPPDATA%\Microsoft\WindowsApps\gguf-tensor-overrider.bat

echo Installing to: %INSTALL_DIR%

REM Remove existing installation if it exists
if exist "%INSTALL_DIR%" (
    echo Removing existing installation...
    rmdir /s /q "%INSTALL_DIR%"
    if !errorlevel! neq 0 (
        echo ERROR: Failed to remove existing installation directory.
        pause
        exit /b 1
    )
)

REM Create installation directory
mkdir "%INSTALL_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to create installation directory.
    pause
    exit /b 1
)

REM Download the main Python script
echo Downloading gguf_tensor_overrider.py...
curl -L -o "%INSTALL_DIR%\gguf_tensor_overrider.py" "https://raw.githubusercontent.com/mattjamo/py-gguf-tensor-overrider/main/gguf_tensor_overrider.py"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download the main script. Please check your internet connection.
    pause
    exit /b 1
)

REM Install required Python packages
echo Installing required Python packages...
python -m pip install requests

if %errorlevel% neq 0 (
    echo ERROR: Failed to install required packages.
    pause
    exit /b 1
)

REM Create batch file wrapper
echo Creating command wrapper...
echo @echo off > "%BIN_FILE%"
echo python "%INSTALL_DIR%\gguf_tensor_overrider.py" %%* >> "%BIN_FILE%"

if %errorlevel% neq 0 (
    echo ERROR: Failed to create command wrapper.
    pause
    exit /b 1
)

echo.
echo ====================================================
echo Installation completed successfully!
echo.
echo You can now use 'gguf-tensor-overrider' from any command prompt.
echo.
echo Example usage:
echo   python gguf-tensor-overrider -g "https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/resolve/main/UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf" -c 32000
echo.
echo Note: Make sure NVIDIA drivers and nvidia-smi are properly installed.
echo ====================================================
pause