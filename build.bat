@echo off
REM Build PCDViewer.exe (PCD Point Cloud Post-Processor) on Windows. Requires 64-bit Python 3.9-3.12 in PATH.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python 3.11 64-bit from python.org
  echo         and tick "Add python.exe to PATH" during installation.
  goto fail
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto fail

python -m PyInstaller --noconfirm --clean --onefile --windowed --name PCDViewer ^
  --icon assets\icon.ico ^
  --collect-submodules OpenGL.platform ^
  --collect-submodules OpenGL.arrays ^
  --exclude-module tkinter --exclude-module matplotlib --exclude-module IPython ^
  --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineWidgets ^
  --exclude-module PySide6.Qt3DCore --exclude-module PySide6.QtQuick --exclude-module PySide6.QtQml ^
  main.py
if errorlevel 1 goto fail

echo.
echo ===== Build OK: dist\PCDViewer.exe =====
pause
exit /b 0

:fail
echo.
echo ===== Build FAILED, see messages above =====
pause
exit /b 1
