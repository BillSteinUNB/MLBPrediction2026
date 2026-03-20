@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Project virtual environment not found at "%PYTHON_EXE%".>&2
  exit /b 1
)

pushd "%REPO_ROOT%" >nul
"%PYTHON_EXE%" -m src.pipeline.daily --date today --mode prod %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul

exit /b %EXIT_CODE%
