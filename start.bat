@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\lookout.exe" (
  echo Lookout is not set up yet - running setup first...
  echo.
  call setup.bat
)

echo.
echo ============================================
echo   Starting Lookout
echo   Web UI: http://127.0.0.1:8765
echo   Close this window to stop.
echo ============================================
echo.

start "" http://127.0.0.1:8765
".venv\Scripts\lookout.exe" serve
pause
endlocal
