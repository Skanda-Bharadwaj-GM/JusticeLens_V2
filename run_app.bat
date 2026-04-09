@echo off
setlocal
set PYTHONUNBUFFERED=1

REM Go to this script's folder
cd /d "%~dp0"

echo Starting Justice Lens Web App...
echo.
echo Open this in browser after startup:
echo http://localhost:8501
echo.
echo Keep this window open. First run may take 2-5 minutes while libraries load.
echo.
start "" "http://localhost:8501"

where py >nul 2>nul
if %errorlevel%==0 (
    py -u -m streamlit run app.py --server.headless true --server.port 8501
    goto :end
)

REM Prefer module launch to avoid PATH issues
where python >nul 2>nul
if %errorlevel%==0 (
    python -u -m streamlit run app.py --server.headless true --server.port 8501
    goto :end
)

echo [ERROR] Python was not found in PATH.
echo Install Python or use the Python Launcher (py.exe), then run again.

:end
echo.
echo App closed. Press any key to exit.
pause >nul
