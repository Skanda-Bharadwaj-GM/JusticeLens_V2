@echo off
setlocal
cd /d "%~dp0"

REM Unbuffered Python so lines appear in launch_log.txt right away (not stuck "empty")
set PYTHONUNBUFFERED=1

set LOG_FILE=launch_log.txt
echo [%date% %time%] Starting Justice Lens app > "%LOG_FILE%"
echo [%date% %time%] PYTHONUNBUFFERED=1 >> "%LOG_FILE%"

echo.
echo Log file: %LOG_FILE%
echo First run can take 2-5 minutes while Python loads Streamlit and Torch.
echo Open browser: http://localhost:8501  (refresh if page fails at first)
echo This window stays open while the server runs. Close it to stop the app.
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    echo [%date% %time%] Quick Python test... >> "%LOG_FILE%"
    py -c "print('python_ok')" >> "%LOG_FILE%" 2>&1
    echo [%date% %time%] Starting Streamlit (this may take a while)... >> "%LOG_FILE%"
    start "" "http://localhost:8501"
    py -u -m streamlit run app.py --server.headless true --server.port 8501 >> "%LOG_FILE%" 2>&1
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    echo [%date% %time%] Quick Python test... >> "%LOG_FILE%"
    python -c "print('python_ok')" >> "%LOG_FILE%" 2>&1
    echo [%date% %time%] Starting Streamlit (this may take a while)... >> "%LOG_FILE%"
    start "" "http://localhost:8501"
    python -u -m streamlit run app.py --server.headless true --server.port 8501 >> "%LOG_FILE%" 2>&1
    goto :end
)

echo [ERROR] Python not found in PATH >> "%LOG_FILE%"
echo [ERROR] Python not found. Check %LOG_FILE%

:end
echo App stopped. Press any key to exit.
pause >nul
