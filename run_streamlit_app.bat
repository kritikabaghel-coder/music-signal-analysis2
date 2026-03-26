@echo off
REM Music Signal Analysis System - Streamlit App Launcher
REM Run this file to start the Streamlit app

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     🎵 Music Signal Analysis System - Streamlit App         ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements_streamlit.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        echo Please run: pip install -r requirements_streamlit.txt
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

streamlit run streamlit_app.py

pause
