@echo off
REM Double-click launcher for the CosmoFit GUI (Windows).
REM Installs whatever's missing on first run, then opens the app in
REM your browser. Safe to run again any time -- just launches the app.

cd /d "%~dp0"

python -c "import CosmoFit" >nul 2>&1
if errorlevel 1 (
    echo Installing CosmoFit...
    python -m pip install -e . --quiet
)

python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing the GUI dependency ^(streamlit^)...
    python -m pip install -e ".[gui]" --quiet
)

REM Pre-answer Streamlit's first-run "email address" prompt so this
REM window doesn't hang waiting for input.
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    (echo [general] & echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)

echo Starting the CosmoFit GUI -- it will open in your browser.
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python -m streamlit run app\streamlit_app.py

pause
