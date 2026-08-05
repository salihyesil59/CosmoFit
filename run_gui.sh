#!/usr/bin/env bash
# Double-click (or "Run in Terminal") launcher for the CosmoFit GUI.
# Installs whatever's missing on first run, then opens the app in
# your browser. Safe to run again any time -- just launches the app.

set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import CosmoFit" >/dev/null 2>&1; then
    echo "Installing CosmoFit..."
    "$PYTHON" -m pip install -e . --quiet
fi

if ! "$PYTHON" -c "import streamlit" >/dev/null 2>&1; then
    echo "Installing the GUI dependency (streamlit)..."
    "$PYTHON" -m pip install -e ".[gui]" --quiet
fi

# Pre-answer Streamlit's first-run "email address" prompt so a
# double-clicked window (no interactive stdin) doesn't hang waiting
# for input.
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo "Starting the CosmoFit GUI -- it will open in your browser."
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
"$PYTHON" -m streamlit run app/streamlit_app.py

# Keep the window open if launched by double-clicking (so any error
# above stays readable instead of the terminal closing immediately).
read -n 1 -s -r -p "Press any key to close this window..."
