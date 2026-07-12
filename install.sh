#!/usr/bin/env bash
# Sets up pipy_catty: Python 3.12 venv + dependencies + .env scaffold.
# Run from Git Bash (or any bash on Windows): ./install.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_VERSION="3.12"
VENV_DIR="venv"
VENV_PY="$VENV_DIR/Scripts/python.exe"

echo "==> Checking for the 'py' launcher..."
if ! command -v py >/dev/null 2>&1; then
    echo "ERROR: 'py' launcher not found. Install Python from https://python.org and re-run." >&2
    exit 1
fi

echo "==> Checking for Python $PYTHON_VERSION..."
# PyAudio (needed for local mic/speaker audio) has no prebuilt wheel on very
# new Python releases and needs MSVC to build from source, so this project
# pins 3.12, where PyAudio wheels are readily available.
if ! py "-$PYTHON_VERSION" --version >/dev/null 2>&1; then
    echo "    Python $PYTHON_VERSION not found."
    if command -v winget >/dev/null 2>&1; then
        echo "==> Installing Python $PYTHON_VERSION via winget (user scope)..."
        winget install --id Python.Python.3.12 --scope user \
            --silent --accept-source-agreements --accept-package-agreements
    else
        echo "ERROR: winget not available. Install Python $PYTHON_VERSION manually and re-run." >&2
        exit 1
    fi
else
    echo "    Found."
fi

echo "==> Setting up virtual environment ($VENV_DIR)..."
if [ ! -f "$VENV_PY" ]; then
    py "-$PYTHON_VERSION" -m venv "$VENV_DIR"
else
    echo "    Already exists, skipping."
fi

echo "==> Upgrading pip, setuptools, wheel..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

echo "==> Installing requirements..."
"$VENV_PY" -m pip install -r requirements.txt

echo "==> Preparing .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "    Created .env from .env.example."
else
    echo "    .env already exists, leaving it untouched."
fi

chmod +x pipy 2>/dev/null || true

echo
echo "Install complete."
echo
if grep -qE "^(OPENAI_API_KEY|SPEECHMATICS_API_KEY)=\s*$" .env 2>/dev/null; then
    echo "  1. Edit .env and add your OPENAI_API_KEY and SPEECHMATICS_API_KEY."
    echo "  2. Run: ./pipy start"
else
    echo "  Run: ./pipy start"
fi
