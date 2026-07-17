#!/usr/bin/env bash
# Sets up pipy_catty: Python venv + dependencies + .env scaffold.
# Works on Raspberry Pi / Debian-based Linux (the deployment target) and on
# Windows via Git Bash (for local dev). Run: ./install.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR="venv"
OS_KIND="$(uname -s)"

case "$OS_KIND" in
    Linux)
        echo "==> Detected Linux ($(uname -m)) — assuming Raspberry Pi / Debian-based."
        echo "==> Installing system packages (requires sudo)..."
        # portaudio19-dev: PyAudio needs PortAudio headers to build.
        # build-essential: compilers for PyAudio and any other sdist builds.
        # (libatlas-base-dev used to be needed for numpy/scipy on ARM, but it
        # no longer exists as a package on Debian trixie/Pi OS bookworm+ —
        # current numpy wheels for aarch64 bundle OpenBLAS, so it's dropped.)
        sudo apt-get update
        sudo apt-get install -y \
            python3 python3-venv python3-pip \
            portaudio19-dev build-essential

        PYTHON_BIN="python3"
        VENV_PY="$VENV_DIR/bin/python"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo "==> Detected Windows (Git Bash)."
        PYTHON_VERSION="3.12"
        VENV_PY="$VENV_DIR/Scripts/python.exe"

        echo "==> Checking for the 'py' launcher..."
        if ! command -v py >/dev/null 2>&1; then
            echo "ERROR: 'py' launcher not found. Install Python from https://python.org and re-run." >&2
            exit 1
        fi

        echo "==> Checking for Python $PYTHON_VERSION..."
        # PyAudio has no prebuilt wheel on very new Python releases on Windows
        # and needs MSVC to build from source, so dev is pinned to 3.12, where
        # PyAudio wheels are readily available. (Not an issue on Linux, where
        # PyAudio builds from source against portaudio19-dev in seconds.)
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
        PYTHON_BIN="py -$PYTHON_VERSION"
        ;;
    *)
        echo "ERROR: unsupported OS '$OS_KIND'. This script supports Raspberry Pi / Debian Linux and Windows (Git Bash)." >&2
        exit 1
        ;;
esac

echo "==> Setting up virtual environment ($VENV_DIR)..."
if [ ! -f "$VENV_PY" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo "    Already exists, skipping."
fi

echo "==> Verifying Python version (pipecat-ai requires >=3.11)..."
"$VENV_PY" - <<'PYEOF'
import sys
if sys.version_info < (3, 11):
    print(f"ERROR: venv Python is {sys.version.split()[0]}, but pipecat-ai requires >=3.11.", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "==> Upgrading pip, setuptools, wheel..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

echo "==> Installing requirements..."
"$VENV_PY" -m pip install -r requirements.txt

echo "==> Installing openwakeword..."
# openwakeword hard-requires tflite-runtime on Linux (platform_system ==
# "Linux" marker), but tflite-runtime has no wheel for many Pi Python
# builds (e.g. Python 3.13 on aarch64), which fails the whole install even
# though we only ever use openwakeword's ONNX backend. Install its actual
# runtime deps ourselves and pull in openwakeword with --no-deps to skip
# the unneeded, unavailable tflite-runtime requirement. Harmless on
# Windows too, since the "full" install works there but this path also
# reaches the exact same result.
"$VENV_PY" -m pip install \
    "onnxruntime<2,>=1.10.0" "tqdm<5.0,>=4.0" "scipy<2,>=1.3" \
    "scikit-learn<2,>=1" "requests<3,>=2.0"
"$VENV_PY" -m pip install --no-deps openwakeword==0.6.0

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
    echo "  2. Run: ./pipy devices     (find your mic/speaker indices, esp. on the Pi)"
    echo "  3. Run: ./pipy start"
else
    echo "  Run: ./pipy devices   (find your mic/speaker indices, esp. on the Pi)"
    echo "  Run: ./pipy start"
fi
