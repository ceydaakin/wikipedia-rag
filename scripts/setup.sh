#!/usr/bin/env bash
# One-shot setup: create venv, install deps, pull Ollama models.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: 'ollama' CLI not found. Install from https://ollama.com first." >&2
    exit 1
fi

# Pick a supported Python interpreter. ChromaDB pulls in protobuf/grpcio
# which don't yet ship wheels for Python 3.14, so we prefer 3.13 -> 3.12 -> 3.11.
# Override with PYTHON=/path/to/pythonX.Y if you know what you're doing.
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
else
    PY=""
    for candidate in python3.13 python3.12 python3.11; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    done
    if [ -z "$PY" ]; then
        echo "ERROR: Need Python 3.11, 3.12, or 3.13 on PATH." >&2
        echo "       Python 3.14 is not yet supported by chromadb's protobuf dep." >&2
        echo "       On macOS: brew install python@3.13" >&2
        exit 1
    fi
fi

PY_VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VER" in
    3.11|3.12|3.13) ;;
    *)
        echo "ERROR: Python $PY_VER is not supported (need 3.11, 3.12, or 3.13)." >&2
        echo "       Python 3.14 breaks chromadb's protobuf C extension." >&2
        exit 1
        ;;
esac
echo ">>> Using $PY (Python $PY_VER)"

if [ ! -d ".venv" ]; then
    echo ">>> Creating virtualenv (.venv)"
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">>> Upgrading pip"
pip install --upgrade pip >/dev/null

echo ">>> Installing Python dependencies"
pip install -r requirements.txt

echo ">>> Pulling Ollama models (this may take several minutes the first time)"
ollama pull llama3.2:3b
ollama pull nomic-embed-text

echo ">>> Setup complete."
echo "Next:"
echo "  1) ollama serve   (in another terminal, if not already running)"
echo "  2) python scripts/ingest.py"
echo "  3) streamlit run src/ui/streamlit_app.py    # or python -m src.ui.cli"
