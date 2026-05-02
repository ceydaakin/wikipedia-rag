#!/usr/bin/env bash
# One-shot setup: create venv, install deps, pull Ollama models.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: 'ollama' CLI not found. Install from https://ollama.com first." >&2
    exit 1
fi

PY="${PYTHON:-python3}"
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
