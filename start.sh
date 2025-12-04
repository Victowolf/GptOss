#!/usr/bin/env bash
set -euo pipefail

echo "=== start.sh: create venv and install requirements ==="

# ---------------------------
# Create venv even if ensurepip is missing
# ---------------------------
if [ ! -d "venv" ]; then
  echo "[1] Creating venv..."
  python3 -m venv venv --without-pip

  echo "[2] Installing pip manually inside venv..."
  curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  ./venv/bin/python get-pip.py
  rm get-pip.py
fi

# ---------------------------
# Activate the venv
# ---------------------------
echo "[3] Activating venv..."
source venv/bin/activate

# ---------------------------
# Upgrade pip + install requirements
# ---------------------------
echo "[4] Upgrading pip..."
pip install --upgrade pip

echo "[5] Installing python requirements..."
pip install -r requirements.txt

# ---------------------------
# START SERVER
# ---------------------------
echo "=== Starting GptOSS-20b FastAPI server ==="
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
