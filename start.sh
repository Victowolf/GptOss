#!/usr/bin/env bash
set -euo pipefail

echo "=== start.sh: create venv and install requirements ==="

# ---------------------------
# Create venv
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
# Upgrade pip + install base requirements
# ---------------------------
echo "[4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[5] Installing requirements..."
pip install -r requirements.txt

# ---------------------------
# Install Transformers (GitHub latest MXFP4 support)
# ---------------------------
echo "[6] Installing transformers from GitHub..."
pip install --upgrade "git+https://github.com/huggingface/transformers.git"

# ---------------------------
# Install Triton + MXFP4 kernels
# ---------------------------
echo "[7] Installing Triton compiler..."
pip install --upgrade triton==3.5.1

echo "[8] Installing Triton MXFP4 kernels (critical for H200)..."
pip install --upgrade "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"

# ---------------------------
# Final verification
# ---------------------------
echo "[9] Verifying MXFP4 availability..."
python3 - << 'EOF'
import importlib, triton
print("Triton version:", triton.__version__)
print("Has triton_kernels:", importlib.util.find_spec("triton_kernels") is not None)
EOF

# ---------------------------
# START SERVER
# ---------------------------
echo "=== Starting GptOSS-20b FastAPI server ==="
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
