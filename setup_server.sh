#!/bin/bash
set -e

echo "=== Starting Server Setup for RBIKM ==="

# 0. Clean up previous environment to ensure no conflicts
if [ -d "venv" ]; then
    echo "Removing existing venv to ensure clean install..."
    rm -rf venv
fi

# 1. System Dependencies
echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl libcurl4-openssl-dev python3-pip python3-venv

# 2. Python Environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade build tools
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# 3. Install Python Requirements
echo "Installing Unsloth and dependencies (Auto-resolving latest compatible versions)..."

# We install unsloth with the cu121 extra, allowing it to determine the best torch version.
# We do NOT manually pin torch or other libs, trusting pip's solver to find the latest compatible set.
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# Install other requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
if [ -f "requirements_finetune.txt" ]; then
    # Exclude unsloth from requirements file if it's there to avoid overwriting/conflicts
    grep -v "unsloth" requirements_finetune.txt > requirements_finetune_clean.txt
    pip install -r requirements_finetune_clean.txt
    rm requirements_finetune_clean.txt
fi

# 4. Build llama.cpp
echo "Building llama.cpp..."
if [ -d "llama.cpp" ]; then
    echo "Cleaning previous llama.cpp build..."
    rm -rf llama.cpp
fi
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
cd ../..

# 5. Install Ollama
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

echo "=== Setup Complete! ==="
echo "Verifying installation..."
python -c "import torch; print(f'Torch version: {torch.__version__}'); import unsloth; print('Unsloth loaded successfully')"

echo ""
echo "To start working:"
echo "1. source venv/bin/activate"
echo "2. python train.py"
