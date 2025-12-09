#!/bin/bash
set -e

echo "=== Starting Server Setup for RBIKM ==="

# 1. System Dependencies
echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl libcurl4-openssl-dev python3-pip python3-venv

# 2. Python Environment
echo "Setting up Python environment..."
# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# 3. Install Python Requirements
echo "Installing Python dependencies..."
# Install PyTorch first (Unsloth needs it) - assuming CUDA 12.1 or similar compatible
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (Robust install)
echo "Installing Unsloth..."
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Install other requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
if [ -f "requirements_finetune.txt" ]; then
    pip install -r requirements_finetune.txt
fi

# 4. Build llama.cpp
echo "Building llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp
git pull
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
echo "To start working:"
echo "1. source venv/bin/activate"
echo "2. python train.py"
