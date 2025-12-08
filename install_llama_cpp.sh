#!/bin/bash
set -e

echo "Installing dependencies..."
apt-get update && apt-get install -y build-essential cmake git

echo "Cloning llama.cpp..."
if [ -d "llama.cpp" ]; then
    echo "llama.cpp directory already exists. Pulling latest..."
    cd llama.cpp
    git pull
else
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
fi

echo "Building llama.cpp..."
make clean
make all -j$(nproc)

echo "llama.cpp installation complete!"
cd ..
