#!/bin/bash
set -e

MODEL_NAME="qwen3-8b-rbi"
GGUF_FILE="qwen3_8b_finetuned-Q4_K_M.gguf"

# 1. Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# 2. Start Ollama server in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5 # Wait for server to start
else
    echo "Ollama server is running."
fi

# 3. Create Modelfile
echo "Creating Modelfile..."
cat <<EOF > Modelfile
FROM ./$GGUF_FILE
SYSTEM "You are a helpful financial assistant trained on RBI reports."
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
EOF

# 4. Create Model
echo "Creating model '$MODEL_NAME'..."
ollama create $MODEL_NAME -f Modelfile

# 5. Run Finetuned Model
echo "Running finetuned model '$MODEL_NAME'..."
echo "Type /bye to exit."
ollama run $MODEL_NAME

# Optional: Run base model for comparison
# echo "Running base model (qwen2.5:7b)..."
# ollama run qwen2.5:7b
