#!/bin/bash
set -e

# Configuration
FINETUNED_MODEL_NAME="qwen2.5-7b-rbi" # Updated to reflect Qwen 2.5
BASE_MODEL_NAME="qwen2.5:7b-instruct" # Using the instruct version of base model for fair comparison

echo "=== Setting up Ollama for Model Serving ==="

# 1. Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5
else
    echo "Ollama server is already running."
fi

# 2. Pull Base Model
echo "Pulling base model: $BASE_MODEL_NAME..."
ollama pull $BASE_MODEL_NAME

# 3. Verify Fine-tuned Model exists
if ollama list | grep -q "$FINETUNED_MODEL_NAME"; then
    echo "Fine-tuned model '$FINETUNED_MODEL_NAME' is ready."
else
    echo "ERROR: Fine-tuned model '$FINETUNED_MODEL_NAME' not found!"
    echo "Please run 'bash run_ollama.sh' first to create it."
    exit 1
fi

echo ""
echo "=== Ready to Serve ==="
echo "Ollama API is listening on http://localhost:11434"
echo ""
echo "Models available:"
echo "1. $FINETUNED_MODEL_NAME (Your Fine-tuned Model)"
echo "2. $BASE_MODEL_NAME (Original Base Model)"
echo ""
echo "You can now run 'python3 compare_models.py' to test them."
