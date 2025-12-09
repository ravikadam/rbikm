# Server Setup Instructions

Follow these steps to set up a fresh GPU server (e.g., Lambda Labs, AWS, GCP) for this project.

## Prerequisites
- NVIDIA GPU (L4, A10, A100, etc.)
- Ubuntu 20.04 or 22.04 LTS
- Git

## 1. Clone the Repository
```bash
git clone https://github.com/ravikadam/rbikm.git
cd rbikm
```

## 2. Run the Setup Script
This script will install all system dependencies, Python libraries (including Unsloth and PyTorch), build `llama.cpp`, and install Ollama.

```bash
chmod +x setup_server.sh
./setup_server.sh
```

## 3. Activate Environment
Always activate the virtual environment before running scripts:
```bash
source venv/bin/activate
```

## 4. Run the Workflow
1.  **Fine-tune**: `python train.py`
2.  **Export**: `python fix_export.py`
3.  **Create Ollama Model**: `bash run_ollama.sh`
4.  **Serve & Compare**: `bash serve_models.sh` then `python compare_models.py`

## Troubleshooting
- **Unsloth Installation**: If `setup_server.sh` fails at Unsloth, check the [Unsloth README](https://github.com/unslothai/unsloth) for the specific command matching your CUDA version.
- **Ollama Service**: If Ollama fails to start, try running `ollama serve` in a separate terminal window.
