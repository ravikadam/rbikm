import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

# Configuration
# You can set HF_USERNAME in .env or replace it here
HF_USERNAME = os.getenv("HF_USERNAME", "ravikadam") 
MODEL_NAME = "qwen2.5-7b-rbi"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
GGUF_FILE = "qwen2.5_7b_finetuned-Q4_K_M.gguf"
LORA_DIR = "lora_model"

def main():
    print("=== Pushing Model to Hugging Face ===")
    
    # Check for HF Token
    if "HF_TOKEN" not in os.environ:
        print("Error: HF_TOKEN not found in environment variables.")
        print("Please add HF_TOKEN=your_token to your .env file.")
        return

    api = HfApi()
    
    print(f"Creating/Verifying repo: {REPO_ID}")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repo creation warning (might already exist): {e}")

    if os.path.exists(GGUF_FILE):
        print(f"Uploading GGUF: {GGUF_FILE}...")
        api.upload_file(
            path_or_fileobj=GGUF_FILE,
            path_in_repo=GGUF_FILE,
            repo_id=REPO_ID,
            repo_type="model",
        )
    else:
        print(f"Warning: GGUF file {GGUF_FILE} not found. Skipping.")

    if os.path.exists(LORA_DIR):
        print(f"Uploading LoRA adapters from {LORA_DIR}...")
        api.upload_folder(
            folder_path=LORA_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            path_in_repo="lora_adapters",
        )
    else:
        print(f"Warning: LoRA directory {LORA_DIR} not found. Skipping.")
    
    print("Upload complete!")
    print(f"Model URL: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
