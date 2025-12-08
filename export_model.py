from unsloth import FastLanguageModel
import torch
import os

# Configuration
model_path = "lora_model" # Path to the saved adapters
max_seq_length = 2048
dtype = None
load_in_4bit = True

def main():
    print(f"Loading model adapters from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    print("Saving to GGUF format (q4_k_m)...")
    # This will merge the LoRA adapters into the base model and convert to GGUF
    # It saves to the current directory
    model.save_pretrained_gguf("qwen3_8b_finetuned", tokenizer, quantization_method = "q4_k_m")
    
    print("Export complete!")
    print("You should see a file like 'qwen3_8b_finetuned-unsloth.Q4_K_M.gguf'")

if __name__ == "__main__":
    main()
