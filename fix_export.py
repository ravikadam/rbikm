import os
import shutil
import json
import subprocess
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Configuration
model_path = "lora_model"
merged_model_dir = "merged_model"
llama_cpp_dir = "llama.cpp"
output_gguf = "qwen2.5_7b_finetuned.gguf"
quantized_gguf = "qwen2.5_7b_finetuned-Q4_K_M.gguf"

def main():
    print("=== Starting Robust Export Fix ===")
    
    # 1. Load and Merge Model
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    
    print(f"Saving merged model to {merged_model_dir}...")
    model.save_pretrained_merged(merged_model_dir, tokenizer, save_method = "merged_16bit")
    
    # 2. FIX TOKENIZER FILES
    # Unsloth/HF might save tokenizer.model which confuses llama.cpp into thinking it's SentencePiece.
    # Qwen 2.5/3 uses BPE (tokenizer.json).
    
    print("=== Fixing Tokenizer Files ===")
    
    # Delete tokenizer.model if it exists
    tokenizer_model_path = os.path.join(merged_model_dir, "tokenizer.model")
    if os.path.exists(tokenizer_model_path):
        print(f"Deleting {tokenizer_model_path} to force BPE mode...")
        os.remove(tokenizer_model_path)
        
    # Ensure tokenizer.json exists
    tokenizer_json_path = os.path.join(merged_model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        print("WARNING: tokenizer.json not found! Saving explicitly...")
        tokenizer.save_pretrained(merged_model_dir)

    # Verify tokenizer_config.json
    config_path = os.path.join(merged_model_dir, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Tokenizer config class: {config.get('tokenizer_class')}")
        # Ensure it says Qwen2Tokenizer
        if config.get('tokenizer_class') != 'Qwen2Tokenizer':
            print("Updating tokenizer_class to Qwen2Tokenizer...")
            config['tokenizer_class'] = 'Qwen2Tokenizer'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

    # 3. Run Conversion
    print("=== Running GGUF Conversion ===")
    
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"Could not find {convert_script}. Is llama.cpp cloned?")

    convert_cmd = [
        "python3",
        convert_script,
        merged_model_dir,
        "--outfile", output_gguf,
        "--outtype", "f16"
    ]
    
    print(f"Running: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)
    
    # 4. Quantize
    print("=== Quantizing to Q4_K_M ===")
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    if not os.path.exists(quantize_bin):
         # Try default location if build/bin doesn't exist (older cmake or make)
         quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")
    
    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(f"Could not find llama-quantize binary at {quantize_bin}")

    quantize_cmd = [
        quantize_bin,
        output_gguf,
        quantized_gguf,
        "Q4_K_M"
    ]
    
    print(f"Running: {' '.join(quantize_cmd)}")
    subprocess.run(quantize_cmd, check=True)
    
    print(f"=== SUCCESS: Saved {quantized_gguf} ===")

if __name__ == "__main__":
    main()
