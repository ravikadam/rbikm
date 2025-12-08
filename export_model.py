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

    # Save the merged model first
    print("Merging and saving model to 'merged_model'...")
    model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit")
    
    # Explicitly save tokenizer to the same directory
    print("Saving tokenizer to 'merged_model'...")
    tokenizer.save_pretrained("merged_model")

    # Qwen uses tokenizer.json, but sometimes convert_hf_to_gguf looks for tokenizer.model
    # We will ensure all tokenizer files are present.
    # Also, we need to make sure the config.json has the correct model_type
    
    print("Converting to GGUF using llama.cpp...")
    
    # Define paths
    llama_cpp_dir = "llama.cpp"
    merged_model_dir = "merged_model"
    output_gguf = "qwen3_8b_finetuned.gguf"
    quantized_gguf = "qwen3_8b_finetuned-Q4_K_M.gguf"

    # 1. Convert HF to GGUF (fp16)
    import subprocess
    
    convert_cmd = [
        "python3", 
        f"{llama_cpp_dir}/convert_hf_to_gguf.py", 
        merged_model_dir, 
        "--outfile", output_gguf,
        "--outtype", "f16"
    ]
    
    print(f"Running: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)
    
    # 2. Quantize to Q4_K_M
    quantize_cmd = [
        f"{llama_cpp_dir}/build/bin/llama-quantize",
        output_gguf,
        quantized_gguf,
        "Q4_K_M"
    ]
    
    print(f"Running: {' '.join(quantize_cmd)}")
    subprocess.run(quantize_cmd, check=True)
    
    print(f"Export complete! File saved as: {quantized_gguf}")

if __name__ == "__main__":
    main()
