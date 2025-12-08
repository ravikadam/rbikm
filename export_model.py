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

    # --- FIX: Prepare environment for Unsloth's internal exporter ---
    import os
    import shutil
    import unsloth_zoo.llama_cpp

    print("Patching Unsloth environment checks...")

    # 1. Bypass internet check
    def mock_do_we_need_sudo():
        return False
    unsloth_zoo.llama_cpp.do_we_need_sudo = mock_do_we_need_sudo

    # 2. Symlink binaries so Unsloth finds them
    # Unsloth looks for 'quantize' or 'llama-quantize' in the llama.cpp folder
    cwd = os.getcwd()
    llama_cpp_dir = os.path.join(cwd, "llama.cpp")
    build_bin_dir = os.path.join(llama_cpp_dir, "build", "bin")
    
    binaries = ["llama-quantize", "llama-cli", "llama-server"]
    for binary in binaries:
        src = os.path.join(build_bin_dir, binary)
        dst = os.path.join(llama_cpp_dir, binary)
        
        # Also try short names (quantize, main) just in case
        short_name = binary.replace("llama-", "")
        dst_short = os.path.join(llama_cpp_dir, short_name)

        if os.path.exists(src):
            if not os.path.exists(dst):
                print(f"Symlinking {src} -> {dst}")
                os.symlink(src, dst)
            if not os.path.exists(dst_short):
                print(f"Symlinking {src} -> {dst_short}")
                os.symlink(src, dst_short)
    
    # 3. Use Unsloth's exporter
    print("Saving to GGUF format (q4_k_m) using Unsloth...")
    try:
        model.save_pretrained_gguf("qwen3_8b_finetuned", tokenizer, quantization_method = "q4_k_m")
        print("Export complete!")
    except Exception as e:
        print(f"Unsloth export failed: {e}")
        print("Falling back to manual conversion is not recommended due to tokenizer issues.")
        raise e

if __name__ == "__main__":
    main()
