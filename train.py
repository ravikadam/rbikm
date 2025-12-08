from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# WandB Project Name
os.environ["WANDB_PROJECT"] = "qwen3-8b-finetune-rbi" 
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # Log checkpoints to wandb

model_name = "Qwen/Qwen2.5-7B-Instruct" # Use Instruct version for better chat performance

def main():
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    # Load Dataset
    dataset = load_dataset("json", data_files="finetune_dataset.json", split="train")
    print(f"Loaded dataset with {len(dataset)} samples")


### Response:
{}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
            texts.append(text)
        return texts

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
        formatting_func = formatting_prompts_func,
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training completed!")

    # Save model
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("Model saved to lora_model")


if __name__ == "__main__":
    main()
