from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
from dotenv import load_dotenv
from unsloth.chat_templates import get_chat_template

load_dotenv()

# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

# WandB Project Name
os.environ["WANDB_PROJECT"] = "qwen3-8b-finetune-rbi" 
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

model_name = "Qwen/Qwen2.5-7B-Instruct"

def main():
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Setup for Qwen 2.5/3 (ChatML)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    )

    # Do model patching and add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # Load Dataset
    dataset = load_dataset("json", data_files="finetune_dataset.json", split="train")
    print(f"Loaded dataset with {len(dataset)} samples")

    # ChatML Formatting Function
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # ChatML format
            text = f"<|im_start|>system\nYou are a helpful financial assistant trained on RBI reports.<|im_end|>\n" \
                   f"<|im_start|>user\n{instruction}\n{input}<|im_end|>\n" \
                   f"<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)
        return texts

    # Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 20,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to="wandb",
        run_name="qwen2.5-7b-finetune-rbi",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
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
