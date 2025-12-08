import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the merged model we are trying to convert
model_path = "merged_model" 

print(f"Loading model from {model_path}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("Model loaded successfully.")

prompt = """We run a prepaid voucher program. We are fintech - we work with issuing bank.
We acquire customer. Voucher is actually issued by bank. They also share transaction details with us.
Can we save it in our system for future use?"""

messages = [
    {"role": "system", "content": "You are a helpful financial assistant trained on RBI reports."},
    {"role": "user", "content": prompt}
]

# Apply chat template if available, otherwise raw prompt
try:
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
except Exception:
    print("Chat template not found, using raw prompt.")
    text = prompt

print(f"\nInput Text:\n{text}\n")

inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nOutput:\n{response}")
