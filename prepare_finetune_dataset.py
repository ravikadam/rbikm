import json
import random
from sklearn.model_selection import train_test_split

INPUT_FILE = "dataset.json"
OUTPUT_FILE = "finetune_dataset.json"

def convert_to_alpaca(entry):
    return {
        "instruction": "Answer the question based on the provided context from the financial report.",
        "input": f"Question: {entry['question']}",
        "output": entry['answer']
    }

def main():
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loaded {len(data)} entries.")
    
    alpaca_data = [convert_to_alpaca(entry) for entry in data]
    
    # Optional: Split into train/val if we had enough data, but for now just save all
    # If we want to be fancy:
    # train, val = train_test_split(alpaca_data, test_size=0.1, random_state=42)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(alpaca_data, f, indent=2)
    
    print(f"Saved {len(alpaca_data)} entries to {OUTPUT_FILE} in Alpaca format.")
    print("Sample entry:")
    print(json.dumps(alpaca_data[0], indent=2))

if __name__ == "__main__":
    main()
