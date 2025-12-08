import requests
import json

# Configuration
FINETUNED_MODEL = "qwen3-8b-rbi"
BASE_MODEL = "qwen2.5:7b-instruct"
API_URL = "http://localhost:11434/api/generate"

def query_ollama(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error: {e}"

def main():
    print(f"=== Model Comparison: {FINETUNED_MODEL} vs {BASE_MODEL} ===\n")
    
    while True:
        prompt = input("\nEnter your prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break
            
        print(f"\n--- Querying {BASE_MODEL} (Base) ---")
        base_response = query_ollama(BASE_MODEL, prompt)
        print(base_response)
        
        print(f"\n--- Querying {FINETUNED_MODEL} (Fine-tuned) ---")
        finetuned_response = query_ollama(FINETUNED_MODEL, prompt)
        print(finetuned_response)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
