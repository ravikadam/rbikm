import os
import json
import re
import time
from pypdf import PdfReader
from tqdm import tqdm
from typing import List, Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configuration
REPORTS_DIR = "Reports"
TRACKER_FILE = "tracker.json"
DATASET_FILE = "dataset.json"
# User needs to set GEMINI_API_KEY in environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash" # Cheaper and fast
CONTEXT_WINDOW_SIZE = 3

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set. Please set it before running.")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class Tracker:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_file_status(self, filename):
        return self.data.get(filename, {"paragraph_index": 0, "completed": False})

    def update_file_status(self, filename, paragraph_index, completed=False):
        self.data[filename] = {"paragraph_index": paragraph_index, "completed": completed}
        self.save()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_paragraphs(text: str) -> List[str]:
    # Split by double newlines which usually indicate paragraph breaks in extracted text
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    # Filter very short paragraphs
    return [p for p in paras if len(p.split()) > 10]

def call_llm(context_paras: List[str], target_para: str) -> Optional[Dict]:
    if not GEMINI_API_KEY:
        return None

    prompt = f"""
    You are an expert at creating datasets for Supervised Fine-Tuning (SFT) from financial reports.
    
    Task:
    1. Analyze the 'Target Paragraph'.
    2. Determine if it contains valuable knowledge suitable for a Q&A dataset. Skip if it is:
       - A table of contents, index, or list of figures.
       - A legal disclaimer or copyright notice.
       - A header, footer, or page number artifact.
       - Too fragmented or lacks semantic meaning.
    3. If valid:
       - Generate a specific Question that the 'Target Paragraph' answers.
       - The Answer should be the 'Target Paragraph'.
       - However, if the 'Target Paragraph' relies on the 'Context Paragraphs' to be fully understood (e.g., it starts with "It also..."), ENHANCE the Answer by incorporating necessary details from the Context.
       - The Answer must be self-contained.
    
    Context Paragraphs (Previous):
    {json.dumps(context_paras)}
    
    Target Paragraph (Current):
    {target_para}
    
    Output JSON ONLY in this format:
    {{
        "valid": true/false,
        "question": "The generated question",
        "answer": "The enhanced answer"
    }}
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        result_text = response.text
        try:
            result = json.loads(result_text)
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    return result[0]
                else:
                    return None
            return result
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e} | Text: {result_text[:100]}...")
            return None
            
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        time.sleep(1) 
        return None

def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found. Exiting.")
        return

    tracker = Tracker(TRACKER_FILE)
    
    dataset = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'r') as f:
            try:
                dataset = json.load(f)
            except json.JSONDecodeError:
                dataset = []

    files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')])
    
    print(f"Found {len(files)} reports.")
    
    for filename in files:
        status = tracker.get_file_status(filename)
        if status['completed']:
            print(f"Skipping {filename} (Completed)")
            continue
            
        print(f"Processing {filename}...")
        filepath = os.path.join(REPORTS_DIR, filename)
        
        try:
            reader = PdfReader(filepath)
            full_text = ""
            print("Extracting text from all pages...")
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            
            all_paras = get_paragraphs(full_text)
            total_paras = len(all_paras)
            start_index = status['paragraph_index']
            
            print(f"Total paragraphs: {total_paras}. Resuming from index {start_index}.")
            
            # Sliding window buffer needs to be reconstructed or just start empty/with what we have?
            # Ideally we should look at previous paragraphs from the full list for context
            
            for i in range(start_index, total_paras):
                target_para = all_paras[i]
                
                # Get context from previous 2 paragraphs in the full list
                context_start = max(0, i - 2)
                context = all_paras[context_start:i]
                
                result = call_llm(context, target_para)
                
                if result and result.get('valid'):
                    entry = {
                        "file": filename,
                        "paragraph_index": i,
                        "question": result['question'],
                        "answer": result['answer']
                    }
                    dataset.append(entry)
                
                # Save progress periodically (e.g., every 5 paragraphs)
                if i % 5 == 0:
                    tracker.update_file_status(filename, i + 1)
                    with open(DATASET_FILE, 'w') as f:
                        json.dump(dataset, f, indent=2)
                    print(f"Processed {filename} Para {i + 1}/{total_paras}. Dataset size: {len(dataset)}")
            
            tracker.update_file_status(filename, total_paras, completed=True)
            # Final save
            with open(DATASET_FILE, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Finished {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            break

if __name__ == "__main__":
    main()
