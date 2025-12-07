# RBI Reports Fine-tuning with Qwen 3 8B

This project contains scripts to generate a dataset from RBI reports and fine-tune a Qwen 3 8B model.

## Workflow

1.  **Data Generation (Mac/Local)**:
    *   Place PDFs in `Reports/`.
    *   Run `python generate_dataset.py` to create `dataset.json`.
    *   Run `python prepare_finetune_dataset.py` to create `finetune_dataset.json`.

2.  **Fine-tuning (GPU/L4)**:
    *   Clone this repository (or transfer files).
    *   Install dependencies: `pip install -r requirements_finetune.txt`.
    *   Run training: `python train.py`.

## Files
*   `generate_dataset.py`: Extracts text from PDFs and uses Gemini API to generate Q&A.
*   `prepare_finetune_dataset.py`: Converts Q&A to Alpaca format.
*   `train.py`: Unsloth training script for Qwen 3 8B.
*   `finetune_dataset.json`: The ready-to-use dataset for fine-tuning.
