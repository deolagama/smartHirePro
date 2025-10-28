import fitz
from datasets import load_dataset, Dataset, DatasetDict
import os
from langdetect import detect, LangDetectException
import pyarrow

DATASET_NAME = "d4rk3r/resumes-raw-pdf"
OUTPUT_DATASET_PATH = "../data/processed_resumes_en"

def extract_text_from_pdf(pdf_input, input_type='bytes'):
    all_text = ""
    doc = None
    try:
        if input_type == 'bytes':
            doc = fitz.open(stream=pdf_input, filetype="pdf")
        elif input_type == 'path':
            doc = fitz.open(pdf_input)
        if doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                all_text += page_text + "\n\n"
            doc.close()
        return all_text.strip()
    except Exception as e:
        if doc:
            doc.close()
        print(f"Error processing PDF: {e}")
        return None

def main():
    print(f"Loading dataset '{DATASET_NAME}'...")
    try:
        ds = load_dataset(DATASET_NAME, split='train')
    except Exception as e:
        print(f"Failed to load dataset. Have you logged in with `huggingface-cli login`?")
        print(f"Error: {e}")
        return

    print(f"Dataset loaded. Processing ALL samples...")
    print(f"Processed dataset will be saved to '{OUTPUT_DATASET_PATH}/' folder.")

    processed_data_list = []

    for i, item in enumerate(ds):
        print(f"\n" + "="*50)
        print(f"Processing Sample {i+1}")
        original_filename = item.get('file_name', f"sample_{i+1}.pdf")
        print(f"Source file: {original_filename}")
        print("="*50)

        pdf_data = item['pdf']
        pdf_bytes = None
        pdf_path = None
        raw_text = None

        if pdf_data:
            if hasattr(pdf_data, 'bytes') and pdf_data.bytes:
                pdf_bytes = pdf_data.bytes
            elif hasattr(pdf_data, 'path') and pdf_data.path:
                pdf_path = pdf_data.path

        if pdf_bytes:
            raw_text = extract_text_from_pdf(pdf_input=pdf_bytes, input_type='bytes')
        elif pdf_path:
            raw_text = extract_text_from_pdf(pdf_input=pdf_path, input_type='path')
        else:
            print("--- Skipping (no PDF bytes or path found in item) ---")
            continue

        if not raw_text:
            print("--- Skipping (no text extracted) ---")
            continue

        try:
            if detect(raw_text) == 'en':
                print("--- Language: English ---")
                processed_data_list.append({
                    "clean_text": raw_text,
                    "original_file": original_filename
                })
                print("Added to dataset list")
            else:
                print("Skipping (non-English)")
        except LangDetectException:
            print("Skipping (language detection failed, likely too little text)")

    print("\n" + "="*50)
    print("All processing complete.")
    if not processed_data_list:
        print("No English resumes were found. Output dataset will be empty.")
        return

    print(f"Creating Hugging Face Dataset from {len(processed_data_list)} processed resumes...")
    final_dataset = Dataset.from_list(processed_data_list)
    dataset_dict = DatasetDict({"train": final_dataset})
    dataset_dict.save_to_disk(OUTPUT_DATASET_PATH)
    print(f"--- Successfully saved dataset to {OUTPUT_DATASET_PATH} ---")

if __name__ == "__main__":
    main()
