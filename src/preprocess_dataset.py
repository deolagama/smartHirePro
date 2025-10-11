from datasets import load_dataset

ds = load_dataset("d4rk3r/resumes-raw-pdf")

def preprocess_pdf(example):
    pdf_bytes = example['content']  # adjust column name if needed
    text = extract_text_from_pdf_bytes(pdf_bytes)
    example['raw_text'] = text
    return example

ds = ds.map(preprocess_pdf)