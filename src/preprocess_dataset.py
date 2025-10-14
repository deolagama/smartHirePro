import fitz
import re
import io
from PIL import Image
import pytesseract
from datasets import load_dataset
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

ds = load_dataset("d4rk3r/resumes-raw-pdf")

def extract_text_from_pdf_or_image(pdf_bytes):
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
            if text.strip():
                return text
    except Exception as e:
        pass  
    try:
        image = Image.open(io.BytesIO(pdf_bytes)) #Convert raw bytes → in-memory file,uses Pillow (PIL) to open that in-memory file as an image.
        text = pytesseract.image_to_string(image) #eads the image and uses machine learning to detect and extract the text printed in it
        return text
    except Exception as e:
        return ""

#We use PDF bytes because it lets us read and process PDFs directly from memory, without saving files, making the workflow faster, scalable, and cleaner .
def preprocess_pdf(example): #Each dictionary (row) represents one resume in your case.
    pdf_bytes = example["pdf"] 
    text = extract_text_from_pdf_bytes(pdf_bytes)
    example['raw_text'] = text
    return example

ds = ds.map(preprocess_pdf)
def clean_text(text):
    """Basic cleaning of resume text."""
    text = re.sub(r'\S+@\S+', ' ', text)  # remove emails
    text = re.sub(r'\+?\d[\d\s-]{8,}\d', ' ', text)  # remove phone numbers
    text = re.sub(r'[^A-Za-z\s]', ' ', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    return text.lower()
def preprocess_pdf(example):
    pdf_bytes = example["pdf"]
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not raw_text or len(raw_text) < 50:
        example["keep"] = False
        return example
    try:
        lang = detect(raw_text)
    except:
        lang = "unknown"
    if lang != "en":  
        example["keep"] = False
        return example
    
    cleaned_text = clean_text(raw_text)
    
    example["raw_text"] = raw_text
    example["clean_text"] = cleaned_text
    example["keep"] = True
    return example

ds = ds.map(preprocess_pdf)
ds_filtered = ds.filter(lambda x: x["keep"])
ds_filtered.save_to_disk("../data/processed_resumes_en")

print("Preprocessing complete, saved English resumes to 'processed_resumes_en/'")