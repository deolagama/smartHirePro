import fitz  
import io
def extract_text_from_pdf_bytes(pdf_bytes):
    pdf_file = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    doc.close()
    return text.strip()
