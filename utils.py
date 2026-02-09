import io
from pypdf import PdfReader
import docx

def parse_pdf(file_bytes):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def parse_docx(file_bytes):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"

def get_text_from_upload(uploaded_file):
    """Dispatch to correct parser based on file type."""
    if uploaded_file is None:
        return ""
    
    file_bytes = uploaded_file.read()
    
    if uploaded_file.name.lower().endswith('.pdf'):
        return parse_pdf(file_bytes)
    elif uploaded_file.name.lower().endswith('.docx'):
        return parse_docx(file_bytes)
    else:
        # Assume text
        try:
            return file_bytes.decode('utf-8')
        except:
            return "Unsupported file format."
