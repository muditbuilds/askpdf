from pypdf import PdfReader;

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]: 
    words = text.split();
    chunks = [];
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size]);
        chunks.append(chunk);
    return chunks;

def chunk_pdf(pdf_path: str) -> list[str]:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file);
        text = '';
        for page in pdf_reader.pages:
            text += page.extract_text();
        return chunk_text(text);