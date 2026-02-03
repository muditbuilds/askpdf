from pypdf import PdfReader
import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]: 
    words = text.split()
    chunks = [];
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size]);
        chunks.append(chunk)
    return chunks

def chunk_pdf(pdf_path: str) -> list[str]:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text();
        return chunk_by_sentences(text)

def chunk_by_sentences(text:str, max_chunk_size: int = 500, overlap: int = 5) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_size + sentence_words > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_size = sum(len(s.split()) for s in current_chunk) if overlap > 0 else 0

        current_chunk.append(sentence)
        current_size += sentence_words

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks