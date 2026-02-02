from chunker import chunk_pdf;
from embedder import get_embeddings;
from vectordb import insert_chunk;
import psycopg2;
import os;
from pgvector.psycopg2 import register_vector;

from dotenv import load_dotenv;
load_dotenv();

def ingest(pdf_path: str):
    chunks = chunk_pdf(pdf_path);
    embeddings = get_embeddings(chunks);
    conn = psycopg2.connect(os.getenv("DATABASE_URL"));
    register_vector(conn);
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        insert_chunk(conn, chunk, embedding, pdf_path, i);
    conn.close();

    print(f"Ingested {len(chunks)} chunks from {pdf_path}")