from embedder import get_embeddings;
from vectordb import search_chunks;

def retrieve(conn, query: str, top_k: int = 5):
    embeddings = get_embeddings([query]);
    chunk_tuples = search_chunks(conn, embeddings[0], top_k);
    return [chunk_tuple[1] for chunk_tuple in chunk_tuples];