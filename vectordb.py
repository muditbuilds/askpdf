def insert_chunk(conn, content, embedding, source, chunk_index):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (content, embedding, source, chunk_index)
        VALUES (%s, %s, %s, %s) RETURNING id;
    """, (content, embedding, source, chunk_index));
    conn.commit();
    return cursor.fetchone()[0];

def search_chunks(conn, query_embedding : list[float], top_k : int = 5):
    cursor = conn.cursor()
    query = """
    SELECT id, content, source, chunk_index
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    cursor.execute(query, (query_embedding, top_k));
    return cursor.fetchall();