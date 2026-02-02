import pytest
from unittest.mock import Mock, patch, MagicMock


class TestIngest:
    @patch('ingest.insert_chunk')
    @patch('ingest.register_vector')
    @patch('ingest.psycopg2.connect')
    @patch('ingest.get_embeddings')
    @patch('ingest.chunk_pdf')
    def test_ingests_pdf_and_stores_chunks(
        self, mock_chunk_pdf, mock_get_embeddings, mock_connect, 
        mock_register_vector, mock_insert_chunk
    ):
        # Setup mocks
        mock_chunk_pdf.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_get_embeddings.return_value = [[0.1]*1536, [0.2]*1536, [0.3]*1536]
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        from ingest import ingest
        ingest("test.pdf")
        
        # Verify chunk_pdf was called with the path
        mock_chunk_pdf.assert_called_once_with("test.pdf")
        
        # Verify embeddings were generated for chunks
        mock_get_embeddings.assert_called_once_with(["chunk1", "chunk2", "chunk3"])
        
        # Verify database connection was made
        mock_connect.assert_called_once()
        
        # Verify vector type was registered
        mock_register_vector.assert_called_once_with(mock_conn)
        
        # Verify each chunk was inserted
        assert mock_insert_chunk.call_count == 3
        
        # Verify connection was closed
        mock_conn.close.assert_called_once()

    @patch('ingest.insert_chunk')
    @patch('ingest.register_vector')
    @patch('ingest.psycopg2.connect')
    @patch('ingest.get_embeddings')
    @patch('ingest.chunk_pdf')
    def test_inserts_chunks_with_correct_parameters(
        self, mock_chunk_pdf, mock_get_embeddings, mock_connect,
        mock_register_vector, mock_insert_chunk
    ):
        mock_chunk_pdf.return_value = ["content A", "content B"]
        mock_get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        from ingest import ingest
        ingest("document.pdf")
        
        # Verify insert_chunk called with correct args for each chunk
        calls = mock_insert_chunk.call_args_list
        
        # First chunk
        assert calls[0][0] == (mock_conn, "content A", [0.1, 0.2], "document.pdf", 0)
        
        # Second chunk
        assert calls[1][0] == (mock_conn, "content B", [0.3, 0.4], "document.pdf", 1)

    @patch('ingest.insert_chunk')
    @patch('ingest.register_vector')
    @patch('ingest.psycopg2.connect')
    @patch('ingest.get_embeddings')
    @patch('ingest.chunk_pdf')
    def test_handles_empty_pdf(
        self, mock_chunk_pdf, mock_get_embeddings, mock_connect,
        mock_register_vector, mock_insert_chunk
    ):
        mock_chunk_pdf.return_value = []
        mock_get_embeddings.return_value = []
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        from ingest import ingest
        ingest("empty.pdf")
        
        # No chunks to insert
        mock_insert_chunk.assert_not_called()
        
        # Connection still closed
        mock_conn.close.assert_called_once()

    @patch('ingest.insert_chunk')
    @patch('ingest.register_vector')
    @patch('ingest.psycopg2.connect')
    @patch('ingest.get_embeddings')
    @patch('ingest.chunk_pdf')
    def test_uses_pdf_path_as_source(
        self, mock_chunk_pdf, mock_get_embeddings, mock_connect,
        mock_register_vector, mock_insert_chunk
    ):
        mock_chunk_pdf.return_value = ["chunk"]
        mock_get_embeddings.return_value = [[0.1]]
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        from ingest import ingest
        ingest("/path/to/my/document.pdf")
        
        # Source should be the full pdf path
        call_args = mock_insert_chunk.call_args[0]
        assert call_args[3] == "/path/to/my/document.pdf"

    @patch('ingest.insert_chunk')
    @patch('ingest.register_vector')
    @patch('ingest.psycopg2.connect')
    @patch('ingest.get_embeddings')
    @patch('ingest.chunk_pdf')
    def test_chunk_indices_are_sequential(
        self, mock_chunk_pdf, mock_get_embeddings, mock_connect,
        mock_register_vector, mock_insert_chunk
    ):
        mock_chunk_pdf.return_value = ["a", "b", "c", "d"]
        mock_get_embeddings.return_value = [[0.1], [0.2], [0.3], [0.4]]
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        from ingest import ingest
        ingest("test.pdf")
        
        # Verify chunk indices are 0, 1, 2, 3
        calls = mock_insert_chunk.call_args_list
        indices = [call[0][4] for call in calls]
        assert indices == [0, 1, 2, 3]
