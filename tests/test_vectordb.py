import pytest
from unittest.mock import Mock, MagicMock
from vectordb import insert_chunk, search_chunks


class TestInsertChunk:
    def test_inserts_chunk_and_returns_id(self):
        # Setup mock connection and cursor
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (42,)
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        result = insert_chunk(
            conn=mock_conn,
            content="test content",
            embedding=[0.1] * 1536,
            source="test.pdf",
            chunk_index=0
        )
        
        assert result == 42
        mock_conn.commit.assert_called_once()

    def test_executes_correct_sql(self):
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        content = "test content"
        embedding = [0.1, 0.2, 0.3]
        source = "doc.pdf"
        chunk_index = 5
        
        insert_chunk(mock_conn, content, embedding, source, chunk_index)
        
        # Verify execute was called with INSERT statement
        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        
        assert "INSERT INTO chunks" in sql
        assert "RETURNING id" in sql
        assert params == (content, embedding, source, chunk_index)

    def test_commits_after_insert(self):
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        insert_chunk(mock_conn, "content", [0.1], "source", 0)
        
        # Verify commit was called on connection
        mock_conn.commit.assert_called_once()


class TestSearchChunks:
    def test_returns_matching_chunks(self):
        # Setup mock cursor with results
        mock_results = [
            (1, "content 1", "source1.pdf", 0),
            (2, "content 2", "source1.pdf", 1),
            (3, "content 3", "source2.pdf", 0),
        ]
        
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = mock_results
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        result = search_chunks(mock_conn, [0.1] * 1536, top_k=3)
        
        assert result == mock_results
        assert len(result) == 3

    def test_uses_cosine_distance_operator(self):
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        search_chunks(mock_conn, [0.1] * 1536, top_k=5)
        
        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        
        # Verify cosine distance operator is used
        assert "<=>" in sql

    def test_respects_top_k_parameter(self):
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        search_chunks(mock_conn, [0.1] * 1536, top_k=10)
        
        call_args = mock_cursor.execute.call_args
        params = call_args[0][1]
        
        # Second parameter should be top_k
        assert params[1] == 10

    def test_default_top_k_is_five(self):
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        search_chunks(mock_conn, [0.1] * 1536)
        
        call_args = mock_cursor.execute.call_args
        params = call_args[0][1]
        
        assert params[1] == 5

    def test_returns_empty_list_when_no_matches(self):
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        result = search_chunks(mock_conn, [0.1] * 1536)
        
        assert result == []
