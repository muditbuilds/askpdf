import pytest
from unittest.mock import Mock, patch


class TestRetrieve:
    @patch('retriever.search_chunks')
    @patch('retriever.get_embeddings')
    def test_retrieves_relevant_chunks(self, mock_get_embeddings, mock_search_chunks):
        mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_search_chunks.return_value = [
            (1, "content about AI", "doc1.pdf", 0),
            (2, "more AI content", "doc1.pdf", 1),
        ]
        mock_conn = Mock()
        
        from retriever import retrieve
        result = retrieve(mock_conn, "What is AI?")
        
        assert result == ["content about AI", "more AI content"]

    @patch('retriever.search_chunks')
    @patch('retriever.get_embeddings')
    def test_passes_query_to_embeddings(self, mock_get_embeddings, mock_search_chunks):
        mock_get_embeddings.return_value = [[0.1]]
        mock_search_chunks.return_value = []
        mock_conn = Mock()
        
        from retriever import retrieve
        retrieve(mock_conn, "my search query")
        
        # Query should be wrapped in a list
        mock_get_embeddings.assert_called_once_with(["my search query"])

    @patch('retriever.search_chunks')
    @patch('retriever.get_embeddings')
    def test_passes_embedding_to_search(self, mock_get_embeddings, mock_search_chunks):
        mock_get_embeddings.return_value = [[0.5, 0.6, 0.7]]
        mock_search_chunks.return_value = []
        mock_conn = Mock()
        
        from retriever import retrieve
        retrieve(mock_conn, "test query")
        
        # First embedding should be passed to search
        call_args = mock_search_chunks.call_args[0]
        assert call_args[0] == mock_conn
        assert call_args[1] == [0.5, 0.6, 0.7]

    @patch('retriever.search_chunks')
    @patch('retriever.get_embeddings')
    def test_respects_top_k_parameter(self, mock_get_embeddings, mock_search_chunks):
        mock_get_embeddings.return_value = [[0.1]]
        mock_search_chunks.return_value = []
        mock_conn = Mock()
        
        from retriever import retrieve
        retrieve(mock_conn, "query", top_k=10)
        
        call_args = mock_search_chunks.call_args[0]
        assert call_args[2] == 10

    @patch('retriever.search_chunks')
    @patch('retriever.get_embeddings')
    def test_default_top_k_is_five(self, mock_get_embeddings, mock_search_chunks):
        mock_get_embeddings.return_value = [[0.1]]
        mock_search_chunks.return_value = []
        mock_conn = Mock()
        
        from retriever import retrieve
        retrieve(mock_conn, "query")
        
        call_args = mock_search_chunks.call_args[0]
        assert call_args[2] == 5

    @patch('retriever.search_chunks')
    @patch('retriever.get_embeddings')
    def test_returns_empty_list_when_no_matches(self, mock_get_embeddings, mock_search_chunks):
        mock_get_embeddings.return_value = [[0.1]]
        mock_search_chunks.return_value = []
        mock_conn = Mock()
        
        from retriever import retrieve
        result = retrieve(mock_conn, "obscure query")
        
        assert result == []

    @patch('retriever.search_chunks')
    @patch('retriever.get_embeddings')
    def test_extracts_only_content_from_tuples(self, mock_get_embeddings, mock_search_chunks):
        mock_get_embeddings.return_value = [[0.1]]
        # Tuples are (id, content, source, chunk_index)
        mock_search_chunks.return_value = [
            (99, "the actual content", "ignored_source.pdf", 42),
        ]
        mock_conn = Mock()
        
        from retriever import retrieve
        result = retrieve(mock_conn, "query")
        
        # Should only get content, not id/source/chunk_index
        assert result == ["the actual content"]
        assert 99 not in result
        assert "ignored_source.pdf" not in result
