import pytest
from unittest.mock import Mock, patch


class TestGetEmbeddings:
    @patch('embedder.openai.embeddings.create')
    def test_returns_embeddings_for_single_text(self, mock_create):
        # Setup mock response
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_create.return_value = mock_response
        
        from embedder import get_embeddings
        result = get_embeddings(["hello world"])
        
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    @patch('embedder.openai.embeddings.create')
    def test_returns_embeddings_for_multiple_texts(self, mock_create):
        # Setup mock response with multiple embeddings
        mock_emb1 = Mock()
        mock_emb1.embedding = [0.1] * 1536
        mock_emb2 = Mock()
        mock_emb2.embedding = [0.2] * 1536
        mock_emb3 = Mock()
        mock_emb3.embedding = [0.3] * 1536
        
        mock_response = Mock()
        mock_response.data = [mock_emb1, mock_emb2, mock_emb3]
        mock_create.return_value = mock_response
        
        from embedder import get_embeddings
        result = get_embeddings(["text1", "text2", "text3"])
        
        assert len(result) == 3
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        assert result[2][0] == 0.3

    @patch('embedder.openai.embeddings.create')
    def test_calls_openai_with_correct_model(self, mock_create):
        mock_embedding = Mock()
        mock_embedding.embedding = [0.0] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_create.return_value = mock_response
        
        from embedder import get_embeddings
        get_embeddings(["test"])
        
        mock_create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["test"]
        )

    @patch('embedder.openai.embeddings.create')
    def test_empty_list_returns_empty_result(self, mock_create):
        mock_response = Mock()
        mock_response.data = []
        mock_create.return_value = mock_response
        
        from embedder import get_embeddings
        result = get_embeddings([])
        
        assert result == []
