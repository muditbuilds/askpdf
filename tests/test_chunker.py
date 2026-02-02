import pytest
from unittest.mock import Mock, patch, mock_open
from chunker import chunk_text, chunk_pdf


class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        result = chunk_text("")
        assert result == []

    def test_text_shorter_than_chunk_size(self):
        text = "hello world this is a test"
        result = chunk_text(text, chunk_size=100, overlap=10)
        assert len(result) == 1
        assert result[0] == text

    def test_chunking_creates_overlap(self):
        # Create text with exactly 20 words
        words = [f"word{i}" for i in range(20)]
        text = " ".join(words)
        
        result = chunk_text(text, chunk_size=10, overlap=2)
        
        # With chunk_size=10 and overlap=2, step is 8
        # First chunk: words 0-9
        # Second chunk: words 8-17
        # Third chunk: words 16-19
        assert len(result) == 3
        
        # Verify overlap exists - last 2 words of chunk 1 should be first 2 of chunk 2
        chunk1_words = result[0].split()
        chunk2_words = result[1].split()
        assert chunk1_words[-2:] == chunk2_words[:2]

    def test_chunk_size_parameter(self):
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        
        result = chunk_text(text, chunk_size=25, overlap=0)
        
        assert len(result) == 4
        for chunk in result:
            assert len(chunk.split()) == 25

    def test_overlap_parameter(self):
        words = [f"w{i}" for i in range(30)]
        text = " ".join(words)
        
        # chunk_size=15, overlap=5 means step=10
        result = chunk_text(text, chunk_size=15, overlap=5)
        
        # words 0-14, 10-24, 20-29
        assert len(result) == 3


class TestChunkPdf:
    @patch('chunker.PdfReader')
    @patch('builtins.open', mock_open())
    def test_extracts_text_from_all_pages(self, mock_reader):
        # Setup mock pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page one content. "
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page two content."
        
        mock_reader.return_value.pages = [mock_page1, mock_page2]
        
        result = chunk_pdf("test.pdf")
        
        # Should have combined text from both pages
        assert len(result) >= 1
        assert "Page one content" in result[0]

    @patch('chunker.PdfReader')
    @patch('builtins.open', mock_open())
    def test_empty_pdf_returns_empty_list(self, mock_reader):
        mock_reader.return_value.pages = []
        
        result = chunk_pdf("empty.pdf")
        
        assert result == []

    @patch('chunker.PdfReader')
    def test_file_not_found_raises_error(self, mock_reader):
        with pytest.raises(FileNotFoundError):
            chunk_pdf("nonexistent.pdf")
