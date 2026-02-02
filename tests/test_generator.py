import pytest
from unittest.mock import Mock, patch


class TestGenerateAnswer:
    @patch('generator.openai.chat.completions.create')
    def test_returns_llm_response(self, mock_create):
        mock_message = Mock()
        mock_message.content = "The answer is 42."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        from generator import generate_answer
        result = generate_answer(["some context"], "What is the answer?")
        
        assert result == "The answer is 42."

    @patch('generator.openai.chat.completions.create')
    def test_uses_correct_model(self, mock_create):
        mock_message = Mock()
        mock_message.content = "response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        from generator import generate_answer
        generate_answer(["context"], "query")
        
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"

    @patch('generator.openai.chat.completions.create')
    def test_includes_system_message(self, mock_create):
        mock_message = Mock()
        mock_message.content = "response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        from generator import generate_answer
        generate_answer(["context"], "query")
        
        call_kwargs = mock_create.call_args[1]
        messages = call_kwargs["messages"]
        
        # First message should be system
        assert messages[0]["role"] == "system"
        assert "context" in messages[0]["content"].lower()

    @patch('generator.openai.chat.completions.create')
    def test_includes_context_in_user_message(self, mock_create):
        mock_message = Mock()
        mock_message.content = "response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        from generator import generate_answer
        generate_answer(["chunk one", "chunk two"], "my question")
        
        call_kwargs = mock_create.call_args[1]
        messages = call_kwargs["messages"]
        user_content = messages[1]["content"]
        
        assert "chunk one" in user_content
        assert "chunk two" in user_content

    @patch('generator.openai.chat.completions.create')
    def test_includes_query_in_user_message(self, mock_create):
        mock_message = Mock()
        mock_message.content = "response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        from generator import generate_answer
        generate_answer(["context"], "What is machine learning?")
        
        call_kwargs = mock_create.call_args[1]
        messages = call_kwargs["messages"]
        user_content = messages[1]["content"]
        
        assert "What is machine learning?" in user_content

    @patch('generator.openai.chat.completions.create')
    def test_joins_context_with_double_newlines(self, mock_create):
        mock_message = Mock()
        mock_message.content = "response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        from generator import generate_answer
        generate_answer(["first chunk", "second chunk", "third chunk"], "query")
        
        call_kwargs = mock_create.call_args[1]
        messages = call_kwargs["messages"]
        user_content = messages[1]["content"]
        
        # Chunks should be separated by double newlines
        assert "first chunk\n\nsecond chunk\n\nthird chunk" in user_content

    @patch('generator.openai.chat.completions.create')
    def test_handles_empty_context(self, mock_create):
        mock_message = Mock()
        mock_message.content = "I don't have context to answer."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response
        
        from generator import generate_answer
        result = generate_answer([], "question without context")
        
        # Should still work, just with empty context
        assert result == "I don't have context to answer."
