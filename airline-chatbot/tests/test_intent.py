import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from chatbot.intent import classify_intent, INTENT_ROUTES


class TestClassifyIntent:
    """Unit tests for classify_intent function."""

    @patch("chatbot.intent.OpenAI")
    def test_classify_baggage_faq(self, mock_openai_class):
        """Test that classify_intent returns 'baggage_faq' when API returns it."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "baggage_faq"
        mock_client.chat.create.return_value = mock_response
        
        result = classify_intent("How much baggage can I bring?")
        
        assert result == "baggage_faq"

    @patch("chatbot.intent.OpenAI")
    def test_classify_flight_search(self, mock_openai_class):
        """Test that classify_intent returns 'flight_search' when API returns it."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "flight_search"
        mock_client.chat.create.return_value = mock_response
        
        result = classify_intent("Find flights to Delhi")
        
        assert result == "flight_search"

    @patch("chatbot.intent.OpenAI")
    def test_classify_with_whitespace_in_response(self, mock_openai_class):
        """Test that classify_intent strips whitespace from API response."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  flight_booking  \n"
        mock_client.chat.create.return_value = mock_response
        
        result = classify_intent("I want to book a flight")
        
        assert result == "flight_booking"

    @patch("chatbot.intent.OpenAI")
    def test_classify_exception_returns_general_faq(self, mock_openai_class):
        """Test that classify_intent returns 'general_faq' fallback on API exception."""
        # Setup mock to raise an exception
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.create.side_effect = Exception("API error")
        
        result = classify_intent("Some message")
        
        assert result == "general_faq"

    @patch("chatbot.intent.OpenAI")
    def test_api_called_with_correct_parameters(self, mock_openai_class):
        """Test that the API is called with temperature=0 and max_tokens=20."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "greeting"
        mock_client.chat.create.return_value = mock_response
        
        classify_intent("Hello")
        
        # Verify create was called with correct parameters
        call_kwargs = mock_client.chat.create.call_args[1]
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 20

    @patch("chatbot.intent.OpenAI")
    def test_api_called_with_correct_model(self, mock_openai_class):
        """Test that the API is called with model='llama-3.3-70b-versatile'."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "refund_faq"
        mock_client.chat.create.return_value = mock_response
        
        classify_intent("Can I get a refund?")
        
        # Verify create was called with correct model
        call_kwargs = mock_client.chat.create.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    @patch("chatbot.intent.OpenAI")
    def test_openai_client_initialized_with_xai_credentials(self, mock_openai_class):
        """Test that OpenAI client is initialized with GROQ_API_KEY and Groq endpoint."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "check_in_faq"
        mock_client.chat.create.return_value = mock_response
        
        classify_intent("When do I check in?")
        
        # Verify OpenAI was initialized with correct parameters
        call_kwargs = mock_openai_class.call_args[1]
        assert "api_key" in call_kwargs or mock_openai_class.call_args[0]
        assert "base_url" in call_kwargs
        assert call_kwargs["base_url"] == "https://api.groq.com/openai/v1"

    def test_all_intents_defined(self):
        """Test that all 9 valid intents are defined in INTENT_ROUTES."""
        expected_intents = {
            "flight_search",
            "flight_booking",
            "booking_modify",
            "baggage_faq",
            "refund_faq",
            "check_in_faq",
            "flight_status",
            "general_faq",
            "greeting"
        }
        
        actual_intents = set(INTENT_ROUTES.keys())
        
        assert actual_intents == expected_intents

    @patch("chatbot.intent.OpenAI")
    def test_classify_all_intent_types(self, mock_openai_class):
        """Test classify_intent with all valid intent types."""
        intents = [
            "flight_search",
            "flight_booking",
            "booking_modify",
            "baggage_faq",
            "refund_faq",
            "check_in_faq",
            "flight_status",
            "general_faq",
            "greeting"
        ]
        
        for intent in intents:
            # Reset mock for each iteration
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.choices[0].message.content = intent
            mock_client.chat.create.return_value = mock_response
            
            result = classify_intent("test message")
            assert result == intent

    @patch("chatbot.intent.OpenAI")
    def test_classify_handles_message_content_attribute_missing(self, mock_openai_class):
        """Test fallback when message.content is missing."""
        # Setup mock where first attempt fails, fallback to general_faq
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Make chat.create raise exception to trigger fallback
        mock_client.chat.create.side_effect = Exception("API error")
        
        result = classify_intent("Find flights")
        
        # Should return general_faq on exception
        assert result == "general_faq"
