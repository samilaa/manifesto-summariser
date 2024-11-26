class LLMException(Exception):
    """Base exception class for LLM-related errors."""
    pass

class TokenLimitException(LLMException):
    """Raised when input exceeds token limit."""
    pass

class APIKeyNotFoundError(LLMException):
    pass