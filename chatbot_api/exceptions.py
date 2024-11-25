from abc import ABC

class LLMException(Exception):
    """Base exception class for LLM-related errors."""
    pass

class TokenLimitException(LLMException):
    """Raised when input exceeds token limit."""
    pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

class APIKeyNotFoundError(LLMException):
    pass