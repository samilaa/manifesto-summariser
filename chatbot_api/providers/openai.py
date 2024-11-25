import os
from openai import OpenAI, OpenAIError
from typing import List, Optional
from chatbot_api.base import LLMProvider, LLMResponse, Message, UsageStats, EmbeddingResponse
from ..base import (
    LLMProvider,
    LLMResponse,
    Message,
    UsageStats
)
from ..exceptions import (
    LLMException, 
    TokenLimitException,
    APIKeyNotFoundError
)

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", embedding_model: str = "text-embedding-ada-002"):
        
        self.api_key = api_key # Get from environment if not provided
        if not self.api_key:
            raise APIKeyNotFoundError(
                "No API key provided. Either pass it to the constructor or set the OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model

    async def count_tokens(self, text: str) -> int:
        try:
            response = await self.client.tools.tokenize.create(
                model=self.model,
                texts=[text]
            )
            return response.data[0].length
            
        except OpenAIError as e:
            raise LLMException(str(e)) from e
        
    async def max_context_tokens(self) -> int:
        try:
            response = await self.client.models.retrieve(model=self.model)
            return response.max_tokens
            
        except OpenAIError as e:
            raise LLMException(str(e)) from e
    
    def estimated_cost(self, prompt_tokens, completion_tokens) -> float:
        input_cost_per_1k = 0.0015
        output_cost_per_1k = 0.002
        
        input_cost = (prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (completion_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost

    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )
            
            usage = UsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=usage,
                model=self.model,
                finish_reason=response.choices[0].finish_reason
            )
            
        except OpenAIError as e:
            if "maximum context length" in str(e):
                raise TokenLimitException(str(e)) from e
            raise LLMException(str(e)) from e
        
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingResponse:
        """
        Get embedding vector for a text string.
        
        Args:
            text: Text to get embedding for
            model: Optional model override (defaults to self.embedding_model)
            
        Returns:
            EmbeddingResponse containing the embedding vector and metadata
            
        Raises:
            LLMException: If the API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=model or self.embedding_model,
                input=text
            )
            
            return EmbeddingResponse(
                embedding=response.data[0].embedding,
                tokens=response.usage.total_tokens,
                model=response.model
            )
            
        except Exception as e:
            raise LLMException(f"OpenAI embedding error: {str(e)}") from e
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[EmbeddingResponse]:
        """
        Get embedding vectors for multiple texts.
        
        Args:
            texts: List of texts to get embeddings for
            model: Optional model override (defaults to self.embedding_model)
            
        Returns:
            List of EmbeddingResponse objects
            
        Raises:
            LLMException: If the API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=model or self.embedding_model,
                input=texts
            )
            
            # Create EmbeddingResponse objects for each embedding
            results = []
            tokens_per_response = response.usage.total_tokens // len(texts)
            
            for data in response.data:
                results.append(
                    EmbeddingResponse(
                        embedding=data.embedding,
                        tokens=tokens_per_response,  # Approximate tokens per text
                        model=response.model
                    )
                )
            
            return results
            
        except Exception as e:
            raise LLMException(f"OpenAI embedding error: {str(e)}") from e