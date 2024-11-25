from typing import Union
import numpy as np
from numpy.typing import NDArray
from typing import List
from chatbot_api.base import EmbeddingResponse
from chatbot_api.providers.openai import OpenAIProvider

def format_prompt(prompt: str, context: str = "") -> str:
    """Format the prompt with additional context."""
    return f"{context}\n{prompt}"

# Utility methods for working with embeddings
@staticmethod
def cosine_similarity(
    embedding1: Union[EmbeddingResponse, NDArray],
    embedding2: Union[EmbeddingResponse, NDArray]
) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Extract numpy arrays if EmbeddingResponse objects are provided
    if isinstance(embedding1, EmbeddingResponse):
        embedding1 = embedding1.embedding
    if isinstance(embedding2, EmbeddingResponse):
        embedding2 = embedding2.embedding
        
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


async def semantic_search(
    provider: OpenAIProvider,
    query: str,
    documents: List[str],
    top_k: int = 3
) -> List[tuple[str, float]]:
    """
    Perform semantic search on documents.
    
    Args:
        provider: OpenAIProvider instance
        query: Search query
        documents: List of documents to search
        top_k: Number of results to return
        
    Returns:
        List of (document, similarity_score) tuples
    """
    # Get query embedding
    query_embedding = await provider.get_embedding(query)
    
    # Get document embeddings
    doc_embeddings = await provider.get_embeddings(documents)
    
    # Calculate similarities
    similarities = [
        (doc, provider.cosine_similarity(query_embedding, emb))
        for doc, emb in zip(documents, doc_embeddings)
    ]
    
    # Sort by similarity and return top_k
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]