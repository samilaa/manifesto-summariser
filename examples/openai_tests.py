import sys
import os

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import asyncio
from chatbot_api.base import Message, Role
from chatbot_api.exceptions import LLMException, TokenLimitException
from chatbot_api.providers.openai import OpenAIProvider
import chatbot_api.utils as utils
from dotenv import load_dotenv
from pathlib import Path 

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")

async def embedding_test():
    # Initialize provider
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Example texts
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A lazy dog sleeps while a swift fox leaps nearby"
    
    try:
        # Get single embedding
        print("\nGetting single embedding...")
        embedding1 = await provider.get_embedding(text1)
        print(f"Embedding dimension: {len(embedding1)}")
        print(f"Tokens used: {embedding1.tokens}")
        print(f"Estimated cost: ${embedding1.estimated_cost:.6f}")
        
        # Get multiple embeddings
        print("\nGetting multiple embeddings...")
        embeddings = await provider.get_embeddings([text1, text2])
        
        # Calculate similarity
        similarity = utils.cosine_similarity(embeddings[0], embeddings[1])
        print(f"\nCosine similarity between texts: {similarity:.4f}")
        
        # Example of using embeddings for semantic search
        search_texts = [
            "A fox jumping",
            "A dog sleeping",
            "Something completely different"
        ]
        
        print("\nPerforming semantic search...")
        # Get embeddings for search texts
        search_embeddings = await provider.get_embeddings(search_texts)
        
        # Compare with our first text
        print(f"\nSimilarity scores with '{text1}':")
        for text, emb in zip(search_texts, search_embeddings):
            similarity = utils.cosine_similarity(embedding1, emb)
            print(f"'{text}': {similarity:.4f}")
        
    except LLMException as e:
        print(f"Error: {e}")

async def prompting_test():
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    messages = [
        Message(role=Role.USER, content="What is Python?")
    ]
    
    try:
        response = await provider.generate(messages)
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        
        if response.was_truncated:
            print("Warning: Response was truncated!")
    
    except TokenLimitException:
        print("Input was too long!")
    except LLMException as e:
        print(f"Error: {e}")

async def main():
    await embedding_test()

if __name__ == "__main__":
    asyncio.run(main())