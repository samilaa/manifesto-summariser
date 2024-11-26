import sys
import os

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from typing import List
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import EmbeddingResponse, Message, Role
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")

# config
n_clusters = 5
n_comments = 15
response_max_tokens = 100
data_source_path = os.path.join(parent_dir, 'data', 'sources', 'kuntavaalit2021.csv')
question_columns = ['q1.explanation_fi', 'q2.explanation_fi']
provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

# Load the open comments from CSV
def load_statements(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    return df[question_columns[1]].dropna()[:n_comments].tolist()  

# Get embeddings for the political statements
async def get_embeddings_for_statements(statements: List[str]) -> List[EmbeddingResponse]:
    embeddings_responses = await provider.get_embeddings(statements)
    return embeddings_responses

# Perform clustering based on embeddings
def cluster_messages(statements: List[str], embeddings_responses: List[EmbeddingResponse], n_clusters: int = 6):
    embeddings = np.array([response.embedding for response in embeddings_responses])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Group statements by their cluster label
    clustered_statements = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clustered_statements[label].append(statements[idx])
    
    return clustered_statements

# Generate a summary for each cluster
async def summarize_cluster(statements: List[str]) -> str:
    prompt = (
        "The following are arguments from a cluster. Summarize them into a single representative statement:\n\n" +
        "\n".join(f"- {s}" for s in statements) +
        "\n\nSummarize these in one concise statement, distilling the essential message in the way that it is relevant to political decisions:"
    )
    
    messages = [Message(role=Role("user"), content=prompt)]
    response = await provider.generate(messages, temperature=0.7, max_tokens=response_max_tokens)
    return response.content

# Enhanced main function with summarization
async def main():
    # Load the statements from CSV
    statements = load_statements(data_source_path)
    
    # Get embeddings for each statement
    embeddings_responses = await get_embeddings_for_statements(statements)
    
    # Cluster the statements based on their embeddings
    clustered_messages = cluster_messages(statements, embeddings_responses, n_clusters=n_clusters)
    
    # Generate summaries for each cluster
    cluster_summaries = {}
    for cluster_id, cluster_statements in clustered_messages.items():
        print(f"Summarizing Cluster {cluster_id}...")
        cluster_summaries[cluster_id] = await summarize_cluster(cluster_statements)
    
    # Print clustered summaries
    for cluster_id, summary in cluster_summaries.items():
        print(f"Cluster {cluster_id} Summary:")
        print(summary)
        print()

# Run the enhanced main function
import asyncio
asyncio.run(main())
