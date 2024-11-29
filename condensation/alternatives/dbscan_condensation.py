import sys
import os

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple
from enum import Enum
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import Message, Role
from dotenv import load_dotenv

class StanceCategory(Enum):
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"
    CONDITIONAL = "conditional"
    ALTERNATIVE = "alternative"

class ArgumentClusterer:
    def __init__(
        self,
        api_key: str,
        min_cluster_size: int = 5,
        max_clusters: int = 10,
        response_max_tokens: int = 150,
        outlier_eps: float = 0.7,  # Increased from 0.5
        outlier_min_samples: int = 2,  # Decreased from 3
        n_samples: int = 200
    ):
        self.provider = OpenAIProvider(api_key=api_key)
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.response_max_tokens = response_max_tokens
        self.outlier_eps = outlier_eps
        self.outlier_min_samples = outlier_min_samples
        self.n_samples = n_samples
        
    async def get_embeddings(self, statements: List[str]) -> np.ndarray:
        """Get embeddings with error handling and retries"""
        embeddings_responses = await self.provider.get_embeddings(statements)
        return np.array([response.embedding for response in embeddings_responses])

    def determine_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int = 2  # Ensure at least 2 clusters
    ) -> Tuple[int, float]:
        """Use silhouette analysis to determine optimal number of clusters"""
        n_samples = len(embeddings)
        if n_samples < min_clusters:
            return min_clusters, 0.0
            
        max_possible_clusters = min(self.max_clusters, n_samples - 1)
        best_score = -1
        optimal_n = min_clusters
        
        for n in range(min_clusters, max_possible_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    optimal_n = n
            except Exception as e:
                print(f"Warning: Error during clustering with {n} clusters: {str(e)}")
                continue
                
        return optimal_n, best_score

    def identify_outliers(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Identify outlier arguments using DBSCAN with adaptive parameters"""
        if len(embeddings) < self.outlier_min_samples:
            return np.zeros(len(embeddings), dtype=bool)
            
        try:
            dbscan = DBSCAN(
                eps=self.outlier_eps,
                min_samples=self.outlier_min_samples,
                n_jobs=-1
            )
            labels = dbscan.fit_predict(embeddings)
            outliers = labels == -1
            
            # If too many outliers (>50%), treat none as outliers
            if np.mean(outliers) > 0.5:
                return np.zeros(len(embeddings), dtype=bool)
                
            return outliers
        except Exception as e:
            print(f"Warning: Error during outlier detection: {str(e)}")
            return np.zeros(len(embeddings), dtype=bool)

    async def analyze_cluster_stance(
        self,
        statements: List[str]
    ) -> StanceCategory:
        """Determine the primary stance of a cluster"""
        prompt = (
            "Analyze these related political arguments and categorize their primary stance as either "
            "'support', 'oppose', 'neutral', 'conditional', or 'alternative':\n\n"
            + "\n".join(f"- {s}" for s in statements)
            + "\n\nPrimary stance (just return the category word):"
        )
        messages = [Message(role=Role("user"), content=prompt)]
        response = await self.provider.generate(
            messages,
            temperature=0.3,
            max_tokens=20
        )
        try:
            return StanceCategory(response.content.strip().lower())
        except ValueError:
            return StanceCategory.NEUTRAL

    async def generate_cluster_summary(
        self,
        statements: List[str],
        stance: StanceCategory
    ) -> str:
        """Generate a nuanced summary incorporating stance analysis"""
        prompt = (
            f"These arguments share a {stance.value} stance on the issue. "
            "Summarize them into a representative statement that captures:\n"
            "1. The core position\n"
            "2. Key supporting reasoning\n"
            "3. Any notable conditions or nuances\n\n"
            + "\n".join(f"- {s}" for s in statements)
            + "\n\nSummary:"
        )
        messages = [Message(role=Role("user"), content=prompt)]
        response = await self.provider.generate(
            messages,
            temperature=0.7,
            max_tokens=self.response_max_tokens
        )
        return response.content

    async def cluster_and_analyze(
        self,
        statements: List[str]
    ) -> Dict:
        """Main clustering and analysis pipeline"""
        if not statements:
            raise ValueError("No statements provided")
            
        # Get embeddings
        embeddings = await self.get_embeddings(statements)
        
        # Scale embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # Identify outliers
        outliers = self.identify_outliers(scaled_embeddings)
        valid_indices = ~outliers
        
        # Handle case where all points are outliers
        if not np.any(valid_indices):
            print("Warning: All points identified as outliers. Proceeding with all points.")
            valid_indices = np.ones(len(statements), dtype=bool)
            outliers = np.zeros(len(statements), dtype=bool)
        
        # Get valid data
        valid_embeddings = scaled_embeddings[valid_indices]
        valid_statements = [s for i, s in enumerate(statements) if valid_indices[i]]
        
        # Ensure we have enough data for clustering
        if len(valid_statements) < 2:
            raise ValueError(f"Not enough valid statements for clustering. Only {len(valid_statements)} remaining after outlier detection.")
        
        # Determine optimal clusters
        n_clusters, silhouette = self.determine_optimal_clusters(valid_embeddings)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(valid_embeddings)
        
        # Group statements by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(valid_statements[idx])
            
        # Analyze each cluster
        results = {
            'metadata': {
                'total_arguments': len(statements),
                'outliers': sum(outliers),
                'clusters': n_clusters,
                'silhouette_score': silhouette
            },
            'clusters': {},
            'outliers': [s for i, s in enumerate(statements) if outliers[i]]
        }
        
        # Analyze each cluster
        for cluster_id, cluster_statements in clusters.items():
            stance = await self.analyze_cluster_stance(cluster_statements)
            summary = await self.generate_cluster_summary(cluster_statements, stance)
            
            results['clusters'][cluster_id] = {
                'stance': stance.value,
                'summary': summary,
                'size': len(cluster_statements),
                'statements': cluster_statements
            }
            
        return results

async def main():
    # Load environment variables
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Initialize clusterer with more lenient outlier detection
    clusterer = ArgumentClusterer(
        api_key=os.getenv("OPENAI_API_KEY"),
        min_cluster_size=2,  # Reduced from 3
        max_clusters=8,
        outlier_eps=0.7,     # Increased from 0.5
        outlier_min_samples=2,
        n_samples=200  # Reduced from 3
    )
    
    # Load and process data
    df = pd.read_csv('data/sources/kuntavaalit2021.csv')
    statements = df['q2.explanation_fi'].dropna().tolist()[:clusterer.n_samples]
    
    try:
        # Run analysis
        results = await clusterer.cluster_and_analyze(statements)
        
        # Print results
        print(f"\nAnalysis Results:")
        print(f"Total Arguments: {results['metadata']['total_arguments']}")
        print(f"Number of Clusters: {results['metadata']['clusters']}")
        print(f"Outliers: {results['metadata']['outliers']}")
        print(f"Silhouette Score: {results['metadata']['silhouette_score']:.3f}\n")
        
        for cluster_id, cluster_data in sorted(
            results['clusters'].items(),
            key=lambda x: x[1]['size'],
            reverse=True
        ):
            print(f"\nCluster {cluster_id}:")
            print(f"Stance: {cluster_data['stance']}")
            print(f"Size: {cluster_data['size']} arguments")
            print(f"Summary: {cluster_data['summary']}\n")
            
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())