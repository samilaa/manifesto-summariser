import sys
import os

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import logging
import pandas as pd
from deep_translator import GoogleTranslator
import numpy as np   
import networkx as nx
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dataclasses import dataclass
from chatbot_api.base import LLMProvider
from chatbot_api.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from pathlib import Path 

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GroupingConfig:
    save_path: str = 'condensation/results/newest_save_path.txt'
    n_test: int = 50
    top_k: int = 10
    batch_size: int = 32
    similarity_threshold: float = 0.93  # Increased threshold for stricter matching
    transitivity_depth: int = 2  # Reduced depth to prevent over-clustering
    csv_path: str = 'data/sources/kuntavaalit2021.csv'
    openai_api_key = os.getenv("OPENAI_API_KEY")

class ArgumentGroupingPipeline:
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: GroupingConfig,
        translator
    ):
        self.llm = llm_provider
        self.config = config
        self.translator = translator
        self.original_explanations = None
        
    async def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[int, int]]:
        """Prepare and split data into sentences"""
        explanations = df['q1.explanation_fi'].dropna().reset_index(drop=True)[:self.config.n_test]
        explanation_indices = df['q1.explanation_fi'].dropna().index[:self.config.n_test]
        answers = df['q1.answer'][explanation_indices]
        explanations_fin = explanations.to_numpy()
        
        self.original_explanations = explanations_fin
        
        # Track sentence origins
        sentence_to_original = {}
        all_sentences_fin = []
        
        # Split into sentences and keep track of original explanation
        for idx, answer in enumerate(explanations_fin):
            sentences = [s.strip() for s in answer.split('.') if s.strip()]
            for sentence in sentences:
                sentence_to_original[len(all_sentences_fin)] = idx
                all_sentences_fin.append(sentence)
        
        logger.info(f"Processed {len(explanations_fin)} explanations into {len(all_sentences_fin)} sentences")
        return explanations_fin, all_sentences_fin, sentence_to_original
    
    def build_similarity_graph(self, embeddings: np.ndarray) -> nx.Graph:
        """Build similarity graph from embeddings with debugging"""
        similarities = cosine_similarity(embeddings)
        
        G = nx.Graph()
        n_sentences = len(embeddings)
        edge_count = 0
        
        # Add edges for similar sentences
        for i in range(n_sentences):
            for j in range(i + 1, n_sentences):
                similarity = similarities[i, j]
                if similarity >= self.config.similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
                    edge_count += 1
        
        logger.info(f"Built similarity graph with {n_sentences} nodes and {edge_count} edges")
        logger.info(f"Similarity stats - Min: {similarities.min():.3f}, Max: {similarities.max():.3f}, Mean: {similarities.mean():.3f}")
        return G
    
    async def translate_sentences(self, sentences: List[str]) -> List[str]:
        """Translate sentences to English"""
        translated = []
        for sentence in tqdm(sentences, desc="Translating"):
            translated_text = self.translator.translate(sentence)
            if not translated_text or not translated_text.strip():
                raise ValueError(f"Invalid translation for sentence: {sentence}")
            translated.append(translated_text)
        return translated
    
    def find_groups(self, G: nx.Graph, sentence_to_original: Dict[int, int]) -> List[List[int]]:
        """Find groups of similar arguments using connected components"""
        # First get the connected components from the graph
        components = list(nx.connected_components(G))
        logger.info(f"Found {len(components)} connected components in the graph")
        
        # Convert sentence components to explanation groups
        grouped_explanations = []
        for component in components:
            # Get unique explanation indices for this component
            explanation_indices = set(sentence_to_original[sent_idx] for sent_idx in component)
            
            # Optional: do you want to keep groups of size 1?
            # if len(explanation_indices) > 1:
            grouped_explanations.append(sorted(list(explanation_indices)))
        
        logger.info(f"Consolidated into {len(grouped_explanations)} explanation groups")
        return grouped_explanations
    
    async def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for all sentences using the LLM provider"""
        embeddings_response = await self.llm.get_embeddings(sentences)
        # Convert list of embedding responses to numpy array
        embeddings = np.array([resp.embedding for resp in embeddings_response])
        return embeddings

    def save_results(self, groups: List[List[int]], save_path: str):
        """Save groups to a text file for qualitative analysis"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"Found {len(groups)} groups of similar arguments\n")
            f.write(f"Largest group size: {max(len(group) for group in groups)}\n")
            f.write(f"Smallest group size: {min(len(group) for group in groups)}\n")
            f.write(f"Transitivity depth: {self.config.transitivity_depth}\n")
            f.write(f"Using similarity threshold: {self.config.similarity_threshold}\n")
            f.write(f"Using top-k: {self.config.top_k}\n")
            f.write(f"Number of explanations left out: {len(self.original_explanations) - sum(len(group) for group in groups)}\n")
            f.write(f"Percentage of explanations grouped: {sum(len(group) for group in groups) / len(self.original_explanations) * 100:.2f}%\n")
            f.write("=" * 80 + "\n\n")
            
            for i, group in enumerate(groups, 1):
                f.write(f"Group {i}:\n")
                f.write("-" * 40 + "\n")
                
                # Write each explanation in the group
                for j, idx in enumerate(group, 1):
                    explanation = self.original_explanations[idx]
                    f.write(f"Argument {j}:\n{explanation}\n\n")
                
                f.write("=" * 80 + "\n\n")

    async def process(self) -> List[List[int]]:
        """Run the complete pipeline"""
        # Load and prepare data
        df = pd.read_csv(self.config.csv_path)
        _, all_sentences_fin, sentence_to_original = await self.prepare_data(df)
        
        # Translate
        all_sentences_en = await self.translate_sentences(all_sentences_fin)
        
        # Get embeddings
        embeddings = await self.get_embeddings(all_sentences_en)
        
        # Calculate similarities
        similarities = cosine_similarity(embeddings)
        
        # Build similarity graph
        G = self.build_similarity_graph(embeddings)
        
        # Find groups
        groups = self.find_groups(G, sentence_to_original)
        
        return groups, similarities

async def main():
    # Initialize OpenAI provider
    openai_provider = OpenAIProvider(
        api_key=GroupingConfig.openai_api_key,
        embedding_model="text-embedding-ada-002"
    )
    
    # Create config
    config = GroupingConfig()
    
    # Initialize pipeline
    pipeline = ArgumentGroupingPipeline(
        llm_provider=openai_provider,
        config=config,
        translator=GoogleTranslator(source='auto', target='en')
    )
    
    # Run pipeline
    groups, similarities = await pipeline.process()
    
    # Save results in text format
    pipeline.save_results(groups, config.save_path)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())