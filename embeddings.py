"""
Embeddings Module using FastEmbed (Local)
Generates embeddings locally without external API calls.
"""

from fastembed import TextEmbedding
from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """
    Local embedding generation using FastEmbed.
    Uses BAAI/bge-small-en-v1.5 model by default (384 dimensions).
    """
    
    # Supported models and their dimensions
    MODELS = {
        "BAAI/bge-small-en-v1.5": 384,  # Fast, efficient, good quality
        "BAAI/bge-base-en-v1.5": 768,   # Better quality, slower
        "sentence-transformers/all-MiniLM-L6-v2": 384,  # Popular alternative
    }
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name of the model to use
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unsupported model. Choose from: {list(self.MODELS.keys())}")
        
        self.model_name = model_name
        self.dimension = self.MODELS[model_name]
        
        print(f"Loading embedding model: {model_name}")
        print(f"Dimension: {self.dimension}")
        
        # Initialize FastEmbed model (downloads on first run)
        self.model = TextEmbedding(model_name=model_name)
        
        print("✅ Embedding model loaded successfully!")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text to embed
        
        Returns:
            List[float]: Embedding vector
        """
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
        
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = list(self.model.embed(batch))
            embeddings.extend([emb.tolist() for emb in batch_embeddings])
            
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding
            embedding2 (List[float]): Second embedding
        
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)


def main():
    """Test the embedding generator"""
    print("=" * 60)
    print("FastEmbed Embedding Generator Test")
    print("=" * 60)
    
    # Initialize
    embedder = EmbeddingGenerator()
    
    # Test single embedding
    print("\n1. Testing single text embedding:")
    text = "This is a test sentence for embedding generation."
    embedding = embedder.embed_text(text)
    print(f"   Text: {text}")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    print("\n2. Testing batch embeddings:")
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Vector databases store high-dimensional embeddings.",
        "FastEmbed provides local embedding generation.",
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"   Generated {len(embeddings)} embeddings")
    
    # Test similarity
    print("\n3. Testing similarity computation:")
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"   Similarity between text 1 and 2: {sim:.4f}")
    sim = embedder.compute_similarity(embeddings[0], embeddings[2])
    print(f"   Similarity between text 1 and 3: {sim:.4f}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
