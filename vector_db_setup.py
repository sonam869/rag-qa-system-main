"""
Vector Database Setup using Pinecone
This script initializes and configures a Pinecone vector database instance.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = "tumpa-embeddings"
DIMENSION = 384  # FastEmbed BAAI/bge-small-en-v1.5 dimension (changed from 1536)
METRIC = "cosine"  # Distance metric for similarity search
CLOUD = "aws"
REGION = "us-east-1"

class VectorDatabaseManager:
    """Manages Pinecone vector database operations"""
    
    def __init__(self):
        """Initialize Pinecone connection"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = INDEX_NAME
        self.dimension = DIMENSION
    
    def create_index(self):
        """
        Create a new Pinecone index if it doesn't exist.
        
        Index Configuration:
        - Name: tumpa-embeddings
        - Dimension: 384 (compatible with FastEmbed BAAI/bge-small-en-v1.5)
        - Metric: cosine (best for text embeddings)
        - Spec: Serverless on AWS us-east-1 (cost-effective, auto-scaling)
        """
        # Check if index already exists
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name in existing_indexes:
            print(f"Index '{self.index_name}' already exists")
            return self.pc.Index(self.index_name)
        
        # Create new index
        print(f"Creating index '{self.index_name}'...")
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=CLOUD,
                region=REGION
            )
        )
        
        print(f"Index '{self.index_name}' created successfully!")
        return self.pc.Index(self.index_name)
    
    def get_index(self):
        """Get reference to existing index"""
        return self.pc.Index(self.index_name)
    
    def get_index_stats(self):
        """Get statistics about the index"""
        index = self.get_index()
        stats = index.describe_index_stats()
        return stats
    
    def delete_index(self):
        """Delete the index (use with caution!)"""
        print(f"Deleting index '{self.index_name}'...")
        self.pc.delete_index(self.index_name)
        print(f"Index '{self.index_name}' deleted successfully!")
    
    def upsert_vectors(self, vectors):
        """
        Upsert vectors into the index.
        
        Upsert Strategy:
        - Batch upserts in chunks of 100 vectors for optimal performance
        - Uses upsert (update or insert) to handle both new and existing vectors
        - Each vector should have: id, values (embedding), metadata (optional)
        
        Args:
            vectors (list): List of tuples (id, embedding, metadata)
                Example: [("doc1", [0.1, 0.2, ...], {"text": "sample", "source": "file1"})]
        
        Returns:
            dict: Upsert response from Pinecone
        """
        index = self.get_index()
        
        # Batch upsert for better performance
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            response = index.upsert(vectors=batch)
            total_upserted += response.upserted_count
            print(f"Upserted {total_upserted}/{len(vectors)} vectors")
        
        return {"total_upserted": total_upserted}
    
    def query_vectors(self, query_vector, top_k=5, include_metadata=True):
        """
        Query the index for similar vectors.
        
        Args:
            query_vector (list): The query embedding vector
            top_k (int): Number of results to return
            include_metadata (bool): Whether to include metadata in results
        
        Returns:
            dict: Query results with matches
        """
        index = self.get_index()
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=False  # Don't return full vectors to save bandwidth
        )
        return results

def main():
    """Main setup function"""
    try:
        # Initialize vector database manager
        db_manager = VectorDatabaseManager()
        
        # Create index
        db_manager.create_index()
        
        # Get index stats
        stats = db_manager.get_index_stats()
        print(f"\nIndex Statistics:")
        print(f"Total vectors: {stats.total_vector_count}")
        print(f"Dimension: {db_manager.dimension}")
        print(f"Index name: {db_manager.index_name}")
        
        print("\n Vector database setup completed successfully!")
        print(f"\nIndex Details:")
        print(f"  - Name: {INDEX_NAME}")
        print(f"  - Dimension: {DIMENSION}")
        print(f"  - Metric: {METRIC}")
        print(f"  - Cloud: {CLOUD}")
        print(f"  - Region: {REGION}")
        
    except Exception as e:
        print(f" Error setting up vector database: {str(e)}")
        raise

if __name__ == "__main__":
    main()
