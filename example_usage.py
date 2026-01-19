"""
Example Usage: Vector Database with Pinecone
Demonstrates how to use the VectorDatabaseManager class for common operations.
"""

import os
from embeddings import EmbeddingGenerator
from vector_db_setup import VectorDatabaseManager

# Initialize FastEmbed for local embedding generation
embedder = EmbeddingGenerator()

def generate_embedding(text):
    """
    Generate an embedding for the given text using FastEmbed (local).
    
    Args:
        text (str): Text to embed
    
    Returns:
        list: 384-dimensional embedding vector
    """
    return embedder.embed_text(text)

def example_1_basic_upsert():
    """Example 1: Basic upsert of vectors"""
    print("\n=== Example 1: Basic Upsert ===")
    
    db = VectorDatabaseManager()
    
    # Sample data
    documents = [
        {"id": "doc_1", "text": "Python is a high-level programming language."},
        {"id": "doc_2", "text": "Machine learning uses algorithms to learn patterns."},
        {"id": "doc_3", "text": "Vector databases store embeddings for similarity search."},
    ]
    
    # Generate embeddings and prepare vectors
    vectors = []
    for doc in documents:
        embedding = generate_embedding(doc["text"])
        vectors.append((
            doc["id"],
            embedding,
            {"text": doc["text"], "source": "example_1"}
        ))
    
    # Upsert to Pinecone
    result = db.upsert_vectors(vectors)
    print(f"✅ Upserted {result['total_upserted']} vectors")

def example_2_query_similarity():
    """Example 2: Query for similar vectors"""
    print("\n=== Example 2: Similarity Search ===")
    
    db = VectorDatabaseManager()
    
    # Query text
    query_text = "What is machine learning?"
    query_embedding = generate_embedding(query_text)
    
    # Search for similar vectors
    results = db.query_vectors(
        query_vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    print(f"\nQuery: '{query_text}'")
    print(f"\nTop {len(results.matches)} similar results:")
    for i, match in enumerate(results.matches, 1):
        print(f"\n{i}. ID: {match.id}")
        print(f"   Score: {match.score:.4f}")
        print(f"   Text: {match.metadata.get('text', 'N/A')}")

def example_3_batch_upsert():
    """Example 3: Batch upsert with larger dataset"""
    print("\n=== Example 3: Batch Upsert ===")
    
    db = VectorDatabaseManager()
    
    # Simulate larger dataset
    documents = []
    for i in range(50):
        documents.append({
            "id": f"batch_doc_{i}",
            "text": f"This is sample document number {i} about various topics.",
        })
    
    # Generate embeddings
    print("Generating embeddings...")
    vectors = []
    for doc in documents:
        embedding = generate_embedding(doc["text"])
        vectors.append((
            doc["id"],
            embedding,
            {
                "text": doc["text"],
                "source": "example_3",
                "doc_number": doc["id"]
            }
        ))
    
    # Upsert in batches
    result = db.upsert_vectors(vectors)
    print(f"✅ Batch upsert completed: {result['total_upserted']} vectors")

def example_4_update_existing():
    """Example 4: Update existing vectors"""
    print("\n=== Example 4: Update Existing Vectors ===")
    
    db = VectorDatabaseManager()
    
    # Update an existing document (same ID)
    doc_id = "doc_1"
    updated_text = "Python is a versatile, high-level programming language used in AI."
    
    embedding = generate_embedding(updated_text)
    vectors = [(
        doc_id,
        embedding,
        {"text": updated_text, "source": "example_4", "updated": True}
    )]
    
    result = db.upsert_vectors(vectors)
    print(f"✅ Updated vector '{doc_id}'")

def example_5_with_filters():
    """Example 5: Query with metadata filtering"""
    print("\n=== Example 5: Query with Metadata ===")
    
    db = VectorDatabaseManager()
    
    # First, upsert documents with specific metadata
    documents = [
        {"id": "tech_1", "text": "Cloud computing provides on-demand resources.", "category": "technology"},
        {"id": "tech_2", "text": "Artificial intelligence mimics human cognition.", "category": "technology"},
        {"id": "science_1", "text": "Photosynthesis converts light into chemical energy.", "category": "science"},
    ]
    
    vectors = []
    for doc in documents:
        embedding = generate_embedding(doc["text"])
        vectors.append((
            doc["id"],
            embedding,
            {
                "text": doc["text"],
                "category": doc["category"],
                "source": "example_5"
            }
        ))
    
    db.upsert_vectors(vectors)
    
    # Query
    query_text = "What is AI?"
    query_embedding = generate_embedding(query_text)
    
    results = db.query_vectors(
        query_vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    print(f"\nQuery: '{query_text}'")
    print("\nResults with metadata:")
    for i, match in enumerate(results.matches, 1):
        print(f"\n{i}. {match.id}")
        print(f"   Category: {match.metadata.get('category', 'N/A')}")
        print(f"   Text: {match.metadata.get('text', 'N/A')}")
        print(f"   Score: {match.score:.4f}")

def example_6_index_stats():
    """Example 6: Get index statistics"""
    print("\n=== Example 6: Index Statistics ===")
    
    db = VectorDatabaseManager()
    stats = db.get_index_stats()
    
    print(f"\nIndex: {db.index_name}")
    print(f"Total vectors: {stats.total_vector_count}")
    print(f"Dimension: {db.dimension}")
    
    # Show namespace stats if available
    if hasattr(stats, 'namespaces') and stats.namespaces:
        print("\nNamespace breakdown:")
        for namespace, data in stats.namespaces.items():
            print(f"  - {namespace}: {data.vector_count} vectors")

def main():
    """Run all examples"""
    print("=" * 60)
    print("Vector Database Usage Examples")
    print("=" * 60)
    
    try:
        # Check if API keys are set
        if not os.getenv("PINECONE_API_KEY"):
            print("❌ Error: PINECONE_API_KEY not found in environment variables")
            return
        
        print("ℹ️  Using FastEmbed for local embedding generation (no API key needed)")
        
        # Run examples
        example_1_basic_upsert()
        example_2_query_similarity()
        example_3_batch_upsert()
        example_4_update_existing()
        example_5_with_filters()
        example_6_index_stats()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        raise

if __name__ == "__main__":
    main()
