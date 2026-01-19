"""
Complete RAG Pipeline
Demonstrates end-to-end retrieval-augmented generation with all components.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict

# Import all components
from embeddings import EmbeddingGenerator
from chunking import TextChunker, Chunk
from vector_db_setup import VectorDatabaseManager
from retriever import Retriever
from reranker import RerankerFlashRank
from llm_answering import GroqAnswerGenerator

# Load environment variables
load_dotenv()


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline.
    
    Pipeline stages:
    1. Chunk documents with metadata
    2. Generate embeddings locally (FastEmbed)
    3. Store in Pinecone vector database
    4. Retrieve with MMR for diversity
    5. Rerank with FlashRank
    6. Generate grounded answer with Groq Llama-3.1
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 1000,
        overlap_percentage: float = 12.5,
        rerank_model: str = "ms-marco-MiniLM-L-12-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: FastEmbed model name
            chunk_size: Tokens per chunk (800-1200)
            overlap_percentage: Chunk overlap (10-15%)
            rerank_model: FlashRank model name
            llm_model: Groq model name
        """
        print("=" * 80)
        print("Initializing RAG Pipeline")
        print("=" * 80)
        
        # Initialize components
        print("\n1. Loading embedding model...")
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        
        print("\n2. Initializing chunker...")
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap_percentage=overlap_percentage
        )
        
        print("\n3. Connecting to vector database...")
        self.db_manager = VectorDatabaseManager()
        
        print("\n4. Setting up retriever...")
        self.retriever = Retriever(self.db_manager, self.embedder)
        
        print("\n5. Loading reranker...")
        self.reranker = RerankerFlashRank(model_name=rerank_model)
        
        print("\n6. Initializing LLM...")
        self.answer_generator = GroqAnswerGenerator(model=llm_model)
        
        print("\n✅ RAG Pipeline initialized successfully!")
        print("=" * 80)
    
    def ingest_documents(self, documents: List[Dict]) -> int:
        """
        Ingest documents into the pipeline.
        
        Args:
            documents (List[Dict]): Documents with keys:
                - text: Document content
                - source: Source identifier
                - title: Document title
                - section: (Optional) Section name
                - metadata: (Optional) Additional metadata
        
        Returns:
            int: Number of chunks created and stored
        """
        print("\n" + "=" * 80)
        print("INGESTING DOCUMENTS")
        print("=" * 80)
        
        # Step 1: Chunk documents
        print("\nStep 1: Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        
        if not chunks:
            print("No chunks created!")
            return 0
        
        # Step 2: Generate embeddings
        print("\nStep 2: Generating embeddings...")
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)
        
        # Step 3: Prepare vectors for Pinecone
        print("\nStep 3: Preparing vectors...")
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append((
                chunk.chunk_id,
                embedding,
                chunk.metadata
            ))
        
        # Step 4: Upsert to vector database
        print("\nStep 4: Storing in vector database...")
        result = self.db_manager.upsert_vectors(vectors)
        
        print(f"\n✅ Successfully ingested {result['total_upserted']} chunks!")
        return result['total_upserted']
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        rerank_top_k: int = 5,
        use_mmr: bool = True,
        lambda_param: float = 0.5
    ) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question (str): User question
            top_k (int): Number of chunks to retrieve
            rerank_top_k (int): Number of chunks after reranking
            use_mmr (bool): Whether to use MMR for diversity
            lambda_param (float): MMR balance parameter
        
        Returns:
            Dict: Answer with sources and metadata
        """
        print("\n" + "=" * 80)
        print("PROCESSING QUERY")
        print("=" * 80)
        print(f"Question: {question}")
        
        # Step 1: Retrieve
        print("\nStep 1: Retrieving relevant chunks...")
        if use_mmr:
            retrieved = self.retriever.retrieve_with_mmr(
                query=question,
                top_k=top_k,
                lambda_param=lambda_param
            )
        else:
            retrieved = self.retriever.retrieve(
                query=question,
                top_k=top_k
            )
        
        if not retrieved:
            print("No relevant chunks found!")
            return self.answer_generator._generate_no_answer_response(question)
        
        print(f"Retrieved {len(retrieved)} chunks")
        
        # Step 2: Rerank
        print("\nStep 2: Reranking results...")
        reranked = self.reranker.rerank(
            query=question,
            documents=retrieved,
            top_k=rerank_top_k
        )
        
        print(f"Reranked to top {len(reranked)} chunks")
        
        # Step 3: Generate answer
        print("\nStep 3: Generating answer with LLM...")
        answer = self.answer_generator.generate_answer(
            query=question,
            context_chunks=reranked
        )
        
        print("✅ Answer generated!")
        
        return answer
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = self.db_manager.get_index_stats()
        
        return {
            "total_chunks": stats.total_vector_count,
            "embedding_dimension": self.embedder.get_dimension(),
            "chunk_size": self.chunker.chunk_size,
            "overlap_tokens": self.chunker.overlap_tokens,
            "index_name": self.db_manager.index_name
        }


def demo_rag_pipeline():
    """Demonstrate the complete RAG pipeline."""
    
    # Check for required API keys
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not found!")
        return
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found!")
        return
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Sample documents
    documents = [
        {
            "text": """
            Machine Learning is a subset of artificial intelligence that enables systems to 
            learn and improve from experience without being explicitly programmed. It focuses 
            on the development of computer programs that can access data and use it to learn 
            for themselves. The process of learning begins with observations or data, such as 
            examples, direct experience, or instruction, in order to look for patterns in data 
            and make better decisions in the future based on the examples that we provide.
            
            There are three main types of machine learning: supervised learning, unsupervised 
            learning, and reinforcement learning. Supervised learning involves training a model 
            on labeled data. Unsupervised learning works with unlabeled data to find hidden 
            patterns. Reinforcement learning involves an agent learning to make decisions by 
            taking actions in an environment to maximize rewards.
            """,
            "source": "ml_intro.txt",
            "title": "Introduction to Machine Learning",
            "section": "Overview",
            "metadata": {"author": "AI Research Team", "date": "2026-01-15"}
        },
        {
            "text": """
            Deep Learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers (hence "deep") to model and understand complex patterns in 
            datasets. These neural networks are inspired by the structure and function of the 
            human brain, with interconnected nodes (neurons) organized in layers.
            
            The key advantage of deep learning is its ability to automatically learn 
            representations from data. This means it can discover the features needed for 
            detection or classification from raw data, without manual feature engineering. 
            Deep learning has achieved remarkable success in computer vision, natural language 
            processing, speech recognition, and many other domains.
            
            Popular deep learning architectures include Convolutional Neural Networks (CNNs) 
            for image processing, Recurrent Neural Networks (RNNs) for sequential data, and 
            Transformers for natural language understanding.
            """,
            "source": "deep_learning.txt",
            "title": "Deep Learning Fundamentals",
            "section": "Neural Networks",
            "metadata": {"author": "AI Research Team", "date": "2026-01-16"}
        },
        {
            "text": """
            Vector databases are specialized database systems designed to store and query 
            high-dimensional vectors efficiently. These vectors are typically embeddings - 
            numerical representations of data like text, images, or audio that capture 
            semantic meaning in a mathematical form.
            
            Unlike traditional databases that use exact matching, vector databases use 
            similarity search to find items that are "close" to a query vector in the 
            embedding space. This enables semantic search, where you can find conceptually 
            similar items even if they don't share exact keywords.
            
            Vector databases are essential infrastructure for modern AI applications, including:
            - Semantic search engines
            - Recommendation systems
            - Retrieval-augmented generation (RAG)
            - Image and video similarity search
            - Anomaly detection
            
            Popular vector databases include Pinecone, Weaviate, Qdrant, and Milvus.
            """,
            "source": "vector_db.txt",
            "title": "Vector Databases Explained",
            "section": "Introduction",
            "metadata": {"author": "Data Engineering Team", "date": "2026-01-17"}
        }
    ]
    
    # Ingest documents
    print("\n" + "=" * 80)
    print("DEMO: Document Ingestion")
    print("=" * 80)
    
    num_chunks = pipeline.ingest_documents(documents)
    print(f"\nIngested {num_chunks} chunks from {len(documents)} documents")
    
    # Display stats
    stats = pipeline.get_stats()
    print("\nPipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Query examples
    queries = [
        "What is machine learning?",
        "Explain deep learning and neural networks",
        "What are vector databases used for?",
        "How does supervised learning differ from unsupervised learning?",
    ]
    
    print("\n" + "=" * 80)
    print("DEMO: Question Answering")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}/{len(queries)}")
        print("=" * 80)
        
        response = pipeline.query(
            question=query,
            top_k=10,
            rerank_top_k=3,
            use_mmr=True
        )
        
        # Format and display
        formatted = pipeline.answer_generator.format_response(response)
        print("\n" + formatted)
        
        input("\nPress Enter to continue to next query...")
    
    print("\n" + "=" * 80)
    print("✅ RAG Pipeline Demo Completed!")
    print("=" * 80)


def main():
    """Main entry point"""
    demo_rag_pipeline()


if __name__ == "__main__":
    main()
