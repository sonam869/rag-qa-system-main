"""
Reranker Module using FlashRank
Reranks retrieved results for better relevance.
"""

from flashrank import Ranker, RerankRequest
from typing import List, Dict, Optional


class RerankerFlashRank:
    """
    Reranks retrieved chunks using FlashRank.
    FlashRank is a free, local reranker that improves retrieval quality.
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize FlashRank reranker.
        
        Args:
            model_name (str): Model to use for reranking
                Available models:
                - ms-marco-MiniLM-L-12-v2 (default, fast)
                - ms-marco-MultiBERT-L-12 (multilingual)
                - rank-T5-flan (more accurate, slower)
        """
        self.model_name = model_name
        
        print(f"Loading FlashRank model: {model_name}")
        self.ranker = Ranker(model_name=model_name, cache_dir="./flashrank_cache")
        print("✅ FlashRank reranker loaded successfully!")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query (str): Query text
            documents (List[Dict]): Retrieved documents with 'text' field
            top_k (int): Number of top results to return (None = all)
        
        Returns:
            List[Dict]: Reranked documents with updated scores
        """
        if not documents:
            return []
        
        # Prepare passages for reranking
        passages = []
        for i, doc in enumerate(documents):
            passages.append({
                "id": i,
                "text": doc.get("text", ""),
                "meta": doc.get("metadata", {})
            })
        
        # Create rerank request
        rerank_request = RerankRequest(
            query=query,
            passages=passages
        )
        
        # Perform reranking
        reranked_results = self.ranker.rerank(rerank_request)
        
        # Map back to original documents with new scores
        reranked_docs = []
        for result in reranked_results:
            original_doc = documents[result["id"]]
            reranked_doc = {
                **original_doc,
                "rerank_score": result["score"],
                "original_score": original_doc.get("score", 0.0),
                "rank_position": len(reranked_docs) + 1
            }
            reranked_docs.append(reranked_doc)
        
        # Apply top_k if specified
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        print(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
        return reranked_docs
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Rerank and filter by score threshold.
        
        Args:
            query (str): Query text
            documents (List[Dict]): Retrieved documents
            top_k (int): Number of top results
            score_threshold (float): Minimum rerank score to include
        
        Returns:
            List[Dict]: Filtered and reranked documents
        """
        reranked = self.rerank(query, documents, top_k=None)
        
        # Filter by threshold
        filtered = [
            doc for doc in reranked
            if doc["rerank_score"] >= score_threshold
        ]
        
        # Apply top_k after filtering
        if top_k:
            filtered = filtered[:top_k]
        
        print(f"After filtering (threshold={score_threshold}): {len(filtered)} documents")
        return filtered


class HybridReranker:
    """
    Combines multiple reranking strategies.
    """
    
    def __init__(self, flashrank_model: str = "ms-marco-MiniLM-L-12-v2"):
        """Initialize hybrid reranker."""
        self.flashrank = RerankerFlashRank(model_name=flashrank_model)
    
    def rerank_with_blending(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
        retrieval_weight: float = 0.3,
        rerank_weight: float = 0.7
    ) -> List[Dict]:
        """
        Blend retrieval and rerank scores.
        
        Args:
            query (str): Query text
            documents (List[Dict]): Retrieved documents
            top_k (int): Number of results
            retrieval_weight (float): Weight for original retrieval score
            rerank_weight (float): Weight for rerank score
        
        Returns:
            List[Dict]: Documents with blended scores
        """
        # Normalize weights
        total_weight = retrieval_weight + rerank_weight
        retrieval_weight /= total_weight
        rerank_weight /= total_weight
        
        # Rerank
        reranked = self.flashrank.rerank(query, documents)
        
        # Blend scores
        for doc in reranked:
            original_score = doc.get("original_score", 0.0)
            rerank_score = doc.get("rerank_score", 0.0)
            
            # Normalize scores to 0-1 range
            normalized_original = (original_score + 1) / 2  # Cosine is -1 to 1
            normalized_rerank = rerank_score  # Already 0-1
            
            # Blend
            blended_score = (
                retrieval_weight * normalized_original +
                rerank_weight * normalized_rerank
            )
            
            doc["blended_score"] = blended_score
        
        # Sort by blended score
        reranked.sort(key=lambda x: x["blended_score"], reverse=True)
        
        return reranked[:top_k]


def main():
    """Test reranker"""
    print("=" * 60)
    print("FlashRank Reranker Test")
    print("=" * 60)
    
    # Sample documents
    query = "What is machine learning?"
    
    documents = [
        {
            "id": "doc1",
            "text": "Python is a programming language used for web development.",
            "score": 0.75,
            "metadata": {"source": "doc1.txt"}
        },
        {
            "id": "doc2",
            "text": "Machine learning is a subset of AI that learns from data.",
            "score": 0.82,
            "metadata": {"source": "doc2.txt"}
        },
        {
            "id": "doc3",
            "text": "Deep learning uses neural networks with multiple layers.",
            "score": 0.78,
            "metadata": {"source": "doc3.txt"}
        },
        {
            "id": "doc4",
            "text": "Machine learning algorithms can be supervised or unsupervised.",
            "score": 0.80,
            "metadata": {"source": "doc4.txt"}
        },
    ]
    
    print(f"\nQuery: {query}")
    print(f"Documents to rerank: {len(documents)}")
    
    # Initialize reranker
    reranker = RerankerFlashRank()
    
    # Rerank
    print("\n" + "=" * 60)
    print("Reranking Results:")
    print("=" * 60)
    
    reranked = reranker.rerank(query, documents, top_k=3)
    
    for i, doc in enumerate(reranked, 1):
        print(f"\n{i}. ID: {doc['id']}")
        print(f"   Original Score: {doc['original_score']:.4f}")
        print(f"   Rerank Score: {doc['rerank_score']:.4f}")
        print(f"   Text: {doc['text'][:60]}...")
    
    print("\n✅ Reranking test completed!")


if __name__ == "__main__":
    main()
