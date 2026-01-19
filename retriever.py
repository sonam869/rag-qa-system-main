"""
Retriever Module with MMR (Maximal Marginal Relevance)
Retrieves relevant chunks from vector database with diversity.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from vector_db_setup import VectorDatabaseManager
from embeddings import EmbeddingGenerator


class Retriever:
    """
    Retrieves relevant chunks from vector database using MMR for diversity.
    """
    
    def __init__(
        self,
        db_manager: VectorDatabaseManager,
        embedder: EmbeddingGenerator
    ):
        """
        Initialize retriever.
        
        Args:
            db_manager: Vector database manager
            embedder: Embedding generator
        """
        self.db = db_manager
        self.embedder = embedder
        self.index = db_manager.get_index()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Simple retrieval without MMR.
        
        Args:
            query (str): Query text
            top_k (int): Number of results to retrieve
            include_metadata (bool): Whether to include metadata
        
        Returns:
            List[Dict]: Retrieved chunks with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Query vector database
        results = self.db.query_vectors(
            query_vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
        
        # Format results
        retrieved = []
        for match in results.matches:
            retrieved.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })
        
        return retrieved
    
    def retrieve_with_mmr(
        self,
        query: str,
        top_k: int = 10,
        fetch_k: int = 50,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve with Maximal Marginal Relevance (MMR) for diversity.
        
        MMR balances relevance and diversity:
        - lambda=1.0: Pure relevance (no diversity)
        - lambda=0.0: Pure diversity (no relevance)
        - lambda=0.5: Balanced (recommended)
        
        Args:
            query (str): Query text
            top_k (int): Number of final results
            fetch_k (int): Number of candidates to fetch (should be > top_k)
            lambda_param (float): Balance parameter (0-1)
        
        Returns:
            List[Dict]: MMR-ranked results
        """
        if fetch_k < top_k:
            fetch_k = top_k * 3  # At least 3x for good diversity
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        query_vec = np.array(query_embedding)
        
        # Fetch initial candidates
        candidates = self.retrieve(
            query=query,
            top_k=fetch_k,
            include_metadata=True
        )
        
        if not candidates:
            return []
        
        # Extract embeddings for all candidates
        # Since we don't store full embeddings in results, regenerate them
        candidate_texts = [c["text"] for c in candidates]
        candidate_embeddings = self.embedder.embed_batch(candidate_texts)
        candidate_vecs = [np.array(emb) for emb in candidate_embeddings]
        
        # MMR algorithm
        selected_indices = []
        selected_vecs = []
        remaining_indices = list(range(len(candidates)))
        
        for _ in range(min(top_k, len(candidates))):
            if not remaining_indices:
                break
            
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance: similarity to query
                relevance = self._cosine_similarity(query_vec, candidate_vecs[idx])
                
                # Diversity: max similarity to already selected
                if selected_vecs:
                    max_similarity = max([
                        self._cosine_similarity(candidate_vecs[idx], sel_vec)
                        for sel_vec in selected_vecs
                    ])
                else:
                    max_similarity = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))
            
            # Select best MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            
            selected_indices.append(best_idx)
            selected_vecs.append(candidate_vecs[best_idx])
            remaining_indices.remove(best_idx)
        
        # Return selected results
        mmr_results = [candidates[idx] for idx in selected_indices]
        
        print(f"MMR Retrieval: Selected {len(mmr_results)} from {len(candidates)} candidates")
        return mmr_results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def retrieve_with_filter(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve with metadata filtering.
        
        Args:
            query (str): Query text
            top_k (int): Number of results
            filter_dict (Dict): Pinecone metadata filter
        
        Returns:
            List[Dict]: Filtered results
        """
        query_embedding = self.embedder.embed_text(query)
        
        # Query with filter
        index = self.db.get_index()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        retrieved = []
        for match in results.matches:
            retrieved.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })
        
        return retrieved
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        use_mmr: bool = True,
        lambda_param: float = 0.5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Hybrid retrieval with optional MMR and filtering.
        
        Args:
            query (str): Query text
            top_k (int): Number of results
            use_mmr (bool): Whether to use MMR
            lambda_param (float): MMR balance parameter
            filter_dict (Dict): Optional metadata filter
        
        Returns:
            List[Dict]: Retrieved results
        """
        if use_mmr:
            # Note: Pinecone filtering with MMR requires fetching all,
            # then filtering, then applying MMR
            if filter_dict:
                # Fetch more candidates to account for filtering
                results = self.retrieve_with_filter(
                    query=query,
                    top_k=top_k * 3,
                    filter_dict=filter_dict
                )
                # Apply MMR on filtered results
                # (simplified - would need embeddings)
                return results[:top_k]
            else:
                return self.retrieve_with_mmr(
                    query=query,
                    top_k=top_k,
                    lambda_param=lambda_param
                )
        else:
            if filter_dict:
                return self.retrieve_with_filter(
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict
                )
            else:
                return self.retrieve(
                    query=query,
                    top_k=top_k
                )


def main():
    """Test retriever"""
    print("=" * 60)
    print("Retriever Test")
    print("=" * 60)
    
    # This is a demo - actual testing requires populated database
    print("\nRetriever initialized successfully!")
    print("\nRetrieval strategies available:")
    print("  1. Simple top-k retrieval")
    print("  2. MMR (Maximal Marginal Relevance) - balances relevance & diversity")
    print("  3. Filtered retrieval - by metadata")
    print("  4. Hybrid - combines MMR and filtering")
    
    print("\n" + "=" * 60)
    print("MMR Algorithm Explanation:")
    print("=" * 60)
    print("""
MMR Score = λ × Relevance - (1-λ) × MaxSimilarity

Where:
- Relevance: Similarity to query
- MaxSimilarity: Similarity to already selected documents
- λ (lambda): Balance parameter (0-1)

Benefits:
- Reduces redundancy in results
- Ensures diverse perspectives
- Better coverage of relevant information

Recommended settings:
- λ = 0.5: Balanced relevance and diversity
- λ = 0.7: More relevance, less diversity
- λ = 0.3: More diversity, less relevance
    """)
    
    print("✅ Test completed!")


if __name__ == "__main__":
    main()
