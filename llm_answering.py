"""
LLM Answering Module using Groq (Llama-3.1-8b)
Generates grounded answers with inline citations.
"""

import os
from groq import Groq
from typing import List, Dict, Optional, Tuple
import re


class GroqAnswerGenerator:
    """
    Generates answers using Groq's Llama-3.1-8b model.
    Produces grounded responses with citations.
    """
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq client.
        
        Args:
            model (str): Model to use
                - llama-3.1-8b-instant (recommended, fast)
                - llama-3.1-70b-versatile (more capable)
                - mixtral-8x7b-32768 (alternative)
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        
        print(f"✅ Groq client initialized with model: {model}")
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.3
    ) -> Dict:
        """
        Generate an answer with citations.
        
        Args:
            query (str): User question
            context_chunks (List[Dict]): Retrieved and reranked chunks
            max_tokens (int): Maximum response tokens
            temperature (float): Generation temperature (0-1)
        
        Returns:
            Dict: Answer with citations and source snippets
        """
        if not context_chunks:
            return self._generate_no_answer_response(query)
        
        # Build context with citations
        context_text, citation_map = self._build_context(context_chunks)
        
        # Create prompt
        prompt = self._create_prompt(query, context_text)
        
        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            answer_text = response.choices[0].message.content
            
            # Extract citations from answer
            citations_used = self._extract_citations(answer_text)
            
            # Build source snippets
            sources = self._build_sources(citations_used, citation_map, context_chunks)
            
            return {
                "answer": answer_text,
                "sources": sources,
                "has_answer": True,
                "model": self.model,
                "query": query
            }
        
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return self._generate_error_response(query, str(e))
    
    def _build_context(self, chunks: List[Dict]) -> Tuple[str, Dict]:
        """
        Build context text with citation markers.
        
        Returns:
            Tuple[str, Dict]: (context_text, citation_map)
        """
        context_parts = []
        citation_map = {}
        
        for i, chunk in enumerate(chunks, 1):
            citation_num = i
            text = chunk.get("text", "")
            
            # Store citation info
            citation_map[citation_num] = {
                "chunk_id": chunk.get("id", f"chunk_{i}"),
                "text": text,
                "metadata": chunk.get("metadata", {}),
                "score": chunk.get("rerank_score") or chunk.get("score", 0.0)
            }
            
            # Add to context with citation marker
            context_parts.append(f"[{citation_num}] {text}")
        
        context_text = "\n\n".join(context_parts)
        return context_text, citation_map
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the user prompt."""
        return f"""Based on the following context, answer the question. Use inline citations [1], [2], etc. to reference specific sources.

Context:
{context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Include inline citations [1], [2], etc. after each claim
3. If the context doesn't contain enough information, say so clearly
4. Be concise but comprehensive
5. Cite specific sources for each fact or claim

Answer:"""
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt."""
        return """You are a helpful AI assistant that provides accurate, grounded answers with citations.

Rules:
- ONLY use information from the provided context
- Include inline citations [1], [2], etc. for every claim
- If you cannot answer based on the context, say "I don't have enough information in the provided context to answer this question."
- Never make up information
- Be clear and concise
- Multiple citations can be used: [1][2] or [1, 2]"""
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text."""
        # Match [1], [2], [1,2], [1][2], etc.
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        citations = sorted(set(int(m) for m in matches))
        return citations
    
    def _build_sources(
        self,
        citations_used: List[int],
        citation_map: Dict,
        chunks: List[Dict]
    ) -> List[Dict]:
        """Build source snippets for citations."""
        sources = []
        
        for citation_num in citations_used:
            if citation_num in citation_map:
                citation_info = citation_map[citation_num]
                metadata = citation_info["metadata"]
                
                source = {
                    "citation": citation_num,
                    "text": citation_info["text"],
                    "source": metadata.get("source", "Unknown"),
                    "title": metadata.get("title", "Untitled"),
                    "section": metadata.get("section"),
                    "score": citation_info["score"]
                }
                sources.append(source)
        
        return sources
    
    def _generate_no_answer_response(self, query: str) -> Dict:
        """Generate response when no context is available."""
        return {
            "answer": "I don't have enough information in the knowledge base to answer this question. Please try rephrasing your question or provide more context.",
            "sources": [],
            "has_answer": False,
            "model": self.model,
            "query": query
        }
    
    def _generate_error_response(self, query: str, error: str) -> Dict:
        """Generate response when an error occurs."""
        return {
            "answer": f"An error occurred while generating the answer: {error}",
            "sources": [],
            "has_answer": False,
            "model": self.model,
            "query": query,
            "error": error
        }
    
    def format_response(self, response: Dict) -> str:
        """
        Format the response for display.
        
        Args:
            response (Dict): Response from generate_answer
        
        Returns:
            str: Formatted response text
        """
        output = []
        
        # Add query
        output.append("=" * 80)
        output.append(f"QUESTION: {response['query']}")
        output.append("=" * 80)
        
        # Add answer
        output.append("\nANSWER:")
        output.append(response['answer'])
        
        # Add sources if available
        if response.get('has_answer') and response.get('sources'):
            output.append("\n" + "=" * 80)
            output.append("SOURCES:")
            output.append("=" * 80)
            
            for source in response['sources']:
                output.append(f"\n[{source['citation']}] {source['title']}")
                if source.get('section'):
                    output.append(f"    Section: {source['section']}")
                output.append(f"    Source: {source['source']}")
                output.append(f"    Score: {source['score']:.4f}")
                output.append(f"    Text: {source['text'][:200]}...")
        
        return "\n".join(output)


def main():
    """Test answer generator"""
    print("=" * 60)
    print("Groq Answer Generator Test")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment variables")
        print("\nTo get started:")
        print("1. Sign up at https://console.groq.com")
        print("2. Get your API key")
        print("3. Add to .env file: GROQ_API_KEY=your_key_here")
        return
    
    # Initialize generator
    generator = GroqAnswerGenerator()
    
    # Sample context chunks
    context_chunks = [
        {
            "id": "chunk_1",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
            "score": 0.95,
            "metadata": {
                "source": "ml_intro.pdf",
                "title": "Introduction to Machine Learning",
                "section": "Overview"
            }
        },
        {
            "id": "chunk_2",
            "text": "Deep learning uses artificial neural networks with multiple layers to model complex patterns in data.",
            "score": 0.88,
            "metadata": {
                "source": "deep_learning.pdf",
                "title": "Deep Learning Fundamentals",
                "section": "Neural Networks"
            }
        }
    ]
    
    # Generate answer
    query = "What is machine learning and how does it relate to deep learning?"
    print(f"\nQuery: {query}")
    print("\nGenerating answer...")
    
    response = generator.generate_answer(query, context_chunks)
    
    # Format and display
    formatted = generator.format_response(response)
    print("\n" + formatted)
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()
