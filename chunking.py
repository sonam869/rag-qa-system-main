"""
Text Chunking Module
Implements chunking strategy with configurable size and overlap.
Stores metadata for citation purposes.
"""

import tiktoken
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: str
    source: str
    title: str
    section: Optional[str]
    position: int
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict


class TextChunker:
    """
    Chunks text into overlapping segments with metadata.
    
    Configuration:
    - Size: 800-1200 tokens (default: 1000)
    - Overlap: 10-15% (default: 12.5% = 125 tokens for 1000 token chunks)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_percentage: float = 12.5,
        encoding_name: str = "cl100k_base"  # GPT-3.5/GPT-4 encoding
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size (int): Target chunk size in tokens (800-1200)
            overlap_percentage (float): Overlap percentage (10-15%)
            encoding_name (str): Tokenizer encoding to use
        """
        if not 800 <= chunk_size <= 1200:
            raise ValueError("Chunk size should be between 800 and 1200 tokens")
        
        if not 10 <= overlap_percentage <= 15:
            raise ValueError("Overlap percentage should be between 10 and 15")
        
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.overlap_tokens = int(chunk_size * (overlap_percentage / 100))
        
        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        print(f"TextChunker initialized:")
        print(f"  - Chunk size: {self.chunk_size} tokens")
        print(f"  - Overlap: {self.overlap_percentage}% ({self.overlap_tokens} tokens)")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(
        self,
        text: str,
        source: str,
        title: str,
        section: Optional[str] = None,
        additional_metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text into overlapping segments with metadata.
        
        Args:
            text (str): Text to chunk
            source (str): Source identifier (e.g., file path, URL)
            title (str): Document title
            section (str): Section name (optional)
            additional_metadata (Dict): Extra metadata to store
        
        Returns:
            List[Chunk]: List of chunks with metadata
        """
        if not text.strip():
            return []
        
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        chunks = []
        position = 0
        start_idx = 0
        
        while start_idx < total_tokens:
            # Define chunk boundaries
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode chunk back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Find character positions in original text
            # This is approximate but good enough for metadata
            char_ratio = len(text) / total_tokens if total_tokens > 0 else 0
            start_char = int(start_idx * char_ratio)
            end_char = int(end_idx * char_ratio)
            
            # Create chunk ID
            chunk_id = f"{source}_{position}"
            
            # Build metadata
            metadata = {
                "source": source,
                "title": title,
                "section": section,
                "position": position,
                "start_char": start_char,
                "end_char": end_char,
                "token_count": len(chunk_tokens),
                "total_tokens": total_tokens,
                "chunk_size": self.chunk_size,
                "overlap_tokens": self.overlap_tokens,
            }
            
            # Add any additional metadata
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Create chunk object
            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source=source,
                title=title,
                section=section,
                position=position,
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_tokens),
                metadata=metadata
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.overlap_tokens)
            position += 1
        
        print(f"Created {len(chunks)} chunks from {total_tokens} tokens")
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Dict]
    ) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents (List[Dict]): List of documents with keys:
                - text: Document text
                - source: Source identifier
                - title: Document title
                - section: (Optional) Section name
                - metadata: (Optional) Additional metadata
        
        Returns:
            List[Chunk]: All chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(
                text=doc["text"],
                source=doc.get("source", "unknown"),
                title=doc.get("title", "Untitled"),
                section=doc.get("section"),
                additional_metadata=doc.get("metadata")
            )
            all_chunks.extend(chunks)
        
        print(f"Total: {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunk_with_context(
        self,
        chunks: List[Chunk],
        chunk_position: int,
        context_before: int = 1,
        context_after: int = 1
    ) -> str:
        """
        Get a chunk with surrounding context.
        
        Args:
            chunks (List[Chunk]): List of chunks from same source
            chunk_position (int): Position of target chunk
            context_before (int): Number of chunks before to include
            context_after (int): Number of chunks after to include
        
        Returns:
            str: Combined text with context
        """
        start = max(0, chunk_position - context_before)
        end = min(len(chunks), chunk_position + context_after + 1)
        
        context_chunks = chunks[start:end]
        return " ".join([c.text for c in context_chunks])


def main():
    """Test the chunker"""
    print("=" * 60)
    print("Text Chunker Test")
    print("=" * 60)
    
    # Initialize chunker
    chunker = TextChunker(
        chunk_size=1000,
        overlap_percentage=12.5
    )
    
    # Test text
    test_text = """
    Machine learning is a subset of artificial intelligence that focuses on the development 
    of algorithms and statistical models that enable computer systems to improve their 
    performance on a specific task through experience. The field has evolved significantly 
    over the past few decades, with deep learning emerging as a particularly powerful approach.
    
    Deep learning uses artificial neural networks with multiple layers to progressively extract 
    higher-level features from raw input. For example, in image processing, lower layers may 
    identify edges, while higher layers may identify concepts relevant to a human such as digits 
    or letters or faces.
    
    Vector databases have become crucial infrastructure for modern AI applications. They enable 
    efficient similarity search over high-dimensional embeddings, which is essential for 
    applications like semantic search, recommendation systems, and retrieval-augmented generation.
    """ * 5  # Repeat to get more tokens
    
    # Chunk the text
    print("\nChunking text...")
    chunks = chunker.chunk_text(
        text=test_text,
        source="test_document.txt",
        title="Machine Learning Overview",
        section="Introduction",
        additional_metadata={"author": "Test", "date": "2026-01-19"}
    )
    
    # Display results
    print(f"\nCreated {len(chunks)} chunks")
    print("\nFirst chunk details:")
    first_chunk = chunks[0]
    print(f"  ID: {first_chunk.chunk_id}")
    print(f"  Position: {first_chunk.position}")
    print(f"  Tokens: {first_chunk.token_count}")
    print(f"  Text preview: {first_chunk.text[:100]}...")
    print(f"  Metadata: {first_chunk.metadata}")
    
    # Test batch chunking
    print("\n" + "=" * 60)
    print("Batch Chunking Test")
    print("=" * 60)
    
    documents = [
        {
            "text": test_text,
            "source": "doc1.txt",
            "title": "Document 1",
            "section": "Introduction"
        },
        {
            "text": test_text[:500],
            "source": "doc2.txt",
            "title": "Document 2",
            "section": "Summary"
        }
    ]
    
    all_chunks = chunker.chunk_documents(documents)
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
